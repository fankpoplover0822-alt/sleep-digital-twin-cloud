from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGET = "osa_event_next_30s"
WINDOW_EPOCHS = 7
RANDOM_SEED = 42
STAGE_LEVELS = ("W", "N1", "N2", "N3", "R")
PREFERRED_FEATURES = (
    "flow_mean_resp", "flow_std_resp", "flow_range_resp", "flow_rms_resp",
    "flow_diff_std_resp", "flow_low_amplitude_fraction", "flow_flattening_proxy",
    "flow_respiratory_rate_bpm", "thermistor_std", "thermistor_range",
    "thorax_std_resp", "thorax_range_resp", "abdomen_std_resp", "abdomen_range_resp",
    "thorax_abdomen_correlation", "thorax_abdomen_opposite_phase_fraction",
    "spo2_mean_resp", "spo2_min_resp", "spo2_std_resp", "spo2_below_90_fraction",
    "spo2_drop_from_start", "spo2_largest_drop", "heart_rate_mean_resp",
    "heart_rate_std_resp", "snore_activity_fraction", "position_mean_resp", "age", "BMI",
)


@dataclass(frozen=True)
class OsaNextEpochConfig:
    window_epochs: int = WINDOW_EPOCHS
    hidden_size: int = 32
    dropout: float = 0.20
    batch_size: int = 256
    max_epochs: int = 40
    patience: int = 6
    learning_rate: float = 0.0015


class UniLSTM(nn.Module):
    """Causal single-direction LSTM: it can only use current and past epochs."""

    def __init__(self, n_features: int, hidden_size: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(values)
        return self.head(encoded[:, -1, :]).squeeze(1)


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def add_strict_osa_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Add t+1 OSA target and exclude central/mixed t+1 labels from supervision.

    Hypopnea is included because the source cohort is OSA-only. Central and mixed
    apnea are never treated as positive OSA labels; their next epochs are marked
    ambiguous and omitted from supervised training/evaluation.
    """
    required = {
        "patient_id", "epoch_index", "has_obstructive_apnea", "has_hypopnea",
        "has_central_apnea", "has_mixed_apnea",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"缺少建立OSA標籤所需欄位：{missing}")
    data = frame.copy()
    data["_osa_now"] = _bool(data["has_obstructive_apnea"]) | _bool(data["has_hypopnea"])
    data["_non_osa_apnea_now"] = _bool(data["has_central_apnea"]) | _bool(data["has_mixed_apnea"])
    data = data.sort_values(["patient_id", "epoch_index"]).reset_index(drop=True)
    grouped = data.groupby(data["patient_id"].astype(str), sort=False)
    data[TARGET] = grouped["_osa_now"].shift(-1)
    data["next_epoch_label_ambiguous"] = grouped["_non_osa_apnea_now"].shift(-1).fillna(False)
    current_index = pd.to_numeric(data["epoch_index"], errors="coerce")
    next_index = grouped["epoch_index"].shift(-1)
    data["next_epoch_contiguous"] = pd.to_numeric(next_index, errors="coerce").eq(current_index + 1)
    return data


def _attach_epoch_event_flags(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Create event flags by true clock-time overlap, not by row-number matching."""
    data = features.copy()
    epoch_start = pd.to_datetime(data["start_time"], errors="coerce")
    epoch_end = pd.to_datetime(data["end_time"], errors="coerce")
    event_start = pd.to_datetime(events["start_time"], errors="coerce")
    event_end = pd.to_datetime(events["end_time"], errors="coerce")
    event_type = events["event_type"].astype(str).str.strip().str.upper()
    if "usable_for_event_training" in events:
        usable = _bool(events["usable_for_event_training"])
    else:
        usable = pd.Series(True, index=events.index)
    definitions = {
        "has_obstructive_apnea": {"OBSTRUCTIVE_APNEA", "A. OBSTRUCTIVE"},
        "has_hypopnea": {"HYPOPNEA"},
        "has_central_apnea": {"CENTRAL_APNEA", "A. CENTRAL"},
        "has_mixed_apnea": {"MIXED_APNEA", "A. MIXED"},
    }
    for column, accepted_types in definitions.items():
        selected = usable & event_type.isin(accepted_types) & event_start.notna() & event_end.notna()
        starts = event_start[selected].to_numpy(dtype="datetime64[ns]")
        ends = event_end[selected].to_numpy(dtype="datetime64[ns]")
        flags = np.zeros(len(data), dtype=bool)
        for index, (start, end) in enumerate(zip(epoch_start, epoch_end)):
            if pd.isna(start) or pd.isna(end) or not len(starts):
                continue
            flags[index] = bool(np.any((starts < np.datetime64(end)) & (ends > np.datetime64(start))))
        data[column] = flags
    return data


def load_training_epochs(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the baseline cohort plus every later PSG with aligned event labels.

    A study is admitted only when it has epoch-level inference features and an
    aligned Event Grid. Night-level AHI, treatment outcomes, or roster rows are
    intentionally not expanded into artificial epoch samples.
    """
    base = pd.read_csv(root / "data" / "processed" / "model_dataset.csv")
    frames = [base]
    included = set(base["patient_id"].astype(str).unique())
    additions: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    inference_root = root / "data" / "inference"
    processed_root = root / "data" / "processed"
    candidate_ids = sorted(
        {path.name for path in processed_root.iterdir() if path.is_dir() and not path.name.startswith("_")}
        | {
            path.name for path in inference_root.iterdir()
            if path.is_dir() and not path.name.startswith("_") and " - " in path.name
        }
    )
    for patient_id in candidate_ids:
        if patient_id in included:
            continue
        feature_path = inference_root / patient_id / "inference_features.csv"
        event_path = processed_root / patient_id / "events_aligned.csv"
        if not feature_path.exists() or not event_path.exists():
            missing = []
            if not feature_path.exists():
                missing.append("逐epoch PSG特徵")
            if not event_path.exists():
                missing.append("對齊Event Grid")
            exclusions.append({"patient_id": patient_id, "reason": "缺少" + "、".join(missing)})
            continue
        features = pd.read_csv(feature_path)
        events = pd.read_csv(event_path)
        required = {"patient_id", "epoch_index", "start_time", "end_time"}
        if not required.issubset(features.columns) or events.empty:
            exclusions.append({"patient_id": patient_id, "reason": "逐epoch時間軸或事件內容無效"})
            continue
        labelled = _attach_epoch_event_flags(features, events)
        labelled["patient_id"] = patient_id
        frames.append(labelled)
        included.add(patient_id)
        additions.append({"patient_id": patient_id, "epoch_rows": int(len(labelled)), "source": "inference_features+aligned_event_grid"})
    combined = pd.concat(frames, ignore_index=True, sort=False)
    provenance = {
        "baseline_patient_count": int(base["patient_id"].nunique()),
        "added_patient_count": len(additions),
        "added_studies": additions,
        "excluded_studies": exclusions,
        "admission_rule": "requires epoch-level PSG features plus time-aligned Event Grid",
    }
    return combined, provenance


def _feature_frame(data: pd.DataFrame, feature_names: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if feature_names is None:
        feature_names = [name for name in PREFERRED_FEATURES if name in data.columns]
        if len(feature_names) < 8:
            raise ValueError("可用的PSG呼吸特徵少於8個，無法可靠建立單向LSTM")
        feature_names += [f"stage_{stage}" for stage in STAGE_LEVELS]
    values = pd.DataFrame(index=data.index)
    for name in feature_names:
        if name.startswith("stage_"):
            stage = name.removeprefix("stage_")
            values[name] = data.get("stage", pd.Series("", index=data.index)).astype(str).eq(stage).astype(float)
        else:
            values[name] = pd.to_numeric(data.get(name), errors="coerce")
    return values, feature_names


def _split_patients(data: pd.DataFrame, manifest_path: Path, root: Path) -> dict[str, list[str]]:
    summary = data.groupby(data["patient_id"].astype(str))[TARGET].agg(["sum", "count"])
    patients = summary.index.to_numpy()
    if len(patients) < 10:
        raise ValueError("患者層級獨立測試至少需要10位患者")
    if manifest_path.exists():
        persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = {
            name: [patient for patient in persisted.get(name, []) if patient in set(patients)]
            for name in ("train", "validation", "test", "external_test")
        }
        assigned = set(split["train"] + split["validation"] + split["test"])
        external_manifest_root = root / "data" / "continual_learning" / "external_epoch_psg"
        external_test: set[str] = set(split.get("external_test", []))
        external_train: set[str] = set()
        external_validation: set[str] = set()
        # Sort manifests so a canonical repartition manifest (named ``manifest_zz_*``)
        # deterministically overrides older shard manifests.
        manifest_files = sorted(external_manifest_root.glob("manifest*.csv")) if external_manifest_root.exists() else []
        if manifest_files:
            external = pd.concat([pd.read_csv(path) for path in manifest_files], ignore_index=True).drop_duplicates("study_id", keep="last")
            completed = external[external.get("status", "").astype(str).eq("COMPLETED")]
            external_test.update(completed.loc[completed["split"].eq("external_test"), "study_id"].astype(str))
            external_train.update(completed.loc[completed["split"].eq("train"), "study_id"].astype(str))
            external_validation.update(completed.loc[completed["split"].eq("validation"), "study_id"].astype(str))
        available = set(patients)
        split["external_test"] = sorted(external_test & available)
        split["train"].extend(sorted((external_train & available) - assigned))
        split["validation"].extend(sorted((external_validation & available) - assigned))
        assigned = set(split["train"] + split["validation"] + split["test"] + split["external_test"])
        # Ordinary newly analysed hospital patients enter development only.
        split["train"].extend(sorted(available - assigned))
        return split
    strata = pd.qcut(summary["sum"].rank(method="first"), q=min(4, len(patients)), labels=False)
    train_val, test = train_test_split(
        patients, test_size=0.20, random_state=RANDOM_SEED, stratify=strata,
    )
    train_val_strata = strata.loc[train_val]
    train, valid = train_test_split(
        train_val, test_size=0.25, random_state=RANDOM_SEED,
        stratify=train_val_strata,
    )
    split = {"train": sorted(train.tolist()), "validation": sorted(valid.tolist()), "test": sorted(test.tolist()), "external_test": []}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    return split


def _make_sequences(
    data: pd.DataFrame,
    scaled_values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    patients: list[str] = []
    row_indices: list[int] = []
    for patient_id, positions in data.groupby(data["patient_id"].astype(str), sort=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        epochs = pd.to_numeric(data.loc[positions, "epoch_index"], errors="coerce").to_numpy()
        for offset in range(window - 1, len(positions)):
            window_positions = positions[offset - window + 1:offset + 1]
            if not np.all(np.diff(epochs[offset - window + 1:offset + 1]) == 1):
                continue
            row = data.loc[positions[offset]]
            if not bool(row["next_epoch_contiguous"]) or bool(row["next_epoch_label_ambiguous"]) or pd.isna(row[TARGET]):
                continue
            sequences.append(scaled_values[window_positions])
            labels.append(int(bool(row[TARGET])))
            patients.append(str(patient_id))
            row_indices.append(int(positions[offset]))
    return (
        np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64),
        np.asarray(patients, dtype=str), np.asarray(row_indices, dtype=np.int64),
    )


def _metric_set(truth: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(truth, probability)),
        "average_precision": float(average_precision_score(truth, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "true_positive": int(tp), "false_positive": int(fp),
        "true_negative": int(tn), "false_negative": int(fn),
    }


def _choose_threshold(truth: np.ndarray, probability: np.ndarray) -> float:
    # A false-positive controlled operating point is selected only on the
    # validation patients. The locked test patients remain untouched. Among
    # thresholds meeting both constraints, preserve as much recall as possible.
    constrained: list[tuple[float, float, float, float]] = []
    fallback: tuple[float, float, float] | None = None
    for threshold in np.arange(0.05, 0.951, 0.005):
        prediction = probability >= threshold
        score = balanced_accuracy_score(truth, prediction)
        recall = recall_score(truth, prediction, zero_division=0)
        specificity = recall_score(truth, prediction, pos_label=0, zero_division=0)
        precision = precision_score(truth, prediction, zero_division=0)
        # Formal high-confidence operating point. The holdout groups are never
        # consulted here; this stricter validation policy prioritises avoiding
        # false alerts before maximising recall.
        if specificity >= 0.91 and precision >= 0.86:
            constrained.append((float(recall), float(score), float(precision), float(threshold)))
        candidate = (float(score), float(recall), float(threshold))
        if fallback is None or candidate > fallback:
            fallback = candidate
    if constrained:
        return max(constrained)[3]
    return fallback[2] if fallback else 0.50


def train(project_root: str | Path, config: OsaNextEpochConfig | None = None) -> dict[str, Any]:
    config = config or OsaNextEpochConfig()
    root = Path(project_root).resolve()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    source_epochs, provenance = load_training_epochs(root)
    data = add_strict_osa_target(source_epochs)
    data = data[data["next_epoch_contiguous"] & ~data["next_epoch_label_ambiguous"] & data[TARGET].notna()].reset_index(drop=True)
    split = _split_patients(data, root / "models" / "osa_next_epoch_lstm" / "patient_split.json", root)
    raw_features, feature_names = _feature_frame(data)
    train_rows = data["patient_id"].astype(str).isin(split["train"]).to_numpy()
    medians = raw_features.loc[train_rows].median(numeric_only=True).fillna(0.0)
    imputed = raw_features.fillna(medians).fillna(0.0)
    means = imputed.loc[train_rows].mean()
    scales = imputed.loc[train_rows].std().replace(0, 1).fillna(1.0)
    scaled = ((imputed - means) / scales).to_numpy(dtype=np.float32)
    x, y, groups, row_indices = _make_sequences(data, scaled, config.window_epochs)
    masks = {name: np.isin(groups, patients) for name, patients in split.items()}
    if any(y[mask].min() == y[mask].max() for mask in masks.values()):
        raise ValueError("患者分組後至少一組沒有同時包含OSA陽性與陰性epoch")

    model = UniLSTM(x.shape[2], config.hidden_size, config.dropout)
    positives = max(int(y[masks["train"]].sum()), 1)
    negatives = int(masks["train"].sum()) - positives
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[masks["train"]]), torch.from_numpy(y[masks["train"]].astype(np.float32))),
        batch_size=config.batch_size, shuffle=True,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    wait = 0
    history: list[dict[str, float | int]] = []
    valid_x = torch.from_numpy(x[masks["validation"]])
    valid_y = torch.from_numpy(y[masks["validation"]].astype(np.float32))
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(valid_x), valid_y))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": valid_loss})
        if valid_loss < best_loss - 1e-4:
            best_loss = valid_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("單向LSTM未產生有效權重")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    threshold = _choose_threshold(y[masks["validation"]], probabilities[masks["validation"]])

    version = "osa_next_epoch_lstm_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    model_root = root / "models" / "osa_next_epoch_lstm"
    model_root.mkdir(parents=True, exist_ok=True)
    bundle_path = model_root / "model.pt"
    torch.save({
        "state_dict": best_state, "n_features": x.shape[2], "feature_names": feature_names,
        "medians": medians.to_dict(), "means": means.to_dict(), "scales": scales.to_dict(),
        "threshold": threshold, "config": asdict(config), "model_version": version,
    }, bundle_path)
    summary: dict[str, Any] = {
        "model_version": version,
        "architecture": "single-direction causal LSTM",
        "target": TARGET,
        "target_definition": "下一個30秒epoch出現阻塞型apnea或hypopnea；中央型與混合型apnea排除",
        "window_epochs": config.window_epochs,
        "history_seconds": config.window_epochs * 30,
        "prediction_horizon_seconds": 30,
        "patient_split": split,
        "patient_count": int(data["patient_id"].nunique()),
        "training_data_provenance": provenance,
        "sequence_count": int(len(y)),
        "positive_sequence_count": int(y.sum()),
        "threshold_selected_on": "validation patients only",
        "threshold_policy": "maximize validation recall subject to specificity>=0.91 and precision>=0.86; fallback=max balanced accuracy",
        "threshold": threshold,
        "validation_metrics": _metric_set(y[masks["validation"]], probabilities[masks["validation"]], threshold),
        "locked_test_metrics": _metric_set(y[masks["test"]], probabilities[masks["test"]], threshold),
        "external_test_metrics": (
            _metric_set(y[masks["external_test"]], probabilities[masks["external_test"]], threshold)
            if masks.get("external_test") is not None
            and int(masks["external_test"].sum()) > 0
            and len(np.unique(y[masks["external_test"]])) == 2
            else None
        ),
        "training_history": history,
        "clinical_validation": False,
        "limitations": [
            "單中心回溯性資料", "患者數有限", "尚未完成外部及前瞻性驗證",
            "hypopnea依OSA-only研究族群納入，來源資料未提供中央型hypopnea子型",
        ],
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    (model_root / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def predict_patient(project_root: str | Path, patient_id: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    model_root = root / "models" / "osa_next_epoch_lstm"
    bundle = torch.load(model_root / "model.pt", map_location="cpu", weights_only=False)
    data = pd.read_csv(root / "data" / "inference" / patient_id / "inference_features.csv")
    data = data.sort_values("epoch_index").reset_index(drop=True)
    raw, _ = _feature_frame(data, list(bundle["feature_names"]))
    medians = pd.Series(bundle["medians"])
    means = pd.Series(bundle["means"])
    scales = pd.Series(bundle["scales"]).replace(0, 1)
    scaled = ((raw.fillna(medians).fillna(0.0) - means) / scales).to_numpy(dtype=np.float32)
    window = int(bundle["config"]["window_epochs"])
    sequence_rows: list[np.ndarray] = []
    end_rows: list[int] = []
    epochs = pd.to_numeric(data["epoch_index"], errors="coerce").to_numpy()
    for index in range(window - 1, len(data)):
        if np.all(np.diff(epochs[index - window + 1:index + 1]) == 1):
            sequence_rows.append(scaled[index - window + 1:index + 1])
            end_rows.append(index)
    model = UniLSTM(int(bundle["n_features"]), int(bundle["config"]["hidden_size"]), float(bundle["config"]["dropout"]))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    probability = np.array([], dtype=float)
    if sequence_rows:
        with torch.no_grad():
            probability = torch.sigmoid(model(torch.from_numpy(np.asarray(sequence_rows, dtype=np.float32)))).numpy()
    threshold = float(bundle["threshold"])
    output = data.loc[end_rows, [c for c in ("patient_id", "epoch_index", "start_time", "end_time", "stage") if c in data]].copy()
    output["history_start_epoch"] = output["epoch_index"].astype(int) - window + 1
    output["predicted_epoch_index"] = output["epoch_index"].astype(int) + 1
    output["osa_next_30s_probability"] = probability
    output["osa_next_30s_alert"] = probability >= threshold
    destination = root / "data" / "inference" / patient_id / "osa_next_epoch_lstm"
    destination.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination / "osa_next_epoch_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "patient_id": patient_id, "model_version": bundle["model_version"],
        "architecture": "single-direction causal LSTM", "window_epochs": window,
        "history_seconds": window * 30, "prediction_horizon_seconds": 30,
        "threshold": threshold, "prediction_count": len(output),
        "alert_count": int(output["osa_next_30s_alert"].sum()) if len(output) else 0,
        "maximum_probability": float(probability.max()) if len(probability) else None,
        "mean_probability": float(probability.mean()) if len(probability) else None,
        "interpretation": "使用前N個30秒epoch預測下一個30秒epoch的OSA事件風險",
        "clinical_validation": False,
    }
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
