from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.models.arousal_model import (
    ArousalModelConfig,
    build_arousal_pipeline,
    sanitize_feature_frame,
)
from src.models.two_stage_sequence import predict_bundle


TARGET = "predict_apnea_next_60s"
THRESHOLD = 0.35
HIGH_SENSITIVITY_RECALL_TARGET = 0.90


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV safely even while Streamlit briefly has the old file open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        last_error = None
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.15 * (attempt + 1))
        raise last_error or PermissionError(f"無法更新 {path}")
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "false": False,
                "0": False,
                "no": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def _feature_definition(root: Path) -> tuple[list[str], list[str]]:
    path = root / "models" / "arousal_next_30s" / "feature_columns.json"
    definition = json.loads(path.read_text(encoding="utf-8"))
    return (
        list(definition["numeric_columns"]),
        list(definition["categorical_columns"]),
    )


def build_baseline_target(data: pd.DataFrame) -> pd.Series:
    """Whether any apnea is present in either of the next two 30-second epochs."""
    if "has_any_apnea" not in data.columns:
        apnea_columns = [
            column
            for column in (
                "has_obstructive_apnea",
                "has_central_apnea",
                "has_mixed_apnea",
            )
            if column in data.columns
        ]
        if not apnea_columns:
            raise ValueError("基礎資料沒有 apnea 事件標籤")
        current = pd.concat(
            [_to_bool(data[column]) for column in apnea_columns],
            axis=1,
        ).any(axis=1)
    else:
        current = _to_bool(data["has_any_apnea"])
    working = pd.DataFrame(
        {
            "patient_id": data["patient_id"].astype(str),
            "epoch_index": pd.to_numeric(data["epoch_index"], errors="coerce"),
            "current_apnea": current,
        },
        index=data.index,
    ).sort_values(["patient_id", "epoch_index"])
    grouped = working.groupby("patient_id", sort=False)["current_apnea"]
    target = grouped.shift(-1).fillna(False) | grouped.shift(-2).fillna(False)
    return target.reindex(data.index).fillna(False).astype(int)


def label_patient_from_events(
    features: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.Series:
    event_type = events.get("event_type", pd.Series(dtype="string")).astype(str)
    apnea = events[event_type.str.contains("APNEA", case=False, na=False)].copy()
    starts = pd.to_numeric(apnea.get("edf_start_seconds"), errors="coerce")
    starts = np.sort(starts.dropna().to_numpy(dtype=float))
    if "edf_end_seconds" in features.columns:
        epoch_end = pd.to_numeric(features["edf_end_seconds"], errors="coerce")
    elif "end_time" in features.columns:
        epoch_end = pd.to_numeric(features["epoch_index"], errors="coerce") * 30 + 30
    else:
        epoch_end = pd.to_numeric(features["epoch_index"], errors="coerce") * 30 + 30
    labels = np.zeros(len(features), dtype=int)
    if starts.size:
        values = epoch_end.to_numpy(dtype=float)
        next_index = np.searchsorted(starts, values, side="right")
        valid = next_index < len(starts)
        next_start = np.full(len(values), np.inf)
        next_start[valid] = starts[next_index[valid]]
        labels = ((next_start > values) & (next_start <= values + 60.0)).astype(int)
    return pd.Series(labels, index=features.index, name=TARGET)


class Apnea60Service:
    def __init__(self, project_root: str | Path):
        self.root = Path(project_root).resolve()
        self.model_root = self.root / "models" / "apnea_next_60s"
        self.champion = self.model_root / "champion"
        self.archive = self.model_root / "archive"
        self.registry_file = self.model_root / "registry.json"
        self.approved = (
            self.root
            / "data"
            / "continual_learning"
            / "apnea60"
            / "approved"
        )
        for folder in (self.champion, self.archive, self.approved):
            folder.mkdir(parents=True, exist_ok=True)
        if not self.registry_file.exists():
            _atomic_json(
                self.registry_file,
                {
                    "champion": None,
                    "previous_champion": None,
                    "processed_studies": [],
                    "models": [],
                    "updated_at": _now(),
                },
            )

    def registry(self) -> dict[str, Any]:
        return json.loads(self.registry_file.read_text(encoding="utf-8"))

    def _dataset(self) -> tuple[pd.DataFrame, list[str], list[str]]:
        base_path = self.root / "data" / "processed" / "model_dataset.csv"
        base = pd.read_csv(base_path)
        base[TARGET] = build_baseline_target(base)
        if "study_id" not in base.columns:
            base["study_id"] = "baseline_" + base["patient_id"].astype(str)
        frames = [base]
        for path in sorted(self.approved.glob("*.csv")):
            frames.append(pd.read_csv(path))
        data = pd.concat(frames, ignore_index=True, sort=False)
        data = data.drop_duplicates(
            ["patient_id", "study_id", "epoch_index"],
            keep="last",
        )
        numeric, categorical = _feature_definition(self.root)
        missing = set(numeric + categorical) - set(data.columns)
        if missing:
            raise ValueError(f"apnea 60秒資料缺少特徵：{sorted(missing)}")
        return data, numeric, categorical

    @staticmethod
    def _metrics(
        truth: np.ndarray,
        probability: np.ndarray,
        threshold: float = THRESHOLD,
    ) -> dict[str, float]:
        prediction = (probability >= threshold).astype(int)
        return {
            "roc_auc": float(roc_auc_score(truth, probability)),
            "average_precision": float(average_precision_score(truth, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "specificity": float(
                recall_score(truth, prediction, pos_label=0, zero_division=0)
            ),
            "f1": float(f1_score(truth, prediction, zero_division=0)),
        }

    def train(self) -> dict[str, Any]:
        data, numeric, categorical = self._dataset()
        target = pd.to_numeric(data[TARGET], errors="raise").astype(int)
        groups = data["patient_id"].astype(str)
        features = sanitize_feature_frame(data[numeric + categorical])
        patient_count = groups.nunique()
        expected_patients = sorted(
            path.name
            for path in (self.root / "data" / "processed").iterdir()
            if path.is_dir() and not path.name.startswith("_")
        )
        included_patients = sorted(groups.unique().tolist())
        excluded_patients = [
            {
                "patient_id": patient_id,
                "reason": "沒有通過時間對齊、必要特徵或有效Apnea標籤資料門檻",
            }
            for patient_id in expected_patients
            if patient_id not in set(included_patients)
        ]
        splits = min(5, int(patient_count))
        if splits < 2 or target.nunique() < 2:
            raise RuntimeError("apnea 模型至少需要兩位病患且正負事件皆存在")
        splitter = StratifiedGroupKFold(
            n_splits=splits,
            shuffle=True,
            random_state=42,
        )
        oof = np.full(len(data), np.nan)
        valid_folds = 0
        for train_index, test_index in splitter.split(features, target, groups):
            y_test = target.iloc[test_index]
            if y_test.nunique() < 2:
                continue
            model = build_arousal_pipeline(
                numeric,
                categorical,
                ArousalModelConfig(n_estimators=300),
            )
            model.fit(features.iloc[train_index], target.iloc[train_index])
            oof[test_index] = model.predict_proba(features.iloc[test_index])[:, 1]
            valid_folds += 1
        valid = ~np.isnan(oof)
        selected_threshold = THRESHOLD
        if valid.sum() and target[valid].nunique() == 2:
            y_valid = target[valid].to_numpy()
            p_valid = oof[valid]
            candidates: list[tuple[float, float, float, float]] = []
            for threshold in np.arange(0.01, 0.991, 0.001):
                prediction = (p_valid >= threshold).astype(int)
                recall = recall_score(y_valid, prediction, zero_division=0)
                if recall >= HIGH_SENSITIVITY_RECALL_TARGET:
                    specificity = recall_score(
                        y_valid, prediction, pos_label=0, zero_division=0
                    )
                    precision = precision_score(y_valid, prediction, zero_division=0)
                    candidates.append(
                        (float(threshold), float(specificity), float(precision), float(recall))
                    )
            if candidates:
                # Retain the high-sensitivity requirement, then jointly prefer
                # fewer false alerts (specificity) and better alert precision.
                selected_threshold = max(
                    candidates, key=lambda item: (item[1] + item[2], item[1], item[2], item[0])
                )[0]
        metrics = (
            self._metrics(target[valid].to_numpy(), oof[valid], selected_threshold)
            if valid.sum() and target[valid].nunique() == 2
            else {}
        )
        final_model = build_arousal_pipeline(
            numeric,
            categorical,
            ArousalModelConfig(n_estimators=300),
        )
        final_model.fit(features, target)
        version = "apnea60_" + datetime.now().strftime("%Y%m%dT%H%M%S")
        candidate = self.model_root / "candidates" / version
        candidate.mkdir(parents=True, exist_ok=False)
        joblib.dump(final_model, candidate / "model.joblib")
        _atomic_json(
            candidate / "feature_columns.json",
            {
                "numeric_columns": numeric,
                "categorical_columns": categorical,
                "target_column": TARGET,
            },
        )
        summary = {
            "model_version": version,
            "target": TARGET,
            "prediction_horizon_seconds": 60,
            "row_count": len(data),
            "patient_count": int(patient_count),
            "expected_patient_count": len(expected_patients),
            "included_patients": included_patients,
            "excluded_patients": excluded_patients,
            "positive_count": int(target.sum()),
            "negative_count": int((target == 0).sum()),
            "threshold": selected_threshold,
            "threshold_policy": "OOF recall >= 0.90, then jointly maximize specificity and precision",
            "high_sensitivity_recall_target": HIGH_SENSITIVITY_RECALL_TARGET,
            "valid_cv_folds": valid_folds,
            "metrics": metrics,
            "created_at": _now(),
        }
        _atomic_json(candidate / "training_summary.json", summary)
        registry = self.registry()
        old_summary_file = self.champion / "training_summary.json"
        old_summary = (
            json.loads(old_summary_file.read_text(encoding="utf-8"))
            if old_summary_file.exists()
            else {}
        )
        old_auc = old_summary.get("metrics", {}).get("roc_auc")
        summary["promotion_passed"] = False
        summary["promotion_allowed"] = False
        summary["promotion_block_reason"] = (
            "尚無獨立外部病人驗證、前瞻性臨床驗證與治理核准；訓練只能建立研究型 Challenger。"
        )
        summary["previous_roc_auc"] = old_auc
        registry["latest_challenger"] = version
        registry["models"].append(summary)
        registry["updated_at"] = _now()
        _atomic_json(self.registry_file, registry)
        return summary

    def _active_model_dir(self, research_preview: bool = False) -> tuple[Path, str]:
        if research_preview:
            latest = self.registry().get("latest_challenger")
            candidate = self.model_root / "candidates" / str(latest)
            if latest and (candidate / "model.joblib").exists():
                return candidate, str(latest)
        return self.champion, str(self.registry().get("champion"))

    def ensure_model(self, research_preview: bool = False) -> dict[str, Any]:
        active_dir, _ = self._active_model_dir(research_preview)
        if not (active_dir / "model.joblib").exists():
            raise RuntimeError("尚無通過獨立驗證與治理核准的 60 秒 Champion；研究型 Challenger 不得自動部署")
        return json.loads(
            (active_dir / "training_summary.json").read_text(encoding="utf-8")
        )

    def predict_patient(self, patient_id: str, research_preview: bool = False) -> dict[str, Any]:
        self.ensure_model(research_preview)
        active_dir, active_version = self._active_model_dir(research_preview)
        feature_path = (
            self.root / "data" / "inference" / patient_id / "inference_features.csv"
        )
        if not feature_path.exists():
            raise FileNotFoundError(f"找不到病患推論特徵：{feature_path}")
        data = pd.read_csv(feature_path)
        metadata_path = (
            self.root / "data" / "processed" / patient_id / "patient_metadata.csv"
        )
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            if not metadata.empty:
                data["age"] = pd.to_numeric(metadata.iloc[0].get("age"), errors="coerce")
                data["BMI"] = pd.to_numeric(metadata.iloc[0].get("BMI"), errors="coerce")
        summary_file = active_dir / "training_summary.json"
        model_summary = json.loads(summary_file.read_text(encoding="utf-8"))
        alert_threshold = float(model_summary.get("threshold", THRESHOLD))
        definition = json.loads(
            (active_dir / "feature_columns.json").read_text(encoding="utf-8")
        )
        columns = definition["numeric_columns"] + definition["categorical_columns"]
        frame = sanitize_feature_frame(data.reindex(columns=columns))
        sequence_bundle = self.model_root / "two_stage_sequence.joblib"
        sequence_summary = None
        temporal_alerts = None
        distribution_guard_fallback = False
        use_sequence_bundle = False
        if sequence_bundle.exists():
            challenger = joblib.load(sequence_bundle).get("summary", {})
            champion_metrics = model_summary.get("metrics", {})
            use_sequence_bundle = (
                float(challenger.get("recall", 0.0)) >= float(model_summary.get("high_sensitivity_recall_target", 0.90))
                and float(challenger.get("specificity", 0.0)) >= float(champion_metrics.get("specificity", 1.0))
                and float(challenger.get("precision", 0.0)) >= float(champion_metrics.get("precision", 1.0))
            )
        if use_sequence_bundle:
            probability, temporal_alerts, sequence_summary = predict_bundle(sequence_bundle, data)
            alert_threshold = float(sequence_summary["threshold"])
            if float(np.mean(probability >= alert_threshold)) > 0.80:
                # An almost-continuous alert is treated as deployment drift,
                # not silently accepted. Fall back to the validated champion
                # and expose the condition for review.
                probability = joblib.load(active_dir / "model.joblib").predict_proba(frame)[:, 1]
                alert_threshold = float(model_summary.get("threshold", THRESHOLD))
                temporal_alerts = None
                sequence_summary = None
                distribution_guard_fallback = True
        else:
            probability = joblib.load(active_dir / "model.joblib").predict_proba(frame)[:, 1]
        output = data[
            [column for column in ("patient_id", "epoch_index", "start_time", "end_time", "stage") if column in data]
        ].copy()
        output["apnea_next_60s_probability"] = probability
        quality_valid = pd.Series(True, index=data.index, dtype=bool)
        for quality_column in (
            "quality_core_features_valid",
            "usable_for_core_respiratory_model",
        ):
            if quality_column in data.columns:
                values = data[quality_column]
                if values.dtype == object:
                    values = values.astype(str).str.strip().str.lower().isin(
                        {"true", "1", "yes"}
                    )
                else:
                    values = values.fillna(False).astype(bool)
                quality_valid &= values
        risk_positive = probability >= alert_threshold
        output["apnea_signal_quality_valid"] = quality_valid.to_numpy()
        output["apnea_sensor_check_alert"] = (~quality_valid).to_numpy()
        if temporal_alerts is None:
            # 單一 epoch 超過低閾值只列入觀察，不直接形成警報。
            # 正式通知要求最近 3 個 epoch 至少 2 個持續高風險，並在
            # distribution fallback 時使用較保守的最低機率，降低警報疲勞。
            # The model threshold remains the high-sensitivity screening
            # threshold. User-facing notifications require stronger evidence
            # to reduce alarm fatigue and are deliberately kept separate.
            notification_threshold = max(alert_threshold, 0.20)
            persistent_risk = pd.Series(
                probability >= notification_threshold,
                index=data.index,
            )
            candidate_alerts = (
                persistent_risk.astype(int).rolling(5, min_periods=5).sum() >= 4
            ).to_numpy()
        else:
            notification_threshold = max(alert_threshold, 0.20)
            persistent_risk = pd.Series(
                probability >= notification_threshold,
                index=data.index,
            )
            persistence_gate = (
                persistent_risk.astype(int).rolling(5, min_periods=5).sum() >= 4
            ).to_numpy()
            candidate_alerts = np.asarray(temporal_alerts, dtype=bool) & persistence_gate
        output["apnea_next_60s_alert"] = candidate_alerts & quality_valid.to_numpy()
        output["apnea_high_risk_but_signal_invalid"] = (
            risk_positive & (~quality_valid.to_numpy())
        )
        output["apnea_risk_level"] = np.select(
            [
                probability >= max(0.70, alert_threshold + 0.30),
                probability >= alert_threshold,
                probability >= max(0.10, alert_threshold - 0.10),
            ],
            ["VERY_HIGH", "HIGH", "WATCH"],
            default="LOW",
        )
        output_dir = self.root / "data" / "inference" / patient_id / "apnea_next_60s"
        if research_preview:
            output_dir = self.root / "data" / "inference" / patient_id / "research_preview" / "apnea_60s"
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_csv(output, output_dir / "apnea_next_60s_predictions.csv")
        registry = self.registry()
        summary = {
            "patient_id": patient_id,
            "model_version": active_version,
            "output_mode": (
                "research_latest_challenger"
                if research_preview
                else "validated_champion"
            ),
            "prediction_horizon_seconds": 60,
            "interpretation": "預測目前 epoch 結束後的未來60秒內是否出現 apnea",
            "threshold": alert_threshold,
            "notification_threshold": notification_threshold,
            "persistence_policy": "4_of_5_consecutive_epochs_at_notification_threshold",
            "threshold_policy": model_summary.get("threshold_policy"),
            "high_sensitivity_recall_target": model_summary.get(
                "high_sensitivity_recall_target"
            ),
            "epoch_count": len(output),
            "alert_count": int(output["apnea_next_60s_alert"].sum()),
            "sensor_check_alert_count": int(output["apnea_sensor_check_alert"].sum()),
            "alert_rate": float(output["apnea_next_60s_alert"].mean()),
            "maximum_probability": float(np.max(probability)),
            "mean_probability": float(np.mean(probability)),
            "generated_at": _now(),
            "alert_logic": (
                "two_stage_xgboost_bilstm_calibrated_apnea_persistence"
                if sequence_summary is not None
                else (
                    "legacy_champion_distribution_guard_fallback"
                    if distribution_guard_fallback
                    else "legacy_single_epoch"
                )
            ),
            "distribution_guard_fallback": distribution_guard_fallback,
        }
        _atomic_json(output_dir / "apnea_next_60s_summary.json", summary)
        return summary

    def ingest_patient_labels(self, patient_id: str) -> dict[str, Any]:
        feature_path = (
            self.root / "data" / "inference" / patient_id / "inference_features.csv"
        )
        event_path = (
            self.root / "data" / "processed" / patient_id / "events_aligned.csv"
        )
        if not feature_path.exists() or not event_path.exists():
            return {"status": "pending_label", "reason": "缺少特徵或 Event Grid"}
        study_hash = hashlib.sha256(
            (_sha256(feature_path) + _sha256(event_path)).encode("utf-8")
        ).hexdigest()
        registry = self.registry()
        if study_hash in registry.get("processed_studies", []):
            return {"status": "already_learned", "study_hash": study_hash}
        features = pd.read_csv(feature_path)
        events = pd.read_csv(event_path)
        features = features.copy()
        features[TARGET] = label_patient_from_events(features, events)
        features["study_id"] = study_hash[:16]
        destination = self.approved / f"{patient_id}_{study_hash[:12]}.csv"
        features.to_csv(destination, index=False, encoding="utf-8-sig")
        registry["processed_studies"].append(study_hash)
        registry["updated_at"] = _now()
        _atomic_json(self.registry_file, registry)
        return {
            "status": "approved",
            "patient_id": patient_id,
            "study_hash": study_hash,
            "row_count": len(features),
            "positive_count": int(features[TARGET].sum()),
            "file": str(destination),
        }

    def process_and_learn(self, patient_id: str) -> dict[str, Any]:
        prediction = self.predict_patient(patient_id)
        ingestion = self.ingest_patient_labels(patient_id)
        training: dict[str, Any] | None = None
        if ingestion.get("status") in {"approved", "already_learned"}:
            training = self.train()
            if ingestion.get("status") == "already_learned":
                ingestion = {
                    **ingestion,
                    "status": "retrained_deduplicated_dataset",
                    "reason": (
                        "相同 PSG/Event Grid 未重複加入樣本；"
                        "已使用去重後完整訓練集重新建立 Challenger"
                    ),
                }
        result = {
            "patient_id": patient_id,
            "prediction_before_learning": prediction,
            "learning_ingestion": ingestion,
            "model_update": training,
            "completed_at": _now(),
        }
        output = self.root / "data" / "inference" / patient_id / "apnea_next_60s"
        _atomic_json(output / "continual_learning_receipt.json", result)
        return result
