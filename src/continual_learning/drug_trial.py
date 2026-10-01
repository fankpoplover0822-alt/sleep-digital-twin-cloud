from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import edfio
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


MODEL_KEY = "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _study_id(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"(\d{8}T\d{6}\s*-\s*[0-9a-fA-F]+)", text)
    return re.sub(r"\s+", " ", match.group(1)) if match else text


def _sex(value: Any) -> float:
    text = str(value).strip().lower()
    if text in {"1", "m", "male", "男"}:
        return 1.0
    if text in {"2", "0", "f", "female", "女"}:
        return 0.0
    return 0.5


def _read_psg_summary(inner_bytes: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        names = inner.namelist()
        event_name = next(n for n in names if "event grid" in n.lower())
        stage_name = next(n for n in names if "signal grid" in n.lower())
        edf_name = next(n for n in names if n.lower().endswith(".edf"))
        events = pd.read_excel(io.BytesIO(inner.read(event_name)))
        stages = pd.read_excel(io.BytesIO(inner.read(stage_name)))
        edf = edfio.read_edf(io.BytesIO(inner.read(edf_name)), lazy_load_data=False)
    event_col = next((c for c in events.columns if str(c).strip().lower() == "event"), None)
    sleep_col = next((c for c in stages.columns if str(c).strip().lower() == "sleep"), None)
    if event_col is None or sleep_col is None:
        raise ValueError("Event Grid 或 Signal Grid 缺少必要欄位")
    labels = events[event_col].fillna("").astype(str).str.lower()
    respiratory = labels.str.contains(r"apnea|hypopnea", regex=True)
    stage = stages[sleep_col].fillna("").astype(str).str.strip().str.lower()
    sleep_epochs = int(stage.isin({"n1", "n2", "n3", "rem"}).sum())
    sleep_hours = sleep_epochs * 30.0 / 3600.0
    event_count = int(respiratory.sum())
    ahi = event_count / sleep_hours if sleep_hours > 0 else np.nan
    stage_counts = stage.value_counts()
    denominator = max(sleep_epochs, 1)
    result = {
        "ahi": float(ahi),
        "respiratory_event_count": event_count,
        "sleep_hours": float(sleep_hours),
        "hypopnea_count": int(labels.str.contains("hypopnea", regex=False).sum()),
        "obstructive_apnea_count": int(labels.str.contains(r"obstructive.*apnea|a\. obstructive", regex=True).sum()),
        "central_apnea_count": int(labels.str.contains(r"central.*apnea|a\. central", regex=True).sum()),
        "mixed_apnea_count": int(labels.str.contains(r"mixed.*apnea|a\. mixed", regex=True).sum()),
        "n1_fraction": float(stage_counts.get("n1", 0) / denominator),
        "n2_fraction": float(stage_counts.get("n2", 0) / denominator),
        "n3_fraction": float(stage_counts.get("n3", 0) / denominator),
        "rem_fraction": float(stage_counts.get("rem", 0) / denominator),
        "wake_epoch_count": int(stage_counts.get("wake", 0)),
        "edf_duration_hours": float(edf.duration / 3600.0),
        "edf_channel_count": int(len(edf.signals)),
    }
    signal_groups = {
        "eeg": {"c3", "c4", "f3", "f4", "o1", "o2"},
        "airflow": {"flow", "thermistor"},
        "respiratory_belt": {"inductance abdom", "inductance thora"},
        "heart_rate": {"heart rate", "pulse"},
        "spo2": {"saturation", "spo2"},
    }
    for group, accepted in signal_groups.items():
        samples: list[np.ndarray] = []
        for signal in edf.signals:
            if str(signal.label).strip().lower() not in accepted:
                continue
            values = np.asarray(signal.data, dtype=float)
            stride = max(int(round(float(signal.sampling_frequency) * 2.0)), 1)
            sampled = values[::stride]
            sampled = sampled[np.isfinite(sampled)]
            if sampled.size:
                samples.append(sampled)
        if samples:
            values = np.concatenate(samples)
            result[f"edf_{group}_median"] = float(np.median(values))
            result[f"edf_{group}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
            result[f"edf_{group}_diff_mad"] = float(
                np.median(np.abs(np.diff(values) - np.median(np.diff(values))))
            ) if values.size > 1 else 0.0
        else:
            result[f"edf_{group}_median"] = np.nan
            result[f"edf_{group}_iqr"] = np.nan
            result[f"edf_{group}_diff_mad"] = np.nan
    return result


def _clinical_package_hash(payload: bytes) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        members = sorted(
            name for name in inner.namelist()
            if name.lower().endswith((".edf", ".xls", ".xlsx"))
        )
        for name in members:
            digest.update(Path(name).name.lower().encode("utf-8"))
            digest.update(hashlib.sha256(inner.read(name)).digest())
    return digest.hexdigest()


def _features(frame: pd.DataFrame) -> np.ndarray:
    age = pd.to_numeric(frame["age"], errors="coerce").fillna(frame["age"].median())
    bmi = pd.to_numeric(frame["bmi"], errors="coerce").fillna(frame["bmi"].median())
    ahi = pd.to_numeric(frame["baseline_ahi"], errors="coerce")
    sex = frame["sex_binary"].astype(float)
    drug = frame["drug"].astype(float)
    columns = [
        age, bmi, ahi, sex, drug,
        drug * age, drug * bmi, drug * ahi, drug * sex,
    ]
    representation_columns = sorted(
        column for column in frame.columns if str(column).startswith("psg_repr_")
    )
    for column in representation_columns:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        columns.extend([values, drug * values])
    return np.column_stack(columns)


def train_drug_trial_challenger(
    project_root: str | Path,
    roster_path: str | Path,
    archive_path: str | Path,
) -> dict[str, Any]:
    """Train a research-only drug-vs-placebo treatment-ranking challenger."""
    root = Path(project_root).resolve()
    roster_path, archive_path = Path(roster_path), Path(archive_path)
    roster = pd.read_excel(roster_path)
    roster.columns = [str(c).strip().lower().replace(" ", "_") for c in roster.columns]
    aliases = {"psg1_id": "psg1", "psg2_id": "psg2", "drug": "drug", "no": "subject_no"}
    roster = roster.rename(columns={k: v for k, v in aliases.items() if k in roster.columns})
    required = {"psg1", "psg2", "drug", "age", "bmi", "sex"}
    missing = required - set(roster.columns)
    if missing:
        raise ValueError(f"Roster 缺少欄位：{sorted(missing)}")

    packages: dict[str, bytes] = {}
    duplicates: list[str] = []
    duplicate_conflicts: list[str] = []
    package_hashes: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as outer:
        for name in outer.namelist():
            if not name.lower().endswith(".zip"):
                continue
            sid = _study_id(Path(name).name)
            payload = outer.read(name)
            if sid in packages:
                duplicates.append(sid)
                if _clinical_package_hash(payload) != package_hashes[sid]:
                    duplicate_conflicts.append(sid)
                continue
            packages[sid] = payload
            package_hashes[sid] = _clinical_package_hash(payload)

    summary_features = [
        "ahi", "respiratory_event_count", "sleep_hours", "hypopnea_count",
        "obstructive_apnea_count", "central_apnea_count", "mixed_apnea_count",
        "n1_fraction", "n2_fraction", "n3_fraction", "rem_fraction",
        "wake_epoch_count",
        "edf_duration_hours", "edf_channel_count",
        "edf_eeg_median", "edf_eeg_iqr", "edf_eeg_diff_mad",
        "edf_airflow_median", "edf_airflow_iqr", "edf_airflow_diff_mad",
        "edf_respiratory_belt_median", "edf_respiratory_belt_iqr", "edf_respiratory_belt_diff_mad",
        "edf_heart_rate_median", "edf_heart_rate_iqr", "edf_heart_rate_diff_mad",
        "edf_spo2_median", "edf_spo2_iqr", "edf_spo2_diff_mad",
    ]
    session_summaries: dict[str, dict[str, Any]] = {}
    unreadable_studies: list[dict[str, str]] = []
    for sid, payload in packages.items():
        try:
            session_summaries[sid] = _read_psg_summary(payload)
        except Exception as exc:
            unreadable_studies.append({"study_id": sid, "reason": f"{type(exc).__name__}: {exc}"})
    representation_frame = pd.DataFrame.from_dict(session_summaries, orient="index")
    representation_frame = representation_frame[summary_features].replace([np.inf, -np.inf], np.nan)
    representation_frame = representation_frame.fillna(representation_frame.median(numeric_only=True)).fillna(0.0)
    component_count = min(5, len(summary_features), max(len(representation_frame) - 1, 1))
    representation_model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=component_count, random_state=20260911)),
    ])
    representation_values = representation_model.fit_transform(representation_frame)
    representations = {
        sid: representation_values[index]
        for index, sid in enumerate(representation_frame.index)
    }

    rows: list[dict[str, Any]] = []
    missing_studies: list[dict[str, str]] = []
    for _, source in roster.iterrows():
        psg1, psg2 = _study_id(source["psg1"]), _study_id(source["psg2"])
        absent = [sid for sid in (psg1, psg2) if sid not in session_summaries]
        if absent:
            missing_studies.append({"subject_no": str(source.get("subject_no", "")), "missing": ", ".join(absent)})
            continue
        before, after = session_summaries[psg1], session_summaries[psg2]
        if not np.isfinite(before["ahi"]) or not np.isfinite(after["ahi"]) or before["ahi"] <= 0:
            continue
        row = {
            "subject_hash": hashlib.sha256(str(source.get("subject_no", len(rows))).encode()).hexdigest()[:12],
            "baseline_study_id": psg1,
            "drug": int(source["drug"]), "age": float(source["age"]), "bmi": float(source["bmi"]),
            "sex_binary": _sex(source["sex"]), "baseline_ahi": before["ahi"],
            "follow_up_ahi": after["ahi"],
            "ahi_reduction_ratio": float(np.clip((before["ahi"] - after["ahi"]) / before["ahi"], -2, 1)),
            "baseline_event_count": before["respiratory_event_count"],
            "follow_up_event_count": after["respiratory_event_count"],
        }
        for column in summary_features:
            row[f"psg_raw_{column}"] = before.get(column)
        for index, value in enumerate(representations[psg1]):
            row[f"psg_repr_{index + 1}"] = float(value)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) < 20 or frame["drug"].nunique() != 2:
        raise ValueError("完整藥物／安慰劑配對不足，無法建立研究 Challenger")
    X, y = _features(frame), frame["ahi_reduction_ratio"].to_numpy(float)
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=5.0))])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260911)
    predictions = np.full(len(frame), np.nan, dtype=float)
    raw_columns = [f"psg_raw_{column}" for column in summary_features]
    for train_index, test_index in folds.split(frame, frame["drug"]):
        fold_representation = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=component_count, random_state=20260911)),
        ])
        train_copy = frame.iloc[train_index].copy()
        test_copy = frame.iloc[test_index].copy()
        train_repr = fold_representation.fit_transform(train_copy[raw_columns])
        test_repr = fold_representation.transform(test_copy[raw_columns])
        for index in range(component_count):
            train_copy[f"psg_repr_{index + 1}"] = train_repr[:, index]
            test_copy[f"psg_repr_{index + 1}"] = test_repr[:, index]
        fold_model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=5.0))])
        fold_model.fit(_features(train_copy), y[train_index])
        predictions[test_index] = fold_model.predict(_features(test_copy))
    model.fit(X, y)
    counterfactual = frame.copy()
    counterfactual["drug"] = 0
    predicted_placebo = model.predict(_features(counterfactual))
    counterfactual["drug"] = 1
    predicted_drug = model.predict(_features(counterfactual))
    relative_effect = float(np.mean(predicted_drug - predicted_placebo))
    # Conservative, bounded research adjustment.  It changes only the
    # medication Challenger score and never the clinical Champion.
    adjustment = float(np.clip(relative_effect * 20.0, -5.0, 5.0))
    version = "drug_trial_" + datetime.now().strftime("%Y%m%dT%H%M%S%fZ")
    model_dir = root / "models" / "registry" / "treatment_trial" / version
    model_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, model_dir / "unspecified_trial_drug.joblib")
    joblib.dump(representation_model, model_dir / "psg_representation_model.joblib")
    representation_export = representation_frame.copy()
    representation_export.insert(0, "study_id", representation_export.index)
    representation_export.to_csv(
        model_dir / "all_psg_representation_training.csv", index=False, encoding="utf-8-sig"
    )
    frame.to_csv(model_dir / "training_summary_deidentified.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "model_version": version, "created_at": _now(), "model_type": "Ridge research challenger",
        "treatment": MODEL_KEY, "drug_name": "UNSPECIFIED_TRIAL_DRUG",
        "training_policy": "all_psg_unsupervised_representation_plus_paired_drug_supervision",
        "unique_psg_session_count": int(len(session_summaries)),
        "representation_training_session_count": int(len(representation_frame)),
        "representation_model": "StandardScaler_plus_PCA",
        "representation_feature_columns": summary_features,
        "representation_component_count": int(component_count),
        "unreadable_study_count": int(len(unreadable_studies)),
        "unreadable_studies": unreadable_studies,
        "complete_pair_count": int(len(frame)), "drug_count": int((frame.drug == 1).sum()),
        "placebo_count": int((frame.drug == 0).sum()), "missing_pair_count": int(len(missing_studies)),
        "missing_studies": missing_studies, "duplicate_package_ids_ignored": sorted(set(duplicates)),
        "duplicate_content_conflict_count": int(len(set(duplicate_conflicts))),
        "duplicate_content_conflicts": sorted(set(duplicate_conflicts)),
        "duplicate_validation": "SHA256_over_EDF_and_Excel_member_contents",
        "target": "PSG1_to_PSG2_AHI_reduction_ratio", "cv_mae": float(mean_absolute_error(y, predictions)),
        "validation_strategy": "patient_level_5_fold_with_fold_local_scaler_and_PCA",
        "pca_test_leakage_prevented": True,
        "raw_edf_waveform_features_used": True,
        "estimated_relative_effect": relative_effect, "research_score_adjustment_points": adjustment,
        "promotion_allowed": False, "clinical_effectiveness_claim": False,
        "affects_tonight_risk_model": False,
        "warning": "未提供試驗藥名；僅可解讀為此未指定試驗藥物相對安慰劑的研究訊號。",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pointer = model_dir.parent / "latest.json"
    pointer.write_text(json.dumps({"model_version": version}, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def load_latest_drug_trial_adjustment(project_root: str | Path) -> dict[str, Any] | None:
    # Prefer the drug component embedded in the single, combined treatment
    # Challenger.  The legacy standalone pointer remains a migration fallback.
    treatment_registry = (
        Path(project_root).resolve() / "models" / "registry" / "treatment" / "registry.json"
    )
    if treatment_registry.exists():
        registry = json.loads(treatment_registry.read_text(encoding="utf-8"))
        version = registry.get("latest_challenger")
        embedded = treatment_registry.parent / str(version) / "drug_trial" / "metadata.json"
        if embedded.exists():
            return json.loads(embedded.read_text(encoding="utf-8"))
    base = Path(project_root).resolve() / "models" / "registry" / "treatment_trial"
    pointer = base / "latest.json"
    if not pointer.exists():
        return None
    version = json.loads(pointer.read_text(encoding="utf-8")).get("model_version")
    metadata = base / str(version) / "metadata.json"
    return json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else None


def refresh_patient_representation_model(project_root: str | Path) -> dict[str, Any]:
    """Rebuild the unlabeled patient representation after each completed PSG run."""
    root = Path(project_root).resolve()
    rows: list[dict[str, Any]] = []
    numeric_keys = [
        "age", "BMI", "ahi", "hypopnea_count", "obstructive_apnea_count",
        "central_apnea_count", "position_event_ratio", "known_position_fraction",
        "coupled_3pct_fraction", "coupled_4pct_fraction", "wake_spo2", "sleep_spo2",
    ]
    for path in sorted((root / "data" / "inference").glob(
        "*/treatment_refinement/refined_treatment_recommendation.json"
    )):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        context = payload.get("patient_context", {})
        if not isinstance(context, dict):
            continue
        row = {"patient_id": payload.get("patient_id") or path.parents[2].name}
        row.update({key: context.get(key) for key in numeric_keys})
        rows.append(row)
    frame = pd.DataFrame(rows).drop_duplicates("patient_id", keep="last")
    if frame.empty:
        return {"status": "skipped", "reason": "no_patient_context"}
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=min(5, len(numeric_keys), max(len(frame) - 1, 1)), random_state=20260911)),
    ])
    values = model.fit_transform(frame[numeric_keys])
    output = frame[["patient_id"]].copy()
    for index in range(values.shape[1]):
        output[f"patient_repr_{index + 1}"] = values[:, index]
    folder = root / "models" / "registry" / "treatment_patient_representation"
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "latest.joblib")
    output.to_csv(folder / "latest_patient_embeddings.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "status": "retrained", "updated_at": _now(), "patient_count": int(len(frame)),
        "feature_columns": numeric_keys, "component_count": int(values.shape[1]),
        "label_usage": "unlabeled_representation_only",
    }
    (folder / "latest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def publish_combined_treatment_challenger(project_root: str | Path) -> dict[str, Any]:
    """Publish one treatment version containing all treatment-model heads.

    Labelled outcomes train their corresponding heads. Newly analysed patients
    without follow-up outcomes are retained in an unlabeled pool and are not
    misrepresented as supervised efficacy labels.
    """
    root = Path(project_root).resolve()
    registry_path = root / "models" / "registry" / "treatment" / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    base_version = registry.get("latest_challenger")
    base_dir = registry_path.parent / str(base_version)
    base_metadata = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8"))
    if base_metadata.get("model_family") == "combined_four_treatment_ranking":
        base_version = base_metadata.get("base_four_treatment_component_version")
        base_dir = registry_path.parent / str(base_version)
        base_metadata = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8"))
    trial_registry = root / "models" / "registry" / "treatment_trial"
    trial_pointer = trial_registry / "latest.json"
    if not trial_pointer.exists():
        raise RuntimeError("找不到藥物試驗 Challenger")
    trial_version = json.loads(trial_pointer.read_text(encoding="utf-8")).get("model_version")
    standalone_trial_dir = trial_registry / str(trial_version)
    trial_metadata_file = standalone_trial_dir / "metadata.json"
    if not trial_metadata_file.exists():
        raise RuntimeError("最新藥物試驗 Challenger 缺少 metadata.json")
    trial = json.loads(trial_metadata_file.read_text(encoding="utf-8"))

    version = "treatment_combined_" + datetime.now().strftime("%Y%m%dT%H%M%S%fZ")
    staging = registry_path.parent / f".staging-{version}"
    output = registry_path.parent / version
    shutil.copytree(base_dir, staging)
    embedded_trial = staging / "drug_trial"
    embedded_trial.mkdir(parents=True, exist_ok=True)
    shutil.copy2(standalone_trial_dir / "unspecified_trial_drug.joblib", embedded_trial)
    shutil.copy2(standalone_trial_dir / "psg_representation_model.joblib", embedded_trial)
    shutil.copy2(standalone_trial_dir / "all_psg_representation_training.csv", embedded_trial)
    shutil.copy2(standalone_trial_dir / "training_summary_deidentified.csv", embedded_trial)
    shutil.copy2(standalone_trial_dir / "metadata.json", embedded_trial)

    pap_metadata = None
    pap_registry = root / "models" / "registry" / "pap_adherence"
    pap_pointer = pap_registry / "latest.json"
    if pap_pointer.exists():
        candidate = json.loads(pap_pointer.read_text(encoding="utf-8"))
        pap_version = candidate.get("model_version")
        pap_source = pap_registry / str(pap_version)
        if pap_version and (pap_source / "metadata.json").exists():
            pap_metadata = json.loads(
                (pap_source / "metadata.json").read_text(encoding="utf-8")
            )
            shutil.copytree(pap_source, staging / "pap_adherence")

    pool_rows: list[dict[str, Any]] = []
    for path in sorted((root / "data" / "inference").glob(
        "*/treatment_refinement/refined_treatment_recommendation.json"
    )):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        context = payload.get("patient_context", {})
        if not isinstance(context, dict):
            context = {}
        pool_rows.append({
            "patient_id": payload.get("patient_id") or path.parents[2].name,
            "generated_at": payload.get("generated_at"),
            "label_status": "pending_follow_up_outcome",
            "age": context.get("age"), "sex": context.get("sex"),
            "BMI": context.get("BMI"), "ahi": context.get("ahi"),
            "ahi_severity": context.get("ahi_severity"),
            "rem_relevance": context.get("rem_relevance"),
            "position_relevance": context.get("position_relevance"),
            "oxygen_burden": context.get("oxygen_burden"),
        })
    pool = pd.DataFrame(pool_rows).drop_duplicates("patient_id", keep="last")
    pool.to_csv(staging / "unlabeled_patient_pool.csv", index=False, encoding="utf-8-sig")

    metadata = dict(base_metadata)
    metadata.update({
        "model_version": version,
        "model_family": "combined_four_treatment_ranking",
        "created_at": _now(),
        "base_four_treatment_component_version": base_version,
        "drug_trial_component_version": trial.get("model_version"),
        "drug_trial_challenger": trial,
        "labelled_drug_trial_rows": trial.get("complete_pair_count", 0),
        "pap_adherence_component_version": (
            pap_metadata.get("model_version") if pap_metadata else None
        ),
        "pap_adherence_challenger": pap_metadata,
        "labelled_pap_adherence_rows": (
            pap_metadata.get("confirmed_followup_adherence_rows", 0)
            if pap_metadata else 0
        ),
        "unlabeled_patient_pool_count": int(len(pool)),
        "combined_training_row_count": int(base_metadata.get("training_row_count", 0))
        + int(trial.get("complete_pair_count", 0))
        + int(
            pap_metadata.get("confirmed_followup_adherence_rows", 0)
            if pap_metadata else 0
        ),
        "training_data_explanation": (
            "ENT治療後AHI、隨機藥物試驗與PAP治療後依從性為不同標記目標；"
            "PAP依從性不會被偽裝成AHI療效。新患者PSG在取得相應治療後結果前"
            "進入未標記池。"
        ),
        "promotion_allowed": False,
        "deployment_status": "COMBINED_RESEARCH_CHALLENGER_REQUIRES_VALIDATION",
    })
    (staging / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    staging.replace(output)
    registry["latest_challenger"] = version
    registry["updated_at"] = _now()
    registry.setdefault("models", []).append(metadata)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
