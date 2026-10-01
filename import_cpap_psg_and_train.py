from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xlrd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_PATTERN = re.compile(r"(\d{8}T\d{6} - [0-9a-fA-F]+)")
WAKE_TOKENS = {"WAKE", "W", "MOVEMENT", "MOVEMENT TIME", "MT"}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe(value: Any) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", ".", "nan", "none", "null"} else text


def _read_study(payload: bytes) -> dict[str, Any]:
    stage_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    arousal_count = 0
    small_hash = hashlib.sha256()
    edf_present = False
    edf_bytes = None
    with zipfile.ZipFile(io.BytesIO(payload)) as nested:
        for info in nested.infolist():
            name = info.filename.lower()
            if name.endswith("signal grid.xls"):
                content = nested.read(info)
                small_hash.update(content)
                workbook = xlrd.open_workbook(file_contents=content)
                sheet = workbook.sheet_by_index(0)
                for row in range(2, sheet.nrows):
                    stage = _safe(sheet.cell_value(row, 2)).upper()
                    if stage:
                        stage_counts[stage] += 1
            elif name.endswith("event grid.xls"):
                content = nested.read(info)
                small_hash.update(content)
                workbook = xlrd.open_workbook(file_contents=content)
                for sheet in workbook.sheets():
                    count = max(0, sheet.nrows - 2)
                    if "AROUSAL" in sheet.name.upper():
                        arousal_count += count
                    else:
                        event_counts[sheet.name] += count
            elif name.endswith(".edf"):
                edf_present = True
                edf_bytes = int(info.file_size)
    sleep_epochs = sum(v for k, v in stage_counts.items() if k not in WAKE_TOKENS)
    total_epochs = sum(stage_counts.values())
    sleep_hours = sleep_epochs * 30.0 / 3600.0
    respiratory_events = int(sum(event_counts.values()))
    def fraction(token: str) -> float | None:
        return stage_counts.get(token, 0) / sleep_epochs if sleep_epochs else None
    return {
        "sleep_hours": sleep_hours,
        "ahi": respiratory_events / sleep_hours if sleep_hours else None,
        "respiratory_event_count": respiratory_events,
        "hypopnea_count": int(event_counts.get("Hypopnea", 0)),
        "obstructive_apnea_count": int(event_counts.get("A. Obstructive", 0)),
        "central_apnea_count": int(event_counts.get("A. Central", 0)),
        "mixed_apnea_count": int(event_counts.get("A. Mixed", 0)),
        "arousal_count": int(arousal_count),
        "sleep_efficiency_proxy": sleep_epochs / total_epochs if total_epochs else None,
        "n1_fraction": fraction("N1"),
        "n2_fraction": fraction("N2"),
        "n3_fraction": fraction("N3"),
        "rem_fraction": fraction("REM"),
        "edf_present": edf_present,
        "edf_bytes": edf_bytes,
        "event_signal_sha256": small_hash.hexdigest(),
    }


def import_psg(zip_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    members: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as outer:
        for info in outer.infolist():
            match = ID_PATTERN.search(info.filename.rsplit("/", 1)[-1])
            if match and info.filename.lower().endswith(".zip"):
                members[match.group(1)].append(info)
        rows: list[dict[str, Any]] = []
        duplicate_checks: dict[str, list[dict[str, Any]]] = {}
        ids = sorted(members)
        for index, study_id in enumerate(ids, start=1):
            versions: list[dict[str, Any]] = []
            selected: dict[str, Any] | None = None
            for info in members[study_id]:
                payload = outer.read(info)
                summary = _read_study(payload)
                versions.append({
                    "member": info.filename,
                    "event_signal_sha256": summary["event_signal_sha256"],
                    "outer_size": info.file_size,
                })
                if selected is None:
                    selected = summary
                del payload
                gc.collect()
            if len(versions) > 1:
                duplicate_checks[study_id] = versions
            rows.append({"patient_id": study_id, **(selected or {})})
            if index % 10 == 0 or index == len(ids):
                print(f"Parsed {index}/{len(ids)} baseline PSG packages", flush=True)
    frame = pd.DataFrame(rows)
    qc = {
        "outer_archive_count": int(sum(len(v) for v in members.values())),
        "unique_study_count": int(len(members)),
        "duplicate_study_count": int(sum(len(v) > 1 for v in members.values())),
        "duplicate_studies": duplicate_checks,
        "missing_edf_count": int((~frame["edf_present"].fillna(False)).sum()),
        "usable_ahi_count": int(frame["ahi"].notna().sum()),
    }
    return frame, qc


def _model(features_numeric: list[str], features_categorical: list[str], classifier: Any) -> Pipeline:
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), features_numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), features_categorical),
        ])),
        ("classifier", classifier),
    ])


def train_adherence(frame: pd.DataFrame, root: Path) -> dict[str, Any]:
    frame = frame.copy()
    safe_sleep_hours = frame["sleep_hours"].replace(0, np.nan)
    safe_event_count = frame["respiratory_event_count"].replace(0, np.nan)
    frame["arousal_index"] = frame["arousal_count"] / safe_sleep_hours
    frame["central_apnea_index"] = frame["central_apnea_count"] / safe_sleep_hours
    frame["obstructive_apnea_index"] = frame["obstructive_apnea_count"] / safe_sleep_hours
    frame["hypopnea_index"] = frame["hypopnea_count"] / safe_sleep_hours
    frame["obstructive_event_fraction"] = (
        frame["obstructive_apnea_count"] / safe_event_count
    )
    frame["hypopnea_fraction"] = frame["hypopnea_count"] / safe_event_count
    frame["jsr2024_pap_cohort_eligible"] = (
        (frame["ahi"] >= 15) & (frame["central_apnea_index"] < 5)
    ).astype(float)

    base_numeric = [
        "BMI", "sex", "age_at_baseline", "ahi", "sleep_hours",
        "hypopnea_count", "obstructive_apnea_count", "arousal_count",
        "sleep_efficiency_proxy", "n1_fraction", "n2_fraction", "n3_fraction", "rem_fraction",
    ]
    mechanism_numeric = [
        "arousal_index", "central_apnea_index", "obstructive_apnea_index",
        "hypopnea_index", "obstructive_event_fraction", "hypopnea_fraction",
        "jsr2024_pap_cohort_eligible",
    ]
    enhanced_numeric = base_numeric + mechanism_numeric
    categorical = ["therapy_mode_inference", "device_brand"]
    trainable = frame.dropna(subset=["adherent_7d", "patient_id"]).copy()
    trainable["adherent_7d"] = trainable["adherent_7d"].astype(int)
    split_dir = root / "data" / "continual_learning" / "model_metrics"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_file = split_dir / "pap_adherence_locked_test_patients.json"
    if split_file.exists():
        locked_ids = set(json.loads(split_file.read_text(encoding="utf-8"))["patient_ids"])
    else:
        _, holdout = train_test_split(
            trainable,
            test_size=0.20,
            random_state=20260930,
            stratify=trainable[["adherent_7d", "therapy_mode_inference"]].astype(str).agg("|".join, axis=1),
        )
        locked_ids = set(holdout["patient_id"].astype(str))
        split_file.write_text(json.dumps({
            "created_at": _now(),
            "policy": "immutable_patient_level_stratified_20_percent_holdout",
            "patient_ids": sorted(locked_ids),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    development = trainable.loc[~trainable["patient_id"].astype(str).isin(locked_ids)].copy()
    holdout = trainable.loc[trainable["patient_id"].astype(str).isin(locked_ids)].copy()
    if len(holdout) < 10 or holdout["adherent_7d"].nunique() != 2:
        raise RuntimeError("鎖定測試集不足或缺少其中一類依從性結果")

    classifiers = {
        "logistic_balanced": LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42),
        "random_forest_leaf_4": RandomForestClassifier(n_estimators=500, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1),
        "extra_trees_leaf_4": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    feature_sets = {
        "baseline": base_numeric,
        "jsr2024_mechanism_informed": enhanced_numeric,
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260930)
    candidate_results: list[dict[str, Any]] = []
    best_name = ""
    best_feature_set = ""
    best_auc = -np.inf
    for feature_set_name, numeric in feature_sets.items():
        features = numeric + categorical
        for name, classifier in classifiers.items():
            pipeline = _model(numeric, categorical, clone(classifier))
            probabilities = cross_val_predict(
                pipeline, development[features], development["adherent_7d"],
                cv=folds, method="predict_proba",
            )[:, 1]
            auc = float(roc_auc_score(development["adherent_7d"], probabilities))
            result = {
                "feature_set": feature_set_name,
                "model": name,
                "development_cv_roc_auc": auc,
                "development_cv_log_loss": float(log_loss(development["adherent_7d"], probabilities)),
                "development_cv_balanced_accuracy": float(balanced_accuracy_score(
                    development["adherent_7d"], (probabilities >= 0.5).astype(int)
                )),
            }
            candidate_results.append(result)
            if auc > best_auc:
                best_auc, best_name, best_feature_set = auc, name, feature_set_name

    selected_numeric = feature_sets[best_feature_set]
    features = selected_numeric + categorical
    model = _model(selected_numeric, categorical, clone(classifiers[best_name]))
    model.fit(development[features], development["adherent_7d"])
    holdout_probability = model.predict_proba(holdout[features])[:, 1]
    holdout_prediction = (holdout_probability >= 0.5).astype(int)
    holdout_auc = float(roc_auc_score(holdout["adherent_7d"], holdout_probability))
    holdout_balanced_accuracy = float(balanced_accuracy_score(holdout["adherent_7d"], holdout_prediction))
    holdout_log_loss = float(log_loss(holdout["adherent_7d"], holdout_probability))

    version = "pap_psg_adherence_" + datetime.now().strftime("%Y%m%dT%H%M%S%f")
    model_dir = root / "models" / "registry" / "pap_adherence" / version
    model_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, model_dir / "model.joblib")
    holdout_export = holdout[["patient_id", "adherent_7d", "therapy_mode_inference"]].copy()
    holdout_export["predicted_adherence_probability"] = holdout_probability
    holdout_export.to_csv(model_dir / "locked_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "model_version": version,
        "created_at": _now(),
        "model_family": "PAP_ADHERENCE_FROM_BASELINE_PSG_AND_CLINICAL_FEATURES",
        "clinical_target": "7_day_device_adherence_not_AHI_treatment_effect",
        "development_patient_count": int(len(development)),
        "locked_holdout_patient_count": int(len(holdout)),
        "confirmed_followup_adherence_rows": int(len(trainable)),
        "selected_algorithm": best_name,
        "selected_feature_set": best_feature_set,
        "features": features,
        "candidate_results": sorted(candidate_results, key=lambda row: row["development_cv_roc_auc"], reverse=True),
        "locked_holdout_roc_auc": holdout_auc,
        "locked_holdout_balanced_accuracy": holdout_balanced_accuracy,
        "locked_holdout_log_loss": holdout_log_loss,
        "baseline_roc_auc": 0.5,
        "outperforms_random_auc": bool(holdout_auc > 0.5),
        "sklearn_version": sklearn.__version__,
        "promotion_allowed": False,
        "affects_formal_treatment_score": False,
        "reason": "Treatment-after data are PAP usage, pressure and leak metrics; no follow-up PSG/AHI is present.",
        "jsr2024_evidence": {
            "citation": "Cheng et al., Journal of Sleep Research 2024;33:e13999",
            "doi": "10.1111/jsr.13999",
            "role": "mechanism-informed feature engineering and cohort eligibility only",
            "exact_pup_endotypes_available": False,
            "safety_note": (
                "Derived indices are observable PSG summaries, not PUP/PUPpy estimates of "
                "arousal threshold, collapsibility, loop gain or upper-airway gain."
            ),
            "feature_set_selected_only_by_development_cv": True,
            "locked_holdout_not_used_for_selection": True,
        },
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = model_dir.parent / "latest.json"
    latest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--cpap-zip", required=True, type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    roster_path = root / "data" / "continual_learning" / "external_clinical_cohorts" / "pap" / "pap_roster_normalized.csv"
    roster = pd.read_csv(roster_path)
    psg, qc = import_psg(args.cpap_zip)
    frame = roster.merge(psg, on="patient_id", how="left", validate="one_to_one")
    cohort_file = roster_path.with_name("cpap_baseline_psg_with_pap_followup.csv")
    frame.to_csv(cohort_file, index=False, encoding="utf-8-sig")
    model = train_adherence(frame, root)
    manifest = {
        "created_at": _now(),
        "source_zip": str(args.cpap_zip),
        "roster_file": str(roster_path),
        "cohort_file": str(cohort_file),
        "row_count": int(len(frame)),
        "baseline_psg_match_count": int(frame["ahi"].notna().sum()),
        "followup_adherence_label_count": int(frame["adherent_7d"].notna().sum()),
        "followup_ahi_count": 0,
        "data_interpretation": "baseline PSG plus post-PAP device adherence/pressure/leak; not paired pre/post PSG efficacy",
        "archive_qc": qc,
        "model": model,
    }
    manifest_file = cohort_file.with_name("cpap_import_manifest_20260930.json")
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
