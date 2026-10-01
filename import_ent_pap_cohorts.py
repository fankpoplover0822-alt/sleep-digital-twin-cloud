from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xlrd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_PATTERN = re.compile(r"(\d{8}T\d{6} - [0-9a-fA-F]+)")
MISSING_TEXT = {"", ".", "nan", "none", "null"}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_text(value: Any) -> str:
    text = str(value).strip()
    return "" if text.lower() in MISSING_TEXT else text


def _number(value: Any) -> float | None:
    text = _safe_text(value)
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _study_datetime(study_id: str) -> datetime:
    return datetime.strptime(study_id[:15], "%Y%m%dT%H%M%S")


def _archive_member_hash(nested: zipfile.ZipFile) -> str:
    digest = hashlib.sha256()
    for name in sorted(nested.namelist()):
        if name.endswith("/"):
            continue
        digest.update(name.rsplit("/", 1)[-1].encode("utf-8", errors="replace"))
        digest.update(nested.read(name))
    return digest.hexdigest()


def _read_ent_study(outer: zipfile.ZipFile, member: zipfile.ZipInfo) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(outer.read(member))) as nested:
        stage_counts: Counter[str] = Counter()
        respiratory_counts: Counter[str] = Counter()
        arousal_count = 0
        edf_bytes = None
        edf_compressed_bytes = None
        for info in nested.infolist():
            name = info.filename.lower()
            if name.endswith("signal grid.xls"):
                workbook = xlrd.open_workbook(file_contents=nested.read(info.filename))
                sheet = workbook.sheet_by_index(0)
                for row in range(2, sheet.nrows):
                    stage = _safe_text(sheet.cell_value(row, 2)).upper()
                    if stage:
                        stage_counts[stage] += 1
            elif name.endswith("event grid.xls"):
                workbook = xlrd.open_workbook(file_contents=nested.read(info.filename))
                for sheet in workbook.sheets():
                    count = max(0, sheet.nrows - 2)
                    if "AROUSAL" in sheet.name.upper():
                        arousal_count += count
                    else:
                        respiratory_counts[sheet.name] += count
            elif name.endswith(".edf"):
                edf_bytes = int(info.file_size)
                edf_compressed_bytes = int(info.compress_size)

        wake_tokens = {"WAKE", "W", "MOVEMENT", "MOVEMENT TIME", "MT"}
        sleep_epochs = sum(
            count for stage, count in stage_counts.items() if stage not in wake_tokens
        )
        sleep_hours = sleep_epochs * 30.0 / 3600.0
        respiratory_event_count = int(sum(respiratory_counts.values()))
        ahi = respiratory_event_count / sleep_hours if sleep_hours > 0 else None
        return {
            "sleep_epoch_count": int(sleep_epochs),
            "sleep_hours": sleep_hours,
            "respiratory_event_count": respiratory_event_count,
            "hypopnea_count": int(respiratory_counts.get("Hypopnea", 0)),
            "obstructive_apnea_count": int(respiratory_counts.get("A. Obstructive", 0)),
            "arousal_count": int(arousal_count),
            "ahi": ahi,
            "stage_counts": dict(stage_counts),
            "event_counts": dict(respiratory_counts),
            "edf_bytes": edf_bytes,
            "edf_compressed_bytes": edf_compressed_bytes,
            "archive_content_sha256": _archive_member_hash(nested),
        }


def import_ent_pairs(ent_csv: Path, ent_zip: Path, output_root: Path) -> dict[str, Any]:
    with ent_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        roster = list(csv.DictReader(handle))
    with zipfile.ZipFile(ent_zip) as outer:
        members_by_id: dict[str, list[zipfile.ZipInfo]] = {}
        for member in outer.infolist():
            match = ID_PATTERN.search(member.filename)
            if match and member.filename.lower().endswith(".zip"):
                members_by_id.setdefault(match.group(1), []).append(member)

        normalized: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []
        duplicates: dict[str, list[str]] = {}
        mapped_ids: set[str] = set()
        for index, row in enumerate(roster, start=1):
            baseline_id = _safe_text(row.get("PSG1"))
            follow_up_id = _safe_text(row.get("PSG2"))
            mapped_ids.update((baseline_id, follow_up_id))
            if baseline_id not in members_by_id or follow_up_id not in members_by_id:
                raise ValueError(f"ENT配對缺少壓縮檔：{baseline_id} -> {follow_up_id}")
            for study_id in (baseline_id, follow_up_id):
                if len(members_by_id[study_id]) > 1:
                    duplicates[study_id] = [item.filename for item in members_by_id[study_id]]
            baseline = _read_ent_study(outer, members_by_id[baseline_id][0])
            follow_up = _read_ent_study(outer, members_by_id[follow_up_id][0])
            follow_up_days = (_study_datetime(follow_up_id) - _study_datetime(baseline_id)).days
            baseline_ahi = baseline["ahi"]
            follow_up_ahi = follow_up["ahi"]
            if baseline_ahi is None or follow_up_ahi is None:
                outcome_status = "needs_review"
            else:
                outcome_status = "confirmed"
            pair_id = f"ENT_{index:03d}"
            reduction = (
                (baseline_ahi - follow_up_ahi) / baseline_ahi
                if baseline_ahi is not None and baseline_ahi > 0 and follow_up_ahi is not None
                else None
            )
            qc_flags: list[str] = []
            if follow_up_days > 730:
                qc_flags.append("FOLLOW_UP_OVER_730_DAYS")
            if baseline_ahi is not None and baseline_ahi < 5:
                qc_flags.append("DERIVED_BASELINE_AHI_BELOW_5")
            if baseline_ahi is not None and baseline_ahi > 120:
                qc_flags.append("DERIVED_BASELINE_AHI_OVER_120")
            if baseline.get("edf_compressed_bytes") and baseline.get("edf_bytes"):
                if baseline["edf_compressed_bytes"] / baseline["edf_bytes"] < 0.01:
                    qc_flags.append("BASELINE_EDF_EXTREME_COMPRESSION_REVIEW")
            if follow_up.get("edf_compressed_bytes") and follow_up.get("edf_bytes"):
                if follow_up["edf_compressed_bytes"] / follow_up["edf_bytes"] < 0.01:
                    qc_flags.append("FOLLOWUP_EDF_EXTREME_COMPRESSION_REVIEW")

            common = {
                "patient_id": pair_id,
                "baseline_study_id": baseline_id,
                "follow_up_study_id": follow_up_id,
                "sex": int(float(row["sex"])),
                "sex_label": "MALE" if int(float(row["sex"])) == 1 else "FEMALE",
                "age": float(row["age"]),
                "BMI": float(row["BMI"]),
                "follow_up_days": follow_up_days,
                "baseline_ahi": baseline_ahi,
                "follow_up_ahi": follow_up_ahi,
                "ahi_reduction_ratio": reduction,
                "qc_flags": "|".join(qc_flags),
                "source_confirmation": "PSG1_PRE_INTERVENTION_PSG2_POST_INTERVENTION_CONFIRMED_20260929",
            }
            normalized.append({
                **common,
                "baseline_sleep_hours": baseline["sleep_hours"],
                "baseline_event_count": baseline["respiratory_event_count"],
                "baseline_hypopnea_count": baseline["hypopnea_count"],
                "baseline_obstructive_apnea_count": baseline["obstructive_apnea_count"],
                "baseline_arousal_count": baseline["arousal_count"],
                "follow_up_sleep_hours": follow_up["sleep_hours"],
                "follow_up_event_count": follow_up["respiratory_event_count"],
                "follow_up_hypopnea_count": follow_up["hypopnea_count"],
                "follow_up_obstructive_apnea_count": follow_up["obstructive_apnea_count"],
                "follow_up_arousal_count": follow_up["arousal_count"],
                "baseline_archive_sha256": baseline["archive_content_sha256"],
                "follow_up_archive_sha256": follow_up["archive_content_sha256"],
            })
            outcomes.append({
                "patient_id": pair_id,
                "study_id": f"{baseline_id}__TO__{follow_up_id}",
                "treatment": "SURGERY",
                "baseline_ahi": baseline_ahi,
                "follow_up_ahi": follow_up_ahi,
                "follow_up_days": follow_up_days,
                "outcome_status": outcome_status,
                "reviewer_id": "DATASET_OWNER_CONFIRMED_20260929",
                "sex": int(float(row["sex"])),
                "age": float(row["age"]),
                "BMI": float(row["BMI"]),
                "sleep_hours": baseline["sleep_hours"],
                "hypopnea_count": baseline["hypopnea_count"],
                "obstructive_apnea_count": baseline["obstructive_apnea_count"],
                "arousal_count": baseline["arousal_count"],
                "intervention_family": "ENT_INTERVENTION_UNSPECIFIED_PROCEDURE",
                "label_provenance": "PAIRED_PSG_EVENT_AND_STAGE_EXPORT",
                "qc_flags": "|".join(qc_flags),
            })

        extras = sorted(set(members_by_id) - mapped_ids)

    cohort_dir = output_root / "data" / "continual_learning" / "external_clinical_cohorts" / "ent"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = cohort_dir / "ent_paired_psg_normalized.csv"
    pd.DataFrame(normalized).to_csv(normalized_path, index=False, encoding="utf-8-sig")
    outcome_path = (
        output_root / "data" / "continual_learning" / "treatment_outcomes"
        / "outcomes_ent_paired_psg_20260929.csv"
    )
    outcome_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outcomes).to_csv(outcome_path, index=False, encoding="utf-8-sig")
    return {
        "pair_count": len(normalized),
        "confirmed_outcome_rows_written": sum(row["outcome_status"] == "confirmed" for row in outcomes),
        "training_eligible_by_current_follow_up_window": sum(
            row["outcome_status"] == "confirmed" and 14 <= int(row["follow_up_days"]) <= 1460
            for row in outcomes
        ),
        "normalized_file": str(normalized_path),
        "outcome_file": str(outcome_path),
        "extra_unpaired_archive_ids": extras,
        "duplicate_archive_ids_ignored": duplicates,
    }


def import_pap_roster(roster_path: Path, output_root: Path) -> dict[str, Any]:
    workbook = xlrd.open_workbook(str(roster_path))
    sheet = workbook.sheet_by_name("roster")
    headers = [_safe_text(sheet.cell_value(0, column)) for column in range(sheet.ncols)]
    rows = [[sheet.cell_value(row, column) for column in range(sheet.ncols)] for row in range(1, sheet.nrows)]
    cpap_values = Counter(_safe_text(row[10]) for row in rows if _safe_text(row[10]))
    # The source codebook and the supplied confirmation establish 是=yes, 否=no.
    # The legacy XLS stores both strings as mojibake; their stable values are retained
    # in raw_cpapuse and mapped by the observed code strings below.
    ordered = [value for value, _ in cpap_values.most_common()]
    no_code = ordered[0] if ordered else ""
    yes_code = ordered[1] if len(ordered) > 1 else ""
    normalized: list[dict[str, Any]] = []
    for row in rows:
        baseline_id = _safe_text(row[0])
        pap_study_id = _safe_text(row[2])
        baseline_date = _study_datetime(baseline_id)
        birth_text = str(int(float(row[4]))) if _number(row[4]) is not None else ""
        birth_date = None
        age = None
        if len(birth_text) == 8:
            birth_date = datetime.strptime(birth_text, "%Y%m%d")
            age = (baseline_date.date() - birth_date.date()).days / 365.2425
        cpap_raw = _safe_text(row[10])
        cpap_use = 1 if cpap_raw == yes_code else (0 if cpap_raw == no_code else None)
        sefam_p90 = _number(row[13])
        sefam_average = _number(row[14])
        resmed_fixed = _number(row[15])
        if sefam_p90 is not None or sefam_average is not None:
            therapy_mode = "APAP_LIKE_VARIABLE_PRESSURE_EXPORT"
        elif resmed_fixed is not None:
            therapy_mode = "CPAP_FIXED_PRESSURE_EXPORT"
        else:
            therapy_mode = "UNKNOWN"
        days7 = _number(row[11])
        hours7 = _number(row[12])
        adherence_7d = (
            int(days7 >= 5 and hours7 >= 4)
            if days7 is not None and hours7 is not None else None
        )
        normalized.append({
            "patient_id": baseline_id,
            "pap_study_id": pap_study_id,
            "BMI": _number(row[1]),
            "sex": int(_number(row[3])) if _number(row[3]) is not None else None,
            "sex_label": "MALE" if _number(row[3]) == 1 else ("FEMALE" if _number(row[3]) == 2 else None),
            "birth_date": birth_date.date().isoformat() if birth_date else None,
            "age_at_baseline": age,
            "cpap_use": cpap_use,
            "raw_cpapuse": cpap_raw,
            "other_treatment_note_raw": _safe_text(row[9]),
            "recent_days_used": _number(row[5]),
            "recent_hours_per_night": _number(row[6]),
            "previous_days_used": _number(row[7]),
            "previous_hours_per_night": _number(row[8]),
            "chip_days_7": days7,
            "chip_hours_7": hours7,
            "adherent_7d": adherence_7d,
            "sefAM_pressure_p90_cmh2o": sefam_p90,
            "sefAM_pressure_average_cmh2o": sefam_average,
            "resmed_fixed_pressure_cmh2o": resmed_fixed,
            "sefAM_leak_p90_l_min": _number(row[16]),
            "sefAM_leak_average_l_min": _number(row[17]),
            "resmed_leak_p95_l_min": _number(row[18]),
            "resmed_leak_average_l_min": _number(row[19]),
            "chip_days_30": _number(row[20]),
            "chip_hours_30": _number(row[21]),
            "device_brand": _safe_text(row[29]),
            "therapy_mode_inference": therapy_mode,
            "source_note": "PAP roster; therapy mode inferred only from pressure export fields",
        })
    frame = pd.DataFrame(normalized)
    cohort_dir = output_root / "data" / "continual_learning" / "external_clinical_cohorts" / "pap"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = cohort_dir / "pap_roster_normalized.csv"
    frame.to_csv(normalized_path, index=False, encoding="utf-8-sig")

    train = frame.dropna(subset=["adherent_7d", "BMI", "sex", "age_at_baseline"]).copy()
    model_result: dict[str, Any] = {"status": "insufficient_data"}
    if len(train) >= 30 and train["adherent_7d"].nunique() == 2:
        features = ["BMI", "sex", "age_at_baseline", "therapy_mode_inference"]
        numeric = ["BMI", "sex", "age_at_baseline"]
        categorical = ["therapy_mode_inference"]
        model = Pipeline([
            ("preprocessor", ColumnTransformer([
                ("numeric", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), numeric),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]), categorical),
            ])),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ])
        folds = min(5, int(train["adherent_7d"].value_counts().min()))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        probabilities = cross_val_predict(
            model, train[features], train["adherent_7d"].astype(int),
            cv=splitter, method="predict_proba",
        )[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        model.fit(train[features], train["adherent_7d"].astype(int))
        version = "pap_adherence_" + datetime.now().strftime("%Y%m%dT%H%M%S%f")
        model_dir = output_root / "models" / "registry" / "pap_adherence" / version
        model_dir.mkdir(parents=True, exist_ok=False)
        joblib.dump(model, model_dir / "model.joblib")
        metadata = {
            "model_version": version,
            "created_at": _now(),
            "model_family": "PAP_7_DAY_ADHERENCE_RESEARCH_CHALLENGER",
            "training_row_count": int(len(train)),
            "positive_count": int(train["adherent_7d"].sum()),
            "negative_count": int((1 - train["adherent_7d"]).sum()),
            "features": features,
            "target": "chip_days_7>=5 and chip_hours_7>=4",
            "cv_folds": folds,
            "cv_roc_auc": float(roc_auc_score(train["adherent_7d"], probabilities)),
            "cv_balanced_accuracy": float(balanced_accuracy_score(train["adherent_7d"], predictions)),
            "sklearn_version": sklearn.__version__,
            "promotion_allowed": False,
            "clinical_effectiveness_claim": False,
            "affects_formal_treatment_score": False,
            "reason": "This cohort supports adherence research, not AHI treatment-effect supervision.",
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        latest = model_dir.parent / "latest.json"
        latest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        model_result = metadata

    return {
        "row_count": int(len(frame)),
        "cpap_use_label_count": int(frame["cpap_use"].notna().sum()),
        "pap_metric_row_count": int(frame["chip_hours_7"].notna().sum()),
        "apap_like_export_rows": int(frame["therapy_mode_inference"].eq("APAP_LIKE_VARIABLE_PRESSURE_EXPORT").sum()),
        "cpap_fixed_export_rows": int(frame["therapy_mode_inference"].eq("CPAP_FIXED_PRESSURE_EXPORT").sum()),
        "normalized_file": str(normalized_path),
        "adherence_model": model_result,
        "raw_headers": headers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import paired ENT PSG and PAP adherence cohorts")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--ent-csv", required=True)
    parser.add_argument("--ent-zip", required=True)
    parser.add_argument("--pap-roster", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    result = {
        "created_at": _now(),
        "ent": import_ent_pairs(Path(args.ent_csv), Path(args.ent_zip), root),
        "pap": import_pap_roster(Path(args.pap_roster), root),
    }
    manifest = root / "data" / "continual_learning" / "external_clinical_cohorts" / "import_manifest_20260929.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["manifest"] = str(manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
