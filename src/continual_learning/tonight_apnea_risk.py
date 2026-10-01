from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.continual_learning.governance import (
    ALLOWED_TONIGHT_FEATURES,
    current_monitoring_window,
    validate_feature_names,
    wearable_quality,
    write_audit_event,
)


FEATURES = [
    "age", "BMI", "prior_ahi", "prior_spo2_mean", "prior_spo2_min",
    "day_spo2_mean", "day_spo2_min", "day_spo2_std", "day_t90",
    "day_hr_mean", "day_hr_std", "day_rr_mean", "coverage_hours",
    "day_accel_x_std", "day_accel_y_std", "day_accel_z_std",
    "day_accel_magnitude_std", "day_movement_mean", "day_active_ratio",
]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_training_frame(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for profile_path in (root / "data" / "inference").glob(
        "*/respiratory_profile/patient_respiratory_profile.json"
    ):
        profile = _read_json(profile_path)
        event = profile.get("event_summary") or {}
        oxygen = profile.get("spo2_profile") or {}
        ahi = pd.to_numeric(event.get("ahi"), errors="coerce")
        if pd.isna(ahi):
            continue
        rows.append({
            "patient_id": profile_path.parents[1].name,
            "age": profile.get("age"), "BMI": profile.get("BMI"),
            "prior_ahi": ahi,
            "prior_spo2_mean": oxygen.get("spo2_mean"),
            "prior_spo2_min": oxygen.get("spo2_min"),
            # Paired daytime-watch data are not yet available historically.
            # These stay missing and are imputed; current watch state is added
            # through a bounded deviation term at prediction time.
            # Neutral placeholders keep the input schema stable. They are not
            # presented as observed daytime data and therefore carry no
            # patient-specific training signal in this bootstrap version.
            "day_spo2_mean": 96.0, "day_spo2_min": 94.0,
            "day_spo2_std": 1.0, "day_t90": 0.0,
            "day_hr_mean": 70.0, "day_hr_std": 8.0,
            "day_rr_mean": 15.0, "coverage_hours": 0.0,
            "day_accel_x_std": 0.0, "day_accel_y_std": 0.0,
            "day_accel_z_std": 0.0, "day_accel_magnitude_std": 0.0,
            "day_movement_mean": 0.0, "day_active_ratio": 0.0,
            "target": int(float(ahi) >= 15.0),
        })
    frame = pd.DataFrame(rows)
    paired_path = root / "data" / "continual_learning" / "tonight_paired_outcomes.csv"
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        # Keep verified rows collected before a newly governed sensor feature
        # was introduced. Missing XYZ fields are imputed by the pipeline rather
        # than silently discarding valid historical outcomes.
        if {"patient_id", "target"}.issubset(paired.columns):
            for feature in FEATURES:
                if feature not in paired.columns:
                    paired[feature] = np.nan
            required = ["patient_id", *FEATURES, "target"]
            frame = pd.concat([frame, paired[required]], ignore_index=True, sort=False)
    return frame


def train(root: Path, model_path: Path) -> dict:
    validate_feature_names(FEATURES, allowed=set(ALLOWED_TONIGHT_FEATURES))
    data = build_training_frame(root)
    if len(data) < 8 or data["target"].nunique() < 2:
        raise RuntimeError("目前 PSG 訓練資料不足或只有單一風險類別。")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42,
        )),
    ])
    model.fit(data[FEATURES], data["target"])
    version = datetime.now(timezone.utc).strftime("tonight_risk_%Y%m%dT%H%M%S%fZ")
    bundle = {
        "model": model, "features": FEATURES, "version": version,
        "trained_patients": int(len(data)), "target": "night_ahi_ge_15",
        "bootstrap": True,
        "limitations": "XYZ 活動特徵已納入輸入；但成對的白天手錶→當晚 PSG 標籤仍少，穿戴狀態另採有界研究修正，須經外部驗證。",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def add_verified_night_outcome_and_retrain(
    root: Path,
    patient_id: str,
    samples: pd.DataFrame,
    outcome: dict[str, Any],
    model_path: Path,
) -> dict[str, Any]:
    """Pair daytime wearable features with a verified/synthetic night result."""
    ahi = float(outcome["night_ahi"])
    if ahi < 0 or ahi > 200:
        raise ValueError("night_ahi must be between 0 and 200")
    source_type = str(outcome.get("source_type", "")).strip().lower()
    if source_type not in {"psg", "hsat", "synthetic_psg", "synthetic_hsat"}:
        raise ValueError("source_type must be PSG, HSAT, synthetic_PSG or synthetic_HSAT")
    reviewer_id = str(outcome.get("reviewer_id", "")).strip()
    if not reviewer_id:
        raise ValueError("reviewer_id is required")
    quality_gate = wearable_quality(samples)
    if not quality_gate["usable"]:
        raise ValueError(
            "穿戴資料未通過品質門檻，不能建立監督式配對訓練列："
            + "；".join(quality_gate.get("warnings") or [quality_gate.get("reason", "資料不足")])
        )
    features = {**_history(root, patient_id), **_watch_features(samples)[0]}
    record = {
        "patient_id": patient_id,
        **{name: features.get(name) for name in FEATURES},
        "target": int(ahi >= 15.0),
        "night_ahi": ahi,
        "night_odi": outcome.get("night_odi"),
        "night_t90": outcome.get("night_t90"),
        "monitoring_date": outcome.get("monitoring_date"),
        "source_type": source_type,
        "reviewer_id": reviewer_id,
        "label_quality": "synthetic_paired_test" if source_type.startswith("synthetic_") else "verified_paired_outcome",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="microseconds"),
    }
    canonical = json.dumps(
        {key: record.get(key) for key in ["patient_id", "monitoring_date", "source_type", "night_ahi", *FEATURES]},
        ensure_ascii=False, sort_keys=True, default=str,
    )
    record["content_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    destination = root / "data" / "continual_learning" / "tonight_paired_outcomes.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    stored = pd.read_csv(destination) if destination.exists() else pd.DataFrame()
    duplicate = not stored.empty and record["content_hash"] in set(
        stored.get("content_hash", pd.Series(dtype=str)).astype(str)
    )
    if not duplicate:
        stored = pd.concat([stored, pd.DataFrame([record])], ignore_index=True, sort=False)
        stored.to_csv(destination, index=False, encoding="utf-8-sig")
    previous_version = None
    if model_path.exists():
        try:
            previous_version = joblib.load(model_path).get("version")
        except Exception:
            previous_version = None
    bundle = train(root, model_path)
    receipt = {
        "status": "retrained",
        "new_training_row": not duplicate,
        "model_version": bundle["version"],
        "previous_model_version": previous_version,
        "training_rows": int(len(build_training_frame(root))),
        "label_quality": record["label_quality"],
        "content_hash": record["content_hash"],
        "quality_gate": quality_gate,
        "promotion_status": "RESEARCH_CHALLENGER_ONLY" if source_type.startswith("synthetic_") else "REQUIRES_VALIDATION",
    }
    write_audit_event(root, patient_id, {
        "event_type": "TONIGHT_MODEL_RETRAINED",
        "receipt": receipt,
        "source_type": source_type,
        "monitoring_date": outcome.get("monitoring_date"),
        "reviewer_id": reviewer_id,
    })
    return receipt


def _history(root: Path, patient_id: str) -> dict:
    path = root / "data" / "inference" / patient_id / "respiratory_profile" / "patient_respiratory_profile.json"
    profile = _read_json(path)
    event, oxygen = profile.get("event_summary") or {}, profile.get("spo2_profile") or {}
    sleep = profile.get("sleep_summary") or {}
    return {
        "age": profile.get("age"), "BMI": profile.get("BMI"),
        "prior_ahi": event.get("ahi"), "prior_spo2_mean": oxygen.get("spo2_mean"),
        "prior_spo2_min": oxygen.get("spo2_min"),
        "prior_sleep_hours": sleep.get("sleep_hours"),
    }


def _watch_features(samples: pd.DataFrame) -> tuple[dict, dict]:
    if samples.empty:
        return {}, {"sample_count": 0, "coverage_hours": 0.0, "quality": "NO_DATA"}
    frame = samples.copy()
    source_values = frame.get("source", pd.Series(index=frame.index, dtype=str)).fillna("").astype(str)
    synthetic = source_values.str.contains("|scenario=", regex=False) | source_values.str.contains(
        "synthetic_watch_test", regex=False
    )
    active_scenario = None
    if synthetic.any():
        # When testing, use the latest uploaded scenario as one isolated batch.
        frame = frame.loc[synthetic].copy()
        scenario_source = str(frame.iloc[-1].get("source", ""))
        active_scenario = scenario_source.split("|scenario=", 1)[-1] if "|scenario=" in scenario_source else "synthetic_watch_test"
    frame = current_monitoring_window(frame)
    ts = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True).dt.tz_convert("Asia/Taipei")
    spo2 = pd.to_numeric(frame.get("spo2"), errors="coerce")
    hr = pd.to_numeric(frame.get("heart_rate"), errors="coerce")
    rr = pd.to_numeric(frame.get("respiratory_rate"), errors="coerce")
    accel_x = pd.to_numeric(frame.get("accel_x"), errors="coerce")
    accel_y = pd.to_numeric(frame.get("accel_y"), errors="coerce")
    accel_z = pd.to_numeric(frame.get("accel_z"), errors="coerce")
    accel_magnitude = pd.to_numeric(frame.get("accel_magnitude"), errors="coerce")
    movement = pd.to_numeric(frame.get("movement"), errors="coerce")
    hours = float((ts.max() - ts.min()).total_seconds() / 3600) if ts.notna().sum() > 1 else 0.0
    robust_low_spo2 = spo2.quantile(.05) if spo2.notna().sum() >= 20 else spo2.min()
    values = {
        "day_spo2_mean": spo2.mean(), "day_spo2_min": robust_low_spo2,
        "day_spo2_std": spo2.std(), "day_t90": (spo2 < 90).mean() * 100,
        "day_hr_mean": hr.mean(), "day_hr_std": hr.std(),
        "day_rr_mean": rr.mean(), "coverage_hours": hours,
        "day_accel_x_std": accel_x.std(), "day_accel_y_std": accel_y.std(),
        "day_accel_z_std": accel_z.std(),
        "day_accel_magnitude_std": accel_magnitude.std(),
        "day_movement_mean": movement.mean(),
        "day_active_ratio": (movement > 0.08).mean() if movement.notna().any() else np.nan,
    }
    valid_ratio = float((spo2.notna() & hr.notna()).mean()) if len(frame) else 0
    hr_valid_ratio = float(hr.notna().mean()) if len(frame) else 0
    if len(frame) >= 30 and hours >= 6 and valid_ratio >= .8:
        quality = "GOOD"
    elif len(frame) >= 30 and hours >= .05 and hr_valid_ratio >= .8 and spo2.notna().sum() == 0:
        # Bangle.js 2 has HR/activity sensing but no SpO2 sensor.  Permit an
        # explicitly limited estimate driven mainly by prior PSG/demographics;
        # never present it as equivalent to the full watch+SpO2 pathway.
        quality = "LIMITED_NO_SPO2"
    else:
        quality = "LIMITED"
    return values, {
        "sample_count": int(len(frame)), "coverage_hours": round(hours, 2),
        "valid_ratio": round(valid_ratio, 3), "quality": quality,
        "heart_rate_valid_ratio": round(hr_valid_ratio, 3),
        "active_scenario": active_scenario,
        "input_mode": "SYNTHETIC_TEST_ISOLATED" if active_scenario else "REAL_ACCUMULATED_STREAM",
        "monitoring_date": str(ts.max().date()) if ts.notna().any() else None,
        "spo2_low_statistic": "5th_percentile" if spo2.notna().sum() >= 20 else "minimum",
    }


def predict(root: Path, patient_id: str, samples: pd.DataFrame, model_path: Path) -> dict:
    if not model_path.exists():
        train(root, model_path)
    bundle = joblib.load(model_path)
    # A stored model from before XYZ support has a different feature schema.
    # Rebuild it before inference so the deployed model and audit version match
    # the current, governed input definition.
    if list(bundle.get("features") or []) != FEATURES:
        bundle = train(root, model_path)
    validate_feature_names(FEATURES, allowed=set(ALLOWED_TONIGHT_FEATURES))
    row = {**_history(root, patient_id)}
    watch, quality = _watch_features(samples)
    strict_frame = samples.copy()
    source_values = strict_frame.get("source", pd.Series(index=strict_frame.index, dtype=str)).fillna("").astype(str)
    synthetic = source_values.str.contains("|scenario=", regex=False) | source_values.str.contains(
        "synthetic_watch_test", regex=False
    )
    if synthetic.any():
        strict_frame = strict_frame.loc[synthetic].copy()
    strict_quality = wearable_quality(strict_frame)
    quality["strict_gate"] = strict_quality
    row.update(watch)
    frame = pd.DataFrame([row], columns=FEATURES)
    base = float(bundle["model"].predict_proba(frame)[0, 1])
    # Temporary bounded state modifier until paired daytime-watch/night-PSG
    # labels exist. It cannot overwhelm established PSG history.
    min_spo2 = pd.to_numeric(row.get("day_spo2_min"), errors="coerce")
    t90 = pd.to_numeric(row.get("day_t90"), errors="coerce")
    hr_std = pd.to_numeric(row.get("day_hr_std"), errors="coerce")
    movement_mean = pd.to_numeric(row.get("day_movement_mean"), errors="coerce")
    active_ratio = pd.to_numeric(row.get("day_active_ratio"), errors="coerce")
    accel_mag_std = pd.to_numeric(row.get("day_accel_magnitude_std"), errors="coerce")
    modifier = 0.0
    reasons = []
    if not pd.isna(min_spo2) and min_spo2 < 92:
        modifier += min((92 - float(min_spo2)) * .012, .12); reasons.append("白天穿戴式 SpO₂ 曾偏低")
    if not pd.isna(t90) and t90 > 1:
        modifier += min(float(t90) * .004, .08); reasons.append("白天 SpO₂<90% 的比例增加")
    if not pd.isna(hr_std) and hr_std > 15:
        modifier += min((float(hr_std) - 15) * .003, .04); reasons.append("白天心率變異較大")
    xyz_available = not pd.isna(accel_mag_std) and not pd.isna(movement_mean)
    if xyz_available:
        if float(movement_mean) > .08:
            modifier += min((float(movement_mean) - .08) * .35, .04)
            reasons.append("XYZ 顯示平均活動波動偏高")
        if not pd.isna(active_ratio) and float(active_ratio) > .35:
            modifier += min((float(active_ratio) - .35) * .06, .03)
            reasons.append("XYZ 顯示高活動波動時段比例偏高")
        if float(accel_mag_std) > .12:
            modifier += min((float(accel_mag_std) - .12) * .20, .03)
            reasons.append("XYZ 合成加速度變異偏高")
        if (
            float(movement_mean) < .03 and float(accel_mag_std) < .04
            and not pd.isna(hr_std) and float(hr_std) < 8
        ):
            modifier -= .05
            reasons.append("今日心率穩定且 XYZ 活動波動低，研究修正下調")
    probability = float(np.clip(base + modifier, .01, .99))
    level = "高" if probability >= .65 else ("中" if probability >= .35 else "低")
    if not reasons:
        reasons.append("本次主要依既往 PSG、年齡與 BMI 估計")
    prior_ahi = pd.to_numeric(row.get("prior_ahi"), errors="coerce")
    prior_sleep_hours = pd.to_numeric(row.get("prior_sleep_hours"), errors="coerce")
    estimated_ahi = None
    event_count_range = None
    if not pd.isna(prior_ahi):
        estimated_ahi = max(float(prior_ahi) * (1.0 + modifier * 1.5), 0.0)
        sleep_hours = float(prior_sleep_hours) if not pd.isna(prior_sleep_hours) else 7.0
        midpoint = estimated_ahi * max(sleep_hours, 1.0)
        event_count_range = [int(max(round(midpoint * .7), 0)), int(max(round(midpoint * 1.3), 0))]
    event_likelihood = "資料不足"
    if quality.get("quality") in {"GOOD", "LIMITED_NO_SPO2"}:
        event_likelihood = "可能" if (
            probability >= .35 or (estimated_ahi is not None and estimated_ahi >= 5)
        ) else "目前較不支持"
    result = {
        "patient_id": patient_id, "probability": round(probability, 4),
        "risk_level": level, "base_probability": round(base, 4),
        "watch_modifier": round(modifier, 4), "reasons": reasons,
        "xyz_features": {
            "movement_mean": None if pd.isna(movement_mean) else round(float(movement_mean), 4),
            "active_ratio": None if pd.isna(active_ratio) else round(float(active_ratio), 4),
            "accel_magnitude_std": None if pd.isna(accel_mag_std) else round(float(accel_mag_std), 4),
        },
        "estimated_ahi": round(estimated_ahi, 2) if estimated_ahi is not None else None,
        "estimated_event_count_range": event_count_range,
        "event_likelihood": event_likelihood,
        "event_count_definition": "整晚呼吸中止與低通氣事件合計的研究粗估範圍，不是 PSG 實測值。",
        "data_quality": quality, "model_version": bundle.get("version"),
        "trained_patients": bundle.get("trained_patients"), "bootstrap": True,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "interpretation": "預測今晚睡眠期間呼吸事件負荷升高（目前以 AHI≥15 為訓練目標）的研究風險；不是 OSA 診斷。",
        "limitation": bundle.get("limitations"),
        "clinical_use_status": "RESEARCH_DECISION_SUPPORT_ONLY",
        "medical_device_status": "NOT_VALIDATED_OR_APPROVED_AS_A_MEDICAL_DEVICE",
        "output_status": "AVAILABLE" if strict_quality["usable"] else "WITHHELD_LOW_DATA_QUALITY",
    }
    limited_no_spo2 = quality.get("quality") == "LIMITED_NO_SPO2"
    if limited_no_spo2:
        result.update({
            "output_status": "AVAILABLE_LIMITED_NO_SPO2",
            "clinical_confidence": "LOW",
            "limitations": [
                "本次沒有 SpO₂，今日穿戴資料只能提供心率與活動資訊。",
                "風險主要由既往 PSG、年齡與 BMI 推估，不等同完整穿戴式模型結果。",
                "事件次數是依既往 AHI 與睡眠時數換算的研究粗估，不是今晚實測。",
            ],
        })
        result["reasons"] = list(result.get("reasons") or []) + [
            "Bangle.js 心率與活動資料已納入，但缺少 SpO₂，因此降低可信度"
        ]
    elif not strict_quality["usable"]:
        result.update({
            "probability": None,
            "risk_level": "資料不足",
            "estimated_ahi": None,
            "estimated_event_count_range": None,
            "event_likelihood": "資料不足，暫不判定",
            "reasons": strict_quality.get("warnings") or [strict_quality.get("reason")],
        })
    output = root / "data" / "inference" / patient_id / "tonight_apnea_risk" / "tonight_risk_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_audit_event(root, patient_id, {
        "event_type": "TONIGHT_INFERENCE_GENERATED",
        "model_version": bundle.get("version"),
        "output_status": result["output_status"],
        "quality_gate": strict_quality,
        "input_mode": quality.get("input_mode"),
    })
    return result
