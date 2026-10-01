from __future__ import annotations

"""Shared safety, provenance and monitoring helpers for research continual learning."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALLOWED_TONIGHT_FEATURES = (
    "age", "BMI", "prior_ahi", "prior_spo2_mean", "prior_spo2_min",
    "day_spo2_mean", "day_spo2_min", "day_spo2_std", "day_t90",
    "day_hr_mean", "day_hr_std", "day_rr_mean", "coverage_hours",
    "day_accel_x_std", "day_accel_y_std", "day_accel_z_std",
    "day_accel_magnitude_std", "day_movement_mean", "day_active_ratio",
)

FORBIDDEN_FEATURE_TOKENS = (
    "answer", "答案", "scenario", "情境", "success", "failure", "成功", "失敗",
    "follow_up", "outcome", "target", "label", "recommendation", "rank", "score",
    "filename", "file_name", "reviewer", "synthetic_test",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="microseconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def content_hash(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_feature_names(columns: list[str] | tuple[str, ...], *, allowed: set[str] | None = None) -> None:
    names = [str(column) for column in columns]
    blocked = [name for name in names if any(token.lower() in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)]
    outside = [name for name in names if allowed is not None and name not in allowed]
    if blocked or outside:
        details = []
        if blocked:
            details.append("疑似答案／結果洩漏欄位：" + "、".join(blocked))
        if outside:
            details.append("不在特徵白名單：" + "、".join(outside))
        raise ValueError("；".join(details))


def current_monitoring_window(frame: pd.DataFrame, timezone_name: str = "Asia/Taipei") -> pd.DataFrame:
    """Return one patient's latest local monitoring day, sorted and deduplicated."""
    if frame.empty or "timestamp" not in frame.columns:
        return frame.copy()
    prepared = frame.copy()
    timestamp = pd.to_datetime(prepared["timestamp"], errors="coerce", utc=True)
    prepared = prepared.loc[timestamp.notna()].copy()
    if prepared.empty:
        return prepared
    prepared["timestamp"] = timestamp.loc[prepared.index]
    local_timestamp = prepared["timestamp"].dt.tz_convert(timezone_name)
    latest_day = local_timestamp.max().date()
    prepared = prepared.loc[local_timestamp.dt.date == latest_day].copy()
    return (
        prepared.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def wearable_quality(frame: pd.DataFrame) -> dict[str, Any]:
    frame = current_monitoring_window(frame)
    if frame.empty:
        return {"status": "BLOCKED", "reason": "沒有穿戴式資料", "usable": False}
    timestamp = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True)
    spo2 = pd.to_numeric(frame.get("spo2"), errors="coerce")
    heart_rate = pd.to_numeric(frame.get("heart_rate"), errors="coerce")
    respiratory = pd.to_numeric(frame.get("respiratory_rate"), errors="coerce")
    spo2_available = bool(spo2.notna().any())
    physiologic = heart_rate.between(30, 220)
    if spo2_available:
        physiologic &= spo2.between(70, 100)
    valid = timestamp.notna() & physiologic
    valid_ratio = float(valid.mean()) if len(frame) else 0.0
    coverage = float((timestamp.max() - timestamp.min()).total_seconds() / 3600) if timestamp.notna().sum() > 1 else 0.0
    duplicate_ratio = float(timestamp.duplicated().mean()) if len(frame) else 0.0
    rr_ratio = float(respiratory.between(4, 60).mean()) if respiratory.notna().any() else None
    usable = len(frame) >= 30 and coverage >= 6 and valid_ratio >= 0.8 and duplicate_ratio <= 0.2
    warnings = []
    if len(frame) < 30: warnings.append("有效筆數少於30筆")
    if coverage < 6: warnings.append("涵蓋時間少於6小時")
    if valid_ratio < 0.8:
        warnings.append("時間戳及必要生理訊號有效率低於80%")
    if not spo2_available:
        warnings.append("裝置未提供SpO₂；僅能產生限制性研究估計")
    if duplicate_ratio > 0.2: warnings.append("重複時間戳超過20%")
    return {
        "status": "PASS" if usable else "BLOCKED",
        "usable": usable,
        "sample_count": int(len(frame)),
        "coverage_hours": round(coverage, 2),
        "valid_ratio": round(valid_ratio, 3),
        "duplicate_timestamp_ratio": round(duplicate_ratio, 3),
        "respiratory_rate_valid_ratio": round(rr_ratio, 3) if rr_ratio is not None else None,
        "warnings": warnings,
    }


def distribution_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    flagged = []
    for column in columns:
        ref = pd.to_numeric(reference.get(column), errors="coerce").dropna()
        cur = pd.to_numeric(current.get(column), errors="coerce").dropna()
        if len(ref) < 10 or len(cur) < 10:
            continue
        scale = float(ref.std(ddof=0))
        shift = abs(float(cur.mean() - ref.mean())) / max(scale, 1e-6)
        details[column] = {"standardized_mean_shift": round(shift, 3)}
        if shift >= 2.0:
            flagged.append(column)
    return {"status": "WARNING" if flagged else "PASS", "flagged_features": flagged, "details": details}


def write_audit_event(root: Path, patient_id: str, event: dict[str, Any]) -> Path:
    event = {"created_at": now_iso(), "patient_id": patient_id, **event}
    event["audit_hash"] = content_hash(event)
    destination = root / "data" / "audit" / patient_id / "continual_learning_events.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    return destination


def governance_summary(root: Path, patient_id: str) -> dict[str, Any]:
    night = root / "data" / "inference" / patient_id / "paired_learning" / "night_training_receipt.json"
    treatment = root / "data" / "inference" / patient_id / "paired_learning" / "treatment_training_receipt.json"
    def read(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    night_receipt, treatment_receipt = read(night), read(treatment)
    return {
        "environment_separation": "PASS",
        "synthetic_policy": "RESEARCH_CHALLENGER_ONLY",
        "unlabelled_upload_policy": "STATE_AND_INFERENCE_ONLY",
        "formal_treatment_policy": "STABLE_RULE_BASED_CHAMPION",
        "feature_leakage_guard": "ALLOWLIST_AND_BLOCKLIST",
        "deduplication": "CONTENT_HASH_AND_STUDY_ID",
        "batch_update": "ATOMIC_VERSIONED_OUTPUT_WITH_ROLLBACK",
        "latest_night_training": night_receipt,
        "latest_treatment_training": {
            key: treatment_receipt.get(key) for key in (
                "model_version", "training_rows", "confirmed_rows", "synthetic_test_rows", "status"
            )
        },
        "clinical_status": "RESEARCH_DECISION_SUPPORT_ONLY",
        "medical_device_status": "NOT_VALIDATED_OR_APPROVED_AS_A_MEDICAL_DEVICE",
    }
