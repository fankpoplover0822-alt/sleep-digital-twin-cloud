from __future__ import annotations

import copy
import json
import os
import time
import uuid
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


MODULE_KEY_MAP: dict[str, str] = {
    "PAP": "pap",
    "ENT": "ent",
    "DISE": "dise",
    "IMAGING": "imaging",
    "HYPOXEMIA": "hypoxemia",
    "ABG": "abg",
    "PULMONARY": "pulmonary",
    "CARDIAC": "cardiac",
    "ECG": "ecg",
    "ECHOCARDIOGRAPHY": "echocardiography",
    "COMORBIDITY": "comorbidities",
    "MEDICATION": "medication",
    "PATIENT_PREFERENCE": "patient_preference",
    "FOLLOW_UP": "follow_up",
    "ANTHROPOMETRY": "anthropometry",
    "WEIGHT_MANAGEMENT": "weight_management",
    "LIFESTYLE": "lifestyle",
    "SLEEP_QUESTIONNAIRE": "sleep_questionnaire",
    "DENTAL_CRANIOFACIAL": "dental_craniofacial",
    "LABORATORY": "laboratory",
    "NEUROLOGICAL": "neurological",
    "PSYCHIATRIC": "psychiatric",
    "OXYGEN_THERAPY": "oxygen_therapy",
    "SURGERY_HISTORY": "surgery_history",
    "DEVICE_DATA": "device_data",
    "OTHER": "other",
}


def _json_safe(value: Any) -> Any:
    """Return a strict-JSON-safe deep copy of *value*."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _clean_confirmed_findings(
    module_id: str,
    confirmed_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove empty values so they cannot erase valid existing findings."""
    if not isinstance(confirmed_data, Mapping):
        raise TypeError("confirmed_data 必須是 mapping/dict。")

    cleaned = {
        str(key): _json_safe(value)
        for key, value in confirmed_data.items()
        if value is not None and value != ""
    }

    # PAP refinement rules require a mode-specific target.
    if module_id == "PAP":
        cleaned.setdefault("therapy_type", "CPAP")

    return cleaned


def merge_clinical_decision_data(
    original_data: Mapping[str, Any],
    *,
    patient_id: str,
    module_id: str,
    confirmed_data: Mapping[str, Any],
    source_files: Sequence[str] | None = None,
    updated_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Merge one physician-confirmed update into ClinicalDecisionData.

    Existing unrelated modules and existing non-empty findings are preserved.
    Returns ``(updated_data, change_record)`` without writing files.
    """
    normalized_module_id = str(module_id).strip().upper()
    module_key = MODULE_KEY_MAP.get(normalized_module_id)
    if module_key is None:
        supported = ", ".join(sorted(MODULE_KEY_MAP))
        raise ValueError(
            f"尚未支援資料類型 {module_id!r}；目前支援：{supported}"
        )

    if not isinstance(original_data, Mapping):
        raise TypeError("original_data 必須是 mapping/dict。")

    updated = copy.deepcopy(_json_safe(dict(original_data)))
    module_record = updated.get(module_key)
    if not isinstance(module_record, dict):
        module_record = {}
        updated[module_key] = module_record

    existing_findings = module_record.get("findings")
    if not isinstance(existing_findings, dict):
        existing_findings = {}

    incoming_findings = _clean_confirmed_findings(
        normalized_module_id,
        confirmed_data,
    )

    changed_fields: dict[str, dict[str, Any]] = {}
    for field_name, new_value in incoming_findings.items():
        old_value = existing_findings.get(field_name)
        if old_value != new_value:
            changed_fields[field_name] = {
                "before": old_value,
                "after": new_value,
            }

    # Identical content is a no-op: do not inflate the patient version merely
    # because the same file was uploaded again.
    if not changed_fields:
        try:
            unchanged_version = int(updated.get("data_version", 0))
        except (TypeError, ValueError):
            unchanged_version = 0
        return updated, {
            "patient_id": str(patient_id).strip(),
            "module_id": normalized_module_id,
            "module_key": module_key,
            "version_before": unchanged_version,
            "version_after": unchanged_version,
            "updated_at": updated.get("last_updated_at"),
            "source_files": list(source_files or []),
            "changed_fields": {},
            "status": "duplicate_content_skipped",
        }

    module_record["available"] = True
    module_record["findings"] = {
        **existing_findings,
        **incoming_findings,
    }
    module_record["source"] = (
        ", ".join(str(name) for name in (source_files or []) if str(name).strip())
        or "Streamlit clinical update"
    )
    module_record["notes"] = (
        "Confirmed through Sleep Digital Twin clinical update workflow."
    )

    try:
        previous_version = int(updated.get("data_version", 0))
    except (TypeError, ValueError):
        previous_version = 0

    timestamp = updated_at or datetime.now().isoformat(timespec="seconds")
    updated["data_version"] = previous_version + 1
    updated["patient_id"] = str(patient_id).strip()
    updated["last_updated_at"] = timestamp
    updated["last_updated_module"] = normalized_module_id

    change_record = {
        "patient_id": str(patient_id).strip(),
        "module_id": normalized_module_id,
        "module_key": module_key,
        "version_before": previous_version,
        "version_after": previous_version + 1,
        "updated_at": timestamp,
        "source_files": list(source_files or []),
        "changed_fields": changed_fields,
    }

    return updated, change_record


def save_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write strict JSON atomically, preventing partial/corrupt files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )

    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            _json_safe(dict(payload)),
            output_file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

        output_file.flush()
        os.fsync(output_file.fileno())

    last_error: PermissionError | None = None
    try:
        for attempt in range(8):
            try:
                os.replace(temporary_path, path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.08 * (attempt + 1))
        raise last_error or PermissionError(f"無法取代檔案：{path}")
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except PermissionError:
            pass


def update_clinical_decision_file(
    clinical_data_path: Path,
    *,
    patient_id: str,
    module_id: str,
    confirmed_data: Mapping[str, Any],
    source_files: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Load, merge and atomically save ``clinical_decision_data.json``.

    Returns ``(original_data, updated_data, change_record)``.
    """
    clinical_data_path = Path(clinical_data_path)
    if not clinical_data_path.is_file():
        raise FileNotFoundError(
            f"找不到 ClinicalDecisionData：{clinical_data_path}"
        )

    with clinical_data_path.open("r", encoding="utf-8") as input_file:
        original_data = json.load(input_file)

    updated_data, change_record = merge_clinical_decision_data(
        original_data,
        patient_id=patient_id,
        module_id=module_id,
        confirmed_data=confirmed_data,
        source_files=source_files,
    )

    save_json_atomic(clinical_data_path, updated_data)
    return original_data, updated_data, change_record
