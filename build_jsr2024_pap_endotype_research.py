"""Build a transparent JSR 2024 PAP/endotype evidence summary.

This module records PSG observables that are relevant to the Cheng et al.
CPAP study.  It deliberately does not manufacture PUP/PUPpy endotypes when
the breath-by-breath ventilation model outputs are unavailable.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _number(value):
    try:
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None


def _divide(numerator, denominator):
    numerator = _number(numerator)
    denominator = _number(denominator)
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def build(patient_id: str) -> Path:
    patient_root = ROOT / "data" / "inference" / patient_id
    respiratory = _load(
        patient_root / "respiratory_profile" / "patient_respiratory_profile.json"
    )
    event = respiratory.get("event_summary") or {}
    sleep = respiratory.get("sleep_summary") or {}
    airflow = respiratory.get("airflow_profile") or {}

    ahi = _number(event.get("ahi"))
    sleep_hours = _divide(sleep.get("sleep_minutes"), 60.0)
    central_count = _number(event.get("central_apnea_count"))
    obstructive_count = _number(event.get("obstructive_apnea_count"))
    hypopnea_count = _number(event.get("hypopnea_count"))
    arousal_count = _number(event.get("arousal_count"))
    central_index = _divide(central_count, sleep_hours)
    obstructive_index = _divide(obstructive_count, sleep_hours)
    hypopnea_index = _divide(hypopnea_count, sleep_hours)
    arousal_index = _divide(arousal_count, sleep_hours)

    cohort_eligible = (
        ahi is not None and ahi >= 15
        and central_index is not None and central_index < 5
    )
    pup_requirements = {
        "nasal_airflow_or_ventilation": bool(airflow),
        "scored_respiratory_events": event.get("respiratory_event_count") is not None,
        "scored_eeg_arousals": arousal_count is not None,
        "breath_by_breath_ventilatory_drive_model": False,
        "validated_puppy_pipeline": False,
    }
    exact_pup_available = all(pup_requirements.values())
    payload = {
        "schema_version": "jsr-2024-pap-endotype-evidence-v1",
        "patient_id": patient_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "title": "Continuous positive airway pressure and adherence in patients with different endotypes of obstructive sleep apnea",
            "journal": "Journal of Sleep Research",
            "year": 2024,
            "doi": "10.1111/jsr.13999",
        },
        "study_cohort_eligibility": {
            "ahi_at_least_15": ahi is not None and ahi >= 15,
            "central_apnea_index_below_5": central_index is not None and central_index < 5,
            "matches_key_psg_criteria": cohort_eligible,
        },
        "observable_psg_features": {
            "ahi": ahi,
            "central_apnea_index": central_index,
            "obstructive_apnea_index": obstructive_index,
            "hypopnea_index": hypopnea_index,
            "arousal_index": arousal_index,
            "sleep_hours": sleep_hours,
        },
        "pup_requirements": pup_requirements,
        "exact_pup_endotypes_available": exact_pup_available,
        "pup_endotypes": {
            "arousal_threshold": {"status": "NOT_ESTIMATED"},
            "collapsibility": {"status": "NOT_ESTIMATED"},
            "loop_gain": {"status": "NOT_ESTIMATED"},
            "upper_airway_gain": {"status": "NOT_ESTIMATED"},
        },
        "model_use": {
            "observable_features_may_enter_pap_adherence_model": True,
            "pup_endotypes_may_affect_formal_score": False,
            "activation_rule": "Only activate an endotype feature after validated PUP/PUPpy extraction and held-out performance improvement.",
        },
        "interpretation": (
            "現有 PSG 可提供 AHI、事件型態、覺醒指數與睡眠結構等可觀察特徵；"
            "這些特徵可參與 PAP 依從性模型的驗證比較。四種 PUP 生理內型尚未正式估算，"
            "因此不會被填入假值，也不會直接改變治療排序。"
        ),
    }
    output_dir = patient_root / "jsr2024_pap_endotype_research"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "jsr2024_pap_endotype_summary.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()
    print(f"JSR 2024 PAP/endotype evidence summary: {build(args.patient_id.strip())}")


if __name__ == "__main__":
    main()
