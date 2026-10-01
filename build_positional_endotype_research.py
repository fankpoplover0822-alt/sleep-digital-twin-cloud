"""Build a transparent positional-OSA/endotype research summary.

Implements the directly reproducible definitions and quality gates described by
Cheng et al. (ERJ 2024). It does not claim to reproduce PUP/PUPpy endotypes when
the required breath-by-breath model outputs are unavailable.
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


def build(patient_id: str) -> Path:
    patient_root = ROOT / "data" / "inference" / patient_id
    respiratory = _load(
        patient_root / "respiratory_profile" / "patient_respiratory_profile.json"
    )
    position_summary = _load(
        patient_root / "position_profile" / "patient_position_profile.json"
    )
    event = respiratory.get("event_summary") or {}
    sleep = respiratory.get("sleep_summary") or {}
    position = respiratory.get("position_profile") or {}
    spo2 = respiratory.get("spo2_profile") or {}
    effort = respiratory.get("respiratory_effort_profile") or {}
    airflow = respiratory.get("airflow_profile") or {}

    ahi = _number(event.get("ahi"))
    sleep_minutes = _number(sleep.get("sleep_minutes"))
    supine_hours = _number(position.get("supine_hours")) or 0.0
    lateral_hours = _number(position.get("lateral_hours")) or 0.0
    supine_ahi = _number(position.get("supine_event_epoch_index"))
    lateral_ahi = _number(position.get("lateral_event_epoch_index"))
    ratio = None
    if supine_ahi is not None and lateral_ahi is not None:
        ratio = float("inf") if lateral_ahi == 0 and supine_ahi > 0 else (
            supine_ahi / lateral_ahi if lateral_ahi > 0 else None
        )

    anatomical_mapping = position_summary.get("anatomical_position_mapping")
    known_fraction = _number(position.get("known_position_epoch_fraction")) or 0.0
    mapping_valid = known_fraction > 0 and anatomical_mapping != "UNCALIBRATED"
    duration_valid = supine_hours >= 0.5 and lateral_hours >= 0.5
    phenotype = "INDETERMINATE"
    if mapping_valid and duration_valid and ratio is not None:
        phenotype = "SUPINE_PREDOMINANT_OSA" if ratio > 2 else "NON_POSITIONAL_OSA"

    spo2_fraction = _number(spo2.get("spo2_valid_epoch_fraction"))
    spo2_min = _number(spo2.get("spo2_valid_absolute_min"))
    quality = {
        "ahi_over_5": ahi is not None and ahi > 5,
        "total_sleep_time_over_240_minutes": (
            sleep_minutes is not None and sleep_minutes > 240
        ),
        "supine_sleep_at_least_30_minutes": supine_hours >= 0.5,
        "lateral_sleep_at_least_30_minutes": lateral_hours >= 0.5,
        "anatomical_position_labels_available": mapping_valid,
        "spo2_available_and_usable": (
            spo2_fraction is not None and spo2_fraction >= 0.8
            and spo2_min is not None and spo2_min >= 40
        ),
        "nasal_flow_features_available": bool(airflow),
        "respiratory_effort_features_available": bool(effort),
        "scored_respiratory_events_available": (
            _number(event.get("respiratory_event_count")) is not None
        ),
        "scored_eeg_arousals_available": _number(event.get("arousal_count")) is not None,
        "periodic_limb_movement_index_at_most_15": None,
    }
    missing = [name for name, passed in quality.items() if passed is False]
    not_evaluated = [name for name, passed in quality.items() if passed is None]

    loop_proxy = respiratory.get("loop_gain_proxy") or {}
    endotypes = {
        "collapsibility": {
            "status": "NOT_ESTIMATED",
            "required_method": "PUP breath-by-breath chemical-drive model; 1 - Vpassive",
        },
        "upper_airway_muscle_compensation": {
            "status": "NOT_ESTIMATED",
            "required_method": "PUP Vactive - Vpassive",
        },
        "arousal_threshold": {
            "status": "NOT_ESTIMATED",
            "required_method": "chemical drive immediately before scored EEG arousal",
        },
        "loop_gain": {
            "status": "PROXY_ONLY" if loop_proxy else "NOT_ESTIMATED",
            "proxy_score": _number(loop_proxy.get("loop_gain_proxy_score")),
            "proxy_level": loop_proxy.get("loop_gain_proxy_level"),
            "warning": "現有分數不是論文 PUP 在 1 cycle/min 的 open-loop gain。",
        },
    }
    if phenotype == "SUPINE_PREDOMINANT_OSA":
        interpretation = (
            "仰睡 AHI／側睡 AHI 大於 2，符合本研究採用的 spOSA 研究定義；"
            "可支持醫師進一步評估姿勢治療，但不可自動取代 PAP 或其他正式治療。"
        )
    elif phenotype == "NON_POSITIONAL_OSA":
        interpretation = (
            "具足夠且已校正的仰睡與側睡資料，AHI 比值未超過 2，"
            "依論文定義屬非姿勢型 OSA。"
        )
    else:
        interpretation = (
            "目前不能判定 spOSA：必須具備已校正的解剖姿勢標籤，且仰睡與側睡各至少 30 分鐘。"
        )

    payload = {
        "schema_version": "erj-2024-positional-endotype-v1",
        "patient_id": patient_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module": "ERJ 2024 positional OSA and endotype research module",
        "paper_doi": "10.1183/13993003.01660-2023",
        "status": "completed",
        "positional_phenotype": phenotype,
        "metrics": {
            "whole_night_ahi": ahi,
            "supine_ahi": supine_ahi,
            "lateral_ahi": lateral_ahi,
            "supine_to_lateral_ahi_ratio": ratio,
            "supine_sleep_minutes": round(supine_hours * 60, 2),
            "lateral_sleep_minutes": round(lateral_hours * 60, 2),
            "total_sleep_minutes": sleep_minutes,
        },
        "quality_gates": quality,
        "failed_quality_gates": missing,
        "not_evaluated_quality_gates": not_evaluated,
        "pup_endotypes": endotypes,
        "interpretation": interpretation,
        "evidence_context": {
            "study_population": "689 adults with OSA from one Taiwan sleep centre",
            "study_sposa_prevalence_percent": 75.8,
            "key_finding": (
                "In the study cohort, supine-predominant OSA showed a larger reduction "
                "in upper-airway muscle compensation from lateral to supine sleep."
            ),
        },
        "research_safety_note": (
            "此模組為論文定義與資料完整度檢查，不是 PUPpy 的替代品；"
            "觀察性單中心亞洲族群結果不可直接當作個別患者的因果診斷。"
        ),
    }
    output_dir = patient_root / "positional_endotype_research"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "positional_endotype_summary.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()
    output = build(args.patient_id.strip())
    print(f"ERJ positional endotype research summary：{output}")


if __name__ == "__main__":
    main()
