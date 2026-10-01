"""Build a deterministic, auditable patient factor summary.

This is the formal report source. It summarizes observed PSG phenotypes and
explicit treatment-rule contributions; it is deliberately not labelled SHAP.
Full-night SHAP is an optional research analysis with separate dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
INFERENCE_ROOT = PROJECT_ROOT / "data" / "inference"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到必要檔案：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON根節點必須是物件：{path}")
    return data


def factor(name: str, group: str, value: Any, direction: str) -> dict[str, Any]:
    return {
        "feature": name,
        "feature_group": group,
        "observed_value": value,
        "direction": direction,
        "mean_signed_shap": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="建立可稽核的患者觀察因素摘要")
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()
    patient_id = str(args.patient_id).strip()
    patient_root = INFERENCE_ROOT / patient_id

    respiratory = load_json(
        patient_root / "respiratory_profile" / "patient_respiratory_profile.json"
    )
    oxygen = load_json(
        patient_root / "oxygen_event_coupling" / "oxygen_event_coupling_summary.json"
    )
    treatment = load_json(
        patient_root / "treatment_refinement" / "refined_treatment_recommendation.json"
    )

    event = respiratory.get("event_summary") or {}
    spo2 = respiratory.get("spo2_profile") or {}
    rem = respiratory.get("rem_profile") or {}
    mechanism = respiratory.get("mechanism_profile") or {}
    coupling = oxygen.get("event_coupling") or {}

    increasing: list[dict[str, Any]] = []
    decreasing: list[dict[str, Any]] = []
    ahi = event.get("ahi")
    severity = str(event.get("ahi_severity") or "UNKNOWN")
    if severity in {"MILD", "MODERATE", "SEVERE"}:
        increasing.append(factor("AHI與嚴重度", "呼吸事件", {"AHI": ahi, "severity": severity}, "INCREASE_RISK"))
    if str(spo2.get("hypoxemia_level") or "UNKNOWN") in {"MODERATE_BURDEN", "HIGH_BURDEN"}:
        increasing.append(factor("夜間低氧負荷", "血氧", spo2.get("hypoxemia_level"), "INCREASE_RISK"))
    coupled = coupling.get("coupled_3pct_fraction")
    if coupled is not None and float(coupled) >= 0.50:
        increasing.append(factor("呼吸事件與至少3%血氧下降耦合", "事件－低氧耦合", coupled, "INCREASE_RISK"))
    if str(rem.get("rem_relevance") or "UNKNOWN") in {"MODERATE", "HIGH"}:
        increasing.append(factor("REM相關呼吸事件變化", "睡眠階段", rem.get("rem_relevance"), "INCREASE_RISK"))

    central_fraction = event.get("central_event_fraction")
    if central_fraction is not None and float(central_fraction) < 0.20:
        decreasing.append(factor("中央型事件比例未偏高", "呼吸事件", central_fraction, "DECREASE_RISK"))

    ranking = mechanism.get("mechanism_ranking") or []
    top_groups = [
        {"feature_group": str(item.get("mechanism")), "score": item.get("score")}
        for item in ranking[:5] if isinstance(item, dict)
    ]

    treatment_rows = treatment.get("personalized_treatment_ranking") or []
    treatment_rule_count = sum(
        len(row.get("supporting_factors") or []) + len(row.get("limiting_factors") or [])
        for row in treatment_rows if isinstance(row, dict)
    )
    summary_text = (
        f"患者{patient_id}：AHI={ahi}（{severity}），"
        f"血氧負荷={spo2.get('hypoxemia_level', 'UNKNOWN')}，"
        f"事件與至少3%血氧下降耦合比例={coupled}。"
        "本摘要來自已觀察PSG表型與可逐項驗算規則，不是SHAP或因果解釋。"
    )
    payload = {
        "schema_version": 1,
        "patient_id": patient_id,
        "explanation_method": "OBSERVED_PSG_PHENOTYPE_AND_RULE_CONTRIBUTIONS",
        "explanation_scope": "formal_digital_twin_report",
        "explanation_warning": "觀察關聯與規則貢獻不等於生理因果；本摘要不是SHAP值。",
        "generated_summary": summary_text,
        "top_physiological_groups": top_groups,
        "risk_increasing_factors": increasing,
        "risk_decreasing_factors": decreasing,
        "treatment_rule_count": treatment_rule_count,
        "source_files": [
            "respiratory_profile/patient_respiratory_profile.json",
            "oxygen_event_coupling/oxygen_event_coupling_summary.json",
            "treatment_refinement/refined_treatment_recommendation.json",
        ],
    }

    output_root = patient_root / "factor_summary"
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "patient_factor_summary.json"
    csv_path = output_root / "patient_factor_summary.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "patient_id", "explanation_method", "ahi", "ahi_severity",
            "oxygen_burden", "coupled_3pct_fraction", "treatment_rule_count",
        ])
        writer.writeheader()
        writer.writerow({
            "patient_id": patient_id,
            "explanation_method": payload["explanation_method"],
            "ahi": ahi,
            "ahi_severity": severity,
            "oxygen_burden": spo2.get("hypoxemia_level"),
            "coupled_3pct_fraction": coupled,
            "treatment_rule_count": treatment_rule_count,
        })
    print(f"Factor Summary JSON：{json_path}")
    print(f"Factor Summary CSV：{csv_path}")


if __name__ == "__main__":
    main()
