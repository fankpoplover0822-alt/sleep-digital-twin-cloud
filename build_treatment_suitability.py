from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clinical_treatment_guardrails import apply_medical_guardrails
from clinical_context_integration import merge_clinical_context

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

INFERENCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "inference"
)


# ============================================================
# 基礎工具
# ============================================================

def safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(result):
        return None

    return result


def safe_int(
    value: Any,
) -> int | None:
    number = safe_float(value)

    if number is None:
        return None

    return int(round(number))


def safe_divide(
    numerator: float | int,
    denominator: float | int,
) -> float | None:
    denominator_value = safe_float(
        denominator
    )

    if (
        denominator_value is None
        or denominator_value == 0
    ):
        return None

    result = (
        float(numerator)
        / denominator_value
    )

    if not np.isfinite(result):
        return None

    return float(result)


def safe_json_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): safe_json_value(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            safe_json_value(item)
            for item in value
        ]

    if isinstance(
        value,
        (
            np.integer,
            np.int32,
            np.int64,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float32,
            np.float64,
        ),
    ):
        if not np.isfinite(value):
            return None

        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat(
            sep=" "
        )

    try:
        if pd.isna(value):
            return None
    except (
        TypeError,
        ValueError,
    ):
        pass

    return value


def load_json(
    file_path: Path,
) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到 JSON：{file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"JSON 內容不是 object：{file_path}"
        )

    return data


def save_json(
    file_path: Path,
    data: dict[str, Any],
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_data = safe_json_value(
        data
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            clean_data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def get_nested(
    data: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(
            current,
            dict,
        ):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def clamp_score(
    value: float,
) -> float:
    return float(
        min(
            max(
                float(value),
                0.0,
            ),
            100.0,
        )
    )


def score_to_level(
    score: float,
) -> str:
    if score >= 75:
        return "HIGH"

    if score >= 50:
        return "MODERATE"

    if score >= 25:
        return "LOW"

    return "VERY_LOW"


def confidence_to_level(
    completeness: float,
) -> str:
    if completeness >= 0.85:
        return "HIGH"

    if completeness >= 0.65:
        return "MODERATE"

    if completeness >= 0.40:
        return "LOW"

    return "VERY_LOW"


def recommendation_status(
    score: float,
    confidence_level: str,
    safety_hold: bool,
) -> str:
    if safety_hold:
        return "CONDITIONAL"

    if (
        score >= 50
        and confidence_level
        in {
            "HIGH",
            "MODERATE",
        }
    ):
        return "CANDIDATE"

    if score >= 25:
        return "POSSIBLE"

    return "NOT_PRIORITIZED"


def add_reason(
    reasons: list[dict[str, Any]],
    code: str,
    description: str,
    weight: float,
) -> None:
    reasons.append(
        {
            "code": code,
            "description": description,
            "weight": float(weight),
        }
    )


def unique_strings(
    values: list[str],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value).strip()

        if not text:
            continue

        if text in seen:
            continue

        seen.add(text)
        result.append(text)

    return result


def calculate_evidence_completeness(
    evidence_items: dict[str, bool],
) -> dict[str, Any]:
    total_count = len(
        evidence_items
    )

    available_count = int(
        sum(
            bool(value)
            for value
            in evidence_items.values()
        )
    )

    completeness = safe_divide(
        available_count,
        total_count,
    )

    if completeness is None:
        completeness = 0.0

    missing_items = [
        name
        for name, available
        in evidence_items.items()
        if not available
    ]

    available_items = [
        name
        for name, available
        in evidence_items.items()
        if available
    ]

    return {
        "available_count": (
            available_count
        ),
        "total_count": (
            total_count
        ),
        "completeness": (
            completeness
        ),
        "confidence_level": (
            confidence_to_level(
                completeness
            )
        ),
        "available_evidence": (
            available_items
        ),
        "missing_evidence": (
            missing_items
        ),
    }


# ============================================================
# 共用患者資料
# ============================================================

def extract_patient_context(
    profile: dict[str, Any],
    position_analysis: dict[str, Any],
    oxygen_coupling: dict[str, Any],
) -> dict[str, Any]:
    event_summary = get_nested(
        profile,
        "event_summary",
        default={},
    )

    spo2_profile = get_nested(
        profile,
        "spo2_profile",
        default={},
    )

    rem_profile = get_nested(
        profile,
        "rem_profile",
        default={},
    )

    loop_gain_profile = get_nested(
        profile,
        "loop_gain_proxy",
        default={},
    )

    mechanism_profile = get_nested(
        profile,
        "mechanism_profile",
        default={},
    )

    low_oxygen_attribution = get_nested(
        oxygen_coupling,
        "low_oxygen_attribution",
        default={},
    )

    event_coupling = get_nested(
        oxygen_coupling,
        "event_coupling",
        default={},
    )

    wake_sleep_oxygen = get_nested(
        oxygen_coupling,
        "wake_sleep_oxygen",
        default={},
    )

    return {
        "sex": (
            profile.get("sex")
        ),
        "age": safe_float(
            profile.get("age")
        ),
        "BMI": safe_float(
            profile.get("BMI")
        ),
        "ahi": safe_float(
            event_summary.get("ahi")
        ),
        "ahi_severity": str(
            event_summary.get(
                "ahi_severity",
                "UNKNOWN",
            )
        ).upper(),
        "hypopnea_count": safe_int(
            event_summary.get(
                "hypopnea_count"
            )
        ),
        "hypopnea_mechanism_classified": bool(
            event_summary.get(
                "hypopnea_mechanism_classified",
                False,
            )
        ),
        "obstructive_apnea_count": (
            safe_int(
                event_summary.get(
                    "obstructive_apnea_count"
                )
            )
        ),
        "central_apnea_count": (
            safe_int(
                event_summary.get(
                    "central_apnea_count"
                )
            )
        ),
        "central_event_fraction": (
            safe_float(
                event_summary.get(
                    "central_event_fraction"
                )
            )
        ),
        "obstructive_event_fraction": (
            safe_float(
                event_summary.get(
                    "obstructive_event_fraction"
                )
            )
        ),
        "oxygen_burden": str(
            spo2_profile.get(
                "hypoxemia_level",
                "UNKNOWN",
            )
        ).upper(),
        "desaturation_depth": str(
            spo2_profile.get(
                "desaturation_depth",
                "UNKNOWN",
            )
        ).upper(),
        "sustained_oxygen_burden": str(
            spo2_profile.get(
                "sustained_low_oxygen_burden",
                "UNKNOWN",
            )
        ).upper(),
        "rem_relevance": str(
            rem_profile.get(
                "rem_relevance",
                "UNKNOWN",
            )
        ).upper(),
        "loop_gain_level": str(
            loop_gain_profile.get(
                "loop_gain_proxy_level",
                "UNKNOWN",
            )
        ).upper(),
        "dominant_mechanism": str(
            mechanism_profile.get(
                "dominant_mechanism",
                "UNKNOWN",
            )
        ).upper(),
        "position_relevance": str(
            position_analysis.get(
                "position_relevance",
                "UNKNOWN",
            )
        ).upper(),
        "position_event_ratio": safe_float(
            position_analysis.get(
                "position_event_index_ratio"
            )
        ),
        "known_position_fraction": safe_float(
            position_analysis.get(
                "known_position_epoch_fraction"
            )
        ),
        "position_mapping": str(
            position_analysis.get(
                "anatomical_position_mapping",
                "UNCALIBRATED",
            )
        ).upper(),
        "oxygen_coupling_level": str(
            oxygen_coupling.get(
                "coupling_level",
                "UNKNOWN",
            )
        ).upper(),
        "coupled_3pct_fraction": safe_float(
            event_coupling.get(
                "coupled_3pct_fraction"
            )
        ),
        "coupled_4pct_fraction": safe_float(
            event_coupling.get(
                "coupled_4pct_fraction"
            )
        ),
        "low_oxygen_away_fraction": (
            safe_float(
                low_oxygen_attribution.get(
                    "spo2_below_90_away_from_event_fraction"
                )
            )
        ),
        "low_oxygen_away_minutes": (
            safe_float(
                low_oxygen_attribution.get(
                    "spo2_below_90_away_from_event_minutes"
                )
            )
        ),
        "low_oxygen_total_minutes": (
            safe_float(
                low_oxygen_attribution.get(
                    "spo2_below_90_minutes"
                )
            )
        ),
        "wake_spo2": safe_float(
            wake_sleep_oxygen.get(
                "wake_spo2_mean"
            )
        ),
        "sleep_spo2": safe_float(
            wake_sleep_oxygen.get(
                "sleep_spo2_mean"
            )
        ),
    }


# ============================================================
# CPAP
# 僅依已觀察證據計分
# 缺資料不加分、不扣分
# ============================================================

def build_projected_score_changes(
    treatment: str,
    evidence: dict[str, Any],
) -> list[str]:
    missing_evidence = evidence.get(
        "missing_evidence",
        [],
    )

    projected_score_changes = []

    if treatment == "CPAP":
        if "PAP 滴定資料" in missing_evidence:
            projected_score_changes.append(
                "完成 PAP 滴定：預估 +5～10 分"
            )

        if "PAP 耐受資料" in missing_evidence:
            projected_score_changes.append(
                "完成 PAP 耐受評估："
                "可重新確認固定 CPAP 是否為最佳治療"
            )

    elif treatment == "APAP":
        if "APAP 壓力範圍" in missing_evidence:
            projected_score_changes.append(
                "完成 APAP 壓力範圍設定："
                "預估 +5～10 分"
            )

        if "PAP 耐受資料" in missing_evidence:
            projected_score_changes.append(
                "完成 PAP 耐受評估："
                "可重新確認 APAP 是否為最佳治療"
            )

    elif treatment == "SURGERY_EVALUATION":
        if "耳鼻喉內視鏡" in missing_evidence:
            projected_score_changes.append(
                "完成耳鼻喉內視鏡："
                "若確認可矯正的上呼吸道阻塞，"
                "手術評估分數可能提高"
            )

    elif (
        treatment
        == "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW"
    ):
        if (
            "完整藥物名稱與作用機轉"
            in missing_evidence
        ):
            projected_score_changes.append(
                "完成目前用藥與作用機轉確認："
                "可重新評估是否適合"
                "睡眠結構調節型藥物"
            )

        if (
            "治療後 PSG 或療效監測"
            in missing_evidence
        ):
            projected_score_changes.append(
                "完成治療後 PSG 或療效監測："
                "可重新評估治療效益"
            )

    return projected_score_changes



def evaluate_cpap(
    context: dict[str, Any],
) -> dict[str, Any]:
    score = 20.0

    supporting: list[
        dict[str, Any]
    ] = []

    limiting: list[
        dict[str, Any]
    ] = []

    severity = context[
        "ahi_severity"
    ]

    oxygen_burden = context[
        "oxygen_burden"
    ]
    coupled_3pct_fraction = context[
        "coupled_3pct_fraction"
    ]

    central_fraction = context[
        "central_event_fraction"
    ]

    loop_gain_level = context[
        "loop_gain_level"
    ]

    coupling_level = context[
        "oxygen_coupling_level"
    ]

    coupled_3pct_fraction = context[
        "coupled_3pct_fraction"
    ]

    low_oxygen_away_fraction = context[
        "low_oxygen_away_fraction"
    ]

    if severity == "SEVERE":
        score += 45

        add_reason(
            supporting,
            "SEVERE_AHI",
            (
                "AHI 位於重度範圍，"
                "支持固定壓力 PAP 評估。"
            ),
            45,
        )

    elif severity == "MODERATE":
        score += 35

        add_reason(
            supporting,
            "MODERATE_AHI",
            (
                "AHI 位於中度範圍，"
                "支持固定壓力 PAP 評估。"
            ),
            35,
        )

    elif severity == "MILD":
        score += 15

        add_reason(
            supporting,
            "MILD_AHI",
            (
                "AHI 位於輕度範圍，"
                "固定壓力 PAP 可列為候選。"
            ),
            15,
        )

    elif severity == "NORMAL":
        score -= 15

        add_reason(
            limiting,
            "NORMAL_AHI",
            (
                "AHI 未達常用異常範圍，"
                "缺少固定壓力 PAP 的支持。"
            ),
            -15,
        )

    if oxygen_burden == "HIGH_BURDEN":
        score += 10

        add_reason(
            supporting,
            "HIGH_OXYGEN_BURDEN",
            (
                "夜間血氧負荷偏高，"
                "支持評估呼吸支持治療反應。"
            ),
            10,
        )

    elif oxygen_burden == "MODERATE_BURDEN":
        score += 5

        add_reason(
            supporting,
            "MODERATE_OXYGEN_BURDEN",
            (
                "存在中度夜間血氧負荷，"
                "可納入 PAP 適配度判斷。"
            ),
            5,
        )

    if (
        coupled_3pct_fraction is not None
        and coupled_3pct_fraction >= 0.50
    ):
        score += 10

        add_reason(
            supporting,
            "EVENT_DESATURATION_COUPLING",
            (
                "至少一半可分析呼吸事件"
                "伴隨 ≥3% 血氧下降，"
                "支持事件具有生理影響。"
            ),
            10,
        )

    if (
        coupling_level
        == "PARTIAL_EVENT_COUPLING"
        and low_oxygen_away_fraction is not None
        and low_oxygen_away_fraction >= 0.70
    ):
        score -= 5

        add_reason(
            limiting,
            "NON_EVENT_HYPOXEMIA",
            (
                "大部分低氧時間遠離呼吸事件，"
                "固定壓力 PAP 未必能處理"
                "全部持續性低氧。"
            ),
            -5,
        )

    if (
        central_fraction is not None
        and central_fraction >= 0.20
    ):
        score -= 25

        add_reason(
            limiting,
            "CENTRAL_EVENT_BURDEN",
            (
                "中央型事件比例偏高，"
                "單純固定 CPAP 需更審慎。"
            ),
            -25,
        )

    if loop_gain_level == "HIGH":
        score -= 10

        add_reason(
            limiting,
            "HIGH_LOOP_GAIN_PROXY",
            (
                "呼吸控制不穩定 proxy 偏高，"
                "固定 CPAP 適配度下降。"
            ),
            -10,
        )

    evidence = calculate_evidence_completeness(
        {
            "AHI 與嚴重度": (
                context["ahi"] is not None
                and severity != "UNKNOWN"
            ),
            "呼吸事件類型": (
                context[
                    "central_apnea_count"
                ]
                is not None
            ),
            "血氧負荷": (
                oxygen_burden
                != "UNKNOWN"
            ),
            "事件－低氧耦合": (
                coupling_level
                != "UNKNOWN"
            ),
            "Loop Gain proxy": (
                loop_gain_level
                != "UNKNOWN"
            ),
            "PAP 滴定資料": False,
            "PAP 耐受資料": False,
        }
    )

    safety_hold = bool(
        (
            context["wake_spo2"]
            is not None
            and context["wake_spo2"]
            < 94.0
        )
        or (
            low_oxygen_away_fraction
            is not None
            and low_oxygen_away_fraction
            >= 0.70
        )
    )

    score = clamp_score(
        score
    )

    projected_score_changes = (
        build_projected_score_changes(
            "CPAP",
            evidence,
        )
    )

    return {
        "treatment": "CPAP",
        "treatment_label": (
            "固定壓力正壓呼吸器"
        ),
        "score": score,
        "suitability_level": (
            score_to_level(score)
        ),
        "recommendation_status": (
            recommendation_status(
                score=score,
                confidence_level=(
                    evidence[
                        "confidence_level"
                    ]
                ),
                safety_hold=(
                    safety_hold
                ),
            )
        ),
        "confidence": evidence,
        "supporting_factors": supporting,
        "limiting_factors": limiting,
        "interpretation": (
            "此分數只依目前已觀察到的患者資料"
            "評估固定壓力 PAP 適配度。"
        ),
        "projected_score_changes": (
            projected_score_changes
        ),
    }

# ============================================================
# APAP
# ============================================================

def evaluate_apap(
    context: dict[str, Any],
) -> dict[str, Any]:
    score = 20.0

    supporting: list[
        dict[str, Any]
    ] = []

    limiting: list[
        dict[str, Any]
    ] = []

    severity = context[
        "ahi_severity"
    ]

    rem_relevance = context[
        "rem_relevance"
    ]

    position_relevance = context[
        "position_relevance"
    ]
    position_event_ratio = context[
        "position_event_ratio"
    ]

    central_count = context[
        "central_apnea_count"
    ]

    loop_gain_level = context[
        "loop_gain_level"
    ]

    oxygen_burden = context[
        "oxygen_burden"
    ]

    coupled_3pct_fraction = context[
        "coupled_3pct_fraction"
    ]

    low_oxygen_away_fraction = context[
        "low_oxygen_away_fraction"
    ]

    if severity == "SEVERE":
        score += 45

        add_reason(
            supporting,
            "SEVERE_AHI",
            (
                "AHI 位於重度範圍，"
                "支持自動調壓 PAP 評估。"
            ),
            45,
        )

    elif severity == "MODERATE":
        score += 35

        add_reason(
            supporting,
            "MODERATE_AHI",
            (
                "AHI 位於中度範圍，"
                "支持自動調壓 PAP 評估。"
            ),
            35,
        )

    elif severity == "MILD":
        score += 15

        add_reason(
            supporting,
            "MILD_AHI",
            (
                "AHI 位於輕度範圍，"
                "APAP 可列為候選。"
            ),
            15,
        )

    elif severity == "NORMAL":
        score -= 15

        add_reason(
            limiting,
            "NORMAL_AHI",
            (
                "AHI 未達常用異常範圍，"
                "缺少 APAP 支持。"
            ),
            -15,
        )

    if rem_relevance == "HIGH":
        score += 15

        add_reason(
            supporting,
            "HIGH_REM_VARIABILITY",
            (
                "呼吸事件具有高度 REM 相關性，"
                "支持壓力需求隨睡眠階段改變。"
            ),
            15,
        )

    elif rem_relevance == "MODERATE":
        score += 8

        add_reason(
            supporting,
            "MODERATE_REM_VARIABILITY",
            (
                "呼吸事件具有中度 REM 相關性，"
                "支持自動調壓候選。"
            ),
            8,
        )

    if position_relevance == "HIGH":
        score += 15

        add_reason(
            supporting,
            "HIGH_POSITION_VARIABILITY",
            (
                "不同姿勢群組間呼吸事件率"
                "具有高度差異，"
                "支持整夜壓力需求可能改變。"
            ),
            15,
        )

    elif position_relevance == "MODERATE":
        score += 8

        add_reason(
            supporting,
            "MODERATE_POSITION_VARIABILITY",
            (
                "不同姿勢群組間呼吸事件率"
                "具有中度差異。"
            ),
            8,
        )

    elif position_relevance == "MILD":
        score += 3

        add_reason(
            supporting,
            "MILD_POSITION_VARIABILITY",
            (
                "存在輕度姿勢相關變異。"
            ),
            3,
        )

    if (
        position_event_ratio is not None
        and position_event_ratio >= 2.0
    ):
        score += 5

        add_reason(
            supporting,
            "POSITION_EVENT_RATIO",
            (
                "最高與最低姿勢群組事件率"
                f"相差約 {position_event_ratio:.2f} 倍。"
            ),
            5,
        )

    if oxygen_burden == "HIGH_BURDEN":
        score += 10

        add_reason(
            supporting,
            "HIGH_OXYGEN_BURDEN",
            (
                "夜間血氧負荷偏高，"
                "支持呼吸支持治療候選。"
            ),
            10,
        )

    elif oxygen_burden == "MODERATE_BURDEN":
        score += 5

        add_reason(
            supporting,
            "MODERATE_OXYGEN_BURDEN",
            (
                "存在中度夜間低氧負荷，"
                "支持呼吸支持治療候選。"
            ),
            5,
        )

    if (
        coupled_3pct_fraction is not None
        and coupled_3pct_fraction >= 0.50
    ):
        score += 10

        add_reason(
            supporting,
            "EVENT_DESATURATION_COUPLING",
            (
                "至少一半可分析呼吸事件伴隨 ≥3% 血氧下降，"
                "支持事件具有生理影響。"
            ),
            10,
        )

    if (
        low_oxygen_away_fraction is not None
        and low_oxygen_away_fraction >= 0.70
    ):
        score -= 5

        add_reason(
            limiting,
            "NON_EVENT_HYPOXEMIA",
            (
                "大部分低氧時間遠離呼吸事件，"
                "APAP 未必能處理全部低氧。"
            ),
            -5,
        )

    if (
        central_count is not None
        and central_count > 0
    ):
        score -= 15

        add_reason(
            limiting,
            "CENTRAL_APNEA_PRESENT",
            (
                "存在中央型事件，"
                "自動調壓 PAP 需更審慎。"
            ),
            -15,
        )

    if loop_gain_level == "HIGH":
        score -= 15

        add_reason(
            limiting,
            "HIGH_LOOP_GAIN_PROXY",
            (
                "呼吸控制不穩定 proxy 偏高，"
                "自動壓力變化需更審慎。"
            ),
            -15,
        )

    evidence = calculate_evidence_completeness(
        {
            "AHI 與嚴重度": (
                context["ahi"] is not None
                and severity != "UNKNOWN"
            ),
            "REM 相關性": (
                rem_relevance
                != "UNKNOWN"
            ),
            "姿勢相關性": (
                position_relevance
                != "UNKNOWN"
            ),
            "中央型事件": (
                central_count
                is not None
            ),
            "血氧負荷": (
                oxygen_burden
                != "UNKNOWN"
            ),
            "Loop Gain proxy": (
                loop_gain_level
                != "UNKNOWN"
            ),
            "APAP 壓力範圍": False,
            "PAP 耐受資料": False,
        }
    )

    safety_hold = bool(
        (
            context["wake_spo2"]
            is not None
            and context["wake_spo2"]
            < 94.0
        )
        or (
            low_oxygen_away_fraction
            is not None
            and low_oxygen_away_fraction
            >= 0.70
        )
    )

    score = clamp_score(
        score
    )

    projected_score_changes = (
        build_projected_score_changes(
            "APAP",
            evidence,
        )
    )

    return {
        "treatment": "APAP",
        "treatment_label": (
            "自動調壓正壓呼吸器"
        ),
        "score": score,
        "suitability_level": (
            score_to_level(score)
        ),
        "recommendation_status": (
            recommendation_status(
                score=score,
                confidence_level=(
                    evidence[
                        "confidence_level"
                    ]
                ),
                safety_hold=(
                    safety_hold
                ),
            )
        ),
        "confidence": evidence,
        "supporting_factors": supporting,
        "limiting_factors": limiting,
        "interpretation": (
            "此分數只依目前已觀察到的患者資料"
            "評估自動調壓 PAP 適配度。"
        ),
        "projected_score_changes": (
            projected_score_changes
        ),
    }


# ============================================================
# 手術
# 缺少解剖資料不扣治療分
# 只降低 confidence
# ============================================================

def evaluate_surgery(
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    評估患者是否值得進一步接受上呼吸道外科檢查。

    重要限制：
    - PSG 可以呈現阻塞型態、REM 與姿勢變化。
    - PSG 無法直接確認實際阻塞位置。
    - 此結果只能表示「是否值得進一步外科評估」，
      不能直接指定鼻中膈、下鼻甲、軟顎或懸壅垂手術。
    """
    score = 10.0

    supporting: list[dict[str, Any]] = []
    limiting: list[dict[str, Any]] = []

    severity = context["ahi_severity"]
    obstructive_fraction = context[
        "obstructive_event_fraction"
    ]
    obstructive_count = context[
        "obstructive_apnea_count"
    ]
    central_fraction = context[
        "central_event_fraction"
    ]
    bmi = context["BMI"]
    rem_relevance = context[
        "rem_relevance"
    ]
    position_relevance = context[
        "position_relevance"
    ]
    hypopnea_mechanism_classified = context[
        "hypopnea_mechanism_classified"
    ]

    # --------------------------------------------------------
    # AHI 只表示治療需求，不表示已證明適合開刀
    # --------------------------------------------------------

    if severity == "SEVERE":
        score += 10

        add_reason(
            supporting,
            "SEVERE_OSA_REQUIRES_MULTIMODAL_REVIEW",
            (
                "AHI 位於重度範圍，患者可能需要多模式治療；"
                "若存在可矯正的解剖性阻塞，可進一步接受外科評估，"
                "但重度 AHI 本身不能證明手術適應性。"
            ),
            10,
        )

    elif severity == "MODERATE":
        score += 10

        add_reason(
            supporting,
            "MODERATE_OSA_SURGICAL_REVIEW_POSSIBLE",
            (
                "AHI 位於中度範圍；若內視鏡或影像確認"
                "局部上呼吸道阻塞，可進一步評估手術。"
            ),
            10,
        )

    elif severity == "MILD":
        score += 5

        add_reason(
            supporting,
            "MILD_OSA_WITH_POSSIBLE_ANATOMICAL_TARGET",
            (
                "AHI 位於輕度範圍；若症狀明顯且存在清楚的"
                "鼻腔或上呼吸道解剖性狹窄，可考慮外科評估。"
            ),
            5,
        )

    elif severity == "NORMAL":
        score -= 15

        add_reason(
            limiting,
            "NORMAL_AHI",
            (
                "AHI 未達常用異常範圍，"
                "目前 PSG 缺少以 OSA 為目的之外科支持。"
            ),
            -15,
        )

    # --------------------------------------------------------
    # 阻塞型態
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 呼吸事件機轉
    #
    # 重要原則：
    # - obstructive apnea 數量不能代表全部阻塞型呼吸負擔。
    # - 未分類的 hypopnea 可能仍由上呼吸道阻塞造成。
    # - 若 hypopnea 沒有 obstructive／central 分型，
    #   不因資訊不足而扣除手術分數，只降低信心。
    # --------------------------------------------------------

    hypopnea_count = context.get(
        "hypopnea_count"
    )

    central_count = context.get(
        "central_apnea_count"
    )

    hypopnea_mechanism_unclassified = (
        hypopnea_count is not None
        and hypopnea_count > 0
        and hypopnea_mechanism_classified is not True
    )

    # 已明確觀察到阻塞型 apnea 時，可提供有限支持。
    # 但不能只用 apnea 數量推估所有 hypopnea 的機轉。
    if (
        obstructive_count is not None
        and obstructive_count >= 5
    ):
        score += 8

        add_reason(
            supporting,
            "CONFIRMED_OBSTRUCTIVE_APNEAS",
            (
                "已觀察到多次明確阻塞型 apnea，"
                "支持存在上呼吸道阻塞機轉；"
                "但實際手術適應性仍須確認阻塞位置。"
            ),
            8,
        )

    elif (
        obstructive_count is not None
        and obstructive_count > 0
    ):
        score += 5

        add_reason(
            supporting,
            "LIMITED_CONFIRMED_OBSTRUCTIVE_APNEAS",
            (
                "已觀察到少量明確阻塞型 apnea，"
                "可提供有限的上呼吸道阻塞支持；"
                "但其餘 hypopnea 的機轉尚未分類。"
            ),
            5,
        )

    # hypopnea 未分類時，不以低 obstructive fraction 扣分。
    if hypopnea_mechanism_unclassified:
        add_reason(
            limiting,
            "HYPOPNEA_MECHANISM_UNCLASSIFIED",
            (
                f"目前有 {int(hypopnea_count)} 次 hypopnea，"
                "但尚未進一步分類為阻塞型、中央型或混合型。"
                "因此不能由明確 obstructive apnea 數量推論"
                "整體阻塞型事件負擔偏低；此項資訊不足只降低"
                "手術推薦信心，不直接扣除分數。"
            ),
            0,
        )

    # 中央型比例明顯偏高時，才構成真正的手術限制。
    if (
        central_fraction is not None
        and central_fraction >= 0.20
    ):
        score -= 20

        add_reason(
            limiting,
            "CENTRAL_EVENT_BURDEN",
            (
                "中央型事件比例偏高；"
                "上呼吸道擴張手術無法直接處理"
                "中央型呼吸控制問題。"
            ),
            -20,
        )



    # 只有在 hypopnea 已經具有可靠機轉分類時，
    # 才使用整體 obstructive_event_fraction 計分。
    if (
        hypopnea_mechanism_classified is True
        and obstructive_fraction is not None
    ):
        if obstructive_fraction >= 0.60:
            score += 12

            add_reason(
                supporting,
                "CLASSIFIED_OBSTRUCTIVE_PREDOMINANCE",
                (
                    "在呼吸事件機轉已完成分類的前提下，"
                    "整體事件以阻塞型為主，支持進一步"
                    "尋找可處理的上呼吸道阻塞位置。"
                ),
                12,
            )

        elif obstructive_fraction < 0.30:
            score -= 10

            add_reason(
                limiting,
                "CLASSIFIED_LOW_OBSTRUCTIVE_FRACTION",
                (
                    "在呼吸事件機轉已完成分類的前提下，"
                    "整體阻塞型比例偏低，降低以解剖性"
                    "上呼吸道手術處理全部事件的支持。"
                ),
                -10,
            )

    if bmi is not None:

        if bmi >= 35:
            score -= 12

            add_reason(
                limiting,
                "HIGH_BMI",
                (
                    "BMI 偏高可能代表多因素或多層次氣道塌陷，"
                    "單一鼻腔或軟組織手術的效果較不確定。"
                ),
                -12,
            )

    # --------------------------------------------------------
    # REM 與姿勢代表睡眠中的動態呼吸型態
    # 不直接否定手術，但降低單一術式的確定性
    # --------------------------------------------------------

    if rem_relevance == "HIGH":
        score -= 5

        add_reason(
            limiting,
            "HIGH_REM_DEPENDENCE",
            (
                "呼吸事件具有高度 REM 相關性。REM 睡眠期間"
                "上呼吸道肌張力下降，可能存在動態或多層塌陷；"
                "固定解剖性手術的效果較難單憑 PSG 預測。"
            ),
            -5,
        )

    elif rem_relevance == "MODERATE":
        score -= 2

        add_reason(
            limiting,
            "MODERATE_REM_DEPENDENCE",
            (
                "呼吸事件具有中度 REM 相關性，"
                "手術效果需結合睡眠期間動態氣道評估。"
            ),
            -2,
        )

    if position_relevance == "HIGH":
        score -= 4

        add_reason(
            limiting,
            "HIGH_POSITION_DEPENDENCE",
            (
                "呼吸事件隨姿勢高度變化，提示氣道阻塞可能具有"
                "動態特性；單一固定部位手術未必能處理全部事件。"
            ),
            -4,
        )

    # --------------------------------------------------------
    # 外科證據完整性
    # --------------------------------------------------------

    evidence = calculate_evidence_completeness(
        {
            "AHI 與嚴重度": (
                context["ahi"] is not None
                and severity != "UNKNOWN"
            ),
            "阻塞事件比例": (
                obstructive_fraction is not None
            ),
            "中央事件比例": (
                central_fraction is not None
            ),
            "BMI": bmi is not None,
            "REM 呼吸型態": (
                rem_relevance != "UNKNOWN"
            ),
            "姿勢呼吸型態": (
                position_relevance != "UNKNOWN"
            ),
            "耳鼻喉內視鏡": False,
            "鼻腔與上呼吸道影像": False,
            "睡眠內視鏡或動態阻塞評估": False,
            "PAP 耐受與治療反應": False,
        }
    )

    score = clamp_score(score)

    projected_score_changes = (
        build_projected_score_changes(
            "SURGERY_EVALUATION",
            evidence,
        )
    )

    return {
    "treatment": "SURGERY",
    "treatment_label": (
        "上呼吸道手術"
    ),
    "score": score,
    "suitability_level": score_to_level(
        score
    ),
    "recommendation_status": (
        recommendation_status(
            score=score,
            confidence_level=(
                evidence["confidence_level"]
            ),
            # 在缺少真正解剖檢查時，
            # 手術推薦仍須保留條件限制。
            safety_hold=True,
        )
    ),
    "confidence": evidence,
    "supporting_factors": supporting,
    "limiting_factors": limiting,
    "candidate_procedures": [
        "鼻中膈矯正",
        "下鼻甲手術",
        "軟顎或懸壅垂修整",
        "其他依阻塞位置決定之上呼吸道手術",
    ],
    "required_confirmation": [
        "耳鼻喉內視鏡檢查",
        "鼻腔與上呼吸道影像評估",
        "必要時睡眠內視鏡或其他動態阻塞評估",
        "確認阻塞位置與預定術式相符",
        "確認 PAP 治療反應、耐受性與患者偏好",
    ],
    "interpretation": (
        "此分數表示患者目前接受上呼吸道手術作為治療方式的"
        "適合度。PSG 可提供阻塞型態、REM 與姿勢相關等資訊，"
        "但無法單獨確認阻塞位置，因此仍需結合耳鼻喉檢查、"
        "影像或睡眠內視鏡等資料，才能決定是否實際接受手術"
        "及選擇術式。補充上述資料後，系統可重新計算手術"
        "適合度與整體治療排名。"
    ),
    "projected_score_changes": (
        projected_score_changes
    ),
}


# ============================================================
# 藥物／其他機轉治療
# 只根據已觀察到的 phenotype 計分
# 缺 CO₂、共病或用藥資料不加分
# ============================================================

def evaluate_medication_or_other(
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    評估睡眠結構調節型藥物是否值得進一步專科評估。

    此類藥物的假設用途：
    - 透過改變睡眠結構，降低部分睡眠階段中的呼吸事件。
    - 可能降低深度睡眠或影響主觀睡眠舒適度。
    - 僅適用特定 phenotype。
    - 不能根治 OSA，也不能自動取代 PAP。
    """
    score = 0.0

    supporting: list[dict[str, Any]] = []
    limiting: list[dict[str, Any]] = []

    severity = context[
        "ahi_severity"
    ]
    rem_relevance = context[
        "rem_relevance"
    ]
    position_relevance = context[
        "position_relevance"
    ]
    oxygen_burden = context[
        "oxygen_burden"
    ]
    coupling_level = context[
        "oxygen_coupling_level"
    ]
    coupled_3pct_fraction = context[
        "coupled_3pct_fraction"
    ]
    central_fraction = context[
        "central_event_fraction"
    ]
    low_oxygen_away_fraction = context[
        "low_oxygen_away_fraction"
    ]
    wake_spo2 = context[
        "wake_spo2"
    ]

    # --------------------------------------------------------
    # 輕度與中度較適合作為條件式候選
    # 重度患者不應依靠此類藥物作為單獨主要治療
    # --------------------------------------------------------

    if severity == "MILD":
        add_reason(
            limiting,
            "MILD_OSA_MEDICATION_PATH_NOT_ESTABLISHED",
            (
                "輕度 OSA 本身不構成藥物加分；仍需明確的目標"
                "表型、共病適應性及 PAP 不耐受或拒用證據。"
            ),
            0,
        )

    elif severity == "MODERATE":
        add_reason(
            limiting,
            "MODERATE_OSA_MEDICATION_PATH_NOT_ESTABLISHED",
            (
                "中度 OSA 本身不構成藥物加分；需由後續臨床"
                "資料確認目標表型、共病適應性及 PAP 替代需求。"
            ),
            0,
        )

    elif severity == "SEVERE":
        score -= 15

        add_reason(
            limiting,
            "SEVERE_OSA_NOT_STANDALONE",
            (
                "AHI 位於重度範圍；睡眠結構調節型藥物不應被視為"
                "可單獨控制疾病或取代主要呼吸支持治療。"
            ),
            -15,
        )

    elif severity == "NORMAL":
        score -= 10

        add_reason(
            limiting,
            "NORMAL_AHI",
            (
                "AHI 未達常用異常範圍，"
                "目前缺少以降低睡眠呼吸中止事件為目的之支持。"
            ),
            -10,
        )

    # --------------------------------------------------------
    # 睡眠階段依賴性是核心 phenotype
    # --------------------------------------------------------

    if rem_relevance == "HIGH":
        score += 20

        add_reason(
            supporting,
            "HIGH_SLEEP_STAGE_DEPENDENCE",
            (
                "呼吸事件具有高度 REM 相關性，表示事件頻率可能"
                "明顯受到睡眠結構與睡眠階段影響；符合進一步評估"
                "睡眠結構調節型藥物的機轉假設。"
            ),
            20,
        )

    elif rem_relevance == "MODERATE":
        score += 10

        add_reason(
            supporting,
            "MODERATE_SLEEP_STAGE_DEPENDENCE",
            (
                "呼吸事件具有中度睡眠階段相關性，"
                "可作為條件式機轉評估依據。"
            ),
            10,
        )

    elif rem_relevance == "LOW":
        score -= 8

        add_reason(
            limiting,
            "LOW_SLEEP_STAGE_DEPENDENCE",
            (
                "呼吸事件缺少明顯睡眠階段相關性，"
                "較不支持透過改變睡眠結構降低事件的假設。"
            ),
            -8,
        )

    # 姿勢差異主要不是這類藥物的直接作用目標
    if position_relevance == "HIGH":
        score -= 5

        add_reason(
            limiting,
            "POSITION_DOMINANT_PATTERN",
            (
                "呼吸事件主要隨姿勢高度變化，"
                "較偏向姿勢或壓力需求問題，而非單純睡眠結構問題。"
            ),
            -5,
        )

    # --------------------------------------------------------
    # 高低氧風險限制
    # --------------------------------------------------------

    if oxygen_burden == "HIGH_BURDEN":
        score -= 15

        add_reason(
            limiting,
            "HIGH_HYPOXEMIC_BURDEN",
            (
                "夜間低氧負荷偏高；此類藥物即使減少部分事件，"
                "仍不能假設足以控制整體低氧風險。"
            ),
            -15,
        )

    elif oxygen_burden == "MODERATE_BURDEN":
        score -= 5

        add_reason(
            limiting,
            "MODERATE_HYPOXEMIC_BURDEN",
            (
                "存在中度夜間低氧負荷，"
                "若評估此類藥物仍須監測是否充分改善血氧。"
            ),
            -5,
        )

    if (
        coupled_3pct_fraction is not None
        and coupled_3pct_fraction >= 0.50
    ):
        score -= 5

        add_reason(
            limiting,
            "PHYSIOLOGICALLY_SIGNIFICANT_EVENTS",
            (
                "至少一半可分析事件伴隨 ≥3% 血氧下降，"
                "表示事件具有明顯生理影響；不宜僅依睡眠結構調節"
                "取代較直接的呼吸治療。"
            ),
            -5,
        )

    if (
        central_fraction is not None
        and central_fraction >= 0.20
    ):
        score -= 10

        add_reason(
            limiting,
            "CENTRAL_EVENT_BURDEN",
            (
                "中央型事件比例偏高；目前無法由 PSG 證明"
                "此睡眠結構調節型藥物可改善中央呼吸控制問題。"
            ),
            -10,
        )

    # 非事件低氧不是此類藥物的適應性支持
    if (
        low_oxygen_away_fraction is not None
        and low_oxygen_away_fraction >= 0.70
    ):
        score -= 15

        add_reason(
            limiting,
            "NON_EVENT_HYPOXEMIA",
            (
                "大部分低氧時間遠離呼吸事件，"
                "提示可能存在心肺疾病、低通氣、基線低氧或量測問題；"
                "不支持以改變睡眠結構作為主要處理方式。"
            ),
            -15,
        )

    if (
        wake_spo2 is not None
        and wake_spo2 < 94.0
    ):
        score -= 10

        add_reason(
            limiting,
            "LOW_WAKE_SPO2",
            (
                f"PSG 清醒期平均 SpO₂ 約為 {wake_spo2:.2f}%；"
                "應先釐清基線心肺與低通氣問題，"
                "不能將其視為此類藥物的支持條件。"
            ),
            -10,
        )

    evidence = calculate_evidence_completeness(
        {
            "AHI 與嚴重度": (
                context["ahi"] is not None
                and severity != "UNKNOWN"
            ),
            "睡眠階段相關性": (
                rem_relevance != "UNKNOWN"
            ),
            "姿勢相關性": (
                position_relevance != "UNKNOWN"
            ),
            "血氧負荷": (
                oxygen_burden != "UNKNOWN"
            ),
            "事件－低氧耦合": (
                coupling_level != "UNKNOWN"
            ),
            "中央型事件比例": (
                central_fraction is not None
            ),
            "完整藥物名稱與作用機轉": False,
            "睡眠品質與白天症狀": False,
            "完整共病與目前用藥": False,
            "肝腎功能與藥物禁忌症": False,
            "治療後 PSG 或療效監測": False,
        }
    )

    # 此治療永遠只能是條件式專科評估。
    # 即使 score 高，也不可在本模型中變成確定用藥建議。
    score = min(
        clamp_score(score),
        60.0,
    )

    projected_score_changes = (
        build_projected_score_changes(
            "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW",
            evidence,
        )
    )

    return {
        "treatment": (
            "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW"
        ),
        "treatment_label": (
            "睡眠結構調節型藥物評估"
        ),
        "score": score,
        "suitability_level": score_to_level(
            score
        ),
        "recommendation_status": "CONDITIONAL",
        "confidence": evidence,
        "supporting_factors": supporting,
        "limiting_factors": limiting,
        "mechanism": (
            "透過改變睡眠結構，降低部分睡眠階段中的"
            "睡眠呼吸中止事件。"
        ),
        "expected_role": (
            "僅限特定 phenotype 的輔助或替代評估，"
            "不能視為根治，也不能自動取代 PAP。"
        ),
        "known_tradeoffs": [
            "可能降低深度睡眠",
            "可能影響主觀睡眠品質或呼吸舒適度",
            "可能只減少部分事件而非完全消除",
            "療效需要治療後睡眠檢查確認",
        ],
        "required_confirmation": [
            "確認實際藥物名稱與作用機轉",
            "確認睡眠階段相關呼吸事件型態",
            "確認白天嗜睡、睡眠品質與患者可接受性",
            "確認共病、目前用藥、肝腎功能與禁忌症",
            "治療後以 PSG 或適當監測確認效果",
        ],
        "interpretation": (
            "此分數只表示患者目前 PSG 是否呈現可能與睡眠結構"
            "相關的呼吸事件 phenotype。此類藥物可能藉由改變睡眠"
            "結構減少部分事件，但可能犧牲深度睡眠或睡眠舒適度，"
            "僅適用特定患者，不能根治睡眠呼吸中止症，亦不能由本"
            "模型指定藥名、劑量或直接開立治療。"
        ),
        "projected_score_changes": (
            projected_score_changes
        ),
    }

# ============================================================
# 治療前置事項
# 不參與治療排名
# ============================================================

def build_clinical_prerequisites(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    prerequisites: list[
        dict[str, Any]
    ] = []

    low_oxygen_away_fraction = context[
        "low_oxygen_away_fraction"
    ]

    low_oxygen_away_minutes = context[
        "low_oxygen_away_minutes"
    ]

    wake_spo2 = context[
        "wake_spo2"
    ]

    if (
        low_oxygen_away_fraction is not None
        and low_oxygen_away_fraction >= 0.70
    ):
        prerequisites.append(
            {
                "priority": 1,
                "code": (
                    "HYPOXEMIA_SOURCE_EVALUATION"
                ),
                "title": (
                    "釐清持續性低氧來源"
                ),
                "reason": (
                    "大部分 SpO₂ <90% 的時間"
                    "遠離已標記呼吸事件。"
                ),
                "observed_value": (
                    low_oxygen_away_fraction
                ),
                "required_information": [
                    "門診或白天靜息 SpO₂",
                    "心肺病史",
                    "完整用藥資料",
                    "必要時 CO₂ 或血氣",
                ],
            }
        )

    if (
        low_oxygen_away_minutes is not None
        and low_oxygen_away_minutes >= 60.0
    ):
        prerequisites.append(
            {
                "priority": 2,
                "code": (
                    "PROLONGED_HYPOXEMIA_REVIEW"
                ),
                "title": (
                    "確認長時間非事件相關低氧"
                ),
                "reason": (
                    "遠離呼吸事件的 SpO₂ <90% "
                    f"時間約 {low_oxygen_away_minutes:.1f} 分鐘。"
                ),
                "observed_value": (
                    low_oxygen_away_minutes
                ),
                "required_information": [
                    "血氧感測器品質確認",
                    "肺部評估",
                    "心血管評估",
                    "低通氣評估",
                ],
            }
        )

    if (
        wake_spo2 is not None
        and wake_spo2 < 94.0
    ):
        prerequisites.append(
            {
                "priority": 3,
                "code": (
                    "LOW_WAKE_SPO2_REVIEW"
                ),
                "title": (
                    "確認清醒期基線血氧"
                ),
                "reason": (
                    "PSG 清醒期平均 SpO₂"
                    f"約為 {wake_spo2:.2f}%。"
                ),
                "observed_value": (
                    wake_spo2
                ),
                "required_information": [
                    "白天靜息 SpO₂",
                    "肺功能或相關心肺評估",
                ],
            }
        )

    prerequisites.append(
        {
            "priority": 4,
            "code": (
                "PAP_TITRATION_AND_TOLERANCE"
            ),
            "title": (
                "完成 PAP 壓力與耐受性評估"
            ),
            "reason": (
                "目前沒有壓力滴定、漏氣、"
                "殘餘 AHI 與實際耐受資料。"
            ),
            "observed_value": None,
            "required_information": [
                "壓力設定",
                "面罩漏氣",
                "殘餘 AHI",
                "治療後血氧",
                "患者耐受度",
            ],
        }
    )

    prerequisites = sorted(
        prerequisites,
        key=lambda item: int(
            item["priority"]
        ),
    )

    return prerequisites


# ============================================================
# 個人化推薦摘要
# ============================================================

def build_recommendation_summary(
    treatments: list[
        dict[str, Any]
    ],
    prerequisites: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    top_treatment = (
        treatments[0]
        if treatments
        else None
    )

    second_treatment = (
        treatments[1]
        if len(treatments) >= 2
        else None
    )

    top_score = (
        safe_float(
            top_treatment.get(
                "score"
            )
        )
        if top_treatment
        else None
    )

    second_score = (
        safe_float(
            second_treatment.get(
                "score"
            )
        )
        if second_treatment
        else None
    )

    score_gap = None

    if (
        top_score is not None
        and second_score is not None
    ):
        score_gap = (
            top_score
            - second_score
        )

    unique_first_choice = bool(
        top_treatment is not None
        and top_score is not None
        and top_score >= 50
        and (
            score_gap is None
            or score_gap >= 10
        )
    )

    highest_prerequisite = (
        prerequisites[0]
        if prerequisites
        else None
    )

    return {
        "recommended_first_candidate": (
            top_treatment[
                "treatment"
            ]
            if top_treatment
            else None
        ),
        "recommended_first_candidate_label": (
            top_treatment[
                "treatment_label"
            ]
            if top_treatment
            else None
        ),
        "first_candidate_score": (
            top_score
        ),
        "first_candidate_confidence": (
            top_treatment[
                "confidence"
            ][
                "confidence_level"
            ]
            if top_treatment
            else None
        ),
        "first_candidate_status": (
            top_treatment[
                "recommendation_status"
            ]
            if top_treatment
            else None
        ),
        "second_candidate": (
            second_treatment[
                "treatment"
            ]
            if second_treatment
            else None
        ),
        "score_gap_to_second": (
            score_gap
        ),
        "unique_first_choice": (
            unique_first_choice
        ),
        "highest_priority_prerequisite": (
            highest_prerequisite
        ),
        "interpretation": (
            "第一候選表示目前患者已觀察 phenotype "
            "最符合的治療方向。前置事項不參與排名，"
            "但可能使推薦狀態維持 CONDITIONAL。"
        ),
    }


# ============================================================
# 顯示
# ============================================================

def print_core_data(
    patient_id: str,
    context: dict[str, Any],
) -> None:
    print("=" * 80)
    print("Internal Treatment Pre-screen")
    print("此步驟僅供最終治療排序引擎使用，不是網頁或報告的正式分數。")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    print()
    print("目前已觀察到的核心資料：")

    print(
        f"AHI：{context['ahi']}"
    )

    print(
        "AHI Severity："
        f"{context['ahi_severity']}"
    )

    print(
        "Obstructive Apnea："
        f"{context['obstructive_apnea_count']}"
    )

    print(
        "Hypopnea："
        f"{context['hypopnea_count']}"
    )

    print(
        "Central Apnea："
        f"{context['central_apnea_count']}"
    )

    print(
        "總體血氧負荷："
        f"{context['oxygen_burden']}"
    )

    print(
        "REM 相關性："
        f"{context['rem_relevance']}"
    )

    print(
        "姿勢相關性："
        f"{context['position_relevance']}"
    )

    print(
        "姿勢事件率比值："
        f"{context['position_event_ratio']}"
    )

    print(
        "低氧－事件耦合："
        f"{context['oxygen_coupling_level']}"
    )

    print(
        "低氧遠離事件比例："
        f"{context['low_oxygen_away_fraction']}"
    )

    print(
        "低氧遠離事件時間："
        f"{context['low_oxygen_away_minutes']} 分鐘"
    )

    print(
        "PSG 清醒期平均 SpO₂："
        f"{context['wake_spo2']}"
    )

    print(
        "睡眠期平均 SpO₂："
        f"{context['sleep_spo2']}"
    )


def print_treatment_ranking(
    treatments: list[
        dict[str, Any]
    ],
) -> None:
    print()
    print("=" * 80)
    print("個人化治療推薦排名")
    print("只依目前已觀察資料計分")
    print("=" * 80)

    for treatment in treatments:
        confidence = treatment[
            "confidence"
        ]

        print(
            f"{treatment['rank']}. "
            f"{treatment['treatment']}："
            f"{treatment['score']:.1f} "
            f"({treatment['suitability_level']})"
        )

        print(
            f"   說明："
            f"{treatment['treatment_label']}"
        )

        print(
            "   推薦狀態："
            f"{treatment['recommendation_status']}"
        )

        print(
            "   推薦信心："
            f"{confidence['confidence_level']} "
            f"({confidence['available_count']}/"
            f"{confidence['total_count']})"
        )

        if treatment[
            "supporting_factors"
        ]:
            print("   支持因素：")

            for reason in treatment[
                "supporting_factors"
            ]:
                print(
                    f"   + "
                    f"{reason['description']} "
                    f"(weight={reason['weight']:+.1f})"
                )

        if treatment[
            "limiting_factors"
        ]:
            print("   限制因素：")

            for reason in treatment[
                "limiting_factors"
            ]:
                print(
                    f"   - "
                    f"{reason['description']} "
                    f"(weight={reason['weight']:+.1f})"
                )

        if confidence[
            "missing_evidence"
        ]:
            print(
                "   尚缺資料，僅降低信心、不影響分數："
            )

            for item in confidence[
                "missing_evidence"
            ]:
                print(
                    f"   * {item}"
                )

        if treatment.get(
            "projected_score_changes"
        ):
            print(
                "   補充資料後的可能分數變化："
            )

            for item in treatment.get(
                "projected_score_changes",
                [],
            ):
                print(
                    f"   * {item}"
                )

        print(
            f"   解讀："
            f"{treatment['interpretation']}"
        )

        print()


def print_prerequisites(
    prerequisites: list[
        dict[str, Any]
    ],
) -> None:
    print("=" * 80)
    print("治療前必要處理")
    print("不參與治療推薦分數")
    print("=" * 80)

    if not prerequisites:
        print(
            "目前沒有額外前置事項。"
        )
        return

    for item in prerequisites:
        print(
            f"{item['priority']}. "
            f"{item['title']}"
        )

        print(
            f"   原因：{item['reason']}"
        )

        print(
            "   需補充／確認："
        )

        for required_item in item[
            "required_information"
        ]:
            print(
                f"   * {required_item}"
            )

        print()


# ============================================================
# 主程式
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "依目前已觀察患者資料產生"
            "四種個人化治療推薦排名，"
            "並將資料缺失與前置事項獨立呈現。"
        )
    )

    parser.add_argument(
        "--patient-id",
        required=True,
        help=(
            "患者 ID，例如："
            "20201014T221256 - d25c6"
        ),
    )

    args = parser.parse_args()

    patient_id = str(
        args.patient_id
    ).strip()

    patient_folder = (
        INFERENCE_ROOT
        / patient_id
    )

    respiratory_profile_file = (
        patient_folder
        / "respiratory_profile"
        / "patient_respiratory_profile.json"
    )

    position_profile_file = (
        patient_folder
        / "position_profile"
        / "patient_position_profile.json"
    )

    oxygen_coupling_file = (
        patient_folder
        / "oxygen_event_coupling"
        / "oxygen_event_coupling_summary.json"
    )

    profile = load_json(
        respiratory_profile_file
    )

    position_analysis = load_json(
        position_profile_file
    )

    oxygen_coupling = load_json(
        oxygen_coupling_file
    )

    context = extract_patient_context(
        profile=profile,
        position_analysis=(
            position_analysis
        ),
        oxygen_coupling=(
            oxygen_coupling
        ),
    )
    clinical_data_file = (
        patient_folder
        / "treatment_refinement"
        / "clinical_decision_data.json"
    )
    clinical_data = (
        load_json(clinical_data_file)
        if clinical_data_file.is_file()
        else {}
    )
    context, clinical_feature_ingestion = merge_clinical_context(
        context,
        clinical_data,
    )

    treatments = [
        evaluate_cpap(
            context
        ),
        evaluate_apap(
            context
        ),
        evaluate_surgery(
            context
        ),
        evaluate_medication_or_other(
            context
        ),
    ]

    treatments, medical_decision_basis = apply_medical_guardrails(
        treatments,
        context,
    )

    treatment_tie_priority = {
        "APAP": 4,
        "CPAP": 3,
        "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW": 2,
        "SURGERY_EVALUATION": 1,
    }

    treatments = sorted(
        treatments,
        key=lambda item: (
            float(
                item["score"]
            ),
            treatment_tie_priority.get(
                str(
                    item["treatment"]
                ),
                0,
            ),
        ),
        reverse=True,
    )

    for rank, treatment in enumerate(
        treatments,
        start=1,
    ):
        treatment["rank"] = rank

    prerequisites = (
        build_clinical_prerequisites(
            context
        )
    )

    recommendation_summary = (
        build_recommendation_summary(
            treatments=treatments,
            prerequisites=prerequisites,
        )
    )

    output_folder = (
        patient_folder
        / "treatment_suitability"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_json = (
        output_folder
        / "personalized_treatment_recommendation.json"
    )

    ranking_csv = (
        output_folder
        / "personalized_treatment_ranking.csv"
    )

    prerequisites_csv = (
        output_folder
        / "clinical_prerequisites.csv"
    )

    result = {
        "patient_id": patient_id,
        "source_files": {
            "respiratory_profile": str(
                respiratory_profile_file
            ),
            "position_profile": str(
                position_profile_file
            ),
            "oxygen_event_coupling": str(
                oxygen_coupling_file
            ),
        },
        "scoring_principle": (
            "治療分數只依目前已觀察到的患者資料計算；"
            "資料缺失不加分、不扣分，只影響推薦信心。"
        ),
        "patient_context": (
            context
        ),
        "clinical_feature_ingestion": clinical_feature_ingestion,
        "personalized_treatment_ranking": (
            treatments
        ),
        "medical_decision_basis": medical_decision_basis,
        "clinical_prerequisites": (
            prerequisites
        ),
        "recommendation_summary": (
            recommendation_summary
        ),
        "safety_note": (
            "此結果為研究型個人化治療候選排序，"
            "不是處方、正式診斷或保證治療成功。"
        ),
    }

    save_json(
        output_json,
        result,
    )

    ranking_rows: list[
        dict[str, Any]
    ] = []

    for treatment in treatments:
        confidence = treatment[
            "confidence"
        ]

        ranking_rows.append(
            {
                "patient_id": patient_id,
                "rank": (
                    treatment["rank"]
                ),
                "treatment": (
                    treatment["treatment"]
                ),
                "treatment_label": (
                    treatment[
                        "treatment_label"
                    ]
                ),
                "score": (
                    treatment["score"]
                ),
                "suitability_level": (
                    treatment[
                        "suitability_level"
                    ]
                ),
                "recommendation_status": (
                    treatment[
                        "recommendation_status"
                    ]
                ),
                "confidence_level": (
                    confidence[
                        "confidence_level"
                    ]
                ),
                "evidence_completeness": (
                    confidence[
                        "completeness"
                    ]
                ),
                "available_evidence_count": (
                    confidence[
                        "available_count"
                    ]
                ),
                "total_evidence_count": (
                    confidence[
                        "total_count"
                    ]
                ),
                "missing_evidence": "；".join(
                    confidence[
                        "missing_evidence"
                    ]
                ),
                "supporting_factors": "；".join(
                    reason["description"]
                    for reason
                    in treatment[
                        "supporting_factors"
                    ]
                ),
                "limiting_factors": "；".join(
                    reason["description"]
                    for reason
                    in treatment[
                        "limiting_factors"
                    ]
                ),
                "interpretation": (
                    treatment[
                        "interpretation"
                    ]
                ),
            }
        )

    pd.DataFrame(
        ranking_rows
    ).to_csv(
        ranking_csv,
        index=False,
        encoding="utf-8-sig",
    )

    prerequisite_rows: list[
        dict[str, Any]
    ] = []

    for item in prerequisites:
        prerequisite_rows.append(
            {
                "patient_id": patient_id,
                "priority": (
                    item["priority"]
                ),
                "code": (
                    item["code"]
                ),
                "title": (
                    item["title"]
                ),
                "reason": (
                    item["reason"]
                ),
                "observed_value": (
                    item[
                        "observed_value"
                    ]
                ),
                "required_information": (
                    "；".join(
                        item[
                            "required_information"
                        ]
                    )
                ),
            }
        )

    pd.DataFrame(
        prerequisite_rows
    ).to_csv(
        prerequisites_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print_core_data(
        patient_id=patient_id,
        context=context,
    )

    print_treatment_ranking(
        treatments
    )

    print_prerequisites(
        prerequisites
    )

    print("=" * 80)
    print("個人化推薦摘要")
    print("=" * 80)

    recommended_first_candidate = (
        recommendation_summary.get(
            "recommended_first_candidate"
        )
    )

    recommended_first_candidate_label = (
        recommendation_summary.get(
            "recommended_first_candidate_label"
        )
    )

    first_candidate_score = (
        recommendation_summary.get(
            "first_candidate_score"
        )
    )   

    first_candidate_status = (
        recommendation_summary.get(
            "first_candidate_status"
        )
    )

    first_candidate_confidence = (
        recommendation_summary.get(
            "first_candidate_confidence"
        )
    )   

    second_candidate = (
        recommendation_summary.get(
            "second_candidate"
        )
    )

    score_gap_to_second = (
        recommendation_summary.get(
            "score_gap_to_second"
        )
    )

    unique_first_choice = (
        recommendation_summary.get(
            "unique_first_choice"
        )
    )

    highest_prerequisite = (
        recommendation_summary.get(
            "highest_priority_prerequisite"
        )   
    )

    print(
        "第一治療候選："
        f"{recommended_first_candidate}"
    )

    print(
        "第一候選說明："
        f"{recommended_first_candidate_label}"
    )

    print(
        "第一候選分數："
        f"{first_candidate_score}"
    )

    print(
        "第一候選推薦狀態："
        f"{first_candidate_status}"
    )

    print(
        "第一候選信心："
        f"{first_candidate_confidence}"
    )

    print(
        "第二治療候選："
        f"{second_candidate}"
    )

    print(
        "與第二名分數差距："
        f"{score_gap_to_second}"
    )

    print(
        "是否為明確唯一第一選擇："
        f"{unique_first_choice}"
    )

    if isinstance(
        highest_prerequisite,
        dict,
    ):
        highest_prerequisite_title = (
            highest_prerequisite.get(
                "title"
            )
        )

    print(
        "最高優先前置事項："
        f"{highest_prerequisite_title}"
    )

    print()

    print(
        f"完整結果 JSON：{output_json}"
    )

    print(
        f"治療排名 CSV：{ranking_csv}"
    )

    print(
        f"前置事項 CSV：{prerequisites_csv}"
    )

print()

print(
    "注意：治療分數不受資料缺失本身影響；"
    "缺少資料只降低推薦信心。"
)

print("=" * 80)


if __name__ == "__main__":
    main()
