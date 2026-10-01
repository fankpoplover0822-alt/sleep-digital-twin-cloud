from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.continual_learning.treatment import (
    apply_learned_treatment_adjustment,
    record_treatment_snapshot,
)
from clinical_treatment_guardrails import apply_medical_guardrails
from clinical_context_integration import merge_clinical_context


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


def utc_now_iso() -> str:
    """
    回傳目前 UTC 時間，使用 ISO 8601 格式。

    範例：
    2026-07-27T10:55:07+00:00
    """
    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )


def load_optional_json(
    file_path: Path,
) -> dict[str, Any] | None:
    """
    嘗試讀取既有 JSON。

    與 load_json() 的差異：
    - 檔案不存在時不拋出錯誤，回傳 None。
    - JSON 格式錯誤時拋出 RuntimeError。
    - JSON 最外層不是 object 時拋出 RuntimeError。

    此函式主要用於讀取既有的治療推薦結果，
    以便在 Follow-up 模式建立歷史快照。
    """
    if not file_path.exists():
        return None

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "既有治療推薦 JSON 格式錯誤："
            f"{file_path}\n"
            f"{error}"
        ) from error

    except OSError as error:
        raise RuntimeError(
            "無法讀取既有治療推薦 JSON："
            f"{file_path}\n"
            f"{error}"
        ) from error

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            "既有治療推薦 JSON 最外層"
            f"不是 object：{file_path}"
        )

    return data


def extract_existing_history(
    previous_result: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    從既有治療推薦 JSON 取得 recommendation_history。

    若舊版 JSON 尚未包含 recommendation_history，
    則回傳空清單。
    """
    if not isinstance(
        previous_result,
        dict,
    ):
        return []

    raw_history = previous_result.get(
        "recommendation_history",
        [],
    )

    if not isinstance(
        raw_history,
        list,
    ):
        return []

    history: list[dict[str, Any]] = []

    for item in raw_history:
        if isinstance(
            item,
            dict,
        ):
            history.append(
                safe_json_value(item)
            )

    return history


def build_recommendation_snapshot(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    將一次治療推薦結果轉換成歷史快照。

    歷史快照只保留比較 Follow-up 所需的重要內容，
    避免把整份 JSON 再完整巢狀複製一次。
    """
    refinement_run = result.get(
        "refinement_run",
        {},
    )

    if not isinstance(
        refinement_run,
        dict,
    ):
        refinement_run = {}

    recommendation_summary = result.get(
        "recommendation_summary",
        {},
    )

    if not isinstance(
        recommendation_summary,
        dict,
    ):
        recommendation_summary = {}

    ranking = result.get(
        "personalized_treatment_ranking",
        [],
    )

    if not isinstance(
        ranking,
        list,
    ):
        ranking = []

    ranking_snapshot: list[
        dict[str, Any]
    ] = []

    for treatment in ranking:
        if not isinstance(
            treatment,
            dict,
        ):
            continue

        confidence = treatment.get(
            "confidence",
            {},
        )

        if not isinstance(
            confidence,
            dict,
        ):
            confidence = {}

        ranking_snapshot.append(
            {
                "rank": safe_int(
                    treatment.get(
                        "rank"
                    )
                ),
                "treatment": treatment.get(
                    "treatment"
                ),
                "treatment_label": (
                    treatment.get(
                        "treatment_label"
                    )
                ),
                "score": safe_float(
                    treatment.get(
                        "score"
                    )
                ),
                "suitability_level": (
                    treatment.get(
                        "suitability_level"
                    )
                ),
                "recommendation_status": (
                    treatment.get(
                        "recommendation_status"
                    )
                ),
                "confidence_level": (
                    confidence.get(
                        "confidence_level"
                    )
                ),
                "evidence_completeness": (
                    safe_float(
                        confidence.get(
                            "completeness"
                        )
                    )
                ),
            }
        )

    return {
        "run_id": refinement_run.get(
            "run_id"
        ),
        "generated_at": (
            refinement_run.get(
                "generated_at"
            )
            or result.get(
                "generated_at"
            )
        ),
        "mode": (
            refinement_run.get(
                "mode"
            )
            or result.get(
                "mode"
            )
            or "INITIAL"
        ),
        "patient_id": result.get(
            "patient_id"
        ),
        "recommended_first_candidate": (
            recommendation_summary.get(
                "recommended_first_candidate"
            )
        ),
        "recommended_first_candidate_label": (
            recommendation_summary.get(
                "recommended_first_candidate_label"
            )
        ),
        "first_candidate_score": safe_float(
            recommendation_summary.get(
                "first_candidate_score"
            )
        ),
        "first_candidate_status": (
            recommendation_summary.get(
                "first_candidate_status"
            )
        ),
        "first_candidate_confidence": (
            recommendation_summary.get(
                "first_candidate_confidence"
            )
        ),
        "second_candidate": (
            recommendation_summary.get(
                "second_candidate"
            )
        ),
        "score_gap_to_second": safe_float(
            recommendation_summary.get(
                "score_gap_to_second"
            )
        ),
        "unique_first_choice": bool(
            recommendation_summary.get(
                "unique_first_choice",
                False,
            )
        ),
        "ranking": ranking_snapshot,
    }


def build_run_id(
    patient_id: str,
    generated_at: str,
    follow_up_mode: bool,
) -> str:
    """
    建立此次推薦執行的識別碼。

    不使用 patient_id 原始空白與特殊符號，
    讓 run_id 適合寫入 JSON、CSV 與檔名。
    """
    clean_patient_id = "".join(
        character
        if character.isalnum()
        or character in {
            "-",
            "_",
        }
        else "_"
        for character in patient_id
    )

    clean_timestamp = (
        generated_at
        .replace(
            ":",
            ""
        )
        .replace(
            "-",
            ""
        )
        .replace(
            "+",
            "_"
        )
    )

    mode_name = (
        "FOLLOW_UP"
        if follow_up_mode
        else "INITIAL"
    )

    return (
        f"{clean_patient_id}_"
        f"{mode_name}_"
        f"{clean_timestamp}"
    )

def normalize_treatment_ranking(
    result: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """
    將治療排名轉換成以 treatment code 為 key 的字典，
    方便比較前次與本次結果。

    支援主要輸出格式：
    personalized_treatment_ranking

    也支援歷史快照格式：
    ranking
    """
    if not isinstance(
        result,
        dict,
    ):
        return {}

    raw_ranking = result.get(
        "personalized_treatment_ranking"
    )

    if not isinstance(
        raw_ranking,
        list,
    ):
        raw_ranking = result.get(
            "ranking",
            [],
        )

    if not isinstance(
        raw_ranking,
        list,
    ):
        return {}

    normalized: dict[
        str,
        dict[str, Any]
    ] = {}

    for item in raw_ranking:
        if not isinstance(
            item,
            dict,
        ):
            continue

        treatment = item.get(
            "treatment"
        )

        if treatment is None:
            continue

        treatment_code = str(
            treatment
        ).strip()

        if not treatment_code:
            continue

        confidence = item.get(
            "confidence",
            {},
        )

        if not isinstance(
            confidence,
            dict,
        ):
            confidence = {}

        normalized[
            treatment_code
        ] = {
            "treatment": treatment_code,
            "treatment_label": (
                item.get(
                    "treatment_label"
                )
            ),
            "rank": safe_int(
                item.get(
                    "rank"
                )
            ),
            "score": safe_float(
                item.get(
                    "score"
                )
            ),
            "suitability_level": (
                item.get(
                    "suitability_level"
                )
            ),
            "recommendation_status": (
                item.get(
                    "recommendation_status"
                )
            ),
            "confidence_level": (
                confidence.get(
                    "confidence_level"
                )
                or item.get(
                    "confidence_level"
                )
            ),
            "evidence_completeness": (
                safe_float(
                    confidence.get(
                        "completeness"
                    )
                )
                if confidence
                else safe_float(
                    item.get(
                        "evidence_completeness"
                    )
                )
            ),
        }

    return normalized


def calculate_numeric_change(
    previous_value: float | int | None,
    current_value: float | int | None,
) -> float | int | None:
    """
    計算數值變化：

    current - previous

    任一值不存在時回傳 None。
    """
    if (
        previous_value is None
        or current_value is None
    ):
        return None

    difference = (
        current_value
        - previous_value
    )

    if isinstance(
        difference,
        float,
    ):
        return round(
            difference,
            6,
        )

    return difference


def classify_score_change(
    score_change: float | int | None,
) -> str:
    """
    將分數變化轉換成方向標記。
    """
    if score_change is None:
        return "NOT_COMPARABLE"

    if score_change > 0:
        return "INCREASED"

    if score_change < 0:
        return "DECREASED"

    return "UNCHANGED"


def classify_rank_change(
    previous_rank: int | None,
    current_rank: int | None,
) -> str:
    """
    判斷排名變化。

    排名數字越小代表排名越高：
    previous=3、current=2，表示 IMPROVED。
    """
    if (
        previous_rank is None
        and current_rank is not None
    ):
        return "NEWLY_RANKED"

    if (
        previous_rank is not None
        and current_rank is None
    ):
        return "NO_LONGER_RANKED"

    if (
        previous_rank is None
        or current_rank is None
    ):
        return "NOT_COMPARABLE"

    if current_rank < previous_rank:
        return "IMPROVED"

    if current_rank > previous_rank:
        return "DECLINED"

    return "UNCHANGED"


def get_top_candidate(
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    從一次推薦結果取得第一候選。

    優先使用 recommendation_summary；
    若不存在，則由排名中找 rank 最小者。
    """
    if not isinstance(
        result,
        dict,
    ):
        return {
            "treatment": None,
            "treatment_label": None,
            "score": None,
            "status": None,
            "confidence_level": None,
        }

    summary = result.get(
        "recommendation_summary",
        {},
    )

    if isinstance(
        summary,
        dict,
    ):
        treatment = summary.get(
            "recommended_first_candidate"
        )

        if treatment is not None:
            return {
                "treatment": treatment,
                "treatment_label": (
                    summary.get(
                        "recommended_first_candidate_label"
                    )
                ),
                "score": safe_float(
                    summary.get(
                        "first_candidate_score"
                    )
                ),
                "status": summary.get(
                    "first_candidate_status"
                ),
                "confidence_level": (
                    summary.get(
                        "first_candidate_confidence"
                    )
                ),
            }

    normalized_ranking = (
        normalize_treatment_ranking(
            result
        )
    )

    ranked_items = [
        item
        for item
        in normalized_ranking.values()
        if item.get(
            "rank"
        )
        is not None
    ]

    if not ranked_items:
        return {
            "treatment": None,
            "treatment_label": None,
            "score": None,
            "status": None,
            "confidence_level": None,
        }

    top_item = min(
        ranked_items,
        key=lambda item: item[
            "rank"
        ],
    )

    return {
        "treatment": top_item.get(
            "treatment"
        ),
        "treatment_label": (
            top_item.get(
                "treatment_label"
            )
        ),
        "score": top_item.get(
            "score"
        ),
        "status": top_item.get(
            "recommendation_status"
        ),
        "confidence_level": (
            top_item.get(
                "confidence_level"
            )
        ),
    }


def build_follow_up_comparison(
    previous_result: dict[str, Any] | None,
    current_treatments: list[dict[str, Any]],
    current_recommendation_summary: dict[str, Any],
    follow_up_mode: bool,
) -> dict[str, Any]:
    """
    比較前次與本次個人化治療推薦結果。
    """
    comparison_available = bool(
        follow_up_mode
        and isinstance(
            previous_result,
            dict,
        )
    )

    if not comparison_available:
        return {
            "comparison_available": False,
            "comparison_reason": (
                "NOT_FOLLOW_UP"
                if not follow_up_mode
                else "PREVIOUS_RESULT_NOT_AVAILABLE"
            ),
            "previous_run_id": None,
            "top_candidate_changed": None,
            "previous_top_candidate": None,
            "current_top_candidate": (
                current_recommendation_summary.get(
                    "recommended_first_candidate"
                )
            ),
            "score_changes": [],
            "rank_changes": [],
            "treatment_changes": [],
            "summary": {
                "score_increased_count": 0,
                "score_decreased_count": 0,
                "score_unchanged_count": 0,
                "rank_improved_count": 0,
                "rank_declined_count": 0,
                "rank_unchanged_count": 0,
                "newly_ranked_count": 0,
                "no_longer_ranked_count": 0,
            },
        }

    current_result_for_comparison = {
        "personalized_treatment_ranking": (
            current_treatments
        ),
        "recommendation_summary": (
            current_recommendation_summary
        ),
    }

    previous_ranking = (
        normalize_treatment_ranking(
            previous_result
        )
    )

    current_ranking = (
        normalize_treatment_ranking(
            current_result_for_comparison
        )
    )

    previous_top = get_top_candidate(
        previous_result
    )

    current_top = get_top_candidate(
        current_result_for_comparison
    )

    all_treatment_codes = sorted(
        set(
            previous_ranking.keys()
        )
        | set(
            current_ranking.keys()
        )
    )

    treatment_changes: list[
        dict[str, Any]
    ] = []

    score_changes: list[
        dict[str, Any]
    ] = []

    rank_changes: list[
        dict[str, Any]
    ] = []

    for treatment_code in all_treatment_codes:
        previous_item = (
            previous_ranking.get(
                treatment_code,
                {},
            )
        )

        current_item = (
            current_ranking.get(
                treatment_code,
                {},
            )
        )

        previous_score = safe_float(
            previous_item.get(
                "score"
            )
        )

        current_score = safe_float(
            current_item.get(
                "score"
            )
        )

        score_change = (
            calculate_numeric_change(
                previous_score,
                current_score,
            )
        )

        score_change_direction = (
            classify_score_change(
                score_change
            )
        )

        previous_rank = safe_int(
            previous_item.get(
                "rank"
            )
        )

        current_rank = safe_int(
            current_item.get(
                "rank"
            )
        )

        rank_change_direction = (
            classify_rank_change(
                previous_rank,
                current_rank,
            )
        )

        rank_position_change = None

        if (
            previous_rank is not None
            and current_rank is not None
        ):
            rank_position_change = (
                previous_rank
                - current_rank
            )

        treatment_label = (
            current_item.get(
                "treatment_label"
            )
            or previous_item.get(
                "treatment_label"
            )
        )

        previous_status = (
            previous_item.get(
                "recommendation_status"
            )
        )

        current_status = (
            current_item.get(
                "recommendation_status"
            )
        )

        previous_confidence = (
            previous_item.get(
                "confidence_level"
            )
        )

        current_confidence = (
            current_item.get(
                "confidence_level"
            )
        )

        change_item = {
            "treatment": treatment_code,
            "treatment_label": (
                treatment_label
            ),
            "previous_rank": (
                previous_rank
            ),
            "current_rank": (
                current_rank
            ),
            "rank_position_change": (
                rank_position_change
            ),
            "rank_change_direction": (
                rank_change_direction
            ),
            "previous_score": (
                previous_score
            ),
            "current_score": (
                current_score
            ),
            "score_change": (
                score_change
            ),
            "score_change_direction": (
                score_change_direction
            ),
            "previous_suitability_level": (
                previous_item.get(
                    "suitability_level"
                )
            ),
            "current_suitability_level": (
                current_item.get(
                    "suitability_level"
                )
            ),
            "previous_recommendation_status": (
                previous_status
            ),
            "current_recommendation_status": (
                current_status
            ),
            "recommendation_status_changed": (
                previous_status
                != current_status
            ),
            "previous_confidence_level": (
                previous_confidence
            ),
            "current_confidence_level": (
                current_confidence
            ),
            "confidence_level_changed": (
                previous_confidence
                != current_confidence
            ),
            "previous_evidence_completeness": (
                previous_item.get(
                    "evidence_completeness"
                )
            ),
            "current_evidence_completeness": (
                current_item.get(
                    "evidence_completeness"
                )
            ),
        }

        treatment_changes.append(
            change_item
        )

        score_changes.append(
            {
                "treatment": treatment_code,
                "treatment_label": (
                    treatment_label
                ),
                "previous_score": (
                    previous_score
                ),
                "current_score": (
                    current_score
                ),
                "score_change": (
                    score_change
                ),
                "direction": (
                    score_change_direction
                ),
            }
        )

        rank_changes.append(
            {
                "treatment": treatment_code,
                "treatment_label": (
                    treatment_label
                ),
                "previous_rank": (
                    previous_rank
                ),
                "current_rank": (
                    current_rank
                ),
                "rank_position_change": (
                    rank_position_change
                ),
                "direction": (
                    rank_change_direction
                ),
            }
        )

    previous_refinement_run = (
        previous_result.get(
            "refinement_run",
            {},
        )
    )

    if not isinstance(
        previous_refinement_run,
        dict,
    ):
        previous_refinement_run = {}

    summary = {
        "score_increased_count": sum(
            item[
                "score_change_direction"
            ]
            == "INCREASED"
            for item
            in treatment_changes
        ),
        "score_decreased_count": sum(
            item[
                "score_change_direction"
            ]
            == "DECREASED"
            for item
            in treatment_changes
        ),
        "score_unchanged_count": sum(
            item[
                "score_change_direction"
            ]
            == "UNCHANGED"
            for item
            in treatment_changes
        ),
        "rank_improved_count": sum(
            item[
                "rank_change_direction"
            ]
            == "IMPROVED"
            for item
            in treatment_changes
        ),
        "rank_declined_count": sum(
            item[
                "rank_change_direction"
            ]
            == "DECLINED"
            for item
            in treatment_changes
        ),
        "rank_unchanged_count": sum(
            item[
                "rank_change_direction"
            ]
            == "UNCHANGED"
            for item
            in treatment_changes
        ),
        "newly_ranked_count": sum(
            item[
                "rank_change_direction"
            ]
            == "NEWLY_RANKED"
            for item
            in treatment_changes
        ),
        "no_longer_ranked_count": sum(
            item[
                "rank_change_direction"
            ]
            == "NO_LONGER_RANKED"
            for item
            in treatment_changes
        ),
    }

    previous_top_treatment = (
        previous_top.get(
            "treatment"
        )
    )

    current_top_treatment = (
        current_top.get(
            "treatment"
        )
    )

    return {
        "comparison_available": True,
        "comparison_reason": None,
        "previous_run_id": (
            previous_refinement_run.get(
                "run_id"
            )
        ),
        "previous_generated_at": (
            previous_refinement_run.get(
                "generated_at"
            )
            or previous_result.get(
                "generated_at"
            )
        ),
        "top_candidate_changed": (
            previous_top_treatment
            != current_top_treatment
        ),
        "previous_top_candidate": (
            previous_top_treatment
        ),
        "previous_top_candidate_label": (
            previous_top.get(
                "treatment_label"
            )
        ),
        "previous_top_candidate_score": (
            previous_top.get(
                "score"
            )
        ),
        "current_top_candidate": (
            current_top_treatment
        ),
        "current_top_candidate_label": (
            current_top.get(
                "treatment_label"
            )
        ),
        "current_top_candidate_score": (
            current_top.get(
                "score"
            )
        ),
        "top_candidate_score_change": (
            calculate_numeric_change(
                safe_float(
                    previous_top.get(
                        "score"
                    )
                ),
                safe_float(
                    current_top.get(
                        "score"
                    )
                ),
            )
        ),
        "score_changes": score_changes,
        "rank_changes": rank_changes,
        "treatment_changes": (
            treatment_changes
        ),
        "summary": summary,
    }

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

    # A numeric position channel is not clinical posture evidence until its
    # device codes have been calibrated to anatomical labels (supine/lateral).
    # Preserve the raw values for audit, but do not expose them to scoring or
    # learned-adjustment features when that mapping is unavailable.
    raw_position_relevance = str(
        position_analysis.get("position_relevance", "UNKNOWN")
    ).upper()
    raw_position_event_ratio = safe_float(
        position_analysis.get("position_event_index_ratio")
    )
    known_position_fraction = safe_float(
        position_analysis.get("known_position_epoch_fraction")
    )
    position_mapping = str(
        position_analysis.get("anatomical_position_mapping", "UNCALIBRATED")
    ).upper()
    position_cluster_quality_valid = (
        raw_position_event_ratio is not None
        and known_position_fraction is not None
        and known_position_fraction >= 0.80
    )
    position_quality_valid = (
        position_mapping not in {"", "NONE", "UNKNOWN", "UNCALIBRATED"}
        and position_cluster_quality_valid
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
        "position_relevance": raw_position_relevance if position_quality_valid else "UNKNOWN",
        # A reliable unnamed-cluster ratio is valid as a device-independent
        # variability feature for learning.  It is not anatomical supine
        # evidence, so formal clinical scoring uses position_scoring_ratio.
        "position_event_ratio": raw_position_event_ratio if position_cluster_quality_valid else None,
        "position_scoring_ratio": raw_position_event_ratio if position_quality_valid else None,
        "position_cluster_quality_valid": position_cluster_quality_valid,
        "known_position_fraction": known_position_fraction,
        "position_mapping": position_mapping,
        "position_quality_valid": position_quality_valid,
        "raw_position_relevance": raw_position_relevance,
        "raw_position_event_ratio": raw_position_event_ratio,
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
    # Neutral origin: every point must come from a visible clinical rule.
    score = 0.0

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
    score = 0.0

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
    position_event_ratio = context.get("position_scoring_ratio")

    if not context.get("position_quality_valid", False):
        add_reason(
            limiting,
            "POSITION_EVIDENCE_NOT_CALIBRATED",
            (
                "姿勢感測代碼尚未校正為仰睡／側睡；可靠的未命名姿勢群組差異"
                "會保留為模型特徵，但本次不作為仰睡型 OSA 證據，也不直接計入 APAP 規則分數。"
            ),
            0,
        )

    central_count = context[
        "central_apnea_count"
    ]

    loop_gain_level = context[
        "loop_gain_level"
    ]

    oxygen_burden = context[
        "oxygen_burden"
    ]
    coupling_level = context["oxygen_coupling_level"]
    coupled_3pct_fraction = context["coupled_3pct_fraction"]

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

    if rem_relevance in {"HIGH", "MODERATE"}:
        add_reason(
            supporting,
            f"{rem_relevance}_REM_VARIABILITY_CONTEXT",
            (
                "呼吸事件具有睡眠階段相關性，可作為後續 PAP 壓力滴定與"
                "整夜追蹤的參考；但單憑 REM 相關性無法證明 APAP 療效"
                "優於 CPAP，因此不直接拉開兩者的治療排序分數。"
            ),
            0,
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
            "存在中度夜間血氧負荷，可納入PAP適配度判斷。",
            5,
        )

    if coupled_3pct_fraction is not None and coupled_3pct_fraction >= 0.50:
        score += 10
        add_reason(
            supporting,
            "EVENT_DESATURATION_COUPLING",
            "至少一半可分析呼吸事件伴隨≥3%血氧下降，支持事件具有生理影響。",
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
    score = 0.0

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
        "上呼吸道手術專科評估"
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
    評估藥物、體重治療與睡眠共病是否值得進一步專科評估。

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
    bmi = context.get("BMI")
    approved_obesity_osa_path = bool(
        bmi is not None
        and bmi >= 30.0
        and severity in {"MODERATE", "SEVERE"}
    )
    pap_tolerance = str(context.get("pap_tolerance") or "").strip().lower()
    pap_alternative_needed = any(
        term in pap_tolerance
        for term in (
            "poor", "intolerant", "cannot tolerate", "unable",
            "差", "不耐受", "無法耐受", "拒絕",
        )
    )
    pap_alternative_target_phenotype = rem_relevance == "HIGH"
    isi_score = safe_float(context.get("isi_score"))
    confirmed_insomnia = bool(
        context.get("insomnia_diagnosis")
        or (isi_score is not None and isi_score >= 15.0)
    )
    respiratory_depressant_medication = bool(
        context.get("sedative_use")
        or context.get("opioid_use")
        or context.get("respiratory_depressant_use")
    )
    current_medications = context.get("current_medications")
    medication_history_available = bool(
        current_medications
        if not isinstance(current_medications, str)
        else current_medications.strip()
    )

    # --------------------------------------------------------
    # 輕度與中度較適合作為條件式候選
    # 重度患者不應依靠此類藥物作為單獨主要治療
    # --------------------------------------------------------

    if approved_obesity_osa_path:
        score += 45
        add_reason(
            supporting,
            "FDA_APPROVED_OBESITY_OSA_MEDICATION_PATH",
            (
                f"BMI={bmi:.1f}，且 AHI 屬中重度 OSA；符合成人肥胖合併"
                "中重度 OSA 的 tirzepatide 核准路徑初步篩檢條件。"
                "此分數表示應由醫師評估適應症、禁忌與共同治療計畫，"
                "不是自動處方或保證療效。"
            ),
            45,
        )

    elif severity == "MILD" and pap_alternative_needed and pap_alternative_target_phenotype:
        score += 15

        add_reason(
            supporting,
            "PAP_ALTERNATIVE_TARGET_PHENOTYPE_RESEARCH_REVIEW",
            (
                "AHI位於輕度範圍，且已記錄PAP不耐受／拒絕，並具有明確"
                "目標表型；可由睡眠專科評估替代治療或研究型藥物路徑。"
            ),
            15,
        )

    elif severity == "MILD":
        add_reason(
            limiting,
            "MILD_OSA_MEDICATION_PATH_NOT_ESTABLISHED",
            "輕度OSA本身不足以支持藥物加分；尚須確認PAP不耐受／拒絕，並具有明確目標表型或可治療睡眠共病。",
            0,
        )

    elif severity == "MODERATE" and pap_alternative_needed and pap_alternative_target_phenotype:
        score += 10

        add_reason(
            supporting,
            "PAP_ALTERNATIVE_TARGET_PHENOTYPE_RESEARCH_REVIEW",
            (
                "AHI位於中度範圍，且已記錄PAP替代需求與明確目標表型；"
                "此類藥物最多作為專科輔助或研究評估。"
            ),
            10,
        )

    elif severity == "MODERATE":
        add_reason(
            limiting,
            "MODERATE_OSA_MEDICATION_PATH_NOT_ESTABLISHED",
            "中度OSA若不符合肥胖核准路徑，僅憑AHI不足以支持藥物加分；須另有PAP替代需求與明確目標表型。",
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

    # 共病失眠是另一條獨立的「睡眠共病治療評估」路徑，
    # 不宣稱失眠藥物可以治療 OSA，也不與肥胖核准路徑重複。
    if confirmed_insomnia:
        score += 15
        add_reason(
            supporting,
            "COMORBID_INSOMNIA_SPECIALIST_REVIEW",
            (
                "已記錄失眠診斷或 ISI≥15，支持進行失眠共病的獨立專科"
                "治療評估；此加分代表共病處理優先度，不代表失眠藥物可改善OSA。"
            ),
            15,
        )

    # 鎮靜安眠藥、鴉片類或其他呼吸抑制藥物需要獨立安全審查。
    # 此路徑提高「用藥安全評估」優先度，不代表推薦新增藥物。
    if respiratory_depressant_medication:
        score += 15
        add_reason(
            supporting,
            "RESPIRATORY_DEPRESSANT_MEDICATION_SAFETY_REVIEW",
            (
                "已記錄鎮靜安眠藥、鴉片類或其他可能抑制呼吸的藥物，"
                "支持優先進行用藥安全與替代方案審查；此分數不是藥物療效分數。"
            ),
            15,
        )

    # --------------------------------------------------------
    # 睡眠階段依賴性是核心 phenotype
    # --------------------------------------------------------

    if rem_relevance == "HIGH":
        add_reason(
            supporting,
            "HIGH_SLEEP_STAGE_DEPENDENCE_DESCRIPTIVE_ONLY",
            (
                "呼吸事件具有高度 REM 相關性；這是病理特徵描述，"
                "目前不能單獨證明患者會對特定 OSA 藥物反應，因此不直接加分。"
            ),
            0,
        )

    elif rem_relevance == "MODERATE":
        add_reason(
            supporting,
            "MODERATE_SLEEP_STAGE_DEPENDENCE",
            (
                "呼吸事件具有中度睡眠階段相關性，"
                "可作為病理特徵資料，但不足以單獨預測藥物療效。"
            ),
            0,
        )

    elif rem_relevance == "LOW":
        add_reason(
            limiting,
            "LOW_SLEEP_STAGE_DEPENDENCE",
            (
                "呼吸事件缺少明顯睡眠階段相關性，"
                "較不支持透過改變睡眠結構降低事件的假設。"
            ),
            0,
        )

    # 姿勢差異主要不是這類藥物的直接作用目標
    if position_relevance == "HIGH":
        add_reason(
            limiting,
            "POSITION_DOMINANT_PATTERN",
            (
                "呼吸事件主要隨姿勢高度變化，"
                "較偏向姿勢或壓力需求問題，而非單純睡眠結構問題。"
            ),
            0,
        )

    # --------------------------------------------------------
    # 高低氧風險限制
    # --------------------------------------------------------

    if approved_obesity_osa_path and oxygen_burden in {
        "HIGH_BURDEN", "MODERATE_BURDEN"
    }:
        add_reason(
            limiting,
            "HYPOXEMIA_REQUIRES_CONCURRENT_MONITORING",
            (
                "目前有夜間低氧負荷；這不否定已核准的肥胖合併中重度"
                "OSA 藥物評估路徑，但必須同步處理呼吸支持需求並追蹤血氧。"
            ),
            0,
        )
    elif oxygen_burden == "HIGH_BURDEN":
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
        if oxygen_burden in {"HIGH_BURDEN", "MODERATE_BURDEN"}:
            add_reason(
                limiting,
                "EVENT_DESATURATION_ALREADY_COUNTED",
                (
                    "至少一半可分析事件伴隨 ≥3% 血氧下降；"
                    "此訊號已包含在夜間缺氧負荷的扣分中，"
                    "為避免重複計算，本項只保留解釋、不再次扣分。"
                ),
                0,
            )
        elif not approved_obesity_osa_path:
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
        central_score_change = 0 if approved_obesity_osa_path else -10
        score += central_score_change
        add_reason(
            limiting,
            "CENTRAL_EVENT_BURDEN",
            (
                "中央型事件比例偏高；目前無法由 PSG 證明"
                "此睡眠結構調節型藥物可改善中央呼吸控制問題。"
            ),
            central_score_change,
        )

    # 非事件低氧不是此類藥物的適應性支持
    if (
        low_oxygen_away_fraction is not None
        and low_oxygen_away_fraction >= 0.70
    ):
        non_event_score_change = 0 if approved_obesity_osa_path else -15
        score += non_event_score_change

        add_reason(
            limiting,
            "NON_EVENT_HYPOXEMIA",
            (
                "大部分低氧時間遠離呼吸事件，"
                "提示可能存在心肺疾病、低通氣、基線低氧或量測問題；"
                "不支持以改變睡眠結構作為主要處理方式。"
            ),
            non_event_score_change,
        )

    if (
        wake_spo2 is not None
        and wake_spo2 < 94.0
    ):
        wake_score_change = 0 if approved_obesity_osa_path else -10
        score += wake_score_change

        add_reason(
            limiting,
            "LOW_WAKE_SPO2",
            (
                f"PSG 清醒期平均 SpO₂ 約為 {wake_spo2:.2f}%；"
                "應先釐清基線心肺與低通氣問題。若符合肥胖合併中重度"
                "OSA 的核准藥物路徑，此項作為安全監測要求，不重複扣分。"
            ),
            wake_score_change,
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
            "完整藥物名稱與作用機轉": medication_history_available,
            "睡眠品質與白天症狀": (
                isi_score is not None or confirmed_insomnia
            ),
            "完整共病與目前用藥": (
                medication_history_available or confirmed_insomnia
            ),
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
            "藥物／體重與睡眠共病專科評估"
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
            "包含成人肥胖合併中重度 OSA 的已核准體重藥物評估、"
            "共病失眠與現用藥安全審查，以及尚屬臨床試驗的 OSA 藥物機轉。"
        ),
        "expected_role": (
            "用於判斷是否需要醫師進一步評估。"
            "除符合核准適應症的 tirzepatide 路徑外，不能視為已證實可取代 PAP 的常規處方。"
        ),
        "known_tradeoffs": [
            "可能降低深度睡眠",
            "可能影響主觀睡眠品質或呼吸舒適度",
            "可能只減少部分事件而非完全消除",
            "療效需要治療後睡眠檢查確認",
        ],
        "required_confirmation": [
            "確認實際藥物名稱與作用機轉",
            "確認 BMI、肥胖診斷及是否符合已核准適應症",
            "確認睡眠階段相關呼吸事件型態",
            "確認白天嗜睡、睡眠品質與患者可接受性",
            "確認共病、目前用藥、肝腎功能與禁忌症",
            "治療後以 PSG 或適當監測確認效果",
        ],
        "interpretation": (
            "此分數是「藥物／體重／共病專科評估優先度」，不是藥物成功率。"
            "成人肥胖合併中重度 OSA 有 tirzepatide 的核准證據；輕度 OSA 的"
            "藥物可有研究訊號，但多數仍非常規核准治療。本模型不指定藥名、"
            "劑量或直接開立處方。"
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

    pap_fields = {
        "壓力設定": context.get("pap_pressure_setting"),
        "面罩漏氣": context.get("pap_mask_leak"),
        "殘餘 AHI": context.get("residual_ahi"),
        "治療後血氧": context.get("pap_treatment_spo2"),
        "患者耐受度": context.get("pap_tolerance"),
    }
    missing_pap_fields = [
        label for label, value in pap_fields.items()
        if value is None or str(value).strip() == ""
    ]
    if missing_pap_fields:
        available_pap_fields = [
            label for label in pap_fields if label not in missing_pap_fields
        ]
        availability_note = (
            "；目前已收到：" + "、".join(available_pap_fields)
            if available_pap_fields else ""
        )
        prerequisites.append({
            "priority": 4,
            "code": (
                "PAP_TITRATION_AND_TOLERANCE"
            ),
            "title": (
                "完成 PAP 壓力與耐受性評估"
            ),
            "reason": (
                "PAP評估資料尚未完整；仍缺少："
                + "、".join(missing_pap_fields)
                + availability_note
                + "。"
            ),
            "observed_value": {
                label: value for label, value in pap_fields.items()
                if label not in missing_pap_fields
            } or None,
            "required_information": missing_pap_fields,
        })

    # The additional medication pathways must only be activated by explicitly
    # confirmed clinical data. Missing fields are collected as a prerequisite;
    # absence must never be silently interpreted as a negative finding.
    medication_pathway_fields = {
        "PAP耐受或拒絕狀態": (
            "clinical_follow_up_tolerance" in context
            or "clinical_pap_tolerance" in context
        ),
        "ISI分數或失眠診斷": (
            "clinical_sleep_questionnaire_isi_score" in context
            or "clinical_sleep_questionnaire_insomnia_diagnosis" in context
            or "clinical_medication_insomnia_diagnosis" in context
        ),
        "目前用藥清單": "clinical_medication_medication_list" in context,
        "鎮靜、鴉片類及呼吸抑制藥物確認": any(
            key in context
            for key in (
                "clinical_medication_sedative_hypnotics",
                "clinical_medication_opioids",
                "clinical_medication_respiratory_depressants",
            )
        ),
    }
    missing_medication_pathway_fields = [
        label for label, available in medication_pathway_fields.items()
        if not available
    ]
    if missing_medication_pathway_fields:
        prerequisites.append({
            "priority": 5,
            "code": "MEDICATION_PATHWAY_CLINICAL_CONFIRMATION",
            "title": "完成藥物、失眠與PAP替代路徑資料確認",
            "reason": (
                "新版藥物／共病專科介入路徑只接受患者訪談、病歷或醫師"
                "確認資料；未填寫不會被推定為陰性，也不會自動產生加分。"
            ),
            "observed_value": None,
            "required_information": missing_medication_pathway_fields,
        })

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

    highest_prerequisite = (
        prerequisites[0]
        if prerequisites
        else None
    )

    co_first_candidates = []
    if top_score is not None:
        co_first_candidates = [
            {
                "treatment": treatment.get("treatment"),
                "treatment_label": treatment.get("treatment_label"),
                "score": safe_float(treatment.get("score")),
            }
            for treatment in treatments
            if safe_float(treatment.get("score")) == top_score
        ]

    first_candidate_is_tie = len(co_first_candidates) > 1
    # "unique" means exactly one treatment has the highest final score.
    # It must not be confused with an additional confidence/score-gap gate.
    # Those safeguards are already carried by recommendation_status and
    # confidence, whereas using them here produced contradictory output such
    # as "not unique" despite a non-tied first rank.
    unique_first_choice = bool(
        top_treatment is not None
        and top_score is not None
        and not first_candidate_is_tie
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
            if second_treatment and not first_candidate_is_tie
            else None
        ),
        "co_first_candidates": co_first_candidates,
        "first_candidate_is_tie": first_candidate_is_tie,
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
            "並列第一表示多個治療方向在目前證據下同分，"
            "不可解讀為清單中較早顯示者較佳。"
            if first_candidate_is_tie
            else "第一候選表示目前患者已觀察 phenotype "
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
    print("Personalized Treatment Recommendation")
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

def build_streamlit_interface(result: dict[str, Any]) -> dict[str, Any]:
    """Build the live Streamlit payload from the same final model output."""
    ranking = [
        item for item in result.get("personalized_treatment_ranking", [])
        if isinstance(item, dict)
    ]
    ranking.sort(key=lambda item: int(item.get("rank", 999)))
    treatments = []
    for item in ranking:
        confidence = item.get("confidence", {})
        if not isinstance(confidence, dict):
            confidence = {}
        score = safe_float(item.get("score")) or 0.0
        rule_score = safe_float(item.get("rule_based_score"))
        if rule_score is None:
            rule_score = score
        fixed_baseline = 0.0
        supporting_total = sum(
            safe_float(factor.get("weight")) or 0.0
            for factor in item.get("supporting_factors", [])
            if isinstance(factor, dict)
        )
        limiting_total = sum(
            safe_float(factor.get("weight")) or 0.0
            for factor in item.get("limiting_factors", [])
            if isinstance(factor, dict)
        )
        rule_factor_total = supporting_total + limiting_total
        learned_adjustment = safe_float(item.get("learned_adjustment")) or 0.0
        drug_adjustment = safe_float(item.get("drug_trial_adjustment")) or 0.0
        treatments.append(
            {
                **item,
                "treatment_id": item.get("treatment"),
                "initial_score": fixed_baseline,
                "fixed_baseline_score": fixed_baseline,
                "supporting_factor_total": supporting_total,
                "limiting_factor_total": limiting_total,
                "rule_factor_total": rule_factor_total,
                "rule_based_score": rule_score,
                "learned_adjustment": learned_adjustment,
                "drug_trial_adjustment": drug_adjustment,
                "final_score": score,
                "display_score": score,
                "score_change": score - fixed_baseline,
                "confidence": {
                    **confidence,
                    "level": confidence.get(
                        "confidence_level",
                        confidence.get("level"),
                    ),
                    "completeness": confidence.get(
                        "evidence_completeness",
                        confidence.get("completeness", 0),
                    ),
                },
            }
        )
    top = treatments[0] if treatments else {}
    top_score = safe_float(top.get("final_score")) if top else None
    top_candidates = [
        item for item in treatments
        if top_score is not None
        and safe_float(item.get("final_score")) == top_score
    ]
    top_treatment_summaries = [
        {
            "treatment": item.get("treatment_id"),
            "treatment_label": item.get("treatment_label"),
            "score": item.get("final_score"),
        }
        for item in top_candidates
    ]
    first_is_tie = len(top_treatment_summaries) > 1
    return {
        "schema_version": 15,
        "engine_version": 15,
        "engine": "clinical_guardrails_plus_adaptive_random_forest",
        "patient_id": result.get("patient_id"),
        "summary": {
            "treatment_count": len(treatments),
            # A singular winner is intentionally null on a tie. Consumers
            # must use top_treatments so a co-leader can never be discarded.
            "top_treatment": (
                None if first_is_tie else top.get("treatment_id")
            ),
            "top_treatment_label": (
                "、".join(
                    str(item.get("treatment_label") or item.get("treatment"))
                    for item in top_treatment_summaries
                ) + "（並列）"
                if first_is_tie
                else top.get("treatment_label")
            ),
            "top_treatments": top_treatment_summaries,
            "top_treatment_is_tie": first_is_tie,
            "top_score": top.get("final_score"),
            "score_semantics": (
                "0–100治療與專科介入評估優先度；不是成功率、療效機率或處方指示。"
            ),
        },
        "top_candidate": top,
        "top_candidates": top_candidates,
        "treatments": treatments,
        "recommendation_summary": result.get("recommendation_summary", {}),
        "clinical_prerequisites": result.get("clinical_prerequisites", []),
        "resolved_clinical_prerequisites": [],
        "follow_up_comparison": result.get("follow_up_comparison", {}),
        "medical_decision_basis": result.get("medical_decision_basis", {}),
        "clinical_feature_ingestion": result.get(
            "clinical_feature_ingestion", {}
        ),
        "treatment_learning": result.get("treatment_learning", {}),
        "treatment_learning_snapshot": result.get(
            "treatment_learning_snapshot", {}
        ),
        "research_treatment_preview": result.get(
            "research_treatment_preview", {}
        ),
        "safety_note": result.get("safety_note"),
    }


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
    parser.add_argument(
        "--follow-up",
        action="store_true",
        help="PSG Follow-up 模式",
    )
    parser.add_argument(
        "--skip-training-snapshot",
        action="store_true",
        help="重新訓練後只刷新輸出，不重複建立相同快照",
    )

    args = parser.parse_args()

    patient_id = str(
        args.patient_id
    ).strip()

    follow_up_mode = bool(args.follow_up)

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
    if clinical_data_file.is_file():
        clinical_data = load_json(clinical_data_file)
    else:
        # Every newly analysed patient needs a persistent, empty clinical
        # decision record before supplementary uploads can be merged.
        clinical_data = {
            "schema_version": 1,
            "patient_id": patient_id,
            "data_version": 0,
            "last_updated_at": None,
            "last_updated_module": None,
        }
        clinical_data_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_clinical_file = clinical_data_file.with_suffix(".json.tmp")
        temporary_clinical_file.write_text(
            json.dumps(clinical_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_clinical_file.replace(clinical_data_file)
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

    score_counts = {}
    for treatment in treatments:
        score_key = round(float(treatment["score"]), 8)
        score_counts[score_key] = score_counts.get(score_key, 0) + 1
    previous_score = None
    dense_rank = 0
    for position, treatment in enumerate(treatments, start=1):
        score_key = round(float(treatment["score"]), 8)
        if previous_score is None or score_key != previous_score:
            dense_rank = position
            previous_score = score_key
        treatment["rank"] = dense_rank
        treatment["is_tied"] = score_counts[score_key] > 1

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
        / "treatment_refinement"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_folder = (
        output_folder
        / "history"
    )

    history_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_json = (
        output_folder
        / "refined_treatment_recommendation.json"
    )

    ranking_csv = (
        output_folder
        / "refined_treatment_ranking.csv"
    )

    prerequisites_csv = (
        output_folder
        / "refined_clinical_prerequisites.csv"
    )

    generated_at = utc_now_iso()

    run_mode = (
        "FOLLOW_UP"
        if follow_up_mode
        else "INITIAL"
    )

    run_id = build_run_id(
        patient_id=patient_id,
        generated_at=generated_at,
        follow_up_mode=follow_up_mode,
    )

    previous_result = load_optional_json(
        output_json
    )

    previous_result_available = bool(
        isinstance(
            previous_result,
            dict,
        )
    )

    previous_refinement_run: dict[
        str,
        Any
    ] = {}

    if isinstance(
        previous_result,
        dict,
    ):
        raw_previous_refinement_run = (
            previous_result.get(
                "refinement_run",
                {},
            )
        )

        if isinstance(
            raw_previous_refinement_run,
            dict,
        ):
            previous_refinement_run = (
                raw_previous_refinement_run
            )

    parent_run_id = (
        previous_refinement_run.get(
            "run_id"
        )
        if follow_up_mode
        else None
    )

    recommendation_history: list[
        dict[str, Any]
    ] = []

    archived_previous_result_file: (
        Path | None
    ) = None

    follow_up_warnings: list[str] = []

    if follow_up_mode:
        recommendation_history = (
            extract_existing_history(
                previous_result
            )
        )

        if isinstance(
            previous_result,
            dict,
        ):
            previous_snapshot = (
                build_recommendation_snapshot(
                    previous_result
                )
            )

            previous_snapshot_run_id = (
                previous_snapshot.get(
                    "run_id"
                )
            )

            existing_history_run_ids = {
                str(
                    item.get(
                        "run_id"
                    )
                )
                for item
                in recommendation_history
                if isinstance(
                    item,
                    dict,
                )
                and item.get(
                    "run_id"
                )
                is not None
            }

            should_append_previous = bool(
                previous_snapshot_run_id
                is None
                or str(
                    previous_snapshot_run_id
                )
                not in existing_history_run_ids
            )

            if should_append_previous:
                recommendation_history.append(
                    previous_snapshot
                )

            archived_previous_result_file = (
                history_folder
                / (
                    "previous_result_before_"
                    f"{run_id}.json"
                )
            )

            save_json(
                archived_previous_result_file,
                previous_result,
            )

        else:
            follow_up_warnings.append(
                "本次使用 FOLLOW_UP 模式執行，"
                "但找不到既有的個人化治療推薦 JSON；"
                "因此無法建立前次推薦比較基準。"
            )

    recommendation_history = sorted(
        recommendation_history,
        key=lambda item: str(
            item.get(
                "generated_at",
                "",
            )
        ),
    )

    run_sequence = (
        len(
            recommendation_history
        )
        + 1
    )

    refinement_run = {
        "run_id": run_id,
        "run_sequence": run_sequence,
        "mode": run_mode,
        "is_follow_up": follow_up_mode,
        "generated_at": generated_at,
        "parent_run_id": parent_run_id,
        "previous_result_available": (
            previous_result_available
        ),
        "history_entry_count": len(
            recommendation_history
        ),
        "archived_previous_result_file": (
            str(
                archived_previous_result_file
            )
            if archived_previous_result_file
            is not None
            else None
        ),
        "warnings": follow_up_warnings,
    }

    follow_up_comparison = (
        build_follow_up_comparison(
            previous_result=previous_result,
            current_treatments=treatments,
            current_recommendation_summary=(
                recommendation_summary
            ),
            follow_up_mode=follow_up_mode,
        )
    )

    result = {
        "patient_id": patient_id,
        "mode": run_mode,
        "generated_at": generated_at,
        "refinement_run": (
            refinement_run
        ),
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
        "clinical_prerequisites": (
            prerequisites
        ),
        "recommendation_summary": (
            recommendation_summary
        ),
        "recommendation_history": (
            recommendation_history
        ),
        "follow_up_comparison": (
            follow_up_comparison
        ),
        "follow_up_status": {
            "requested": (
                follow_up_mode
            ),
            "baseline_available": (
                previous_result_available
                if follow_up_mode
                else None
            ),
            "parent_run_id": (
                parent_run_id
            ),
            "archived_previous_result": (
                str(
                    archived_previous_result_file
                )
                if archived_previous_result_file
                is not None
                else None
            ),
            "warnings": (
                follow_up_warnings
            ),
        },
        "safety_note": (
            "此結果為研究型個人化治療候選排序，"
            "不是處方、正式診斷或保證治療成功。"
        ),
    }

    # Keep an untouched rule-based copy for the research preview.  Clinical
    # guardrails must be applied exactly once, after the selected learned
    # adjustment; applying them both before and after learning duplicated
    # factors such as DOCUMENTED_PAP_INTOLERANCE and deducted them twice.
    research_source = copy.deepcopy(result)
    if args.skip_training_snapshot:
        result["treatment_learning_snapshot"] = {
            "status": "skipped_post_training_refresh",
            "reason": "本次只以新模型刷新輸出，避免同一補充資料重複建立訓練快照。",
        }
    else:
        result["treatment_learning_snapshot"] = record_treatment_snapshot(
            PROJECT_ROOT,
            result,
        )
    result = apply_learned_treatment_adjustment(
        PROJECT_ROOT,
        result,
    )
    guarded_treatments, medical_decision_basis = apply_medical_guardrails(
        result.get("personalized_treatment_ranking", []),
        context,
    )
    result["personalized_treatment_ranking"] = guarded_treatments
    result["medical_decision_basis"] = medical_decision_basis
    result["recommendation_summary"] = build_recommendation_summary(
        treatments=guarded_treatments,
        prerequisites=prerequisites,
    )
    # Guardrails can change scores and rank order. Rebuild the comparison only
    # after the final guarded ranking exists so every summary names the same
    # current top candidate shown on screen.
    result["follow_up_comparison"] = build_follow_up_comparison(
        previous_result=previous_result,
        current_treatments=guarded_treatments,
        current_recommendation_summary=result["recommendation_summary"],
        follow_up_mode=follow_up_mode,
    )
    research_result = apply_learned_treatment_adjustment(
        PROJECT_ROOT,
        research_source,
        research_preview=True,
    )
    research_ranking, research_basis = apply_medical_guardrails(
        research_result.get("personalized_treatment_ranking", []),
        context,
    )
    result["research_treatment_preview"] = {
        "warning": "研究測試輸出；含弱監督資料，不可作為診斷、處方或醫療警報。",
        "model": research_result.get("treatment_learning", {}),
        "ranking": research_ranking,
        "medical_decision_basis": research_basis,
    }
    # Clinical decision stability: a durable treatment choice must not flip
    # because of a short wearable fluctuation or one unconfirmed supplement.
    # The new ranking remains visible as a pending candidate, while the prior
    # top choice stays first until material evidence persists twice.
    previous_output = {}
    if output_json.exists():
        try:
            previous_output = json.loads(output_json.read_text(encoding="utf-8"))
        except Exception:
            previous_output = {}
    previous_items = previous_output.get("personalized_treatment_ranking") or []
    previous_items = sorted(
        [item for item in previous_items if isinstance(item, dict)],
        key=lambda item: int(item.get("rank", 999)),
    )
    previous_top = str(previous_items[0].get("treatment") or "") if previous_items else ""
    proposed_top = str(guarded_treatments[0].get("treatment") or "") if guarded_treatments else ""
    decision_path = PROJECT_ROOT / "data" / "inference" / args.patient_id / "clinical_decision" / "clinical_decision_data.json"
    decision_data = {}
    if decision_path.exists():
        try:
            decision_data = json.loads(decision_path.read_text(encoding="utf-8"))
        except Exception:
            decision_data = {}
    last_module = str(decision_data.get("last_updated_module") or "")
    material_modules = {
        "PSG_FOLLOWUP", "ENT_STRUCTURE", "DISE", "PAP_THERAPY",
        "TREATMENT_OUTCOME", "SURGERY_HISTORY",
    }
    state_path = output_folder / "treatment_decision_stability.json"
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    accepted_switch = False
    if previous_top and proposed_top and proposed_top != previous_top:
        qualifying = last_module in material_modules
        consecutive = int(state.get("consecutive_qualifying_updates", 0))
        if qualifying and state.get("pending_candidate") == proposed_top:
            consecutive += 1
        elif qualifying:
            consecutive = 1
        else:
            consecutive = 0
        accepted_switch = qualifying and consecutive >= 2
        # Stability is retained as an advisory state only.  It must never
        # rewrite the displayed ranking, otherwise rank 1 can have a lower
        # score than rank 2 after new evidence or a scoring-rule update.
        state = {
            "locked_top_treatment": proposed_top if accepted_switch else previous_top,
            "pending_candidate": None if accepted_switch else proposed_top,
            "consecutive_qualifying_updates": 0 if accepted_switch else consecutive,
            "last_module": last_module,
            "material_evidence": qualifying,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    else:
        state = {
            **state,
            "locked_top_treatment": proposed_top or previous_top,
            "pending_candidate": None,
            "consecutive_qualifying_updates": 0,
            "last_module": last_module,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    result["treatment_decision_stability"] = {
        **state,
        "policy": "短期穿戴式波動不切換；重大臨床證據需連續兩次支持，才更換正式首選。",
        "accepted_switch": accepted_switch,
    }
    result["personalized_treatment_ranking"] = guarded_treatments
    treatments = guarded_treatments
    recommendation_summary = result["recommendation_summary"]

    save_json(
        output_json,
        result,
    )
    save_json(
        output_folder / "treatment_refinement_interface.json",
        build_streamlit_interface(result),
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
                "run_id": run_id,
                "run_sequence": (
                    run_sequence
                ),
                "mode": run_mode,
                "is_follow_up": (
                    follow_up_mode
                ),
                "generated_at": (
                    generated_at
                ),
                "parent_run_id": (
                    parent_run_id
                ),
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
                "run_id": run_id,
                "run_sequence": (
                    run_sequence
                ),
                "mode": run_mode,
                "is_follow_up": (
                    follow_up_mode
                ),
                "generated_at": (
                    generated_at
                ),
                "parent_run_id": (
                    parent_run_id
                ),
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

    print()
    print("=" * 80)
    print("治療推薦執行資訊")
    print("=" * 80)

    print(
        f"執行模式：{run_mode}"
    )

    print(
        f"Run ID：{run_id}"
    )

    print(
        f"Run Sequence：{run_sequence}"
    )

    print(
        f"執行時間：{generated_at}"
    )

    print(
        "是否為 Follow-up："
        f"{follow_up_mode}"
    )

    print(
        "是否找到前次推薦結果："
        f"{previous_result_available}"
    )

    print(
        f"Parent Run ID：{parent_run_id}"
    )

    print(
        "歷史推薦筆數："
        f"{len(recommendation_history)}"
    )

    if (
        archived_previous_result_file
        is not None
    ):
        print(
            "前次完整結果備份："
            f"{archived_previous_result_file}"
        )

    if follow_up_warnings:
        print("Follow-up 警示：")

        for warning in follow_up_warnings:
            print(
                f"* {warning}"
            )

    if follow_up_mode:
        print()
        print("-" * 80)
        print("Follow-up 推薦比較")
        print("-" * 80)

        if follow_up_comparison.get(
            "comparison_available"
        ):
            print(
                "前次第一候選："
                f"{follow_up_comparison.get('previous_top_candidate')}"
            )

            print(
                "本次第一候選："
                f"{follow_up_comparison.get('current_top_candidate')}"
            )

            print(
                "第一候選是否改變："
                f"{follow_up_comparison.get('top_candidate_changed')}"
            )

            print(
                "第一候選分數變化："
                f"{follow_up_comparison.get('top_candidate_score_change')}"
            )

            print()
            print("各治療分數與排名變化：")

            for item in follow_up_comparison.get(
                "treatment_changes",
                [],
            ):
                print(
                    f"* {item.get('treatment')}："
                    f"分數 "
                    f"{item.get('previous_score')} → "
                    f"{item.get('current_score')} "
                    f"({item.get('score_change_direction')})；"
                    f"排名 "
                    f"{item.get('previous_rank')} → "
                    f"{item.get('current_rank')} "
                    f"({item.get('rank_change_direction')})"
                )

        else:
            print(
                "無法建立比較："
                f"{follow_up_comparison.get('comparison_reason')}"
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

    co_first_candidates = recommendation_summary.get(
        "co_first_candidates",
        [],
    )
    first_candidate_is_tie = bool(
        recommendation_summary.get("first_candidate_is_tie")
        and isinstance(co_first_candidates, list)
        and len(co_first_candidates) > 1
    )

    if first_candidate_is_tie:
        print(
            "並列第一治療候選："
            + "、".join(
                str(item.get("treatment"))
                for item in co_first_candidates
                if isinstance(item, dict)
            )
        )
    else:
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

    if not first_candidate_is_tie:
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
