from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

INFERENCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "inference"
)

DEFAULT_OUTPUT_FOLDER = (
    INFERENCE_ROOT
    / "_treatment_rule_audit"
)

EXPECTED_TREATMENTS = [
    "APAP",
    "CPAP",
    "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW",
    "SURGERY",
]

TREATMENT_DISPLAY_NAMES = {
    "APAP": (
        "自動調壓正壓呼吸器"
    ),
    "CPAP": (
        "固定壓力正壓呼吸器"
    ),
    "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW": (
        "藥物／體重與睡眠共病專科評估"
    ),
    "SURGERY": (
        "上呼吸道手術專科評估"
    ),
}

BASELINE_SCORES = {
    "APAP": 0.0,
    "CPAP": 0.0,
    "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW": 0.0,
    "SURGERY": 0.0,
}

NEAR_FIRST_THRESHOLD = 10.0
VERY_NEAR_FIRST_THRESHOLD = 5.0
RULE_DOMINANCE_FRACTION = 0.50
MIN_PATIENTS_FOR_AUDIT_WARNING = 5


# ============================================================
# 基礎工具
# ============================================================

def safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(number):
        return None

    return float(number)


def safe_int(
    value: Any,
) -> int | None:
    number = safe_float(
        value
    )

    if number is None:
        return None

    return int(
        round(number)
    )


def safe_bool(
    value: Any,
) -> bool | None:
    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    text = str(
        value
    ).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "n",
    }:
        return False

    return None


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
    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

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


def safe_divide(
    numerator: float | int,
    denominator: float | int,
) -> float | None:
    numerator_value = safe_float(
        numerator
    )

    denominator_value = safe_float(
        denominator
    )

    if (
        numerator_value is None
        or denominator_value is None
        or denominator_value == 0
    ):
        return None

    result = (
        numerator_value
        / denominator_value
    )

    if not np.isfinite(result):
        return None

    return float(result)


def join_text(
    values: list[str],
) -> str:
    return "；".join(
        str(value).strip()
        for value in values
        if str(value).strip()
    )


def find_recommendation_files(
    inference_root: Path,
) -> list[Path]:
    files = sorted(
        inference_root.glob(
            "*/treatment_refinement/"
            "refined_treatment_recommendation.json"
        )
    )

    return [
        file_path
        for file_path in files
        if file_path.is_file()
    ]


# ============================================================
# JSON 解析
# ============================================================

def extract_confidence(
    treatment: dict[str, Any],
) -> dict[str, Any]:
    confidence = treatment.get(
        "confidence",
        {},
    )

    if not isinstance(
        confidence,
        dict,
    ):
        confidence = {}

    missing_evidence = confidence.get(
        "missing_evidence",
        [],
    )

    if not isinstance(
        missing_evidence,
        list,
    ):
        missing_evidence = []

    available_evidence = confidence.get(
        "available_evidence",
        [],
    )

    if not isinstance(
        available_evidence,
        list,
    ):
        available_evidence = []

    return {
        "confidence_level": (
            confidence.get(
                "confidence_level"
            )
        ),
        "completeness": safe_float(
            confidence.get(
                "completeness"
            )
        ),
        "available_count": safe_int(
            confidence.get(
                "available_count"
            )
        ),
        "total_count": safe_int(
            confidence.get(
                "total_count"
            )
        ),
        "missing_evidence": (
            missing_evidence
        ),
        "available_evidence": (
            available_evidence
        ),
    }


def normalize_factor_rows(
    patient_id: str,
    treatment_name: str,
    factor_type: str,
    factors: Any,
) -> list[dict[str, Any]]:
    if not isinstance(
        factors,
        list,
    ):
        return []

    rows: list[
        dict[str, Any]
    ] = []

    for factor in factors:
        if not isinstance(
            factor,
            dict,
        ):
            continue

        code = str(
            factor.get(
                "code",
                "",
            )
        ).strip()

        description = str(
            factor.get(
                "description",
                "",
            )
        ).strip()

        weight = safe_float(
            factor.get(
                "weight"
            )
        )

        if not code:
            code = "UNKNOWN_RULE"

        rows.append(
            {
                "patient_id": (
                    patient_id
                ),
                "treatment": (
                    treatment_name
                ),
                "factor_type": (
                    factor_type
                ),
                "rule_code": (
                    code
                ),
                "description": (
                    description
                ),
                "weight": (
                    weight
                ),
                "absolute_weight": (
                    abs(weight)
                    if weight is not None
                    else None
                ),
                "is_positive_weight": (
                    weight is not None
                    and weight > 0
                ),
                "is_negative_weight": (
                    weight is not None
                    and weight < 0
                ),
                "is_zero_weight": (
                    weight is not None
                    and np.isclose(
                        weight,
                        0.0,
                    )
                ),
            }
        )

    return rows


def parse_recommendation_file(
    file_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    data = load_json(
        file_path
    )

    patient_id = str(
        data.get(
            "patient_id",
            file_path.parents[1].name,
        )
    ).strip()

    context = data.get(
        "patient_context",
        {},
    )

    if not isinstance(
        context,
        dict,
    ):
        context = {}

    ranking = data.get(
        "personalized_treatment_ranking",
        [],
    )

    if not isinstance(
        ranking,
        list,
    ):
        raise RuntimeError(
            "personalized_treatment_ranking "
            f"不是 list：{file_path}"
        )

    treatments = [
        item
        for item in ranking
        if isinstance(
            item,
            dict,
        )
    ]

    if not treatments:
        raise RuntimeError(
            f"沒有治療排名資料：{file_path}"
        )

    treatments = sorted(
        treatments,
        key=lambda item: (
            safe_int(
                item.get(
                    "rank"
                )
            )
            if safe_int(
                item.get(
                    "rank"
                )
            )
            is not None
            else 999,
            -(
                safe_float(
                    item.get(
                        "score"
                    )
                )
                or 0.0
            ),
        ),
    )

    first_treatment = (
        treatments[0]
    )

    first_name = str(
        first_treatment.get(
            "treatment",
            "",
        )
    )

    first_score = safe_float(
        first_treatment.get(
            "score"
        )
    )

    second_score = None

    if len(treatments) >= 2:
        second_score = safe_float(
            treatments[1].get(
                "score"
            )
        )

    patient_row = {
        "patient_id": (
            patient_id
        ),
        "source_file": str(
            file_path
        ),
        "sex": (
            context.get(
                "sex"
            )
        ),
        "age": safe_float(
            context.get(
                "age"
            )
        ),
        "BMI": safe_float(
            context.get(
                "BMI"
            )
        ),
        "ahi": safe_float(
            context.get(
                "ahi"
            )
        ),
        "ahi_severity": (
            context.get(
                "ahi_severity"
            )
        ),
        "hypopnea_count": safe_int(
            context.get(
                "hypopnea_count"
            )
        ),
        "obstructive_apnea_count": (
            safe_int(
                context.get(
                    "obstructive_apnea_count"
                )
            )
        ),
        "central_apnea_count": (
            safe_int(
                context.get(
                    "central_apnea_count"
                )
            )
        ),
        "central_event_fraction": (
            safe_float(
                context.get(
                    "central_event_fraction"
                )
            )
        ),
        "obstructive_event_fraction": (
            safe_float(
                context.get(
                    "obstructive_event_fraction"
                )
            )
        ),
        "oxygen_burden": (
            context.get(
                "oxygen_burden"
            )
        ),
        "desaturation_depth": (
            context.get(
                "desaturation_depth"
            )
        ),
        "sustained_oxygen_burden": (
            context.get(
                "sustained_oxygen_burden"
            )
        ),
        "rem_relevance": (
            context.get(
                "rem_relevance"
            )
        ),
        "loop_gain_level": (
            context.get(
                "loop_gain_level"
            )
        ),
        "dominant_mechanism": (
            context.get(
                "dominant_mechanism"
            )
        ),
        "position_relevance": (
            context.get(
                "position_relevance"
            )
        ),
        "position_event_ratio": (
            safe_float(
                context.get(
                    "position_event_ratio"
                )
            )
        ),
        "known_position_fraction": (
            safe_float(
                context.get(
                    "known_position_fraction"
                )
            )
        ),
        "position_mapping": (
            context.get(
                "position_mapping"
            )
        ),
        "oxygen_coupling_level": (
            context.get(
                "oxygen_coupling_level"
            )
        ),
        "coupled_3pct_fraction": (
            safe_float(
                context.get(
                    "coupled_3pct_fraction"
                )
            )
        ),
        "coupled_4pct_fraction": (
            safe_float(
                context.get(
                    "coupled_4pct_fraction"
                )
            )
        ),
        "low_oxygen_away_fraction": (
            safe_float(
                context.get(
                    "low_oxygen_away_fraction"
                )
            )
        ),
        "low_oxygen_away_minutes": (
            safe_float(
                context.get(
                    "low_oxygen_away_minutes"
                )
            )
        ),
        "low_oxygen_total_minutes": (
            safe_float(
                context.get(
                    "low_oxygen_total_minutes"
                )
            )
        ),
        "wake_spo2": safe_float(
            context.get(
                "wake_spo2"
            )
        ),
        "sleep_spo2": safe_float(
            context.get(
                "sleep_spo2"
            )
        ),
        "rank_1_treatment": (
            first_name
        ),
        "rank_1_score": (
            first_score
        ),
        "rank_2_score": (
            second_score
        ),
        "score_gap_rank_1_to_2": (
            (
                first_score
                - second_score
            )
            if (
                first_score is not None
                and second_score is not None
            )
            else None
        ),
    }

    treatment_rows: list[
        dict[str, Any]
    ] = []

    factor_rows: list[
        dict[str, Any]
    ] = []

    score_lookup: dict[
        str,
        float | None,
    ] = {}

    for treatment in treatments:
        treatment_name = str(
            treatment.get(
                "treatment",
                "",
            )
        ).strip()

        if not treatment_name:
            continue

        score = safe_float(
            treatment.get(
                "score"
            )
        )

        score_lookup[
            treatment_name
        ] = score

        confidence = (
            extract_confidence(
                treatment
            )
        )

        supporting = treatment.get(
            "supporting_factors",
            [],
        )

        limiting = treatment.get(
            "limiting_factors",
            [],
        )

        supporting_rows = (
            normalize_factor_rows(
                patient_id=patient_id,
                treatment_name=(
                    treatment_name
                ),
                factor_type="SUPPORTING",
                factors=supporting,
            )
        )

        limiting_rows = (
            normalize_factor_rows(
                patient_id=patient_id,
                treatment_name=(
                    treatment_name
                ),
                factor_type="LIMITING",
                factors=limiting,
            )
        )

        factor_rows.extend(
            supporting_rows
        )

        factor_rows.extend(
            limiting_rows
        )

        supporting_weight_sum = float(
            sum(
                row["weight"]
                for row in supporting_rows
                if row["weight"]
                is not None
            )
        )

        limiting_weight_sum = float(
            sum(
                row["weight"]
                for row in limiting_rows
                if row["weight"]
                is not None
            )
        )

        rule_weight_sum = (
            supporting_weight_sum
            + limiting_weight_sum
        )

        baseline_score = (
            BASELINE_SCORES.get(
                treatment_name
            )
        )

        reconstructed_raw_score = None

        if baseline_score is not None:
            reconstructed_raw_score = (
                baseline_score
                + rule_weight_sum
            )

        reconstructed_clamped_score = None

        if reconstructed_raw_score is not None:
            reconstructed_clamped_score = float(
                min(
                    max(
                        reconstructed_raw_score,
                        0.0,
                    ),
                    100.0,
                )
            )

        reconstruction_error = None

        if (
            score is not None
            and reconstructed_clamped_score
            is not None
        ):
            reconstruction_error = (
                score
                - reconstructed_clamped_score
            )

        treatment_rows.append(
            {
                "patient_id": (
                    patient_id
                ),
                "treatment": (
                    treatment_name
                ),
                "treatment_label": (
                    treatment.get(
                        "treatment_label"
                    )
                ),
                "rank": safe_int(
                    treatment.get(
                        "rank"
                    )
                ),
                "score": (
                    score
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
                "missing_evidence": (
                    join_text(
                        [
                            str(item)
                            for item
                            in confidence[
                                "missing_evidence"
                            ]
                        ]
                    )
                ),
                "baseline_score": (
                    baseline_score
                ),
                "supporting_rule_count": (
                    len(
                        supporting_rows
                    )
                ),
                "limiting_rule_count": (
                    len(
                        limiting_rows
                    )
                ),
                "supporting_weight_sum": (
                    supporting_weight_sum
                ),
                "limiting_weight_sum": (
                    limiting_weight_sum
                ),
                "rule_weight_sum": (
                    rule_weight_sum
                ),
                "reconstructed_raw_score": (
                    reconstructed_raw_score
                ),
                "reconstructed_clamped_score": (
                    reconstructed_clamped_score
                ),
                "reconstruction_error": (
                    reconstruction_error
                ),
                "is_first_choice": (
                    safe_int(
                        treatment.get(
                            "rank"
                        )
                    )
                    == 1
                ),
                "gap_to_patient_first": (
                    (
                        first_score
                        - score
                    )
                    if (
                        first_score is not None
                        and score is not None
                    )
                    else None
                ),
                "within_5_points_of_first": (
                    (
                        first_score
                        - score
                    )
                    <= VERY_NEAR_FIRST_THRESHOLD
                    if (
                        first_score is not None
                        and score is not None
                    )
                    else None
                ),
                "within_10_points_of_first": (
                    (
                        first_score
                        - score
                    )
                    <= NEAR_FIRST_THRESHOLD
                    if (
                        first_score is not None
                        and score is not None
                    )
                    else None
                ),
            }
        )

    for treatment_name in (
        EXPECTED_TREATMENTS
    ):
        score = score_lookup.get(
            treatment_name
        )

        patient_row[
            (
                treatment_name
                + "_score"
            )
        ] = score

    apap_score = score_lookup.get(
        "APAP"
    )

    cpap_score = score_lookup.get(
        "CPAP"
    )

    other_score = score_lookup.get(
        "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW"
    )

    surgery_score = score_lookup.get(
        "SURGERY"
    )

    patient_row[
        "apap_minus_cpap"
    ] = (
        (
            apap_score
            - cpap_score
        )
        if (
            apap_score is not None
            and cpap_score is not None
        )
        else None
    )

    patient_row[
        "other_gap_to_first"
    ] = (
        (
            first_score
            - other_score
        )
        if (
            first_score is not None
            and other_score is not None
        )
        else None
    )

    patient_row[
        "surgery_gap_to_first"
    ] = (
        (
            first_score
            - surgery_score
        )
        if (
            first_score is not None
            and surgery_score is not None
        )
        else None
    )

    return (
        patient_row,
        treatment_rows,
        factor_rows,
    )


# ============================================================
# 規則統計
# ============================================================

def build_rule_summary(
    factor_df: pd.DataFrame,
    patient_count: int,
) -> pd.DataFrame:
    if factor_df.empty:
        return pd.DataFrame()

    rows: list[
        dict[str, Any]
    ] = []

    group_columns = [
        "treatment",
        "factor_type",
        "rule_code",
    ]

    grouped = factor_df.groupby(
        group_columns,
        dropna=False,
    )

    for group_key, group in grouped:
        treatment = group_key[0]
        factor_type = group_key[1]
        rule_code = group_key[2]

        weights = pd.to_numeric(
            group["weight"],
            errors="coerce",
        ).dropna()

        unique_patient_count = int(
            group[
                "patient_id"
            ].nunique()
        )

        trigger_fraction = safe_divide(
            unique_patient_count,
            patient_count,
        )

        description_counts = Counter(
            group[
                "description"
            ]
            .dropna()
            .astype(str)
        )

        most_common_description = None

        if description_counts:
            most_common_description = (
                description_counts
                .most_common(1)[0][0]
            )

        rows.append(
            {
                "treatment": (
                    treatment
                ),
                "treatment_label": (
                    TREATMENT_DISPLAY_NAMES.get(
                        str(treatment),
                        str(treatment),
                    )
                ),
                "factor_type": (
                    factor_type
                ),
                "rule_code": (
                    rule_code
                ),
                "description": (
                    most_common_description
                ),
                "trigger_count": int(
                    len(group)
                ),
                "unique_patient_count": (
                    unique_patient_count
                ),
                "trigger_fraction": (
                    trigger_fraction
                ),
                "weight_mean": (
                    float(
                        weights.mean()
                    )
                    if not weights.empty
                    else None
                ),
                "weight_min": (
                    float(
                        weights.min()
                    )
                    if not weights.empty
                    else None
                ),
                "weight_max": (
                    float(
                        weights.max()
                    )
                    if not weights.empty
                    else None
                ),
                "total_weight_contribution": (
                    float(
                        weights.sum()
                    )
                    if not weights.empty
                    else 0.0
                ),
                "absolute_total_weight": (
                    float(
                        weights.abs().sum()
                    )
                    if not weights.empty
                    else 0.0
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        [
            "treatment",
            "factor_type",
            "absolute_total_weight",
            "rule_code",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


def build_treatment_summary(
    treatment_df: pd.DataFrame,
    patient_count: int,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for treatment_name in (
        EXPECTED_TREATMENTS
    ):
        subset = treatment_df[
            treatment_df[
                "treatment"
            ]
            == treatment_name
        ].copy()

        scores = pd.to_numeric(
            subset[
                "score"
            ],
            errors="coerce",
        ).dropna()

        ranks = pd.to_numeric(
            subset[
                "rank"
            ],
            errors="coerce",
        ).dropna()

        gaps = pd.to_numeric(
            subset[
                "gap_to_patient_first"
            ],
            errors="coerce",
        ).dropna()

        completeness = pd.to_numeric(
            subset[
                "evidence_completeness"
            ],
            errors="coerce",
        ).dropna()

        supporting_weight = pd.to_numeric(
            subset[
                "supporting_weight_sum"
            ],
            errors="coerce",
        ).dropna()

        limiting_weight = pd.to_numeric(
            subset[
                "limiting_weight_sum"
            ],
            errors="coerce",
        ).dropna()

        first_count = int(
            (
                ranks == 1
            ).sum()
        )

        near_first_count = int(
            (
                gaps
                <= NEAR_FIRST_THRESHOLD
            ).sum()
        )

        very_near_first_count = int(
            (
                gaps
                <= VERY_NEAR_FIRST_THRESHOLD
            ).sum()
        )

        rows.append(
            {
                "treatment": (
                    treatment_name
                ),
                "treatment_label": (
                    TREATMENT_DISPLAY_NAMES.get(
                        treatment_name,
                        treatment_name,
                    )
                ),
                "baseline_score": (
                    BASELINE_SCORES.get(
                        treatment_name
                    )
                ),
                "patient_count": int(
                    len(subset)
                ),
                "score_available_count": int(
                    len(scores)
                ),
                "score_mean": (
                    float(
                        scores.mean()
                    )
                    if not scores.empty
                    else None
                ),
                "score_std": (
                    float(
                        scores.std(
                            ddof=0
                        )
                    )
                    if not scores.empty
                    else None
                ),
                "score_min": (
                    float(
                        scores.min()
                    )
                    if not scores.empty
                    else None
                ),
                "score_p25": (
                    float(
                        scores.quantile(
                            0.25
                        )
                    )
                    if not scores.empty
                    else None
                ),
                "score_median": (
                    float(
                        scores.median()
                    )
                    if not scores.empty
                    else None
                ),
                "score_p75": (
                    float(
                        scores.quantile(
                            0.75
                        )
                    )
                    if not scores.empty
                    else None
                ),
                "score_max": (
                    float(
                        scores.max()
                    )
                    if not scores.empty
                    else None
                ),
                "mean_rank": (
                    float(
                        ranks.mean()
                    )
                    if not ranks.empty
                    else None
                ),
                "rank_1_count": (
                    first_count
                ),
                "rank_1_fraction": (
                    safe_divide(
                        first_count,
                        patient_count,
                    )
                ),
                "within_5_points_count": (
                    very_near_first_count
                ),
                "within_10_points_count": (
                    near_first_count
                ),
                "mean_gap_to_first": (
                    float(
                        gaps.mean()
                    )
                    if not gaps.empty
                    else None
                ),
                "minimum_gap_to_first": (
                    float(
                        gaps.min()
                    )
                    if not gaps.empty
                    else None
                ),
                "mean_supporting_weight": (
                    float(
                        supporting_weight.mean()
                    )
                    if not supporting_weight.empty
                    else None
                ),
                "mean_limiting_weight": (
                    float(
                        limiting_weight.mean()
                    )
                    if not limiting_weight.empty
                    else None
                ),
                "mean_evidence_completeness": (
                    float(
                        completeness.mean()
                    )
                    if not completeness.empty
                    else None
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_patient_rule_contribution(
    factor_df: pd.DataFrame,
) -> pd.DataFrame:
    if factor_df.empty:
        return pd.DataFrame()

    grouped = (
        factor_df.groupby(
            [
                "patient_id",
                "treatment",
                "rule_code",
                "factor_type",
            ],
            dropna=False,
        )[
            "weight"
        ]
        .sum(
            min_count=1
        )
        .reset_index()
    )

    return grouped.sort_values(
        [
            "patient_id",
            "treatment",
            "factor_type",
            "weight",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


# ============================================================
# APAP 與 CPAP 差異分析
# ============================================================

def build_apap_cpap_comparison(
    patient_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "patient_id",
        "ahi",
        "ahi_severity",
        "rem_relevance",
        "position_relevance",
        "position_event_ratio",
        "oxygen_burden",
        "oxygen_coupling_level",
        "coupled_3pct_fraction",
        "low_oxygen_away_fraction",
        "central_apnea_count",
        "loop_gain_level",
        "APAP_score",
        "CPAP_score",
        "apap_minus_cpap",
        "rank_1_treatment",
    ]

    available_columns = [
        column
        for column in columns
        if column in patient_df.columns
    ]

    result = patient_df[
        available_columns
    ].copy()

    result[
        "pap_preference"
    ] = np.select(
        [
            pd.to_numeric(
                result[
                    "apap_minus_cpap"
                ],
                errors="coerce",
            )
            >= 5.0,
            pd.to_numeric(
                result[
                    "apap_minus_cpap"
                ],
                errors="coerce",
            )
            <= -5.0,
        ],
        [
            "APAP_FAVORED",
            "CPAP_FAVORED",
        ],
        default="CLOSE_OR_TIED",
    )

    return result.sort_values(
        [
            "pap_preference",
            "apap_minus_cpap",
            "patient_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


# ============================================================
# 可達性與警告
# ============================================================

def build_audit_findings(
    patient_df: pd.DataFrame,
    treatment_summary_df: pd.DataFrame,
    rule_summary_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    findings: list[
        dict[str, Any]
    ] = []

    patient_count = int(
        len(patient_df)
    )

    if patient_count == 0:
        return [
            {
                "severity": "ERROR",
                "code": "NO_PATIENT_DATA",
                "message": (
                    "沒有可用患者資料。"
                ),
            }
        ]

    for _, row in (
        treatment_summary_df.iterrows()
    ):
        treatment = str(
            row.get(
                "treatment"
            )
        )

        first_count = safe_int(
            row.get(
                "rank_1_count"
            )
        )

        near_count = safe_int(
            row.get(
                "within_10_points_count"
            )
        )

        maximum_score = safe_float(
            row.get(
                "score_max"
            )
        )

        mean_gap = safe_float(
            row.get(
                "mean_gap_to_first"
            )
        )

        if (
            first_count is not None
            and first_count == 0
        ):
            findings.append(
                {
                    "severity": "INFO",
                    "code": (
                        "TREATMENT_NEVER_FIRST"
                    ),
                    "treatment": (
                        treatment
                    ),
                    "message": (
                        f"{treatment} 在目前 "
                        f"{patient_count} 位患者中"
                        "從未成為第一名。"
                    ),
                }
            )

        if (
            first_count == 0
            and near_count is not None
            and near_count == 0
        ):
            findings.append(
                {
                    "severity": "WARNING",
                    "code": (
                        "TREATMENT_NOT_COMPETITIVE"
                    ),
                    "treatment": (
                        treatment
                    ),
                    "message": (
                        f"{treatment} 不但從未第一，"
                        f"也從未進入第一名 "
                        f"{NEAR_FIRST_THRESHOLD:.0f} 分以內；"
                        "需檢查評分尺度或患者 phenotype 覆蓋。"
                    ),
                }
            )

        elif (
            first_count == 0
            and near_count is not None
            and near_count > 0
        ):
            findings.append(
                {
                    "severity": "INFO",
                    "code": (
                        "TREATMENT_NEAR_FIRST"
                    ),
                    "treatment": (
                        treatment
                    ),
                    "patient_count": (
                        near_count
                    ),
                    "message": (
                        f"{treatment} 雖未成為第一，"
                        f"但有 {near_count} 位患者"
                        f"距第一名不超過 "
                        f"{NEAR_FIRST_THRESHOLD:.0f} 分。"
                    ),
                }
            )

        if (
            maximum_score is not None
            and maximum_score < 50.0
        ):
            findings.append(
                {
                    "severity": "WARNING",
                    "code": (
                        "LOW_OBSERVED_MAXIMUM_SCORE"
                    ),
                    "treatment": (
                        treatment
                    ),
                    "observed_maximum": (
                        maximum_score
                    ),
                    "message": (
                        f"{treatment} 在目前資料中的"
                        f"最高分僅 {maximum_score:.1f}，"
                        "可能缺少適合 phenotype，"
                        "或評分上限偏低。"
                    ),
                }
            )

        if (
            mean_gap is not None
            and mean_gap >= 30.0
        ):
            findings.append(
                {
                    "severity": "WARNING",
                    "code": (
                        "LARGE_MEAN_GAP_TO_FIRST"
                    ),
                    "treatment": (
                        treatment
                    ),
                    "mean_gap": (
                        mean_gap
                    ),
                    "message": (
                        f"{treatment} 與各患者第一名"
                        f"平均相差 {mean_gap:.1f} 分，"
                        "在目前評分尺度中競爭力偏低。"
                    ),
                }
            )

    if (
        patient_count
        >= MIN_PATIENTS_FOR_AUDIT_WARNING
        and not rule_summary_df.empty
    ):
        for treatment_name in (
            EXPECTED_TREATMENTS
        ):
            subset = rule_summary_df[
                rule_summary_df[
                    "treatment"
                ]
                == treatment_name
            ].copy()

            positive_rules = subset[
                pd.to_numeric(
                    subset[
                        "total_weight_contribution"
                    ],
                    errors="coerce",
                )
                > 0
            ]

            total_positive_weight = (
                pd.to_numeric(
                    positive_rules[
                        "total_weight_contribution"
                    ],
                    errors="coerce",
                )
                .fillna(0.0)
                .sum()
            )

            if total_positive_weight <= 0:
                continue

            positive_rules = (
                positive_rules.copy()
            )

            positive_rules[
                "positive_share"
            ] = (
                pd.to_numeric(
                    positive_rules[
                        "total_weight_contribution"
                    ],
                    errors="coerce",
                )
                .fillna(0.0)
                / total_positive_weight
            )

            dominant_rows = positive_rules[
                positive_rules[
                    "positive_share"
                ]
                >= RULE_DOMINANCE_FRACTION
            ]

            for _, dominant_row in (
                dominant_rows.iterrows()
            ):
                rule_code = (
                    dominant_row.get(
                        "rule_code"
                    )
                )

                positive_share = safe_float(
                    dominant_row.get(
                        "positive_share"
                    )
                )

                if positive_share is None:
                    continue

                findings.append(
                    {
                        "severity": "INFO",
                        "code": (
                            "DOMINANT_POSITIVE_RULE"
                        ),
                        "treatment": (
                            treatment_name
                        ),
                        "rule_code": (
                            rule_code
                        ),
                        "positive_weight_share": (
                            positive_share
                        ),
                        "message": (
                            f"{treatment_name} 的正向加分中，"
                            f"{rule_code} 約占 "
                            f"{positive_share:.1%}；"
                            "需確認是否過度依賴單一規則。"
                        ),
                    }
                )

    other_rows = patient_df[
        pd.to_numeric(
            patient_df[
                "other_gap_to_first"
            ],
            errors="coerce",
        )
        <= NEAR_FIRST_THRESHOLD
    ]

    surgery_rows = patient_df[
        pd.to_numeric(
            patient_df[
                "surgery_gap_to_first"
            ],
            errors="coerce",
        )
        <= NEAR_FIRST_THRESHOLD
    ]

    findings.append(
        {
            "severity": "SUMMARY",
            "code": (
                "OTHER_NEAR_FIRST_PATIENT_COUNT"
            ),
            "patient_count": int(
                len(other_rows)
            ),
            "message": (
                "藥物或其他機轉方向距第一名 "
                f"{NEAR_FIRST_THRESHOLD:.0f} 分內的患者數："
                f"{len(other_rows)}。"
            ),
        }
    )

    findings.append(
        {
            "severity": "SUMMARY",
            "code": (
                "SURGERY_NEAR_FIRST_PATIENT_COUNT"
            ),
            "patient_count": int(
                len(surgery_rows)
            ),
            "message": (
                "手術評估距第一名 "
                f"{NEAR_FIRST_THRESHOLD:.0f} 分內的患者數："
                f"{len(surgery_rows)}。"
            ),
        }
    )

    return findings


# ============================================================
# 顯示
# ============================================================

def print_treatment_summary(
    dataframe: pd.DataFrame,
) -> None:
    columns = [
        "treatment",
        "baseline_score",
        "score_mean",
        "score_median",
        "score_max",
        "rank_1_count",
        "within_5_points_count",
        "within_10_points_count",
        "mean_gap_to_first",
        "mean_evidence_completeness",
    ]

    available_columns = [
        column
        for column in columns
        if column in dataframe.columns
    ]

    print(
        dataframe[
            available_columns
        ].to_string(
            index=False
        )
    )


def print_top_rules(
    rule_summary_df: pd.DataFrame,
    top_n: int = 10,
) -> None:
    if rule_summary_df.empty:
        print(
            "沒有規則資料。"
        )
        return

    for treatment_name in (
        EXPECTED_TREATMENTS
    ):
        print()
        print(
            f"{treatment_name}："
        )

        subset = rule_summary_df[
            rule_summary_df[
                "treatment"
            ]
            == treatment_name
        ].copy()

        subset = subset.sort_values(
            "absolute_total_weight",
            ascending=False,
        ).head(
            top_n
        )

        columns = [
            "factor_type",
            "rule_code",
            "unique_patient_count",
            "trigger_fraction",
            "weight_mean",
            "total_weight_contribution",
        ]

        print(
            subset[
                columns
            ].to_string(
                index=False
            )
        )


def print_near_first_patients(
    patient_df: pd.DataFrame,
) -> None:
    columns = [
        "patient_id",
        "rank_1_treatment",
        "rank_1_score",
        "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW_score",
        "other_gap_to_first",
        "SURGERY_score",
        "surgery_gap_to_first",
        "ahi",
        "ahi_severity",
        "BMI",
        "obstructive_event_fraction",
        "low_oxygen_away_fraction",
        "wake_spo2",
    ]

    available_columns = [
        column
        for column in columns
        if column in patient_df.columns
    ]

    subset = patient_df[
        (
            pd.to_numeric(
                patient_df[
                    "other_gap_to_first"
                ],
                errors="coerce",
            )
            <= NEAR_FIRST_THRESHOLD
        )
        |
        (
            pd.to_numeric(
                patient_df[
                    "surgery_gap_to_first"
                ],
                errors="coerce",
            )
            <= NEAR_FIRST_THRESHOLD
        )
    ].copy()

    if subset.empty:
        print(
            "沒有 OTHER 或 SURGERY "
            f"距第一名 {NEAR_FIRST_THRESHOLD:.0f} 分內的患者。"
        )
        return

    print(
        subset[
            available_columns
        ].sort_values(
            [
                "other_gap_to_first",
                "surgery_gap_to_first",
            ],
            na_position="last",
        ).to_string(
            index=False
        )
    )


# ============================================================
# 主程式
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "稽核四種個人化治療評分規則的"
            "實際覆蓋率、分數貢獻與排名可達性。"
        )
    )

    parser.add_argument(
        "--inference-root",
        type=Path,
        default=INFERENCE_ROOT,
        help=(
            "推論資料根目錄；預設為 data/inference。"
        ),
    )

    parser.add_argument(
        "--output-folder",
        type=Path,
        default=DEFAULT_OUTPUT_FOLDER,
        help=(
            "規則稽核輸出資料夾。"
        ),
    )

    args = parser.parse_args()

    inference_root = (
        args.inference_root
        .expanduser()
        .resolve()
    )

    output_folder = (
        args.output_folder
        .expanduser()
        .resolve()
    )

    print("=" * 80)
    print("Treatment Rule Coverage Audit")
    print("=" * 80)

    print(
        f"Inference Root：{inference_root}"
    )

    recommendation_files = (
        find_recommendation_files(
            inference_root
        )
    )

    print(
        "找到個人化推薦結果："
        f"{len(recommendation_files)} 份"
    )

    if not recommendation_files:
        raise FileNotFoundError(
            "找不到任何 "
            "personalized_treatment_recommendation.json。"
        )

    patient_rows: list[
        dict[str, Any]
    ] = []

    treatment_rows: list[
        dict[str, Any]
    ] = []

    factor_rows: list[
        dict[str, Any]
    ] = []

    failed_rows: list[
        dict[str, Any]
    ] = []

    print()
    print("=" * 80)
    print("逐患者讀取")
    print("=" * 80)

    for index, file_path in enumerate(
        recommendation_files,
        start=1,
    ):
        patient_id = (
            file_path.parents[1].name
        )

        try:
            (
                patient_row,
                patient_treatment_rows,
                patient_factor_rows,
            ) = parse_recommendation_file(
                file_path
            )

            patient_rows.append(
                patient_row
            )

            treatment_rows.extend(
                patient_treatment_rows
            )

            factor_rows.extend(
                patient_factor_rows
            )

            first_treatment = (
                patient_row[
                    "rank_1_treatment"
                ]
            )

            first_score = (
                patient_row[
                    "rank_1_score"
                ]
            )

            print(
                f"[OK] {index}/"
                f"{len(recommendation_files)} "
                f"{patient_id} → "
                f"{first_treatment} "
                f"({first_score})"
            )

        except Exception as error:
            failed_rows.append(
                {
                    "patient_id": (
                        patient_id
                    ),
                    "source_file": str(
                        file_path
                    ),
                    "error_type": (
                        type(error).__name__
                    ),
                    "error_message": (
                        str(error)
                    ),
                }
            )

            print(
                f"[FAILED] {patient_id}："
                f"{type(error).__name__}: "
                f"{error}"
            )

    if not patient_rows:
        raise RuntimeError(
            "所有推薦結果都解析失敗。"
        )

    patient_df = pd.DataFrame(
        patient_rows
    ).sort_values(
        "patient_id"
    ).reset_index(
        drop=True
    )

    treatment_df = pd.DataFrame(
        treatment_rows
    ).sort_values(
        [
            "patient_id",
            "rank",
            "treatment",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    factor_df = pd.DataFrame(
        factor_rows
    ).sort_values(
        [
            "patient_id",
            "treatment",
            "factor_type",
            "rule_code",
        ]
    ).reset_index(
        drop=True
    )

    failed_df = pd.DataFrame(
        failed_rows
    )

    patient_count = int(
        len(patient_df)
    )

    rule_summary_df = (
        build_rule_summary(
            factor_df=factor_df,
            patient_count=patient_count,
        )
    )

    treatment_summary_df = (
        build_treatment_summary(
            treatment_df=(
                treatment_df
            ),
            patient_count=(
                patient_count
            ),
        )
    )

    patient_rule_df = (
        build_patient_rule_contribution(
            factor_df
        )
    )

    apap_cpap_df = (
        build_apap_cpap_comparison(
            patient_df
        )
    )

    findings = (
        build_audit_findings(
            patient_df=patient_df,
            treatment_summary_df=(
                treatment_summary_df
            ),
            rule_summary_df=(
                rule_summary_df
            ),
        )
    )

    reconstruction_errors = pd.to_numeric(
        treatment_df[
            "reconstruction_error"
        ],
        errors="coerce",
    ).dropna()

    maximum_reconstruction_error = (
        float(
            reconstruction_errors
            .abs()
            .max()
        )
        if not reconstruction_errors.empty
        else None
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_output = (
        output_folder
        / "patient_phenotype_and_scores.csv"
    )

    treatment_output = (
        output_folder
        / "patient_treatment_rule_totals.csv"
    )

    factor_output = (
        output_folder
        / "patient_rule_triggers.csv"
    )

    rule_summary_output = (
        output_folder
        / "rule_coverage_summary.csv"
    )

    treatment_summary_output = (
        output_folder
        / "treatment_competitiveness_summary.csv"
    )

    patient_rule_output = (
        output_folder
        / "patient_rule_contribution_matrix.csv"
    )

    apap_cpap_output = (
        output_folder
        / "apap_cpap_comparison.csv"
    )

    failed_output = (
        output_folder
        / "failed_audit_files.csv"
    )

    audit_json = (
        output_folder
        / "treatment_rule_audit_summary.json"
    )

    patient_df.to_csv(
        patient_output,
        index=False,
        encoding="utf-8-sig",
    )

    treatment_df.to_csv(
        treatment_output,
        index=False,
        encoding="utf-8-sig",
    )

    factor_df.to_csv(
        factor_output,
        index=False,
        encoding="utf-8-sig",
    )

    rule_summary_df.to_csv(
        rule_summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    treatment_summary_df.to_csv(
        treatment_summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    patient_rule_df.to_csv(
        patient_rule_output,
        index=False,
        encoding="utf-8-sig",
    )

    apap_cpap_df.to_csv(
        apap_cpap_output,
        index=False,
        encoding="utf-8-sig",
    )

    failed_df.to_csv(
        failed_output,
        index=False,
        encoding="utf-8-sig",
    )

    first_choice_counts = (
        patient_df[
            "rank_1_treatment"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    summary = {
        "inference_root": str(
            inference_root
        ),
        "patient_count": (
            patient_count
        ),
        "failed_file_count": int(
            len(failed_df)
        ),
        "first_choice_counts": (
            first_choice_counts
        ),
        "baseline_scores": (
            BASELINE_SCORES
        ),
        "maximum_score_reconstruction_error": (
            maximum_reconstruction_error
        ),
        "treatment_competitiveness": (
            treatment_summary_df.to_dict(
                orient="records"
            )
        ),
        "rule_coverage": (
            rule_summary_df.to_dict(
                orient="records"
            )
        ),
        "audit_findings": (
            findings
        ),
        "interpretation": (
            "此稽核描述目前 20 位患者中各項規則"
            "實際觸發情況與四種治療的競爭力；"
            "不能單靠本稽核證明規則具有臨床效度。"
        ),
    }

    save_json(
        audit_json,
        summary,
    )

    print()
    print("=" * 80)
    print("四種治療競爭力摘要")
    print("=" * 80)

    print_treatment_summary(
        treatment_summary_df
    )

    print()
    print("=" * 80)
    print("各治療主要規則")
    print("=" * 80)

    print_top_rules(
        rule_summary_df=(
            rule_summary_df
        ),
        top_n=10,
    )

    print()
    print("=" * 80)
    print("APAP 與 CPAP 分布")
    print("=" * 80)

    pap_counts = (
        apap_cpap_df[
            "pap_preference"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        pap_counts.to_string()
    )

    print()
    print("=" * 80)
    print("OTHER／SURGERY 接近第一名患者")
    print("=" * 80)

    print_near_first_patients(
        patient_df
    )

    print()
    print("=" * 80)
    print("稽核發現")
    print("=" * 80)

    if not findings:
        print(
            "目前沒有產生稽核警告。"
        )
    else:
        for finding in findings:
            severity = (
                finding.get(
                    "severity"
                )
            )

            message = (
                finding.get(
                    "message"
                )
            )

            print(
                f"[{severity}] {message}"
            )

    print()
    print("=" * 80)
    print("分數重建檢查")
    print("=" * 80)

    print(
        "最大分數重建誤差："
        f"{maximum_reconstruction_error}"
    )

    if (
        maximum_reconstruction_error
        is not None
        and maximum_reconstruction_error
        > 1e-6
    ):
        print(
            "警告：JSON 中的分數無法完全由"
            "基礎分與規則權重重建，"
            "請檢查是否有未輸出的隱藏計分規則。"
        )
    else:
        print(
            "分數可由基礎分與輸出規則完整重建。"
        )

    print()
    print("=" * 80)
    print("輸出檔案")
    print("=" * 80)

    print(
        f"患者 phenotype 與分數：{patient_output}"
    )

    print(
        f"患者治療加扣分：{treatment_output}"
    )

    print(
        f"逐條規則觸發：{factor_output}"
    )

    print(
        f"規則覆蓋摘要：{rule_summary_output}"
    )

    print(
        f"治療競爭力摘要：{treatment_summary_output}"
    )

    print(
        f"患者規則貢獻：{patient_rule_output}"
    )

    print(
        f"APAP／CPAP 比較：{apap_cpap_output}"
    )

    print(
        f"失敗紀錄：{failed_output}"
    )

    print(
        f"完整稽核 JSON：{audit_json}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
