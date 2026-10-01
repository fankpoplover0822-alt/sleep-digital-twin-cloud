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
    / "_treatment_validation"
)

EXPECTED_TREATMENTS = [
    "APAP",
    "CPAP",
    "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW",
    "SURGERY",
]

DOMINANCE_WARNING_THRESHOLD = 0.80
MINIMUM_PATIENTS_FOR_DOMINANCE_CHECK = 5
SMALL_SCORE_GAP_THRESHOLD = 5.0
NEAR_TIE_SCORE_GAP_THRESHOLD = 2.0


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
    number = safe_float(value)

    if number is None:
        return None

    return int(round(number))


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


def join_text(
    values: list[str],
) -> str:
    return "；".join(
        str(value).strip()
        for value in values
        if str(value).strip()
    )


# ============================================================
# 搜尋患者結果
# ============================================================

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


def extract_treatment_lookup(
    treatments: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup: dict[
        str,
        dict[str, Any],
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

        lookup[
            treatment_name
        ] = treatment

    return lookup


def extract_factor_text(
    treatment: dict[str, Any],
    key: str,
) -> str:
    items = treatment.get(
        key,
        [],
    )

    if not isinstance(
        items,
        list,
    ):
        return ""

    descriptions: list[str] = []

    for item in items:
        if isinstance(
            item,
            dict,
        ):
            description = str(
                item.get(
                    "description",
                    "",
                )
            ).strip()

            if description:
                descriptions.append(
                    description
                )

    return join_text(
        descriptions
    )


def extract_missing_evidence(
    treatment: dict[str, Any],
) -> str:
    confidence = treatment.get(
        "confidence",
        {},
    )

    if not isinstance(
        confidence,
        dict,
    ):
        return ""

    missing = confidence.get(
        "missing_evidence",
        [],
    )

    if not isinstance(
        missing,
        list,
    ):
        return ""

    return join_text(
        [
            str(item)
            for item in missing
        ]
    )


# ============================================================
# 單一患者解析
# ============================================================

def parse_patient_result(
    file_path: Path,
) -> tuple[
    dict[str, Any],
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
    )

    treatments = data.get(
        "personalized_treatment_ranking",
        [],
    )

    if not isinstance(
        treatments,
        list,
    ):
        raise RuntimeError(
            "personalized_treatment_ranking "
            f"不是 list：{file_path}"
        )

    valid_treatments = [
        treatment
        for treatment in treatments
        if isinstance(
            treatment,
            dict,
        )
    ]

    if not valid_treatments:
        raise RuntimeError(
            f"沒有治療排名資料：{file_path}"
        )

    valid_treatments = sorted(
        valid_treatments,
        key=lambda item: (
            safe_int(
                item.get("rank")
            )
            if safe_int(
                item.get("rank")
            )
            is not None
            else 999,
            -(
                safe_float(
                    item.get("score")
                )
                or 0.0
            ),
        ),
    )

    lookup = extract_treatment_lookup(
        valid_treatments
    )

    first = valid_treatments[0]

    second = (
        valid_treatments[1]
        if len(valid_treatments) >= 2
        else {}
    )

    first_score = safe_float(
        first.get("score")
    )

    second_score = safe_float(
        second.get("score")
    )

    score_gap = None

    if (
        first_score is not None
        and second_score is not None
    ):
        score_gap = (
            first_score
            - second_score
        )

    score_values = [
        safe_float(
            treatment.get("score")
        )
        for treatment in valid_treatments
    ]

    score_values = [
        value
        for value in score_values
        if value is not None
    ]

    exact_top_tie_count = 0

    if first_score is not None:
        exact_top_tie_count = int(
            sum(
                np.isclose(
                    value,
                    first_score,
                    atol=1e-9,
                )
                for value in score_values
            )
        )

    recommendation_summary = data.get(
        "recommendation_summary",
        {},
    )

    if not isinstance(
        recommendation_summary,
        dict,
    ):
        recommendation_summary = {}

    highest_prerequisite = (
        recommendation_summary.get(
            "highest_priority_prerequisite"
        )
    )

    prerequisite_title = None

    if isinstance(
        highest_prerequisite,
        dict,
    ):
        prerequisite_title = (
            highest_prerequisite.get(
                "title"
            )
        )

    patient_row: dict[
        str,
        Any,
    ] = {
        "patient_id": patient_id,
        "source_file": str(
            file_path
        ),
        "treatment_count": len(
            valid_treatments
        ),
        "rank_1_treatment": (
            first.get("treatment")
        ),
        "rank_1_label": (
            first.get(
                "treatment_label"
            )
        ),
        "rank_1_score": (
            first_score
        ),
        "rank_1_level": (
            first.get(
                "suitability_level"
            )
        ),
        "rank_1_status": (
            first.get(
                "recommendation_status"
            )
        ),
        "rank_1_confidence": (
            first.get(
                "confidence",
                {},
            ).get(
                "confidence_level"
            )
            if isinstance(
                first.get(
                    "confidence",
                    {},
                ),
                dict,
            )
            else None
        ),
        "rank_2_treatment": (
            second.get("treatment")
        ),
        "rank_2_score": (
            second_score
        ),
        "score_gap_rank_1_to_2": (
            score_gap
        ),
        "exact_top_tie_count": (
            exact_top_tie_count
        ),
        "has_exact_top_tie": (
            exact_top_tie_count > 1
        ),
        "near_tie_under_2_points": (
            score_gap is not None
            and score_gap
            <= NEAR_TIE_SCORE_GAP_THRESHOLD
        ),
        "small_gap_under_5_points": (
            score_gap is not None
            and score_gap
            <= SMALL_SCORE_GAP_THRESHOLD
        ),
        "unique_first_choice": (
            recommendation_summary.get(
                "unique_first_choice"
            )
        ),
        "highest_priority_prerequisite": (
            prerequisite_title
        ),
        "rank_1_supporting_factors": (
            extract_factor_text(
                first,
                "supporting_factors",
            )
        ),
        "rank_1_limiting_factors": (
            extract_factor_text(
                first,
                "limiting_factors",
            )
        ),
        "rank_1_missing_evidence": (
            extract_missing_evidence(
                first
            )
        ),
    }

    treatment_rows: list[
        dict[str, Any]
    ] = []

    for treatment_name in (
        EXPECTED_TREATMENTS
    ):
        treatment = lookup.get(
            treatment_name,
            {},
        )

        confidence = treatment.get(
            "confidence",
            {},
        )

        if not isinstance(
            confidence,
            dict,
        ):
            confidence = {}

        treatment_rows.append(
            {
                "patient_id": (
                    patient_id
                ),
                "treatment": (
                    treatment_name
                ),
                "present": bool(
                    treatment
                ),
                "rank": safe_int(
                    treatment.get(
                        "rank"
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
                "available_evidence_count": (
                    safe_int(
                        confidence.get(
                            "available_count"
                        )
                    )
                ),
                "total_evidence_count": (
                    safe_int(
                        confidence.get(
                            "total_count"
                        )
                    )
                ),
                "supporting_factors": (
                    extract_factor_text(
                        treatment,
                        "supporting_factors",
                    )
                ),
                "limiting_factors": (
                    extract_factor_text(
                        treatment,
                        "limiting_factors",
                    )
                ),
                "missing_evidence": (
                    extract_missing_evidence(
                        treatment
                    )
                ),
            }
        )

    return (
        patient_row,
        treatment_rows,
    )


# ============================================================
# 統計分析
# ============================================================

def build_first_choice_distribution(
    patients_df: pd.DataFrame,
) -> pd.DataFrame:
    total_patients = int(
        len(patients_df)
    )

    counts = Counter(
        patients_df[
            "rank_1_treatment"
        ]
        .dropna()
        .astype(str)
    )

    rows: list[
        dict[str, Any]
    ] = []

    for treatment in (
        EXPECTED_TREATMENTS
    ):
        count = int(
            counts.get(
                treatment,
                0,
            )
        )

        fraction = (
            float(
                count
                / total_patients
            )
            if total_patients > 0
            else None
        )

        rows.append(
            {
                "treatment": (
                    treatment
                ),
                "first_choice_count": (
                    count
                ),
                "first_choice_fraction": (
                    fraction
                ),
                "patient_count": (
                    total_patients
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_treatment_score_summary(
    treatment_rows_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for treatment in (
        EXPECTED_TREATMENTS
    ):
        subset = treatment_rows_df[
            treatment_rows_df[
                "treatment"
            ]
            == treatment
        ].copy()

        scores = pd.to_numeric(
            subset["score"],
            errors="coerce",
        ).dropna()

        ranks = pd.to_numeric(
            subset["rank"],
            errors="coerce",
        ).dropna()

        evidence = pd.to_numeric(
            subset[
                "evidence_completeness"
            ],
            errors="coerce",
        ).dropna()

        rows.append(
            {
                "treatment": treatment,
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
                "rank_1_count": int(
                    (
                        ranks == 1
                    ).sum()
                ),
                "rank_4_count": int(
                    (
                        ranks == 4
                    ).sum()
                ),
                "mean_evidence_completeness": (
                    float(
                        evidence.mean()
                    )
                    if not evidence.empty
                    else None
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_validation_warnings(
    patients_df: pd.DataFrame,
    first_choice_df: pd.DataFrame,
    score_summary_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    warnings: list[
        dict[str, Any]
    ] = []

    patient_count = int(
        len(patients_df)
    )

    if patient_count == 0:
        warnings.append(
            {
                "severity": "ERROR",
                "code": "NO_PATIENT_RESULTS",
                "message": (
                    "沒有找到任何個人化治療推薦結果。"
                ),
            }
        )

        return warnings

    if (
        patient_count
        >= MINIMUM_PATIENTS_FOR_DOMINANCE_CHECK
    ):
        for _, row in (
            first_choice_df.iterrows()
        ):
            fraction = safe_float(
                row.get(
                    "first_choice_fraction"
                )
            )

            if (
                fraction is not None
                and fraction
                >= DOMINANCE_WARNING_THRESHOLD
            ):
                warnings.append(
                    {
                        "severity": "WARNING",
                        "code": (
                            "FIRST_CHOICE_DOMINANCE"
                        ),
                        "treatment": (
                            row.get(
                                "treatment"
                            )
                        ),
                        "observed_fraction": (
                            fraction
                        ),
                        "threshold": (
                            DOMINANCE_WARNING_THRESHOLD
                        ),
                        "message": (
                            f"{row.get('treatment')} "
                            f"在 {fraction:.1%} 患者中成為第一名，"
                            "需檢查是否存在規則偏向。"
                        ),
                    }
                )

    exact_tie_count = int(
        patients_df[
            "has_exact_top_tie"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    if exact_tie_count > 0:
        warnings.append(
            {
                "severity": "INFO",
                "code": (
                    "EXACT_TOP_SCORE_TIES"
                ),
                "patient_count": (
                    exact_tie_count
                ),
                "message": (
                    f"有 {exact_tie_count} 位患者"
                    "出現第一名分數同分，"
                    "最終順序會受同分優先規則影響。"
                ),
            }
        )

    near_tie_count = int(
        patients_df[
            "near_tie_under_2_points"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    if near_tie_count > 0:
        warnings.append(
            {
                "severity": "INFO",
                "code": (
                    "NEAR_TIE_RECOMMENDATIONS"
                ),
                "patient_count": (
                    near_tie_count
                ),
                "message": (
                    f"有 {near_tie_count} 位患者"
                    "第一與第二名差距不超過 "
                    f"{NEAR_TIE_SCORE_GAP_THRESHOLD:.1f} 分，"
                    "不應視為明確唯一首選。"
                ),
            }
        )

    for _, row in (
        score_summary_df.iterrows()
    ):
        rank_1_count = safe_int(
            row.get(
                "rank_1_count"
            )
        )

        rank_4_count = safe_int(
            row.get(
                "rank_4_count"
            )
        )

        if (
            rank_1_count is not None
            and rank_1_count == 0
        ):
            warnings.append(
                {
                    "severity": "INFO",
                    "code": (
                        "TREATMENT_NEVER_FIRST"
                    ),
                    "treatment": (
                        row.get(
                            "treatment"
                        )
                    ),
                    "message": (
                        f"{row.get('treatment')} "
                        "目前從未成為第一名；"
                        "需確認是患者族群特性，"
                        "還是規則永遠無法支持該治療。"
                    ),
                }
            )

        if (
            rank_4_count is not None
            and rank_4_count
            == patient_count
        ):
            warnings.append(
                {
                    "severity": "WARNING",
                    "code": (
                        "TREATMENT_ALWAYS_LAST"
                    ),
                    "treatment": (
                        row.get(
                            "treatment"
                        )
                    ),
                    "message": (
                        f"{row.get('treatment')} "
                        "在所有患者中均為最後一名，"
                        "需檢查規則是否過度壓低該治療。"
                    ),
                }
            )

    return warnings


# ============================================================
# 顯示
# ============================================================

def print_patient_table(
    patients_df: pd.DataFrame,
) -> None:
    display_columns = [
        "patient_id",
        "rank_1_treatment",
        "rank_1_score",
        "rank_1_confidence",
        "rank_2_treatment",
        "rank_2_score",
        "score_gap_rank_1_to_2",
        "has_exact_top_tie",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in patients_df.columns
    ]

    print(
        patients_df[
            available_columns
        ].to_string(
            index=False
        )
    )


def print_distribution(
    distribution_df: pd.DataFrame,
) -> None:
    display = (
        distribution_df.copy()
    )

    display[
        "first_choice_fraction"
    ] = pd.to_numeric(
        display[
            "first_choice_fraction"
        ],
        errors="coerce",
    )

    display[
        "first_choice_percentage"
    ] = (
        display[
            "first_choice_fraction"
        ]
        * 100.0
    )

    print(
        display[
            [
                "treatment",
                "first_choice_count",
                "first_choice_percentage",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 主程式
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "彙整所有患者的個人化治療推薦結果，"
            "檢查推薦分布、同分、分數差距與規則偏向。"
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
            "驗證結果輸出資料夾。"
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
    print("Treatment Recommendation Validation")
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
        f"找到推薦結果："
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

    failed_rows: list[
        dict[str, Any]
    ] = []

    print()
    print("=" * 80)
    print("逐患者讀取")
    print("=" * 80)

    for file_index, file_path in enumerate(
        recommendation_files,
        start=1,
    ):
        patient_id = (
            file_path.parents[1].name
        )

        try:
            patient_row, rows = (
                parse_patient_result(
                    file_path
                )
            )

            patient_rows.append(
                patient_row
            )

            treatment_rows.extend(
                rows
            )

            print(
                f"[OK] {file_index}/"
                f"{len(recommendation_files)} "
                f"{patient_id} → "
                f"{patient_row['rank_1_treatment']} "
                f"({patient_row['rank_1_score']})"
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
            "所有推薦結果都讀取失敗。"
        )

    patients_df = pd.DataFrame(
        patient_rows
    ).sort_values(
        "patient_id"
    ).reset_index(
        drop=True
    )

    treatment_rows_df = pd.DataFrame(
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

    failed_df = pd.DataFrame(
        failed_rows
    )

    first_choice_df = (
        build_first_choice_distribution(
            patients_df
        )
    )

    score_summary_df = (
        build_treatment_score_summary(
            treatment_rows_df
        )
    )

    warnings = (
        build_validation_warnings(
            patients_df=patients_df,
            first_choice_df=(
                first_choice_df
            ),
            score_summary_df=(
                score_summary_df
            ),
        )
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_output = (
        output_folder
        / "patient_recommendation_summary.csv"
    )

    treatment_rows_output = (
        output_folder
        / "patient_treatment_scores.csv"
    )

    first_choice_output = (
        output_folder
        / "first_choice_distribution.csv"
    )

    score_summary_output = (
        output_folder
        / "treatment_score_summary.csv"
    )

    failed_output = (
        output_folder
        / "failed_recommendation_files.csv"
    )

    validation_json = (
        output_folder
        / "treatment_validation_summary.json"
    )

    patients_df.to_csv(
        patient_output,
        index=False,
        encoding="utf-8-sig",
    )

    treatment_rows_df.to_csv(
        treatment_rows_output,
        index=False,
        encoding="utf-8-sig",
    )

    first_choice_df.to_csv(
        first_choice_output,
        index=False,
        encoding="utf-8-sig",
    )

    score_summary_df.to_csv(
        score_summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    failed_df.to_csv(
        failed_output,
        index=False,
        encoding="utf-8-sig",
    )

    score_gaps = pd.to_numeric(
        patients_df[
            "score_gap_rank_1_to_2"
        ],
        errors="coerce",
    ).dropna()

    summary = {
        "inference_root": str(
            inference_root
        ),
        "recommendation_file_count": int(
            len(recommendation_files)
        ),
        "successful_patient_count": int(
            len(patients_df)
        ),
        "failed_patient_count": int(
            len(failed_df)
        ),
        "first_choice_distribution": (
            first_choice_df.to_dict(
                orient="records"
            )
        ),
        "treatment_score_summary": (
            score_summary_df.to_dict(
                orient="records"
            )
        ),
        "score_gap_summary": {
            "available_count": int(
                len(score_gaps)
            ),
            "mean": (
                float(
                    score_gaps.mean()
                )
                if not score_gaps.empty
                else None
            ),
            "median": (
                float(
                    score_gaps.median()
                )
                if not score_gaps.empty
                else None
            ),
            "minimum": (
                float(
                    score_gaps.min()
                )
                if not score_gaps.empty
                else None
            ),
            "maximum": (
                float(
                    score_gaps.max()
                )
                if not score_gaps.empty
                else None
            ),
            "small_gap_under_5_count": int(
                patients_df[
                    "small_gap_under_5_points"
                ]
                .fillna(False)
                .astype(bool)
                .sum()
            ),
            "near_tie_under_2_count": int(
                patients_df[
                    "near_tie_under_2_points"
                ]
                .fillna(False)
                .astype(bool)
                .sum()
            ),
            "exact_top_tie_count": int(
                patients_df[
                    "has_exact_top_tie"
                ]
                .fillna(False)
                .astype(bool)
                .sum()
            ),
        },
        "warnings": warnings,
        "interpretation": (
            "此驗證檢查推薦規則是否對所有患者"
            "過度偏向同一治療，以及第一、第二名"
            "是否經常接近或同分。"
        ),
    }

    save_json(
        validation_json,
        summary,
    )

    print()
    print("=" * 80)
    print("患者推薦摘要")
    print("=" * 80)

    print_patient_table(
        patients_df
    )

    print()
    print("=" * 80)
    print("第一治療候選分布")
    print("=" * 80)

    print_distribution(
        first_choice_df
    )

    print()
    print("=" * 80)
    print("各治療分數摘要")
    print("=" * 80)

    print(
        score_summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("驗證警告")
    print("=" * 80)

    if not warnings:
        print(
            "目前沒有觸發偏向或同分警告。"
        )
    else:
        for warning in warnings:
            print(
                f"[{warning.get('severity')}] "
                f"{warning.get('message')}"
            )

    print()
    print("=" * 80)
    print("輸出檔案")
    print("=" * 80)

    print(
        f"患者摘要：{patient_output}"
    )

    print(
        f"逐治療分數："
        f"{treatment_rows_output}"
    )

    print(
        f"第一候選分布："
        f"{first_choice_output}"
    )

    print(
        f"治療分數摘要："
        f"{score_summary_output}"
    )

    print(
        f"讀取失敗紀錄："
        f"{failed_output}"
    )

    print(
        f"完整驗證 JSON："
        f"{validation_json}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
