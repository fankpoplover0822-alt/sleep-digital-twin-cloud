from __future__ import annotations

import argparse
import html
import json
import os
import shutil
from datetime import datetime
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


def load_json_optional(
    file_path: Path,
) -> dict[str, Any]:
    if not file_path.exists():
        return {}

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
        return {}

    return data


def load_csv_optional(
    file_path: Path,
) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            file_path
        )
    except Exception:
        return pd.DataFrame()


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


def atomic_save_json(
    file_path: Path,
    data: dict[str, Any],
) -> None:
    """Strict JSON write through a temporary file and atomic replace."""
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_data = safe_json_value(
        data
    )

    temporary_file = file_path.with_suffix(
        file_path.suffix + ".tmp"
    )

    with temporary_file.open(
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
        file.flush()
        os.fsync(file.fileno())

    temporary_file.replace(
        file_path
    )


def snapshot_existing_outputs(
    output_folder: Path,
    output_files: list[Path],
    timestamp: str,
) -> list[str]:
    """Archive existing report outputs before a follow-up overwrite."""
    history_folder = (
        output_folder
        / "history"
    )

    snapshots: list[str] = []

    for file_path in output_files:
        if not file_path.is_file():
            continue

        history_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        snapshot_path = (
            history_folder
            / f"{timestamp}_{file_path.name}"
        )

        shutil.copy2(
            file_path,
            snapshot_path,
        )

        snapshots.append(
            str(snapshot_path)
        )

    return snapshots


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


def first_non_none(
    *values: Any,
) -> Any:
    for value in values:
        if value is not None:
            return value

    return None


def format_number(
    value: Any,
    digits: int = 2,
    suffix: str = "",
) -> str:
    number = safe_float(
        value
    )

    if number is None:
        return "NOT_AVAILABLE"

    return (
        f"{number:.{digits}f}"
        f"{suffix}"
    )


def format_percentage(
    value: Any,
    digits: int = 1,
) -> str:
    number = safe_float(
        value
    )

    if number is None:
        return "NOT_AVAILABLE"

    return (
        f"{number * 100.0:.{digits}f}%"
    )


def normalize_text(
    value: Any,
    default: str = "NOT_AVAILABLE",
) -> str:
    if value is None:
        return default

    text = str(
        value
    ).strip()

    if not text:
        return default

    return text


def join_text(
    values: list[str],
    separator: str = "、",
) -> str:
    clean_values = [
        str(value).strip()
        for value in values
        if str(value).strip()
    ]

    if not clean_values:
        return "NOT_AVAILABLE"

    return separator.join(
        clean_values
    )


# ============================================================
# SHAP／模型因素摘要
# ============================================================

def extract_factor_summary(
    factor_json: dict[str, Any],
    factor_csv: pd.DataFrame,
) -> dict[str, Any]:
    summary_text = first_non_none(
        factor_json.get(
            "generated_summary"
        ),
        factor_json.get(
            "automatic_summary"
        ),
        factor_json.get(
            "summary_text"
        ),
        factor_json.get(
            "patient_summary"
        ),
    )

    top_groups: list[str] = []

    group_candidates = first_non_none(
        factor_json.get(
            "top_physiological_groups"
        ),
        factor_json.get(
            "physiological_group_summary"
        ),
        factor_json.get(
            "top_groups"
        ),
    )

    if isinstance(
        group_candidates,
        list,
    ):
        for item in group_candidates:
            if isinstance(
                item,
                dict,
            ):
                name = first_non_none(
                    item.get(
                        "feature_group"
                    ),
                    item.get(
                        "group"
                    ),
                    item.get(
                        "name"
                    ),
                )
            else:
                name = item

            if name is not None:
                top_groups.append(
                    str(name)
                )

    increasing_factors: list[dict[str, Any]] = []
    decreasing_factors: list[dict[str, Any]] = []

    increasing_candidates = first_non_none(
        factor_json.get(
            "risk_increasing_factors"
        ),
        factor_json.get(
            "top_increasing_factors"
        ),
        [],
    )

    decreasing_candidates = first_non_none(
        factor_json.get(
            "risk_decreasing_factors"
        ),
        factor_json.get(
            "top_decreasing_factors"
        ),
        [],
    )

    if isinstance(
        increasing_candidates,
        list,
    ):
        for item in increasing_candidates:
            if isinstance(
                item,
                dict,
            ):
                increasing_factors.append(
                    {
                        "feature": first_non_none(
                            item.get(
                                "feature"
                            ),
                            item.get(
                                "name"
                            ),
                        ),
                        "feature_group": (
                            item.get(
                                "feature_group"
                            )
                        ),
                        "mean_signed_shap": (
                            safe_float(
                                first_non_none(
                                    item.get(
                                        "mean_signed_shap"
                                    ),
                                    item.get(
                                        "signed_shap"
                                    ),
                                )
                            )
                        ),
                    }
                )

    if isinstance(
        decreasing_candidates,
        list,
    ):
        for item in decreasing_candidates:
            if isinstance(
                item,
                dict,
            ):
                decreasing_factors.append(
                    {
                        "feature": first_non_none(
                            item.get(
                                "feature"
                            ),
                            item.get(
                                "name"
                            ),
                        ),
                        "feature_group": (
                            item.get(
                                "feature_group"
                            )
                        ),
                        "mean_signed_shap": (
                            safe_float(
                                first_non_none(
                                    item.get(
                                        "mean_signed_shap"
                                    ),
                                    item.get(
                                        "signed_shap"
                                    ),
                                )
                            )
                        ),
                    }
                )

    if not factor_csv.empty:
        first_row = factor_csv.iloc[0]

        if not top_groups:
            for column in [
                "top_group_1",
                "top_group_2",
                "top_group_3",
            ]:
                if column in factor_csv.columns:
                    value = first_row.get(
                        column
                    )

                    if (
                        value is not None
                        and not pd.isna(
                            value
                        )
                    ):
                        top_groups.append(
                            str(value)
                        )

        if not increasing_factors:
            for column in [
                "top_increasing_factor_1",
                "top_increasing_factor_2",
                "top_increasing_factor_3",
            ]:
                if column in factor_csv.columns:
                    value = first_row.get(
                        column
                    )

                    if (
                        value is not None
                        and not pd.isna(
                            value
                        )
                    ):
                        increasing_factors.append(
                            {
                                "feature": str(
                                    value
                                ),
                                "feature_group": None,
                                "mean_signed_shap": None,
                            }
                        )

    model_performance = {
        "roc_auc": safe_float(
            first_non_none(
                factor_json.get(
                    "roc_auc"
                ),
                (
                    factor_csv.iloc[0].get(
                        "roc_auc"
                    )
                    if (
                        not factor_csv.empty
                        and "roc_auc"
                        in factor_csv.columns
                    )
                    else None
                ),
            )
        ),
        "recall": safe_float(
            first_non_none(
                factor_json.get(
                    "recall"
                ),
                (
                    factor_csv.iloc[0].get(
                        "recall"
                    )
                    if (
                        not factor_csv.empty
                        and "recall"
                        in factor_csv.columns
                    )
                    else None
                ),
            )
        ),
        "specificity": safe_float(
            first_non_none(
                factor_json.get(
                    "specificity"
                ),
                (
                    factor_csv.iloc[0].get(
                        "specificity"
                    )
                    if (
                        not factor_csv.empty
                        and "specificity"
                        in factor_csv.columns
                    )
                    else None
                ),
            )
        ),
        "precision": safe_float(
            first_non_none(
                factor_json.get(
                    "precision"
                ),
                (
                    factor_csv.iloc[0].get(
                        "precision"
                    )
                    if (
                        not factor_csv.empty
                        and "precision"
                        in factor_csv.columns
                    )
                    else None
                ),
            )
        ),
    }

    return {
        "available": bool(
            factor_json
            or not factor_csv.empty
        ),
        "summary_text": (
            summary_text
        ),
        "top_physiological_groups": (
            top_groups[:8]
        ),
        "risk_increasing_factors": (
            increasing_factors[:10]
        ),
        "risk_decreasing_factors": (
            decreasing_factors[:10]
        ),
        "model_performance": (
            model_performance
        ),
        "explanation_method": factor_json.get("explanation_method"),
        "explanation_scope": factor_json.get("explanation_scope"),
        "explanation_warning": factor_json.get("explanation_warning"),
    }


# ============================================================
# 治療結果
# ============================================================

def extract_arousal_shap_summary(
    shap_group_csv: pd.DataFrame,
    shap_feature_csv: pd.DataFrame,
) -> dict[str, Any]:
    if (
        shap_group_csv.empty
        or shap_feature_csv.empty
    ):
        return {
            "available": False,
            "analysis_scope": "ALERT_EPOCHS",
            "group_importance": [],
            "risk_increasing_factors": [],
            "risk_decreasing_factors": [],
        }

    group_df = shap_group_csv.copy()

    feature_df = shap_feature_csv.copy()

    if "analysis" in group_df.columns:
        group_df = group_df[
            group_df["analysis"].astype(str)
            == "ALERT_EPOCHS"
        ].copy()

    if "analysis" in feature_df.columns:
        feature_df = feature_df[
            feature_df["analysis"].astype(str)
            == "ALERT_EPOCHS"
        ].copy()

    for column in [
        "rank",
        "total_mean_absolute_shap",
        "mean_feature_absolute_shap",
        "total_mean_signed_shap",
    ]:
        if column in group_df.columns:
            group_df[column] = pd.to_numeric(
                group_df[column],
                errors="coerce",
            )

    for column in [
        "rank",
        "mean_absolute_shap",
        "mean_signed_shap",
        "positive_effect_fraction",
        "negative_effect_fraction",
    ]:
        if column in feature_df.columns:
            feature_df[column] = pd.to_numeric(
                feature_df[column],
                errors="coerce",
            )

    group_df = group_df.sort_values(
        [
            "rank",
            "total_mean_absolute_shap",
        ],
        ascending=[
            True,
            False,
        ],
        na_position="last",
    )

    feature_df = feature_df.sort_values(
        [
            "rank",
            "mean_absolute_shap",
        ],
        ascending=[
            True,
            False,
        ],
        na_position="last",
    )

    group_rows: list[dict[str, Any]] = []

    total_group_importance = pd.to_numeric(
        group_df.get(
            "total_mean_absolute_shap",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    ).fillna(0.0).sum()

    for _, row in group_df.head(8).iterrows():
        absolute_value = safe_float(
            row.get(
                "total_mean_absolute_shap"
            )
        )

        relative_share = None

        if (
            absolute_value is not None
            and total_group_importance > 0
        ):
            relative_share = float(
                absolute_value
                / total_group_importance
            )

        signed_value = safe_float(
            row.get(
                "total_mean_signed_shap"
            )
        )

        if signed_value is None:
            direction = "UNKNOWN"
        elif signed_value > 0:
            direction = "RISK_INCREASING"
        elif signed_value < 0:
            direction = "RISK_DECREASING"
        else:
            direction = "NEUTRAL"

        group_rows.append(
            {
                "rank": safe_int(
                    row.get(
                        "rank"
                    )
                ),
                "feature_group": (
                    row.get(
                        "feature_group"
                    )
                ),
                "total_mean_absolute_shap": (
                    absolute_value
                ),
                "total_mean_signed_shap": (
                    signed_value
                ),
                "relative_importance_share": (
                    relative_share
                ),
                "direction": (
                    direction
                ),
                "feature_count": safe_int(
                    row.get(
                        "feature_count"
                    )
                ),
                "epoch_count": safe_int(
                    row.get(
                        "epoch_count"
                    )
                ),
            }
        )

    increasing_df = feature_df[
        pd.to_numeric(
            feature_df.get(
                "mean_signed_shap",
                pd.Series(
                    index=feature_df.index,
                    dtype=float,
                ),
            ),
            errors="coerce",
        )
        > 0
    ].copy()

    decreasing_df = feature_df[
        pd.to_numeric(
            feature_df.get(
                "mean_signed_shap",
                pd.Series(
                    index=feature_df.index,
                    dtype=float,
                ),
            ),
            errors="coerce",
        )
        < 0
    ].copy()

    increasing_df = increasing_df.sort_values(
        "mean_signed_shap",
        ascending=False,
        na_position="last",
    ).head(8)

    decreasing_df = decreasing_df.sort_values(
        "mean_signed_shap",
        ascending=True,
        na_position="last",
    ).head(8)

    def convert_feature_rows(
        dataframe: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for _, row in dataframe.iterrows():
            rows.append(
                {
                    "rank": safe_int(
                        row.get(
                            "rank"
                        )
                    ),
                    "feature": (
                        row.get(
                            "feature"
                        )
                    ),
                    "feature_group": (
                        row.get(
                            "feature_group"
                        )
                    ),
                    "mean_absolute_shap": (
                        safe_float(
                            row.get(
                                "mean_absolute_shap"
                            )
                        )
                    ),
                    "mean_signed_shap": (
                        safe_float(
                            row.get(
                                "mean_signed_shap"
                            )
                        )
                    ),
                    "positive_effect_fraction": (
                        safe_float(
                            row.get(
                                "positive_effect_fraction"
                            )
                        )
                    ),
                    "negative_effect_fraction": (
                        safe_float(
                            row.get(
                                "negative_effect_fraction"
                            )
                        )
                    ),
                }
            )

        return rows

    return {
        "available": True,
        "analysis_scope": "ALERT_EPOCHS",
        "group_importance": (
            group_rows
        ),
        "risk_increasing_factors": (
            convert_feature_rows(
                increasing_df
            )
        ),
        "risk_decreasing_factors": (
            convert_feature_rows(
                decreasing_df
            )
        ),
        "interpretation_note": (
            "此區塊描述模型在 ALERT_EPOCHS 中的患者層級 SHAP 影響。"
            "正向 SHAP 表示推高模型 Arousal 警報輸出，"
            "負向 SHAP 表示降低模型警報輸出；"
            "不代表生理因果。"
        ),
    }


def extract_treatment_result(
    treatment_json: dict[str, Any],
    follow_up_mode: bool = False,
) -> dict[str, Any]:
    ranking = treatment_json.get(
        "personalized_treatment_ranking",
        [],
    )

    if not isinstance(
        ranking,
        list,
    ):
        ranking = []

    ranking = [
        item
        for item in ranking
        if isinstance(
            item,
            dict,
        )
    ]

    ranking = sorted(
        ranking,
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
            else 999
        ),
    )

    recommendation_summary = (
        treatment_json.get(
            "recommendation_summary",
            {}
        )
    )

    if not isinstance(
        recommendation_summary,
        dict,
    ):
        recommendation_summary = {}

    prerequisites = treatment_json.get(
        "clinical_prerequisites",
        [],
    )

    if not isinstance(
        prerequisites,
        list,
    ):
        prerequisites = []

    first_candidate = (
        ranking[0]
        if ranking
        else {}
    )

    second_candidate = (
        ranking[1]
        if len(ranking) >= 2
        else {}
    )

    return {
        "available": bool(
            treatment_json
        ),
        "first_candidate": first_non_none(
            recommendation_summary.get(
                "recommended_first_candidate"
            ),
            first_candidate.get(
                "treatment"
            ),
        ),
        "first_candidate_label": first_non_none(
            recommendation_summary.get(
                "recommended_first_candidate_label"
            ),
            first_candidate.get(
                "treatment_label"
            ),
        ),
        "first_candidate_score": safe_float(
            first_non_none(
                recommendation_summary.get(
                    "first_candidate_score"
                ),
                first_candidate.get(
                    "score"
                ),
            )
        ),
        "first_candidate_status": first_non_none(
            recommendation_summary.get(
                "first_candidate_status"
            ),
            first_candidate.get(
                "recommendation_status"
            ),
        ),
        "first_candidate_confidence": first_non_none(
            recommendation_summary.get(
                "first_candidate_confidence"
            ),
            get_nested(
                first_candidate,
                "confidence",
                "confidence_level",
            ),
        ),
        "second_candidate": first_non_none(
            recommendation_summary.get(
                "second_candidate"
            ),
            second_candidate.get(
                "treatment"
            ),
        ),
        "second_candidate_score": safe_float(
            second_candidate.get(
                "score"
            )
        ),
        "score_gap_to_second": safe_float(
            recommendation_summary.get(
                "score_gap_to_second"
            )
        ),
        "unique_first_choice": safe_bool(
            recommendation_summary.get(
                "unique_first_choice"
            )
        ),
        "first_candidate_is_tie": safe_bool(
            recommendation_summary.get(
                "first_candidate_is_tie"
            )
        ),
        "co_first_candidates": (
            recommendation_summary.get(
                "co_first_candidates",
                [],
            )
            if isinstance(
                recommendation_summary.get(
                    "co_first_candidates",
                    [],
                ),
                list,
            )
            else []
        ),
        "ranking": ranking,
        "clinical_prerequisites": (
            prerequisites
        ),
        "safety_note": (
            treatment_json.get(
                "safety_note"
            )
        ),
        "medical_decision_basis": (
            treatment_json.get(
                "medical_decision_basis",
                {},
            )
            if isinstance(
                treatment_json.get(
                    "medical_decision_basis",
                    {},
                ),
                dict,
            )
            else {}
        ),
        "clinical_feature_ingestion": (
            treatment_json.get(
                "clinical_feature_ingestion",
                {},
            )
            if isinstance(
                treatment_json.get(
                    "clinical_feature_ingestion",
                    {},
                ),
                dict,
            )
            else {}
        ),
        "recommendation_history": (
            treatment_json.get(
                "recommendation_history",
                [],
            )
            if isinstance(
                treatment_json.get(
                    "recommendation_history",
                    [],
                ),
                list,
            )
            else []
        ),
        "refinement_run": (
            treatment_json.get(
                "refinement_run",
                {},
            )
            if isinstance(
                treatment_json.get(
                    "refinement_run",
                    {},
                ),
                dict,
            )
            else {}
        ),
    }


# ============================================================
# 整合患者資料
# ============================================================

def build_integrated_report(
    patient_id: str,
    respiratory_json: dict[str, Any],
    factor_json: dict[str, Any],
    factor_csv: pd.DataFrame,
    spo2_quality_json: dict[str, Any],
    oxygen_json: dict[str, Any],
    position_json: dict[str, Any],
    shap_group_csv: pd.DataFrame,
    shap_feature_csv: pd.DataFrame,
    arousal_early_warning_json: dict[str, Any],
    representative_case: dict[str, Any],
    representative_timeline: dict[str, Any],
    representative_timeline_window: list[dict[str, Any]],
    treatment_json: dict[str, Any],
    follow_up_mode: bool = False,
) -> dict[str, Any]:
    event_summary = get_nested(
        respiratory_json,
        "event_summary",
        default={},
    )

    spo2_profile = get_nested(
        respiratory_json,
        "spo2_profile",
        default={},
    )

    position_profile = get_nested(
        respiratory_json,
        "position_profile",
        default={},
    )

    rem_profile = get_nested(
        respiratory_json,
        "rem_profile",
        default={},
    )

    loop_gain_profile = get_nested(
        respiratory_json,
        "loop_gain_proxy",
        default={},
    )

    mechanism_profile = get_nested(
        respiratory_json,
        "mechanism_profile",
        default={},
    )

    low_oxygen_attribution = get_nested(
        oxygen_json,
        "low_oxygen_attribution",
        default={},
    )

    event_coupling = get_nested(
        oxygen_json,
        "event_coupling",
        default={},
    )

    wake_sleep_oxygen = get_nested(
        oxygen_json,
        "wake_sleep_oxygen",
        default={},
    )

    factor_summary = extract_factor_summary(
        factor_json=factor_json,
        factor_csv=factor_csv,
    )

    arousal_shap_summary = (
        extract_arousal_shap_summary(
            shap_group_csv=shap_group_csv,
            shap_feature_csv=shap_feature_csv,
        )
    )

    early_warning_data_counts = get_nested(
        arousal_early_warning_json,
        "data_counts",
        default={},
    )

    early_warning_event_level = get_nested(
        arousal_early_warning_json,
        "event_level_warning",
        default={},
    )

    early_warning_alert_level = get_nested(
        arousal_early_warning_json,
        "alert_level_performance",
        default={},
    )

    early_warning_lead_seconds = get_nested(
        early_warning_event_level,
        "lead_seconds",
        default={},
    )

    arousal_early_warning_summary = {
        "available": bool(
            arousal_early_warning_json
        ),
        "model_target": (
            arousal_early_warning_json.get(
                "model_target"
            )
        ),
        "prediction_horizon_seconds": safe_float(
            arousal_early_warning_json.get(
                "prediction_horizon_seconds"
            )
        ),
        "true_arousal_count": safe_int(
            early_warning_data_counts.get(
                "true_arousal_count"
            )
        ),
        "monitoring_hours": safe_float(
            early_warning_data_counts.get(
                "monitoring_hours"
            )
        ),
        "successfully_warned_count": safe_int(
            early_warning_event_level.get(
                "successfully_warned_count"
            )
        ),
        "missed_event_count": safe_int(
            early_warning_event_level.get(
                "missed_event_count"
            )
        ),
        "warning_recall": safe_float(
            early_warning_event_level.get(
                "warning_recall"
            )
        ),
        "lead_seconds_mean": safe_float(
            early_warning_lead_seconds.get(
                "mean"
            )
        ),
        "lead_seconds_median": safe_float(
            early_warning_lead_seconds.get(
                "median"
            )
        ),
        "lead_seconds_min": safe_float(
            early_warning_lead_seconds.get(
                "minimum"
            )
        ),
        "lead_seconds_max": safe_float(
            early_warning_lead_seconds.get(
                "maximum"
            )
    ),
        "alert_threshold": safe_float(
            early_warning_alert_level.get(
                "alert_threshold"
            )
        ),
        "alert_epoch_count": safe_int(
            early_warning_alert_level.get(
                "alert_epoch_count"
            )
        ),
        "true_alert_epoch_count": safe_int(
            early_warning_alert_level.get(
                "true_alert_epoch_count"
            )
        ),
        "false_alert_epoch_count": safe_int(
            early_warning_alert_level.get(
                "false_alert_epoch_count"
            )
        ),
        "false_alert_episode_count": safe_int(
            early_warning_alert_level.get(
                "false_alert_episode_count"
            )
        ),
        "alert_epoch_positive_predictive_value": safe_float(
            early_warning_alert_level.get(
                "alert_epoch_positive_predictive_value"
            )
        ),
        "false_alert_epochs_per_hour": safe_float(
            early_warning_alert_level.get(
                "false_alert_epochs_per_hour"
            )
        ),
        "false_alert_episodes_per_hour": safe_float(
            early_warning_alert_level.get(
                "false_alert_episodes_per_hour"
            )
        ),
        "subtype_results": (
            arousal_early_warning_json.get(
                "subtype_results",
                [],
            )
        ),
        "interpretation": (
            arousal_early_warning_json.get(
                "interpretation"
            )
        ),
        "safety_note": (
            arousal_early_warning_json.get(
                "safety_note"
            )
        ),
    }

    treatment_result = extract_treatment_result(
        treatment_json
    )

    patient_information = {
        "patient_id": patient_id,
        "sex": first_non_none(
            respiratory_json.get(
                "sex"
            ),
            treatment_json.get(
                "patient_context",
                {}
            ).get(
                "sex"
            )
            if isinstance(
                treatment_json.get(
                    "patient_context",
                    {}
                ),
                dict,
            )
            else None,
        ),
        "age": safe_float(
            first_non_none(
                respiratory_json.get(
                    "age"
                ),
                get_nested(
                    treatment_json,
                    "patient_context",
                    "age",
                ),
            )
        ),
        "BMI": safe_float(
            first_non_none(
                respiratory_json.get(
                    "BMI"
                ),
                get_nested(
                    treatment_json,
                    "patient_context",
                    "BMI",
                ),
            )
        ),
        "sleep_minutes": safe_float(
            first_non_none(
                get_nested(
                    respiratory_json,
                    "sleep_summary",
                    "sleep_minutes",
                ),
                respiratory_json.get(
                    "sleep_minutes"
                ),
                event_summary.get(
                    "sleep_minutes"
                ),
            )
        ),
    }

    respiratory_phenotype = {
        "ahi": safe_float(
            event_summary.get(
                "ahi"
            )
        ),
        "ahi_severity": (
            event_summary.get(
                "ahi_severity"
            )
        ),
        "respiratory_event_count": (
            safe_int(
                event_summary.get(
                    "respiratory_event_count"
                )
            )
        ),

        "arousal_count": safe_int(
            event_summary.get(
                "arousal_count"
            )
        ),
        "arousal_index": safe_float(
            event_summary.get(
                "arousal_index"
            )
        ),
        "respiratory_arousal_index": safe_float(
            event_summary.get(
                "respiratory_arousal_index"
            )
        ),

        "hypopnea_count": safe_int(
            event_summary.get(
                "hypopnea_count"
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
        "mixed_apnea_count": (
            safe_int(
                event_summary.get(
                    "mixed_apnea_count"
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
        "central_event_fraction": (
            safe_float(
                event_summary.get(
                    "central_event_fraction"
                )
            )
        ),
        "rem_relevance": first_non_none(
            rem_profile.get(
                "rem_relevance"
            ),
            get_nested(
                treatment_json,
                "patient_context",
                "rem_relevance",
            ),
        ),
        "position_relevance": first_non_none(
            get_nested(
                treatment_json,
                "patient_context",
                "position_relevance",
            ),
            position_json.get(
                "position_relevance"
            ),
            position_profile.get(
                "position_relevance"
            ),
        ),
        "position_event_ratio": safe_float(
            get_nested(
                treatment_json,
                "patient_context",
                "position_scoring_ratio",
            )
            if treatment_json
            else first_non_none(
                position_json.get(
                    "position_event_index_ratio"
                ),
            )
        ),
        "known_position_fraction": safe_float(
            first_non_none(
                position_json.get(
                    "known_position_epoch_fraction"
                ),
                get_nested(
                    treatment_json,
                    "patient_context",
                    "known_position_fraction",
                ),
            )
        ),
        "loop_gain_proxy_score": safe_float(
            loop_gain_profile.get(
                "loop_gain_proxy_score"
            )
        ),
        "loop_gain_proxy_level": first_non_none(
            loop_gain_profile.get(
                "loop_gain_proxy_level"
            ),
            get_nested(
                treatment_json,
                "patient_context",
                "loop_gain_level",
            ),
        ),
        "dominant_mechanism": first_non_none(
            mechanism_profile.get(
                "dominant_mechanism"
            ),
            get_nested(
                treatment_json,
                "patient_context",
                "dominant_mechanism",
            ),
        ),

        "mechanism_ranking": (
            mechanism_profile.get(
                "mechanism_ranking",
            [],
            )
            if isinstance(
                mechanism_profile.get(
                    "mechanism_ranking",
                [],
                ),
                list,
            )
            else []
        ),
    }

    oxygen_phenotype = {
        "spo2_mean": safe_float(
            spo2_profile.get(
                "spo2_mean"
            )
        ),
        "spo2_median": safe_float(
            spo2_profile.get(
                "spo2_median"
            )
        ),
        "robust_spo2_min": safe_float(
            spo2_profile.get(
                "spo2_min"
            )
        ),
        "observed_absolute_min": safe_float(
            spo2_profile.get(
                "spo2_observed_absolute_min"
            )
        ),
        "spo2_below_90_fraction": safe_float(
            spo2_profile.get(
                "spo2_below_90_fraction"
            )
        ),
        "spo2_below_88_fraction": safe_float(
            spo2_profile.get(
                "spo2_below_88_fraction"
            )
        ),
        "desaturation_depth": (
            spo2_profile.get(
                "desaturation_depth"
            )
        ),
        "sustained_low_oxygen_burden": (
            spo2_profile.get(
                "sustained_low_oxygen_burden"
            )
        ),
        "hypoxemia_level": (
            spo2_profile.get(
                "hypoxemia_level"
            )
        ),
        "spo2_quality_valid_fraction": safe_float(
            first_non_none(
                spo2_quality_json.get(
                    "valid_fraction"
                ),
                get_nested(
                    spo2_quality_json,
                    "quality_counts",
                    "valid_fraction",
                ),
                get_nested(
                    spo2_quality_json,
                    "quality_summary",
                    "valid_fraction",
                ),
            )
        ),
        "spo2_quality_suspect_fraction": safe_float(
            first_non_none(
                spo2_quality_json.get(
                    "suspect_fraction"
                ),
                get_nested(
                    spo2_quality_json,
                    "quality_counts",
                    "suspect_fraction",
                ),
            )
        ),
        "spo2_quality_invalid_fraction": safe_float(
            first_non_none(
                spo2_quality_json.get(
                    "invalid_fraction"
                ),
                get_nested(
                    spo2_quality_json,
                    "quality_counts",
                    "invalid_fraction",
                ),
            )
        ),
        "oxygen_coupling_level": (
            oxygen_json.get(
                "coupling_level"
            )
        ),
        "coupled_3pct_fraction": safe_float(
            first_non_none(
                event_coupling.get(
                    "coupled_3pct_fraction"
                ),
                oxygen_json.get(
                    "coupled_3pct_fraction"
                ),
            )
        ),
        "coupled_4pct_fraction": safe_float(
            first_non_none(
                event_coupling.get(
                    "coupled_4pct_fraction"
                ),
                oxygen_json.get(
                    "coupled_4pct_fraction"
                ),
            )
        ),
        "low_oxygen_total_minutes": safe_float(
            low_oxygen_attribution.get(
                "spo2_below_90_minutes"
            )
        ),
        "low_oxygen_near_event_minutes": safe_float(
            low_oxygen_attribution.get(
                "spo2_below_90_near_event_minutes"
            )
        ),
        "low_oxygen_away_from_event_minutes": (
            safe_float(
                low_oxygen_attribution.get(
                    "spo2_below_90_away_from_event_minutes"
                )
            )
        ),
        "low_oxygen_near_event_fraction": safe_float(
            low_oxygen_attribution.get(
                "spo2_below_90_near_event_fraction"
            )
        ),
        "low_oxygen_away_from_event_fraction": (
            safe_float(
                low_oxygen_attribution.get(
                    "spo2_below_90_away_from_event_fraction"
                )
            )
        ),
        "wake_spo2_mean": safe_float(
            wake_sleep_oxygen.get(
                "wake_spo2_mean"
            )
        ),
        "sleep_spo2_mean": safe_float(
            wake_sleep_oxygen.get(
                "sleep_spo2_mean"
            )
        ),
    }

    refinement_run = treatment_result.get(
        "refinement_run",
        {},
    )

    if not isinstance(
        refinement_run,
        dict,
    ):
        refinement_run = {}

    inferred_follow_up = (
        str(
            refinement_run.get(
                "mode",
                "",
            )
        ).strip().upper()
        == "PSG_FOLLOW_UP"
    )

    effective_follow_up_mode = bool(
        follow_up_mode
        or inferred_follow_up
    )

    report_metadata = {
        "report_mode": (
            "PSG_FOLLOW_UP"
            if effective_follow_up_mode
            else "INITIAL_OR_STANDARD"
        ),
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "patient_id": patient_id,
        "report_schema_version": "2.0",
        "psg_derived_data_refreshed": bool(
            effective_follow_up_mode
        ),
        "accumulated_clinical_data_preserved": (
            refinement_run.get(
                "clinical_data_preserved"
            )
        ),
        "treatment_refinement_run": (
            refinement_run
        ),
        "recommendation_history_count": len(
            treatment_result.get(
                "recommendation_history",
                [],
            )
        ),
    }

    report = {
        "patient_id": patient_id,
        "report_type": (
            "PATIENT_DIGITAL_TWIN_REPORT"
        ),
        "report_metadata": (
            report_metadata
        ),
        "recommendation_history": (
            treatment_result.get(
                "recommendation_history",
                [],
            )
        ),
        "patient_information": (
            patient_information
        ),
        "respiratory_phenotype": (
            respiratory_phenotype
        ),
        "oxygen_phenotype": (
            oxygen_phenotype
        ),
        "model_explanation": (
            factor_summary
        ),

        "arousal_risk_explanation": (
            arousal_shap_summary
        ),

        "arousal_early_warning": (
            arousal_early_warning_summary
        ),

        "representative_early_warning_case": (
            representative_case
        ),

        "representative_early_warning_timeline": (
            representative_timeline
        ),

        "representative_early_warning_timeline_window": (
            representative_timeline_window
        ),


        "personalized_treatment": (
            treatment_result
        ),
        "source_availability": {
            "respiratory_profile": bool(
                respiratory_json
            ),
            "factor_summary": bool(
                factor_json
                or not factor_csv.empty
            ),
            "spo2_quality": bool(
                spo2_quality_json
            ),
            "oxygen_event_coupling": bool(
                oxygen_json
            ),
            "position_profile": bool(
                position_json
            ),

            "arousal_early_warning": bool(
                arousal_early_warning_json
            ),

            "treatment_recommendation": bool(
                treatment_json
            ),
        },
        "safety_note": (
            "本報告為研究型患者數位孿生整合摘要，"
            "不可單獨作為正式臨床診斷、處方或治療決策。"
        ),
    }

    # ========================================================
    # 同步 Treatment Refinement Follow-up 資料
    # ========================================================

    personalized_treatment = report.get(
        "personalized_treatment",
        {},
    )

    if not isinstance(
        personalized_treatment,
        dict,
    ):
        personalized_treatment = {}

    if isinstance(
        treatment_json,
        dict,
    ):
        follow_up_comparison = treatment_json.get(
            "follow_up_comparison",
            {},
        )

        recommendation_history = treatment_json.get(
            "recommendation_history",
            [],
        )

        refinement_run = treatment_json.get(
            "refinement_run",
            {},
        )

        personalized_treatment[
            "follow_up_comparison"
        ] = (
            follow_up_comparison
            if isinstance(
                follow_up_comparison,
                dict,
            )
            else {}
        )

        personalized_treatment[
            "recommendation_history"
        ] = (
            recommendation_history
            if isinstance(
                recommendation_history,
                list,
            )
            else []
        )

        personalized_treatment[
            "refinement_run"
        ] = (
            refinement_run
            if isinstance(
                refinement_run,
                dict,
            )
            else {}
        )

    report[
        "personalized_treatment"
    ] = personalized_treatment

    report[
        "digital_twin_narrative"
    ] = build_digital_twin_narrative(
        report
    )

    return report


# ============================================================
# 自動文字摘要
# ============================================================

def build_digital_twin_narrative(
    report: dict[str, Any],
) -> str:
    patient = report[
        "patient_information"
    ]

    respiratory = report[
        "respiratory_phenotype"
    ]

    oxygen = report[
        "oxygen_phenotype"
    ]

    model = report[
        "model_explanation"
    ]

    treatment = report[
        "personalized_treatment"
    ]

    patient_id = normalize_text(
        patient.get(
            "patient_id"
        )
    )

    lines: list[str] = []

    report_metadata = report.get(
        "report_metadata",
        {},
    )

    if not isinstance(
        report_metadata,
        dict,
    ):
        report_metadata = {}

    report_mode = normalize_text(
        report_metadata.get(
            "report_mode"
        )
    )

    if report_mode == "PSG_FOLLOW_UP":
        lines.append(
            "本報告為 PSG Follow-up 更新："
            "本次重新整理 PSG 衍生生理表型，"
            "並沿用既有累積臨床資料與治療精煉資訊。"
        )

    lines.append(
        f"患者 {patient_id} 的數位孿生整合摘要如下。"
    )

    demographic_parts: list[str] = []

    if patient.get(
        "sex"
    ) is not None:
        demographic_parts.append(
            f"sex={patient.get('sex')}"
        )

    if patient.get(
        "age"
    ) is not None:
        demographic_parts.append(
            "年齡="
            f"{format_number(patient.get('age'), 1)} 歲"
        )

    if patient.get(
        "BMI"
    ) is not None:
        demographic_parts.append(
            "BMI="
            f"{format_number(patient.get('BMI'), 1)}"
        )

    if demographic_parts:
        lines.append(
            "基本資料："
            + "、".join(
                demographic_parts
            )
            + "。"
        )

    ahi = respiratory.get(
        "ahi"
    )

    severity = normalize_text(
        respiratory.get(
            "ahi_severity"
        )
    )

    if ahi is not None:
        lines.append(
            "睡眠呼吸事件："
            f"AHI={format_number(ahi, 2)}，"
            f"嚴重度={severity}；"
            f"Hypopnea="
            f"{normalize_text(respiratory.get('hypopnea_count'))}、"
            f"Obstructive Apnea="
            f"{normalize_text(respiratory.get('obstructive_apnea_count'))}、"
            f"Central Apnea="
            f"{normalize_text(respiratory.get('central_apnea_count'))}。"
        )

    rem_relevance = normalize_text(
        respiratory.get(
            "rem_relevance"
        )
    )

    position_relevance = normalize_text(
        respiratory.get(
            "position_relevance"
        )
    )

    position_ratio = respiratory.get(
        "position_event_ratio"
    )

    phenotype_sentence = (
        "呼吸表型："
        f"REM 相關性={rem_relevance}，"
        f"姿勢相關性={position_relevance}"
    )

    if position_ratio is not None:
        phenotype_sentence += (
            "，姿勢群組最高與最低事件率"
            f"約相差 {format_number(position_ratio, 2)} 倍"
        )

    phenotype_sentence += (
        "；Loop Gain proxy="
        f"{normalize_text(respiratory.get('loop_gain_proxy_level'))}，"
        "主要研究型機轉="
        f"{normalize_text(respiratory.get('dominant_mechanism'))}。"
    )

    lines.append(
        phenotype_sentence
    )

    oxygen_sentence = (
        "血氧表型："
        "平均 SpO₂="
        f"{format_number(oxygen.get('spo2_mean'), 2, '%')}，"
        "穩健最低 SpO₂="
        f"{format_number(oxygen.get('robust_spo2_min'), 2, '%')}，"
        "血氧下降深度="
        f"{normalize_text(oxygen.get('desaturation_depth'))}，"
        "持續性低血氧負荷="
        f"{normalize_text(oxygen.get('sustained_low_oxygen_burden'))}，"
        "總體血氧負荷="
        f"{normalize_text(oxygen.get('hypoxemia_level'))}。"
    )

    lines.append(
        oxygen_sentence
    )

    coupling_level = normalize_text(
        oxygen.get(
            "oxygen_coupling_level"
        )
    )

    coupled_fraction = oxygen.get(
        "coupled_3pct_fraction"
    )

    away_fraction = oxygen.get(
        "low_oxygen_away_from_event_fraction"
    )

    coupling_parts = [
        "低氧－事件關係："
        f"耦合等級={coupling_level}"
    ]

    if coupled_fraction is not None:
        coupling_parts.append(
            "有效呼吸事件中"
            f"{format_percentage(coupled_fraction)} "
            "伴隨至少 3% 血氧下降"
        )

    if away_fraction is not None:
        coupling_parts.append(
            "SpO₂ <90% 時間中"
            f"{format_percentage(away_fraction)} "
            "遠離已標記呼吸事件"
        )

    lines.append(
        "；".join(
            coupling_parts
        )
        + "。"
    )

    wake_spo2 = oxygen.get(
        "wake_spo2_mean"
    )

    sleep_spo2 = oxygen.get(
        "sleep_spo2_mean"
    )

    if (
        wake_spo2 is not None
        or sleep_spo2 is not None
    ):
        lines.append(
            "清醒與睡眠血氧："
            "PSG 清醒期平均 SpO₂="
            f"{format_number(wake_spo2, 2, '%')}，"
            "睡眠期平均 SpO₂="
            f"{format_number(sleep_spo2, 2, '%')}。"
        )

    top_groups = model.get(
        "top_physiological_groups",
        [],
    )
    observed_factor_summary = (
        model.get("explanation_method")
        == "OBSERVED_PSG_PHENOTYPE_AND_RULE_CONTRIBUTIONS"
    )

    if isinstance(
        top_groups,
        list,
    ) and top_groups:
        group_prefix = (
            "目前觀察到的主要生理表型包括："
            if observed_factor_summary
            else "模型整體較關注的生理系統包括："
        )
        lines.append(
            group_prefix
            + f"{join_text([str(item) for item in top_groups[:5]])}。"
        )

    increasing = model.get(
        "risk_increasing_factors",
        [],
    )

    if isinstance(
        increasing,
        list,
    ) and increasing:
        increasing_names = [
            normalize_text(
                item.get(
                    "feature"
                )
            )
            for item in increasing[:5]
            if isinstance(
                item,
                dict,
            )
        ]

        if increasing_names:
            factor_prefix = (
                "目前已觀察到的睡眠呼吸風險因素包括："
                if observed_factor_summary
                else "警報期間主要提高模型風險的因素包括："
            )
            lines.append(
                factor_prefix
                + f"{join_text(increasing_names)}。"
            )

    first_candidate = normalize_text(
        treatment.get(
            "first_candidate"
        )
    )

    first_label = normalize_text(
        treatment.get(
            "first_candidate_label"
        )
    )

    second_candidate = normalize_text(
        treatment.get(
            "second_candidate"
        )
    )

    co_first_candidates = treatment.get(
        "co_first_candidates",
        [],
    )
    first_candidate_is_tie = bool(
        treatment.get("first_candidate_is_tie")
        and isinstance(co_first_candidates, list)
        and len(co_first_candidates) > 1
    )

    if first_candidate_is_tie:
        co_first_labels = [
            normalize_text(
                item.get("treatment_label")
                or item.get("treatment")
            )
            for item in co_first_candidates
            if isinstance(item, dict)
        ]
        lines.append(
            "依目前可觀察資料，"
            f"{'、'.join(co_first_labels)}並列第一治療候選，"
            f"分數均為{format_number(treatment.get('first_candidate_score'), 1)}；"
            "目前證據不足以判定其中一項優於另一項。"
        )
    elif first_candidate != "NOT_AVAILABLE":
        lines.append(
            "依目前可觀察資料，"
            f"{first_candidate}（{first_label}）"
            "為第一治療候選，"
            f"分數={format_number(treatment.get('first_candidate_score'), 1)}，"
            "推薦狀態="
            f"{normalize_text(treatment.get('first_candidate_status'))}，"
            "推薦信心="
            f"{normalize_text(treatment.get('first_candidate_confidence'))}；"
            f"第二候選為 {second_candidate}。"
        )

    prerequisites = treatment.get(
        "clinical_prerequisites",
        [],
    )

    if isinstance(
        prerequisites,
        list,
    ) and prerequisites:
        first_prerequisite = prerequisites[0]

        if isinstance(
            first_prerequisite,
            dict,
        ):
            lines.append(
                "最高優先治療前置事項為："
                f"{normalize_text(first_prerequisite.get('title'))}。"
            )

    lines.append(
        "以上為研究型模型整合結果；"
        "特徵關聯不等於生理因果，"
        "治療候選亦不代表可直接開始治療。"
    )

    return "\n".join(
        lines
    )


# ============================================================
# CSV 平坦化
# ============================================================

def flatten_report_for_csv(
    report: dict[str, Any],
) -> dict[str, Any]:
    patient = report[
        "patient_information"
    ]

    respiratory = report[
        "respiratory_phenotype"
    ]

    oxygen = report[
        "oxygen_phenotype"
    ]

    model = report[
        "model_explanation"
    ]

    treatment = report[
        "personalized_treatment"
    ]

    top_groups = model.get(
        "top_physiological_groups",
        [],
    )

    increasing = model.get(
        "risk_increasing_factors",
        [],
    )

    decreasing = model.get(
        "risk_decreasing_factors",
        [],
    )

    prerequisites = treatment.get(
        "clinical_prerequisites",
        [],
    )

    report_metadata = report.get(
        "report_metadata",
        {},
    )

    if not isinstance(
        report_metadata,
        dict,
    ):
        report_metadata = {}

    row: dict[str, Any] = {
        "patient_id": patient.get(
            "patient_id"
        ),
        "report_mode": report_metadata.get(
            "report_mode"
        ),
        "report_generated_at": report_metadata.get(
            "generated_at"
        ),
        "recommendation_history_count": report_metadata.get(
            "recommendation_history_count"
        ),
        "sex": patient.get(
            "sex"
        ),
        "age": patient.get(
            "age"
        ),
        "BMI": patient.get(
            "BMI"
        ),
        "sleep_minutes": patient.get(
            "sleep_minutes"
        ),
        **respiratory,
        **oxygen,
        "first_treatment_candidate": (
            treatment.get(
                "first_candidate"
            )
        ),
        "first_treatment_candidate_label": (
            treatment.get(
                "first_candidate_label"
            )
        ),
        "first_treatment_score": (
            treatment.get(
                "first_candidate_score"
            )
        ),
        "first_treatment_status": (
            treatment.get(
                "first_candidate_status"
            )
        ),
        "first_treatment_confidence": (
            treatment.get(
                "first_candidate_confidence"
            )
        ),
        "second_treatment_candidate": (
            treatment.get(
                "second_candidate"
            )
        ),
        "second_treatment_score": (
            treatment.get(
                "second_candidate_score"
            )
        ),
        "score_gap_to_second": (
            treatment.get(
                "score_gap_to_second"
            )
        ),
        "unique_first_choice": (
            treatment.get(
                "unique_first_choice"
            )
        ),
        "model_roc_auc": get_nested(
            model,
            "model_performance",
            "roc_auc",
        ),
        "model_recall": get_nested(
            model,
            "model_performance",
            "recall",
        ),
        "model_specificity": get_nested(
            model,
            "model_performance",
            "specificity",
        ),
        "model_precision": get_nested(
            model,
            "model_performance",
            "precision",
        ),
        "digital_twin_narrative": (
            report.get(
                "digital_twin_narrative"
            )
        ),
    }

    for index in range(
        5
    ):
        group_value = (
            top_groups[index]
            if (
                isinstance(
                    top_groups,
                    list,
                )
                and index < len(
                    top_groups
                )
            )
            else None
        )

        row[
            f"top_physiological_group_{index + 1}"
        ] = group_value

    for index in range(
        5
    ):
        increasing_item = (
            increasing[index]
            if (
                isinstance(
                    increasing,
                    list,
                )
                and index < len(
                    increasing
                )
                and isinstance(
                    increasing[index],
                    dict,
                )
            )
            else {}
        )

        row[
            f"risk_increasing_factor_{index + 1}"
        ] = increasing_item.get(
            "feature"
        )

        decreasing_item = (
            decreasing[index]
            if (
                isinstance(
                    decreasing,
                    list,
                )
                and index < len(
                    decreasing
                )
                and isinstance(
                    decreasing[index],
                    dict,
                )
            )
            else {}
        )

        row[
            f"risk_decreasing_factor_{index + 1}"
        ] = decreasing_item.get(
            "feature"
        )

    for index in range(
        4
    ):
        prerequisite = (
            prerequisites[index]
            if (
                isinstance(
                    prerequisites,
                    list,
                )
                and index < len(
                    prerequisites
                )
                and isinstance(
                    prerequisites[index],
                    dict,
                )
            )
            else {}
        )

        row[
            f"clinical_prerequisite_{index + 1}"
        ] = prerequisite.get(
            "title"
        )

    return row


# ============================================================
# 純文字報告
# ============================================================

def build_text_report(
    report: dict[str, Any],
) -> str:
    patient = report[
        "patient_information"
    ]

    respiratory = report[
        "respiratory_phenotype"
    ]

    oxygen = report[
        "oxygen_phenotype"
    ]

    model = report[
        "model_explanation"
    ]

    treatment = report[
        "personalized_treatment"
    ]

    lines: list[str] = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "患者睡眠數位孿生整合報告"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Patient ID：{patient.get('patient_id')}"
    )

    lines.append(
        ""
    )

    report_metadata = report.get(
        "report_metadata",
        {},
    )

    if not isinstance(
        report_metadata,
        dict,
    ):
        report_metadata = {}

    lines.append(
        "【報告資訊】"
    )

    lines.append(
        "報告模式："
        f"{normalize_text(report_metadata.get('report_mode'))}"
    )

    lines.append(
        "產生時間："
        f"{normalize_text(report_metadata.get('generated_at'))}"
    )

    lines.append(
        "歷史推薦筆數："
        f"{normalize_text(report_metadata.get('recommendation_history_count'))}"
    )

    lines.append(
        ""
    )

    lines.append(
        "【一、患者基本資訊】"
    )

    lines.append(
        f"sex：{normalize_text(patient.get('sex'))}"
    )

    lines.append(
        f"年齡：{format_number(patient.get('age'), 1)} 歲"
    )

    lines.append(
        f"BMI：{format_number(patient.get('BMI'), 1)}"
    )

    lines.append(
        "總睡眠時間："
        f"{format_number(patient.get('sleep_minutes'), 1)} 分鐘"
    )

    lines.append(
        ""
    )

    lines.append(
        "【二、睡眠呼吸表型】"
    )

    lines.append(
        f"AHI：{format_number(respiratory.get('ahi'), 2)}"
    )

    lines.append(
        "AHI 嚴重度："
        f"{normalize_text(respiratory.get('ahi_severity'))}"
    )

    lines.append(
        "呼吸事件總數："
        f"{normalize_text(respiratory.get('respiratory_event_count'))}"
    )

    lines.append(
        "Hypopnea："
        f"{normalize_text(respiratory.get('hypopnea_count'))}"
    )

    lines.append(
        "Obstructive Apnea："
        f"{normalize_text(respiratory.get('obstructive_apnea_count'))}"
    )

    lines.append(
        "Central Apnea："
        f"{normalize_text(respiratory.get('central_apnea_count'))}"
    )

    lines.append(
        "REM 相關性："
        f"{normalize_text(respiratory.get('rem_relevance'))}"
    )

    lines.append(
        "姿勢相關性："
        f"{normalize_text(respiratory.get('position_relevance'))}"
    )

    lines.append(
        "姿勢事件率比值："
        f"{format_number(respiratory.get('position_event_ratio'), 2)}"
    )

    lines.append(
        "Loop Gain proxy："
        f"{normalize_text(respiratory.get('loop_gain_proxy_level'))}"
    )

    lines.append(
        "主要研究型機轉："
        f"{normalize_text(respiratory.get('dominant_mechanism'))}"
    )

    lines.append(
        ""
    )

    lines.append(
        "【三、血氧表型】"
    )

    lines.append(
        "平均 SpO₂："
        f"{format_number(oxygen.get('spo2_mean'), 2, '%')}"
    )

    lines.append(
        "穩健最低 SpO₂："
        f"{format_number(oxygen.get('robust_spo2_min'), 2, '%')}"
    )

    lines.append(
        "SpO₂ <90% 比例："
        f"{format_percentage(oxygen.get('spo2_below_90_fraction'))}"
    )

    lines.append(
        "SpO₂ <88% 比例："
        f"{format_percentage(oxygen.get('spo2_below_88_fraction'))}"
    )

    lines.append(
        "血氧下降深度："
        f"{normalize_text(oxygen.get('desaturation_depth'))}"
    )

    lines.append(
        "持續性低血氧負荷："
        f"{normalize_text(oxygen.get('sustained_low_oxygen_burden'))}"
    )

    lines.append(
        "總體血氧負荷："
        f"{normalize_text(oxygen.get('hypoxemia_level'))}"
    )

    lines.append(
        "低氧－事件耦合："
        f"{normalize_text(oxygen.get('oxygen_coupling_level'))}"
    )

    lines.append(
        "呼吸事件伴隨 ≥3% 血氧下降比例："
        f"{format_percentage(oxygen.get('coupled_3pct_fraction'))}"
    )

    lines.append(
        "低氧遠離事件比例："
        f"{format_percentage(oxygen.get('low_oxygen_away_from_event_fraction'))}"
    )

    lines.append(
        "低氧遠離事件時間："
        f"{format_number(oxygen.get('low_oxygen_away_from_event_minutes'), 1)} 分鐘"
    )

    lines.append(
        "PSG 清醒期平均 SpO₂："
        f"{format_number(oxygen.get('wake_spo2_mean'), 2, '%')}"
    )

    lines.append(
        "睡眠期平均 SpO₂："
        f"{format_number(oxygen.get('sleep_spo2_mean'), 2, '%')}"
    )

    lines.append(
        ""
    )

    lines.append(
        "【四、模型解釋】"
    )

    top_groups = model.get(
        "top_physiological_groups",
        [],
    )

    lines.append(
        "模型較關注的生理系統："
        f"{join_text([str(item) for item in top_groups])}"
    )

    increasing = model.get(
        "risk_increasing_factors",
        [],
    )

    lines.append(
        "主要提高風險因素："
    )

    if isinstance(
        increasing,
        list,
    ) and increasing:
        for index, item in enumerate(
            increasing[:10],
            start=1,
        ):
            if not isinstance(
                item,
                dict,
            ):
                continue

            lines.append(
                f"{index}. "
                f"{normalize_text(item.get('feature'))}"
            )
    else:
        lines.append(
            "NOT_AVAILABLE"
        )

    decreasing = model.get(
        "risk_decreasing_factors",
        [],
    )

    lines.append(
        "主要降低風險因素："
    )

    if isinstance(
        decreasing,
        list,
    ) and decreasing:
        for index, item in enumerate(
            decreasing[:10],
            start=1,
        ):
            if not isinstance(
                item,
                dict,
            ):
                continue

            lines.append(
                f"{index}. "
                f"{normalize_text(item.get('feature'))}"
            )
    else:
        lines.append(
            "NOT_AVAILABLE"
        )

    lines.append(
        ""
    )

    lines.append(
        "【五、個人化治療候選】"
    )

    ranking = treatment.get(
        "ranking",
        [],
    )

    if isinstance(
        ranking,
        list,
    ) and ranking:
        for item in ranking:
            if not isinstance(
                item,
                dict,
            ):
                continue

            lines.append(
                f"{normalize_text(item.get('rank'))}. "
                f"{normalize_text(item.get('treatment'))}："
                f"{format_number(item.get('score'), 1)} "
                f"({normalize_text(item.get('suitability_level'))})"
            )

            lines.append(
                "   推薦狀態："
                f"{normalize_text(item.get('recommendation_status'))}"
            )

            lines.append(
                "   推薦信心："
                f"{normalize_text(get_nested(item, 'confidence', 'confidence_level'))}"
            )
    else:
        lines.append(
            "NOT_AVAILABLE"
        )

    lines.append(
        ""
    )

    lines.append(
        "【六、治療前必要處理】"
    )

    prerequisites = treatment.get(
        "clinical_prerequisites",
        [],
    )

    if isinstance(
        prerequisites,
        list,
    ) and prerequisites:
        for item in prerequisites:
            if not isinstance(
                item,
                dict,
            ):
                continue

            lines.append(
                f"{normalize_text(item.get('priority'))}. "
                f"{normalize_text(item.get('title'))}"
            )

            lines.append(
                "   原因："
                f"{normalize_text(item.get('reason'))}"
            )
    else:
        lines.append(
            "NOT_AVAILABLE"
        )

    lines.append(
        ""
    )

    lines.append(
        "【七、數位孿生整合摘要】"
    )

    lines.append(
        report.get(
            "digital_twin_narrative",
            "NOT_AVAILABLE",
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "【研究與安全聲明】"
    )

    lines.append(
        report.get(
            "safety_note",
            "",
        )
    )

    lines.append(
        "=" * 80
    )

    return "\n".join(
        lines
    )


# ============================================================
# HTML 報告
# ============================================================

def html_escape(
    value: Any,
) -> str:
    return html.escape(
        normalize_text(
            value
        )
    )


def html_value(
    value: Any,
    digits: int = 2,
    suffix: str = "",
) -> str:
    number = safe_float(
        value
    )

    if number is None:
        return "NOT_AVAILABLE"

    return html.escape(
        f"{number:.{digits}f}{suffix}"
    )


def build_html_report(
    report: dict[str, Any],
) -> str:
    patient = report["patient_information"]
    respiratory = report["respiratory_phenotype"]
    oxygen = report["oxygen_phenotype"]
    model = report["model_explanation"]
    treatment = report["personalized_treatment"]
    medical_basis = treatment.get("medical_decision_basis", {})
    if not isinstance(medical_basis, dict):
        medical_basis = {}

    medical_signals_html = "".join(
        f"<li>{html_escape(signal)}</li>"
        for signal in medical_basis.get("signals_used", [])
    )
    medical_missing_html = "".join(
        f"<li>{html_escape(item)}</li>"
        for item in medical_basis.get("missing_data", [])
    ) or "<li>目前規則層未標示缺少資料；仍須由臨床醫師確認原始訊號品質。</li>"
    medical_references_html = "".join(
        (
            "<li><a href=\""
            + html_escape(reference.get("url"))
            + "\" target=\"_blank\" rel=\"noopener noreferrer\">"
            + html_escape(reference.get("title"))
            + "</a>："
            + html_escape(reference.get("use"))
            + "</li>"
        )
        for reference in medical_basis.get("guideline_references", [])
        if isinstance(reference, dict)
    )
    surgery_gate = medical_basis.get("surgery_local_gate", {})
    if not isinstance(surgery_gate, dict):
        surgery_gate = {}
    surgery_gate_status = (
        "目前符合系統候選條件"
        if surgery_gate.get("eligible") is True
        else "目前不符合或資料不足"
    )
    clinical_ingestion = treatment.get("clinical_feature_ingestion", {})
    if not isinstance(clinical_ingestion, dict):
        clinical_ingestion = {}
    clinical_feature_rows_html = "".join(
        (
            "<tr><td>" + html_escape(item.get("module")) + "</td>"
            "<td>" + html_escape(item.get("source_field")) + "</td>"
            "<td>" + html_escape(item.get("model_feature")) + "</td>"
            "<td>" + html_escape(item.get("value")) + "</td>"
            "<td>已納入治療 context／訓練快照</td></tr>"
        )
        for item in clinical_ingestion.get("features", [])
        if isinstance(item, dict)
    )
    if not clinical_feature_rows_html:
        clinical_feature_rows_html = (
            "<tr><td colspan=\"5\">目前沒有經醫師確認的補充欄位。</td></tr>"
        )

    report_metadata = report.get(
        "report_metadata",
        {},
    )

    if not isinstance(
        report_metadata,
        dict,
    ):
        report_metadata = {}

    report_mode = normalize_text(
        report_metadata.get(
            "report_mode"
        )
    )

    report_generated_at = normalize_text(
        report_metadata.get(
            "generated_at"
        )
    )

    recommendation_history_count = safe_int(
        report_metadata.get(
            "recommendation_history_count"
        )
    )

    arousal_shap = report.get(
        "arousal_risk_explanation",
        {},
    )

    if not isinstance(
        arousal_shap,
        dict,
    ):
        arousal_shap = {}


    arousal_early_warning = report.get(
        "arousal_early_warning",
        {},
    )
    warning = arousal_early_warning

    timeline_case = {}

    subtype_results = warning.get(
        "subtype_results",
        [],
    )

    warning_dashboard_rows: list[str] = []

    dashboard_items = [
        (
            "事件層級 Recall",
            f"{(safe_float(warning.get('warning_recall')) or 0.0) * 100:.1f}%",
        ),
        (
            "成功預警",
            (
                f"{warning.get('successfully_warned_count', 0)} / "
                f"{warning.get('true_arousal_count', 0)}"
            ),
        ),
        (
            "中位提前時間",
            f"{(safe_float(warning.get('lead_seconds_median')) or 0.0):.1f} 秒",
        ),
        (
            "平均提前時間",
            f"{(safe_float(warning.get('lead_seconds_mean')) or 0.0):.1f} 秒",
        ),
        (
            "每小時誤警事件",
            (
                f"{(safe_float(warning.get('false_alert_episodes_per_hour')) or 0.0):.1f}"
            ),
        ),
    ]

    for title, value in dashboard_items:
        warning_dashboard_rows.append(
            f"""
            <div class="metric">
                <div class="metric-label">
                    {html.escape(title)}
                </div>

                <div class="metric-value">
                    {html.escape(str(value))}
                </div>
            </div>
            """
        )   
        warning_dashboard_html = "".join(
            warning_dashboard_rows
        )

    if not isinstance(
        arousal_early_warning,
        dict,
    ):
        arousal_early_warning = {}

    representative_case = report.get(
        "representative_early_warning_case",
        {},
    )

    if not isinstance(
        representative_case,
        dict,
    ):
        representative_case = {}

    representative_timeline = report.get(
        "representative_early_warning_timeline",
        {},
    )

    if not isinstance(
        representative_timeline,
        dict,
    ):
        representative_timeline = {}

    representative_timeline_window = report.get(
        "representative_early_warning_timeline_window",
        [],
    )

    if not isinstance(
        representative_timeline_window,
        list,
    ):
        representative_timeline_window = []

    group_label_map = {
        "SPO2": "血氧與缺氧變化",
        "AIRFLOW": "鼻氣流與呼吸氣流",
        "RESPIRATORY_EFFORT": "胸腹呼吸努力",
        "CARDIOVASCULAR": "心血管訊號",
        "EEG": "腦波活動",
        "SLEEP_STAGE": "睡眠分期",
        "EOG": "眼動訊號",
        "SNORE": "鼾聲",
        "POSITION": "睡眠姿勢",
        "DATA_QUALITY": "資料品質",
        "OTHER": "其他因素",
    }

    feature_label_map = {
        "stage_N1": "處於 N1 淺睡期",
        "stage_W": "處於清醒期",
        "spo2_std_resp": "呼吸分析中的血氧波動程度",
        "spo2_std": "整體血氧波動程度",
        "spo2_range": "血氧變化範圍",
        "spo2_iqr": "血氧四分位變異",
        "spo2_below_88_fraction": "SpO₂ 低於 88% 的比例",
        "spo2_range_resp": "呼吸分析中的血氧變化範圍",
        "spo2_iqr_resp": "呼吸分析中的血氧四分位變異",
        "flow_p75": "鼻氣流振幅上四分位數",
        "flow_max": "鼻氣流最大振幅",
        "flow_max_resp": "呼吸分析中的鼻氣流最大振幅",
        "thermistor_amplitude_p90_p10": "熱敏氣流振幅範圍",
        "thermistor_std": "熱敏氣流波動程度",
        "eeg_c4_beta_relative": "C4 腦波 Beta 相對能量",
    }

    group_rows_html: list[str] = []

    for item in arousal_shap.get(
        "group_importance",
        [],
    )[:6]:
        if not isinstance(
            item,
            dict,
        ):
            continue

        group_code = str(
            item.get(
                "feature_group",
                "NOT_AVAILABLE",
            )
        )

        group_label = group_label_map.get(
            group_code,
            group_code,
        )

        relative_share = safe_float(
            item.get(
                "relative_importance_share"
            )
        )

        share_percent = (
            relative_share * 100.0
            if relative_share is not None
            else 0.0
        )

        direction = str(
            item.get(
                "direction",
                "UNKNOWN",
            )
        )

        if direction == "RISK_INCREASING":
            direction_label = "整體推高警報輸出"
            direction_class = "direction-up"
            direction_symbol = "↑"

        elif direction == "RISK_DECREASING":
            direction_label = "整體降低警報輸出"
            direction_class = "direction-down"
            direction_symbol = "↓"

        else:
            direction_label = "方向不明確"
            direction_class = "direction-neutral"
            direction_symbol = "—"

        bounded_share = min(
            max(
                share_percent,
                0.0,
            ),
            100.0,
        )

        group_rows_html.append(
            f"""
            <div class="shap-item">
                <div class="shap-item-header">
                    <span class="shap-rank">
                        {html_escape(item.get("rank"))}
                    </span>

                    <span class="shap-name">
                        {html.escape(group_label)}
                    </span>

                    <span class="shap-share">
                        {share_percent:.1f}%
                    </span>
                </div>

                <div class="shap-bar-track">
                    <div
                        class="shap-bar-fill"
                        style="width: {bounded_share:.1f}%"
                    ></div>
                </div>

                <div class="shap-direction {direction_class}">
                    {direction_symbol}
                    {html.escape(direction_label)}
                </div>
            </div>
            """
        )

    def build_factor_rows(
        items: Any,
        positive: bool,
    ) -> list[str]:
        rows: list[str] = []

        if not isinstance(
            items,
            list,
        ):
            return rows

        for item in items[:5]:
            if not isinstance(
                item,
                dict,
            ):
                continue

            feature_code = str(
                item.get(
                    "feature",
                    "NOT_AVAILABLE",
                )
            )

            feature_label = feature_label_map.get(
                feature_code,
                feature_code,
            )

            signed_value = safe_float(
                item.get(
                    "mean_signed_shap"
                )
            )

            fraction_key = (
                "positive_effect_fraction"
                if positive
                else "negative_effect_fraction"
            )

            fraction = safe_float(
                item.get(
                    fraction_key
                )
            )

            shap_text = (
                f"{signed_value:+.4f}"
                if signed_value is not None
                else "NOT_AVAILABLE"
            )

            fraction_text = (
                f"{fraction * 100.0:.1f}%"
                if fraction is not None
                else "NOT_AVAILABLE"
            )

            row_class = (
                "factor-positive"
                if positive
                else "factor-negative"
            )

            fraction_label = (
                "正向影響 Epoch"
                if positive
                else "負向影響 Epoch"
            )

            rows.append(
                f"""
                <div class="factor-row {row_class}">
                    <div>
                        <strong>
                            {html.escape(feature_label)}
                        </strong>

                        <div class="factor-code">
                            {html.escape(feature_code)}
                        </div>
                    </div>

                    <div class="factor-value">
                        SHAP {shap_text}

                        <div class="factor-fraction">
                            {fraction_label}：{fraction_text}
                        </div>
                    </div>
                </div>
                """
            )

        return rows

    increasing_rows_html = build_factor_rows(
        arousal_shap.get(
            "risk_increasing_factors",
            [],
        ),
        positive=True,
    )

    decreasing_rows_html = build_factor_rows(
        arousal_shap.get(
            "risk_decreasing_factors",
            [],
        ),
        positive=False,
    )


    subtype_rows_html: list[str] = []

    subtype_results = arousal_early_warning.get(
        "subtype_results",
        [],
    )

    if isinstance(
        subtype_results,
        list,
    ):
        for item in subtype_results:
            if not isinstance(
                item,
                dict,
            ):
                continue

            subtype_name = html_escape(
                item.get(
                    "subtype"
                )
            )

            arousal_count = safe_int(
                item.get(
                    "arousal_count"
                )
            )

            warned_count = safe_int(
                item.get(
                 "successfully_warned_count"
                )
            )

            warning_recall = safe_float(
                item.get(
                    "warning_recall"
                )
            )

            lead_mean = safe_float(
                item.get(
                    "lead_seconds_mean"
                )
            )

            lead_median = safe_float(
                item.get(
                    "lead_seconds_median"
                )
            )

            subtype_rows_html.append(
                f"""
                <tr>
                    <td>{subtype_name}</td>
                    <td>{html_escape(arousal_count)}</td>
                    <td>{html_escape(warned_count)}</td>
                    <td>{html_escape(format_percentage(warning_recall))}</td>
                    <td>{html_value(lead_mean, 2)} 秒</td>
                    <td>{html_value(lead_median, 2)} 秒</td>
             </tr>
                """
            )

    true_arousal_count = safe_int(
        arousal_early_warning.get(
            "true_arousal_count"
        )
    )

    successfully_warned_count = safe_int(
        arousal_early_warning.get(
            "successfully_warned_count"
        )
    )

    warning_recall = safe_float(
        arousal_early_warning.get(
            "warning_recall"
        )
    )

    lead_seconds_mean = safe_float(
        arousal_early_warning.get(
            "lead_seconds_mean"
        )
    )

    lead_seconds_median = safe_float(
        arousal_early_warning.get(
            "lead_seconds_median"
        )
    )

    lead_seconds_min = safe_float(
        arousal_early_warning.get(
            "lead_seconds_min"
        )
    )

    lead_seconds_max = safe_float(
        arousal_early_warning.get(
            "lead_seconds_max"
        )
    )

    false_alert_episodes_per_hour = safe_float(
        arousal_early_warning.get(
            "false_alert_episodes_per_hour"
        )
    )

    alert_ppv = safe_float(
        arousal_early_warning.get(
            "alert_epoch_positive_predictive_value"
        )
    )

    prediction_horizon_seconds = safe_float(
        arousal_early_warning.get(
            "prediction_horizon_seconds"
        )
    )

    representative_subtype = html_escape(
        representative_case.get(
            "subtype"
        )
    )

    representative_stage = html_escape(
        representative_timeline.get(
            "stage"
        )
    )

    representative_risk_level = html_escape(
        representative_timeline.get(
            "arousal_risk_level"
        )
    )

    representative_alert_classification = html_escape(
        representative_timeline.get(
            "alert_classification"
        )
    )

    representative_future_arousal_exists = (
        str(
            representative_timeline.get(
                "future_arousal_exists"
            )
        ).strip().lower()
        in {
            "true",
            "1",
            "yes",
        }
    )

    representative_future_arousal_count = safe_int(
        representative_timeline.get(
            "future_arousal_count"
        )
    )

    representative_quality_issue = (
        str(
            representative_timeline.get(
                "quality_any_issue"
            )
        ).strip().lower()
        in {
            "true",
            "1",
            "yes",
        }
    )

    representative_epoch_index = safe_int(
        representative_timeline.get(
            "epoch_index"
        )
    )

    representative_arousal_time = html_escape(
        representative_case.get(
            "arousal_start_time"
        )
    )

    representative_alert_time = html_escape(
        representative_case.get(
            "selected_alert_epoch_end"
        )
    )

    representative_probability = safe_float(
        representative_case.get(
            "selected_alert_probability"
        )
    )

    representative_lead_seconds = safe_float(
        representative_case.get(
            "lead_seconds"
        )
    )

    representative_timeline_window = report.get(
        "representative_early_warning_timeline_window",
        [],
    )

    if not isinstance(
        representative_timeline_window,
        list,
    ):
        representative_timeline_window = []



    timeline_window_rows: list[str] = []

    for row in representative_timeline_window:
        if not isinstance(
            row,
            dict,
        ):
            continue

        row_probability = safe_float(
            row.get(
                "arousal_next_30s_probability"
            )
        )

        row_threshold = safe_float(
            row.get(
                "arousal_alert_threshold"
            )
        )

        row_alert = (
            str(
                row.get(
                    "arousal_next_30s_alert"
                )
            ).strip().lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

        row_future_arousal = (
            str(
                row.get(
                    "future_arousal_exists"
                )
            ).strip().lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

        row_quality_issue = (
            str(
                row.get(
                    "quality_any_issue"
                )
            ).strip().lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

        timeline_window_rows.append(
            f"""
            <tr>
                <td>
                    {html_escape(row.get("epoch_index"))}
                </td>
                <td>
                    {html_escape(row.get("start_time"))}
                </td>
                <td>
                    {html_escape(row.get("end_time"))}
                </td>
                <td>
                    {html_escape(row.get("stage"))}
                </td>
                <td>
                    {html_escape(format_percentage(row_probability))}
                </td>
                <td>
                    {html_escape(format_percentage(row_threshold))}
                </td>
                <td>
                    {"是" if row_alert else "否"}
                </td>
                <td>
                    {html_escape(row.get("arousal_risk_level"))}
                </td>
                <td>
                    {"是" if row_future_arousal else "否"}
                </td>
                <td>
                    {html_escape(row.get("alert_classification"))}
                </td>
                <td>
                    {"有" if row_quality_issue else "無"}
                </td>
            </tr>
            """
        )

    timeline_window_rows_html = "".join(
        timeline_window_rows
    )

    timeline_chart_points: list[dict[str, Any]] = []

    for row in representative_timeline_window:
        if not isinstance(
            row,
            dict,
        ):
            continue

        epoch_index = safe_int(
            row.get(
                "epoch_index"
            )
        )

        probability = safe_float(
            row.get(
                "arousal_next_30s_probability"
            )
        )

        threshold = safe_float(
            row.get(
                "arousal_alert_threshold"
            )
        )

        if (
            epoch_index is None
            or probability is None
        ):
            continue

        timeline_chart_points.append(
            {
                "epoch_index": epoch_index,
                "probability": probability,
                "threshold": threshold,
                "stage": str(
                    row.get(
                        "stage",
                        "",
                    )
                ),
                "alert": (
                    str(
                        row.get(
                            "arousal_next_30s_alert"
                        )
                    ).strip().lower()
                    in {
                        "true",
                        "1",
                        "yes",
                    }
                ),
                "future_arousal": (
                    str(
                        row.get(
                            "future_arousal_exists"
                        )
                    ).strip().lower()
                    in {
                        "true",
                        "1",
                        "yes",
                    }
                ),
            }
        )

    timeline_chart_width = 900
    timeline_chart_height = 300

    timeline_chart_left = 70
    timeline_chart_right = 30
    timeline_chart_top = 30
    timeline_chart_bottom = 55

    timeline_plot_width = (
        timeline_chart_width
        - timeline_chart_left
        - timeline_chart_right
    )

    timeline_plot_height = (
        timeline_chart_height
        - timeline_chart_top
        - timeline_chart_bottom
    )

    timeline_chart_svg = ""

    if timeline_chart_points:
        probability_values = [
            point["probability"]
            for point in timeline_chart_points
        ]

        threshold_values = [
            point["threshold"]
            for point in timeline_chart_points
            if point["threshold"] is not None
        ]

        all_probability_values = (
            probability_values
            + threshold_values
        )

        y_min = max(
            0.0,
            min(
                all_probability_values
            ) - 0.05,
        )

        y_max = min(
            1.0,
            max(
                all_probability_values
            ) + 0.05,
        )

        if y_max <= y_min:
            y_max = min(
                1.0,
                y_min + 0.1,
            )

        point_count = len(
            timeline_chart_points
        )

        if point_count == 1:
            x_positions = [
                timeline_chart_left
                + timeline_plot_width / 2
            ]
        else:
            x_positions = [
                timeline_chart_left
                + (
                    index
                    * timeline_plot_width
                    / (point_count - 1)
                )
                for index in range(
                    point_count
                )
            ]

        def probability_to_y(
            value: float,
        ) -> float:
            return (
                timeline_chart_top
                + (
                    y_max - value
                )
                / (
                    y_max - y_min
                )
                * timeline_plot_height
            )
        

        svg_elements: list[str] = []

        # 圖表背景與邊框
        svg_elements.append(
            f"""
            <rect
                x="0"
                y="0"
                width="{timeline_chart_width}"
                height="{timeline_chart_height}"
                fill="#ffffff"
                stroke="#dddddd"
                rx="8"
            />
            """
        )

        # Y 軸網格線與百分比標籤
        y_tick_count = 4

        for tick_index in range(
            y_tick_count + 1
        ):
            tick_value = (
                y_min
                + (
                    y_max - y_min
                )
                * tick_index
                / y_tick_count
            )

            tick_y = probability_to_y(
                tick_value
            )

            svg_elements.append(
                f"""
                <line
                    x1="{timeline_chart_left}"
                    y1="{tick_y:.2f}"
                    x2="{
                        timeline_chart_left
                        + timeline_plot_width
                    }"
                    y2="{tick_y:.2f}"
                    stroke="#e5e7eb"
                    stroke-width="1"
                />
                """
            )

            svg_elements.append(
                f"""
                <text
                    x="{timeline_chart_left - 10}"
                    y="{tick_y + 4:.2f}"
                    text-anchor="end"
                    font-size="12"
                    fill="#666666"
                >
                    {tick_value * 100.0:.1f}%
                </text>
                """
            )

        # X 軸
        x_axis_y = (
            timeline_chart_top
            + timeline_plot_height
        )

        svg_elements.append(
            f"""
            <line
                x1="{timeline_chart_left}"
                y1="{x_axis_y}"
                x2="{
                    timeline_chart_left
                    + timeline_plot_width
                }"
                y2="{x_axis_y}"
                stroke="#999999"
                stroke-width="1"
            />
            """
        )

        # 警報閾值線
        first_threshold = next(
            (
                point["threshold"]
                for point in timeline_chart_points
                if point["threshold"] is not None
            ),
            None,
        )

        if first_threshold is not None:
            threshold_y = probability_to_y(
                first_threshold
            )

            svg_elements.append(
                f"""
                <line
                    x1="{timeline_chart_left}"
                    y1="{threshold_y:.2f}"
                    x2="{
                        timeline_chart_left
                        + timeline_plot_width
                    }"
                    y2="{threshold_y:.2f}"
                    stroke="#b45309"
                    stroke-width="2"
                    stroke-dasharray="7 5"
                />
                """
            )

            svg_elements.append(
                f"""
                <text
                    x="{
                        timeline_chart_left
                        + timeline_plot_width
                        - 4
                    }"
                    y="{threshold_y - 7:.2f}"
                    text-anchor="end"
                    font-size="12"
                    font-weight="700"
                    fill="#92400e"
                >
                    警報閾值 {first_threshold * 100.0:.1f}%
                </text>
                """
            )

        # 預測機率折線
        polyline_points: list[str] = []

        for index, point in enumerate(
            timeline_chart_points
        ):
            point_x = x_positions[
                index
            ]

            point_y = probability_to_y(
                point["probability"]
            )

            polyline_points.append(
                f"{point_x:.2f},{point_y:.2f}"
            )

        svg_elements.append(
            f"""
            <polyline
                points="{" ".join(polyline_points)}"
                fill="none"
                stroke="#2563eb"
                stroke-width="3"
                stroke-linejoin="round"
                stroke-linecap="round"
            />
            """
        )

        for index, point in enumerate(
            timeline_chart_points
        ):
            point_x = x_positions[
                index
            ]

            point_y = probability_to_y(
                point["probability"]
            )

            is_selected_epoch = (
                representative_epoch_index is not None
                and point["epoch_index"]
                == representative_epoch_index
            )

            point_radius = (
                7
                if is_selected_epoch
                else 5
            )

            point_fill = (
                "#dc2626"
                if is_selected_epoch
                else "#2563eb"
            )

            svg_elements.append(
                f"""
                <circle
                    cx="{point_x:.2f}"
                    cy="{point_y:.2f}"
                    r="{point_radius}"
                    fill="{point_fill}"
                    stroke="#ffffff"
                    stroke-width="2"
                />
                """
            )

            # 機率標籤
            svg_elements.append(
                f"""
                <text
                    x="{point_x:.2f}"
                    y="{point_y - 12:.2f}"
                    text-anchor="middle"
                    font-size="12"
                    font-weight="700"
                    fill="#111111"
                >
                    {point["probability"] * 100.0:.1f}%
                </text>
                """
            )

            # Epoch 標籤
            svg_elements.append(
                f"""
                <text
                    x="{point_x:.2f}"
                    y="{x_axis_y + 20:.2f}"
                    text-anchor="middle"
                    font-size="12"
                    fill="#333333"
                >
                    Epoch {point["epoch_index"]}
                </text>
                """
            )

            # 睡眠分期標籤
            svg_elements.append(
                f"""
                <text
                    x="{point_x:.2f}"
                    y="{x_axis_y + 38:.2f}"
                    text-anchor="middle"
                    font-size="11"
                    fill="#777777"
                >
                    Stage {html.escape(point["stage"])}
                </text>
                """
            )

            # 代表警報 Epoch 標示
            if is_selected_epoch:
                svg_elements.append(
                    f"""
                    <text
                        x="{point_x:.2f}"
                        y="{timeline_chart_top + 14}"
                        text-anchor="middle"
                        font-size="12"
                        font-weight="700"
                        fill="#b91c1c"
                    >
                        代表警報 Epoch
                    </text>
                    """
                )



            timeline_chart_svg = (
            f"""
            <svg
                viewBox="
                    0 0
                    {timeline_chart_width}
                    {timeline_chart_height}
                "
                width="100%"
                role="img"
                aria-label="代表案例 Arousal 預測機率變化圖"
                xmlns="http://www.w3.org/2000/svg"
            >
                {"".join(svg_elements)}
            </svg>
            """
        )

    representative_case_html = (
        f"""
        <div class="card">
            <h2>
                Representative Early Warning Case
            </h2>

            <p class="section-intro">
                本案例選取此患者成功預警事件中，
                提前時間最長的一次作為代表案例。
            </p>

            <div class="grid">
                <div class="metric">
                    <div class="metric-label">
                        Arousal 類型
                    </div>

                    <div class="metric-value">
                        {representative_subtype}
                    </div>
                </div>

                <div class="metric">
                    <div class="metric-label">
                        警報時間
                    </div>

                    <div class="metric-value">
                        {representative_alert_time}
                    </div>
                </div>

                <div class="metric">
                    <div class="metric-label">
                        Arousal 發生時間
                    </div>

                    <div class="metric-value">
                        {representative_arousal_time}
                    </div>
                </div>

                <div class="metric">
                    <div class="metric-label">
                        警報機率
                    </div>

                    <div class="metric-value">
                        {html_escape(format_percentage(representative_probability))}
                    </div>
                </div>

                <div class="metric">
                    <div class="metric-label">
                        提前時間
                    </div>

                    <div class="metric-value">
                        {html_value(representative_lead_seconds, 2)} 秒
                    </div>
                    <div class="metric">
    <div class="metric-label">
        警報 Epoch
    </div>

    <div class="metric-value">
        {html_escape(representative_epoch_index)}
    </div>
</div>

<div class="metric">
    <div class="metric-label">
        當下睡眠分期
    </div>

    <div class="metric-value">
        {representative_stage}
    </div>
</div>

<div class="metric">
    <div class="metric-label">
        Arousal 風險等級
    </div>

    <div class="metric-value">
        {representative_risk_level}
    </div>
</div>

<div class="metric">
    <div class="metric-label">
        警報分類
    </div>

    <div class="metric-value">
        {representative_alert_classification}
    </div>
</div>

<div class="metric">
    <div class="metric-label">
        未來 30 秒真實 Arousal
    </div>

    <div class="metric-value">
        {
            "是"
            if representative_future_arousal_exists
            else "否"
        }
    </div>
</div>

<div class="metric">
    <div class="metric-label">
        未來 30 秒 Arousal 數
    </div>

    <div class="metric-value">
        {html_escape(representative_future_arousal_count)}
    </div>
</div>

<div class="metric">
    <div class="metric-label">
        該 Epoch 資料品質問題
    </div>

    <div class="metric-value">
        {
            "有"
            if representative_quality_issue
            else "無"
        }
    </div>
</div>
                </div>
            </div>

            <h3>
    代表案例事件前後預測變化
</h3>

<p class="section-intro">
    下表顯示代表警報 Epoch 前 4 個 Epoch、
    警報 Epoch 本身，以及警報後 1 個 Epoch。
    每個 Epoch 為 30 秒。
</p>


<h3>
    Prediction Evolution
</h3>

<p class="section-intro">
    顯示代表案例於事件前後各 Epoch 的 Arousal 預測機率變化；
    虛線表示模型警報閾值，紅點表示代表警報 Epoch。
</p>

<div
    style="
        margin: 20px 0;
        background: #ffffff;
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 16px;
        overflow-x: auto;
    "
>
    {timeline_chart_svg}
</div>

<div class="table-wrap">
    <table>
        <thead>
            <tr>
                <th>Epoch</th>
                <th>開始時間</th>
                <th>結束時間</th>
                <th>睡眠分期</th>
                <th>Arousal 機率</th>
                <th>警報閾值</th>
                <th>觸發警報</th>
                <th>風險等級</th>
                <th>未來 30 秒有 Arousal</th>
                <th>警報分類</th>
                <th>資料品質問題</th>
            </tr>
        </thead>

        <tbody>
            {
                timeline_window_rows_html
                if timeline_window_rows_html
                else (
                    "<tr>"
                    "<td colspan='11'>"
                    "NOT_AVAILABLE"
                    "</td>"
                    "</tr>"
                )
            }
        </tbody>
    </table>
</div>


            <div class="notice">
                模型於 Arousal 發生前
                {html_value(representative_lead_seconds, 2)} 秒
                達到警報條件；該次事件類型為
                {representative_subtype}。
            </div>
        </div>
        </div>
        """
        if representative_case
        else ""
    )

    early_warning_section_html = (
        f"""
        <div class="card">
            <h2>
                六、Arousal 提前預警
            </h2>

            <p class="section-intro">
                本區塊評估模型是否能在真實 Arousal 發生前，
                於未來 {html_value(prediction_horizon_seconds, 0)} 秒預測窗內發出警報。
                此結果為離線事件時間匹配分析。
            </p>

            <div class="grid">
                {warning_dashboard_html}
            </div>


            <h3>
                不同 Arousal 類型的預警結果
            </h3>

            <table>
                <thead>
                    <tr>
                        <th>Arousal 類型</th>
                        <th>事件數</th>
                        <th>成功預警數</th>
                        <th>召回率</th>
                        <th>平均提前時間</th>
                        <th>中位提前時間</th>
                    </tr>
                </thead>

                <tbody>
                    {
                        "".join(
                            subtype_rows_html
                        )
                        if subtype_rows_html
                        else (
                            "<tr>"
                            "<td colspan='6'>NOT_AVAILABLE</td>"
                            "</tr>"
                        )
                    }
                </tbody>
            </table>

            <div class="notice">
                <strong>研究型解讀：</strong>
                {
                    html_escape(
                        arousal_early_warning.get(
                            "interpretation"
                        )
                    )
                }
            </div>
        </div>
        """
        if arousal_early_warning.get(
            "available"
        )
        else ""
    )

    mechanism_label_map = {
        "HYPOXEMIA": "低氧與缺氧負荷",
        "REM_RELATED": "REM 睡眠相關",
        "RESPIRATORY_INSTABILITY": "呼吸控制不穩定",
        "RESPIRATORY_EFFORT_DYSYNCHRONY": "胸腹呼吸努力不同步",
        "OBSTRUCTIVE_ANATOMICAL": "阻塞性解剖因素",
        "POSITIONAL": "睡姿相關因素",
    }

    mechanism_description_map = {
        "HYPOXEMIA": (
            "血氧下降、持續性低氧或低氧負荷對患者表型的影響。"
        ),
        "REM_RELATED": (
            "呼吸事件在 REM 睡眠期間較明顯。"
        ),
        "RESPIRATORY_INSTABILITY": (
            "呼吸控制穩定度與 Loop Gain proxy 相關的影響。"
        ),
        "RESPIRATORY_EFFORT_DYSYNCHRONY": (
            "胸部與腹部呼吸努力不同步的影響。"
        ),
        "OBSTRUCTIVE_ANATOMICAL": (
            "阻塞型事件與可能的上呼吸道解剖因素。"
        ),
        "POSITIONAL": (
            "不同睡姿下呼吸事件風險差異的影響。"
        ),
    }


    mechanism_clinical_interpretation_map = {
        "HYPOXEMIA": {
            "clinical_interpretation": (
                "目前資料顯示，血氧下降、持續性低氧或整體低氧負荷，"
                "可能是此患者睡眠生理表型中較重要的組成。"
            ),
            "clinical_implication": (
                "建議優先確認低氧是否主要由阻塞型呼吸事件造成，"
                "並評估基線低氧、心肺疾病、睡眠低通氣或其他非事件相關來源。"
                "若後續使用 PAP，仍應追蹤治療後夜間 SpO₂。"
            ),
        },
        "REM_RELATED": {
            "clinical_interpretation": (
                "目前資料顯示，患者的呼吸事件或相關生理負荷在 REM 睡眠期間較明顯。"
            ),
            "clinical_implication": (
                "建議比較 REM 與 NREM 期間的事件率、血氧反應及治療效果，"
                "並注意整夜治療是否能涵蓋後半夜較多的 REM 睡眠。"
            ),
        },
        "RESPIRATORY_INSTABILITY": {
            "clinical_interpretation": (
                "目前資料顯示，呼吸控制穩定度可能對患者表型有一定影響，"
                "但此結果來自研究型 Loop Gain proxy。"
            ),
            "clinical_implication": (
                "建議搭配中央型事件比例、呼吸頻率變異、血氧波動及呼吸相關覺醒一起判讀，"
                "不可單獨視為正式 Loop Gain 測量。"
            ),
        },
        "RESPIRATORY_EFFORT_DYSYNCHRONY": {
            "clinical_interpretation": (
                "目前資料顯示，胸部與腹部呼吸努力之間存在一定程度的不同步或相關性下降。"
            ),
            "clinical_implication": (
                "建議搭配原始胸腹呼吸帶訊號、呼吸事件時段與訊號品質進一步確認，"
                "以判斷是否具有阻塞、呼吸努力增加或感測器問題。"
            ),
        },
        "OBSTRUCTIVE_ANATOMICAL": {
            "clinical_interpretation": (
                "目前 PSG 事件型態對固定上呼吸道解剖阻塞的支持程度相對較低。"
            ),
            "clinical_implication": (
                "若臨床仍懷疑明確解剖阻塞，需搭配耳鼻喉檢查、鼻阻力、"
                "扁桃腺、舌根、下顎、影像或 DISE 評估，不能只由 PSG 分數判定。"
            ),
        },
        "POSITIONAL": {
            "clinical_interpretation": (
                "目前機轉分數對已校正姿勢型表型的支持程度較低。"
            ),
            "clinical_implication": (
                "此結果不等於排除姿勢影響。若姿勢訊號尚未校正為仰睡、側睡或趴睡，"
                "仍應搭配姿勢群組事件率與實際身體姿勢標註判讀。"
            ),
        },
    }

    mechanism_rows_html: list[str] = []

    mechanism_ranking = respiratory.get(
        "mechanism_ranking",
        [],
    )

    if isinstance(
        mechanism_ranking,
        list,
    ):
        for item in mechanism_ranking:
            if not isinstance(
                item,
                dict,
            ):
                continue

            mechanism_code = str(
                item.get(
                    "mechanism",
                    "NOT_AVAILABLE",
                )
            )

            mechanism_label = mechanism_label_map.get(
                mechanism_code,
                mechanism_code,
            )

            mechanism_description = (
                mechanism_description_map.get(
                    mechanism_code,
                    "目前沒有補充說明。",
                )
            )


            clinical_content = (
                mechanism_clinical_interpretation_map.get(
                    mechanism_code,
                    {},
                )
            )

            if not isinstance(
                clinical_content,
                dict,
            ):
                clinical_content = {}

            clinical_interpretation = str(
                clinical_content.get(
                    "clinical_interpretation",
                    "目前沒有個別患者臨床解讀。",
                )
            )

            clinical_implication = str(
                clinical_content.get(
                    "clinical_implication",
                    "目前沒有補充的臨床判讀建議。",
                )
            )

            mechanism_score = safe_float(
                item.get(
                    "score"
                )
            )

            score_percent = (
                mechanism_score * 100.0
                if mechanism_score is not None
                else 0.0
            )

            bounded_score_percent = min(
                max(
                    score_percent,
                    0.0,
                ),
                100.0,
            )

            mechanism_rows_html.append(
                f"""
                <div class="mechanism-item">
                    <div class="mechanism-header">
                        <span class="mechanism-rank">
                            {html_escape(item.get("rank"))}
                        </span>

                        <span class="mechanism-name">
                            {html.escape(mechanism_label)}
                        </span>

                        <span class="mechanism-score">
                            {html_value(mechanism_score, 3)}
                        </span>
                    </div>

                    <div class="mechanism-bar-track">
                        <div
                            class="mechanism-bar-fill"
                            style="width: {bounded_score_percent:.1f}%"
                        ></div>
                    </div>

                    <div class="mechanism-description">
                        {html.escape(mechanism_description)}
                    </div>

                    <div class="mechanism-description">
                        <strong>Clinical interpretation</strong><br>
                        {html.escape(clinical_interpretation)}
                    </div>

                    <div class="mechanism-description">
                        <strong>Clinical implication</strong><br>
                        {html.escape(clinical_implication)}
                    </div>

                    <div class="mechanism-code">
                        {html.escape(mechanism_code)}
                    </div>
                </div>
                """
            )

    phenotype_section_html = (
        f"""
        <div class="card">
            <h2>
                三、Digital Twin 個別患者機轉排名
            </h2>

            <p class="section-intro">
                本區塊依目前可觀察的 PSG 表型，
                比較不同研究型生理機轉對此患者的相對支持程度。
                分數越高表示目前資料越符合該機轉，
                但不等同正式病理生理診斷或因果證明。
            </p>

            <div class="mechanism-list">
                {
                    "".join(
                        mechanism_rows_html
                    )
                    if mechanism_rows_html
                    else "<p>NOT_AVAILABLE</p>"
                }
            </div>
        </div>
        """
        if mechanism_rows_html
        else ""
    )

    ranking_rows: list[str] = []

    for item in treatment.get(
        "ranking",
        [],
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue

        supporting = item.get(
            "supporting_factors",
            [],
        )

        limiting = item.get(
            "limiting_factors",
            [],
        )

        supporting_html = "".join(
            (
                "<li>"
                + html_escape(
                    reason.get(
                        "description"
                    )
                )
                + "</li>"
            )
            for reason in supporting
            if isinstance(
                reason,
                dict,
            )
        )

        limiting_html = "".join(
            (
                "<li>"
                + html_escape(
                    reason.get(
                        "description"
                    )
                )
                + "</li>"
            )
            for reason in limiting
            if isinstance(
                reason,
                dict,
            )
        )

        ranking_rows.append(
            f"""
            <tr>
                <td>
                    {html_escape(item.get("rank"))}
                </td>

                <td>
                    {html_escape(item.get("treatment"))}
                </td>

                <td>
                    {html_value(item.get("score"), 1)}
                </td>

                <td>
                    {html_escape(item.get("suitability_level"))}
                </td>

                <td>
                    {html_escape(item.get("recommendation_status"))}
                </td>

                <td>
                    {
                        html_escape(
                            get_nested(
                                item,
                                "confidence",
                                "confidence_level",
                            )
                        )
                    }
                </td>

                <td>
                    <ul>
                        {supporting_html}
                    </ul>
                </td>

                <td>
                    <ul>
                        {limiting_html}
                    </ul>
                </td>
            </tr>
            """
        )


    treatment_ranking = treatment.get(
        "ranking",
        [],
    )

    top_treatment: dict = {}

    for treatment_item in treatment_ranking:
        if not isinstance(
            treatment_item,
            dict,
        ):
            continue

        if treatment_item.get("rank") == 1:
            top_treatment = treatment_item
            break

    if not top_treatment:
        for treatment_item in treatment_ranking:
            if isinstance(
                treatment_item,
                dict,
            ):
                top_treatment = treatment_item
                break

    top_treatment_name = top_treatment.get(
        "treatment",
        "資料不足",
    )

    top_treatment_score = top_treatment.get(
        "score",
    )

    top_treatment_status = top_treatment.get(
        "recommendation_status",
        "資料不足",
    )

    top_treatment_confidence = get_nested(
        top_treatment,
        "confidence",
        "confidence_level",
    )

    top_treatment_supporting_factors = top_treatment.get(
        "supporting_factors",
        [],
    )


    second_treatment = {}

    for treatment_item in treatment_ranking:
        if (
            isinstance(treatment_item, dict)
            and treatment_item.get("rank") == 2
        ):
            second_treatment = treatment_item
            break

    top_treatment_second = second_treatment.get(
        "treatment",
        "無",
    )

    top_treatment_second_score = second_treatment.get(
        "score",
    )

    top_treatment_supporting_html = "".join(
        (
            "<li>"
            + html_escape(
                factor.get(
                    "description",
                    "未提供說明",
                )
            )
            + "</li>"
        )
        for factor in top_treatment_supporting_factors
        if isinstance(
            factor,
            dict,
        )
    )

    if not top_treatment_supporting_html:
        top_treatment_supporting_html = (
            "<li>目前沒有可顯示的支持因素</li>"
        )

    prerequisite_rows: list[str] = []

    prerequisite_rows: list[str] = []

    for item in treatment.get(
        "clinical_prerequisites",
        [],
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue

        required_information = item.get(
            "required_information",
            [],
        )

        required_html = "".join(
            (
                "<li>"
                + html_escape(
                    required_item
                )
                + "</li>"
            )
            for required_item in required_information
        )

        prerequisite_rows.append(
            f"""
            <tr>
                <td>
                    {html_escape(item.get("priority"))}
                </td>

                <td>
                    {html_escape(item.get("title"))}
                </td>

                <td>
                    {html_escape(item.get("reason"))}
                </td>

                <td>
                    <ul>
                        {required_html}
                    </ul>
                </td>
            </tr>
            """
        )

    increasing_html = "".join(
        (
            "<li>"
            + html_escape(
                item.get(
                    "feature"
                )
            )
            + "</li>"
        )
        for item in model.get(
            "risk_increasing_factors",
            [],
        )
        if isinstance(
            item,
            dict,
        )
    )

    decreasing_html = "".join(
        (
            "<li>"
            + html_escape(
                item.get(
                    "feature"
                )
            )
            + "</li>"
        )
        for item in model.get(
            "risk_decreasing_factors",
            [],
        )
        if isinstance(
            item,
            dict,
        )
    )

    top_groups_html = "".join(
        (
            "<li>"
            + html_escape(
                item
            )
            + "</li>"
        )
        for item in model.get(
            "top_physiological_groups",
            [],
        )
    )

    narrative_html = "<br>".join(
        html.escape(
            report.get(
                "digital_twin_narrative",
                "",
            )
        ).splitlines()
    )

    shap_section_html = (
        f"""
        <div class="card">
            <h2>
                五、Arousal 高風險時段的個別生理影響
            </h2>

            <p class="section-intro">
                此區塊使用 ALERT_EPOCHS 的患者層級 SHAP 結果，
                顯示模型在高風險時段主要依賴哪些生理系統與特徵。
                正向 SHAP 代表推高警報輸出，
                負向 SHAP 代表降低警報輸出；
                不代表生理因果。
            </p>

            <div class="shap-layout">
                <div class="shap-panel">
                    <h3>
                        主要影響生理系統
                    </h3>

                    {
                        "".join(
                            group_rows_html
                        )
                        if group_rows_html
                        else "<p>NOT_AVAILABLE</p>"
                    }
                </div>

                <div class="shap-panel">
                    <h3>
                        主要提高 Arousal 警報輸出的因素
                    </h3>

                    {
                        "".join(
                            increasing_rows_html
                        )
                        if increasing_rows_html
                        else "<p>NOT_AVAILABLE</p>"
                    }
                </div>

                <div class="shap-panel">
                    <h3>
                        主要降低 Arousal 警報輸出的因素
                    </h3>

                    {
                        "".join(
                            decreasing_rows_html
                        )
                        if decreasing_rows_html
                        else "<p>NOT_AVAILABLE</p>"
                    }
                </div>
            </div>
        </div>
        """
        if arousal_shap.get(
            "available"
        )
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>
    患者睡眠數位孿生報告 -
    {html_escape(patient.get("patient_id"))}
</title>

<style>
body {{
    font-family:
        Arial,
        "Microsoft JhengHei",
        sans-serif;
    margin: 0;
    background: #f4f6f8;
    color: #222222;
    line-height: 1.6;
}}

.container {{
    width: min(1180px, 94%);
    margin: 24px auto;
}}

.header {{
    background: #ffffff;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 18px;
}}

.card {{
    background: #ffffff;
    padding: 22px;
    border-radius: 12px;
    margin-bottom: 18px;
}}

.decision-flow {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    margin-top: 20px;
}}

.decision-step {{
    width: min(720px, 90%);
    padding: 16px 20px;
    border: 1px solid #d7dee8;
    border-radius: 12px;
    background: #f8fafc;
    text-align: center;
    box-sizing: border-box;
}}

.decision-step-title {{
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 6px;
}}

.decision-step-content {{
    font-size: 14px;
    line-height: 1.6;
    color: #475569;
}}

.decision-arrow {{
    font-size: 28px;
    font-weight: 700;
    line-height: 1;
    color: #64748b;
}}

.decision-step-final {{
    border-width: 2px;
    background: #eef6ff;
}}

.decision-factor-list {{
    margin-top: 12px;
    text-align: left;
}}

.decision-factor-list ul {{
    margin: 8px 0 0;
    padding-left: 22px;
}}

.decision-factor-list li {{
    margin-bottom: 5px;
    line-height: 1.5;
}}

h1,
h2 {{
    margin-top: 0;
}}

.grid {{
    display: grid;
    grid-template-columns:
        repeat(
            auto-fit,
            minmax(210px, 1fr)
        );
    gap: 12px;
}}

.metric {{
    border: 1px solid #dddddd;
    border-radius: 8px;
    padding: 12px;
    min-width: 0;
    overflow: hidden;
}}

.metric-label {{
    font-size: 13px;
    color: #666666;
}}

.metric-value {{
    font-size: 19px;
    font-weight: 700;
    line-height: 1.35;
    overflow-wrap: anywhere;
    word-break: break-word;
    white-space: normal;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}}

th,
td {{
    border: 1px solid #dddddd;
    padding: 8px;
    vertical-align: top;
}}

th {{
    background: #eef1f4;
}}

.notice {{
    border-left: 5px solid #555555;
    padding: 14px;
    background: #f2f2f2;
}}

ul {{
    margin-top: 4px;
    margin-bottom: 4px;
}}

.section-intro {{
    color: #555555;
    margin-bottom: 18px;
}}

.shap-layout {{
    display: grid;
    grid-template-columns:
        repeat(
            auto-fit,
            minmax(280px, 1fr)
        );
    gap: 16px;
}}

.shap-panel {{
    border: 1px solid #dddddd;
    border-radius: 10px;
    padding: 16px;
    min-width: 0;
}}

.shap-item {{
    padding: 10px 0;
    border-bottom: 1px solid #eeeeee;
}}

.shap-item:last-child {{
    border-bottom: none;
}}

.shap-item-header {{
    display: grid;
    grid-template-columns:
        30px
        1fr
        auto;
    gap: 8px;
    align-items: center;
}}

.shap-rank {{
    font-weight: 700;
}}

.shap-name {{
    font-weight: 700;
    overflow-wrap: anywhere;
}}

.shap-share {{
    font-weight: 700;
}}

.shap-bar-track {{
    height: 9px;
    background: #e9edf1;
    border-radius: 999px;
    margin: 8px 0 5px;
    overflow: hidden;
}}

.shap-bar-fill {{
    height: 100%;
    background: #566573;
    border-radius: 999px;
}}

.shap-direction {{
    font-size: 13px;
    color: #666666;
}}

.direction-up {{
    color: #8a2f2f;
}}

.direction-down {{
    color: #245b45;
}}

.factor-row {{
    display: grid;
    grid-template-columns:
        1fr
        auto;
    gap: 12px;
    align-items: start;
    padding: 12px 0;
    border-bottom: 1px solid #eeeeee;
}}

.factor-row:last-child {{
    border-bottom: none;
}}

.factor-positive {{
    border-left: 4px solid #8a2f2f;
    padding-left: 10px;
}}

.factor-negative {{
    border-left: 4px solid #245b45;
    padding-left: 10px;
}}
.mechanism-list {{
    display: flex;
    flex-direction: column;
    gap: 18px;
}}

.mechanism-item {{
    border: 1px solid #dddddd;
    border-radius: 10px;
    padding: 16px;
    background: #ffffff;
}}

.mechanism-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 10px;
}}

.mechanism-rank {{
    font-weight: 700;
    color: #2563eb;
    min-width: 24px;
}}

.mechanism-name {{
    flex: 1;
    font-size: 18px;
    font-weight: 700;
}}

.mechanism-score {{
    font-size: 18px;
    font-weight: 700;
}}

.mechanism-bar-track {{
    width: 100%;
    height: 10px;
    background: #e5e7eb;
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 10px;
}}

.mechanism-bar-fill {{
    height: 100%;
    background: #2563eb;
}}

.mechanism-description {{
    color: #555555;
    line-height: 1.6;
    margin-bottom: 6px;
}}

.mechanism-code {{
    font-size: 12px;
    color: #999999;
}}

.factor-code {{
    color: #777777;
    font-size: 12px;
    overflow-wrap: anywhere;
}}

.factor-value {{
    text-align: right;
    font-weight: 700;
    white-space: nowrap;
}}

.factor-fraction {{
    color: #666666;
    font-size: 12px;
    font-weight: 400;
}}

@media (max-width: 700px) {{
    .factor-row {{
        grid-template-columns: 1fr;
    }}

    .factor-value {{
        text-align: left;
        white-space: normal;
    }}

    .card {{
        padding: 16px;
    }}
}}
</style>
</head>

<body>
<div class="container">

<div class="header">
    <h1>
        患者睡眠數位孿生整合報告
    </h1>

    <p>
        <strong>
            Patient ID：
        </strong>

        {html_escape(patient.get("patient_id"))}
    </p>

    <p>
        <strong>報告模式：</strong>
        {html.escape(report_mode)}
        ｜
        <strong>產生時間：</strong>
        {html.escape(report_generated_at)}
        ｜
        <strong>歷史推薦筆數：</strong>
        {html.escape(str(recommendation_history_count or 0))}
    </p>
</div>

<div class="card">
    <h2>
        一、患者基本資訊
    </h2>

    <div class="grid">
        <div class="metric">
    <div class="metric-label">
        性別
    </div>

    <div class="metric-value">
        {
            "男"
            if safe_int(patient.get("sex")) == 1
            else "女"
            if safe_int(patient.get("sex")) == 0
            else html_escape(patient.get("sex"))
        }
    </div>
</div>
        <div class="metric">
            <div class="metric-label">
                年齡
            </div>

            <div class="metric-value">
                {html_value(patient.get("age"), 1)} 歲
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                BMI
            </div>

            <div class="metric-value">
                {html_value(patient.get("BMI"), 1)}
            </div>
        </div>


                <div class="metric">
            <div class="metric-label">
                AHI
            </div>

            <div class="metric-value">
                {html_value(respiratory.get("ahi"), 2)}
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                Arousal Index
            </div>

            <div class="metric-value">
                {html_value(respiratory.get("arousal_index"), 2)}
                次／小時
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                平均 SpO₂
            </div>

            <div class="metric-value">
                {html_value(oxygen.get("spo2_mean"), 2, "%")}
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                穩健最低 SpO₂
            </div>

            <div class="metric-value">
                {html_value(oxygen.get("robust_spo2_min"), 2, "%")}
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                總睡眠時間
            </div>

            <div class="metric-value">
                {html_value(patient.get("sleep_minutes"), 1)}
                分鐘
            </div>
        </div>
    </div>
</div>

<div class="card">
    <h2>
        二、睡眠呼吸表型
    </h2>

    <div class="grid">
        <div class="metric">
            <div class="metric-label">
                AHI
            </div>

            <div class="metric-value">
                {html_value(respiratory.get("ahi"), 2)}
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                Arousal Index
            </div>

            <div class="metric-value">
                {html_value(respiratory.get("arousal_index"), 2)}
                次／小時
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                AHI 嚴重度
            </div>

            <div class="metric-value">
                {html_escape(respiratory.get("ahi_severity"))}
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                REM 相關性
            </div>

            <div class="metric-value">
                {html_escape(respiratory.get("rem_relevance"))}
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                姿勢相關性
            </div>

            <div class="metric-value">
                {html_escape(respiratory.get("position_relevance"))}
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                姿勢事件率比值
            </div>

            <div class="metric-value">
                {
                    html_value(
                        respiratory.get(
                            "position_event_ratio"
                        ),
                        2,
                    )
                }
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                Loop Gain proxy
            </div>

            <div class="metric-value">
                {
                    html_escape(
                        respiratory.get(
                            "loop_gain_proxy_level"
                        )
                    )
                }
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                主要研究型機轉
            </div>

            <div class="metric-value">
                {
                    html_escape(
                        respiratory.get(
                            "dominant_mechanism"
                        )
                    )
                }
            </div>
        </div>
    </div>
</div>
{phenotype_section_html}
<div class="card">
    <h2>
        三、血氧表型
    </h2>

    <div class="grid">
        <div class="metric">
            <div class="metric-label">
                平均 SpO₂
            </div>

            <div class="metric-value">
                {
                    html_value(
                        oxygen.get(
                            "spo2_mean"
                        ),
                        2,
                        "%",
                    )
                }
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                穩健最低 SpO₂
            </div>

            <div class="metric-value">
                {
                    html_value(
                        oxygen.get(
                            "robust_spo2_min"
                        ),
                        2,
                        "%",
                    )
                }
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                SpO₂ &lt;90%
            </div>

            <div class="metric-value">
                {
                    html_escape(
                        format_percentage(
                            oxygen.get(
                                "spo2_below_90_fraction"
                            )
                        )
                    )
                }
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                血氧下降深度
            </div>

            <div class="metric-value">
                {
                    html_escape(
                        oxygen.get(
                            "desaturation_depth"
                        )
                    )
                }
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                持續性低血氧負荷
            </div>

            <div class="metric-value">
                {
                    html_escape(
                        oxygen.get(
                            "sustained_low_oxygen_burden"
                        )
                    )
                }
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                低氧－事件耦合
            </div>

            <div class="metric-value">
                {
                    html_escape(
                        oxygen.get(
                            "oxygen_coupling_level"
                        )
                    )
                }
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                低氧遠離事件比例
            </div>

            <div class="metric-value">
                {
                    html_escape(
                        format_percentage(
                            oxygen.get(
                                "low_oxygen_away_from_event_fraction"
                            )
                        )
                    )
                }
            </div>
        </div>

        <div class="metric">
            <div class="metric-label">
                清醒期平均 SpO₂
            </div>

            <div class="metric-value">
                {
                    html_value(
                        oxygen.get(
                            "wake_spo2_mean"
                        ),
                        2,
                        "%",
                    )
                }
            </div>
        </div>
    </div>
</div>

<div class="card">
    <h2>
        四、模型整體解釋
    </h2>

    <h3>
        模型較關注的生理系統
    </h3>

    <ul>
        {top_groups_html}
    </ul>

    <h3>
        主要提高模型風險因素
    </h3>

    <ul>
        {increasing_html}
    </ul>

    <h3>
        主要降低模型風險因素
    </h3>

    <ul>
        {decreasing_html}
    </ul>
</div>

{shap_section_html}

{early_warning_section_html}

{representative_case_html}

<div class="card">
    <h2>治療判斷的醫學依據與安全限制</h2>
    <p>
        本區規則在機器學習調整之後執行，因此模型分數不能越過手術適應條件或
        將研究性藥物評估誤當成已核准處方。0–100 為相對適配／轉介優先指數，
        不是治療成功率，也尚未經外部前瞻性臨床驗證。
    </p>
    <h3>本次納入判斷的訊號</h3>
    <ul>{medical_signals_html}</ul>
    <h3>手術候選安全門檻</h3>
    <p>
        {html_escape(surgery_gate.get("rule"))}<br>
        <strong>{html_escape(surgery_gate_status)}</strong>
    </p>
    <h3>仍缺少或需要確認的資料</h3>
    <ul>{medical_missing_html}</ul>
    <p>
        <strong>重要校正：</strong>
        {html_escape(medical_basis.get("important_correction"))}
    </p>
    <h3>規則參考來源</h3>
    <ul>{medical_references_html}</ul>
    <h3>醫師補充資料接入模型紀錄</h3>
    <p>
        已接收 {html_escape(clinical_ingestion.get("feature_count", 0))}
        個標準化特徵。即使本次分數沒有改變，非空白欄位仍會保留於
        Digital Twin、治療訓練快照及新版模型特徵。
    </p>
    <table>
        <thead>
            <tr>
                <th>模組</th><th>原始欄位</th><th>模型特徵</th>
                <th>確認值</th><th>處理狀態</th>
            </tr>
        </thead>
        <tbody>{clinical_feature_rows_html}</tbody>
    </table>
</div>

<div class="card">
    <h2>
        七、個人化治療候選
    </h2>

    <table>
        <thead>
            <tr>
                <th>排名</th>
                <th>治療</th>
                <th>相對適配指數（非成功率）</th>
                <th>適配度</th>
                <th>狀態</th>
                <th>信心</th>
                <th>支持因素</th>
                <th>限制因素</th>
            </tr>
        </thead>

        <tbody>
            {"".join(ranking_rows)}
        </tbody>
    </table>
</div>

<div class="card">
    <h2>
        治療決策流程
    </h2>

    <div class="decision-flow">

                   <div class="decision-step">
            <div class="decision-step-title">
                疾病分類
            </div>

            <div class="decision-step-content">
                <strong>
                    {html_escape(
                        respiratory.get(
                            "ahi_severity"
                        )
                    )}
                    OSA
                </strong>
            </div>
        </div>

        <div class="decision-arrow">
            ↓
        </div>

        <div class="decision-step">
            <div class="decision-step-title">
                主要病理機轉
            </div>

            <div class="decision-step-content">
                <strong>
                    {html_escape(
                        respiratory.get(
                            "dominant_mechanism"
                        )
                    )}
                </strong>
            </div>
        </div>

        <div class="decision-arrow">
            ↓
        </div>

        <div class="decision-step">
            <div class="decision-step-title">
                主要支持證據
            </div>

            <div class="decision-step-content">
                REM 相關性：
                <strong>
                    {html_escape(respiratory.get("rem_relevance"))}
                </strong>

                <br>

                姿勢相關性：
                <strong>
                    {html_escape(respiratory.get("position_relevance"))}
                </strong>

                <br>

                持續性低血氧負荷：
                <strong>
                    {html_escape(
                        oxygen.get(
                            "sustained_low_oxygen_burden"
                        )
                    )}
                </strong>
            </div>
        </div>

        <div class="decision-arrow">
            ↓
        </div>


        <div class="decision-step">
    <div class="decision-step-title">
        臨床前置處理
    </div>

    <div class="decision-step-content">
        {html_escape(
            treatment.get(
                "clinical_prerequisites"
            )[0].get("title")
        )}
    </div>
</div>

<div class="decision-arrow">
    ↓
</div>

                <div class="decision-step decision-step-final">
            <div class="decision-step-title">
                第一治療候選：
                {html_escape(top_treatment_name)}
            </div>

            <div class="decision-step-content">
                <div>
                    證據分數：
                    {html_value(top_treatment_score, 1)}
                </div>

                <div>
                    推薦狀態：
                    {html_escape(top_treatment_status)}
                </div>

                <div>
                    推薦信心：
                    {html_escape(top_treatment_confidence)}
                </div>

                

                <div>
                    第二候選：
                    <strong>
                        {html_escape(top_treatment_second)}
                    </strong>
                    （分數：
                        {html_value(top_treatment_second_score, 1)}
                    )
                </div>
                <hr>

                <div class="decision-factor-list">
                    <strong>主要支持因素</strong>

                    <ul>
                        {top_treatment_supporting_html}
                    </ul>
                </div>
            </div>
        </div>

    </div>
</div>
<div class="card">
    <h2>
        八、治療前必要處理
    </h2>

    <table>
        <thead>
            <tr>
                <th>優先度</th>
                <th>事項</th>
                <th>原因</th>
                <th>需補充／確認</th>
            </tr>
        </thead>

        <tbody>
            {"".join(prerequisite_rows)}
        </tbody>
    </table>
</div>

<div class="card">
    <h2>
        九、數位孿生整合摘要
    </h2>

    <p>
        {narrative_html}
    </p>
</div>

<div class="card notice">
    <strong>
        研究與安全聲明：
    </strong>

    {html_escape(report.get("safety_note"))}
</div>

</div>
</body>
</html>
"""


# ============================================================
# 主程式
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "整合患者 PSG 生理表型、"
            "模型解釋與個人化治療候選，"
            "建立患者睡眠數位孿生報告。"
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
        help=(
            "以 PSG Follow-up 模式建立報告；"
            "保留既有臨床資訊與推薦歷史。"
        ),
    )

    args = parser.parse_args()

    patient_id = str(
        args.patient_id
    ).strip()

    if not patient_id:
        parser.error(
            "--patient-id 不可為空白。"
        )

    follow_up_mode = bool(
        args.follow_up
    )

    patient_folder = (
        INFERENCE_ROOT
        / patient_id
    )

    respiratory_file = (
        patient_folder
        / "respiratory_profile"
        / "patient_respiratory_profile.json"
    )

    factor_json_file = (
        patient_folder
        / "factor_summary"
        / "patient_factor_summary.json"
    )

    factor_csv_file = (
        patient_folder
        / "factor_summary"
        / "patient_factor_summary.csv"
    )

    spo2_quality_file = (
        patient_folder
        / "spo2_quality"
        / "spo2_quality_summary.json"
    )

    oxygen_file = (
        patient_folder
        / "oxygen_event_coupling"
        / "oxygen_event_coupling_summary.json"
    )

    position_file = (
        patient_folder
        / "position_profile"
        / "patient_position_profile.json"
    )

    shap_group_file = (
        patient_folder
        / "shap"
        / "patient_level"
        / "patient_group_importance.csv"
    )

    shap_feature_file = (
        patient_folder
        / "shap"
        / "patient_level"
        / "patient_feature_importance.csv"
    )

    arousal_early_warning_file = (
        patient_folder
        / "arousal_early_warning"
        / "arousal_early_warning_summary.json"
    )


    arousal_early_warning_events_file = (
        patient_folder
        / "arousal_early_warning"
        / "arousal_early_warning_events.csv"
    )

    arousal_early_warning_timeline_file = (
        patient_folder
        / "arousal_early_warning"
        / "arousal_early_warning_timeline.csv"
    )

    refined_treatment_file = (
        patient_folder
        / "treatment_refinement"
        / "refined_treatment_recommendation.json"
    )

    stage1_treatment_file = (
        patient_folder
        / "treatment_suitability"
        / "personalized_treatment_recommendation.json"
    )

    treatment_file = (
        refined_treatment_file
        if refined_treatment_file.exists()
        else stage1_treatment_file
    )

    if not respiratory_file.exists():
        raise FileNotFoundError(
            "缺少必要呼吸生理 Profile："
            f"{respiratory_file}"
        )

    respiratory_json = load_json_optional(
        respiratory_file
    )

    factor_json = load_json_optional(
        factor_json_file
    )

    factor_csv = load_csv_optional(
        factor_csv_file
    )

    spo2_quality_json = load_json_optional(
        spo2_quality_file
    )

    oxygen_json = load_json_optional(
        oxygen_file
    )

    position_json = load_json_optional(
        position_file
    )


    shap_group_csv = load_csv_optional(
        shap_group_file
    )

    shap_feature_csv = load_csv_optional(
        shap_feature_file
    )

    arousal_early_warning_json = load_json_optional(
        arousal_early_warning_file
    )



    arousal_early_warning_events_csv = load_csv_optional(
        arousal_early_warning_events_file
    )

    arousal_early_warning_timeline_csv = load_csv_optional(
        arousal_early_warning_timeline_file
    )

    # ============================================================
    # Early Warning 代表案例
    # 即使 Early Warning 檔案不存在，也必須先建立安全預設值
    # ============================================================

    representative_case: dict[str, Any] = {}

    representative_timeline: dict[str, Any] = {}

    representative_timeline_window: list[
        dict[str, Any]
    ] = []

    if not arousal_early_warning_events_csv.empty:
        successful_cases = (
            arousal_early_warning_events_csv.copy()
        )

        if (
            "successfully_warned"
            in successful_cases.columns
        ):
            successful_mask = (
                successful_cases[
                    "successfully_warned"
                ]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                        "yes",
                    ]
                )
            )

            successful_cases = successful_cases[
                successful_mask
            ].copy()

        if (
            not successful_cases.empty
            and "lead_seconds"
            in successful_cases.columns
        ):
            successful_cases[
                "lead_seconds"
            ] = pd.to_numeric(
                successful_cases[
                    "lead_seconds"
                ],
                errors="coerce",
            )

            successful_cases = (
                successful_cases.sort_values(
                    "lead_seconds",
                    ascending=False,
                    na_position="last",
                )
            )

            representative_case = (
                successful_cases
                .iloc[0]
                .to_dict()
            )

    # ============================================================
    # Early Warning 代表時間軸
    # ============================================================

    selected_epoch: int | None = None

    if representative_case:
        selected_epoch_value = (
            representative_case.get(
                "selected_alert_epoch_index"
            )
        )

        selected_epoch_number = safe_float(
            selected_epoch_value
        )

        if selected_epoch_number is not None:
            selected_epoch = int(
                round(
                    selected_epoch_number
                )
            )

    if (
        selected_epoch is not None
        and not arousal_early_warning_timeline_csv.empty
        and "epoch_index"
        in arousal_early_warning_timeline_csv.columns
    ):
        timeline_df = (
            arousal_early_warning_timeline_csv.copy()
        )

        timeline_df[
            "epoch_index"
        ] = pd.to_numeric(
            timeline_df[
                "epoch_index"
            ],
            errors="coerce",
        )

        matched = timeline_df[
            timeline_df[
                "epoch_index"
            ]
            == selected_epoch
        ]

        if not matched.empty:
            representative_timeline = (
                matched
                .iloc[0]
                .to_dict()
            )

        nearby_epochs = timeline_df[
            timeline_df[
                "epoch_index"
            ].between(
                selected_epoch - 4,
                selected_epoch + 1,
            )
        ].copy()

        nearby_epochs = (
            nearby_epochs.sort_values(
                "epoch_index"
            )
        )

        keep_columns = [
            "epoch_index",
            "start_time",
            "end_time",
            "stage",
            "arousal_next_30s_probability",
            "arousal_alert_threshold",
            "arousal_next_30s_alert",
            "arousal_risk_level",
            "future_arousal_exists",
            "matched_arousal_sequences",
            "alert_classification",
            "quality_any_issue",
        ]

        available_columns = [
            column
            for column in keep_columns
            if column in nearby_epochs.columns
        ]

        representative_timeline_window = (
            nearby_epochs[
                available_columns
            ].to_dict(
                orient="records"
            )
        )

    treatment_json = load_json_optional(
        treatment_file
    )

    print("=" * 80)
    print("Patient Digital Twin Report Builder")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    print(
        "Report Mode："
        + (
            "PSG Follow-up"
            if follow_up_mode
            else "Initial / Standard"
        )
    )

    print()
    print("來源資料：")

    source_files = [
        (
            "Respiratory Profile",
            respiratory_file,
        ),
        (
            "Factor Summary JSON",
            factor_json_file,
        ),
        (
            "Factor Summary CSV",
            factor_csv_file,
        ),
        (
            "SpO₂ Quality",
            spo2_quality_file,
        ),
        (
            "Oxygen Event Coupling",
            oxygen_file,
        ),
        (
            "Position Profile",
            position_file,
        ),
        (
            "Treatment Recommendation",
            treatment_file,
        ),
    ]

    for label, file_path in source_files:
        status = (
            "FOUND"
            if file_path.exists()
            else "NOT FOUND"
        )

        print(
            f"  [{status}] "
            f"{label}：{file_path}"
        )

    report = build_integrated_report(
        patient_id=patient_id,
        respiratory_json=(
            respiratory_json
        ),
        factor_json=(
            factor_json
        ),
        factor_csv=(
            factor_csv
        ),
        spo2_quality_json=(
            spo2_quality_json
        ),
        oxygen_json=(
            oxygen_json
        ),
        position_json=(
            position_json
        ),

        shap_group_csv=(
            shap_group_csv
        ),
        shap_feature_csv=(
            shap_feature_csv
        ),

        arousal_early_warning_json=(
            arousal_early_warning_json
        ),

        representative_case=(
            representative_case
        ),

        representative_timeline=(
            representative_timeline
        ),

        representative_timeline_window=(
            representative_timeline_window
        ),

        treatment_json=(
            treatment_json
        ),
        follow_up_mode=(
            follow_up_mode
        ),
    )

    output_folder = (
        patient_folder
        / "digital_twin_report"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_json = (
        output_folder
        / "patient_digital_twin_report.json"
    )

    output_csv = (
        output_folder
        / "patient_digital_twin_report.csv"
    )

    output_txt = (
        output_folder
        / "patient_digital_twin_report.txt"
    )

    output_html = (
        output_folder
        / "patient_digital_twin_report.html"
    )

    report_timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    previous_output_snapshots: list[str] = []

    if follow_up_mode:
        previous_output_snapshots = (
            snapshot_existing_outputs(
                output_folder=output_folder,
                output_files=[
                    output_json,
                    output_csv,
                    output_txt,
                    output_html,
                ],
                timestamp=report_timestamp,
            )
        )

    report.setdefault(
        "report_metadata",
        {},
    )[
        "previous_output_snapshots"
    ] = previous_output_snapshots

    atomic_save_json(
        output_json,
        report,
    )

    csv_row = flatten_report_for_csv(
        report
    )

    pd.DataFrame(
        [
            safe_json_value(
                csv_row
            )
        ]
    ).to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    text_report = build_text_report(
        report
    )

    output_txt.write_text(
        text_report,
        encoding="utf-8",
    )

    html_report = build_html_report(
        report
    )

    output_html.write_text(
        html_report,
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("數位孿生整合摘要")
    print("=" * 80)

    print(
        report[
            "digital_twin_narrative"
        ]
    )

    print()
    print("=" * 80)
    print("輸出檔案")
    print("=" * 80)

    print(
        f"完整 JSON：{output_json}"
    )

    print(
        f"摘要 CSV：{output_csv}"
    )

    print(
        f"文字報告：{output_txt}"
    )

    print(
        f"HTML 報告：{output_html}"
    )

    print()
    print(
        "注意：本報告為研究型患者數位孿生整合摘要，"
        "不可單獨作為正式臨床診斷或治療決策。"
    )

    print("=" * 80)


    


if __name__ == "__main__":
    main()
