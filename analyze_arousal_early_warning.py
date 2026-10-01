from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 專案路徑
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INFERENCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "inference"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# 終端機 UTF-8
# ============================================================

def configure_console_encoding() -> None:
    for stream_name in [
        "stdout",
        "stderr",
    ]:
        stream = getattr(
            sys,
            stream_name,
            None,
        )

        if stream is None:
            continue

        reconfigure = getattr(
            stream,
            "reconfigure",
            None,
        )

        if callable(reconfigure):
            try:
                reconfigure(
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception:
                pass


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
        "t",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "n",
        "f",
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
        pd.Timestamp,
    ):
        if pd.isna(value):
            return None

        return value.isoformat(
            sep=" "
        )

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    try:
        if pd.isna(value):
            return None
    except (
        TypeError,
        ValueError,
    ):
        pass

    return value


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


def finite_summary(
    values: pd.Series,
) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    numeric = numeric[
        np.isfinite(
            numeric
        )
    ]

    if numeric.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "p25": None,
            "p75": None,
        }

    return {
        "count": int(
            len(numeric)
        ),
        "mean": float(
            numeric.mean()
        ),
        "median": float(
            numeric.median()
        ),
        "minimum": float(
            numeric.min()
        ),
        "maximum": float(
            numeric.max()
        ),
        "p25": float(
            numeric.quantile(
                0.25
            )
        ),
        "p75": float(
            numeric.quantile(
                0.75
            )
        ),
    }


def format_number(
    value: Any,
    digits: int = 2,
) -> str:
    number = safe_float(
        value
    )

    if number is None:
        return "NOT_AVAILABLE"

    return f"{number:.{digits}f}"


def format_percentage(
    numerator: int,
    denominator: int,
    digits: int = 1,
) -> str:
    if denominator <= 0:
        return "NOT_AVAILABLE"

    percentage = (
        numerator
        / denominator
        * 100.0
    )

    return (
        f"{percentage:.{digits}f}%"
    )


# ============================================================
# 資料讀取與驗證
# ============================================================

def load_predictions(
    file_path: Path,
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            "找不到 Arousal 預測檔："
            f"{file_path}"
        )

    dataframe = pd.read_csv(
        file_path
    )

    required_columns = [
        "epoch_index",
        "start_time",
        "end_time",
        "arousal_next_30s_probability",
        "arousal_next_30s_alert",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Arousal 預測檔缺少必要欄位："
            + ", ".join(
                missing_columns
            )
        )

    dataframe = dataframe.copy()

    dataframe[
        "start_time"
    ] = pd.to_datetime(
        dataframe[
            "start_time"
        ],
        errors="coerce",
    )

    dataframe[
        "end_time"
    ] = pd.to_datetime(
        dataframe[
            "end_time"
        ],
        errors="coerce",
    )

    dataframe[
        "epoch_index"
    ] = pd.to_numeric(
        dataframe[
            "epoch_index"
        ],
        errors="coerce",
    )

    dataframe[
        "arousal_next_30s_probability"
    ] = pd.to_numeric(
        dataframe[
            "arousal_next_30s_probability"
        ],
        errors="coerce",
    )

    dataframe[
        "arousal_next_30s_alert"
    ] = dataframe[
        "arousal_next_30s_alert"
    ].map(
        safe_bool
    )

    if (
        "arousal_alert_threshold"
        in dataframe.columns
    ):
        dataframe[
            "arousal_alert_threshold"
        ] = pd.to_numeric(
            dataframe[
                "arousal_alert_threshold"
            ],
            errors="coerce",
        )

    valid_time_mask = (
        dataframe[
            "start_time"
        ].notna()
        & dataframe[
            "end_time"
        ].notna()
        & (
            dataframe[
                "end_time"
            ]
            > dataframe[
                "start_time"
            ]
        )
    )

    valid_probability_mask = (
        dataframe[
            "arousal_next_30s_probability"
        ].notna()
    )

    valid_alert_mask = (
        dataframe[
            "arousal_next_30s_alert"
        ].notna()
    )

    dataframe = dataframe[
        valid_time_mask
        & valid_probability_mask
        & valid_alert_mask
    ].copy()

    dataframe[
        "arousal_next_30s_alert"
    ] = dataframe[
        "arousal_next_30s_alert"
    ].astype(
        bool
    )

    dataframe = dataframe.sort_values(
        [
            "end_time",
            "epoch_index",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    return dataframe


def load_arousal_events(
    file_path: Path,
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            "找不到 Event 對齊檔："
            f"{file_path}"
        )

    dataframe = pd.read_csv(
        file_path
    )

    required_columns = [
        "event_type",
        "start_time",
        "end_time",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Event 對齊檔缺少必要欄位："
            + ", ".join(
                missing_columns
            )
        )

    dataframe = dataframe.copy()

    dataframe[
        "event_type"
    ] = dataframe[
        "event_type"
    ].astype(
        str
    ).str.strip().str.upper()

    arousal_mask = (
        dataframe[
            "event_type"
        ]
        == "AROUSAL"
    )

    if (
        "usable_for_event_training"
        in dataframe.columns
    ):
        usable_mask = dataframe[
            "usable_for_event_training"
        ].map(
            safe_bool
        )

        usable_mask = (
            usable_mask
            == True
        )
    else:
        usable_mask = pd.Series(
            True,
            index=dataframe.index,
        )

    dataframe = dataframe[
        arousal_mask
        & usable_mask
    ].copy()

    dataframe[
        "start_time"
    ] = pd.to_datetime(
        dataframe[
            "start_time"
        ],
        errors="coerce",
    )

    dataframe[
        "end_time"
    ] = pd.to_datetime(
        dataframe[
            "end_time"
        ],
        errors="coerce",
    )

    dataframe = dataframe[
        dataframe[
            "start_time"
        ].notna()
    ].copy()

    if "subtype" not in dataframe.columns:
        dataframe[
            "subtype"
        ] = "UNKNOWN"

    dataframe[
        "subtype"
    ] = dataframe[
        "subtype"
    ].fillna(
        "UNKNOWN"
    ).astype(
        str
    ).str.strip()

    if "event_index" not in dataframe.columns:
        dataframe[
            "event_index"
        ] = np.arange(
            len(dataframe)
        )

    dataframe[
        "event_index"
    ] = pd.to_numeric(
        dataframe[
            "event_index"
        ],
        errors="coerce",
    )

    dataframe = dataframe.sort_values(
        [
            "start_time",
            "event_index",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    dataframe[
        "arousal_sequence"
    ] = np.arange(
        1,
        len(dataframe) + 1,
    )

    return dataframe


# ============================================================
# 事件層級提前預警分析
# ============================================================

def analyze_event_level_warning(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    horizon_seconds: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    horizon_delta = pd.to_timedelta(
        horizon_seconds,
        unit="s",
    )

    timeline = predictions.copy()

    timeline[
        "prediction_window_start"
    ] = timeline[
        "end_time"
    ]

    timeline[
        "prediction_window_end"
    ] = (
        timeline[
            "end_time"
        ]
        + horizon_delta
    )

    timeline[
        "future_arousal_count"
    ] = 0

    timeline[
        "future_arousal_exists"
    ] = False

    timeline[
        "matched_arousal_sequences"
    ] = ""

    event_rows: list[
        dict[str, Any]
    ] = []

    for _, event in events.iterrows():
        event_start = event[
            "start_time"
        ]

        event_end = event.get(
            "end_time"
        )

        candidate_mask = (
            (
                timeline[
                    "prediction_window_start"
                ]
                <= event_start
            )
            & (
                event_start
                <= timeline[
                    "prediction_window_end"
                ]
            )
        )

        candidate_predictions = timeline[
            candidate_mask
        ].copy()

        if not candidate_predictions.empty:
            lead_seconds_series = (
                event_start
                - candidate_predictions[
                    "prediction_window_start"
                ]
            ).dt.total_seconds()

            candidate_predictions[
                "lead_seconds"
            ] = lead_seconds_series

            candidate_predictions = (
                candidate_predictions[
                    (
                        candidate_predictions[
                            "lead_seconds"
                        ]
                        >= 0.0
                    )
                    & (
                        candidate_predictions[
                            "lead_seconds"
                        ]
                        <= horizon_seconds
                    )
                ].copy()
            )

        alert_candidates = candidate_predictions[
            candidate_predictions[
                "arousal_next_30s_alert"
            ]
        ].copy()

        predicted_probability = None
        predicted_epoch_index = None
        predicted_epoch_start = None
        predicted_epoch_end = None
        lead_seconds = None
        successfully_warned = False

        if not alert_candidates.empty:
            alert_candidates = (
                alert_candidates.sort_values(
                    [
                        "lead_seconds",
                        "arousal_next_30s_probability",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )
            )

            selected_alert = (
                alert_candidates.iloc[0]
            )

            successfully_warned = True

            predicted_probability = safe_float(
                selected_alert.get(
                    "arousal_next_30s_probability"
                )
            )

            predicted_epoch_index = safe_int(
                selected_alert.get(
                    "epoch_index"
                )
            )

            predicted_epoch_start = (
                selected_alert.get(
                    "start_time"
                )
            )

            predicted_epoch_end = (
                selected_alert.get(
                    "end_time"
                )
            )

            lead_seconds = safe_float(
                selected_alert.get(
                    "lead_seconds"
                )
            )

        maximum_pre_event_probability = None
        closest_pre_event_probability = None
        closest_pre_event_lead_seconds = None

        if not candidate_predictions.empty:
            probability_series = pd.to_numeric(
                candidate_predictions[
                    "arousal_next_30s_probability"
                ],
                errors="coerce",
            )

            finite_probability = probability_series[
                np.isfinite(
                    probability_series
                )
            ]

            if not finite_probability.empty:
                maximum_pre_event_probability = (
                    float(
                        finite_probability.max()
                    )
                )

            closest_candidates = (
                candidate_predictions.sort_values(
                    "lead_seconds",
                    ascending=True,
                )
            )

            if not closest_candidates.empty:
                closest_row = (
                    closest_candidates.iloc[0]
                )

                closest_pre_event_probability = (
                    safe_float(
                        closest_row.get(
                            "arousal_next_30s_probability"
                        )
                    )
                )

                closest_pre_event_lead_seconds = (
                    safe_float(
                        closest_row.get(
                            "lead_seconds"
                        )
                    )
                )

        event_rows.append(
            {
                "arousal_sequence": safe_int(
                    event.get(
                        "arousal_sequence"
                    )
                ),
                "event_index": safe_int(
                    event.get(
                        "event_index"
                    )
                ),
                "subtype": event.get(
                    "subtype"
                ),
                "arousal_start_time": (
                    event_start
                ),
                "arousal_end_time": (
                    event_end
                ),
                "successfully_warned": (
                    successfully_warned
                ),
                "lead_seconds": (
                    lead_seconds
                ),
                "selected_alert_epoch_index": (
                    predicted_epoch_index
                ),
                "selected_alert_epoch_start": (
                    predicted_epoch_start
                ),
                "selected_alert_epoch_end": (
                    predicted_epoch_end
                ),
                "selected_alert_probability": (
                    predicted_probability
                ),
                "maximum_pre_event_probability": (
                    maximum_pre_event_probability
                ),
                "closest_pre_event_probability": (
                    closest_pre_event_probability
                ),
                "closest_pre_event_lead_seconds": (
                    closest_pre_event_lead_seconds
                ),
                "candidate_prediction_count": int(
                    len(
                        candidate_predictions
                    )
                ),
                "candidate_alert_count": int(
                    len(
                        alert_candidates
                    )
                ),
            }
        )

        if not candidate_predictions.empty:
            candidate_indices = (
                candidate_predictions.index
            )

            timeline.loc[
                candidate_indices,
                "future_arousal_count",
            ] = (
                timeline.loc[
                    candidate_indices,
                    "future_arousal_count",
                ]
                + 1
            )

            sequence_text = str(
                safe_int(
                    event.get(
                        "arousal_sequence"
                    )
                )
            )

            for timeline_index in candidate_indices:
                existing_text = str(
                    timeline.at[
                        timeline_index,
                        "matched_arousal_sequences",
                    ]
                ).strip()

                if existing_text:
                    timeline.at[
                        timeline_index,
                        "matched_arousal_sequences",
                    ] = (
                        existing_text
                        + ";"
                        + sequence_text
                    )
                else:
                    timeline.at[
                        timeline_index,
                        "matched_arousal_sequences",
                    ] = sequence_text

    timeline[
        "future_arousal_exists"
    ] = (
        timeline[
            "future_arousal_count"
        ]
        > 0
    )

    timeline[
        "alert_classification"
    ] = np.select(
        [
            (
                timeline[
                    "arousal_next_30s_alert"
                ]
                & timeline[
                    "future_arousal_exists"
                ]
            ),
            (
                timeline[
                    "arousal_next_30s_alert"
                ]
                & ~timeline[
                    "future_arousal_exists"
                ]
            ),
            (
                ~timeline[
                    "arousal_next_30s_alert"
                ]
                & timeline[
                    "future_arousal_exists"
                ]
            ),
        ],
        [
            "TRUE_ALERT_EPOCH",
            "FALSE_ALERT_EPOCH",
            "MISSED_WARNING_EPOCH",
        ],
        default="TRUE_NON_ALERT_EPOCH",
    )

    event_results = pd.DataFrame(
        event_rows
    )

    return (
        event_results,
        timeline,
    )


# ============================================================
# 連續誤警報合併
# ============================================================

def build_false_alert_episodes(
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    false_alerts = timeline[
        timeline[
            "alert_classification"
        ]
        == "FALSE_ALERT_EPOCH"
    ].copy()

    if false_alerts.empty:
        return pd.DataFrame(
            columns=[
                "false_alert_episode",
                "episode_start_time",
                "episode_end_time",
                "epoch_count",
                "duration_seconds",
                "maximum_probability",
                "mean_probability",
            ]
        )

    false_alerts = false_alerts.sort_values(
        [
            "start_time",
            "end_time",
        ]
    ).reset_index(
        drop=True
    )

    episode_numbers: list[int] = []

    current_episode = 0
    previous_end: pd.Timestamp | None = None

    for _, row in false_alerts.iterrows():
        row_start = row[
            "start_time"
        ]

        if (
            previous_end is None
            or pd.isna(
                previous_end
            )
            or row_start > previous_end
        ):
            current_episode += 1

        episode_numbers.append(
            current_episode
        )

        row_end = row[
            "end_time"
        ]

        if (
            previous_end is None
            or pd.isna(
                previous_end
            )
            or row_end > previous_end
        ):
            previous_end = row_end

    false_alerts[
        "false_alert_episode"
    ] = episode_numbers

    episode_rows: list[
        dict[str, Any]
    ] = []

    for episode_number, group in (
        false_alerts.groupby(
            "false_alert_episode",
            sort=True,
        )
    ):
        episode_start = group[
            "start_time"
        ].min()

        episode_end = group[
            "end_time"
        ].max()

        duration_seconds = (
            episode_end
            - episode_start
        ).total_seconds()

        probabilities = pd.to_numeric(
            group[
                "arousal_next_30s_probability"
            ],
            errors="coerce",
        )

        finite_probabilities = probabilities[
            np.isfinite(
                probabilities
            )
        ]

        episode_rows.append(
            {
                "false_alert_episode": int(
                    episode_number
                ),
                "episode_start_time": (
                    episode_start
                ),
                "episode_end_time": (
                    episode_end
                ),
                "epoch_count": int(
                    len(group)
                ),
                "duration_seconds": float(
                    duration_seconds
                ),
                "maximum_probability": (
                    float(
                        finite_probabilities.max()
                    )
                    if not finite_probabilities.empty
                    else None
                ),
                "mean_probability": (
                    float(
                        finite_probabilities.mean()
                    )
                    if not finite_probabilities.empty
                    else None
                ),
            }
        )

    return pd.DataFrame(
        episode_rows
    )


# ============================================================
# Subtype 摘要
# ============================================================

def build_subtype_summary(
    event_results: pd.DataFrame,
) -> pd.DataFrame:
    if event_results.empty:
        return pd.DataFrame(
            columns=[
                "subtype",
                "arousal_count",
                "successfully_warned_count",
                "warning_recall",
                "lead_seconds_mean",
                "lead_seconds_median",
                "lead_seconds_min",
                "lead_seconds_max",
            ]
        )

    rows: list[
        dict[str, Any]
    ] = []

    for subtype, group in (
        event_results.groupby(
            "subtype",
            dropna=False,
            sort=True,
        )
    ):
        event_count = int(
            len(group)
        )

        warned_mask = (
            group[
                "successfully_warned"
            ]
            == True
        )

        warned_count = int(
            warned_mask.sum()
        )

        lead_summary = finite_summary(
            group.loc[
                warned_mask,
                "lead_seconds",
            ]
        )

        rows.append(
            {
                "subtype": (
                    subtype
                    if pd.notna(
                        subtype
                    )
                    else "UNKNOWN"
                ),
                "arousal_count": (
                    event_count
                ),
                "successfully_warned_count": (
                    warned_count
                ),
                "warning_recall": (
                    warned_count
                    / event_count
                    if event_count > 0
                    else None
                ),
                "lead_seconds_mean": (
                    lead_summary[
                        "mean"
                    ]
                ),
                "lead_seconds_median": (
                    lead_summary[
                        "median"
                    ]
                ),
                "lead_seconds_min": (
                    lead_summary[
                        "minimum"
                    ]
                ),
                "lead_seconds_max": (
                    lead_summary[
                        "maximum"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 整體摘要
# ============================================================

def build_summary(
    patient_id: str,
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    event_results: pd.DataFrame,
    timeline: pd.DataFrame,
    false_alert_episodes: pd.DataFrame,
    subtype_summary: pd.DataFrame,
    horizon_seconds: float,
    predictions_file: Path,
    events_file: Path,
) -> dict[str, Any]:
    true_event_count = int(
        len(
            event_results
        )
    )

    warned_event_count = int(
        (
            event_results[
                "successfully_warned"
            ]
            == True
        ).sum()
    )

    missed_event_count = (
        true_event_count
        - warned_event_count
    )

    warning_recall = (
        warned_event_count
        / true_event_count
        if true_event_count > 0
        else None
    )

    warned_lead_seconds = (
        event_results.loc[
            event_results[
                "successfully_warned"
            ]
            == True,
            "lead_seconds",
        ]
    )

    lead_summary = finite_summary(
        warned_lead_seconds
    )

    alert_epoch_count = int(
        timeline[
            "arousal_next_30s_alert"
        ].sum()
    )

    true_alert_epoch_count = int(
        (
            timeline[
                "alert_classification"
            ]
            == "TRUE_ALERT_EPOCH"
        ).sum()
    )

    false_alert_epoch_count = int(
        (
            timeline[
                "alert_classification"
            ]
            == "FALSE_ALERT_EPOCH"
        ).sum()
    )

    false_alert_episode_count = int(
        len(
            false_alert_episodes
        )
    )

    if predictions.empty:
        monitoring_seconds = 0.0
    else:
        monitoring_start = predictions[
            "start_time"
        ].min()

        monitoring_end = predictions[
            "end_time"
        ].max()

        monitoring_seconds = float(
            (
                monitoring_end
                - monitoring_start
            ).total_seconds()
        )

    monitoring_hours = (
        monitoring_seconds
        / 3600.0
    )

    false_alert_epochs_per_hour = (
        false_alert_epoch_count
        / monitoring_hours
        if monitoring_hours > 0
        else None
    )

    false_alert_episodes_per_hour = (
        false_alert_episode_count
        / monitoring_hours
        if monitoring_hours > 0
        else None
    )

    alert_positive_predictive_value = (
        true_alert_epoch_count
        / alert_epoch_count
        if alert_epoch_count > 0
        else None
    )

    threshold = None

    if (
        "arousal_alert_threshold"
        in predictions.columns
    ):
        threshold_values = pd.to_numeric(
            predictions[
                "arousal_alert_threshold"
            ],
            errors="coerce",
        )

        threshold_values = threshold_values[
            np.isfinite(
                threshold_values
            )
        ]

        if not threshold_values.empty:
            threshold = float(
                threshold_values.median()
            )

    subtype_records = (
        subtype_summary.to_dict(
            orient="records"
        )
        if not subtype_summary.empty
        else []
    )

    interpretation_parts: list[str] = []

    interpretation_parts.append(
        f"共有 {true_event_count} 次可用真實 Arousal。"
    )

    interpretation_parts.append(
        f"其中 {warned_event_count} 次在事件開始前最多 "
        f"{horizon_seconds:.0f} 秒的預測窗內出現模型警報。"
    )

    if warning_recall is not None:
        interpretation_parts.append(
            "事件層級提前預警召回率約為 "
            f"{warning_recall * 100.0:.1f}%。"
        )

    if lead_summary[
        "median"
    ] is not None:
        interpretation_parts.append(
            "成功預警事件的中位提前時間約為 "
            f"{lead_summary['median']:.1f} 秒。"
        )

    if false_alert_episodes_per_hour is not None:
        interpretation_parts.append(
            "合併連續警報後，每小時約有 "
            f"{false_alert_episodes_per_hour:.2f} 個誤警報事件。"
        )

    interpretation_parts.append(
        "此分析以 Epoch 結束後的未來 30 秒作為預測窗，"
        "屬研究型離線評估，並不代表已完成即時臨床警報驗證。"
    )

    return {
        "patient_id": patient_id,
        "analysis_name": (
            "AROUSAL_EARLY_WARNING_ANALYSIS"
        ),
        "model_target": (
            "predict_arousal_next_30s"
        ),
        "prediction_horizon_seconds": (
            float(
                horizon_seconds
            )
        ),
        "matching_rule": (
            "若真實 Arousal 開始時間落在某 Epoch 結束後的未來"
            f" {horizon_seconds:.0f} 秒內，則該 Epoch 可預測此事件；"
            "若該 Epoch 同時為 alert=True，視為成功提前預警。"
        ),
        "source_files": {
            "predictions": str(
                predictions_file
            ),
            "events": str(
                events_file
            ),
        },
        "data_counts": {
            "prediction_epoch_count": int(
                len(
                    predictions
                )
            ),
            "true_arousal_count": (
                true_event_count
            ),
            "monitoring_seconds": (
                monitoring_seconds
            ),
            "monitoring_hours": (
                monitoring_hours
            ),
        },
        "event_level_warning": {
            "successfully_warned_count": (
                warned_event_count
            ),
            "missed_event_count": (
                missed_event_count
            ),
            "warning_recall": (
                warning_recall
            ),
            "lead_seconds": (
                lead_summary
            ),
        },
        "alert_level_performance": {
            "alert_threshold": (
                threshold
            ),
            "alert_epoch_count": (
                alert_epoch_count
            ),
            "true_alert_epoch_count": (
                true_alert_epoch_count
            ),
            "false_alert_epoch_count": (
                false_alert_epoch_count
            ),
            "false_alert_episode_count": (
                false_alert_episode_count
            ),
            "alert_epoch_positive_predictive_value": (
                alert_positive_predictive_value
            ),
            "false_alert_epochs_per_hour": (
                false_alert_epochs_per_hour
            ),
            "false_alert_episodes_per_hour": (
                false_alert_episodes_per_hour
            ),
        },
        "subtype_results": (
            subtype_records
        ),
        "interpretation": (
            "".join(
                interpretation_parts
            )
        ),
        "safety_note": (
            "此結果為研究型離線事件時間匹配分析。"
            "它不能證明模型可安全用於即時醫療警報，"
            "也不可單獨作為診斷或治療依據。"
        ),
    }


def flatten_summary_for_csv(
    summary: dict[str, Any],
) -> dict[str, Any]:
    data_counts = summary.get(
        "data_counts",
        {},
    )

    event_level = summary.get(
        "event_level_warning",
        {},
    )

    alert_level = summary.get(
        "alert_level_performance",
        {},
    )

    lead_seconds = event_level.get(
        "lead_seconds",
        {},
    )

    return {
        "patient_id": summary.get(
            "patient_id"
        ),
        "model_target": summary.get(
            "model_target"
        ),
        "prediction_horizon_seconds": (
            summary.get(
                "prediction_horizon_seconds"
            )
        ),
        "prediction_epoch_count": (
            data_counts.get(
                "prediction_epoch_count"
            )
        ),
        "true_arousal_count": (
            data_counts.get(
                "true_arousal_count"
            )
        ),
        "monitoring_hours": (
            data_counts.get(
                "monitoring_hours"
            )
        ),
        "successfully_warned_count": (
            event_level.get(
                "successfully_warned_count"
            )
        ),
        "missed_event_count": (
            event_level.get(
                "missed_event_count"
            )
        ),
        "warning_recall": (
            event_level.get(
                "warning_recall"
            )
        ),
        "lead_seconds_mean": (
            lead_seconds.get(
                "mean"
            )
        ),
        "lead_seconds_median": (
            lead_seconds.get(
                "median"
            )
        ),
        "lead_seconds_min": (
            lead_seconds.get(
                "minimum"
            )
        ),
        "lead_seconds_max": (
            lead_seconds.get(
                "maximum"
            )
        ),
        "alert_threshold": (
            alert_level.get(
                "alert_threshold"
            )
        ),
        "alert_epoch_count": (
            alert_level.get(
                "alert_epoch_count"
            )
        ),
        "true_alert_epoch_count": (
            alert_level.get(
                "true_alert_epoch_count"
            )
        ),
        "false_alert_epoch_count": (
            alert_level.get(
                "false_alert_epoch_count"
            )
        ),
        "false_alert_episode_count": (
            alert_level.get(
                "false_alert_episode_count"
            )
        ),
        "alert_epoch_positive_predictive_value": (
            alert_level.get(
                "alert_epoch_positive_predictive_value"
            )
        ),
        "false_alert_epochs_per_hour": (
            alert_level.get(
                "false_alert_epochs_per_hour"
            )
        ),
        "false_alert_episodes_per_hour": (
            alert_level.get(
                "false_alert_episodes_per_hour"
            )
        ),
        "interpretation": summary.get(
            "interpretation"
        ),
    }


# ============================================================
# 終端輸出
# ============================================================

def print_summary(
    summary: dict[str, Any],
    subtype_summary: pd.DataFrame,
) -> None:
    data_counts = summary[
        "data_counts"
    ]

    event_level = summary[
        "event_level_warning"
    ]

    alert_level = summary[
        "alert_level_performance"
    ]

    lead_seconds = event_level[
        "lead_seconds"
    ]

    print("=" * 80)
    print("Arousal Early Warning Analysis")
    print("=" * 80)

    print(
        f"Patient ID："
        f"{summary['patient_id']}"
    )

    print(
        "模型目標："
        f"{summary['model_target']}"
    )

    print(
        "預測時間窗："
        f"{summary['prediction_horizon_seconds']:.0f} 秒"
    )

    print()

    print("=" * 80)
    print("資料摘要")
    print("=" * 80)

    print(
        "預測 Epoch 數："
        f"{data_counts['prediction_epoch_count']}"
    )

    print(
        "可用真實 Arousal 數："
        f"{data_counts['true_arousal_count']}"
    )

    print(
        "監測時間："
        f"{format_number(data_counts['monitoring_hours'], 2)} 小時"
    )

    print()

    print("=" * 80)
    print("事件層級提前預警")
    print("=" * 80)

    warned_count = int(
        event_level[
            "successfully_warned_count"
        ]
    )

    true_count = int(
        data_counts[
            "true_arousal_count"
        ]
    )

    print(
        "成功提前預警："
        f"{warned_count}/{true_count} "
        f"({format_percentage(warned_count, true_count)})"
    )

    print(
        "未成功提前預警："
        f"{event_level['missed_event_count']}"
    )

    print(
        "平均提前時間："
        f"{format_number(lead_seconds['mean'], 2)} 秒"
    )

    print(
        "中位提前時間："
        f"{format_number(lead_seconds['median'], 2)} 秒"
    )

    print(
        "最短提前時間："
        f"{format_number(lead_seconds['minimum'], 2)} 秒"
    )

    print(
        "最長提前時間："
        f"{format_number(lead_seconds['maximum'], 2)} 秒"
    )

    print()

    print("=" * 80)
    print("警報層級結果")
    print("=" * 80)

    print(
        "警報閾值："
        f"{format_number(alert_level['alert_threshold'], 4)}"
    )

    print(
        "警報 Epoch 數："
        f"{alert_level['alert_epoch_count']}"
    )

    print(
        "命中未來 Arousal 的警報 Epoch："
        f"{alert_level['true_alert_epoch_count']}"
    )

    print(
        "未命中未來 Arousal 的警報 Epoch："
        f"{alert_level['false_alert_epoch_count']}"
    )

    print(
        "連續誤警合併後的誤警事件數："
        f"{alert_level['false_alert_episode_count']}"
    )

    alert_ppv = safe_float(
        alert_level.get(
            "alert_epoch_positive_predictive_value"
        )
    )

    alert_ppv_percent = (
        alert_ppv * 100.0
        if alert_ppv is not None
        else None
    )

    print(
        "警報 Epoch 陽性預測值："
        f"{format_number(alert_ppv_percent, 1)}%"
    )

    false_alert_epochs_per_hour = safe_float(
        alert_level.get(
            "false_alert_epochs_per_hour"
        )
    )

    print(
        "每小時誤警 Epoch："
        f"{format_number(false_alert_epochs_per_hour, 2)}"
    )

    false_alert_episodes_per_hour = safe_float(
        alert_level.get(
            "false_alert_episodes_per_hour"
        )
    )

    print(
        "每小時誤警事件："
        f"{format_number(false_alert_episodes_per_hour, 2)}"
    )

    print()

    print("=" * 80)
    print("Arousal Subtype 結果")
    print("=" * 80)

    if subtype_summary.empty:
        print("沒有 subtype 結果。")
    else:
        display_columns = [
            "subtype",
            "arousal_count",
            "successfully_warned_count",
            "warning_recall",
            "lead_seconds_mean",
            "lead_seconds_median",
        ]

        print(
            subtype_summary[
                display_columns
            ].to_string(
                index=False
            )
        )

    print()

    print("=" * 80)
    print("研究型解讀")
    print("=" * 80)

    print(
        summary[
            "interpretation"
        ]
    )

    print()

    print(
        "注意："
        f"{summary['safety_note']}"
    )

    print("=" * 80)


# ============================================================
# 主程式
# ============================================================

def main() -> None:
    configure_console_encoding()

    parser = argparse.ArgumentParser(
        description=(
            "以真實 Arousal 事件與逐 Epoch 未來 30 秒預測，"
            "評估患者層級 Arousal 提前預警能力。"
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
        "--horizon-seconds",
        type=float,
        default=30.0,
        help=(
            "預測未來時間窗秒數。"
            "目前模型目標為 next 30s，預設 30。"
        ),
    )

    args = parser.parse_args()

    patient_id = str(
        args.patient_id
    ).strip()

    horizon_seconds = float(
        args.horizon_seconds
    )

    if horizon_seconds <= 0:
        raise ValueError(
            "--horizon-seconds 必須大於 0。"
        )

    patient_inference_folder = (
        INFERENCE_ROOT
        / patient_id
    )

    patient_processed_folder = (
        PROCESSED_ROOT
        / patient_id
    )

    predictions_file = (
        patient_inference_folder
        / "arousal_predictions.csv"
    )

    events_file = (
        patient_processed_folder
        / "events_aligned.csv"
    )

    output_folder = (
        patient_inference_folder
        / "arousal_early_warning"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_results_file = (
        output_folder
        / "arousal_early_warning_events.csv"
    )

    timeline_file = (
        output_folder
        / "arousal_early_warning_timeline.csv"
    )

    false_alert_episodes_file = (
        output_folder
        / "false_alert_episodes.csv"
    )

    subtype_summary_file = (
        output_folder
        / "arousal_early_warning_subtype_summary.csv"
    )

    summary_csv_file = (
        output_folder
        / "arousal_early_warning_summary.csv"
    )

    summary_json_file = (
        output_folder
        / "arousal_early_warning_summary.json"
    )

    print("=" * 80)
    print("載入資料")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    print(
        f"Predictions：{predictions_file}"
    )

    print(
        f"Events：{events_file}"
    )

    predictions = load_predictions(
        predictions_file
    )

    events = load_arousal_events(
        events_file
    )

    print(
        f"有效預測 Epoch：{len(predictions)}"
    )

    print(
        f"可用 Arousal：{len(events)}"
    )

    print()

    event_results, timeline = (
        analyze_event_level_warning(
            predictions=predictions,
            events=events,
            horizon_seconds=(
                horizon_seconds
            ),
        )
    )

    false_alert_episodes = (
        build_false_alert_episodes(
            timeline
        )
    )

    subtype_summary = (
        build_subtype_summary(
            event_results
        )
    )

    summary = build_summary(
        patient_id=patient_id,
        predictions=predictions,
        events=events,
        event_results=event_results,
        timeline=timeline,
        false_alert_episodes=(
            false_alert_episodes
        ),
        subtype_summary=(
            subtype_summary
        ),
        horizon_seconds=(
            horizon_seconds
        ),
        predictions_file=(
            predictions_file
        ),
        events_file=(
            events_file
        ),
    )

    event_results.to_csv(
        event_results_file,
        index=False,
        encoding="utf-8-sig",
    )

    timeline.to_csv(
        timeline_file,
        index=False,
        encoding="utf-8-sig",
    )

    false_alert_episodes.to_csv(
        false_alert_episodes_file,
        index=False,
        encoding="utf-8-sig",
    )

    subtype_summary.to_csv(
        subtype_summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            flatten_summary_for_csv(
                summary
            )
        ]
    ).to_csv(
        summary_csv_file,
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        summary_json_file,
        summary,
    )

    print_summary(
        summary=summary,
        subtype_summary=(
            subtype_summary
        ),
    )

    print()

    print("=" * 80)
    print("輸出檔案")
    print("=" * 80)

    print(
        "逐 Arousal 事件結果："
        f"{event_results_file}"
    )

    print(
        "逐 Epoch 警報時間軸："
        f"{timeline_file}"
    )

    print(
        "誤警報事件："
        f"{false_alert_episodes_file}"
    )

    print(
        "Subtype 摘要："
        f"{subtype_summary_file}"
    )

    print(
        "摘要 CSV："
        f"{summary_csv_file}"
    )

    print(
        "摘要 JSON："
        f"{summary_json_file}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()