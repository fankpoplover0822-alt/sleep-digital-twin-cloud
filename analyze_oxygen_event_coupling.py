from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

INCOMING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "incoming"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INFERENCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "inference"
)


RESPIRATORY_EVENT_TYPES = {
    "HYPOPNEA",
    "OBSTRUCTIVE_APNEA",
    "CENTRAL_APNEA",
    "MIXED_APNEA",
}


SPO2_CHANNEL_KEYWORDS = [
    "saturation",
    "spo2",
    "sao2",
    "oxygen saturation",
    "oximetry",
]


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
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    denominator_value = safe_float(
        denominator
    )

    if (
        denominator_value is None
        or denominator_value == 0
    ):
        return None

    result = float(
        numerator
    ) / denominator_value

    if not np.isfinite(result):
        return None

    return result


def safe_json_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(value, dict):
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
        (
            pd.Timestamp,
            datetime,
        ),
    ):
        return pd.Timestamp(
            value
        ).isoformat(
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


def load_csv(
    file_path: Path,
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到 CSV：{file_path}"
        )

    return pd.read_csv(
        file_path
    )


def normalize_datetime(
    value: Any,
) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(timestamp):
        return None

    timestamp = pd.Timestamp(
        timestamp
    )

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(
            None
        )

    return timestamp


def normalize_boolean_series(
    values: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        values
    ):
        return values.fillna(False)

    normalized = (
        values.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        [
            "true",
            "1",
            "yes",
            "y",
        ]
    )


# ============================================================
# 找檔案與 Channel
# ============================================================

def find_edf_file(
    patient_folder: Path,
) -> Path:
    if not patient_folder.exists():
        raise FileNotFoundError(
            f"找不到患者資料夾：{patient_folder}"
        )

    edf_files = sorted(
        patient_folder.glob("*.edf")
    )

    if not edf_files:
        raise FileNotFoundError(
            f"患者資料夾內沒有 EDF：{patient_folder}"
        )

    exact_matches = [
        path
        for path in edf_files
        if path.name.endswith(
            "_EDF.edf"
        )
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    return max(
        edf_files,
        key=lambda path: path.stat().st_size,
    )


def find_spo2_channel(
    channel_names: list[str],
) -> str:
    for channel_name in channel_names:
        normalized = (
            str(channel_name)
            .strip()
            .lower()
        )

        for keyword in SPO2_CHANNEL_KEYWORDS:
            if keyword in normalized:
                return channel_name

    raise RuntimeError(
        "EDF 中找不到 Saturation／SpO₂ Channel。"
    )


# ============================================================
# EDF 與時間軸
# ============================================================

def read_measurement_start(
    raw: mne.io.BaseRaw,
    metadata: pd.DataFrame,
) -> pd.Timestamp:
    """
    優先使用前處理階段校正後的 EDF 開始時間。

    若 metadata 無法使用，再退回 EDF header。
    """
    if not metadata.empty:
        candidate_columns = [
            "corrected_edf_start",
            "edf_start",
            "original_edf_start",
        ]

        for column in candidate_columns:
            if column not in metadata.columns:
                continue

            value = metadata.iloc[0][
                column
            ]

            timestamp = normalize_datetime(
                value
            )

            if timestamp is not None:
                return timestamp

    meas_date = raw.info.get(
        "meas_date"
    )

    timestamp = normalize_datetime(
        meas_date
    )

    if timestamp is None:
        raise RuntimeError(
            "無法取得 EDF 開始時間。"
        )

    return timestamp


def normalize_spo2_scale(
    values: np.ndarray,
) -> tuple[
    np.ndarray,
    str,
]:
    """
    將 SpO₂ 訊號標準化到百分比尺度。

    常見可能形式：
    - 0.90～1.00
    - 90～100
    - 某些 EDF 經 MNE 後可能有額外縮放
    """
    data = np.asarray(
        values,
        dtype=float,
    ).copy()

    finite = data[
        np.isfinite(data)
    ]

    if len(finite) == 0:
        return data, "NO_FINITE_DATA"

    median_value = float(
        np.median(finite)
    )

    p95_value = float(
        np.percentile(
            finite,
            95,
        )
    )

    if (
        median_value >= 0.40
        and p95_value <= 1.50
    ):
        data = data * 100.0
        return data, "FRACTION_TO_PERCENT"

    if (
        median_value >= 40.0
        and p95_value <= 120.0
    ):
        return data, "ALREADY_PERCENT"

    # MNE 對部分 EDF auxiliary channel
    # 可能套用 1e-6 類型的尺度。
    if (
        median_value > 0
        and median_value < 0.01
    ):
        candidate = data * 1_000_000.0

        candidate_finite = candidate[
            np.isfinite(candidate)
        ]

        candidate_median = float(
            np.median(
                candidate_finite
            )
        )

        if (
            40.0
            <= candidate_median
            <= 110.0
        ):
            return (
                candidate,
                "MICRO_SCALE_TO_PERCENT",
            )

    return data, "UNDETERMINED_SCALE"


def create_valid_spo2_mask(
    spo2: np.ndarray,
) -> np.ndarray:
    return (
        np.isfinite(spo2)
        & (spo2 >= 40.0)
        & (spo2 <= 100.0)
    )


def sample_to_timestamp(
    sample_index: int,
    measurement_start: pd.Timestamp,
    sampling_rate: float,
) -> pd.Timestamp:
    return (
        measurement_start
        + pd.to_timedelta(
            sample_index
            / sampling_rate,
            unit="s",
        )
    )


def timestamp_to_sample(
    timestamp: pd.Timestamp,
    measurement_start: pd.Timestamp,
    sampling_rate: float,
    total_samples: int,
) -> int:
    seconds = (
        timestamp
        - measurement_start
    ).total_seconds()

    sample_index = int(
        round(
            seconds
            * sampling_rate
        )
    )

    return int(
        np.clip(
            sample_index,
            0,
            total_samples,
        )
    )


# ============================================================
# 訊號摘要
# ============================================================

def summarize_window(
    spo2: np.ndarray,
    valid_mask: np.ndarray,
    start_sample: int,
    end_sample: int,
) -> dict[str, Any]:
    start = max(
        int(start_sample),
        0,
    )

    end = min(
        int(end_sample),
        len(spo2),
    )

    if end <= start:
        return {
            "sample_count": 0,
            "valid_count": 0,
            "valid_fraction": None,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p05": None,
            "p95": None,
        }

    window = spo2[
        start:end
    ]

    valid = valid_mask[
        start:end
    ]

    valid_values = window[
        valid
    ]

    sample_count = int(
        len(window)
    )

    valid_count = int(
        len(valid_values)
    )

    if valid_count == 0:
        return {
            "sample_count": sample_count,
            "valid_count": 0,
            "valid_fraction": 0.0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p05": None,
            "p95": None,
        }

    return {
        "sample_count": sample_count,
        "valid_count": valid_count,
        "valid_fraction": safe_divide(
            valid_count,
            sample_count,
        ),
        "mean": safe_float(
            np.mean(valid_values)
        ),
        "median": safe_float(
            np.median(valid_values)
        ),
        "min": safe_float(
            np.min(valid_values)
        ),
        "max": safe_float(
            np.max(valid_values)
        ),
        "p05": safe_float(
            np.percentile(
                valid_values,
                5,
            )
        ),
        "p95": safe_float(
            np.percentile(
                valid_values,
                95,
            )
        ),
    }


def find_nadir(
    spo2: np.ndarray,
    valid_mask: np.ndarray,
    start_sample: int,
    end_sample: int,
) -> tuple[
    float | None,
    int | None,
]:
    start = max(
        int(start_sample),
        0,
    )

    end = min(
        int(end_sample),
        len(spo2),
    )

    if end <= start:
        return None, None

    values = spo2[
        start:end
    ]

    valid = valid_mask[
        start:end
    ]

    valid_indices = np.flatnonzero(
        valid
    )

    if len(valid_indices) == 0:
        return None, None

    valid_values = values[
        valid_indices
    ]

    minimum_local_valid_index = int(
        np.argmin(
            valid_values
        )
    )

    local_sample = int(
        valid_indices[
            minimum_local_valid_index
        ]
    )

    absolute_sample = (
        start
        + local_sample
    )

    minimum_value = float(
        spo2[
            absolute_sample
        ]
    )

    return (
        minimum_value,
        absolute_sample,
    )


# ============================================================
# 每個呼吸事件分析
# ============================================================

def analyze_respiratory_events(
    events: pd.DataFrame,
    spo2: np.ndarray,
    valid_mask: np.ndarray,
    measurement_start: pd.Timestamp,
    sampling_rate: float,
    total_samples: int,
    baseline_seconds: float,
    post_event_seconds: float,
) -> pd.DataFrame:
    respiratory_events = events.copy()

    respiratory_events[
        "event_type"
    ] = (
        respiratory_events[
            "event_type"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    respiratory_events = respiratory_events[
        respiratory_events[
            "event_type"
        ].isin(
            RESPIRATORY_EVENT_TYPES
        )
    ].copy()

    if (
        "usable_for_event_training"
        in respiratory_events.columns
    ):
        usable_mask = (
            normalize_boolean_series(
                respiratory_events[
                    "usable_for_event_training"
                ]
            )
        )

        respiratory_events = (
            respiratory_events[
                usable_mask
            ].copy()
        )

    respiratory_events[
        "start_time"
    ] = pd.to_datetime(
        respiratory_events[
            "start_time"
        ],
        errors="coerce",
    )

    respiratory_events[
        "end_time"
    ] = pd.to_datetime(
        respiratory_events[
            "end_time"
        ],
        errors="coerce",
    )

    respiratory_events = (
        respiratory_events.dropna(
            subset=[
                "start_time",
                "end_time",
            ]
        )
        .sort_values(
            "start_time"
        )
        .reset_index(
            drop=True
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    baseline_samples = int(
        round(
            baseline_seconds
            * sampling_rate
        )
    )

    post_samples = int(
        round(
            post_event_seconds
            * sampling_rate
        )
    )

    for event_number, event in (
        respiratory_events.iterrows()
    ):
        event_start = normalize_datetime(
            event["start_time"]
        )

        event_end = normalize_datetime(
            event["end_time"]
        )

        if (
            event_start is None
            or event_end is None
        ):
            continue

        event_start_sample = (
            timestamp_to_sample(
                event_start,
                measurement_start,
                sampling_rate,
                total_samples,
            )
        )

        event_end_sample = (
            timestamp_to_sample(
                event_end,
                measurement_start,
                sampling_rate,
                total_samples,
            )
        )

        baseline_start_sample = max(
            event_start_sample
            - baseline_samples,
            0,
        )

        baseline_end_sample = (
            event_start_sample
        )

        nadir_search_start = (
            event_start_sample
        )

        nadir_search_end = min(
            event_end_sample
            + post_samples,
            total_samples,
        )

        baseline_summary = summarize_window(
            spo2=spo2,
            valid_mask=valid_mask,
            start_sample=(
                baseline_start_sample
            ),
            end_sample=(
                baseline_end_sample
            ),
        )

        event_summary = summarize_window(
            spo2=spo2,
            valid_mask=valid_mask,
            start_sample=(
                event_start_sample
            ),
            end_sample=(
                event_end_sample
            ),
        )

        post_summary = summarize_window(
            spo2=spo2,
            valid_mask=valid_mask,
            start_sample=(
                event_end_sample
            ),
            end_sample=(
                nadir_search_end
            ),
        )

        nadir_value, nadir_sample = (
            find_nadir(
                spo2=spo2,
                valid_mask=valid_mask,
                start_sample=(
                    nadir_search_start
                ),
                end_sample=(
                    nadir_search_end
                ),
            )
        )

        baseline_value = (
            baseline_summary[
                "median"
            ]
        )

        desaturation_drop = None

        if (
            baseline_value is not None
            and nadir_value is not None
        ):
            desaturation_drop = (
                float(
                    baseline_value
                    - nadir_value
                )
            )

        nadir_delay_from_event_end = None
        nadir_delay_from_event_start = None
        nadir_time = None

        if nadir_sample is not None:
            nadir_time = (
                sample_to_timestamp(
                    nadir_sample,
                    measurement_start,
                    sampling_rate,
                )
            )

            nadir_delay_from_event_end = (
                (
                    nadir_sample
                    - event_end_sample
                )
                / sampling_rate
            )

            nadir_delay_from_event_start = (
                (
                    nadir_sample
                    - event_start_sample
                )
                / sampling_rate
            )

        baseline_valid = (
            baseline_summary[
                "valid_fraction"
            ]
            is not None
            and baseline_summary[
                "valid_fraction"
            ]
            >= 0.80
        )

        nadir_valid = (
            nadir_value is not None
        )

        analysis_valid = bool(
            baseline_valid
            and nadir_valid
        )

        coupled_3pct = bool(
            analysis_valid
            and desaturation_drop is not None
            and desaturation_drop >= 3.0
        )

        coupled_4pct = bool(
            analysis_valid
            and desaturation_drop is not None
            and desaturation_drop >= 4.0
        )

        duration_seconds = (
            event_end
            - event_start
        ).total_seconds()

        rows.append(
            {
                "event_number": (
                    event_number + 1
                ),
                "source_event_index": (
                    event.get(
                        "event_index"
                    )
                ),
                "event_type": (
                    event["event_type"]
                ),
                "subtype": (
                    event.get(
                        "subtype"
                    )
                ),
                "event_start_time": (
                    event_start
                ),
                "event_end_time": (
                    event_end
                ),
                "event_duration_seconds": (
                    duration_seconds
                ),
                "event_start_sample": (
                    event_start_sample
                ),
                "event_end_sample": (
                    event_end_sample
                ),
                "baseline_spo2_median": (
                    baseline_value
                ),
                "baseline_spo2_mean": (
                    baseline_summary[
                        "mean"
                    ]
                ),
                "baseline_valid_fraction": (
                    baseline_summary[
                        "valid_fraction"
                    ]
                ),
                "event_spo2_mean": (
                    event_summary[
                        "mean"
                    ]
                ),
                "event_spo2_min": (
                    event_summary[
                        "min"
                    ]
                ),
                "post_event_spo2_mean": (
                    post_summary[
                        "mean"
                    ]
                ),
                "post_event_spo2_min": (
                    post_summary[
                        "min"
                    ]
                ),
                "nadir_spo2": (
                    nadir_value
                ),
                "nadir_time": (
                    nadir_time
                ),
                "nadir_delay_from_event_start_seconds": (
                    nadir_delay_from_event_start
                ),
                "nadir_delay_from_event_end_seconds": (
                    nadir_delay_from_event_end
                ),
                "desaturation_drop_percent": (
                    desaturation_drop
                ),
                "analysis_valid": (
                    analysis_valid
                ),
                "coupled_desaturation_3pct": (
                    coupled_3pct
                ),
                "coupled_desaturation_4pct": (
                    coupled_4pct
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 全夜低氧歸因
# ============================================================

def build_event_influence_mask(
    events: pd.DataFrame,
    measurement_start: pd.Timestamp,
    sampling_rate: float,
    total_samples: int,
    pre_event_seconds: float,
    post_event_seconds: float,
) -> np.ndarray:
    influence_mask = np.zeros(
        total_samples,
        dtype=bool,
    )

    pre_samples = int(
        round(
            pre_event_seconds
            * sampling_rate
        )
    )

    post_samples = int(
        round(
            post_event_seconds
            * sampling_rate
        )
    )

    event_types = (
        events[
            "event_type"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    respiratory_events = events[
        event_types.isin(
            RESPIRATORY_EVENT_TYPES
        )
    ].copy()

    if (
        "usable_for_event_training"
        in respiratory_events.columns
    ):
        respiratory_events = (
            respiratory_events[
                normalize_boolean_series(
                    respiratory_events[
                        "usable_for_event_training"
                    ]
                )
            ].copy()
        )

    for _, event in (
        respiratory_events.iterrows()
    ):
        event_start = normalize_datetime(
            event.get("start_time")
        )

        event_end = normalize_datetime(
            event.get("end_time")
        )

        if (
            event_start is None
            or event_end is None
        ):
            continue

        start_sample = (
            timestamp_to_sample(
                event_start,
                measurement_start,
                sampling_rate,
                total_samples,
            )
            - pre_samples
        )

        end_sample = (
            timestamp_to_sample(
                event_end,
                measurement_start,
                sampling_rate,
                total_samples,
            )
            + post_samples
        )

        start_sample = max(
            start_sample,
            0,
        )

        end_sample = min(
            end_sample,
            total_samples,
        )

        if end_sample > start_sample:
            influence_mask[
                start_sample:end_sample
            ] = True

    return influence_mask


def calculate_low_oxygen_attribution(
    spo2: np.ndarray,
    valid_mask: np.ndarray,
    influence_mask: np.ndarray,
    sampling_rate: float,
) -> dict[str, Any]:
    low_90_mask = (
        valid_mask
        & (spo2 < 90.0)
    )

    low_88_mask = (
        valid_mask
        & (spo2 < 88.0)
    )

    low_90_near_event = (
        low_90_mask
        & influence_mask
    )

    low_90_away_from_event = (
        low_90_mask
        & ~influence_mask
    )

    low_88_near_event = (
        low_88_mask
        & influence_mask
    )

    low_88_away_from_event = (
        low_88_mask
        & ~influence_mask
    )

    low_90_count = int(
        low_90_mask.sum()
    )

    low_88_count = int(
        low_88_mask.sum()
    )

    return {
        "valid_spo2_minutes": (
            float(
                valid_mask.sum()
                / sampling_rate
                / 60.0
            )
        ),
        "event_influence_minutes": (
            float(
                (
                    influence_mask
                    & valid_mask
                ).sum()
                / sampling_rate
                / 60.0
            )
        ),
        "spo2_below_90_minutes": (
            float(
                low_90_count
                / sampling_rate
                / 60.0
            )
        ),
        "spo2_below_88_minutes": (
            float(
                low_88_count
                / sampling_rate
                / 60.0
            )
        ),
        "spo2_below_90_near_event_minutes": (
            float(
                low_90_near_event.sum()
                / sampling_rate
                / 60.0
            )
        ),
        "spo2_below_90_away_from_event_minutes": (
            float(
                low_90_away_from_event.sum()
                / sampling_rate
                / 60.0
            )
        ),
        "spo2_below_88_near_event_minutes": (
            float(
                low_88_near_event.sum()
                / sampling_rate
                / 60.0
            )
        ),
        "spo2_below_88_away_from_event_minutes": (
            float(
                low_88_away_from_event.sum()
                / sampling_rate
                / 60.0
            )
        ),
        "spo2_below_90_near_event_fraction": (
            safe_divide(
                int(
                    low_90_near_event.sum()
                ),
                low_90_count,
            )
        ),
        "spo2_below_90_away_from_event_fraction": (
            safe_divide(
                int(
                    low_90_away_from_event.sum()
                ),
                low_90_count,
            )
        ),
        "spo2_below_88_near_event_fraction": (
            safe_divide(
                int(
                    low_88_near_event.sum()
                ),
                low_88_count,
            )
        ),
        "spo2_below_88_away_from_event_fraction": (
            safe_divide(
                int(
                    low_88_away_from_event.sum()
                ),
                low_88_count,
            )
        ),
    }


# ============================================================
# Stage 對應血氧分析
# ============================================================

def analyze_stage_oxygen(
    stages: pd.DataFrame,
    spo2: np.ndarray,
    valid_mask: np.ndarray,
    measurement_start: pd.Timestamp,
    sampling_rate: float,
    total_samples: int,
) -> pd.DataFrame:
    data = stages.copy()

    data[
        "start_time"
    ] = pd.to_datetime(
        data["start_time"],
        errors="coerce",
    )

    data[
        "end_time"
    ] = pd.to_datetime(
        data["end_time"],
        errors="coerce",
    )

    if (
        "usable_for_stage_training"
        in data.columns
    ):
        data = data[
            normalize_boolean_series(
                data[
                    "usable_for_stage_training"
                ]
            )
        ].copy()

    rows: list[
        dict[str, Any]
    ] = []

    for _, row in data.iterrows():
        start_time = normalize_datetime(
            row.get("start_time")
        )

        end_time = normalize_datetime(
            row.get("end_time")
        )

        if (
            start_time is None
            or end_time is None
        ):
            continue

        start_sample = (
            timestamp_to_sample(
                start_time,
                measurement_start,
                sampling_rate,
                total_samples,
            )
        )

        end_sample = (
            timestamp_to_sample(
                end_time,
                measurement_start,
                sampling_rate,
                total_samples,
            )
        )

        summary = summarize_window(
            spo2=spo2,
            valid_mask=valid_mask,
            start_sample=start_sample,
            end_sample=end_sample,
        )

        rows.append(
            {
                "epoch_index": (
                    row.get(
                        "epoch_index"
                    )
                ),
                "start_time": (
                    start_time
                ),
                "end_time": (
                    end_time
                ),
                "stage": (
                    str(
                        row.get(
                            "stage",
                            "UNKNOWN",
                        )
                    ).upper()
                ),
                "spo2_mean": (
                    summary["mean"]
                ),
                "spo2_median": (
                    summary["median"]
                ),
                "spo2_min": (
                    summary["min"]
                ),
                "spo2_valid_fraction": (
                    summary[
                        "valid_fraction"
                    ]
                ),
            }
        )

    epoch_data = pd.DataFrame(
        rows
    )

    if epoch_data.empty:
        return pd.DataFrame()

    epoch_data = epoch_data[
        (
            epoch_data[
                "spo2_valid_fraction"
            ]
            >= 0.80
        )
    ].copy()

    summaries: list[
        dict[str, Any]
    ] = []

    stage_order = [
        "W",
        "N1",
        "N2",
        "N3",
        "REM",
        "SLEEP_ALL",
    ]

    for stage_name in stage_order:
        if stage_name == "SLEEP_ALL":
            subset = epoch_data[
                epoch_data[
                    "stage"
                ].isin(
                    [
                        "N1",
                        "N2",
                        "N3",
                        "REM",
                    ]
                )
            ].copy()
        else:
            subset = epoch_data[
                epoch_data[
                    "stage"
                ]
                == stage_name
            ].copy()

        if subset.empty:
            summaries.append(
                {
                    "stage_group": (
                        stage_name
                    ),
                    "epoch_count": 0,
                    "minutes": 0.0,
                    "spo2_mean": None,
                    "spo2_median": None,
                    "spo2_min_p01": None,
                    "spo2_min_p05": None,
                    "epoch_fraction_mean_below_90": None,
                    "epoch_fraction_mean_below_88": None,
                }
            )

            continue

        spo2_mean_values = pd.to_numeric(
            subset["spo2_mean"],
            errors="coerce",
        )

        spo2_min_values = pd.to_numeric(
            subset["spo2_min"],
            errors="coerce",
        )

        summaries.append(
            {
                "stage_group": stage_name,
                "epoch_count": int(
                    len(subset)
                ),
                "minutes": float(
                    len(subset)
                    * 0.5
                ),
                "spo2_mean": safe_float(
                    spo2_mean_values.mean()
                ),
                "spo2_median": safe_float(
                    spo2_mean_values.median()
                ),
                "spo2_min_p01": safe_float(
                    spo2_min_values.quantile(
                        0.01
                    )
                ),
                "spo2_min_p05": safe_float(
                    spo2_min_values.quantile(
                        0.05
                    )
                ),
                "epoch_fraction_mean_below_90": (
                    safe_float(
                        (
                            spo2_mean_values
                            < 90.0
                        ).mean()
                    )
                ),
                "epoch_fraction_mean_below_88": (
                    safe_float(
                        (
                            spo2_mean_values
                            < 88.0
                        ).mean()
                    )
                ),
            }
        )

    return pd.DataFrame(
        summaries
    )


# ============================================================
# 結論規則
# ============================================================

def classify_coupling(
    event_coupled_3pct_fraction: float | None,
    low_90_near_event_fraction: float | None,
) -> str:
    if (
        event_coupled_3pct_fraction is None
        or low_90_near_event_fraction is None
    ):
        return "UNKNOWN"

    if (
        event_coupled_3pct_fraction >= 0.60
        and low_90_near_event_fraction >= 0.60
    ):
        return "STRONG_EVENT_COUPLING"

    if (
        event_coupled_3pct_fraction >= 0.35
        or low_90_near_event_fraction >= 0.35
    ):
        return "PARTIAL_EVENT_COUPLING"

    return "LOW_EVENT_COUPLING"


def build_interpretation(
    coupling_level: str,
    event_coupled_fraction: float | None,
    low_90_near_event_fraction: float | None,
    wake_spo2: float | None,
    sleep_spo2: float | None,
) -> str:
    parts: list[str] = []

    if coupling_level == "STRONG_EVENT_COUPLING":
        parts.append(
            (
                "呼吸事件與血氧下降呈較強時間關聯，"
                "夜間低氧較可能有相當部分與呼吸事件相關。"
            )
        )

    elif coupling_level == "PARTIAL_EVENT_COUPLING":
        parts.append(
            (
                "呼吸事件與血氧下降呈部分時間關聯；"
                "低氧可能同時包含事件相關與非事件相關成分。"
            )
        )

    elif coupling_level == "LOW_EVENT_COUPLING":
        parts.append(
            (
                "呼吸事件與血氧下降的時間關聯偏低，"
                "夜間持續低氧可能無法僅由目前呼吸事件解釋。"
            )
        )

    else:
        parts.append(
            "目前資料不足以可靠判定呼吸事件與低氧的時間關聯。"
        )

    if event_coupled_fraction is not None:
        parts.append(
            (
                f"有效呼吸事件中約 "
                f"{event_coupled_fraction:.1%} "
                "伴隨至少 3% 血氧下降。"
            )
        )

    if low_90_near_event_fraction is not None:
        parts.append(
            (
                f"SpO₂ 低於 90% 的時間中約 "
                f"{low_90_near_event_fraction:.1%} "
                "位於呼吸事件影響時間窗內。"
            )
        )

    if (
        wake_spo2 is not None
        and sleep_spo2 is not None
    ):
        difference = (
            wake_spo2
            - sleep_spo2
        )

        parts.append(
            (
                f"PSG 清醒期平均 SpO₂ 約 "
                f"{wake_spo2:.2f}%，睡眠期約 "
                f"{sleep_spo2:.2f}%，差異約 "
                f"{difference:.2f} 個百分點。"
            )
        )

    parts.append(
        (
            "此分析屬研究型時間關聯分析，"
            "不能單獨確定低氧病因或治療方式。"
        )
    )

    return "".join(
        parts
    )


# ============================================================
# 主程式
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "分析呼吸事件與 SpO₂ 下降的時間耦合，"
            "並比較清醒與睡眠期血氧。"
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
        "--baseline-seconds",
        type=float,
        default=30.0,
        help=(
            "事件開始前基線視窗秒數，預設 30 秒。"
        ),
    )

    parser.add_argument(
        "--post-event-seconds",
        type=float,
        default=60.0,
        help=(
            "事件結束後搜尋血氧最低點的秒數，預設 60 秒。"
        ),
    )

    args = parser.parse_args()

    patient_id = str(
        args.patient_id
    ).strip()

    patient_folder = (
        INCOMING_ROOT
        / patient_id
    )

    processed_folder = (
        PROCESSED_ROOT
        / patient_id
    )

    spo2_quality_file = (
        INFERENCE_ROOT
        / patient_id
        / "spo2_quality"
        / "spo2_quality_mask.csv"
    )

    output_folder = (
        INFERENCE_ROOT
        / patient_id
        / "oxygen_event_coupling"
    )

    edf_file = find_edf_file(
        patient_folder
    )

    events_file = (
        processed_folder
        / "events_aligned.csv"
    )

    stages_file = (
        processed_folder
        / "stages_aligned.csv"
    )

    metadata_file = (
        processed_folder
        / "patient_metadata.csv"
    )

    events = load_csv(
        events_file
    )

    stages = load_csv(
        stages_file
    )

    metadata = load_csv(
        metadata_file
    )

    spo2_quality = load_csv(
        spo2_quality_file
    )

    required_quality_columns = {
        "second_index",
        "spo2",
        "quality_class",
    }

    missing_quality_columns = (
        required_quality_columns
        - set(spo2_quality.columns)
    )

    if missing_quality_columns:
        raise RuntimeError(
            "SpO₂ quality mask 缺少必要欄位："
            f"{sorted(missing_quality_columns)}"
        )

    spo2_quality[
        "second_index"
    ] = pd.to_numeric(
        spo2_quality[
            "second_index"
        ],
        errors="coerce",
    )

    spo2_quality[
        "spo2"
    ] = pd.to_numeric(
        spo2_quality[
            "spo2"
        ],
        errors="coerce",
    )

    spo2_quality = (
        spo2_quality.dropna(
            subset=[
                "second_index",
            ]
        )
        .copy()
    )

    spo2_quality[
        "second_index"
    ] = (
        spo2_quality[
            "second_index"
        ]
        .astype(int)
    )

    spo2_quality = (
        spo2_quality.sort_values(
            "second_index"
        )
        .drop_duplicates(
            subset=[
                "second_index",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    print("=" * 80)
    print("Oxygen–Respiratory Event Coupling Analysis")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    print(
        f"EDF：{edf_file}"
    )

    print()
    print("=" * 80)
    print("步驟 1：讀取 Saturation Channel")
    print("=" * 80)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "default"
        )

        raw = mne.io.read_raw_edf(
            edf_file,
            preload=False,
            verbose=False,
        )

    spo2_channel = find_spo2_channel(
        list(raw.ch_names)
    )

    print(
        f"SpO₂ Channel：{spo2_channel}"
    )

    measurement_start = (
        read_measurement_start(
            raw=raw,
            metadata=metadata,
        )
    )

    sampling_rate = float(
        raw.info["sfreq"]
    )

    total_samples = int(
        raw.n_times
    )

    print(
        f"EDF 開始時間：{measurement_start}"
    )

    print(
        f"Sampling Rate：{sampling_rate} Hz"
    )

    print(
        f"Samples：{total_samples}"
    )

    print()
    print("載入 Saturation 訊號……")

    raw.load_data()

    raw_spo2 = raw.get_data(
        picks=[spo2_channel]
    )[0]

    spo2, scale_method = (
        normalize_spo2_scale(
            raw_spo2
        )
    )

    basic_valid_mask = (
        create_valid_spo2_mask(
            spo2
        )
    )

    quality_valid_seconds = set(
        spo2_quality.loc[
            spo2_quality[
                "quality_class"
            ]
            == "VALID_PHYSIOLOGICAL",
            "second_index",
        ].astype(int)
    )

    sample_second_indices = np.floor(
        np.arange(
            total_samples,
            dtype=float,
        )
        / sampling_rate
    ).astype(int)

    quality_valid_mask = np.isin(
        sample_second_indices,
        list(
            quality_valid_seconds
        ),
    )

    valid_mask = (
        basic_valid_mask
        & quality_valid_mask
    )

    valid_fraction = safe_divide(
        int(valid_mask.sum()),
        len(valid_mask),
    )

    basic_valid_fraction = safe_divide(
        int(
            basic_valid_mask.sum()
        ),
        len(
            basic_valid_mask
        ),
    )

    quality_valid_fraction = safe_divide(
        int(
            quality_valid_mask.sum()
        ),
        len(
            quality_valid_mask
        ),
    )

    print(
        f"SpO₂ 尺度處理：{scale_method}"
    )

    print(
        f"基本範圍有效比例："
        f"{basic_valid_fraction:.2%}"
        if basic_valid_fraction is not None
        else "基本範圍有效比例：無法計算"
    )

    print(
        f"Quality mask 合格比例："
        f"{quality_valid_fraction:.2%}"
        if quality_valid_fraction is not None
        else "Quality mask 合格比例：無法計算"
    )

    print(
        f"最終有效 SpO₂ 比例："
        f"{valid_fraction:.2%}"
        if valid_fraction is not None
        else "最終有效 SpO₂ 比例：無法計算"
    )

    if valid_mask.sum() == 0:
        raise RuntimeError(
            "Saturation Channel 沒有 40–100% 的有效資料。"
        )

    valid_values = spo2[
        valid_mask
    ]

    print(
        f"SpO₂ 中位數："
        f"{np.median(valid_values):.3f}%"
    )

    print(
        f"SpO₂ P1："
        f"{np.percentile(valid_values, 1):.3f}%"
    )

    print(
        f"SpO₂ P99："
        f"{np.percentile(valid_values, 99):.3f}%"
    )

    print()
    print("=" * 80)
    print("步驟 2：逐呼吸事件分析")
    print("=" * 80)

    event_results = (
        analyze_respiratory_events(
            events=events,
            spo2=spo2,
            valid_mask=valid_mask,
            measurement_start=(
                measurement_start
            ),
            sampling_rate=(
                sampling_rate
            ),
            total_samples=(
                total_samples
            ),
            baseline_seconds=float(
                args.baseline_seconds
            ),
            post_event_seconds=float(
                args.post_event_seconds
            ),
        )
    )

    if event_results.empty:
        raise RuntimeError(
            "沒有可分析的呼吸事件。"
        )

    valid_event_results = (
        event_results[
            event_results[
                "analysis_valid"
            ]
            == True
        ].copy()
    )

    total_event_count = int(
        len(event_results)
    )

    valid_event_count = int(
        len(valid_event_results)
    )

    coupled_3pct_count = int(
        valid_event_results[
            "coupled_desaturation_3pct"
        ].sum()
    )

    coupled_4pct_count = int(
        valid_event_results[
            "coupled_desaturation_4pct"
        ].sum()
    )

    coupled_3pct_fraction = (
        safe_divide(
            coupled_3pct_count,
            valid_event_count,
        )
    )

    coupled_4pct_fraction = (
        safe_divide(
            coupled_4pct_count,
            valid_event_count,
        )
    )

    mean_drop = safe_float(
        pd.to_numeric(
            valid_event_results[
                "desaturation_drop_percent"
            ],
            errors="coerce",
        ).mean()
    )

    median_drop = safe_float(
        pd.to_numeric(
            valid_event_results[
                "desaturation_drop_percent"
            ],
            errors="coerce",
        ).median()
    )

    median_nadir_delay = safe_float(
        pd.to_numeric(
            valid_event_results[
                "nadir_delay_from_event_end_seconds"
            ],
            errors="coerce",
        ).median()
    )

    print(
        f"呼吸事件總數：{total_event_count}"
    )

    print(
        f"可分析事件："
        f"{valid_event_count}/{total_event_count}"
    )

    print(
        f"伴隨 ≥3% 血氧下降："
        f"{coupled_3pct_count}"
        f"/{valid_event_count} "
        f"({coupled_3pct_fraction:.2%})"
        if coupled_3pct_fraction is not None
        else "伴隨 ≥3% 血氧下降：無法計算"
    )

    print(
        f"伴隨 ≥4% 血氧下降："
        f"{coupled_4pct_count}"
        f"/{valid_event_count} "
        f"({coupled_4pct_fraction:.2%})"
        if coupled_4pct_fraction is not None
        else "伴隨 ≥4% 血氧下降：無法計算"
    )

    print(
        f"平均下降幅度：{mean_drop}"
    )

    print(
        f"下降幅度中位數：{median_drop}"
    )

    print(
        f"最低點相對事件結束時間中位數："
        f"{median_nadir_delay} 秒"
    )

    print()
    print("=" * 80)
    print("步驟 3：低氧時間歸因")
    print("=" * 80)

    influence_mask = (
        build_event_influence_mask(
            events=events,
            measurement_start=(
                measurement_start
            ),
            sampling_rate=(
                sampling_rate
            ),
            total_samples=(
                total_samples
            ),
            pre_event_seconds=0.0,
            post_event_seconds=float(
                args.post_event_seconds
            ),
        )
    )

    low_oxygen_attribution = (
        calculate_low_oxygen_attribution(
            spo2=spo2,
            valid_mask=valid_mask,
            influence_mask=(
                influence_mask
            ),
            sampling_rate=(
                sampling_rate
            ),
        )
    )

    print(
        f"SpO₂ <90% 總時間："
        f"{low_oxygen_attribution['spo2_below_90_minutes']:.2f} 分鐘"
    )

    print(
        f"SpO₂ <90% 且位於事件影響窗："
        f"{low_oxygen_attribution['spo2_below_90_near_event_minutes']:.2f} 分鐘"
    )

    print(
        f"SpO₂ <90% 且遠離事件："
        f"{low_oxygen_attribution['spo2_below_90_away_from_event_minutes']:.2f} 分鐘"
    )

    near_fraction = (
        low_oxygen_attribution[
            "spo2_below_90_near_event_fraction"
        ]
    )

    print(
        f"低於 90% 時間的事件鄰近比例："
        f"{near_fraction:.2%}"
        if near_fraction is not None
        else "低於 90% 時間的事件鄰近比例：無法計算"
    )

    print()
    print("=" * 80)
    print("步驟 4：清醒與睡眠期 SpO₂")
    print("=" * 80)

    stage_oxygen = (
        analyze_stage_oxygen(
            stages=stages,
            spo2=spo2,
            valid_mask=valid_mask,
            measurement_start=(
                measurement_start
            ),
            sampling_rate=(
                sampling_rate
            ),
            total_samples=(
                total_samples
            ),
        )
    )

    if not stage_oxygen.empty:
        print(
            stage_oxygen.to_string(
                index=False
            )
        )

    wake_spo2 = None
    sleep_spo2 = None

    if not stage_oxygen.empty:
        wake_rows = stage_oxygen[
            stage_oxygen[
                "stage_group"
            ]
            == "W"
        ]

        sleep_rows = stage_oxygen[
            stage_oxygen[
                "stage_group"
            ]
            == "SLEEP_ALL"
        ]

        if not wake_rows.empty:
            wake_spo2 = safe_float(
                wake_rows.iloc[0][
                    "spo2_mean"
                ]
            )

        if not sleep_rows.empty:
            sleep_spo2 = safe_float(
                sleep_rows.iloc[0][
                    "spo2_mean"
                ]
            )

    coupling_level = (
        classify_coupling(
            event_coupled_3pct_fraction=(
                coupled_3pct_fraction
            ),
            low_90_near_event_fraction=(
                near_fraction
            ),
        )
    )

    interpretation = (
        build_interpretation(
            coupling_level=(
                coupling_level
            ),
            event_coupled_fraction=(
                coupled_3pct_fraction
            ),
            low_90_near_event_fraction=(
                near_fraction
            ),
            wake_spo2=(
                wake_spo2
            ),
            sleep_spo2=(
                sleep_spo2
            ),
        )
    )

    print()
    print("=" * 80)
    print("分析結論")
    print("=" * 80)

    print(
        f"呼吸事件－低氧耦合等級："
        f"{coupling_level}"
    )

    print(
        interpretation
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_output_file = (
        output_folder
        / "oxygen_event_coupling_events.csv"
    )

    stage_output_file = (
        output_folder
        / "stage_oxygen_summary.csv"
    )

    summary_csv_file = (
        output_folder
        / "oxygen_event_coupling_summary.csv"
    )

    summary_json_file = (
        output_folder
        / "oxygen_event_coupling_summary.json"
    )

    event_results.to_csv(
        event_output_file,
        index=False,
        encoding="utf-8-sig",
    )

    stage_oxygen.to_csv(
        stage_output_file,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "patient_id": patient_id,
        "edf_file": str(edf_file),
        "spo2_channel": spo2_channel,
        "spo2_scale_method": (
            scale_method
        ),
        "measurement_start": (
            measurement_start
        ),
        "sampling_rate": (
            sampling_rate
        ),
        "valid_spo2_fraction": (
            valid_fraction
        ),
        "analysis_parameters": {
            "baseline_seconds": (
                float(
                    args.baseline_seconds
                )
            ),
            "post_event_seconds": (
                float(
                    args.post_event_seconds
                )
            ),
            "desaturation_3pct_threshold": 3.0,
            "desaturation_4pct_threshold": 4.0,
        },
        "event_coupling": {
            "total_respiratory_event_count": (
                total_event_count
            ),
            "valid_event_count": (
                valid_event_count
            ),
            "coupled_3pct_count": (
                coupled_3pct_count
            ),
            "coupled_3pct_fraction": (
                coupled_3pct_fraction
            ),
            "coupled_4pct_count": (
                coupled_4pct_count
            ),
            "coupled_4pct_fraction": (
                coupled_4pct_fraction
            ),
            "mean_desaturation_drop_percent": (
                mean_drop
            ),
            "median_desaturation_drop_percent": (
                median_drop
            ),
            "median_nadir_delay_from_event_end_seconds": (
                median_nadir_delay
            ),
        },
        "low_oxygen_attribution": (
            low_oxygen_attribution
        ),
        "wake_sleep_oxygen": {
            "wake_spo2_mean": (
                wake_spo2
            ),
            "sleep_spo2_mean": (
                sleep_spo2
            ),
            "wake_minus_sleep_spo2": (
                (
                    wake_spo2
                    - sleep_spo2
                )
                if (
                    wake_spo2 is not None
                    and sleep_spo2 is not None
                )
                else None
            ),
        },
        "coupling_level": (
            coupling_level
        ),
        "interpretation": (
            interpretation
        ),
        "research_safety_note": (
            "此分析為研究型時間關聯分析，"
            "不能單獨確定低氧病因、診斷或治療方式。"
        ),
    }

    summary_row = {
        "patient_id": patient_id,
        "spo2_channel": (
            spo2_channel
        ),
        "valid_spo2_fraction": (
            valid_fraction
        ),
        "total_respiratory_event_count": (
            total_event_count
        ),
        "valid_event_count": (
            valid_event_count
        ),
        "coupled_3pct_count": (
            coupled_3pct_count
        ),
        "coupled_3pct_fraction": (
            coupled_3pct_fraction
        ),
        "coupled_4pct_count": (
            coupled_4pct_count
        ),
        "coupled_4pct_fraction": (
            coupled_4pct_fraction
        ),
        "mean_desaturation_drop_percent": (
            mean_drop
        ),
        "median_desaturation_drop_percent": (
            median_drop
        ),
        "median_nadir_delay_from_event_end_seconds": (
            median_nadir_delay
        ),
        **low_oxygen_attribution,
        "wake_spo2_mean": (
            wake_spo2
        ),
        "sleep_spo2_mean": (
            sleep_spo2
        ),
        "wake_minus_sleep_spo2": (
            (
                wake_spo2
                - sleep_spo2
            )
            if (
                wake_spo2 is not None
                and sleep_spo2 is not None
            )
            else None
        ),
        "coupling_level": (
            coupling_level
        ),
        "interpretation": (
            interpretation
        ),
    }

    pd.DataFrame(
        [summary_row]
    ).to_csv(
        summary_csv_file,
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        summary_json_file,
        summary,
    )

    print()
    print("=" * 80)
    print("輸出檔案")
    print("=" * 80)

    print(
        f"逐事件分析：{event_output_file}"
    )

    print(
        f"Stage 血氧摘要：{stage_output_file}"
    )

    print(
        f"摘要 CSV：{summary_csv_file}"
    )

    print(
        f"摘要 JSON：{summary_json_file}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()