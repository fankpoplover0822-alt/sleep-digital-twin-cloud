from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


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


POSITION_CHANNEL_ALIASES = [
    "posangle",
    "position",
    "body position",
    "sleep position",
]

RESPIRATORY_EVENT_TYPES = {
    "HYPOPNEA",
    "OBSTRUCTIVE_APNEA",
    "CENTRAL_APNEA",
    "MIXED_APNEA",
}

MIN_EPOCH_COVERAGE = 0.80
MAX_EPOCH_ANGLE_IQR = 30.0
MIN_CLUSTER_EXPOSURE_HOURS = 0.50

RANDOM_STATE = 42


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
            np.datetime64,
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


def normalize_datetime(
    value: Any,
) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(timestamp):
        return None

    result = pd.Timestamp(
        timestamp
    )

    if result.tzinfo is not None:
        result = result.tz_localize(
            None
        )

    return result


# ============================================================
# 檔案與 Channel
# ============================================================

def find_edf_file(
    patient_id: str,
) -> Path:
    patient_folder = (
        INCOMING_ROOT
        / patient_id
    )

    if not patient_folder.exists():
        raise FileNotFoundError(
            f"找不到患者資料夾：{patient_folder}"
        )

    preferred_file = (
        patient_folder
        / f"{patient_id}_EDF.edf"
    )

    if preferred_file.exists():
        return preferred_file

    candidates = sorted(
        patient_folder.glob("*.edf")
    )

    if not candidates:
        candidates = sorted(
            patient_folder.glob("*.EDF")
        )

    if not candidates:
        raise FileNotFoundError(
            f"找不到 EDF：{patient_folder}"
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_size,
    )


def find_position_channel(
    channel_names: list[str],
) -> str:
    normalized_names = {
        channel: (
            str(channel)
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )
        for channel in channel_names
    }

    for channel, normalized in (
        normalized_names.items()
    ):
        for alias in POSITION_CHANNEL_ALIASES:
            if alias in normalized:
                return channel

    raise RuntimeError(
        "EDF 中找不到 PosAngle／Position Channel。"
    )


def load_required_csv(
    file_path: Path,
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到 CSV：{file_path}"
        )

    return pd.read_csv(
        file_path
    )


def get_measurement_start(
    raw: mne.io.BaseRaw,
    metadata: pd.DataFrame,
) -> pd.Timestamp:
    if not metadata.empty:
        for column in [
            "corrected_edf_start",
            "edf_start",
            "original_edf_start",
        ]:
            if column not in metadata.columns:
                continue

            timestamp = normalize_datetime(
                metadata.iloc[0][column]
            )

            if timestamp is not None:
                return timestamp

    timestamp = normalize_datetime(
        raw.info.get("meas_date")
    )

    if timestamp is None:
        raise RuntimeError(
            "無法取得 EDF 開始時間。"
        )

    return timestamp


# ============================================================
# PosAngle 訊號處理
# ============================================================

def resample_angle_to_seconds(
    angle: np.ndarray,
    sampling_rate: float,
) -> pd.DataFrame:
    total_seconds = int(
        np.floor(
            len(angle)
            / sampling_rate
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for second_index in range(
        total_seconds
    ):
        start_sample = int(
            round(
                second_index
                * sampling_rate
            )
        )

        end_sample = int(
            round(
                (
                    second_index + 1
                )
                * sampling_rate
            )
        )

        segment = angle[
            start_sample:end_sample
        ]

        finite_values = segment[
            np.isfinite(segment)
        ]

        if finite_values.size == 0:
            rows.append(
                {
                    "second_index": second_index,
                    "angle_median": None,
                    "angle_iqr": None,
                    "valid": False,
                }
            )
            continue

        rows.append(
            {
                "second_index": second_index,
                "angle_median": float(
                    np.median(
                        finite_values
                    )
                ),
                "angle_iqr": float(
                    np.percentile(
                        finite_values,
                        75,
                    )
                    - np.percentile(
                        finite_values,
                        25,
                    )
                ),
                "valid": True,
            }
        )

    result = pd.DataFrame(
        rows
    )

    numeric_angle = pd.to_numeric(
        result["angle_median"],
        errors="coerce",
    )

    finite_angle = numeric_angle.dropna()

    if finite_angle.empty:
        raise RuntimeError(
            "PosAngle 沒有可用數值。"
        )

    lower_limit = float(
        finite_angle.quantile(
            0.005
        )
    )

    upper_limit = float(
        finite_angle.quantile(
            0.995
        )
    )

    result[
        "angle_outlier"
    ] = (
        numeric_angle
        < lower_limit
    ) | (
        numeric_angle
        > upper_limit
    )

    result[
        "valid"
    ] = (
        result["valid"]
        & ~result[
            "angle_outlier"
        ]
    )

    result.loc[
        ~result["valid"],
        "angle_median",
    ] = np.nan

    return result


def timestamp_to_second(
    timestamp: pd.Timestamp,
    measurement_start: pd.Timestamp,
    total_seconds: int,
) -> int:
    seconds = int(
        round(
            (
                timestamp
                - measurement_start
            ).total_seconds()
        )
    )

    return int(
        np.clip(
            seconds,
            0,
            total_seconds,
        )
    )


def build_epoch_position_features(
    stages: pd.DataFrame,
    angle_seconds: pd.DataFrame,
    measurement_start: pd.Timestamp,
) -> pd.DataFrame:
    stage_data = stages.copy()

    stage_data[
        "start_time"
    ] = pd.to_datetime(
        stage_data["start_time"],
        errors="coerce",
    )

    stage_data[
        "end_time"
    ] = pd.to_datetime(
        stage_data["end_time"],
        errors="coerce",
    )

    if (
        "usable_for_stage_training"
        in stage_data.columns
    ):
        stage_data = stage_data[
            normalize_boolean_series(
                stage_data[
                    "usable_for_stage_training"
                ]
            )
        ].copy()

    total_seconds = int(
        len(angle_seconds)
    )

    rows: list[
        dict[str, Any]
    ] = []

    for _, stage_row in (
        stage_data.iterrows()
    ):
        start_time = normalize_datetime(
            stage_row.get(
                "start_time"
            )
        )

        end_time = normalize_datetime(
            stage_row.get(
                "end_time"
            )
        )

        if (
            start_time is None
            or end_time is None
        ):
            continue

        start_second = (
            timestamp_to_second(
                start_time,
                measurement_start,
                total_seconds,
            )
        )

        end_second = (
            timestamp_to_second(
                end_time,
                measurement_start,
                total_seconds,
            )
        )

        if end_second <= start_second:
            continue

        window = angle_seconds.iloc[
            start_second:end_second
        ]

        valid_window = window[
            window["valid"]
            & window[
                "angle_median"
            ].notna()
        ]

        epoch_seconds = int(
            end_second
            - start_second
        )

        valid_seconds = int(
            len(valid_window)
        )

        coverage = safe_divide(
            valid_seconds,
            epoch_seconds,
        )

        angle_median = None
        angle_mean = None
        angle_iqr = None
        angle_std = None

        if valid_seconds > 0:
            angle_values = pd.to_numeric(
                valid_window[
                    "angle_median"
                ],
                errors="coerce",
            ).dropna()

            if not angle_values.empty:
                angle_median = safe_float(
                    angle_values.median()
                )

                angle_mean = safe_float(
                    angle_values.mean()
                )

                angle_iqr = safe_float(
                    angle_values.quantile(
                        0.75
                    )
                    - angle_values.quantile(
                        0.25
                    )
                )

                angle_std = safe_float(
                    angle_values.std(
                        ddof=0
                    )
                )

        position_quality_valid = bool(
            coverage is not None
            and coverage
            >= MIN_EPOCH_COVERAGE
            and angle_median is not None
            and (
                angle_iqr is None
                or angle_iqr
                <= MAX_EPOCH_ANGLE_IQR
            )
        )

        rows.append(
            {
                "epoch_index": (
                    stage_row.get(
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
                        stage_row.get(
                            "stage",
                            "UNKNOWN",
                        )
                    )
                ),
                "position_start_second": (
                    start_second
                ),
                "position_end_second": (
                    end_second
                ),
                "position_valid_seconds": (
                    valid_seconds
                ),
                "position_coverage_fraction": (
                    coverage
                ),
                "position_angle_median": (
                    angle_median
                ),
                "position_angle_mean": (
                    angle_mean
                ),
                "position_angle_iqr": (
                    angle_iqr
                ),
                "position_angle_std": (
                    angle_std
                ),
                "position_quality_valid": (
                    position_quality_valid
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 姿勢分群
# ============================================================

def choose_cluster_count(
    values: np.ndarray,
    requested_cluster_count: int | None,
) -> tuple[
    int,
    dict[int, float | None],
]:
    if requested_cluster_count is not None:
        return (
            int(requested_cluster_count),
            {},
        )

    sample_count = len(values)

    if sample_count < 20:
        return 2, {}

    candidate_scores: dict[
        int,
        float | None,
    ] = {}

    maximum_clusters = min(
        4,
        max(
            2,
            sample_count // 20,
        ),
    )

    scaler = StandardScaler()

    scaled_values = (
        scaler.fit_transform(
            values.reshape(
                -1,
                1,
            )
        )
    )

    best_cluster_count = 2
    best_score = -np.inf

    for cluster_count in range(
        2,
        maximum_clusters + 1,
    ):
        model = KMeans(
            n_clusters=cluster_count,
            random_state=RANDOM_STATE,
            n_init=20,
        )

        labels = model.fit_predict(
            scaled_values
        )

        if len(
            np.unique(labels)
        ) < 2:
            candidate_scores[
                cluster_count
            ] = None
            continue

        score = float(
            silhouette_score(
                scaled_values,
                labels,
            )
        )

        candidate_scores[
            cluster_count
        ] = score

        if score > best_score:
            best_score = score
            best_cluster_count = (
                cluster_count
            )

    return (
        best_cluster_count,
        candidate_scores,
    )


def assign_position_clusters(
    epoch_data: pd.DataFrame,
    requested_cluster_count: int | None,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    result = epoch_data.copy()

    result[
        "position_cluster"
    ] = "UNKNOWN"

    valid_mask = (
        result[
            "position_quality_valid"
        ]
        & result[
            "position_angle_median"
        ].notna()
    )

    valid_values = pd.to_numeric(
        result.loc[
            valid_mask,
            "position_angle_median",
        ],
        errors="coerce",
    ).dropna()

    if len(valid_values) < 10:
        return (
            result,
            {
                "cluster_count": 0,
                "cluster_centers": [],
                "silhouette_scores": {},
                "warning": (
                    "可用品質合格 Epoch 太少，"
                    "無法進行姿勢分群。"
                ),
            },
        )

    cluster_count, silhouette_scores = (
        choose_cluster_count(
            valid_values.to_numpy(
                dtype=float
            ),
            requested_cluster_count,
        )
    )

    scaler = StandardScaler()

    scaled_values = (
        scaler.fit_transform(
            valid_values.to_numpy(
                dtype=float
            ).reshape(
                -1,
                1,
            )
        )
    )

    model = KMeans(
        n_clusters=cluster_count,
        random_state=RANDOM_STATE,
        n_init=30,
    )

    raw_labels = model.fit_predict(
        scaled_values
    )

    original_centers = (
        scaler.inverse_transform(
            model.cluster_centers_
        )
        .reshape(-1)
    )

    center_order = np.argsort(
        original_centers
    )

    label_mapping = {
        int(original_label): (
            f"POSITION_CLUSTER_"
            f"{ordered_index + 1}"
        )
        for ordered_index, original_label
        in enumerate(center_order)
    }

    mapped_labels = [
        label_mapping[
            int(label)
        ]
        for label in raw_labels
    ]

    result.loc[
        valid_values.index,
        "position_cluster",
    ] = mapped_labels

    sorted_centers = [
        float(
            original_centers[
                original_label
            ]
        )
        for original_label
        in center_order
    ]

    return (
        result,
        {
            "cluster_count": (
                cluster_count
            ),
            "cluster_centers": (
                sorted_centers
            ),
            "silhouette_scores": (
                silhouette_scores
            ),
            "warning": (
                "姿勢群組為 PosAngle 資料分群結果；"
                "尚未經裝置校正，因此不能直接命名為"
                "仰睡、左側睡、右側睡或趴睡。"
            ),
        },
    )


# ============================================================
# 呼吸事件與姿勢
# ============================================================

def mark_respiratory_events(
    epoch_data: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    result = epoch_data.copy()

    result[
        "respiratory_event_count"
    ] = 0

    result[
        "has_respiratory_event"
    ] = False

    event_data = events.copy()

    event_data[
        "event_type"
    ] = (
        event_data[
            "event_type"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    event_data = event_data[
        event_data[
            "event_type"
        ].isin(
            RESPIRATORY_EVENT_TYPES
        )
    ].copy()

    if (
        "usable_for_event_training"
        in event_data.columns
    ):
        event_data = event_data[
            normalize_boolean_series(
                event_data[
                    "usable_for_event_training"
                ]
            )
        ].copy()

    event_data[
        "start_time"
    ] = pd.to_datetime(
        event_data["start_time"],
        errors="coerce",
    )

    event_data[
        "end_time"
    ] = pd.to_datetime(
        event_data["end_time"],
        errors="coerce",
    )

    event_data = event_data.dropna(
        subset=[
            "start_time",
            "end_time",
        ]
    )

    result[
        "start_time"
    ] = pd.to_datetime(
        result["start_time"],
        errors="coerce",
    )

    result[
        "end_time"
    ] = pd.to_datetime(
        result["end_time"],
        errors="coerce",
    )

    for _, event in event_data.iterrows():
        event_start = event[
            "start_time"
        ]

        event_end = event[
            "end_time"
        ]

        overlap_seconds = (
            np.minimum(
                result[
                    "end_time"
                ].values.astype(
                    "datetime64[ns]"
                ),
                np.datetime64(
                    event_end
                ),
            )
            - np.maximum(
                result[
                    "start_time"
                ].values.astype(
                    "datetime64[ns]"
                ),
                np.datetime64(
                    event_start
                ),
            )
        ) / np.timedelta64(
            1,
            "s",
        )

        overlap_seconds = np.asarray(
            overlap_seconds,
            dtype=float,
        )

        overlap_seconds[
            overlap_seconds < 0
        ] = 0.0

        if not np.any(
            overlap_seconds > 0
        ):
            continue

        assigned_position = int(
            np.argmax(
                overlap_seconds
            )
        )

        result.iloc[
            assigned_position,
            result.columns.get_loc(
                "respiratory_event_count"
            ),
        ] += 1

        result.iloc[
            assigned_position,
            result.columns.get_loc(
                "has_respiratory_event"
            ),
        ] = True

    return result


def classify_position_relevance(
    cluster_summary: pd.DataFrame,
) -> tuple[
    str,
    float | None,
]:
    eligible = cluster_summary[
        cluster_summary[
            "sleep_hours"
        ]
        >= MIN_CLUSTER_EXPOSURE_HOURS
    ].copy()

    eligible = eligible[
        eligible[
            "event_index_per_hour"
        ].notna()
    ]

    if len(eligible) < 2:
        return "UNKNOWN", None

    maximum_index = safe_float(
        eligible[
            "event_index_per_hour"
        ].max()
    )

    minimum_index = safe_float(
        eligible[
            "event_index_per_hour"
        ].min()
    )

    if (
        maximum_index is None
        or minimum_index is None
    ):
        return "UNKNOWN", None

    ratio = safe_divide(
        maximum_index,
        max(
            minimum_index,
            0.1,
        ),
    )

    if ratio is None:
        return "UNKNOWN", None

    if ratio >= 2.0:
        return "HIGH", ratio

    if ratio >= 1.5:
        return "MODERATE", ratio

    if ratio >= 1.2:
        return "MILD", ratio

    return "LOW", ratio


def build_cluster_summary(
    epoch_data: pd.DataFrame,
) -> pd.DataFrame:
    sleep_data = epoch_data[
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

    rows: list[
        dict[str, Any]
    ] = []

    for cluster_name in sorted(
        sleep_data[
            "position_cluster"
        ].dropna().unique()
    ):
        subset = sleep_data[
            sleep_data[
                "position_cluster"
            ]
            == cluster_name
        ]

        epoch_count = int(
            len(subset)
        )

        sleep_hours = float(
            epoch_count
            * 30.0
            / 3600.0
        )

        event_epoch_count = int(
            subset[
                "has_respiratory_event"
            ].sum()
        )

        respiratory_event_count = int(
            subset[
                "respiratory_event_count"
            ].sum()
        )

        event_index = safe_divide(
            respiratory_event_count,
            sleep_hours,
        )

        rows.append(
            {
                "position_cluster": (
                    cluster_name
                ),
                "epoch_count": (
                    epoch_count
                ),
                "sleep_hours": (
                    sleep_hours
                ),
                "event_epoch_count": (
                    event_epoch_count
                ),
                "respiratory_event_count": (
                    respiratory_event_count
                ),
                "event_index_per_hour": (
                    event_index
                ),
                "angle_median": safe_float(
                    pd.to_numeric(
                        subset[
                            "position_angle_median"
                        ],
                        errors="coerce",
                    ).median()
                ),
                "angle_p05": safe_float(
                    pd.to_numeric(
                        subset[
                            "position_angle_median"
                        ],
                        errors="coerce",
                    ).quantile(
                        0.05
                    )
                ),
                "angle_p95": safe_float(
                    pd.to_numeric(
                        subset[
                            "position_angle_median"
                        ],
                        errors="coerce",
                    ).quantile(
                        0.95
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def count_position_changes(
    epoch_data: pd.DataFrame,
) -> int:
    sleep_data = epoch_data[
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

    clusters = (
        sleep_data[
            "position_cluster"
        ]
        .astype(str)
    )

    known_mask = (
        clusters
        != "UNKNOWN"
    )

    known_clusters = clusters[
        known_mask
    ]

    if len(known_clusters) <= 1:
        return 0

    return int(
        (
            known_clusters
            != known_clusters.shift()
        ).sum()
        - 1
    )


# ============================================================
# 主程式
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "建立患者 PosAngle 姿勢群組 Profile。"
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
        "--cluster-count",
        type=int,
        default=None,
        choices=[
            2,
            3,
            4,
        ],
        help=(
            "指定姿勢群組數；未指定時自動選擇。"
        ),
    )

    args = parser.parse_args()

    patient_id = str(
        args.patient_id
    ).strip()

    processed_folder = (
        PROCESSED_ROOT
        / patient_id
    )

    output_folder = (
        INFERENCE_ROOT
        / patient_id
        / "position_profile"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    edf_file = find_edf_file(
        patient_id
    )

    stages_file = (
        processed_folder
        / "stages_aligned.csv"
    )

    events_file = (
        processed_folder
        / "events_aligned.csv"
    )

    metadata_file = (
        processed_folder
        / "patient_metadata.csv"
    )

    stages = load_required_csv(
        stages_file
    )

    events = load_required_csv(
        events_file
    )

    metadata = load_required_csv(
        metadata_file
    )

    print("=" * 80)
    print("Patient Position Profile")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    print(
        f"EDF：{edf_file}"
    )

    print()
    print("=" * 80)
    print("步驟 1：讀取 PosAngle")
    print("=" * 80)

    raw = mne.io.read_raw_edf(
        edf_file,
        preload=False,
        verbose="ERROR",
    )

    position_channel = (
        find_position_channel(
            list(raw.ch_names)
        )
    )

    measurement_start = (
        get_measurement_start(
            raw,
            metadata,
        )
    )

    sampling_rate = float(
        raw.info["sfreq"]
    )

    print(
        f"Position Channel："
        f"{position_channel}"
    )

    print(
        f"EDF 開始時間："
        f"{measurement_start}"
    )

    print(
        f"Sampling Rate："
        f"{sampling_rate} Hz"
    )

    position_signal = raw.get_data(
        picks=[
            position_channel
        ]
    )[0]

    print(
        f"Position Samples："
        f"{len(position_signal)}"
    )

    print()
    print("=" * 80)
    print("步驟 2：每秒角度與 Epoch 特徵")
    print("=" * 80)

    angle_seconds = (
        resample_angle_to_seconds(
            angle=position_signal,
            sampling_rate=(
                sampling_rate
            ),
        )
    )

    epoch_data = (
        build_epoch_position_features(
            stages=stages,
            angle_seconds=(
                angle_seconds
            ),
            measurement_start=(
                measurement_start
            ),
        )
    )

    valid_epoch_count = int(
        epoch_data[
            "position_quality_valid"
        ].sum()
    )

    total_epoch_count = int(
        len(epoch_data)
    )

    valid_epoch_fraction = safe_divide(
        valid_epoch_count,
        total_epoch_count,
    )

    print(
        f"Epoch 數量："
        f"{total_epoch_count}"
    )

    print(
        f"姿勢品質合格 Epoch："
        f"{valid_epoch_count}/"
        f"{total_epoch_count}"
    )

    print(
        f"姿勢品質合格比例："
        f"{valid_epoch_fraction:.2%}"
        if valid_epoch_fraction
        is not None
        else "姿勢品質合格比例：無法計算"
    )

    print()
    print("=" * 80)
    print("步驟 3：姿勢群組分群")
    print("=" * 80)

    epoch_data, clustering_info = (
        assign_position_clusters(
            epoch_data=epoch_data,
            requested_cluster_count=(
                args.cluster_count
            ),
        )
    )

    epoch_data = mark_respiratory_events(
        epoch_data=epoch_data,
        events=events,
    )

    cluster_summary = (
        build_cluster_summary(
            epoch_data
        )
    )

    position_relevance, event_ratio = (
        classify_position_relevance(
            cluster_summary
        )
    )

    position_change_count = (
        count_position_changes(
            epoch_data
        )
    )

    known_position_epochs = int(
        (
            epoch_data[
                "position_cluster"
            ]
            != "UNKNOWN"
        ).sum()
    )

    known_position_fraction = (
        safe_divide(
            known_position_epochs,
            len(epoch_data),
        )
    )

    print(
        f"自動選擇群組數："
        f"{clustering_info['cluster_count']}"
    )

    print(
        "群組中心角度："
        f"{clustering_info['cluster_centers']}"
    )

    print(
        f"已知群組 Epoch 比例："
        f"{known_position_fraction:.2%}"
        if known_position_fraction
        is not None
        else "已知群組 Epoch 比例：無法計算"
    )

    print(
        f"睡眠期間姿勢群組變換次數："
        f"{position_change_count}"
    )

    print()
    print("姿勢群組摘要：")

    if cluster_summary.empty:
        print(
            "沒有可用姿勢群組摘要。"
        )
    else:
        print(
            cluster_summary.to_string(
                index=False
            )
        )

    print()
    print("=" * 80)
    print("姿勢相關性結果")
    print("=" * 80)

    print(
        f"Position Relevance："
        f"{position_relevance}"
    )

    print(
        f"最高／最低事件率比值："
        f"{event_ratio}"
    )

    print()
    print(
        "注意：POSITION_CLUSTER_1、2、3、4 "
        "只是角度分群，不代表已確認的仰睡、"
        "左側睡、右側睡或趴睡。"
    )

    epoch_output = (
        output_folder
        / "epoch_position_profile.csv"
    )

    cluster_output = (
        output_folder
        / "position_cluster_summary.csv"
    )

    summary_csv = (
        output_folder
        / "patient_position_profile.csv"
    )

    summary_json = (
        output_folder
        / "patient_position_profile.json"
    )

    epoch_data.to_csv(
        epoch_output,
        index=False,
        encoding="utf-8-sig",
    )

    cluster_summary.to_csv(
        cluster_output,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "patient_id": patient_id,
        "edf_file": str(edf_file),
        "position_channel": (
            position_channel
        ),
        "measurement_start": (
            measurement_start
        ),
        "sampling_rate": (
            sampling_rate
        ),
        "total_epoch_count": (
            total_epoch_count
        ),
        "valid_position_epoch_count": (
            valid_epoch_count
        ),
        "valid_position_epoch_fraction": (
            valid_epoch_fraction
        ),
        "known_position_epoch_count": (
            known_position_epochs
        ),
        "known_position_epoch_fraction": (
            known_position_fraction
        ),
        "position_change_count": (
            position_change_count
        ),
        "cluster_count": (
            clustering_info[
                "cluster_count"
            ]
        ),
        "cluster_centers": (
            clustering_info[
                "cluster_centers"
            ]
        ),
        "silhouette_scores": (
            clustering_info[
                "silhouette_scores"
            ]
        ),
        "position_relevance": (
            position_relevance
        ),
        "position_event_index_ratio": (
            event_ratio
        ),
        "anatomical_position_mapping": (
            "UNCALIBRATED"
        ),
        "interpretation": (
            "Position relevance 使用未命名姿勢群組間的"
            "呼吸事件率差異計算，可用於評估整夜壓力需求"
            "是否可能隨姿勢改變；不可直接視為仰睡相關 OSA。"
        ),
        "warning": (
            clustering_info[
                "warning"
            ]
        ),
    }

    save_json(
        summary_json,
        summary,
    )

    summary_row = {
        "patient_id": patient_id,
        "position_channel": (
            position_channel
        ),
        "total_epoch_count": (
            total_epoch_count
        ),
        "valid_position_epoch_count": (
            valid_epoch_count
        ),
        "valid_position_epoch_fraction": (
            valid_epoch_fraction
        ),
        "known_position_epoch_fraction": (
            known_position_fraction
        ),
        "position_change_count": (
            position_change_count
        ),
        "cluster_count": (
            clustering_info[
                "cluster_count"
            ]
        ),
        "cluster_centers": json.dumps(
            clustering_info[
                "cluster_centers"
            ],
            ensure_ascii=False,
        ),
        "position_relevance": (
            position_relevance
        ),
        "position_event_index_ratio": (
            event_ratio
        ),
        "anatomical_position_mapping": (
            "UNCALIBRATED"
        ),
    }

    pd.DataFrame(
        [summary_row]
    ).to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("Position Profile 完成")
    print("=" * 80)

    print(
        f"Epoch Position："
        f"{epoch_output}"
    )

    print(
        f"Cluster Summary："
        f"{cluster_output}"
    )

    print(
        f"Profile CSV："
        f"{summary_csv}"
    )

    print(
        f"Profile JSON："
        f"{summary_json}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()