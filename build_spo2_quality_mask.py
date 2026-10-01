from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import sys

if sys.stdout.encoding is not None:
    try:
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
        )
    except AttributeError:
        pass

if sys.stderr.encoding is not None:
    try:
        sys.stderr.reconfigure(
            encoding="utf-8",
            errors="replace",
        )
    except AttributeError:
        pass


PROJECT_ROOT = Path(__file__).resolve().parent

INCOMING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "incoming"
)

INFERENCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "inference"
)


OUTPUT_SAMPLE_RATE = 1.0

ABSOLUTE_MIN_SPO2 = 40.0
ABSOLUTE_MAX_SPO2 = 100.0

SUSPECT_LOW_SPO2 = 60.0

MAX_STEP_CHANGE_PER_SECOND = 8.0

V_DROP_MINIMUM = 12.0
V_DROP_MAX_DURATION_SECONDS = 10
V_RECOVERY_TOLERANCE = 5.0

FLATLINE_MINIMUM_SECONDS = 120
FLATLINE_TOLERANCE = 0.01


QUALITY_VALID = "VALID_PHYSIOLOGICAL"
QUALITY_SUSPECT = "SUSPECT_ARTIFACT"
QUALITY_INVALID = "INVALID"


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

    return number


def safe_json_value(
    value: Any,
) -> Any:
    if value is None:
        return None

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


def sanitize_json_object(
    value: Any,
) -> Any:
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): sanitize_json_object(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            sanitize_json_object(item)
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return [
            sanitize_json_object(item)
            for item in value
        ]

    return safe_json_value(value)


def save_json(
    file_path: Path,
    data: dict[str, Any],
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_data = sanitize_json_object(
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


def find_edf_file(
    patient_id: str,
) -> Path:
    patient_folder = (
        INCOMING_ROOT
        / patient_id
    )

    if not patient_folder.exists():
        raise FileNotFoundError(
            "找不到患者資料夾："
            f"{patient_folder}"
        )

    preferred_file = (
        patient_folder
        / f"{patient_id}_EDF.edf"
    )

    if preferred_file.exists():
        return preferred_file

    edf_files = sorted(
        patient_folder.glob("*.edf")
    )

    if not edf_files:
        edf_files = sorted(
            patient_folder.glob("*.EDF")
        )

    if not edf_files:
        raise FileNotFoundError(
            "患者資料夾中找不到 EDF："
            f"{patient_folder}"
        )

    return edf_files[0]


def normalize_channel_name(
    value: str,
) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def find_channel(
    channel_names: list[str],
    aliases: list[str],
) -> str | None:
    normalized_channels = {
        channel: normalize_channel_name(
            channel
        )
        for channel in channel_names
    }

    normalized_aliases = [
        normalize_channel_name(alias)
        for alias in aliases
    ]

    for channel, normalized in (
        normalized_channels.items()
    ):
        if normalized in normalized_aliases:
            return channel

    for channel, normalized in (
        normalized_channels.items()
    ):
        for alias in normalized_aliases:
            if alias in normalized:
                return channel

    return None


def read_raw_edf(
    edf_file: Path,
) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(
        edf_file,
        preload=False,
        verbose="ERROR",
    )

    return raw


def get_channel_data(
    raw: mne.io.BaseRaw,
    channel_name: str,
) -> tuple[np.ndarray, float]:
    sampling_rate = float(
        raw.info["sfreq"]
    )

    data = raw.get_data(
        picks=[channel_name],
    )[0]

    data = np.asarray(
        data,
        dtype=float,
    )

    return (
        data,
        sampling_rate,
    )


def detect_spo2_scale(
    values: np.ndarray,
) -> tuple[np.ndarray, str]:
    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.size == 0:
        return (
            values.astype(float),
            "NO_FINITE_DATA",
        )

    median_value = float(
        np.nanmedian(finite_values)
    )

    p99_value = float(
        np.nanpercentile(
            finite_values,
            99,
        )
    )

    if (
        median_value <= 1.5
        and p99_value <= 2.0
    ):
        return (
            values * 100.0,
            "FRACTION_TO_PERCENT",
        )

    return (
        values.astype(float),
        "ALREADY_PERCENT",
    )


def resample_to_one_second(
    values: np.ndarray,
    sampling_rate: float,
) -> np.ndarray:
    if sampling_rate <= 0:
        raise ValueError(
            "sampling_rate 必須大於 0。"
        )

    total_seconds = int(
        np.floor(
            len(values)
            / sampling_rate
        )
    )

    if total_seconds <= 0:
        return np.asarray(
            [],
            dtype=float,
        )

    output = np.full(
        total_seconds,
        np.nan,
        dtype=float,
    )

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
                    second_index
                    + 1
                )
                * sampling_rate
            )
        )

        start_sample = max(
            start_sample,
            0,
        )

        end_sample = min(
            end_sample,
            len(values),
        )

        if end_sample <= start_sample:
            continue

        segment = values[
            start_sample:end_sample
        ]

        finite_segment = segment[
            np.isfinite(segment)
        ]

        if finite_segment.size == 0:
            continue

        output[
            second_index
        ] = float(
            np.nanmedian(
                finite_segment
            )
        )

    return output


def resample_auxiliary_channel(
    raw: mne.io.BaseRaw,
    channel_name: str | None,
    target_length: int,
) -> np.ndarray:
    output = np.full(
        target_length,
        np.nan,
        dtype=float,
    )

    if channel_name is None:
        return output

    values, sampling_rate = (
        get_channel_data(
            raw,
            channel_name,
        )
    )

    second_values = (
        resample_to_one_second(
            values,
            sampling_rate,
        )
    )

    copy_length = min(
        target_length,
        len(second_values),
    )

    if copy_length > 0:
        output[
            :copy_length
        ] = second_values[
            :copy_length
        ]

    return output


def mark_range(
    mask: np.ndarray,
    start_index: int,
    end_index: int,
) -> None:
    start_index = max(
        int(start_index),
        0,
    )

    end_index = min(
        int(end_index),
        len(mask),
    )

    if end_index <= start_index:
        return

    mask[
        start_index:end_index
    ] = True


def detect_flatline_mask(
    spo2: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(
        len(spo2),
        dtype=bool,
    )

    if len(spo2) == 0:
        return mask

    run_start = 0

    for index in range(
        1,
        len(spo2) + 1,
    ):
        should_break = False

        if index >= len(spo2):
            should_break = True
        else:
            previous_value = spo2[
                index - 1
            ]

            current_value = spo2[
                index
            ]

            if (
                not np.isfinite(
                    previous_value
                )
                or not np.isfinite(
                    current_value
                )
            ):
                should_break = True
            elif (
                abs(
                    current_value
                    - previous_value
                )
                > FLATLINE_TOLERANCE
            ):
                should_break = True

        if not should_break:
            continue

        run_length = (
            index
            - run_start
        )

        if (
            run_length
            >= FLATLINE_MINIMUM_SECONDS
        ):
            mark_range(
                mask,
                run_start,
                index,
            )

        run_start = index

    return mask


def detect_rapid_change_mask(
    spo2: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(
        len(spo2),
        dtype=bool,
    )

    if len(spo2) < 2:
        return mask

    differences = np.diff(
        spo2
    )

    rapid_indices = np.where(
        np.isfinite(differences)
        & (
            np.abs(differences)
            > MAX_STEP_CHANGE_PER_SECOND
        )
    )[0]

    for index in rapid_indices:
        mark_range(
            mask,
            index,
            index + 2,
        )

    return mask


def detect_short_v_drop_mask(
    spo2: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(
        len(spo2),
        dtype=bool,
    )

    if len(spo2) < 3:
        return mask

    window_seconds = (
        V_DROP_MAX_DURATION_SECONDS
    )

    for center_index in range(
        1,
        len(spo2) - 1,
    ):
        center_value = spo2[
            center_index
        ]

        if not np.isfinite(
            center_value
        ):
            continue

        left_start = max(
            0,
            center_index
            - window_seconds,
        )

        right_end = min(
            len(spo2),
            center_index
            + window_seconds
            + 1,
        )

        left_values = spo2[
            left_start:center_index
        ]

        right_values = spo2[
            center_index + 1:right_end
        ]

        left_values = left_values[
            np.isfinite(left_values)
        ]

        right_values = right_values[
            np.isfinite(right_values)
        ]

        if (
            left_values.size == 0
            or right_values.size == 0
        ):
            continue

        left_reference = float(
            np.nanmax(left_values)
        )

        right_reference = float(
            np.nanmax(right_values)
        )

        drop_from_left = (
            left_reference
            - center_value
        )

        recovery_to_right = (
            right_reference
            - center_value
        )

        baseline_difference = abs(
            left_reference
            - right_reference
        )

        if (
            drop_from_left
            >= V_DROP_MINIMUM
            and recovery_to_right
            >= V_DROP_MINIMUM
            and baseline_difference
            <= V_RECOVERY_TOLERANCE
        ):
            mark_range(
                mask,
                center_index - 1,
                center_index + 2,
            )

    return mask


def build_quality_mask(
    spo2: np.ndarray,
) -> pd.DataFrame:
    sample_count = len(spo2)

    invalid_nonfinite = (
        ~np.isfinite(spo2)
    )

    invalid_absolute_range = (
        np.isfinite(spo2)
        & (
            (
                spo2
                < ABSOLUTE_MIN_SPO2
            )
            | (
                spo2
                > ABSOLUTE_MAX_SPO2
            )
        )
    )

    suspect_low_value = (
        np.isfinite(spo2)
        & (
            spo2
            < SUSPECT_LOW_SPO2
        )
        & (
            spo2
            >= ABSOLUTE_MIN_SPO2
        )
    )

    suspect_rapid_change = (
        detect_rapid_change_mask(
            spo2
        )
    )

    suspect_short_v_drop = (
        detect_short_v_drop_mask(
            spo2
        )
    )

    # 單靠 SpO₂ 長時間不變不足以判定 artifact。
# 目前先保留 flatline 偵測結果供報告，
# 但不將其直接列入 SUSPECT_ARTIFACT。
    suspect_flatline = (
        detect_flatline_mask(
         spo2
        )
    )

    invalid_mask = (
        invalid_nonfinite
        | invalid_absolute_range
    )

    suspect_mask = (
        suspect_low_value
        | suspect_rapid_change
        | suspect_short_v_drop
    )

    suspect_mask = (
        suspect_mask
        & ~invalid_mask
    )

    quality = np.full(
        sample_count,
        QUALITY_VALID,
        dtype=object,
    )

    quality[
        suspect_mask
    ] = QUALITY_SUSPECT

    quality[
        invalid_mask
    ] = QUALITY_INVALID

    quality_score = np.ones(
        sample_count,
        dtype=float,
    )

    quality_score[
        suspect_mask
    ] = 0.5

    quality_score[
        invalid_mask
    ] = 0.0

    dataframe = pd.DataFrame(
        {
            "second_index": np.arange(
                sample_count,
                dtype=int,
            ),
            "spo2": spo2,
            "quality_class": quality,
            "quality_score": (
                quality_score
            ),
            "invalid_nonfinite": (
                invalid_nonfinite
            ),
            "invalid_absolute_range": (
                invalid_absolute_range
            ),
            "suspect_low_value": (
                suspect_low_value
            ),
            "suspect_rapid_change": (
                suspect_rapid_change
            ),
            "suspect_short_v_drop": (
                suspect_short_v_drop
            ),
            "suspect_flatline": (
                suspect_flatline
            ),
        }
    )

    return dataframe


def finite_summary(
    values: np.ndarray,
) -> dict[str, Any]:
    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.size == 0:
        return {
            "finite_count": 0,
            "minimum": None,
            "p01": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }

    return {
        "finite_count": int(
            finite_values.size
        ),
        "minimum": float(
            np.nanmin(finite_values)
        ),
        "p01": float(
            np.nanpercentile(
                finite_values,
                1,
            )
        ),
        "p05": float(
            np.nanpercentile(
                finite_values,
                5,
            )
        ),
        "median": float(
            np.nanmedian(
                finite_values
            )
        ),
        "mean": float(
            np.nanmean(
                finite_values
            )
        ),
        "p95": float(
            np.nanpercentile(
                finite_values,
                95,
            )
        ),
        "p99": float(
            np.nanpercentile(
                finite_values,
                99,
            )
        ),
        "maximum": float(
            np.nanmax(finite_values)
        ),
    }


def calculate_summary(
    quality_df: pd.DataFrame,
    scale_mode: str,
    channel_names: dict[str, str | None],
) -> dict[str, Any]:
    total_count = len(
        quality_df
    )

    valid_mask = (
        quality_df[
            "quality_class"
        ]
        == QUALITY_VALID
    )

    suspect_mask = (
        quality_df[
            "quality_class"
        ]
        == QUALITY_SUSPECT
    )

    invalid_mask = (
        quality_df[
            "quality_class"
        ]
        == QUALITY_INVALID
    )

    valid_values = quality_df.loc[
        valid_mask,
        "spo2",
    ].to_numpy(
        dtype=float
    )

    all_values = quality_df[
        "spo2"
    ].to_numpy(
        dtype=float
    )

    def fraction(
        count: int,
    ) -> float | None:
        if total_count <= 0:
            return None

        return float(
            count
            / total_count
        )

    valid_count = int(
        valid_mask.sum()
    )

    suspect_count = int(
        suspect_mask.sum()
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    summary = {
        "output_sample_rate_hz": (
            OUTPUT_SAMPLE_RATE
        ),
        "spo2_scale_mode": scale_mode,
        "channels": channel_names,
        "total_second_count": int(
            total_count
        ),
        "valid_second_count": (
            valid_count
        ),
        "suspect_second_count": (
            suspect_count
        ),
        "invalid_second_count": (
            invalid_count
        ),
        "valid_fraction": fraction(
            valid_count
        ),
        "suspect_fraction": fraction(
            suspect_count
        ),
        "invalid_fraction": fraction(
            invalid_count
        ),
        "all_spo2_summary": (
            finite_summary(
                all_values
            )
        ),
        "valid_spo2_summary": (
            finite_summary(
                valid_values
            )
        ),
        "flag_counts": {
            "invalid_nonfinite": int(
                quality_df[
                    "invalid_nonfinite"
                ].sum()
            ),
            "invalid_absolute_range": int(
                quality_df[
                    "invalid_absolute_range"
                ].sum()
            ),
            "suspect_low_value": int(
                quality_df[
                    "suspect_low_value"
                ].sum()
            ),
            "suspect_rapid_change": int(
                quality_df[
                    "suspect_rapid_change"
                ].sum()
            ),
            "suspect_short_v_drop": int(
                quality_df[
                    "suspect_short_v_drop"
                ].sum()
            ),
            "suspect_flatline": int(
                quality_df[
                    "suspect_flatline"
                ].sum()
            ),
        },
        "quality_interpretation": (
            "VALID_PHYSIOLOGICAL 表示第一版規則未發現明顯 artifact；"
            "SUSPECT_ARTIFACT 表示保留原始值但建議排除於主要生理統計；"
            "INVALID 表示缺值或超出絕對允許範圍。"
        ),
        "safety_note": (
            "此品質遮罩為研究型訊號清理工具，"
            "不是臨床血氧判讀器。"
        ),
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "建立患者 SpO2 訊號品質遮罩。"
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

    edf_file = find_edf_file(
        patient_id
    )

    output_folder = (
        INFERENCE_ROOT
        / patient_id
        / "spo2_quality"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_csv = (
        output_folder
        / "spo2_quality_mask.csv"
    )

    summary_csv = (
        output_folder
        / "spo2_quality_summary.csv"
    )

    summary_json = (
        output_folder
        / "spo2_quality_summary.json"
    )

    print("=" * 80)
    print("SpO2 Signal Quality Mask")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    print(
        f"EDF：{edf_file}"
    )

    print()
    print("=" * 80)
    print("步驟 1：讀取 EDF Channel")
    print("=" * 80)

    raw = read_raw_edf(
        edf_file
    )

    saturation_channel = find_channel(
        raw.ch_names,
        [
            "Saturation",
            "SpO2",
            "SaO2",
            "Oxygen Saturation",
        ],
    )

    pulse_channel = find_channel(
        raw.ch_names,
        [
            "Pulse",
            "Pulse Rate",
        ],
    )

    pwa_channel = find_channel(
        raw.ch_names,
        [
            "PWA",
            "Pleth",
            "Plethysmography",
        ],
    )

    heart_rate_channel = find_channel(
        raw.ch_names,
        [
            "Heart Rate",
            "HR",
        ],
    )

    if saturation_channel is None:
        raise RuntimeError(
            "EDF 中找不到 Saturation / SpO2 Channel。"
        )

    print(
        "SpO2 Channel："
        f"{saturation_channel}"
    )

    print(
        "Pulse Channel："
        f"{pulse_channel}"
    )

    print(
        "PWA Channel："
        f"{pwa_channel}"
    )

    print(
        "Heart Rate Channel："
        f"{heart_rate_channel}"
    )

    print()
    print("=" * 80)
    print("步驟 2：建立每秒 SpO2")
    print("=" * 80)

    spo2_raw, sampling_rate = (
        get_channel_data(
            raw,
            saturation_channel,
        )
    )

    print(
        f"Sampling Rate："
        f"{sampling_rate} Hz"
    )

    print(
        f"Raw Samples："
        f"{len(spo2_raw)}"
    )

    spo2_raw, scale_mode = (
        detect_spo2_scale(
            spo2_raw
        )
    )

    print(
        f"SpO2 尺度處理："
        f"{scale_mode}"
    )

    spo2_second = (
        resample_to_one_second(
            spo2_raw,
            sampling_rate,
        )
    )

    print(
        f"每秒資料點："
        f"{len(spo2_second)}"
    )

    print()
    print("=" * 80)
    print("步驟 3：建立品質遮罩")
    print("=" * 80)

    quality_df = build_quality_mask(
        spo2_second
    )

    pulse_second = (
        resample_auxiliary_channel(
            raw,
            pulse_channel,
            len(quality_df),
        )
    )

    pwa_second = (
        resample_auxiliary_channel(
            raw,
            pwa_channel,
            len(quality_df),
        )
    )

    heart_rate_second = (
        resample_auxiliary_channel(
            raw,
            heart_rate_channel,
            len(quality_df),
        )
    )

    quality_df[
        "pulse"
    ] = pulse_second

    quality_df[
        "pwa"
    ] = pwa_second

    quality_df[
        "heart_rate"
    ] = heart_rate_second

    valid_count = int(
        (
            quality_df[
                "quality_class"
            ]
            == QUALITY_VALID
        ).sum()
    )

    suspect_count = int(
        (
            quality_df[
                "quality_class"
            ]
            == QUALITY_SUSPECT
        ).sum()
    )

    invalid_count = int(
        (
            quality_df[
                "quality_class"
            ]
            == QUALITY_INVALID
        ).sum()
    )

    total_count = len(
        quality_df
    )

    print(
        f"VALID_PHYSIOLOGICAL："
        f"{valid_count} "
        f"({valid_count / total_count:.2%})"
    )

    print(
        f"SUSPECT_ARTIFACT："
        f"{suspect_count} "
        f"({suspect_count / total_count:.2%})"
    )

    print(
        f"INVALID："
        f"{invalid_count} "
        f"({invalid_count / total_count:.2%})"
    )

    print()
    print("Artifact flags：")

    flag_columns = [
        "invalid_nonfinite",
        "invalid_absolute_range",
        "suspect_low_value",
        "suspect_rapid_change",
        "suspect_short_v_drop",
        "suspect_flatline",
    ]

    for column in flag_columns:
        print(
            f"  {column}："
            f"{int(quality_df[column].sum())}"
        )

    channel_names = {
        "spo2": saturation_channel,
        "pulse": pulse_channel,
        "pwa": pwa_channel,
        "heart_rate": (
            heart_rate_channel
        ),
    }

    summary = calculate_summary(
        quality_df,
        scale_mode,
        channel_names,
    )

    print()
    print("=" * 80)
    print("步驟 4：SpO2 統計比較")
    print("=" * 80)

    all_summary = summary[
        "all_spo2_summary"
    ]

    valid_summary = summary[
        "valid_spo2_summary"
    ]

    print("原始每秒 SpO2 ：")

    print(
        f"  minimum："
        f"{all_summary['minimum']}"
    )

    print(
        f"  p01："
        f"{all_summary['p01']}"
    )

    print(
        f"  p05："
        f"{all_summary['p05']}"
    )

    print(
        f"  median："
        f"{all_summary['median']}"
    )

    print(
        f"  mean："
        f"{all_summary['mean']}"
    )

    print()
    print("品質合格 SpO2：")

    print(
        f"  minimum："
        f"{valid_summary['minimum']}"
    )

    print(
        f"  p01："
        f"{valid_summary['p01']}"
    )

    print(
        f"  p05："
        f"{valid_summary['p05']}"
    )

    print(
        f"  median："
        f"{valid_summary['median']}"
    )

    print(
        f"  mean："
        f"{valid_summary['mean']}"
    )

    quality_df.to_csv(
        quality_csv,
        index=False,
        encoding="utf-8-sig",
    )

    summary_row = {
        "patient_id": patient_id,
        "total_second_count": (
            summary[
                "total_second_count"
            ]
        ),
        "valid_second_count": (
            summary[
                "valid_second_count"
            ]
        ),
        "suspect_second_count": (
            summary[
                "suspect_second_count"
            ]
        ),
        "invalid_second_count": (
            summary[
                "invalid_second_count"
            ]
        ),
        "valid_fraction": (
            summary[
                "valid_fraction"
            ]
        ),
        "suspect_fraction": (
            summary[
                "suspect_fraction"
            ]
        ),
        "invalid_fraction": (
            summary[
                "invalid_fraction"
            ]
        ),
        "raw_spo2_min": (
            all_summary["minimum"]
        ),
        "raw_spo2_p01": (
            all_summary["p01"]
        ),
        "raw_spo2_p05": (
            all_summary["p05"]
        ),
        "raw_spo2_mean": (
            all_summary["mean"]
        ),
        "raw_spo2_median": (
            all_summary["median"]
        ),
        "valid_spo2_min": (
            valid_summary["minimum"]
        ),
        "valid_spo2_p01": (
            valid_summary["p01"]
        ),
        "valid_spo2_p05": (
            valid_summary["p05"]
        ),
        "valid_spo2_mean": (
            valid_summary["mean"]
        ),
        "valid_spo2_median": (
            valid_summary["median"]
        ),
    }

    pd.DataFrame(
        [summary_row]
    ).to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    output_json_data = {
        "patient_id": patient_id,
        "edf_file": str(edf_file),
        **summary,
    }

    save_json(
        summary_json,
        output_json_data,
    )

    print()
    print("=" * 80)
    print("SpO2 品質遮罩完成")
    print("=" * 80)

    print(
        f"品質遮罩 CSV："
        f"{quality_csv}"
    )

    print(
        f"摘要 CSV："
        f"{summary_csv}"
    )

    print(
        f"摘要 JSON："
        f"{summary_json}"
    )

    print()
    print(
        "注意：SUSPECT_ARTIFACT 不代表一定是假訊號；"
        "目前採保守策略，先從主要生理統計排除，"
        "但保留原始值與原因旗標供後續檢查。"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()