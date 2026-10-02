from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import mne
import numpy as np
import pandas as pd
try:
    import pyedflib
except ModuleNotFoundError:  # Python 3.13 Windows may not have a wheel.
    pyedflib = None
from scipy.signal import (
    find_peaks,
    resample,
    welch,
)

from src.features.channel_mapper import ChannelMapper


RESPIRATORY_CHANNELS = [
    "flow",
    "thermistor",
    "spo2",
    "thorax",
    "abdomen",
    "heart_rate",
    "position",
    "snore",
]


def to_boolean_series(
    series: pd.Series,
) -> pd.Series:
    """
    將 CSV 讀進來的 True/False、1/0、yes/no
    統一轉成布林值。
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "false": False,
                "0": False,
                "no": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def finite_values(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return values[
        np.isfinite(values)
    ]


def calculate_basic_features(
    values: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """
    通用時域統計特徵。
    """
    x = finite_values(values)

    names = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "range",
        "rms",
        "p05",
        "p25",
        "p75",
        "p95",
        "iqr",
        "diff_std",
    ]

    if x.size == 0:
        return {
            f"{prefix}_{name}": np.nan
            for name in names
        }

    differences = np.diff(x)

    p05, p25, p75, p95 = np.percentile(
        x,
        [5, 25, 75, 95],
    )

    return {
        f"{prefix}_mean": float(
            np.mean(x)
        ),
        f"{prefix}_std": float(
            np.std(x)
        ),
        f"{prefix}_min": float(
            np.min(x)
        ),
        f"{prefix}_max": float(
            np.max(x)
        ),
        f"{prefix}_median": float(
            np.median(x)
        ),
        f"{prefix}_range": float(
            np.max(x) - np.min(x)
        ),
        f"{prefix}_rms": float(
            np.sqrt(
                np.mean(
                    np.square(x)
                )
            )
        ),
        f"{prefix}_p05": float(p05),
        f"{prefix}_p25": float(p25),
        f"{prefix}_p75": float(p75),
        f"{prefix}_p95": float(p95),
        f"{prefix}_iqr": float(
            p75 - p25
        ),
        f"{prefix}_diff_std": (
            float(
                np.std(differences)
            )
            if differences.size > 0
            else 0.0
        ),
    }


def estimate_respiratory_rate(
    values: np.ndarray,
    sampling_rate: float,
) -> dict[str, float]:
    """
    使用功率頻譜估計呼吸頻率。

    合理搜尋範圍：
    0.08～0.7 Hz
    約等於 4.8～42 次/分鐘。
    """
    result = {
        "respiratory_rate_bpm": np.nan,
        "respiratory_peak_power": np.nan,
        "respiratory_band_power": np.nan,
    }

    x = finite_values(values)

    if (
        x.size < 8
        or sampling_rate <= 0
    ):
        return result

    x = x - np.mean(x)

    if np.std(x) <= 1e-12:
        return result

    nperseg = min(
        len(x),
        max(
            64,
            int(sampling_rate * 20),
        ),
    )

    frequencies, power = welch(
        x,
        fs=sampling_rate,
        nperseg=nperseg,
    )

    respiratory_mask = (
        (frequencies >= 0.08)
        & (frequencies <= 0.70)
    )

    respiratory_frequencies = (
        frequencies[respiratory_mask]
    )

    respiratory_power = (
        power[respiratory_mask]
    )

    if respiratory_power.size == 0:
        return result

    peak_index = int(
        np.argmax(respiratory_power)
    )

    peak_frequency = float(
        respiratory_frequencies[
            peak_index
        ]
    )

    result[
        "respiratory_rate_bpm"
    ] = peak_frequency * 60.0

    result[
        "respiratory_peak_power"
    ] = float(
        respiratory_power[
            peak_index
        ]
    )

    if respiratory_power.size >= 2:
        result[
            "respiratory_band_power"
        ] = float(
            np.trapezoid(
                respiratory_power,
                respiratory_frequencies,
            )
        )

    return result


def calculate_airflow_features(
    values: np.ndarray,
    sampling_rate: float,
    prefix: str,
) -> dict[str, float]:
    """
    Flow 或 Thermistor 的呼吸型態特徵。

    flattening_proxy、low_amplitude_fraction 等為研究型 proxy，
    不能直接視為臨床診斷。
    """
    result = calculate_basic_features(
        values,
        prefix,
    )

    x = finite_values(values)

    extra_names = [
        "amplitude_p90_p10",
        "low_amplitude_fraction",
        "zero_crossing_rate",
        "flattening_proxy",
        "respiratory_rate_bpm",
        "respiratory_peak_power",
        "respiratory_band_power",
    ]

    for name in extra_names:
        result[
            f"{prefix}_{name}"
        ] = np.nan

    if (
        x.size < 8
        or sampling_rate <= 0
    ):
        return result

    centered = x - np.median(x)

    p10, p90 = np.percentile(
        centered,
        [10, 90],
    )

    amplitude = float(
        p90 - p10
    )

    result[
        f"{prefix}_amplitude_p90_p10"
    ] = amplitude

    absolute = np.abs(centered)

    amplitude_threshold = float(
        np.percentile(
            absolute,
            25,
        )
    )

    result[
        f"{prefix}_low_amplitude_fraction"
    ] = float(
        np.mean(
            absolute
            <= amplitude_threshold
        )
    )

    signs = np.signbit(centered)

    zero_crossings = int(
        np.count_nonzero(
            signs[1:] != signs[:-1]
        )
    )

    duration_seconds = (
        len(centered)
        / sampling_rate
    )

    result[
        f"{prefix}_zero_crossing_rate"
    ] = (
        zero_crossings
        / duration_seconds
        if duration_seconds > 0
        else np.nan
    )

    # 平台區域 proxy：
    # 局部斜率越小的樣本比例越高，值越大。
    differences = np.abs(
        np.diff(centered)
    )

    if differences.size > 0:
        slope_threshold = float(
            np.percentile(
                differences,
                25,
            )
        )

        result[
            f"{prefix}_flattening_proxy"
        ] = float(
            np.mean(
                differences
                <= slope_threshold
            )
        )

    rate_features = (
        estimate_respiratory_rate(
            centered,
            sampling_rate,
        )
    )

    for name, value in (
        rate_features.items()
    ):
        result[
            f"{prefix}_{name}"
        ] = value

    return result


def normalize_spo2(
    values: np.ndarray,
) -> np.ndarray:
    """
    將可能為 0～1 或 0～100 的 SpO₂ 統一成百分比。

    超出合理範圍的數值改為 NaN。
    """
    x = np.asarray(
        values,
        dtype=np.float64,
    ).copy()

    valid = x[
        np.isfinite(x)
    ]

    if valid.size == 0:
        return x

    median_value = float(
        np.median(valid)
    )

    # 某些來源可能把 97% 記成 0.97
    if 0.5 <= median_value <= 1.2:
        x = x * 100.0

    invalid_mask = (
        ~np.isfinite(x)
        | (x < 40.0)
        | (x > 100.5)
    )

    x[invalid_mask] = np.nan

    return x


def calculate_spo2_features(
    values: np.ndarray,
    sampling_rate: float,
) -> dict[str, float]:
    """
    30 秒內的血氧特徵。

    這裡的 below_90_fraction 是 epoch 內比例，
    不是整晚 T90。
    """
    spo2 = normalize_spo2(values)

    result = calculate_basic_features(
        spo2,
        "spo2",
    )

    extra = {
        "spo2_valid_fraction": np.nan,
        "spo2_below_90_fraction": np.nan,
        "spo2_below_88_fraction": np.nan,
        "spo2_drop_from_start": np.nan,
        "spo2_end_minus_start": np.nan,
        "spo2_linear_slope_per_minute": np.nan,
        "spo2_largest_drop": np.nan,
    }

    result.update(extra)

    valid_mask = np.isfinite(spo2)

    result["spo2_valid_fraction"] = float(
        np.mean(valid_mask)
    )

    x = spo2[valid_mask]

    if (
        x.size < 2
        or sampling_rate <= 0
    ):
        return result

    result[
        "spo2_below_90_fraction"
    ] = float(
        np.mean(x < 90.0)
    )

    result[
        "spo2_below_88_fraction"
    ] = float(
        np.mean(x < 88.0)
    )

    start_window = max(
        1,
        min(
            x.size,
            int(
                round(
                    sampling_rate * 3
                )
            ),
        ),
    )

    start_value = float(
        np.median(
            x[:start_window]
        )
    )

    end_value = float(
        np.median(
            x[-start_window:]
        )
    )

    minimum_value = float(
        np.min(x)
    )

    result[
        "spo2_drop_from_start"
    ] = float(
        start_value - minimum_value
    )

    result[
        "spo2_end_minus_start"
    ] = float(
        end_value - start_value
    )

    times = np.arange(
        x.size,
        dtype=np.float64,
    ) / sampling_rate

    if np.ptp(times) > 0:
        slope_per_second = float(
            np.polyfit(
                times,
                x,
                deg=1,
            )[0]
        )

        result[
            "spo2_linear_slope_per_minute"
        ] = (
            slope_per_second * 60.0
        )

    differences = np.diff(x)

    if differences.size > 0:
        result[
            "spo2_largest_drop"
        ] = float(
            abs(
                min(
                    np.min(differences),
                    0.0,
                )
            )
        )

    return result


def calculate_heart_rate_features(
    values: np.ndarray,
) -> dict[str, float]:
    """
    心率 channel 的 epoch 特徵。
    """
    result = calculate_basic_features(
        values,
        "heart_rate",
    )

    x = finite_values(values)

    result[
        "heart_rate_valid_fraction"
    ] = np.nan

    result[
        "heart_rate_outlier_fraction"
    ] = np.nan

    if x.size == 0:
        return result

    valid_mask = (
        (x >= 25.0)
        & (x <= 220.0)
    )

    result[
        "heart_rate_valid_fraction"
    ] = float(
        np.mean(valid_mask)
    )

    result[
        "heart_rate_outlier_fraction"
    ] = float(
        np.mean(~valid_mask)
    )

    return result


def calculate_position_features(
    values: np.ndarray,
    sampling_rate: float,
) -> dict[str, float]:
    """
    姿勢角度的變動特徵。

    尚未套用使用者的姿勢 codebook，因此這裡不直接輸出
    左躺、右躺、仰睡或趴睡。
    """
    result = calculate_basic_features(
        values,
        "position",
    )

    result[
        "position_change_count"
    ] = np.nan

    result[
        "position_change_rate_per_minute"
    ] = np.nan

    result[
        "position_stability_fraction"
    ] = np.nan

    x = finite_values(values)

    if (
        x.size < 2
        or sampling_rate <= 0
    ):
        return result

    differences = np.abs(
        np.diff(x)
    )

    # 角度變動門檻先採 10 度。
    change_mask = (
        differences >= 10.0
    )

    change_count = int(
        np.count_nonzero(
            change_mask
        )
    )

    duration_minutes = (
        x.size
        / sampling_rate
        / 60.0
    )

    result[
        "position_change_count"
    ] = change_count

    result[
        "position_change_rate_per_minute"
    ] = (
        change_count
        / duration_minutes
        if duration_minutes > 0
        else np.nan
    )

    result[
        "position_stability_fraction"
    ] = float(
        np.mean(
            differences < 2.0
        )
    )

    return result


def calculate_snore_features(
    values: np.ndarray,
) -> dict[str, float]:
    """
    鼾聲訊號的活動程度特徵。
    """
    result = calculate_basic_features(
        values,
        "snore",
    )

    result[
        "snore_activity_fraction"
    ] = np.nan

    result[
        "snore_peak_count"
    ] = np.nan

    x = finite_values(values)

    if x.size < 4:
        return result

    centered = x - np.median(x)

    absolute = np.abs(centered)

    median_absolute = float(
        np.median(absolute)
    )

    mad = float(
        np.median(
            np.abs(
                absolute
                - median_absolute
            )
        )
    )

    threshold = (
        median_absolute
        + 3.0 * max(
            mad,
            1e-12,
        )
    )

    result[
        "snore_activity_fraction"
    ] = float(
        np.mean(
            absolute > threshold
        )
    )

    peaks, _ = find_peaks(
        absolute,
        height=threshold,
    )

    result[
        "snore_peak_count"
    ] = int(
        len(peaks)
    )

    return result


def resample_to_length(
    values: np.ndarray,
    target_length: int,
) -> np.ndarray:
    x = finite_values(values)

    if (
        x.size == 0
        or target_length <= 0
    ):
        return np.empty(
            0,
            dtype=np.float64,
        )

    if x.size == target_length:
        return x

    return np.asarray(
        resample(
            x,
            target_length,
        ),
        dtype=np.float64,
    )


def calculate_thoracoabdominal_features(
    thorax: np.ndarray,
    abdomen: np.ndarray,
    thorax_rate: float,
    abdomen_rate: float,
) -> dict[str, float]:
    """
    胸帶與腹帶同步程度。

    correlation 與 lag 是 proxy，
    不是正式的 RIP phase angle 臨床測量。
    """
    result = {
        "thorax_abdomen_correlation": np.nan,
        "thorax_abdomen_absolute_correlation": np.nan,
        "thorax_abdomen_lag_seconds": np.nan,
        "thorax_abdomen_amplitude_ratio": np.nan,
        "thorax_abdomen_opposite_phase_fraction": np.nan,
    }

    thorax_x = finite_values(thorax)
    abdomen_x = finite_values(abdomen)

    if (
        thorax_x.size < 8
        or abdomen_x.size < 8
        or thorax_rate <= 0
        or abdomen_rate <= 0
    ):
        return result

    target_rate = min(
        thorax_rate,
        abdomen_rate,
        50.0,
    )

    duration_seconds = min(
        thorax_x.size / thorax_rate,
        abdomen_x.size / abdomen_rate,
    )

    target_length = int(
        round(
            duration_seconds
            * target_rate
        )
    )

    if target_length < 8:
        return result

    thorax_resampled = resample_to_length(
        thorax_x,
        target_length,
    )

    abdomen_resampled = resample_to_length(
        abdomen_x,
        target_length,
    )

    thorax_centered = (
        thorax_resampled
        - np.mean(thorax_resampled)
    )

    abdomen_centered = (
        abdomen_resampled
        - np.mean(abdomen_resampled)
    )

    thorax_std = float(
        np.std(thorax_centered)
    )

    abdomen_std = float(
        np.std(abdomen_centered)
    )

    if (
        thorax_std <= 1e-12
        or abdomen_std <= 1e-12
    ):
        return result

    correlation = float(
        np.corrcoef(
            thorax_centered,
            abdomen_centered,
        )[0, 1]
    )

    result[
        "thorax_abdomen_correlation"
    ] = correlation

    result[
        "thorax_abdomen_absolute_correlation"
    ] = abs(correlation)

    result[
        "thorax_abdomen_amplitude_ratio"
    ] = float(
        thorax_std
        / abdomen_std
    )

    result[
        "thorax_abdomen_opposite_phase_fraction"
    ] = float(
        np.mean(
            np.sign(thorax_centered)
            != np.sign(abdomen_centered)
        )
    )

    normalized_thorax = (
        thorax_centered
        / thorax_std
    )

    normalized_abdomen = (
        abdomen_centered
        / abdomen_std
    )

    full_correlation = np.correlate(
        normalized_thorax,
        normalized_abdomen,
        mode="full",
    )

    lags = np.arange(
        -target_length + 1,
        target_length,
    )

    max_lag_samples = int(
        round(
            target_rate * 5.0
        )
    )

    allowed = (
        np.abs(lags)
        <= max_lag_samples
    )

    if np.any(allowed):
        allowed_correlation = (
            full_correlation[allowed]
        )

        allowed_lags = lags[allowed]

        best_index = int(
            np.argmax(
                np.abs(
                    allowed_correlation
                )
            )
        )

        best_lag_samples = int(
            allowed_lags[
                best_index
            ]
        )

        result[
            "thorax_abdomen_lag_seconds"
        ] = float(
            best_lag_samples
            / target_rate
        )

    return result


class RespiratoryFeatureBuilder:
    def __init__(
        self,
        channel_mapper: ChannelMapper,
    ) -> None:
        self.channel_mapper = (
            channel_mapper
        )

    def build_patient_features(
        self,
        patient_id: str,
        edf_file: str | Path,
        stages_file: str | Path,
    ) -> pd.DataFrame:
        edf_path = Path(edf_file)
        stages_path = Path(
            stages_file
        )

        if not edf_path.exists():
            raise FileNotFoundError(
                f"找不到 EDF：{edf_path}"
            )

        if not stages_path.exists():
            raise FileNotFoundError(
                f"找不到 Stage CSV："
                f"{stages_path}"
            )

        stages = pd.read_csv(
            stages_path,
            parse_dates=[
                "start_time",
                "end_time",
            ],
        )

        required_columns = {
            "epoch_index",
            "original_epoch",
            "start_time",
            "end_time",
            "stage",
            "edf_start_seconds",
            "edf_end_seconds",
            "usable_for_stage_training",
        }

        missing_columns = (
            required_columns
            - set(stages.columns)
        )

        if missing_columns:
            raise RuntimeError(
                "stages_aligned.csv 缺少欄位："
                f"{sorted(missing_columns)}"
            )

        usable_mask = to_boolean_series(
            stages[
                "usable_for_stage_training"
            ]
        )

        stages = stages[
            usable_mask
        ].copy()

        if stages.empty:
            raise RuntimeError(
                f"{patient_id} 沒有可用 Epoch。"
            )

        source = self._load_signals(
            edf_path
        )

        try:
            rows: list[dict] = []

            for stage_row in (
                stages.itertuples(
                    index=False
                )
            ):
                start_seconds = float(
                    stage_row
                    .edf_start_seconds
                )

                end_seconds = float(
                    stage_row
                    .edf_end_seconds
                )

                segments: dict[
                    str,
                    np.ndarray,
                ] = {}

                rates: dict[
                    str,
                    float,
                ] = {}

                feature_row: dict[
                    str,
                    Any,
                ] = {
                    "patient_id": patient_id,
                    "epoch_index": int(
                        stage_row.epoch_index
                    ),
                    "original_epoch": int(
                        stage_row
                        .original_epoch
                    ),
                    "start_time": (
                        stage_row.start_time
                    ),
                    "end_time": (
                        stage_row.end_time
                    ),
                    "stage": (
                        stage_row.stage
                    ),
                    "edf_start_seconds": (
                        start_seconds
                    ),
                    "edf_end_seconds": (
                        end_seconds
                    ),
                    "respiratory_edf_backend": (
                        source["backend"]
                    ),
                }

                epoch_valid = True

                for standard_name in (
                    RESPIRATORY_CHANNELS
                ):
                    sampling_rate = float(
                        source[
                            "sampling_rates"
                        ][standard_name]
                    )

                    expected_samples = int(
                        round(
                            30.0
                            * sampling_rate
                        )
                    )

                    if "read_segment" in source:
                        segment = source["read_segment"](
                            standard_name,
                            start_seconds,
                            end_seconds,
                        )
                    else:
                        signal = source["signals"][standard_name]
                        start_sample = max(0, int(round(start_seconds * sampling_rate)))
                        end_sample = min(len(signal), int(round(end_seconds * sampling_rate)))
                        segment = np.asarray(
                            signal[start_sample:end_sample],
                            dtype=np.float64,
                        ) if end_sample > start_sample else np.empty(0, dtype=np.float64)

                    coverage = (
                        len(segment)
                        / max(
                            expected_samples,
                            1,
                        )
                    )

                    if coverage < 0.99:
                        epoch_valid = False

                    feature_row[
                        f"{standard_name}_source"
                    ] = (
                        source[
                            "source_channels"
                        ][standard_name]
                    )

                    feature_row[
                        f"{standard_name}_sampling_rate"
                    ] = sampling_rate

                    feature_row[
                        f"{standard_name}_samples"
                    ] = len(segment)

                    feature_row[
                        f"{standard_name}_coverage"
                    ] = coverage

                    segments[
                        standard_name
                    ] = segment

                    rates[
                        standard_name
                    ] = sampling_rate

                feature_row.update(
                    calculate_airflow_features(
                        values=segments[
                            "flow"
                        ],
                        sampling_rate=rates[
                            "flow"
                        ],
                        prefix="flow",
                    )
                )

                feature_row.update(
                    calculate_airflow_features(
                        values=segments[
                            "thermistor"
                        ],
                        sampling_rate=rates[
                            "thermistor"
                        ],
                        prefix="thermistor",
                    )
                )

                feature_row.update(
                    calculate_basic_features(
                        values=segments[
                            "thorax"
                        ],
                        prefix="thorax",
                    )
                )

                feature_row.update(
                    estimate_respiratory_rate(
                        values=segments[
                            "thorax"
                        ],
                        sampling_rate=rates[
                            "thorax"
                        ],
                    )
                )

                # 避免 thorax 的 respiratory rate
                # 和其他欄位名稱衝突。
                for key in [
                    "respiratory_rate_bpm",
                    "respiratory_peak_power",
                    "respiratory_band_power",
                ]:
                    feature_row[
                        f"thorax_{key}"
                    ] = feature_row.pop(
                        key
                    )

                feature_row.update(
                    calculate_basic_features(
                        values=segments[
                            "abdomen"
                        ],
                        prefix="abdomen",
                    )
                )

                abdomen_rate_features = (
                    estimate_respiratory_rate(
                        values=segments[
                            "abdomen"
                        ],
                        sampling_rate=rates[
                            "abdomen"
                        ],
                    )
                )

                for key, value in (
                    abdomen_rate_features.items()
                ):
                    feature_row[
                        f"abdomen_{key}"
                    ] = value

                feature_row.update(
                    calculate_thoracoabdominal_features(
                        thorax=segments[
                            "thorax"
                        ],
                        abdomen=segments[
                            "abdomen"
                        ],
                        thorax_rate=rates[
                            "thorax"
                        ],
                        abdomen_rate=rates[
                            "abdomen"
                        ],
                    )
                )

                feature_row.update(
                    calculate_spo2_features(
                        values=segments[
                            "spo2"
                        ],
                        sampling_rate=rates[
                            "spo2"
                        ],
                    )
                )

                feature_row.update(
                    calculate_heart_rate_features(
                        values=segments[
                            "heart_rate"
                        ]
                    )
                )

                feature_row.update(
                    calculate_position_features(
                        values=segments[
                            "position"
                        ],
                        sampling_rate=rates[
                            "position"
                        ],
                    )
                )

                feature_row.update(
                    calculate_snore_features(
                        values=segments[
                            "snore"
                        ]
                    )
                )

                feature_row[
                    "epoch_signal_valid"
                ] = epoch_valid

                rows.append(
                    feature_row
                )

            result = pd.DataFrame(
                rows
            )

            if result.empty:
                return result

            result = result[
                result[
                    "epoch_signal_valid"
                ].astype(bool)
            ].reset_index(
                drop=True
            )

            return result

        finally:
            close_callback: (
                Callable[[], None] | None
            ) = source.get("close")

            if callable(close_callback):
                close_callback()

    def _load_signals(
        self,
        edf_path: Path,
    ) -> dict[str, Any]:
        try:
            source = (
                self._load_with_pyedflib(
                    edf_path
                )
            )

            print(
                "Respiratory backend："
                "pyEDFlib"
            )

            return source

        except Exception as exc:
            print(
                "pyEDFlib 無法讀取，"
                "呼吸特徵改用 "
                "MNE preload："
                f"{exc}"
            )

            source = (
                self._load_with_mne(
                    edf_path
                )
            )

            print(
                "Respiratory backend："
                "MNE preload"
            )

            return source

    def _load_with_pyedflib(
        self,
        edf_path: Path,
    ) -> dict[str, Any]:
        if pyedflib is None:
            raise RuntimeError("pyEDFlib 未安裝；改用 MNE/edfio EDF backend")
        reader = pyedflib.EdfReader(
            str(edf_path)
        )

        try:
            labels = [
                str(label).strip()
                for label in (
                    reader.getSignalLabels()
                )
            ]

            mapped = (
                self.channel_mapper
                .map_channels(labels)
            )

            missing = (
                self.channel_mapper
                .validate_required(
                    mapped,
                    RESPIRATORY_CHANNELS,
                )
            )

            if missing:
                raise RuntimeError(
                    f"缺少呼吸 Channel："
                    f"{missing}"
                )

            label_to_index = {
                label: index
                for index, label
                in enumerate(labels)
            }

            signals: dict[
                str,
                np.ndarray,
            ] = {}

            sampling_rates: dict[
                str,
                float,
            ] = {}

            source_channels: dict[
                str,
                str,
            ] = {}

            for standard_name in (
                RESPIRATORY_CHANNELS
            ):
                source_channel = (
                    mapped[standard_name]
                )

                if source_channel is None:
                    raise RuntimeError(
                        f"{standard_name} "
                        "找不到來源 Channel。"
                    )

                index = label_to_index[
                    source_channel
                ]

                signals[
                    standard_name
                ] = np.asarray(
                    reader.readSignal(index),
                    dtype=np.float64,
                )

                sampling_rates[
                    standard_name
                ] = float(
                    reader.getSampleFrequency(
                        index
                    )
                )

                source_channels[
                    standard_name
                ] = source_channel

            return {
                "backend": "pyedflib",
                "signals": signals,
                "sampling_rates": (
                    sampling_rates
                ),
                "source_channels": (
                    source_channels
                ),
                "close": reader.close,
            }

        except Exception:
            reader.close()
            raise

    def _load_with_mne(
        self,
        edf_path: Path,
    ) -> dict[str, Any]:
        raw = mne.io.read_raw_edf(
            edf_path,
            preload=False,
            verbose="ERROR",
        )

        mapped = (
            self.channel_mapper
            .map_channels(
                raw.ch_names
            )
        )

        missing = (
            self.channel_mapper
            .validate_required(
                mapped,
                RESPIRATORY_CHANNELS,
            )
        )

        if missing:
            raw.close()

            raise RuntimeError(
                f"缺少呼吸 Channel："
                f"{missing}"
            )

        source_channels = {
            standard_name: mapped[
                standard_name
            ]
            for standard_name in (
                RESPIRATORY_CHANNELS
            )
        }

        selected_channels = list(
            dict.fromkeys(
                source_channel
                for source_channel
                in source_channels.values()
                if source_channel
                is not None
            )
        )

        common_rate = float(
            raw.info["sfreq"]
        )

        # Decode only one 30-second segment at a time.  This keeps the memory
        # bound to an epoch instead of the full-night recording.
        def read_segment(
            standard_name: str,
            start_seconds: float,
            end_seconds: float,
        ) -> np.ndarray:
            source_channel = source_channels[standard_name]
            start = max(0, int(round(start_seconds * common_rate)))
            stop = min(raw.n_times, int(round(end_seconds * common_rate)))
            if stop <= start:
                return np.empty(0, dtype=np.float64)
            return np.asarray(
                raw.get_data(picks=[source_channel], start=start, stop=stop)[0],
                dtype=np.float64,
            )

        sampling_rates = {
            standard_name: common_rate
            for standard_name
            in RESPIRATORY_CHANNELS
        }

        return {
            "backend": "mne_epoch_segments",
            "sampling_rates": (
                sampling_rates
            ),
            "source_channels": (
                source_channels
            ),
            "read_segment": read_segment,
            "close": raw.close,
        }
