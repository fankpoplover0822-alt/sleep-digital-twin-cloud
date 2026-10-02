from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
try:
    import pyedflib
except ModuleNotFoundError:  # Use MNE/edfio on Python 3.13 Windows.
    pyedflib = None
from scipy.signal import welch

from src.features.channel_mapper import ChannelMapper


STAGE_CHANNELS = [
    "eeg_c3",
    "eeg_c4",
    "eog_left",
    "eog_right",
    "flow",
    "spo2",
    "thorax",
    "abdomen",
    "heart_rate",
    "position",
]

EEG_CHANNELS = {
    "eeg_c3",
    "eeg_c4",
}

FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}


def calculate_basic_features(
    values: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = values[
        np.isfinite(values)
    ]

    feature_names = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "range",
        "rms",
        "diff_std",
        "p05",
        "p95",
        "iqr",
    ]

    if finite.size == 0:
        return {
            f"{prefix}_{name}": np.nan
            for name in feature_names
        }

    differences = np.diff(finite)

    p05 = float(
        np.percentile(finite, 5)
    )

    p25 = float(
        np.percentile(finite, 25)
    )

    p75 = float(
        np.percentile(finite, 75)
    )

    p95 = float(
        np.percentile(finite, 95)
    )

    return {
        f"{prefix}_mean": float(
            np.mean(finite)
        ),
        f"{prefix}_std": float(
            np.std(finite)
        ),
        f"{prefix}_min": float(
            np.min(finite)
        ),
        f"{prefix}_max": float(
            np.max(finite)
        ),
        f"{prefix}_median": float(
            np.median(finite)
        ),
        f"{prefix}_range": float(
            np.max(finite)
            - np.min(finite)
        ),
        f"{prefix}_rms": float(
            np.sqrt(
                np.mean(
                    np.square(finite)
                )
            )
        ),
        f"{prefix}_diff_std": (
            float(
                np.std(differences)
            )
            if differences.size > 0
            else 0.0
        ),
        f"{prefix}_p05": p05,
        f"{prefix}_p95": p95,
        f"{prefix}_iqr": p75 - p25,
    }


def calculate_spectral_features(
    values: np.ndarray,
    sampling_rate: float,
    prefix: str,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    result: dict[str, float] = {}

    for band_name in FREQUENCY_BANDS:
        result[
            f"{prefix}_{band_name}_power"
        ] = np.nan

        result[
            f"{prefix}_{band_name}_relative"
        ] = np.nan

    result[
        f"{prefix}_spectral_entropy"
    ] = np.nan

    result[
        f"{prefix}_dominant_frequency"
    ] = np.nan

    if (
        values.size < 4
        or sampling_rate <= 0
    ):
        return result

    centered = (
        values - np.mean(values)
    )

    nperseg = min(
        len(centered),
        max(
            64,
            int(sampling_rate * 4),
        ),
    )

    frequencies, power = welch(
        centered,
        fs=sampling_rate,
        nperseg=nperseg,
    )

    valid_mask = (
        (frequencies >= 0.5)
        & (frequencies <= 30.0)
    )

    valid_frequencies = (
        frequencies[valid_mask]
    )

    valid_power = (
        power[valid_mask]
    )

    if valid_power.size == 0:
        return result

    total_power = float(
        np.trapezoid(
            valid_power,
            valid_frequencies,
        )
    )

    for band_name, (
        low_frequency,
        high_frequency,
    ) in FREQUENCY_BANDS.items():
        band_mask = (
            (frequencies >= low_frequency)
            & (frequencies < high_frequency)
        )

        if band_mask.sum() < 2:
            band_power = 0.0
        else:
            band_power = float(
                np.trapezoid(
                    power[band_mask],
                    frequencies[band_mask],
                )
            )

        result[
            f"{prefix}_{band_name}_power"
        ] = band_power

        result[
            f"{prefix}_{band_name}_relative"
        ] = (
            band_power / total_power
            if total_power > 0
            else np.nan
        )

    total_discrete_power = float(
        np.sum(valid_power)
    )

    if total_discrete_power > 0:
        normalized_power = (
            valid_power
            / total_discrete_power
        )

        positive_power = (
            normalized_power[
                normalized_power > 0
            ]
        )

        entropy = -np.sum(
            positive_power
            * np.log2(positive_power)
        )

        maximum_entropy = np.log2(
            len(normalized_power)
        )

        result[
            f"{prefix}_spectral_entropy"
        ] = (
            float(
                entropy
                / maximum_entropy
            )
            if maximum_entropy > 0
            else 0.0
        )

    dominant_index = int(
        np.argmax(valid_power)
    )

    result[
        f"{prefix}_dominant_frequency"
    ] = float(
        valid_frequencies[
            dominant_index
        ]
    )

    return result


def to_boolean_series(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
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


class StageFeatureBuilder:
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
        corrected_edf_start: Any = None,
    ) -> pd.DataFrame:
        # stages_aligned.csv 已經包含相對 EDF 秒數，
        # 因此這裡不再使用日期直接切割。
        del corrected_edf_start

        edf_path = Path(edf_file)
        stages_path = Path(stages_file)

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

        required_stage_columns = {
            "epoch_index",
            "original_epoch",
            "start_time",
            "end_time",
            "stage",
            "edf_start_seconds",
            "edf_end_seconds",
            "usable_for_stage_training",
        }

        missing_stage_columns = (
            required_stage_columns
            - set(stages.columns)
        )

        if missing_stage_columns:
            raise RuntimeError(
                "stages_aligned.csv 缺少欄位："
                f"{sorted(missing_stage_columns)}"
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

        # 優先使用 pyEDFlib。
        # 若 EDF header 不合規，則改用 MNE 整晚 preload。
        try:
            source = (
                self._load_with_pyedflib(
                    edf_path
                )
            )

            print(
                "EDF backend：pyEDFlib"
            )

        except Exception as exc:
            print(
                "pyEDFlib 無法讀取，"
                "改用 MNE 指定通道讀取："
                f"{exc}"
            )

            source = self._load_with_mne(
                edf_path
            )

            print(
                "EDF backend：MNE 指定通道讀取"
            )

        try:
            mapped_channels = (
                self.channel_mapper
                .map_channels(
                    source[
                        "signal_labels"
                    ]
                )
            )

            missing_channels = (
                self.channel_mapper
                .validate_required(
                    mapped_channels,
                    STAGE_CHANNELS,
                )
            )

            if missing_channels:
                raise RuntimeError(
                    f"{patient_id} 缺少 Channel："
                    f"{missing_channels}"
                )

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
                    "edf_backend": (
                        source["backend"]
                    ),
                }

                epoch_valid = True

                for standard_name in (
                    STAGE_CHANNELS
                ):
                    source_channel = (
                        mapped_channels[
                            standard_name
                        ]
                    )

                    if source_channel is None:
                        epoch_valid = False
                        continue

                    sampling_rate = float(
                        source[
                            "sampling_rates"
                        ][source_channel]
                    )

                    expected_samples = int(
                        round(
                            30.0
                            * sampling_rate
                        )
                    )

                    if "read_segment" in source:
                        signal_segment = source["read_segment"](
                            source_channel,
                            start_seconds,
                            end_seconds,
                        )
                    else:
                        full_signal = source["signals"][source_channel]
                        start_sample = max(int(round(start_seconds * sampling_rate)), 0)
                        end_sample = min(int(round(end_seconds * sampling_rate)), len(full_signal))
                        signal_segment = (
                            np.empty(0, dtype=np.float64)
                            if end_sample <= start_sample
                            else full_signal[start_sample:end_sample]
                        )

                    actual_samples = len(
                        signal_segment
                    )

                    coverage = (
                        actual_samples
                        / max(
                            expected_samples,
                            1,
                        )
                    )

                    feature_row[
                        f"{standard_name}_source"
                    ] = source_channel

                    feature_row[
                        f"{standard_name}_sampling_rate"
                    ] = sampling_rate

                    feature_row[
                        f"{standard_name}_samples"
                    ] = actual_samples

                    feature_row[
                        f"{standard_name}_coverage"
                    ] = coverage

                    if coverage < 0.99:
                        epoch_valid = False

                    feature_row.update(
                        calculate_basic_features(
                            values=signal_segment,
                            prefix=standard_name,
                        )
                    )

                    if (
                        standard_name
                        in EEG_CHANNELS
                    ):
                        feature_row.update(
                            calculate_spectral_features(
                                values=(
                                    signal_segment
                                ),
                                sampling_rate=(
                                    sampling_rate
                                ),
                                prefix=(
                                    standard_name
                                ),
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
            close_callback = source.get(
                "close"
            )

            if callable(close_callback):
                close_callback()

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
            signal_labels = [
                str(label).strip()
                for label in (
                    reader.getSignalLabels()
                )
            ]

            signals: dict[
                str,
                np.ndarray,
            ] = {}

            sampling_rates: dict[
                str,
                float,
            ] = {}

            for index, label in enumerate(
                signal_labels
            ):
                signals[label] = np.asarray(
                    reader.readSignal(
                        index
                    ),
                    dtype=np.float64,
                )

                sampling_rates[label] = float(
                    reader.getSampleFrequency(
                        index
                    )
                )

            return {
                "backend": "pyedflib",
                "signal_labels": (
                    signal_labels
                ),
                "signals": signals,
                "sampling_rates": (
                    sampling_rates
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
        # Do not preload the whole PSG. A compressed EDF can expand to several
        # GB when every channel is decoded, which exceeds Community Cloud's
        # memory limit. Read only the mapped channels required by this model.
        raw = mne.io.read_raw_edf(
            edf_path,
            preload=False,
            verbose="ERROR",
        )

        all_signal_labels = [
            str(label).strip()
            for label in raw.ch_names
        ]

        mapped_channels = self.channel_mapper.map_channels(all_signal_labels)
        missing_channels = self.channel_mapper.validate_required(
            mapped_channels,
            STAGE_CHANNELS,
        )
        if missing_channels:
            raw.close()
            raise RuntimeError(f"缺少 Stage Channel：{missing_channels}")

        signal_labels = list(dict.fromkeys(
            mapped_channels[channel]
            for channel in STAGE_CHANNELS
            if mapped_channels.get(channel) is not None
        ))

        common_sampling_rate = float(
            raw.info["sfreq"]
        )

        # Read only the requested 30-second epoch.  Keeping a full-night array
        # per channel still exceeds a small cloud worker for high-rate PSG.
        def read_segment(
            label: str,
            start_seconds: float,
            end_seconds: float,
        ) -> np.ndarray:
            start = max(0, int(round(start_seconds * common_sampling_rate)))
            stop = min(raw.n_times, int(round(end_seconds * common_sampling_rate)))
            if stop <= start:
                return np.empty(0, dtype=np.float64)
            return np.asarray(
                raw.get_data(picks=[label], start=start, stop=stop)[0],
                dtype=np.float64,
            )

        sampling_rates = {
            label: common_sampling_rate
            for label in signal_labels
        }

        return {
            "backend": "mne_epoch_segments",
            "signal_labels": (
                signal_labels
            ),
            "sampling_rates": (
                sampling_rates
            ),
            "read_segment": read_segment,
            "close": raw.close,
        }
