from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import mne


class EDFReader:
    def __init__(
        self,
        edf_path: str | Path,
    ) -> None:
        self.edf_path = Path(edf_path)
        self.raw = None

        # 若 EDF header 日期錯誤，可在程式中設定校正時間
        self._measurement_start_override: datetime | None = None

    def load(self) -> None:
        if not self.edf_path.exists():
            raise FileNotFoundError(
                f"EDF 檔案不存在：{self.edf_path}"
            )

        print(f"讀取 EDF：{self.edf_path}")

        self.raw = mne.io.read_raw_edf(
            self.edf_path,
            preload=False,
            verbose="ERROR",
        )

        print("EDF 讀取成功。")

    @property
    def sampling_rate(self) -> float:
        self._ensure_loaded()

        return float(
            self.raw.info["sfreq"]
        )

    @property
    def duration_seconds(self) -> float:
        self._ensure_loaded()

        return float(
            self.raw.n_times
            / self.sampling_rate
        )

    @property
    def original_measurement_start(
        self,
    ) -> datetime:
        """
        EDF header 中原始記錄開始時間。

        即使之後設定了校正時間，
        此屬性仍回傳 EDF 原始 header 的時間。
        """
        self._ensure_loaded()

        meas_date = self.raw.info.get(
            "meas_date"
        )

        if meas_date is None:
            raise RuntimeError(
                "EDF 中沒有 measurement start time。"
            )

        if hasattr(
            meas_date,
            "to_pydatetime",
        ):
            meas_date = (
                meas_date.to_pydatetime()
            )

        if meas_date.tzinfo is not None:
            meas_date = meas_date.replace(
                tzinfo=None
            )

        return meas_date

    @property
    def measurement_start(
        self,
    ) -> datetime:
        """
        實際用於資料對齊的 EDF 開始時間。

        若有設定 override，優先使用校正後時間；
        否則使用 EDF header 原始時間。
        """
        self._ensure_loaded()

        if (
            self._measurement_start_override
            is not None
        ):
            return (
                self._measurement_start_override
            )

        return self.original_measurement_start

    @property
    def measurement_end(
        self,
    ) -> datetime:
        return (
            self.measurement_start
            + timedelta(
                seconds=self.duration_seconds
            )
        )

    @property
    def measurement_start_was_corrected(
        self,
    ) -> bool:
        return (
            self._measurement_start_override
            is not None
        )

    def set_measurement_start_override(
        self,
        timestamp: datetime,
    ) -> None:
        """
        設定程式內使用的 EDF 開始時間。

        不會修改原始 EDF 檔案。
        """
        if not isinstance(
            timestamp,
            datetime,
        ):
            raise TypeError(
                "timestamp 必須是 datetime。"
            )

        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(
                tzinfo=None
            )

        self._measurement_start_override = (
            timestamp
        )

    def clear_measurement_start_override(
        self,
    ) -> None:
        self._measurement_start_override = None

    def time_to_seconds(
        self,
        timestamp: datetime,
    ) -> float:
        self._ensure_loaded()

        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(
                tzinfo=None
            )

        return float(
            (
                timestamp
                - self.measurement_start
            ).total_seconds()
        )

    def time_to_sample(
        self,
        timestamp: datetime,
    ) -> int:
        seconds = self.time_to_seconds(
            timestamp
        )

        return int(
            round(
                seconds
                * self.sampling_rate
            )
        )

    def is_time_inside_recording(
        self,
        timestamp: datetime,
    ) -> bool:
        seconds = self.time_to_seconds(
            timestamp
        )

        return bool(
            0
            <= seconds
            <= self.duration_seconds
        )

    def print_info(self) -> None:
        self._ensure_loaded()

        print("=" * 70)
        print(
            f"病人檔案："
            f"{self.edf_path.name}"
        )
        print(
            f"Channel 數量："
            f"{len(self.raw.ch_names)}"
        )
        print(
            f"Sampling Rate："
            f"{self.sampling_rate} Hz"
        )
        print(
            f"Duration："
            f"{self.duration_seconds / 60:.2f} 分鐘"
        )
        print(
            f"EDF 原始開始時間："
            f"{self.original_measurement_start}"
        )

        if (
            self.measurement_start_was_corrected
        ):
            print(
                f"EDF 校正開始時間："
                f"{self.measurement_start}"
            )
        else:
            print(
                f"EDF 開始時間："
                f"{self.measurement_start}"
            )

        print(
            f"EDF 結束時間："
            f"{self.measurement_end}"
        )

        print("\nChannels：")

        for index, channel_name in enumerate(
            self.raw.ch_names,
            start=1,
        ):
            print(
                f"{index:02d}. "
                f"{channel_name}"
            )

        print("=" * 70)

    def _ensure_loaded(self) -> None:
        if self.raw is None:
            raise RuntimeError(
                "尚未讀取 EDF，"
                "請先執行 load()。"
            )