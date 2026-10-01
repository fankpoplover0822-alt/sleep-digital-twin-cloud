from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.io_utils.edf_reader import EDFReader


class TimeAligner:
    def __init__(
        self,
        edf_reader: EDFReader,
    ) -> None:
        self.edf_reader = edf_reader

    def align_stages(
        self,
        stages: pd.DataFrame,
    ) -> pd.DataFrame:
        data = stages.copy()

        edf_start = pd.Timestamp(
            self.edf_reader.measurement_start
        )

        duration = float(
            self.edf_reader.duration_seconds
        )

        sampling_rate = float(
            self.edf_reader.sampling_rate
        )

        total_samples = int(
            self.edf_reader.raw.n_times
        )

        data["edf_start_seconds"] = (
            data["start_time"] - edf_start
        ).dt.total_seconds()

        data["edf_end_seconds"] = (
            data["end_time"] - edf_start
        ).dt.total_seconds()

        # 完整位於 EDF 中
        data["fully_inside_edf"] = (
            (data["edf_start_seconds"] >= 0)
            & (data["edf_end_seconds"] <= duration)
        )

        # 與 EDF 至少有部分重疊
        data["overlaps_edf"] = (
            (data["edf_end_seconds"] > 0)
            & (data["edf_start_seconds"] < duration)
        )

        data["partially_inside_edf"] = (
            data["overlaps_edf"]
            & ~data["fully_inside_edf"]
        )

        data["completely_outside_edf"] = (
            ~data["overlaps_edf"]
        )

        # 原始 sample
        data["original_start_sample"] = (
            data["edf_start_seconds"]
            * sampling_rate
        ).round().astype("int64")

        data["original_end_sample"] = (
            data["edf_end_seconds"]
            * sampling_rate
        ).round().astype("int64")

        # 裁切後 sample，避免超出 EDF
        data["start_sample"] = (
            data["original_start_sample"]
            .clip(lower=0, upper=total_samples)
            .astype("int64")
        )

        data["end_sample"] = (
            data["original_end_sample"]
            .clip(lower=0, upper=total_samples)
            .astype("int64")
        )

        data["available_samples"] = (
            data["end_sample"]
            - data["start_sample"]
        ).clip(lower=0)

        data["available_seconds"] = (
            data["available_samples"]
            / sampling_rate
        )

        data["coverage_fraction"] = (
            data["available_seconds"] / 30.0
        ).clip(lower=0.0, upper=1.0)

        # 睡眠分期訓練只使用完整 30 秒 epoch
        data["usable_for_stage_training"] = (
            data["fully_inside_edf"]
            & np.isclose(
                data["available_seconds"],
                30.0,
                atol=0.01,
            )
        )

        # 相容舊程式欄位
        data["inside_edf"] = (
            data["fully_inside_edf"]
        )

        return data

    def align_events(
        self,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        data = events.copy()

        if data.empty:
            for column in [
                "edf_start_seconds",
                "edf_end_seconds",
                "fully_inside_edf",
                "overlaps_edf",
                "partially_inside_edf",
                "completely_outside_edf",
                "original_start_sample",
                "original_end_sample",
                "start_sample",
                "end_sample",
                "available_samples",
                "available_seconds",
                "coverage_fraction",
                "usable_for_event_training",
                "inside_edf",
            ]:
                data[column] = pd.Series(dtype="float64")

            return data

        edf_start = pd.Timestamp(
            self.edf_reader.measurement_start
        )

        duration = float(
            self.edf_reader.duration_seconds
        )

        sampling_rate = float(
            self.edf_reader.sampling_rate
        )

        total_samples = int(
            self.edf_reader.raw.n_times
        )

        data["edf_start_seconds"] = (
            data["start_time"] - edf_start
        ).dt.total_seconds()

        data["edf_end_seconds"] = (
            data["end_time"] - edf_start
        ).dt.total_seconds()

        data["fully_inside_edf"] = (
            (data["edf_start_seconds"] >= 0)
            & (data["edf_end_seconds"] <= duration)
        )

        data["overlaps_edf"] = (
            (data["edf_end_seconds"] > 0)
            & (data["edf_start_seconds"] < duration)
        )

        data["partially_inside_edf"] = (
            data["overlaps_edf"]
            & ~data["fully_inside_edf"]
        )

        data["completely_outside_edf"] = (
            ~data["overlaps_edf"]
        )

        data["original_start_sample"] = (
            data["edf_start_seconds"]
            * sampling_rate
        ).round().astype("int64")

        data["original_end_sample"] = (
            data["edf_end_seconds"]
            * sampling_rate
        ).round().astype("int64")

        data["start_sample"] = (
            data["original_start_sample"]
            .clip(lower=0, upper=total_samples)
            .astype("int64")
        )

        data["end_sample"] = (
            data["original_end_sample"]
            .clip(lower=0, upper=total_samples)
            .astype("int64")
        )

        data["available_samples"] = (
            data["end_sample"]
            - data["start_sample"]
        ).clip(lower=0)

        data["available_seconds"] = (
            data["available_samples"]
            / sampling_rate
        )

        original_duration = (
            data["duration_seconds"]
            .replace(0, np.nan)
        )

        data["coverage_fraction"] = (
            data["available_seconds"]
            / original_duration
        ).clip(lower=0.0, upper=1.0)

        # 事件至少要有 80% 訊號覆蓋才納入事件模型
        data["usable_for_event_training"] = (
            data["overlaps_edf"]
            & (data["coverage_fraction"] >= 0.80)
        )

        # 相容舊程式欄位
        data["inside_edf"] = (
            data["fully_inside_edf"]
        )

        return data

    @staticmethod
    def print_alignment_report(
        stages: pd.DataFrame,
        events: pd.DataFrame,
    ) -> None:
        print("=" * 70)
        print("EDF、Stage、Event 時間軸覆蓋結果")
        print("=" * 70)

        print("\nStage：")
        print(f"總數：{len(stages)}")
        print(
            "完整位於 EDF："
            f"{int(stages['fully_inside_edf'].sum())}"
        )
        print(
            "部分重疊："
            f"{int(stages['partially_inside_edf'].sum())}"
        )
        print(
            "完全位於 EDF 外："
            f"{int(stages['completely_outside_edf'].sum())}"
        )
        print(
            "可用於睡眠分期訓練："
            f"{int(stages['usable_for_stage_training'].sum())}"
        )

        print("\nEvent：")
        print(f"總數：{len(events)}")

        if events.empty:
            print("沒有事件。")
        else:
            print(
                "完整位於 EDF："
                f"{int(events['fully_inside_edf'].sum())}"
            )
            print(
                "部分重疊："
                f"{int(events['partially_inside_edf'].sum())}"
            )
            print(
                "完全位於 EDF 外："
                f"{int(events['completely_outside_edf'].sum())}"
            )
            print(
                "可用於事件模型訓練："
                f"{int(events['usable_for_event_training'].sum())}"
            )

        print("=" * 70)

    @staticmethod
    def save_csv(
        data: pd.DataFrame,
        output_path: str | Path,
    ) -> None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"已儲存對齊資料：{output_path}")