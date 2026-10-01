from pathlib import Path

import pandas as pd


VALID_STAGES = {
    "Wake": "W",
    "W": "W",
    "N1": "N1",
    "N2": "N2",
    "N3": "N3",
    "N4": "N3",
    "REM": "REM",
    "R": "REM",
}


class StageReader:
    def __init__(self, stage_path: str | Path) -> None:
        self.stage_path = Path(stage_path)
        self.raw_data: pd.DataFrame | None = None
        self.data: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if not self.stage_path.exists():
            raise FileNotFoundError(
                f"Stage 檔案不存在：{self.stage_path}"
            )

        print(f"讀取 Stage：{self.stage_path}")

        raw_df = pd.read_excel(
            self.stage_path,
            sheet_name=0,
            engine="xlrd",
            header=None,
        )

        self.raw_data = raw_df

        # 前兩列是：
        # Time Stamp / Epoch Number / Sleep
        # [] / [#] / []
        data = raw_df.iloc[2:, :3].copy()

        data.columns = [
            "start_time",
            "original_epoch",
            "stage_raw",
        ]

        data["start_time"] = pd.to_datetime(
            data["start_time"],
            errors="coerce",
        )

        data["original_epoch"] = pd.to_numeric(
            data["original_epoch"],
            errors="coerce",
        )

        data["stage_raw"] = (
            data["stage_raw"]
            .astype(str)
            .str.strip()
        )

        data["stage"] = data["stage_raw"].map(
            VALID_STAGES
        )

        # 移除無法解析的列
        data = data.dropna(
            subset=[
                "start_time",
                "original_epoch",
                "stage",
            ]
        ).copy()

        data["original_epoch"] = (
            data["original_epoch"].astype(int)
        )

        data["end_time"] = (
            data["start_time"]
            + pd.Timedelta(seconds=30)
        )

        data = data.reset_index(drop=True)

        data.insert(
            0,
            "epoch_index",
            range(len(data)),
        )

        self.data = data[
            [
                "epoch_index",
                "original_epoch",
                "start_time",
                "end_time",
                "stage",
                "stage_raw",
            ]
        ]

        print("Stage 解析成功。")
        return self.data

    def print_info(self) -> None:
        if self.data is None:
            raise RuntimeError(
                "尚未讀取 Stage，請先執行 load()。"
            )

        print("=" * 70)
        print(f"Stage 檔案：{self.stage_path.name}")
        print(f"有效 Epoch 數量：{len(self.data)}")

        if len(self.data) > 0:
            first_time = self.data.iloc[0]["start_time"]
            last_time = self.data.iloc[-1]["end_time"]

            duration_minutes = (
                last_time - first_time
            ).total_seconds() / 60

            print(f"開始時間：{first_time}")
            print(f"結束時間：{last_time}")
            print(
                f"Stage 時間長度："
                f"{duration_minutes:.2f} 分鐘"
            )

        print("\n睡眠階段數量：")
        print(
            self.data["stage"]
            .value_counts()
            .reindex(
                ["W", "N1", "N2", "N3", "REM"],
                fill_value=0,
            )
            .to_string()
        )

        print("\n前 20 個 Epoch：")
        print(
            self.data.head(20).to_string(
                index=False
            )
        )

        print("=" * 70)

    def save_csv(
        self,
        output_path: str | Path,
    ) -> None:
        if self.data is None:
            raise RuntimeError(
                "尚未讀取 Stage，請先執行 load()。"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.data.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Stage CSV 已儲存：{output_path}")