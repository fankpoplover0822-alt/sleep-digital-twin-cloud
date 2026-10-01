from pathlib import Path

import pandas as pd


SHEET_TYPE_MAP = {
    "Hypopnea": "HYPOPNEA",
    "A. Obstructive": "OBSTRUCTIVE_APNEA",
    "A. Central": "CENTRAL_APNEA",
    "A. Mixed": "MIXED_APNEA",
    "Arousal": "AROUSAL",
}


class EventReader:
    def __init__(self, event_path: str | Path) -> None:
        self.event_path = Path(event_path)
        self.data: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if not self.event_path.exists():
            raise FileNotFoundError(
                f"Event Grid 檔案不存在：{self.event_path}"
            )

        print(f"讀取 Event Grid：{self.event_path}")

        excel = pd.ExcelFile(
            self.event_path,
            engine="xlrd",
        )

        print(f"Event 工作表：{excel.sheet_names}")

        all_events: list[pd.DataFrame] = []

        for sheet_name in excel.sheet_names:
            raw_df = pd.read_excel(
                self.event_path,
                sheet_name=sheet_name,
                engine="xlrd",
                header=None,
            )

            parsed = self._parse_sheet(
                raw_df,
                sheet_name,
            )

            if not parsed.empty:
                all_events.append(parsed)

        if not all_events:
            self.data = self._empty_dataframe()
            return self.data

        data = pd.concat(
            all_events,
            ignore_index=True,
        )

        data = data.sort_values(
            ["start_time", "event_type"]
        ).reset_index(drop=True)

        data.insert(
            0,
            "event_index",
            range(len(data)),
        )

        self.data = data

        print("Event Grid 解析成功。")
        return self.data

    def _parse_sheet(
        self,
        raw_df: pd.DataFrame,
        sheet_name: str,
    ) -> pd.DataFrame:
        if raw_df.shape[0] <= 2:
            print(
                f"工作表 {sheet_name} 沒有事件資料。"
            )
            return self._empty_dataframe()

        data = raw_df.iloc[2:, :5].copy()

        data.columns = [
            "start_time",
            "end_time",
            "start_epoch",
            "end_epoch",
            "event_label",
        ]

        data["start_time"] = pd.to_datetime(
            data["start_time"],
            errors="coerce",
        )

        data["end_time"] = pd.to_datetime(
            data["end_time"],
            errors="coerce",
        )

        data["start_epoch"] = pd.to_numeric(
            data["start_epoch"],
            errors="coerce",
        )

        data["end_epoch"] = pd.to_numeric(
            data["end_epoch"],
            errors="coerce",
        )

        data["event_label"] = (
            data["event_label"]
            .astype(str)
            .str.strip()
        )

        data = data.dropna(
            subset=["start_time", "end_time"]
        ).copy()

        if data.empty:
            return self._empty_dataframe()

        data["duration_seconds"] = (
            data["end_time"]
            - data["start_time"]
        ).dt.total_seconds()

        data = data[
            (data["duration_seconds"] > 0)
            & (data["duration_seconds"] <= 600)
        ].copy()

        data["event_type"] = (
            SHEET_TYPE_MAP.get(
                sheet_name,
                sheet_name.upper().replace(" ", "_"),
            )
        )

        data["subtype"] = data["event_label"]
        data["source_sheet"] = sheet_name

        data["start_epoch"] = (
            data["start_epoch"]
            .round()
            .astype("Int64")
        )

        data["end_epoch"] = (
            data["end_epoch"]
            .round()
            .astype("Int64")
        )

        return data[
            [
                "event_type",
                "subtype",
                "start_time",
                "end_time",
                "duration_seconds",
                "start_epoch",
                "end_epoch",
                "source_sheet",
            ]
        ]

    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "event_type",
                "subtype",
                "start_time",
                "end_time",
                "duration_seconds",
                "start_epoch",
                "end_epoch",
                "source_sheet",
            ]
        )

    def print_info(self) -> None:
        if self.data is None:
            raise RuntimeError(
                "尚未讀取 Event Grid，請先執行 load()。"
            )

        print("=" * 70)
        print(f"Event 檔案：{self.event_path.name}")
        print(f"有效事件數量：{len(self.data)}")

        if not self.data.empty:
            print("\n事件類型數量：")
            print(
                self.data["event_type"]
                .value_counts()
                .to_string()
            )

        print("=" * 70)