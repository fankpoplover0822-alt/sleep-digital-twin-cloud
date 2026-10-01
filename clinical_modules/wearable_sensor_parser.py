from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from clinical_modules.base_parser import BaseClinicalParser, ParsedClinicalData
from clinical_modules.generic_parser import GenericClinicalParser


class WearableSensorParser(BaseClinicalParser):
    """Parse watch/sensor summaries and timestamped monitoring exports."""

    module_id = "DEVICE_DATA"
    display_name = "穿戴式手錶／感測器同步資料"
    supported_extensions = {".csv", ".txt", ".xls", ".xlsx", ".pdf", ".docx"}

    COLUMN_ALIASES = {
        "timestamp": ("timestamp", "time", "datetime", "recorded_at", "時間", "時間戳"),
        "spo2": ("spo2", "oxygen_saturation", "blood_oxygen", "血氧"),
        "heart_rate": ("heart_rate", "heartrate", "pulse", "hr", "心率", "脈搏"),
        "sleep_stage": ("sleep_stage", "stage", "睡眠階段"),
        "position": ("position", "sleep_position", "posture", "睡姿", "姿勢"),
        "apnea_event": ("apnea_event", "apnea", "respiratory_event", "呼吸中止", "呼吸事件"),
        "respiratory_rate": ("respiratory_rate", "resp_rate", "rr", "呼吸速率"),
        "device_name": ("device_name", "device", "watch", "sensor", "裝置名稱"),
    }

    def parse(self, uploaded_files: list[BinaryIO]) -> ParsedClinicalData:
        self.validate_files(uploaded_files)
        merged: dict[str, Any] = {}
        warnings: list[str] = []
        sources: list[str] = []
        total_rows = 0

        for uploaded_file in uploaded_files:
            filename = str(getattr(uploaded_file, "name", "sensor_export"))
            sources.append(filename)
            extension = Path(filename).suffix.lower()
            try:
                if extension == ".csv":
                    frame = pd.read_csv(BytesIO(self._bytes(uploaded_file)))
                elif extension == ".txt":
                    frame = pd.read_csv(BytesIO(self._bytes(uploaded_file)), sep=None, engine="python")
                elif extension in {".xls", ".xlsx"}:
                    frame = pd.read_excel(BytesIO(self._bytes(uploaded_file)))
                else:
                    generic_file = BytesIO(self._bytes(uploaded_file))
                    generic_file.name = filename
                    generic_file.size = len(generic_file.getvalue())
                    result = GenericClinicalParser("DEVICE_DATA").parse([generic_file])
                    merged.update(result.data)
                    warnings.extend(result.warnings)
                    continue
            except Exception as exc:
                warnings.append(f"`{filename}` 感測資料讀取失敗：{type(exc).__name__}: {exc}")
                continue

            if frame.empty:
                warnings.append(f"`{filename}` 沒有感測資料列。")
                continue
            total_rows += len(frame)
            normalized = {self._normal(column): column for column in frame.columns}

            def column_for(name: str):
                for alias in self.COLUMN_ALIASES[name]:
                    column = normalized.get(self._normal(alias))
                    if column is not None:
                        return column
                return None

            time_col = column_for("timestamp")
            spo2_col = column_for("spo2")
            hr_col = column_for("heart_rate")
            stage_col = column_for("sleep_stage")
            position_col = column_for("position")
            apnea_col = column_for("apnea_event")
            rr_col = column_for("respiratory_rate")
            device_col = column_for("device_name")

            timestamps = pd.to_datetime(frame[time_col], errors="coerce") if time_col else pd.Series(dtype="datetime64[ns]")
            valid_times = timestamps.dropna()
            if not valid_times.empty:
                start, end = valid_times.min(), valid_times.max()
                merged["monitoring_period"] = f"{start.isoformat()} ~ {end.isoformat()}"
                duration_hours = max((end - start).total_seconds() / 3600.0, 0.0)
            else:
                duration_hours = 0.0

            if device_col:
                devices = frame[device_col].dropna().astype(str)
                if not devices.empty:
                    merged["device_name"] = devices.iloc[-1]
            merged.setdefault("device_name", Path(filename).stem)

            if spo2_col:
                values = pd.to_numeric(frame[spo2_col], errors="coerce").dropna()
                values = values[(values >= 20) & (values <= 100)]
                if not values.empty:
                    merged["minimum_spo2"] = round(float(values.min()), 2)
                    merged["average_spo2"] = round(float(values.mean()), 2)
                    merged["time_below_90_percent"] = round(float((values < 90).mean() * 100), 2)

            if hr_col:
                values = pd.to_numeric(frame[hr_col], errors="coerce").dropna()
                values = values[(values >= 20) & (values <= 250)]
                if not values.empty:
                    merged["average_heart_rate"] = round(float(values.mean()), 2)
                    merged["minimum_heart_rate"] = round(float(values.min()), 2)
                    merged["maximum_heart_rate"] = round(float(values.max()), 2)

            if rr_col:
                values = pd.to_numeric(frame[rr_col], errors="coerce").dropna()
                values = values[(values >= 2) & (values <= 80)]
                if not values.empty:
                    merged["average_respiratory_rate"] = round(float(values.mean()), 2)

            if stage_col:
                stages = frame[stage_col].fillna("").astype(str).str.strip().str.lower()
                sleep_mask = ~stages.isin({"", "w", "wake", "awake", "清醒"})
                if len(stages):
                    if duration_hours > 0 and len(valid_times) > 1:
                        merged["sleep_duration_hours"] = round(duration_hours * float(sleep_mask.mean()), 2)
                    else:
                        merged["sleep_duration_hours"] = round(float(sleep_mask.sum()) * 30 / 3600, 2)
                    merged["rem_sleep_percent"] = round(float(stages.str.contains("rem").mean() * 100), 2)
                    merged["deep_sleep_percent"] = round(float(stages.isin({"n3", "deep", "深睡"}).mean() * 100), 2)

            if position_col:
                positions = frame[position_col].fillna("").astype(str).str.strip().str.lower()
                supine = positions.isin({"supine", "back", "仰睡", "平蹺"})
                merged["supine_sleep_percent"] = round(float(supine.mean() * 100), 2)

            if apnea_col:
                events = frame[apnea_col].map(self._truthy)
                event_count = int(events.sum())
                merged["apnea_event_count"] = event_count
                denominator = duration_hours or (len(frame) * 30 / 3600)
                if denominator > 0:
                    merged["estimated_ahi"] = round(event_count / denominator, 2)

        merged["sensor_data_points"] = total_rows
        merged["sync_received_at"] = datetime.now().isoformat(timespec="seconds")
        merged["data_source_type"] = "wearable_sensor_sync"
        if total_rows == 0 and not merged:
            warnings.append("未辨識到可同步的穿戴式感測資料。")

        return ParsedClinicalData(
            module_id=self.module_id,
            data=merged,
            source_files=sources,
            warnings=warnings,
            metadata={
                "parser": self.__class__.__name__,
                "parser_version": "1.0.0",
                "sensor_rows": total_rows,
                "sync_mode": "file_export_as_new_monitoring_update",
            },
        )

    @staticmethod
    def _bytes(uploaded_file: BinaryIO) -> bytes:
        if hasattr(uploaded_file, "getvalue"):
            return uploaded_file.getvalue()
        position = uploaded_file.tell() if hasattr(uploaded_file, "tell") else None
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        data = uploaded_file.read()
        if position is not None and hasattr(uploaded_file, "seek"):
            uploaded_file.seek(position)
        return data

    @staticmethod
    def _normal(value: Any) -> str:
        return "".join(character for character in str(value).strip().lower() if character.isalnum() or "\u4e00" <= character <= "\u9fff")

    @staticmethod
    def _truthy(value: Any) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y", "apnea", "hypopnea", "event", "是", "有", "呼吸中止"}
