from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.io_utils.edf_reader import EDFReader
from src.preprocessing.alignment_diagnostics import (
    build_alignment_diagnostics,
    save_alignment_diagnostics,
)
from src.preprocessing.time_alignment import TimeAligner
from src.respiratory.event_reader import EventReader
from src.staging.stage_reader import StageReader


@dataclass
class PatientFiles:
    patient_id: str
    folder: Path
    edf_file: Path | None
    stage_file: Path | None
    event_file: Path | None


class DatasetScanner:
    def __init__(
        self,
        raw_root: str | Path,
        processed_root: str | Path,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)

        self.alignment_diagnostics: list[dict] = []

    def discover_patients(
        self,
    ) -> list[PatientFiles]:
        if not self.raw_root.exists():
            raise FileNotFoundError(
                f"raw 資料夾不存在：{self.raw_root}"
            )

        patient_folders = sorted(
            path
            for path in self.raw_root.iterdir()
            if path.is_dir()
        )

        patients: list[PatientFiles] = []

        for folder in patient_folders:
            patients.append(
                PatientFiles(
                    patient_id=folder.name,
                    folder=folder,
                    edf_file=self._find_optional_file(
                        folder=folder,
                        pattern="*_EDF.edf",
                    ),
                    stage_file=self._find_optional_file(
                        folder=folder,
                        pattern="*_stage.xls",
                    ),
                    event_file=self._find_optional_file(
                        folder=folder,
                        pattern="*_Event Grid.xls",
                    ),
                )
            )

        return patients

    @staticmethod
    def _find_optional_file(
        folder: Path,
        pattern: str,
    ) -> Path | None:
        matches = list(folder.glob(pattern))

        if len(matches) == 0:
            return None

        if len(matches) > 1:
            raise RuntimeError(
                f"{folder.name} 找到多個符合 "
                f"{pattern} 的檔案：{matches}"
            )

        return matches[0]

    @staticmethod
    def _parse_recording_start_from_patient_id(
        patient_id: str,
    ) -> datetime | None:
        """
        從資料夾名稱解析記錄開始時間。

        範例：
        20220902T232511 - 05138
        ->
        2022-09-02 23:25:11
        """
        prefix = patient_id.split(" - ")[0].strip()

        try:
            return datetime.strptime(
                prefix,
                "%Y%m%dT%H%M%S",
            )
        except ValueError:
            return None

    @staticmethod
    def _is_invalid_edf_start(
        timestamp: datetime,
    ) -> bool:
        """
        目前部分 EDF 的無效日期固定為 1985-01-01。
        """
        return timestamp.year == 1985

    def process_all(
        self,
    ) -> pd.DataFrame:
        # 避免重複呼叫時累積舊診斷資料
        self.alignment_diagnostics = []

        patients = self.discover_patients()

        print("=" * 80)
        print(
            f"找到病人資料夾數量：{len(patients)}"
        )
        print("=" * 80)

        results: list[dict] = []

        for index, patient in enumerate(
            patients,
            start=1,
        ):
            print()
            print("=" * 80)
            print(
                f"[{index}/{len(patients)}] "
                f"處理患者：{patient.patient_id}"
            )
            print("=" * 80)

            result = self._process_patient(
                patient
            )

            results.append(result)

        manifest = pd.DataFrame(results)

        self.processed_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_path = (
            self.processed_root
            / "manifest.csv"
        )

        manifest.to_csv(
            manifest_path,
            index=False,
            encoding="utf-8-sig",
        )

        diagnostics_df = pd.DataFrame(
            self.alignment_diagnostics
        )

        diagnostics_path = (
            self.processed_root
            / "alignment_diagnostics.csv"
        )

        save_alignment_diagnostics(
            diagnostics=diagnostics_df,
            output_path=diagnostics_path,
        )

        print()
        print("=" * 80)
        print("全部患者處理完成")
        print(f"Manifest：{manifest_path}")
        print("=" * 80)

        return manifest

    def _process_patient(
        self,
        patient: PatientFiles,
    ) -> dict:
        result = {
            "patient_id": patient.patient_id,
            "folder": str(patient.folder),

            "edf_file": (
                str(patient.edf_file)
                if patient.edf_file is not None
                else None
            ),
            "stage_file": (
                str(patient.stage_file)
                if patient.stage_file is not None
                else None
            ),
            "event_file": (
                str(patient.event_file)
                if patient.event_file is not None
                else None
            ),

            "has_edf": (
                patient.edf_file is not None
            ),
            "has_stage": (
                patient.stage_file is not None
            ),
            "has_event": (
                patient.event_file is not None
            ),

            "edf_ok": False,
            "stage_ok": False,
            "event_ok": False,
            "alignment_ok": False,
            "coverage_ok": False,

            # EDF 時間校正
            "original_edf_start": None,
            "corrected_edf_start": None,
            "edf_start_corrected": False,
            "alignment_correction_reason": None,

            # EDF 基本資訊
            "channel_count": None,
            "sampling_rate": None,
            "duration_minutes": None,

            # Stage 統計
            "stage_count": None,
            "stage_inside_edf": None,
            "stage_fully_inside": None,
            "stage_partially_inside": None,
            "stage_completely_outside": None,
            "stage_usable_count": None,
            "stage_coverage_fraction": None,

            "sleep_stage_w": None,
            "sleep_stage_n1": None,
            "sleep_stage_n2": None,
            "sleep_stage_n3": None,
            "sleep_stage_rem": None,

            # Event 統計
            "event_count": None,
            "event_inside_edf": None,
            "event_fully_inside": None,
            "event_partially_inside": None,
            "event_completely_outside": None,
            "event_usable_count": None,
            "event_coverage_fraction": None,

            "hypopnea_count": None,
            "obstructive_apnea_count": None,
            "central_apnea_count": None,
            "mixed_apnea_count": None,
            "arousal_count": None,

            "error": None,
        }

        missing_files: list[str] = []

        if patient.edf_file is None:
            missing_files.append("EDF")

        if patient.stage_file is None:
            missing_files.append("Stage")

        if patient.event_file is None:
            missing_files.append("Event")

        if missing_files:
            result["error"] = (
                "缺少檔案："
                + ", ".join(missing_files)
            )

            print(result["error"])
            return result

        try:
            # ==========================================
            # 1. EDF
            # ==========================================
            edf_reader = EDFReader(
                patient.edf_file
            )

            edf_reader.load()

            original_edf_start = (
                edf_reader
                .original_measurement_start
            )

            correction_reason: str | None = None

            if self._is_invalid_edf_start(
                original_edf_start
            ):
                folder_start = (
                    self
                    ._parse_recording_start_from_patient_id(
                        patient.patient_id
                    )
                )

                if folder_start is None:
                    raise RuntimeError(
                        "EDF 日期無效，且無法從"
                        "患者資料夾名稱解析記錄時間。"
                    )

                edf_reader.set_measurement_start_override(
                    folder_start
                )

                correction_reason = (
                    "EDF header 使用 "
                    "1985-01-01 假日期；"
                    "改用患者資料夾名稱中的 "
                    "YYYYMMDDTHHMMSS 時間"
                )

                print(
                    "校正 EDF 開始時間："
                    f"{original_edf_start} "
                    "→ "
                    f"{folder_start}"
                )

            result["edf_ok"] = True

            result["original_edf_start"] = (
                original_edf_start
            )

            result["corrected_edf_start"] = (
                edf_reader.measurement_start
            )

            result["edf_start_corrected"] = (
                edf_reader
                .measurement_start_was_corrected
            )

            result[
                "alignment_correction_reason"
            ] = correction_reason

            result["channel_count"] = len(
                edf_reader.raw.ch_names
            )

            result["sampling_rate"] = (
                edf_reader.sampling_rate
            )

            result["duration_minutes"] = (
                edf_reader.duration_seconds
                / 60.0
            )

            # ==========================================
            # 2. Stage
            # ==========================================
            stage_reader = StageReader(
                patient.stage_file
            )

            stages = stage_reader.load()

            result["stage_ok"] = True
            result["stage_count"] = len(stages)

            stage_counts = (
                stages["stage"]
                .value_counts()
                .to_dict()
            )

            result["sleep_stage_w"] = int(
                stage_counts.get("W", 0)
            )

            result["sleep_stage_n1"] = int(
                stage_counts.get("N1", 0)
            )

            result["sleep_stage_n2"] = int(
                stage_counts.get("N2", 0)
            )

            result["sleep_stage_n3"] = int(
                stage_counts.get("N3", 0)
            )

            result["sleep_stage_rem"] = int(
                stage_counts.get("REM", 0)
            )

            # ==========================================
            # 3. Event
            # ==========================================
            event_reader = EventReader(
                patient.event_file
            )

            events = event_reader.load()

            result["event_ok"] = True
            result["event_count"] = len(events)

            event_counts = (
                events["event_type"]
                .value_counts()
                .to_dict()
            )

            result["hypopnea_count"] = int(
                event_counts.get(
                    "HYPOPNEA",
                    0,
                )
            )

            result[
                "obstructive_apnea_count"
            ] = int(
                event_counts.get(
                    "OBSTRUCTIVE_APNEA",
                    0,
                )
            )

            result[
                "central_apnea_count"
            ] = int(
                event_counts.get(
                    "CENTRAL_APNEA",
                    0,
                )
            )

            result[
                "mixed_apnea_count"
            ] = int(
                event_counts.get(
                    "MIXED_APNEA",
                    0,
                )
            )

            result["arousal_count"] = int(
                event_counts.get(
                    "AROUSAL",
                    0,
                )
            )

            # ==========================================
            # 4. 時間軸對齊
            # ==========================================
            aligner = TimeAligner(
                edf_reader
            )

            aligned_stages = (
                aligner.align_stages(
                    stages
                )
            )

            aligned_events = (
                aligner.align_events(
                    events
                )
            )

            # ==========================================
            # 5. Stage 覆蓋統計
            # ==========================================
            result["stage_fully_inside"] = int(
                aligned_stages[
                    "fully_inside_edf"
                ].sum()
            )

            result[
                "stage_partially_inside"
            ] = int(
                aligned_stages[
                    "partially_inside_edf"
                ].sum()
            )

            result[
                "stage_completely_outside"
            ] = int(
                aligned_stages[
                    "completely_outside_edf"
                ].sum()
            )

            result["stage_usable_count"] = int(
                aligned_stages[
                    "usable_for_stage_training"
                ].sum()
            )

            # 相容舊欄位
            result["stage_inside_edf"] = (
                result["stage_fully_inside"]
            )

            stage_total = len(
                aligned_stages
            )

            result[
                "stage_coverage_fraction"
            ] = float(
                result["stage_usable_count"]
                / max(stage_total, 1)
            )

            # ==========================================
            # 6. Event 覆蓋統計
            # ==========================================
            if aligned_events.empty:
                result["event_fully_inside"] = 0
                result[
                    "event_partially_inside"
                ] = 0
                result[
                    "event_completely_outside"
                ] = 0
                result["event_usable_count"] = 0

            else:
                result[
                    "event_fully_inside"
                ] = int(
                    aligned_events[
                        "fully_inside_edf"
                    ].sum()
                )

                result[
                    "event_partially_inside"
                ] = int(
                    aligned_events[
                        "partially_inside_edf"
                    ].sum()
                )

                result[
                    "event_completely_outside"
                ] = int(
                    aligned_events[
                        "completely_outside_edf"
                    ].sum()
                )

                result[
                    "event_usable_count"
                ] = int(
                    aligned_events[
                        "usable_for_event_training"
                    ].sum()
                )

            # 相容舊欄位
            result["event_inside_edf"] = (
                result["event_fully_inside"]
            )

            event_total = len(
                aligned_events
            )

            if event_total == 0:
                # 沒有事件標註時，不視為覆蓋失敗
                result[
                    "event_coverage_fraction"
                ] = 1.0
            else:
                result[
                    "event_coverage_fraction"
                ] = float(
                    result["event_usable_count"]
                    / event_total
                )

            # ==========================================
            # 7. 判斷時間軸與資料覆蓋
            # ==========================================
            stage_has_overlap = bool(
                aligned_stages.empty
                or aligned_stages[
                    "overlaps_edf"
                ].any()
            )

            event_has_overlap = bool(
                aligned_events.empty
                or aligned_events[
                    "overlaps_edf"
                ].any()
            )

            # 日期及主要時間軸有重疊，即視為對齊成功
            result["alignment_ok"] = bool(
                stage_has_overlap
                and event_has_overlap
            )

            # Stage 至少 90% 可用；
            # Event 至少 80% 可用
            result["coverage_ok"] = bool(
                result[
                    "stage_coverage_fraction"
                ] >= 0.90
                and result[
                    "event_coverage_fraction"
                ] >= 0.80
            )

            # ==========================================
            # 8. Diagnostics
            # ==========================================
            diagnostic = (
                build_alignment_diagnostics(
                    patient_id=(
                        patient.patient_id
                    ),
                    edf_start=(
                        edf_reader
                        .measurement_start
                    ),
                    edf_end=(
                        edf_reader
                        .measurement_end
                    ),
                    stages=aligned_stages,
                    events=aligned_events,
                )
            )

            diagnostic[
                "original_edf_start"
            ] = original_edf_start

            diagnostic[
                "corrected_edf_start"
            ] = (
                edf_reader.measurement_start
            )

            diagnostic[
                "edf_start_corrected"
            ] = (
                edf_reader
                .measurement_start_was_corrected
            )

            diagnostic[
                "alignment_correction_reason"
            ] = correction_reason

            diagnostic[
                "stage_usable_count"
            ] = result[
                "stage_usable_count"
            ]

            diagnostic[
                "stage_coverage_fraction"
            ] = result[
                "stage_coverage_fraction"
            ]

            diagnostic[
                "event_usable_count"
            ] = result[
                "event_usable_count"
            ]

            diagnostic[
                "event_coverage_fraction"
            ] = result[
                "event_coverage_fraction"
            ]

            diagnostic["alignment_ok"] = (
                result["alignment_ok"]
            )

            diagnostic["coverage_ok"] = (
                result["coverage_ok"]
            )

            self.alignment_diagnostics.append(
                diagnostic
            )

            # ==========================================
            # 9. 儲存患者處理結果
            # ==========================================
            patient_output_folder = (
                self.processed_root
                / patient.patient_id
            )

            patient_output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            aligned_stages.to_csv(
                patient_output_folder
                / "stages_aligned.csv",
                index=False,
                encoding="utf-8-sig",
            )

            aligned_events.to_csv(
                patient_output_folder
                / "events_aligned.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print(
                f"完成："
                f"Stage={len(stages)}, "
                f"Stage usable="
                f"{result['stage_usable_count']}, "
                f"Event={len(events)}, "
                f"Event usable="
                f"{result['event_usable_count']}, "
                f"Alignment="
                f"{result['alignment_ok']}, "
                f"Coverage="
                f"{result['coverage_ok']}"
            )

        except Exception as exc:
            result["error"] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f"失敗：{result['error']}"
            )

        return result