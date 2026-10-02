from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.importers.new_patient_importer import NewPatientImporter
from src.io_utils.edf_reader import EDFReader
from src.preprocessing.time_alignment import TimeAligner
from src.respiratory.event_reader import EventReader
from src.staging.stage_reader import StageReader


PROJECT_ROOT = Path(__file__).resolve().parent

INCOMING_ROOT = PROJECT_ROOT / "data" / "incoming"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

DEMOGRAPHICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "demographics"
    / "patients.xlsx"
)

STAGE_EPOCH_SECONDS = 30.0
MIN_ACCEPTABLE_STAGE_COVERAGE = 0.95
TIME_TOLERANCE_SECONDS = 1.0


def write_csv_resilient(
    data: pd.DataFrame,
    destination: Path,
    *,
    attempts: int = 20,
    delay_seconds: float = 0.25,
) -> None:
    """Write a CSV atomically and tolerate short-lived Windows file locks."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{datetime.now().strftime('%H%M%S%f')}.part"
    )
    data.to_csv(temporary, index=False, encoding="utf-8-sig")
    try:
        for attempt in range(1, attempts + 1):
            try:
                os.replace(temporary, destination)
                return
            except PermissionError as exc:
                if attempt >= attempts:
                    raise PermissionError(
                        "無法更新處理後資料，檔案持續被其他程式占用："
                        f"{destination}。請關閉正在開啟此CSV的Excel或其他程式後重試。"
                    ) from exc
                time.sleep(delay_seconds)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass


def parse_datetime_from_patient_id(
    patient_id: str,
) -> datetime | None:
    """
    從患者 ID 開頭解析 YYYYMMDDTHHMMSS。

    例如：
    20201014T221256 - d25c6
    會解析成：
    2020-10-14 22:12:56
    """
    matched = re.match(
        r"^\s*(\d{8}T\d{6})",
        str(patient_id),
    )

    if matched is None:
        return None

    try:
        return datetime.strptime(
            matched.group(1),
            "%Y%m%dT%H%M%S",
        )
    except ValueError:
        return None


def should_correct_edf_start(
    original_edf_start: datetime,
    folder_timestamp: datetime | None,
) -> tuple[bool, str]:
    """
    判斷 EDF header 的日期是否需要校正。

    規則：
    1. EDF 年份小於 1990，視為假日期。
    2. EDF 日期與患者資料夾日期相差超過 7 天，
       視為 header 日期不可信。

    注意：
    這裡只校正明顯錯誤的 EDF 日期，不會為了提高覆蓋率，
    任意平移 Stage 或 Event 的時間。
    """
    if folder_timestamp is None:
        return (
            False,
            "患者 ID 無法解析日期，保留 EDF header 時間",
        )

    if original_edf_start.year < 1990:
        return (
            True,
            (
                "EDF header 使用早於 1990 年的假日期；"
                "改用患者資料夾名稱中的 YYYYMMDDTHHMMSS 時間"
            ),
        )

    difference_days = abs(
        (
            original_edf_start
            - folder_timestamp
        ).total_seconds()
    ) / 86400.0

    if difference_days > 7.0:
        return (
            True,
            (
                "EDF header 日期與患者資料夾日期相差超過 7 天；"
                "改用患者資料夾名稱中的 YYYYMMDDTHHMMSS 時間"
            ),
        )

    return (
        False,
        "EDF header 時間與患者資料夾日期一致",
    )


def correct_companion_timestamps_when_header_is_known_wrong(
    data: pd.DataFrame,
    *,
    original_edf_start: datetime,
    corrected_edf_start: datetime,
    label: str,
) -> tuple[pd.DataFrame, str | None]:
    """Apply the same audited EDF-header correction to matching companion files.

    This is deliberately narrow: timestamps are changed only when the companion
    starts on the same calendar date as the EDF's known-wrong original header.
    It prevents a demonstrated recorder-date error from making an otherwise
    complete PSG unusable, without guessing an arbitrary offset.
    """
    if data.empty or "start_time" not in data:
        return data, None
    start = pd.to_datetime(data["start_time"], errors="coerce").min()
    if pd.isna(start):
        return data, None
    if abs((pd.Timestamp(start).normalize() - pd.Timestamp(original_edf_start).normalize()).days) > 1:
        return data, None
    shift = pd.Timestamp(corrected_edf_start) - pd.Timestamp(original_edf_start)
    corrected = data.copy()
    for column in ("start_time", "end_time"):
        if column in corrected:
            corrected[column] = pd.to_datetime(corrected[column], errors="coerce") + shift
    return corrected, (
        f"{label}日期與已判定錯誤的EDF原始header日期一致；"
        f"已套用相同校正量 {shift}。"
    )


def to_json_safe(
    value: Any,
) -> Any:
    """
    將 pandas、numpy 或 datetime 值轉成可寫入 JSON 的格式。
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        (
            datetime,
            pd.Timestamp,
        ),
    ):
        return value.isoformat(
            sep=" "
        )

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    return value


def save_metadata_json(
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    safe_metadata = {
        key: to_json_safe(value)
        for key, value in metadata.items()
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            safe_metadata,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def build_empty_events() -> pd.DataFrame:
    """
    當新患者沒有 Event Grid 時，建立空事件表。
    """
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


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    missing_columns = sorted(
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"{dataframe_name} 缺少必要欄位："
            f"{missing_columns}"
        )


def normalize_datetime_columns(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...],
    dataframe_name: str,
) -> pd.DataFrame:
    """
    將指定欄位安全轉成 datetime。
    """
    result = dataframe.copy()

    for column in columns:
        if column not in result.columns:
            continue

        result[column] = pd.to_datetime(
            result[column],
            errors="coerce",
        )

        invalid_count = int(
            result[column].isna().sum()
        )

        if invalid_count > 0:
            raise RuntimeError(
                f"{dataframe_name}.{column} 有 "
                f"{invalid_count} 筆無法解析的時間。"
            )

    return result


def build_alignment_diagnostics(
    stages: pd.DataFrame,
    events: pd.DataFrame,
    edf_start: datetime,
    edf_end: datetime,
    edf_duration_seconds: float,
) -> dict[str, Any]:
    """
    建立時間對齊診斷。

    重要原則：
    若 Stage 的總時長遠大於 EDF，單靠平移時間軸不可能讓全部
    Stage 進入 EDF。這種情況應標記為 EDF_RECORDING_SHORTER，
    而不是自動偽造 offset。
    """
    stage_start = pd.Timestamp(
        stages["start_time"].min()
    )

    stage_end = pd.Timestamp(
        stages["end_time"].max()
    )

    stage_span_seconds = float(
        (
            stage_end
            - stage_start
        ).total_seconds()
    )

    stage_start_offset_seconds = float(
        (
            stage_start
            - pd.Timestamp(edf_start)
        ).total_seconds()
    )

    stage_end_offset_seconds = float(
        (
            stage_end
            - pd.Timestamp(edf_end)
        ).total_seconds()
    )

    theoretical_max_stage_epochs = max(
        0,
        int(
            math.floor(
                (
                    float(edf_duration_seconds)
                    + TIME_TOLERANCE_SECONDS
                )
                / STAGE_EPOCH_SECONDS
            )
        ),
    )

    theoretical_max_stage_coverage = min(
        1.0,
        (
            theoretical_max_stage_epochs
            / len(stages)
        )
        if len(stages) > 0
        else 0.0,
    )

    overlap_start = max(
        stage_start,
        pd.Timestamp(edf_start),
    )

    overlap_end = min(
        stage_end,
        pd.Timestamp(edf_end),
    )

    overlap_seconds = max(
        0.0,
        float(
            (
                overlap_end
                - overlap_start
            ).total_seconds()
        ),
    )

    stage_duration_ratio = (
        stage_span_seconds
        / float(edf_duration_seconds)
        if edf_duration_seconds > 0
        else None
    )

    if (
        edf_duration_seconds > 0
        and stage_span_seconds
        > edf_duration_seconds
        + STAGE_EPOCH_SECONDS
    ):
        status = (
            "EDF_RECORDING_SHORTER_THAN_STAGE"
        )
        explanation = (
            "Stage 的時間範圍長於 EDF 訊號。"
            "單靠時間平移不可能讓所有 Stage 進入 EDF；"
            "目前只能分析 EDF 實際包含的重疊區段。"
        )
    elif abs(
        stage_start_offset_seconds
    ) > STAGE_EPOCH_SECONDS:
        status = (
            "POSSIBLE_START_TIME_OFFSET"
        )
        explanation = (
            "Stage 與 EDF 的開始時間存在明顯偏移，"
            "但在沒有可靠同步標記前，不應自動平移。"
        )
    else:
        status = "TIME_RANGE_COMPATIBLE"
        explanation = (
            "Stage 與 EDF 的時間長度及開始時間大致相容。"
        )

    event_start = None
    event_end = None
    event_span_seconds = None

    if not events.empty:
        event_start = pd.Timestamp(
            events["start_time"].min()
        )

        event_end = pd.Timestamp(
            events["end_time"].max()
        )

        event_span_seconds = float(
            (
                event_end
                - event_start
            ).total_seconds()
        )

    return {
        "alignment_status": status,
        "alignment_explanation": explanation,
        "edf_start": edf_start,
        "edf_end": edf_end,
        "edf_duration_seconds": float(
            edf_duration_seconds
        ),
        "stage_start": stage_start,
        "stage_end": stage_end,
        "stage_span_seconds": stage_span_seconds,
        "stage_start_offset_seconds": (
            stage_start_offset_seconds
        ),
        "stage_end_offset_seconds": (
            stage_end_offset_seconds
        ),
        "stage_duration_to_edf_ratio": (
            stage_duration_ratio
        ),
        "edf_stage_overlap_seconds": (
            overlap_seconds
        ),
        "theoretical_max_stage_epochs": (
            theoretical_max_stage_epochs
        ),
        "theoretical_max_stage_coverage": (
            theoretical_max_stage_coverage
        ),
        "event_start": event_start,
        "event_end": event_end,
        "event_span_seconds": (
            event_span_seconds
        ),
    }


def print_alignment_diagnostics(
    diagnostics: dict[str, Any],
) -> None:
    print()
    print("-" * 80)
    print("時間範圍診斷")
    print("-" * 80)

    print(
        f"診斷狀態："
        f"{diagnostics['alignment_status']}"
    )

    print(
        f"EDF 時長："
        f"{diagnostics['edf_duration_seconds'] / 60:.2f} 分鐘"
    )

    print(
        f"Stage 時間範圍："
        f"{diagnostics['stage_span_seconds'] / 60:.2f} 分鐘"
    )

    print(
        f"Stage 相對 EDF 起點："
        f"{diagnostics['stage_start_offset_seconds'] / 60:.2f} 分鐘"
    )

    print(
        f"Stage 與 EDF 重疊："
        f"{diagnostics['edf_stage_overlap_seconds'] / 60:.2f} 分鐘"
    )

    print(
        "依 EDF 長度可容納的理論最大 Stage："
        f"{diagnostics['theoretical_max_stage_epochs']} 個 "
        f"({diagnostics['theoretical_max_stage_coverage']:.2%})"
    )

    print(
        f"說明："
        f"{diagnostics['alignment_explanation']}"
    )

def process_patient(
    patient_id: str,
    allow_partial_edf: bool = False,
) -> None:
    patient_folder = (
        INCOMING_ROOT
        / patient_id
    )

    print("=" * 80)
    print("新患者前處理")
    print("=" * 80)
    print(f"Patient ID：{patient_id}")
    print(f"輸入資料夾：{patient_folder}")

    # ============================================================
    # 1. 尋找 EDF、Stage、Event Grid 與基本資料
    # ============================================================
    importer = NewPatientImporter(
        patient_folder=patient_folder,
        demographics_file=(
            DEMOGRAPHICS_FILE
        ),
    )

    patient_files = importer.inspect()

    importer.print_summary(
        patient_files
    )

    output_folder = (
        PROCESSED_ROOT
        / patient_id
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # 2. 讀取 EDF
    # ============================================================
    print()
    print("=" * 80)
    print("步驟 1：讀取 EDF")
    print("=" * 80)

    edf_reader = EDFReader(
        patient_files.edf_file
    )

    edf_reader.load()

    original_edf_start = (
        edf_reader
        .original_measurement_start
    )

    folder_timestamp = (
        parse_datetime_from_patient_id(
            patient_id
        )
    )

    (
        correction_needed,
        correction_reason,
    ) = should_correct_edf_start(
        original_edf_start=(
            original_edf_start
        ),
        folder_timestamp=(
            folder_timestamp
        ),
    )

    if correction_needed:
        if folder_timestamp is None:
            raise RuntimeError(
                "判定需要校正 EDF 時間，"
                "但無法解析患者資料夾日期。"
            )

        edf_reader.set_measurement_start_override(
            folder_timestamp
        )

        print("EDF 開始時間已校正：")
        print(
            f"原始時間："
            f"{original_edf_start}"
        )
        print(
            f"校正時間："
            f"{edf_reader.measurement_start}"
        )
        print(
            f"原因：{correction_reason}"
        )
    else:
        print("EDF 開始時間不需校正。")
        print(
            f"EDF 開始時間："
            f"{edf_reader.measurement_start}"
        )
        print(
            f"判斷原因："
            f"{correction_reason}"
        )

    print(
        f"EDF 結束時間："
        f"{edf_reader.measurement_end}"
    )
    print(
        f"Sampling Rate："
        f"{edf_reader.sampling_rate} Hz"
    )
    print(
        f"Duration："
        f"{edf_reader.duration_seconds / 60:.2f} 分鐘"
    )

    # ============================================================
    # 3. 讀取 Stage
    # ============================================================
    print()
    print("=" * 80)
    print("步驟 2：讀取 Stage")
    print("=" * 80)

    stage_reader = StageReader(
        patient_files.stage_file
    )

    stages = stage_reader.load()

    if stages.empty:
        raise RuntimeError(
            "Stage 解析結果為空。"
        )

    require_columns(
        stages,
        {
            "start_time",
            "end_time",
        },
        "Stage",
    )

    stages = normalize_datetime_columns(
        stages,
        (
            "start_time",
            "end_time",
        ),
        "Stage",
    )

    companion_correction_notes: list[str] = []
    if correction_needed:
        stages, stage_correction_note = correct_companion_timestamps_when_header_is_known_wrong(
            stages,
            original_edf_start=original_edf_start,
            corrected_edf_start=edf_reader.measurement_start,
            label="Stage",
        )
        if stage_correction_note:
            companion_correction_notes.append(stage_correction_note)
            print(stage_correction_note)

    stages_csv = (
        output_folder
        / "stages.csv"
    )

    write_csv_resilient(stages, stages_csv)

    print(
        f"Stage CSV 已儲存："
        f"{stages_csv}"
    )
    print(
        f"有效 Stage Epoch："
        f"{len(stages)}"
    )
    print(
        f"Stage 開始時間："
        f"{stages['start_time'].min()}"
    )
    print(
        f"Stage 結束時間："
        f"{stages['end_time'].max()}"
    )

    # ============================================================
    # 4. 讀取 Event Grid
    # ============================================================
    print()
    print("=" * 80)
    print("步驟 3：讀取 Event Grid")
    print("=" * 80)

    if patient_files.event_file is not None:
        event_reader = EventReader(
            patient_files.event_file
        )

        events = event_reader.load()

        if not events.empty:
            require_columns(
                events,
                {
                    "start_time",
                    "end_time",
                },
                "Event",
            )

            events = normalize_datetime_columns(
                events,
                (
                    "start_time",
                    "end_time",
                ),
                "Event",
            )
            if correction_needed:
                events, event_correction_note = correct_companion_timestamps_when_header_is_known_wrong(
                    events,
                    original_edf_start=original_edf_start,
                    corrected_edf_start=edf_reader.measurement_start,
                    label="Event Grid",
                )
                if event_correction_note:
                    companion_correction_notes.append(event_correction_note)
                    print(event_correction_note)

        print(
            f"有效 Event："
            f"{len(events)}"
        )
    else:
        events = build_empty_events()

        print("未提供 Event Grid。")
        print(
            "仍可完成推論，"
            "但不能以真實事件驗證模型。"
        )

    events_csv = (
        output_folder
        / "events.csv"
    )

    write_csv_resilient(events, events_csv)

    print(
        f"Event CSV："
        f"{events_csv}"
    )

    # ============================================================
    # 5. 時間範圍診斷
    # ============================================================
    diagnostics = build_alignment_diagnostics(
        stages=stages,
        events=events,
        edf_start=edf_reader.measurement_start,
        edf_end=edf_reader.measurement_end,
        edf_duration_seconds=(
            edf_reader.duration_seconds
        ),
    )

    print_alignment_diagnostics(
        diagnostics
    )

    # ============================================================
    # EDF 完整性檢查
    # ============================================================

    theoretical_coverage = float(
        diagnostics["theoretical_max_stage_coverage"]
    )

    if (
        diagnostics["alignment_status"]
        == "EDF_RECORDING_SHORTER_THAN_STAGE"
        and theoretical_coverage < 0.90
        and not allow_partial_edf
    ):
        raise RuntimeError(
            "\n".join(
                [
                    "",
                    "=" * 80,
                    "停止分析：EDF 記錄不完整",
                    "=" * 80,
                    f"EDF 時長：{diagnostics['edf_duration_seconds']/60:.2f} 分鐘",
                    f"Stage 時間範圍：{diagnostics['stage_span_seconds']/60:.2f} 分鐘",
                    (
                        "理論最大 Stage："
                        f"{diagnostics['theoretical_max_stage_epochs']}"
                        f"/{len(stages)} "
                        f"({theoretical_coverage:.2%})"
                    ),
                    "",
                    "目前 EDF 無法涵蓋完整 PSG。",
                    "請重新確認是否取得完整 EDF。",
                    "",
                    "Pipeline 已停止。",
                    "=" * 80,
                ]
            )
        )


    if (
            diagnostics["alignment_status"]
            == "EDF_RECORDING_SHORTER_THAN_STAGE"
            and theoretical_coverage < 0.90
            and allow_partial_edf
        ):
            print()
            print("=" * 80)
            print("警告：允許分析不完整 EDF")
            print("=" * 80)
            print(
                f"EDF 僅能涵蓋理論上 "
                f"{theoretical_coverage:.2%} 的 Stage。"
            )
            print(
                "本次執行已使用 --allow-partial-edf，"
                "因此 Pipeline 將繼續。"
            )
            print(
                "後續 AHI、血氧、呼吸事件與治療推薦，"
                "只代表 EDF 實際記錄的片段，"
                "不能視為整晚 PSG 結果。"
            )
            print("=" * 80)

    # ============================================================
    # 6. 時間對齊
    # ============================================================
    print()
    print("=" * 80)
    print("步驟 4：時間軸對齊")
    print("=" * 80)

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

    require_columns(
        aligned_stages,
        {
            "fully_inside_edf",
            "usable_for_stage_training",
        },
        "Aligned Stage",
    )

    if not aligned_events.empty:
        require_columns(
            aligned_events,
            {
                "fully_inside_edf",
                "usable_for_event_training",
            },
            "Aligned Event",
        )

    stages_aligned_csv = (
        output_folder
        / "stages_aligned.csv"
    )

    events_aligned_csv = (
        output_folder
        / "events_aligned.csv"
    )

    write_csv_resilient(aligned_stages, stages_aligned_csv)
    print(f"已儲存對齊資料：{stages_aligned_csv}")

    write_csv_resilient(aligned_events, events_aligned_csv)
    print(f"已儲存對齊資料：{events_aligned_csv}")

    stage_total = len(
        aligned_stages
    )

    stage_fully_inside = int(
        aligned_stages[
            "fully_inside_edf"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    stage_usable = int(
        aligned_stages[
            "usable_for_stage_training"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    event_total = len(
        aligned_events
    )

    if aligned_events.empty:
        event_fully_inside = 0
        event_usable = 0
    else:
        event_fully_inside = int(
            aligned_events[
                "fully_inside_edf"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        event_usable = int(
            aligned_events[
                "usable_for_event_training"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    stage_coverage = (
        stage_fully_inside
        / stage_total
        if stage_total > 0
        else 0.0
    )

    event_coverage = (
        event_fully_inside
        / event_total
        if event_total > 0
        else None
    )

    print()
    print("時間對齊結果：")

    print(
        f"Stage fully inside："
        f"{stage_fully_inside}"
        f"/{stage_total}"
        f" ({stage_coverage:.2%})"
    )

    print(
        f"Stage usable："
        f"{stage_usable}"
        f"/{stage_total}"
    )

    if event_total > 0:
        print(
            f"Event fully inside："
            f"{event_fully_inside}"
            f"/{event_total}"
            f" ({event_coverage:.2%})"
        )

        print(
            f"Event usable："
            f"{event_usable}"
            f"/{event_total}"
        )
    else:
        print("Event：沒有事件資料")

    theoretical_coverage = float(
        diagnostics[
            "theoretical_max_stage_coverage"
        ]
    )

    if (
        diagnostics["alignment_status"]
        == "EDF_RECORDING_SHORTER_THAN_STAGE"
    ):
        difference = abs(
            stage_coverage
            - theoretical_coverage
        )

        if difference <= 0.05:
            print()
            print(
                "判讀：實際 Stage 覆蓋率接近 EDF 長度"
                "所允許的理論上限。"
            )
            print(
                "因此低覆蓋率主要是 EDF 訊號較短，"
                "不是單純的時間對齊程式錯誤。"
            )

    # ============================================================
    # 7. 儲存新患者基本資料與處理摘要
    # ============================================================
    print()
    print("=" * 80)
    print("步驟 5：儲存患者資訊")
    print("=" * 80)

    metadata = {
        "patient_id": patient_id,
        "allow_partial_edf": bool(
            allow_partial_edf
        ),
        "partial_edf_analysis": bool(
            diagnostics["alignment_status"]
            == "EDF_RECORDING_SHORTER_THAN_STAGE"
            and theoretical_coverage < 0.90
        ),
        "analysis_scope": (
            "PARTIAL_EDF_SEGMENT"
            if (
                diagnostics["alignment_status"]
                == "EDF_RECORDING_SHORTER_THAN_STAGE"
                and theoretical_coverage < 0.90
            )
            else "FULL_RECORDING"
        ),
        "analysis_scope_warning": (
            (
            "本次結果只代表 EDF 實際記錄片段，"
            "不能視為整晚 PSG 結果。"
            )
            if (
                diagnostics["alignment_status"]
                == "EDF_RECORDING_SHORTER_THAN_STAGE"
                and theoretical_coverage < 0.90
            )
            else None
        ),
        "patient_folder": str(
            patient_files.patient_folder
        ),
        "edf_file": str(
            patient_files.edf_file
        ),
        "stage_file": str(
            patient_files.stage_file
        ),
        "event_file": (
            str(
                patient_files.event_file
            )
            if patient_files.event_file
            is not None
            else None
        ),
        "demographics_file": (
            str(
                patient_files.demographics_file
            )
            if patient_files.demographics_file
            is not None
            else None
        ),
        "demographics_found": (
            patient_files.demographics_found
        ),
        "demographics_sheet": (
            patient_files.demographics_sheet
        ),
        "sex": patient_files.sex,
        "age": patient_files.age,
        "BMI": patient_files.bmi,
        "original_edf_start": (
            original_edf_start
        ),
        "corrected_edf_start": (
            edf_reader.measurement_start
        ),
        "edf_end": (
            edf_reader.measurement_end
        ),
        "edf_start_was_corrected": (
            edf_reader
            .measurement_start_was_corrected
        ),
        "alignment_correction_reason": (
            correction_reason
        ),
        "companion_timestamp_correction": "；".join(companion_correction_notes),
        "sampling_rate": (
            edf_reader.sampling_rate
        ),
        "duration_seconds": (
            edf_reader.duration_seconds
        ),
        "stage_count": stage_total,
        "stage_fully_inside_edf": (
            stage_fully_inside
        ),
        "stage_usable_count": (
            stage_usable
        ),
        "stage_coverage_fraction": (
            stage_coverage
        ),
        "event_count": event_total,
        "event_fully_inside_edf": (
            event_fully_inside
        ),
        "event_usable_count": (
            event_usable
        ),
        "event_coverage_fraction": (
            event_coverage
        ),
        **diagnostics,
    }

    metadata_json = (
        output_folder
        / "patient_metadata.json"
    )

    save_metadata_json(
        metadata_json,
        metadata,
    )

    metadata_csv = (
        output_folder
        / "patient_metadata.csv"
    )

    pd.DataFrame(
        [
            {
                key: to_json_safe(
                    value
                )
                for key, value
                in metadata.items()
            }
        ]
    ).to_csv(
        metadata_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"患者資訊 JSON："
        f"{metadata_json}"
    )
    print(
        f"患者資訊 CSV："
        f"{metadata_csv}"
    )

    # ============================================================
    # 8. 最終檢查
    # ============================================================
    if stage_usable == 0:
        raise RuntimeError(
            "沒有任何可用 Stage Epoch，"
            "不能進行後續特徵建立與推論。"
        )

    if stage_coverage >= MIN_ACCEPTABLE_STAGE_COVERAGE:
        print()
        print("Stage 時間對齊覆蓋合格。")
    elif (
        diagnostics["alignment_status"]
        == "EDF_RECORDING_SHORTER_THAN_STAGE"
    ):
        print()
        print(
            "警告：Stage 覆蓋率低於 95%，"
            "但主要原因是 EDF 記錄長度短於 Stage 時間範圍。"
        )
        print(
            "目前分析只代表 EDF 實際記錄的重疊區段，"
            "不能視為整晚 PSG 結果。"
        )
        print(
            "請確認是否取得了完整 EDF，"
            "或該 EDF 本來就是截短片段。"
        )
    else:
        print()
        print(
            "警告：Stage 完整位於 EDF 內的比例"
            f"只有 {stage_coverage:.2%}。"
        )
        print(
            "請檢查 EDF、Stage 的開始時間、日期、"
            "時區或同步標記。"
        )

    print()
    print("=" * 80)
    print("新患者前處理完成")
    print("=" * 80)
    print(f"Patient ID：{patient_id}")
    print(
        f"輸出資料夾："
        f"{output_folder}"
    )
    print(
        f"stages_aligned.csv："
        f"{stages_aligned_csv}"
    )
    print(
        f"events_aligned.csv："
        f"{events_aligned_csv}"
    )
    print(
        "沒有修改原本 19 位訓練患者的 "
        "manifest.csv。"
    )
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "處理 data/incoming 中的單一新患者，"
            "建立 Stage、Event 與 EDF 時間對齊資料。"
        )
    )

    parser.add_argument(
        "--patient-id",
        required=True,
        help=(
            "data/incoming 中的患者資料夾名稱，"
            "例如：20201014T221256 - d25c6"
        ),
    )

    parser.add_argument(
        "--allow-partial-edf",
        action="store_true",
        help=(
            "允許 EDF 僅涵蓋部分 Stage 時仍繼續執行。"
            "輸出結果只代表 EDF 實際記錄片段，"
            "   不能視為整晚 PSG。"
        ),
    )

    args = parser.parse_args()

    process_patient(
        patient_id=str(
            args.patient_id
        ).strip(),
        allow_partial_edf=bool(
            args.allow_partial_edf
        ),
    )


if __name__ == "__main__":
    main()
