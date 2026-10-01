from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

INCOMING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "incoming"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

DEMOGRAPHICS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "demographics"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "_metadata_build"
)


PATIENT_TIMESTAMP_PATTERN = re.compile(
    r"(?P<timestamp>\d{8}T\d{6})"
)

MIN_REASONABLE_EDF_YEAR = 1990

MAX_HEADER_FOLDER_DATE_DIFFERENCE_DAYS = 2.0


# ============================================================
# 基礎工具
# ============================================================

def safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(result):
        return None

    return float(result)


def safe_int(
    value: Any,
) -> int | None:
    number = safe_float(value)

    if number is None:
        return None

    return int(round(number))


def safe_json_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): safe_json_value(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            safe_json_value(item)
            for item in value
        ]

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
        (
            pd.Timestamp,
            np.datetime64,
            datetime,
        ),
    ):
        return pd.Timestamp(
            value
        ).isoformat(
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


def save_json(
    file_path: Path,
    data: dict[str, Any],
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_data = safe_json_value(
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


def normalize_datetime(
    value: Any,
) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(timestamp):
        return None

    result = pd.Timestamp(
        timestamp
    )

    if result.tzinfo is not None:
        result = result.tz_localize(
            None
        )

    return result


def normalize_boolean_series(
    values: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        values
    ):
        return values.fillna(False)

    normalized = (
        values.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        [
            "true",
            "1",
            "yes",
            "y",
        ]
    )


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    normalized_lookup = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized_lookup:
            return normalized_lookup[key]

    return None


# ============================================================
# 患者與檔案搜尋
# ============================================================

def parse_patient_timestamp(
    patient_id: str,
) -> pd.Timestamp | None:
    match = PATIENT_TIMESTAMP_PATTERN.search(
        patient_id
    )

    if match is None:
        return None

    timestamp_text = match.group(
        "timestamp"
    )

    try:
        parsed = datetime.strptime(
            timestamp_text,
            "%Y%m%dT%H%M%S",
        )
    except ValueError:
        return None

    return pd.Timestamp(
        parsed
    )


def find_patient_edf(
    patient_id: str,
) -> Path:
    preferred_candidates = [
        (
            INCOMING_ROOT
            / patient_id
            / f"{patient_id}_EDF.edf"
        ),
        (
            RAW_ROOT
            / patient_id
            / f"{patient_id}_EDF.edf"
        ),
    ]

    for candidate in preferred_candidates:
        if candidate.exists():
            return candidate

    matches: list[Path] = []

    for root in [
        INCOMING_ROOT,
        RAW_ROOT,
    ]:
        if not root.exists():
            continue

        direct_folder = (
            root
            / patient_id
        )

        if direct_folder.exists():
            matches.extend(
                direct_folder.glob(
                    "*.edf"
                )
            )

            matches.extend(
                direct_folder.glob(
                    "*.EDF"
                )
            )

    matches = sorted(
        {
            path.resolve()
            for path in matches
            if path.is_file()
        }
    )

    if not matches:
        raise FileNotFoundError(
            f"找不到患者 EDF：{patient_id}"
        )

    return max(
        matches,
        key=lambda path: path.stat().st_size,
    )


def find_stage_source_file(
    patient_id: str,
) -> Path | None:
    for root in [
        INCOMING_ROOT,
        RAW_ROOT,
    ]:
        patient_folder = (
            root
            / patient_id
        )

        if not patient_folder.exists():
            continue

        candidates = sorted(
            list(
                patient_folder.glob(
                    "*stage*.xls"
                )
            )
            + list(
                patient_folder.glob(
                    "*Stage*.xls"
                )
            )
            + list(
                patient_folder.glob(
                    "*stage*.xlsx"
                )
            )
            + list(
                patient_folder.glob(
                    "*Stage*.xlsx"
                )
            )
        )

        if candidates:
            return candidates[0]

    return None


def find_event_source_file(
    patient_id: str,
) -> Path | None:
    for root in [
        INCOMING_ROOT,
        RAW_ROOT,
    ]:
        patient_folder = (
            root
            / patient_id
        )

        if not patient_folder.exists():
            continue

        candidates = sorted(
            list(
                patient_folder.glob(
                    "*Event*Grid*.xls"
                )
            )
            + list(
                patient_folder.glob(
                    "*event*grid*.xls"
                )
            )
            + list(
                patient_folder.glob(
                    "*Event*Grid*.xlsx"
                )
            )
            + list(
                patient_folder.glob(
                    "*event*grid*.xlsx"
                )
            )
        )

        if candidates:
            return candidates[0]

    return None


# ============================================================
# Demographics
# ============================================================

def find_demographics_files() -> list[Path]:
    if not DEMOGRAPHICS_ROOT.exists():
        return []

    candidates = sorted(
        list(
            DEMOGRAPHICS_ROOT.glob(
                "*.xlsx"
            )
        )
        + list(
            DEMOGRAPHICS_ROOT.glob(
                "*.xls"
            )
        )
        + list(
            DEMOGRAPHICS_ROOT.glob(
                "*.csv"
            )
        )
    )

    return [
        path
        for path in candidates
        if path.is_file()
    ]


def normalize_patient_id_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (
        TypeError,
        ValueError,
    ):
        pass

    return str(
        value
    ).strip()


def load_demographics_lookup() -> tuple[
    dict[str, dict[str, Any]],
    list[str],
]:
    lookup: dict[
        str,
        dict[str, Any],
    ] = {}

    source_descriptions: list[str] = []

    for file_path in find_demographics_files():
        try:
            if (
                file_path.suffix.lower()
                == ".csv"
            ):
                sheets = {
                    "CSV": pd.read_csv(
                        file_path
                    )
                }
            else:
                excel = pd.ExcelFile(
                    file_path
                )

                sheets = {
                    sheet_name: pd.read_excel(
                        file_path,
                        sheet_name=sheet_name,
                    )
                    for sheet_name
                    in excel.sheet_names
                }

        except Exception as error:
            source_descriptions.append(
                (
                    f"{file_path}：讀取失敗 "
                    f"{type(error).__name__}: {error}"
                )
            )
            continue

        for sheet_name, dataframe in (
            sheets.items()
        ):
            if dataframe.empty:
                continue

            id_column = first_existing_column(
                dataframe,
                [
                    "ID",
                    "patient_id",
                    "patient id",
                    "Patient ID",
                    "患者ID",
                    "患者 ID",
                ],
            )

            if id_column is None:
                continue

            sex_column = first_existing_column(
                dataframe,
                [
                    "sex",
                    "gender",
                    "性別",
                ],
            )

            age_column = first_existing_column(
                dataframe,
                [
                    "age",
                    "年齡",
                ],
            )

            bmi_column = first_existing_column(
                dataframe,
                [
                    "BMI",
                    "bmi",
                ],
            )

            for _, row in dataframe.iterrows():
                patient_id = (
                    normalize_patient_id_text(
                        row.get(
                            id_column
                        )
                    )
                )

                if not patient_id:
                    continue

                lookup[
                    patient_id
                ] = {
                    "sex": (
                        row.get(
                            sex_column
                        )
                        if sex_column
                        is not None
                        else None
                    ),
                    "age": (
                        safe_float(
                            row.get(
                                age_column
                            )
                        )
                        if age_column
                        is not None
                        else None
                    ),
                    "BMI": (
                        safe_float(
                            row.get(
                                bmi_column
                            )
                        )
                        if bmi_column
                        is not None
                        else None
                    ),
                    "demographics_file": str(
                        file_path
                    ),
                    "demographics_sheet": (
                        sheet_name
                    ),
                }

            source_descriptions.append(
                (
                    f"{file_path} / {sheet_name}："
                    f"已讀取 {len(dataframe)} 列"
                )
            )

    return (
        lookup,
        source_descriptions,
    )


def match_demographics(
    patient_id: str,
    lookup: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    if patient_id in lookup:
        result = dict(
            lookup[patient_id]
        )

        result[
            "demographics_found"
        ] = True

        return result

    normalized_target = (
        patient_id.lower()
        .replace(" ", "")
    )

    for lookup_id, information in (
        lookup.items()
    ):
        normalized_lookup_id = (
            lookup_id.lower()
            .replace(" ", "")
        )

        if (
            normalized_lookup_id
            == normalized_target
        ):
            result = dict(
                information
            )

            result[
                "demographics_found"
            ] = True

            return result

    return {
        "sex": None,
        "age": None,
        "BMI": None,
        "demographics_file": None,
        "demographics_sheet": None,
        "demographics_found": False,
    }


# ============================================================
# EDF 資訊
# ============================================================

def read_edf_information(
    edf_file: Path,
    patient_id: str,
) -> dict[str, Any]:
    raw = mne.io.read_raw_edf(
        edf_file,
        preload=False,
        verbose="ERROR",
    )

    sampling_rate = float(
        raw.info["sfreq"]
    )

    sample_count = int(
        raw.n_times
    )

    duration_seconds = float(
        sample_count
        / sampling_rate
    )

    header_start = normalize_datetime(
        raw.info.get(
            "meas_date"
        )
    )

    patient_timestamp = (
        parse_patient_timestamp(
            patient_id
        )
    )

    corrected_start = header_start

    start_was_corrected = False

    correction_reason = (
        "EDF header 開始時間直接使用"
    )

    if header_start is None:
        if patient_timestamp is None:
            raise RuntimeError(
                (
                    "EDF header 無開始時間，"
                    "患者 ID 也無法解析時間："
                    f"{patient_id}"
                )
            )

        corrected_start = (
            patient_timestamp
        )

        start_was_corrected = True

        correction_reason = (
            "EDF header 缺少開始時間；"
            "改用患者 ID 中的 YYYYMMDDTHHMMSS"
        )

    elif (
        header_start.year
        < MIN_REASONABLE_EDF_YEAR
    ):
        if patient_timestamp is not None:
            corrected_start = (
                patient_timestamp
            )

            start_was_corrected = True

            correction_reason = (
                "EDF header 年份早於 1990；"
                "改用患者 ID 中的 YYYYMMDDTHHMMSS"
            )

    elif patient_timestamp is not None:
        date_difference_days = abs(
            (
                header_start
                - patient_timestamp
            ).total_seconds()
        ) / 86400.0

        if (
            date_difference_days
            > MAX_HEADER_FOLDER_DATE_DIFFERENCE_DAYS
        ):
            corrected_start = (
                patient_timestamp
            )

            start_was_corrected = True

            correction_reason = (
                "EDF header 日期與患者 ID 日期"
                "相差超過允許範圍；"
                "改用患者 ID 中的 YYYYMMDDTHHMMSS"
            )
        else:
            correction_reason = (
                "EDF header 日期與患者 ID 日期一致"
            )

    if corrected_start is None:
        raise RuntimeError(
            f"無法決定 EDF 開始時間：{patient_id}"
        )

    edf_end = (
        corrected_start
        + pd.Timedelta(
            seconds=duration_seconds
        )
    )

    return {
        "original_edf_start": (
            header_start
        ),
        "corrected_edf_start": (
            corrected_start
        ),
        "edf_end": (
            edf_end
        ),
        "edf_start_was_corrected": (
            start_was_corrected
        ),
        "alignment_correction_reason": (
            correction_reason
        ),
        "sampling_rate": (
            sampling_rate
        ),
        "duration_seconds": (
            duration_seconds
        ),
        "sample_count": (
            sample_count
        ),
        "channel_count": int(
            len(
                raw.ch_names
            )
        ),
    }


# ============================================================
# Stage / Event 統計
# ============================================================

def read_csv_optional(
    file_path: Path,
) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        file_path
    )


def summarize_stages(
    stages: pd.DataFrame,
) -> dict[str, Any]:
    if stages.empty:
        return {
            "stage_count": 0,
            "stage_fully_inside_edf": 0,
            "stage_usable_count": 0,
            "stage_coverage_fraction": None,
            "stage_start_time": None,
            "stage_end_time": None,
        }

    stage_count = int(
        len(stages)
    )

    if (
        "fully_inside_edf"
        in stages.columns
    ):
        fully_inside_mask = (
            normalize_boolean_series(
                stages[
                    "fully_inside_edf"
                ]
            )
        )
    else:
        fully_inside_mask = pd.Series(
            False,
            index=stages.index,
        )

    if (
        "usable_for_stage_training"
        in stages.columns
    ):
        usable_mask = (
            normalize_boolean_series(
                stages[
                    "usable_for_stage_training"
                ]
            )
        )
    elif (
        "inside_edf"
        in stages.columns
    ):
        usable_mask = (
            normalize_boolean_series(
                stages[
                    "inside_edf"
                ]
            )
        )
    else:
        usable_mask = (
            fully_inside_mask.copy()
        )

    stage_fully_inside = int(
        fully_inside_mask.sum()
    )

    stage_usable_count = int(
        usable_mask.sum()
    )

    stage_coverage_fraction = (
        float(
            stage_fully_inside
            / stage_count
        )
        if stage_count > 0
        else None
    )

    stage_start_time = None
    stage_end_time = None

    if (
        "start_time"
        in stages.columns
    ):
        start_values = pd.to_datetime(
            stages[
                "start_time"
            ],
            errors="coerce",
        ).dropna()

        if not start_values.empty:
            stage_start_time = (
                start_values.min()
            )

    if (
        "end_time"
        in stages.columns
    ):
        end_values = pd.to_datetime(
            stages[
                "end_time"
            ],
            errors="coerce",
        ).dropna()

        if not end_values.empty:
            stage_end_time = (
                end_values.max()
            )

    return {
        "stage_count": (
            stage_count
        ),
        "stage_fully_inside_edf": (
            stage_fully_inside
        ),
        "stage_usable_count": (
            stage_usable_count
        ),
        "stage_coverage_fraction": (
            stage_coverage_fraction
        ),
        "stage_start_time": (
            stage_start_time
        ),
        "stage_end_time": (
            stage_end_time
        ),
    }


def summarize_events(
    events: pd.DataFrame,
) -> dict[str, Any]:
    if events.empty:
        return {
            "event_count": 0,
            "event_fully_inside_edf": 0,
            "event_usable_count": 0,
            "event_coverage_fraction": None,
            "event_start_time": None,
            "event_end_time": None,
        }

    event_count = int(
        len(events)
    )

    if (
        "fully_inside_edf"
        in events.columns
    ):
        fully_inside_mask = (
            normalize_boolean_series(
                events[
                    "fully_inside_edf"
                ]
            )
        )
    else:
        fully_inside_mask = pd.Series(
            False,
            index=events.index,
        )

    if (
        "usable_for_event_training"
        in events.columns
    ):
        usable_mask = (
            normalize_boolean_series(
                events[
                    "usable_for_event_training"
                ]
            )
        )
    elif (
        "inside_edf"
        in events.columns
    ):
        usable_mask = (
            normalize_boolean_series(
                events[
                    "inside_edf"
                ]
            )
        )
    else:
        usable_mask = (
            fully_inside_mask.copy()
        )

    event_fully_inside = int(
        fully_inside_mask.sum()
    )

    event_usable_count = int(
        usable_mask.sum()
    )

    event_coverage_fraction = (
        float(
            event_fully_inside
            / event_count
        )
        if event_count > 0
        else None
    )

    event_start_time = None
    event_end_time = None

    if (
        "start_time"
        in events.columns
    ):
        start_values = pd.to_datetime(
            events[
                "start_time"
            ],
            errors="coerce",
        ).dropna()

        if not start_values.empty:
            event_start_time = (
                start_values.min()
            )

    if (
        "end_time"
        in events.columns
    ):
        end_values = pd.to_datetime(
            events[
                "end_time"
            ],
            errors="coerce",
        ).dropna()

        if not end_values.empty:
            event_end_time = (
                end_values.max()
            )

    return {
        "event_count": (
            event_count
        ),
        "event_fully_inside_edf": (
            event_fully_inside
        ),
        "event_usable_count": (
            event_usable_count
        ),
        "event_coverage_fraction": (
            event_coverage_fraction
        ),
        "event_start_time": (
            event_start_time
        ),
        "event_end_time": (
            event_end_time
        ),
    }


# ============================================================
# 建立單一患者 metadata
# ============================================================

def build_patient_metadata(
    patient_id: str,
    demographics_lookup: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    processed_folder = (
        PROCESSED_ROOT
        / patient_id
    )

    stages_file = (
        processed_folder
        / "stages_aligned.csv"
    )

    events_file = (
        processed_folder
        / "events_aligned.csv"
    )

    if not stages_file.exists():
        raise FileNotFoundError(
            f"找不到 stages_aligned.csv：{stages_file}"
        )

    if not events_file.exists():
        raise FileNotFoundError(
            f"找不到 events_aligned.csv：{events_file}"
        )

    edf_file = find_patient_edf(
        patient_id
    )

    stage_source_file = (
        find_stage_source_file(
            patient_id
        )
    )

    event_source_file = (
        find_event_source_file(
            patient_id
        )
    )

    demographics = match_demographics(
        patient_id,
        demographics_lookup,
    )

    edf_information = (
        read_edf_information(
            edf_file=edf_file,
            patient_id=patient_id,
        )
    )

    stages = read_csv_optional(
        stages_file
    )

    events = read_csv_optional(
        events_file
    )

    stage_summary = summarize_stages(
        stages
    )

    event_summary = summarize_events(
        events
    )

    source_root = (
        "incoming"
        if INCOMING_ROOT
        in edf_file.parents
        else "raw"
    )

    metadata = {
        "patient_id": (
            patient_id
        ),
        "patient_folder": str(
            edf_file.parent
        ),
        "data_source_root": (
            source_root
        ),
        "edf_file": str(
            edf_file
        ),
        "stage_file": (
            str(
                stage_source_file
            )
            if stage_source_file
            is not None
            else None
        ),
        "event_file": (
            str(
                event_source_file
            )
            if event_source_file
            is not None
            else None
        ),
        "stages_aligned_file": str(
            stages_file
        ),
        "events_aligned_file": str(
            events_file
        ),
        "demographics_file": (
            demographics.get(
                "demographics_file"
            )
        ),
        "demographics_found": (
            demographics.get(
                "demographics_found",
                False,
            )
        ),
        "demographics_sheet": (
            demographics.get(
                "demographics_sheet"
            )
        ),
        "sex": (
            demographics.get(
                "sex"
            )
        ),
        "age": (
            demographics.get(
                "age"
            )
        ),
        "BMI": (
            demographics.get(
                "BMI"
            )
        ),
        **edf_information,
        **stage_summary,
        **event_summary,
        "metadata_build_method": (
            "build_missing_patient_metadata.py"
        ),
        "metadata_build_time": (
            pd.Timestamp.now()
        ),
    }

    return metadata


def write_patient_metadata(
    patient_id: str,
    metadata: dict[str, Any],
) -> tuple[
    Path,
    Path,
]:
    processed_folder = (
        PROCESSED_ROOT
        / patient_id
    )

    processed_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_file = (
        processed_folder
        / "patient_metadata.csv"
    )

    json_file = (
        processed_folder
        / "patient_metadata.json"
    )

    clean_metadata = {
        key: safe_json_value(value)
        for key, value
        in metadata.items()
    }

    pd.DataFrame(
        [
            clean_metadata
        ]
    ).to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        json_file,
        clean_metadata,
    )

    return (
        csv_file,
        json_file,
    )


# ============================================================
# 主程式
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "替 data/processed 中缺少 metadata 的患者"
            "補建 patient_metadata.csv 與 JSON。"
        )
    )

    parser.add_argument(
        "--patient-id",
        default=None,
        help=(
            "只處理指定患者；未指定時處理全部患者。"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "覆蓋已存在的 patient_metadata.csv。"
        ),
    )

    args = parser.parse_args()

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    demographics_lookup, demographics_sources = (
        load_demographics_lookup()
    )

    patient_folders = sorted(
        [
            path
            for path in PROCESSED_ROOT.iterdir()
            if (
                path.is_dir()
                and not path.name.startswith(
                    "_"
                )
            )
        ],
        key=lambda path: path.name,
    )

    if args.patient_id is not None:
        requested_patient_id = str(
            args.patient_id
        ).strip()

        patient_folders = [
            path
            for path in patient_folders
            if path.name
            == requested_patient_id
        ]

        if not patient_folders:
            raise FileNotFoundError(
                (
                    "data/processed 中找不到患者："
                    f"{requested_patient_id}"
                )
            )

    print("=" * 80)
    print("Build Missing Patient Metadata")
    print("=" * 80)

    print(
        f"Processed Root：{PROCESSED_ROOT}"
    )

    print(
        f"患者資料夾數：{len(patient_folders)}"
    )

    print(
        f"Demographics 可匹配患者數："
        f"{len(demographics_lookup)}"
    )

    print(
        f"覆蓋既有 metadata：{args.overwrite}"
    )

    print()
    print("=" * 80)
    print("逐患者處理")
    print("=" * 80)

    result_rows: list[
        dict[str, Any]
    ] = []

    for index, patient_folder in enumerate(
        patient_folders,
        start=1,
    ):
        patient_id = (
            patient_folder.name
        )

        metadata_csv = (
            patient_folder
            / "patient_metadata.csv"
        )

        if (
            metadata_csv.exists()
            and not args.overwrite
        ):
            print(
                f"[SKIP] {index}/"
                f"{len(patient_folders)} "
                f"{patient_id}："
                "patient_metadata.csv 已存在"
            )

            result_rows.append(
                {
                    "patient_id": patient_id,
                    "status": "SKIPPED_EXISTS",
                    "metadata_csv": str(
                        metadata_csv
                    ),
                    "metadata_json": str(
                        patient_folder
                        / "patient_metadata.json"
                    ),
                    "error_type": None,
                    "error_message": None,
                }
            )

            continue

        try:
            metadata = (
                build_patient_metadata(
                    patient_id=patient_id,
                    demographics_lookup=(
                        demographics_lookup
                    ),
                )
            )

            csv_file, json_file = (
                write_patient_metadata(
                    patient_id=patient_id,
                    metadata=metadata,
                )
            )

            corrected_start = (
                metadata.get(
                    "corrected_edf_start"
                )
            )

            corrected_flag = (
                metadata.get(
                    "edf_start_was_corrected"
                )
            )

            demographics_found = (
                metadata.get(
                    "demographics_found"
                )
            )

            print(
                f"[OK] {index}/"
                f"{len(patient_folders)} "
                f"{patient_id}"
            )

            print(
                "     EDF 開始時間："
                f"{corrected_start}"
            )

            print(
                "     時間是否校正："
                f"{corrected_flag}"
            )

            print(
                "     Demographics："
                f"{demographics_found}"
            )

            result_rows.append(
                {
                    "patient_id": patient_id,
                    "status": "CREATED",
                    "metadata_csv": str(
                        csv_file
                    ),
                    "metadata_json": str(
                        json_file
                    ),
                    "edf_start": safe_json_value(
                        corrected_start
                    ),
                    "edf_start_was_corrected": (
                        corrected_flag
                    ),
                    "demographics_found": (
                        demographics_found
                    ),
                    "error_type": None,
                    "error_message": None,
                }
            )

        except Exception as error:
            print(
                f"[FAILED] {index}/"
                f"{len(patient_folders)} "
                f"{patient_id}："
                f"{type(error).__name__}: "
                f"{error}"
            )

            result_rows.append(
                {
                    "patient_id": patient_id,
                    "status": "FAILED",
                    "metadata_csv": None,
                    "metadata_json": None,
                    "error_type": (
                        type(error).__name__
                    ),
                    "error_message": (
                        str(error)
                    ),
                }
            )

    result_df = pd.DataFrame(
        result_rows
    )

    report_csv = (
        REPORT_ROOT
        / "metadata_build_report.csv"
    )

    report_json = (
        REPORT_ROOT
        / "metadata_build_report.json"
    )

    result_df.to_csv(
        report_csv,
        index=False,
        encoding="utf-8-sig",
    )

    status_counts = (
        result_df[
            "status"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
        if not result_df.empty
        else {}
    )

    report = {
        "processed_root": str(
            PROCESSED_ROOT
        ),
        "patient_count": int(
            len(patient_folders)
        ),
        "status_counts": (
            status_counts
        ),
        "demographics_sources": (
            demographics_sources
        ),
        "results": (
            result_df.to_dict(
                orient="records"
            )
        ),
    }

    save_json(
        report_json,
        report,
    )

    created_count = int(
        (
            result_df["status"]
            == "CREATED"
        ).sum()
    )

    skipped_count = int(
        (
            result_df["status"]
            == "SKIPPED_EXISTS"
        ).sum()
    )

    failed_count = int(
        (
            result_df["status"]
            == "FAILED"
        ).sum()
    )

    print()
    print("=" * 80)
    print("Metadata 補建完成")
    print("=" * 80)

    print(
        f"建立成功：{created_count}"
    )

    print(
        f"既有檔案跳過：{skipped_count}"
    )

    print(
        f"失敗：{failed_count}"
    )

    print(
        f"報告 CSV：{report_csv}"
    )

    print(
        f"報告 JSON：{report_json}"
    )

    if failed_count > 0:
        print()
        print("失敗患者：")

        failed_rows = result_df[
            result_df[
                "status"
            ]
            == "FAILED"
        ]

        print(
            failed_rows[
                [
                    "patient_id",
                    "error_type",
                    "error_message",
                ]
            ].to_string(
                index=False
            )
        )

    print("=" * 80)


if __name__ == "__main__":
    main()