from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predict_new_patient import (
    build_inference_dataset,
)
from src.importers.new_patient_importer import (
    NewPatientImporter,
)


PROJECT_ROOT = Path(__file__).resolve().parent

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

INFERENCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "inference"
)

DEMOGRAPHICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "demographics"
    / "patients.xlsx.xlsx"
)


RESPIRATORY_EVENT_TYPES = {
    "HYPOPNEA",
    "OBSTRUCTIVE_APNEA",
    "CENTRAL_APNEA",
    "MIXED_APNEA",
}


def load_csv(
    file_path: Path,
    required: bool = True,
) -> pd.DataFrame:
    if not file_path.exists():
        if required:
            raise FileNotFoundError(
                f"找不到檔案：{file_path}"
            )

        return pd.DataFrame()

    return pd.read_csv(
        file_path
    )


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

    return result


def safe_int(
    value: Any,
) -> int | None:
    number = safe_float(value)

    if number is None:
        return None

    return int(round(number))


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    try:
        denominator_value = float(
            denominator
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if (
        not np.isfinite(
            denominator_value
        )
        or denominator_value == 0
    ):
        return None

    result = float(
        numerator
    ) / denominator_value

    if not np.isfinite(result):
        return None

    return result


def safe_json_value(
    value: Any,
) -> Any:
    if value is None:
        return None

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
        pd.Timestamp,
    ):
        return value.isoformat(
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

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=safe_json_value,
        )


def numeric_series(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in data.columns:
        return pd.Series(
            np.nan,
            index=data.index,
            dtype="float64",
        )

    return pd.to_numeric(
        data[column],
        errors="coerce",
    )


def first_existing_column(
    data: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in data.columns:
            return column

    return None


def mean_or_none(
    data: pd.DataFrame,
    column: str,
) -> float | None:
    values = numeric_series(
        data,
        column,
    )

    if values.notna().sum() == 0:
        return None

    return safe_float(
        values.mean()
    )


def median_or_none(
    data: pd.DataFrame,
    column: str,
) -> float | None:
    values = numeric_series(
        data,
        column,
    )

    if values.notna().sum() == 0:
        return None

    return safe_float(
        values.median()
    )


def min_or_none(
    data: pd.DataFrame,
    column: str,
) -> float | None:
    values = numeric_series(
        data,
        column,
    )

    if values.notna().sum() == 0:
        return None

    return safe_float(
        values.min()
    )


def max_or_none(
    data: pd.DataFrame,
    column: str,
) -> float | None:
    values = numeric_series(
        data,
        column,
    )

    if values.notna().sum() == 0:
        return None

    return safe_float(
        values.max()
    )


def classify_ahi(
    ahi: float | None,
) -> str:
    if ahi is None:
        return "UNKNOWN"

    if ahi < 5:
        return "NORMAL"

    if ahi < 15:
        return "MILD"

    if ahi < 30:
        return "MODERATE"

    return "SEVERE"


def classify_oxygen_burden(
    robust_spo2_min: float | None,
    t90_fraction: float | None,
    t88_fraction: float | None,
) -> dict[str, str]:
    """
    研究型夜間血氧負荷分級。

    將血氧問題拆成兩個維度：

    1. desaturation_depth
       血氧下降深度。

    2. sustained_low_oxygen_burden
       持續性低血氧負荷。

    這是研究型分級，
    不是正式臨床診斷標準。
    """

    # ================================================================
    # 1. 血氧下降深度
    # ================================================================

    if robust_spo2_min is None:
        desaturation_depth = "UNKNOWN"

    elif robust_spo2_min < 80.0:
        desaturation_depth = "SEVERE"

    elif robust_spo2_min < 85.0:
        desaturation_depth = "MODERATE"

    elif robust_spo2_min < 90.0:
        desaturation_depth = "MILD_TO_MODERATE"

    else:
        desaturation_depth = "LOW"


    # ================================================================
    # 2. 持續性低血氧負荷
    # ================================================================

    if (
        t90_fraction is None
        and t88_fraction is None
    ):
        sustained_low_oxygen_burden = "UNKNOWN"

    else:
        t90 = (
            float(t90_fraction)
            if t90_fraction is not None
            else 0.0
        )

        t88 = (
            float(t88_fraction)
            if t88_fraction is not None
            else 0.0
        )

        if (
            t90 >= 0.20
            or t88 >= 0.10
        ):
            sustained_low_oxygen_burden = "HIGH"

        elif (
            t90 >= 0.10
            or t88 >= 0.05
        ):
            sustained_low_oxygen_burden = "MODERATE"

        elif (
            t90 > 0.01
            or t88 > 0.01
        ):
            sustained_low_oxygen_burden = "MILD"

        else:
            sustained_low_oxygen_burden = "LOW"


    # ================================================================
    # 3. 相容舊程式的總體研究型標籤
    # ================================================================

    if (
        desaturation_depth == "SEVERE"
        or sustained_low_oxygen_burden == "HIGH"
    ):
        hypoxemia_level = "HIGH_BURDEN"

    elif (
        desaturation_depth == "MODERATE"
        or sustained_low_oxygen_burden == "MODERATE"
    ):
        hypoxemia_level = "MODERATE_BURDEN"

    elif (
        desaturation_depth == "MILD_TO_MODERATE"
        or sustained_low_oxygen_burden == "MILD"
    ):
        hypoxemia_level = "MILD_BURDEN"

    elif (
        desaturation_depth == "UNKNOWN"
        and sustained_low_oxygen_burden == "UNKNOWN"
    ):
        hypoxemia_level = "UNKNOWN"

    else:
        hypoxemia_level = "LOW_BURDEN"


    return {
        "desaturation_depth": (
            desaturation_depth
        ),
        "sustained_low_oxygen_burden": (
            sustained_low_oxygen_burden
        ),
        "hypoxemia_level": (
            hypoxemia_level
        ),
    }


def normalize_position_value(
    value: Any,
) -> str:
    if value is None:
        return "UNKNOWN"

    try:
        if pd.isna(value):
            return "UNKNOWN"
    except (
        TypeError,
        ValueError,
    ):
        pass

    text = str(
        value
    ).strip().upper()

    mapping = {
        "SUPINE": "SUPINE",
        "BACK": "SUPINE",
        "仰睡": "SUPINE",
        "平躺": "SUPINE",
        "PRONE": "PRONE",
        "趴睡": "PRONE",
        "LEFT": "LEFT",
        "LEFT_SIDE": "LEFT",
        "LEFT SIDE": "LEFT",
        "左側": "LEFT",
        "左躺": "LEFT",
        "RIGHT": "RIGHT",
        "RIGHT_SIDE": "RIGHT",
        "RIGHT SIDE": "RIGHT",
        "右側": "RIGHT",
        "右躺": "RIGHT",
        "UPRIGHT": "UPRIGHT",
        "SITTING": "UPRIGHT",
    }

    if text in mapping:
        return mapping[text]

    if "SUPINE" in text:
        return "SUPINE"

    if "PRONE" in text:
        return "PRONE"

    if "LEFT" in text:
        return "LEFT"

    if "RIGHT" in text:
        return "RIGHT"

    if "UPRIGHT" in text:
        return "UPRIGHT"

    return "UNKNOWN"


def infer_position_series(
    data: pd.DataFrame,
) -> pd.Series:
    """
    優先使用已分類的姿勢欄位。

    若目前資料只有數值 PosAngle，則不在這裡
    強行定義角度區間，避免錯誤推論。
    """
    candidates = [
        "position_label",
        "position_class",
        "sleep_position",
        "position_category",
        "position_type",
        "body_position",
    ]

    column = first_existing_column(
        data,
        candidates,
    )

    if column is None:
        return pd.Series(
            "UNKNOWN",
            index=data.index,
            dtype="object",
        )

    return data[column].apply(
        normalize_position_value
    )


def stage_sleep_mask(
    stages: pd.DataFrame,
) -> pd.Series:
    if "stage" not in stages.columns:
        return pd.Series(
            False,
            index=stages.index,
        )

    return (
        stages["stage"]
        .astype(str)
        .str.upper()
        .isin(
            [
                "N1",
                "N2",
                "N3",
                "REM",
            ]
        )
    )


def calculate_sleep_minutes(
    stages: pd.DataFrame,
) -> float:
    sleep_mask = stage_sleep_mask(
        stages
    )

    return float(
        sleep_mask.sum()
        * 30.0
        / 60.0
    )


def count_events(
    events: pd.DataFrame,
    event_type: str,
) -> int:
    if (
        events.empty
        or "event_type"
        not in events.columns
    ):
        return 0

    return int(
        (
            events[
                "event_type"
            ]
            .astype(str)
            .str.upper()
            == event_type
        ).sum()
    )


def calculate_event_summary(
    events: pd.DataFrame,
    sleep_hours: float,
) -> dict[str, Any]:
    hypopnea_count = count_events(
        events,
        "HYPOPNEA",
    )

    obstructive_count = count_events(
        events,
        "OBSTRUCTIVE_APNEA",
    )

    central_count = count_events(
        events,
        "CENTRAL_APNEA",
    )

    mixed_count = count_events(
        events,
        "MIXED_APNEA",
    )

    apnea_count = (
        obstructive_count
        + central_count
        + mixed_count
    )

    respiratory_event_count = (
        hypopnea_count
        + apnea_count
    )

    ahi = safe_divide(
        respiratory_event_count,
        sleep_hours,
    )

    obstructive_fraction = safe_divide(
        obstructive_count,
        respiratory_event_count,
    )

    central_fraction = safe_divide(
        central_count,
        respiratory_event_count,
    )

    mixed_fraction = safe_divide(
        mixed_count,
        respiratory_event_count,
    )

    apnea_fraction = safe_divide(
        apnea_count,
        respiratory_event_count,
    )

    arousal_count = count_events(
        events,
        "AROUSAL",
    )

    respiratory_arousal_count = 0
    spontaneous_arousal_count = 0

    if (
        not events.empty
        and "event_type"
        in events.columns
        and "subtype"
        in events.columns
    ):
        arousal_rows = events[
            events[
                "event_type"
            ]
            .astype(str)
            .str.upper()
            == "AROUSAL"
        ]

        subtype = (
            arousal_rows["subtype"]
            .astype(str)
            .str.lower()
        )

        respiratory_arousal_count = int(
            subtype.str.contains(
                "respiratory",
                na=False,
            ).sum()
        )

        spontaneous_arousal_count = int(
            subtype.str.contains(
                "spontaneous",
                na=False,
            ).sum()
        )

    arousal_index = safe_divide(
        arousal_count,
        sleep_hours,
    )

    respiratory_arousal_index = (
        safe_divide(
            respiratory_arousal_count,
            sleep_hours,
        )
    )

    return {
        "hypopnea_count": (
            hypopnea_count
        ),
        "hypopnea_mechanism_classified": False,
        "hypopnea_mechanism_classification_basis": (
            "目前事件資料僅提供通用 HYPOPNEA 標籤，"
            "未提供阻塞型、中央型或混合型 hypopnea 分類。"
        ),
        "obstructive_apnea_count": (
            obstructive_count
        ),
        "central_apnea_count": (
            central_count
        ),
        "mixed_apnea_count": (
            mixed_count
        ),
        "total_apnea_count": (
            apnea_count
        ),
        "respiratory_event_count": (
            respiratory_event_count
        ),
        "ahi": ahi,
        "ahi_severity": classify_ahi(
            ahi
        ),
        "apnea_fraction": (
            apnea_fraction
        ),
        "obstructive_event_fraction": (
            obstructive_fraction
        ),
        "central_event_fraction": (
            central_fraction
        ),
        "mixed_event_fraction": (
            mixed_fraction
        ),
        "arousal_count": (
            arousal_count
        ),
        "respiratory_arousal_count": (
            respiratory_arousal_count
        ),
        "spontaneous_arousal_count": (
            spontaneous_arousal_count
        ),
        "arousal_index": (
            arousal_index
        ),
        "respiratory_arousal_index": (
            respiratory_arousal_index
        ),
    }


def build_epoch_event_flags(
    epochs: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    result = epochs[
        [
            "epoch_index",
            "start_time",
            "end_time",
            "stage",
        ]
    ].copy()

    result["has_respiratory_event"] = (
        False
    )

    result[
        "has_obstructive_apnea"
    ] = False

    result[
        "has_central_apnea"
    ] = False

    result[
        "has_hypopnea"
    ] = False

    if events.empty:
        return result

    result["start_time"] = pd.to_datetime(
        result["start_time"],
        errors="coerce",
    )

    result["end_time"] = pd.to_datetime(
        result["end_time"],
        errors="coerce",
    )

    event_data = events.copy()

    event_data[
        "start_time"
    ] = pd.to_datetime(
        event_data["start_time"],
        errors="coerce",
    )

    event_data[
        "end_time"
    ] = pd.to_datetime(
        event_data["end_time"],
        errors="coerce",
    )

    event_data = event_data.dropna(
        subset=[
            "start_time",
            "end_time",
            "event_type",
        ]
    )

    for _, event in event_data.iterrows():
        event_type = str(
            event["event_type"]
        ).upper()

        if (
            event_type
            not in RESPIRATORY_EVENT_TYPES
        ):
            continue

        overlap = (
            (
                result["start_time"]
                < event["end_time"]
            )
            & (
                result["end_time"]
                > event["start_time"]
            )
        )

        result.loc[
            overlap,
            "has_respiratory_event",
        ] = True

        if (
            event_type
            == "OBSTRUCTIVE_APNEA"
        ):
            result.loc[
                overlap,
                "has_obstructive_apnea",
            ] = True

        elif (
            event_type
            == "CENTRAL_APNEA"
        ):
            result.loc[
                overlap,
                "has_central_apnea",
            ] = True

        elif (
            event_type
            == "HYPOPNEA"
        ):
            result.loc[
                overlap,
                "has_hypopnea",
            ] = True

    return result


def calculate_stage_specific_indices(
    epoch_flags: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    stage_values = (
        epoch_flags["stage"]
        .astype(str)
        .str.upper()
    )

    for stage_name in [
        "N1",
        "N2",
        "N3",
        "REM",
    ]:
        mask = (
            stage_values
            == stage_name
        )

        stage_hours = (
            mask.sum()
            * 30.0
            / 3600.0
        )

        event_epochs = int(
            epoch_flags.loc[
                mask,
                "has_respiratory_event",
            ].sum()
        )

        result[
            f"{stage_name.lower()}_hours"
        ] = float(
            stage_hours
        )

        result[
            (
                f"{stage_name.lower()}_"
                "respiratory_event_epoch_count"
            )
        ] = event_epochs

        result[
            (
                f"{stage_name.lower()}_"
                "event_epoch_index"
            )
        ] = safe_divide(
            event_epochs,
            stage_hours,
        )

    rem_index = result.get(
        "rem_event_epoch_index"
    )

    non_rem_mask = (
        stage_values.isin(
            [
                "N1",
                "N2",
                "N3",
            ]
        )
    )

    non_rem_hours = (
        non_rem_mask.sum()
        * 30.0
        / 3600.0
    )

    non_rem_event_epochs = int(
        epoch_flags.loc[
            non_rem_mask,
            "has_respiratory_event",
        ].sum()
    )

    non_rem_index = safe_divide(
        non_rem_event_epochs,
        non_rem_hours,
    )

    result[
        "non_rem_hours"
    ] = float(
        non_rem_hours
    )

    result[
        "non_rem_event_epoch_index"
    ] = non_rem_index

    result[
        "rem_to_non_rem_event_ratio"
    ] = (
        safe_divide(
            rem_index,
            non_rem_index,
        )
        if (
            rem_index is not None
            and non_rem_index is not None
        )
        else None
    )

    ratio = result[
        "rem_to_non_rem_event_ratio"
    ]

    if ratio is None:
        result[
            "rem_relevance"
        ] = "UNKNOWN"
    elif ratio >= 2.0:
        result[
            "rem_relevance"
        ] = "HIGH"
    elif ratio >= 1.3:
        result[
            "rem_relevance"
        ] = "MODERATE"
    else:
        result[
            "rem_relevance"
        ] = "LOW"

    return result


def calculate_position_profile(
    inference_data: pd.DataFrame,
    epoch_flags: pd.DataFrame,
) -> dict[str, Any]:
    positions = infer_position_series(
        inference_data
    )

    merged = epoch_flags.copy()

    merged[
        "position_category"
    ] = positions.to_numpy()

    result: dict[str, Any] = {}

    for position in [
        "SUPINE",
        "LEFT",
        "RIGHT",
        "PRONE",
        "UPRIGHT",
        "UNKNOWN",
    ]:
        mask = (
            merged[
                "position_category"
            ]
            == position
        )

        epoch_count = int(
            mask.sum()
        )

        event_epoch_count = int(
            merged.loc[
                mask,
                "has_respiratory_event",
            ].sum()
        )

        hours = (
            epoch_count
            * 30.0
            / 3600.0
        )

        result[
            f"{position.lower()}_epoch_count"
        ] = epoch_count

        result[
            f"{position.lower()}_hours"
        ] = float(
            hours
        )

        result[
            (
                f"{position.lower()}_"
                "respiratory_event_epoch_count"
            )
        ] = event_epoch_count

        result[
            (
                f"{position.lower()}_"
                "event_epoch_index"
            )
        ] = safe_divide(
            event_epoch_count,
            hours,
        )

    known_position_count = int(
        (
            merged[
                "position_category"
            ]
            != "UNKNOWN"
        ).sum()
    )

    result[
        "known_position_epoch_fraction"
    ] = safe_divide(
        known_position_count,
        len(merged),
    )

    supine_index = result.get(
        "supine_event_epoch_index"
    )

    lateral_hours = (
        result.get(
            "left_hours",
            0.0,
        )
        + result.get(
            "right_hours",
            0.0,
        )
    )

    lateral_events = (
        result.get(
            (
                "left_respiratory_"
                "event_epoch_count"
            ),
            0,
        )
        + result.get(
            (
                "right_respiratory_"
                "event_epoch_count"
            ),
            0,
        )
    )

    lateral_index = safe_divide(
        lateral_events,
        lateral_hours,
    )

    result[
        "lateral_hours"
    ] = lateral_hours

    result[
        "lateral_event_epoch_index"
    ] = lateral_index

    if (
        supine_index is None
        or lateral_index is None
    ):
        ratio = None
    else:
        ratio = safe_divide(
            supine_index,
            lateral_index,
        )

    result[
        "supine_to_lateral_event_ratio"
    ] = ratio

    if ratio is None:
        result[
            "position_relevance"
        ] = "UNKNOWN"
    elif ratio >= 2.0:
        result[
            "position_relevance"
        ] = "HIGH"
    elif ratio >= 1.3:
        result[
            "position_relevance"
        ] = "MODERATE"
    else:
        result[
            "position_relevance"
        ] = "LOW"

    return result


def calculate_spo2_profile(
    inference_data: pd.DataFrame,
) -> dict[str, Any]:
    """
    建立品質感知的 SpO₂ Profile。

    不直接使用全夜單一最低值作為缺氧分級依據，
    而是先篩選品質合格 Epoch，再使用穩健統計量。
    """
    data = inference_data.copy()

    spo2_mean_series = numeric_series(
        data,
        "spo2_mean",
    )

    spo2_min_series = numeric_series(
        data,
        "spo2_min",
    )

    spo2_median_series = numeric_series(
        data,
        "spo2_median",
    )

    spo2_std_series = numeric_series(
        data,
        "spo2_std",
    )

    t90_series = numeric_series(
        data,
        "spo2_below_90_fraction",
    )

    t88_series = numeric_series(
        data,
        "spo2_below_88_fraction",
    )

    validity_series = numeric_series(
        data,
        "spo2_valid_fraction",
    )

    # 品質合格條件：
    # 1. 至少 80% 樣本為有效 SpO₂
    # 2. 平均與最低 SpO₂ 在合理生理範圍
    quality_mask = (
        (validity_series >= 0.80)
        & spo2_mean_series.between(
            50.0,
            100.0,
            inclusive="both",
        )
        & spo2_min_series.between(
            50.0,
            100.0,
            inclusive="both",
        )
    )

    valid_data = data[
        quality_mask
    ].copy()

    valid_spo2_mean = numeric_series(
        valid_data,
        "spo2_mean",
    )

    valid_spo2_min = numeric_series(
        valid_data,
        "spo2_min",
    )

    valid_spo2_median = numeric_series(
        valid_data,
        "spo2_median",
    )

    valid_spo2_std = numeric_series(
        valid_data,
        "spo2_std",
    )

    valid_t90 = numeric_series(
        valid_data,
        "spo2_below_90_fraction",
    )

    valid_t88 = numeric_series(
        valid_data,
        "spo2_below_88_fraction",
    )

    # 原始極端值保留做品質警告，但不直接拿來分級
    observed_absolute_min = (
        safe_float(
            spo2_min_series.min()
        )
        if spo2_min_series.notna().sum() > 0
        else None
    )

    if valid_spo2_min.notna().sum() > 0:
        robust_spo2_min_p01 = safe_float(
            valid_spo2_min.quantile(0.01)
        )

        robust_spo2_min_p05 = safe_float(
            valid_spo2_min.quantile(0.05)
        )

        valid_absolute_min = safe_float(
            valid_spo2_min.min()
        )
    else:
        robust_spo2_min_p01 = None
        robust_spo2_min_p05 = None
        valid_absolute_min = None

    spo2_mean = (
        safe_float(
            valid_spo2_mean.mean()
        )
        if valid_spo2_mean.notna().sum() > 0
        else None
    )

    spo2_median = (
        safe_float(
            valid_spo2_median.median()
        )
        if valid_spo2_median.notna().sum() > 0
        else None
    )

    spo2_std = (
        safe_float(
            valid_spo2_std.mean()
        )
        if valid_spo2_std.notna().sum() > 0
        else None
    )

    t90_fraction = (
        safe_float(
            valid_t90.mean()
        )
        if valid_t90.notna().sum() > 0
        else None
    )

    t88_fraction = (
        safe_float(
            valid_t88.mean()
        )
        if valid_t88.notna().sum() > 0
        else None
    )

    total_epoch_count = int(
        len(data)
    )

    valid_epoch_count = int(
        quality_mask.sum()
    )

    invalid_epoch_count = (
        total_epoch_count
        - valid_epoch_count
    )

    valid_epoch_fraction = safe_divide(
        valid_epoch_count,
        total_epoch_count,
    )

    robust_min_for_classification = (
        robust_spo2_min_p01
        if robust_spo2_min_p01 is not None
        else robust_spo2_min_p05
    )

    oxygen_burden = classify_oxygen_burden(
        robust_spo2_min=(
            robust_min_for_classification
            ),
        t90_fraction=(
            t90_fraction
        ),
        t88_fraction=(
            t88_fraction
        ),
    )

    desaturation_depth = oxygen_burden[
     "desaturation_depth"
    ]

    sustained_low_oxygen_burden = oxygen_burden[
        "sustained_low_oxygen_burden"
    ]

    hypoxemia_level = oxygen_burden[
        "hypoxemia_level"
    ]

    outlier_warning = False

    if (
        observed_absolute_min is not None
        and robust_spo2_min_p01 is not None
        and observed_absolute_min
        < robust_spo2_min_p01 - 5.0
    ):
        outlier_warning = True

    low_quality_warning = (
        valid_epoch_fraction is None
        or valid_epoch_fraction < 0.90
    )

    return {
        "spo2_mean": spo2_mean,
        "spo2_median": spo2_median,

        # 原始全夜最低值，只供參考
        "spo2_observed_absolute_min": (
            observed_absolute_min
        ),

        # 品質合格 Epoch 中的最低值
        "spo2_valid_absolute_min": (
            valid_absolute_min
        ),

        # 穩健最低值，主要供分級使用
        "spo2_min_p01_valid_epochs": (
            robust_spo2_min_p01
        ),
        "spo2_min_p05_valid_epochs": (
            robust_spo2_min_p05
        ),

        # 為了相容原本其他程式，
        # spo2_min 改代表品質合格資料的 P1
        "spo2_min": (
            robust_min_for_classification
        ),

        "spo2_std_mean": spo2_std,
        "spo2_below_90_fraction": (
            t90_fraction
        ),
        "spo2_below_88_fraction": (
            t88_fraction
        ),

        "spo2_total_epoch_count": (
            total_epoch_count
        ),
        "spo2_valid_epoch_count": (
            valid_epoch_count
        ),
        "spo2_invalid_epoch_count": (
            invalid_epoch_count
        ),
        "spo2_valid_epoch_fraction": (
            valid_epoch_fraction
        ),

        "spo2_quality_threshold": 0.80,
        "spo2_outlier_warning": (
            outlier_warning
        ),
        "spo2_low_quality_warning": (
            low_quality_warning
        ),

        "desaturation_depth": (
            desaturation_depth
        ),

        "sustained_low_oxygen_burden": (
         sustained_low_oxygen_burden
        ),

        "hypoxemia_level": (
            hypoxemia_level
        ),

        "spo2_interpretation_note": (
            "缺氧分級使用品質合格 Epoch 的 "
            "SpO₂ 最低值第 1 百分位、T90 與 T88；"
            "不直接使用全夜單一極端最低值。"
        ),
    }


def calculate_airflow_profile(
    inference_data: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "flow_respiratory_rate_mean_bpm": (
            mean_or_none(
                inference_data,
                (
                    "flow_respiratory_"
                    "rate_bpm"
                ),
            )
        ),
        "flow_respiratory_rate_median_bpm": (
            median_or_none(
                inference_data,
                (
                    "flow_respiratory_"
                    "rate_bpm"
                ),
            )
        ),
        "flow_amplitude_mean": (
            mean_or_none(
                inference_data,
                "flow_amplitude_p90_p10",
            )
        ),
        "flow_iqr_mean": (
            mean_or_none(
                inference_data,
                "flow_iqr",
            )
        ),
        "flow_std_mean": (
            mean_or_none(
                inference_data,
                "flow_std",
            )
        ),
        "thermistor_respiratory_rate_mean_bpm": (
            mean_or_none(
                inference_data,
                (
                    "thermistor_"
                    "respiratory_rate_bpm"
                ),
            )
        ),
        "thermistor_amplitude_mean": (
            mean_or_none(
                inference_data,
                (
                    "thermistor_"
                    "amplitude_p90_p10"
                ),
            )
        ),
    }


def calculate_effort_profile(
    inference_data: pd.DataFrame,
) -> dict[str, Any]:
    correlation = numeric_series(
        inference_data,
        (
            "thorax_abdomen_"
            "correlation"
        ),
    )

    lag = numeric_series(
        inference_data,
        (
            "thorax_abdomen_"
            "lag_seconds"
        ),
    )

    low_correlation_fraction = None

    if correlation.notna().sum() > 0:
        low_correlation_fraction = (
            safe_float(
                (
                    correlation
                    < 0.30
                ).mean()
            )
        )

    large_lag_fraction = None

    if lag.notna().sum() > 0:
        large_lag_fraction = (
            safe_float(
                (
                    lag.abs()
                    > 1.0
                ).mean()
            )
        )

    return {
        "thorax_respiratory_rate_mean_bpm": (
            mean_or_none(
                inference_data,
                (
                    "thorax_"
                    "respiratory_rate_bpm"
                ),
            )
        ),
        "abdomen_respiratory_rate_mean_bpm": (
            mean_or_none(
                inference_data,
                (
                    "abdomen_"
                    "respiratory_rate_bpm"
                ),
            )
        ),
        "thorax_abdomen_correlation_mean": (
            safe_float(
                correlation.mean()
            )
            if correlation.notna().sum() > 0
            else None
        ),
        "thorax_abdomen_correlation_median": (
            safe_float(
                correlation.median()
            )
            if correlation.notna().sum() > 0
            else None
        ),
        "thorax_abdomen_low_correlation_fraction": (
            low_correlation_fraction
        ),
        "thorax_abdomen_lag_mean_seconds": (
            safe_float(
                lag.mean()
            )
            if lag.notna().sum() > 0
            else None
        ),
        "thorax_abdomen_large_lag_fraction": (
            large_lag_fraction
        ),
    }


def calculate_loop_gain_proxy(
    event_summary: dict[str, Any],
    spo2_profile: dict[str, Any],
    inference_data: pd.DataFrame,
) -> dict[str, Any]:
    """
    建立研究型 Loop Gain 傾向分數。

    這不是臨床 Loop Gain 測量，也不是診斷。
    僅將中央型事件、週期性血氧波動及
    呼吸頻率變異等現有特徵整合為 proxy。
    """
    components: dict[
        str,
        float | None
    ] = {}

    central_fraction = safe_float(
        event_summary.get(
            "central_event_fraction"
        )
    )

    components[
        "central_event_component"
    ] = (
        min(
            central_fraction / 0.20,
            1.0,
        )
        if central_fraction is not None
        else None
    )

    spo2_std = safe_float(
        spo2_profile.get(
            "spo2_std_mean"
        )
    )

    components[
        "spo2_variability_component"
    ] = (
        min(
            spo2_std / 3.0,
            1.0,
        )
        if spo2_std is not None
        else None
    )

    flow_rate = numeric_series(
        inference_data,
        (
            "flow_respiratory_"
            "rate_bpm"
        ),
    )

    if flow_rate.notna().sum() >= 5:
        flow_rate_cv = safe_divide(
            flow_rate.std(),
            abs(
                flow_rate.mean()
            ),
        )
    else:
        flow_rate_cv = None

    components[
        "respiratory_rate_variability_component"
    ] = (
        min(
            flow_rate_cv / 0.35,
            1.0,
        )
        if flow_rate_cv is not None
        else None
    )

    respiratory_arousal_index = (
        safe_float(
            event_summary.get(
                (
                    "respiratory_"
                    "arousal_index"
                )
            )
        )
    )

    components[
        "respiratory_arousal_component"
    ] = (
        min(
            respiratory_arousal_index
            / 20.0,
            1.0,
        )
        if (
            respiratory_arousal_index
            is not None
        )
        else None
    )

    available_components = [
        value
        for value in components.values()
        if value is not None
    ]

    if not available_components:
        score = None
    else:
        score = float(
            np.mean(
                available_components
            )
        )

    if score is None:
        level = "UNKNOWN"
    elif score >= 0.67:
        level = "HIGH"
    elif score >= 0.33:
        level = "MODERATE"
    else:
        level = "LOW"

    return {
        "loop_gain_proxy_score": (
            score
        ),
        "loop_gain_proxy_level": (
            level
        ),
        "loop_gain_proxy_components": (
            components
        ),
        "loop_gain_proxy_warning": (
            "此分數為研究型 proxy，"
            "並非臨床 Loop Gain 測量或診斷。"
        ),
    }


def determine_dominant_mechanism(
    event_summary: dict[str, Any],
    spo2_profile: dict[str, Any],
    position_profile: dict[str, Any],
    rem_profile: dict[str, Any],
    effort_profile: dict[str, Any],
    loop_gain_profile: dict[str, Any],
) -> dict[str, Any]:
    scores: dict[str, float] = {
        "OBSTRUCTIVE_ANATOMICAL": 0.0,
        "HYPOXEMIA": 0.0,
        "POSITIONAL": 0.0,
        "REM_RELATED": 0.0,
        "RESPIRATORY_INSTABILITY": 0.0,
        "RESPIRATORY_EFFORT_DYSYNCHRONY": 0.0,
    }

    obstructive_fraction = safe_float(
        event_summary.get(
            "obstructive_event_fraction"
        )
    )

    if obstructive_fraction is not None:
        scores[
            "OBSTRUCTIVE_ANATOMICAL"
        ] += obstructive_fraction

    hypoxemia_level = str(
        spo2_profile.get(
            "hypoxemia_level",
            "UNKNOWN",
        )
    )

    hypoxemia_scores = {
        "LOW_BURDEN": 0.10,
        "MILD_BURDEN": 0.35,
        "MODERATE_BURDEN": 0.70,
        "HIGH_BURDEN": 1.00,
        "UNKNOWN": 0.0,

    # 向下相容舊版標籤
        "NONE": 0.0,
        "MILD": 0.35,
        "MODERATE": 0.70,
        "SEVERE": 1.00,
    }

    scores[
        "HYPOXEMIA"
    ] = hypoxemia_scores.get(
        hypoxemia_level,
        0.0,
    )

    position_level = str(
        position_profile.get(
            "position_relevance",
            "UNKNOWN",
        )
    )

    position_scores = {
        "LOW": 0.20,
        "MODERATE": 0.60,
        "HIGH": 1.0,
        "UNKNOWN": 0.0,
    }

    scores[
        "POSITIONAL"
    ] = position_scores.get(
        position_level,
        0.0,
    )

    rem_level = str(
        rem_profile.get(
            "rem_relevance",
            "UNKNOWN",
        )
    )

    scores[
        "REM_RELATED"
    ] = position_scores.get(
        rem_level,
        0.0,
    )

    loop_score = safe_float(
        loop_gain_profile.get(
            "loop_gain_proxy_score"
        )
    )

    scores[
        "RESPIRATORY_INSTABILITY"
    ] = (
        loop_score
        if loop_score is not None
        else 0.0
    )

    low_corr_fraction = safe_float(
        effort_profile.get(
            (
                "thorax_abdomen_low_"
                "correlation_fraction"
            )
        )
    )

    large_lag_fraction = safe_float(
        effort_profile.get(
            (
                "thorax_abdomen_large_"
                "lag_fraction"
            )
        )
    )

    effort_components = [
        value
        for value in [
            low_corr_fraction,
            large_lag_fraction,
        ]
        if value is not None
    ]

    if effort_components:
        scores[
            (
                "RESPIRATORY_EFFORT_"
                "DYSYNCHRONY"
            )
        ] = float(
            np.mean(
                effort_components
            )
        )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    dominant = (
        ranked[0][0]
        if ranked
        else "UNKNOWN"
    )

    return {
        "dominant_mechanism": (
            dominant
        ),
        "mechanism_scores": (
            scores
        ),
        "mechanism_ranking": [
            {
                "rank": index,
                "mechanism": mechanism,
                "score": score,
            }
            for index, (
                mechanism,
                score,
            ) in enumerate(
                ranked,
                start=1,
            )
        ],
        "interpretation_warning": (
            "機轉分數為研究型資料摘要，"
            "不等同正式病理生理診斷。"
        ),
    }


def build_readable_summary(
    patient_id: str,
    patient_files: Any,
    sleep_minutes: float,
    event_summary: dict[str, Any],
    spo2_profile: dict[str, Any],
    position_profile: dict[str, Any],
    rem_profile: dict[str, Any],
    loop_gain_profile: dict[str, Any],
    mechanism_profile: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.append(
        f"患者 {patient_id} 呼吸生理摘要。"
    )

    lines.append(
        "基本資料："
        f"sex={patient_files.sex}、"
        f"年齡={patient_files.age}、"
        f"BMI={patient_files.bmi}。"
    )

    ahi = event_summary.get(
        "ahi"
    )

    ahi_text = (
        f"{ahi:.2f}"
        if ahi is not None
        else "無法計算"
    )

    lines.append(
        f"總睡眠時間約 {sleep_minutes:.1f} 分鐘；"
        f"AHI={ahi_text}，"
        f"嚴重度={event_summary.get('ahi_severity')}。"
    )

    lines.append(
        "呼吸事件："
        f"Hypopnea={event_summary.get('hypopnea_count')}、"
        f"Obstructive Apnea="
        f"{event_summary.get('obstructive_apnea_count')}、"
        f"Central Apnea="
        f"{event_summary.get('central_apnea_count')}、"
        f"Mixed Apnea="
        f"{event_summary.get('mixed_apnea_count')}。"
    )

    lines.append(
        "血氧："
        f"平均 SpO₂={spo2_profile.get('spo2_mean')}、"
        f"穩健最低 SpO₂={spo2_profile.get('spo2_min')}、"
        f"血氧下降深度="
        f"{spo2_profile.get('desaturation_depth')}、"
        f"持續性低血氧負荷="
        f"{spo2_profile.get('sustained_low_oxygen_burden')}、"
        f"總體血氧負荷="
        f"{spo2_profile.get('hypoxemia_level')}。"
    )

    lines.append(
        "姿勢相關性："
        f"{position_profile.get('position_relevance')}；"
        "REM 相關性："
        f"{rem_profile.get('rem_relevance')}。"
    )

    lines.append(
        "研究型 Loop Gain proxy："
        f"{loop_gain_profile.get('loop_gain_proxy_level')} "
        f"(score="
        f"{loop_gain_profile.get('loop_gain_proxy_score')})。"
    )

    lines.append(
        "目前資料推測的主要模型機轉："
        f"{mechanism_profile.get('dominant_mechanism')}。"
    )

    lines.append(
        "此摘要僅供研究與模型開發，"
        "不可單獨作為臨床診斷或治療選擇依據。"
    )

    return "\n".join(
        lines
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "建立 incoming 新患者的"
            "患者層級呼吸生理摘要。"
        )
    )

    parser.add_argument(
        "--patient-id",
        required=True,
        help=(
            "患者 ID，例如："
            "20201014T221256 - d25c6"
        ),
    )

    args = parser.parse_args()

    patient_id = str(
        args.patient_id
    ).strip()

    patient_folder = (
        INCOMING_ROOT
        / patient_id
    )

    processed_folder = (
        PROCESSED_ROOT
        / patient_id
    )

    inference_folder = (
        INFERENCE_ROOT
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
            "找不到 stages_aligned.csv："
            f"{stages_file}"
        )

    if not events_file.exists():
        raise FileNotFoundError(
            "找不到 events_aligned.csv："
            f"{events_file}"
        )

    print("=" * 80)
    print("患者呼吸生理 Profile")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    importer = NewPatientImporter(
        patient_folder=patient_folder,
        demographics_file=(
            DEMOGRAPHICS_FILE
        ),
    )

    patient_files = importer.inspect()

    stages = load_csv(
        stages_file
    )

    events = load_csv(
        events_file
    )

    stages[
        "start_time"
    ] = pd.to_datetime(
        stages["start_time"],
        errors="coerce",
    )

    stages[
        "end_time"
    ] = pd.to_datetime(
        stages["end_time"],
        errors="coerce",
    )

    if (
        "usable_for_stage_training"
        in stages.columns
    ):
        usable = (
            stages[
                "usable_for_stage_training"
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                ]
            )
        )

        stages = stages[
            usable
        ].copy()

    if stages.empty:
        raise RuntimeError(
            "沒有可用 Stage Epoch。"
        )
    

    inference_features_file = (
        INFERENCE_ROOT
        / patient_id
        / "inference_features.csv"
    )

    print()
    print("=" * 80)
    print("重新建立呼吸特徵")
    print("=" * 80)

    

    if inference_features_file.exists():

        print(
            f"使用既有推論特徵："
            f"{inference_features_file}"
        )

        inference_data = pd.read_csv(
            inference_features_file
        )

    else:

        inference_data = (
            build_inference_dataset(
                patient_id=patient_id,
                edf_file=(
                    patient_files.edf_file
                ),
                stages_file=stages_file,
            )
        )

    if inference_data.empty:
        raise RuntimeError(
            "呼吸特徵資料為空。"
        )

    inference_data = (
        inference_data.sort_values(
            "epoch_index"
        )
        .reset_index(
            drop=True
        )
    )

    stages = (
        stages.sort_values(
            "epoch_index"
        )
        .reset_index(
            drop=True
        )
    )

    sleep_minutes = (
        calculate_sleep_minutes(
            stages
        )
    )

    sleep_hours = (
        sleep_minutes
        / 60.0
    )

    event_summary = (
        calculate_event_summary(
            events=events,
            sleep_hours=sleep_hours,
        )
    )

    epoch_flags = (
        build_epoch_event_flags(
            epochs=stages,
            events=events,
        )
    )

    rem_profile = (
        calculate_stage_specific_indices(
            epoch_flags
        )
    )

    position_profile = (
        calculate_position_profile(
            inference_data=(
                inference_data
            ),
            epoch_flags=(
                epoch_flags
            ),
        )
    )

    spo2_profile = (
        calculate_spo2_profile(
            inference_data
        )
    )

    airflow_profile = (
        calculate_airflow_profile(
            inference_data
        )
    )

    effort_profile = (
        calculate_effort_profile(
            inference_data
        )
    )

    loop_gain_profile = (
        calculate_loop_gain_proxy(
            event_summary=(
                event_summary
            ),
            spo2_profile=(
                spo2_profile
            ),
            inference_data=(
                inference_data
            ),
        )
    )

    mechanism_profile = (
        determine_dominant_mechanism(
            event_summary=(
                event_summary
            ),
            spo2_profile=(
                spo2_profile
            ),
            position_profile=(
                position_profile
            ),
            rem_profile=(
                rem_profile
            ),
            effort_profile=(
                effort_profile
            ),
            loop_gain_profile=(
                loop_gain_profile
            ),
        )
    )

    readable_summary = (
        build_readable_summary(
            patient_id=patient_id,
            patient_files=patient_files,
            sleep_minutes=(
                sleep_minutes
            ),
            event_summary=(
                event_summary
            ),
            spo2_profile=(
                spo2_profile
            ),
            position_profile=(
                position_profile
            ),
            rem_profile=(
                rem_profile
            ),
            loop_gain_profile=(
                loop_gain_profile
            ),
            mechanism_profile=(
                mechanism_profile
            ),
        )
    )

    profile = {
        "patient_id": patient_id,
        "sex": patient_files.sex,
        "age": patient_files.age,
        "BMI": patient_files.bmi,
        "sleep_summary": {
            "usable_epoch_count": int(
                len(stages)
            ),
            "sleep_minutes": (
                sleep_minutes
            ),
            "sleep_hours": (
                sleep_hours
            ),
            "stage_counts": (
                stages[
                    "stage"
                ]
                .astype(str)
                .value_counts()
                .to_dict()
            ),
        },
        "event_summary": (
            event_summary
        ),
        "spo2_profile": (
            spo2_profile
        ),
        "airflow_profile": (
            airflow_profile
        ),
        "respiratory_effort_profile": (
            effort_profile
        ),
        "position_profile": (
            position_profile
        ),
        "rem_profile": (
            rem_profile
        ),
        "loop_gain_proxy": (
            loop_gain_profile
        ),
        "mechanism_profile": (
            mechanism_profile
        ),
        "clinical_readable_summary": (
            readable_summary
        ),
        "research_safety_note": (
            "此 Profile 屬研究型資料摘要。"
            "AHI 以目前 Event Grid 與有效睡眠時間計算；"
            "Loop Gain、機轉與治療相關判斷皆為 proxy，"
            "不可取代正式睡眠醫學評估。"
        ),
    }

    output_folder = (
        inference_folder
        / "respiratory_profile"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_json_file = (
        output_folder
        / "patient_respiratory_profile.json"
    )

    profile_csv_file = (
        output_folder
        / "patient_respiratory_profile.csv"
    )

    text_file = (
        output_folder
        / "patient_respiratory_profile.txt"
    )

    epoch_flags_file = (
        output_folder
        / "epoch_respiratory_event_flags.csv"
    )

    save_json(
        profile_json_file,
        profile,
    )

    flat_row = {
        "patient_id": patient_id,
        "sex": patient_files.sex,
        "age": patient_files.age,
        "BMI": patient_files.bmi,
        "sleep_minutes": (
            sleep_minutes
        ),
        **event_summary,
        **spo2_profile,
        **airflow_profile,
        **effort_profile,
        **{
            key: value
            for key, value
            in position_profile.items()
            if not isinstance(
                value,
                dict,
            )
        },
        **{
            key: value
            for key, value
            in rem_profile.items()
            if not isinstance(
                value,
                dict,
            )
        },
        "loop_gain_proxy_score": (
            loop_gain_profile.get(
                "loop_gain_proxy_score"
            )
        ),
        "loop_gain_proxy_level": (
            loop_gain_profile.get(
                "loop_gain_proxy_level"
            )
        ),
        "dominant_mechanism": (
            mechanism_profile.get(
                "dominant_mechanism"
            )
        ),
    }

    pd.DataFrame(
        [flat_row]
    ).to_csv(
        profile_csv_file,
        index=False,
        encoding="utf-8-sig",
    )

    epoch_flags.to_csv(
        epoch_flags_file,
        index=False,
        encoding="utf-8-sig",
    )

    with text_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            readable_summary
        )

    print()
    print("=" * 80)
    print("患者呼吸生理 Profile 完成")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    print(
        f"總睡眠時間："
        f"{sleep_minutes:.2f} 分鐘"
    )

    ahi = event_summary[
        "ahi"
    ]

    if ahi is not None:
        print(
            f"AHI：{ahi:.2f}"
        )
    else:
        print(
            "AHI：無法計算"
        )

    print(
        "AHI 嚴重度："
        f"{event_summary['ahi_severity']}"
    )

    print(
        "呼吸事件總數："
        f"{event_summary['respiratory_event_count']}"
    )

    print(
        "阻塞型 Apnea："
        f"{event_summary['obstructive_apnea_count']}"
    )

    print(
        "Hypopnea："
        f"{event_summary['hypopnea_count']}"
    )

    print(
        "Central Apnea："
        f"{event_summary['central_apnea_count']}"
    )

    print(
        "最低 SpO₂："
        f"{spo2_profile['spo2_min']}"
    )

    print(
        "血氧下降深度："
        f"{spo2_profile['desaturation_depth']}"
    )

    print(
        "持續性低血氧負荷："
        f"{spo2_profile['sustained_low_oxygen_burden']}"
    )

    print(
        "總體血氧負荷："
        f"{spo2_profile['hypoxemia_level']}"
    )

    print(
        "姿勢相關性："
        f"{position_profile['position_relevance']}"
    )

    print(
        "REM 相關性："
        f"{rem_profile['rem_relevance']}"
    )

    print(
        "Loop Gain proxy："
        f"{loop_gain_profile['loop_gain_proxy_level']} "
        f"(score="
        f"{loop_gain_profile['loop_gain_proxy_score']})"
    )

    print(
        "主要研究型機轉："
        f"{mechanism_profile['dominant_mechanism']}"
    )

    print()
    print("=" * 80)
    print("自動摘要")
    print("=" * 80)

    print(
        readable_summary
    )

    print()
    print("=" * 80)
    print("輸出檔案")
    print("=" * 80)

    print(
        f"完整 Profile JSON："
        f"{profile_json_file}"
    )

    print(
        f"Profile CSV："
        f"{profile_csv_file}"
    )

    print(
        f"文字摘要："
        f"{text_file}"
    )

    print(
        f"Epoch Event Flags："
        f"{epoch_flags_file}"
    )

    print()
    print(
        "注意：Loop Gain 與主要機轉皆為研究型 proxy，"
        "不等於正式臨床診斷。"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()