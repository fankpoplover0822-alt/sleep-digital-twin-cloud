from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.channel_mapper import (
    ChannelMapper,
)
from src.features.respiratory_feature_builder import (
    RespiratoryFeatureBuilder,
)
from src.features.stage_feature_builder import (
    StageFeatureBuilder,
)
from src.inference.arousal_inference import (
    ArousalInferenceEngine,
)
from src.continual_learning.service import (
    resolve_arousal_artifacts,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parent

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

CHANNEL_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "channel_map.yaml"
)

AROUSAL_ARTIFACTS = resolve_arousal_artifacts(PROJECT_ROOT)
MODEL_FILE = AROUSAL_ARTIFACTS["model"]
FEATURE_CONFIG_FILE = AROUSAL_ARTIFACTS["features"]
THRESHOLD_FILE = AROUSAL_ARTIFACTS["thresholds"]

MANIFEST_FILE = (
    PROCESSED_ROOT
    / "manifest.csv"
)


MERGE_KEYS = [
    "patient_id",
    "epoch_index",
]


DUPLICATE_METADATA_COLUMNS = {
    "original_epoch",
    "start_time",
    "end_time",
    "stage",
    "edf_start_seconds",
    "edf_end_seconds",
    "epoch_signal_valid",
}


def find_patient_row(
    manifest: pd.DataFrame,
    patient_id: str,
) -> pd.Series:
    matched = manifest[
        manifest[
            "patient_id"
        ].astype(str)
        == str(patient_id)
    ]

    if matched.empty:
        raise RuntimeError(
            "Manifest 中找不到患者："
            f"{patient_id}"
        )

    if len(matched) > 1:
        raise RuntimeError(
            "Manifest 中患者 ID 不唯一："
            f"{patient_id}"
        )

    return matched.iloc[0]


def remove_duplicate_metadata(
    data: pd.DataFrame,
) -> pd.DataFrame:
    columns_to_drop = [
        column
        for column in (
            DUPLICATE_METADATA_COLUMNS
        )
        if column in data.columns
    ]

    return data.drop(
        columns=columns_to_drop,
        errors="ignore",
    )


def get_numeric_series(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    安全取得數值欄位。

    若欄位不存在，建立整欄 NaN。
    """
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


def mark_outside_range_as_nan(
    data: pd.DataFrame,
    column: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    """
    將超出合理範圍的數值改為 NaN，
    與訓練資料的清理規則保持一致。
    """
    if column not in data.columns:
        return

    values = pd.to_numeric(
        data[column],
        errors="coerce",
    )

    invalid = pd.Series(
        False,
        index=data.index,
    )

    if minimum is not None:
        invalid |= (
            values.notna()
            & (values < minimum)
        )

    if maximum is not None:
        invalid |= (
            values.notna()
            & (values > maximum)
        )

    data.loc[
        invalid,
        column,
    ] = np.nan


def apply_inference_cleaning(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    套用與 clean_respiratory_features.py
    相同的主要合理值清理規則。
    """
    result = data.copy()

    result.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    respiratory_rate_columns = [
        "flow_respiratory_rate_bpm",
        "thermistor_respiratory_rate_bpm",
        "thorax_respiratory_rate_bpm",
        "abdomen_respiratory_rate_bpm",
    ]

    for column in (
        respiratory_rate_columns
    ):
        mark_outside_range_as_nan(
            data=result,
            column=column,
            minimum=4.0,
            maximum=45.0,
        )

    spo2_level_columns = [
        "spo2_mean",
        "spo2_min",
        "spo2_max",
        "spo2_median",
        "spo2_p05",
        "spo2_p25",
        "spo2_p75",
        "spo2_p95",
    ]

    for column in spo2_level_columns:
        mark_outside_range_as_nan(
            data=result,
            column=column,
            minimum=50.0,
            maximum=100.5,
        )

    spo2_change_columns = [
        "spo2_drop_from_start",
        "spo2_largest_drop",
    ]

    for column in (
        spo2_change_columns
    ):
        mark_outside_range_as_nan(
            data=result,
            column=column,
            minimum=0.0,
            maximum=50.0,
        )

    fraction_columns = [
        column
        for column in result.columns
        if (
            column.endswith(
                "_fraction"
            )
            or column.endswith(
                "_coverage"
            )
        )
    ]

    for column in fraction_columns:
        mark_outside_range_as_nan(
            data=result,
            column=column,
            minimum=0.0,
            maximum=1.0,
        )

    heart_rate_columns = [
        "heart_rate_mean",
        "heart_rate_min",
        "heart_rate_max",
        "heart_rate_median",
        "heart_rate_p05",
        "heart_rate_p25",
        "heart_rate_p75",
        "heart_rate_p95",
    ]

    for column in heart_rate_columns:
        mark_outside_range_as_nan(
            data=result,
            column=column,
            minimum=25.0,
            maximum=220.0,
        )

    mark_outside_range_as_nan(
        data=result,
        column=(
            "thorax_abdomen_"
            "correlation"
        ),
        minimum=-1.0,
        maximum=1.0,
    )

    mark_outside_range_as_nan(
        data=result,
        column=(
            "thorax_abdomen_"
            "lag_seconds"
        ),
        minimum=-5.0,
        maximum=5.0,
    )

    return result


def add_quality_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    建立與訓練資料相同的 8 個品質欄位。
    """
    result = data.copy()

    spo2_valid_fraction = (
        get_numeric_series(
            result,
            "spo2_valid_fraction",
        )
    )

    flow_rate = get_numeric_series(
        result,
        "flow_respiratory_rate_bpm",
    )

    thorax_abdomen_corr = (
        get_numeric_series(
            result,
            (
                "thorax_abdomen_"
                "correlation"
            ),
        )
    )

    heart_rate = get_numeric_series(
        result,
        "heart_rate_mean",
    )

    result[
        "quality_low_spo2_validity"
    ] = (
        spo2_valid_fraction.isna()
        | (
            spo2_valid_fraction
            < 0.80
        )
    )

    result[
        "quality_missing_flow_rate"
    ] = flow_rate.isna()

    result[
        "quality_missing_"
        "thorax_abdomen_correlation"
    ] = thorax_abdomen_corr.isna()

    result[
        "quality_missing_heart_rate"
    ] = heart_rate.isna()

    quality_columns = [
        "quality_low_spo2_validity",
        "quality_missing_flow_rate",
        (
            "quality_missing_"
            "thorax_abdomen_correlation"
        ),
        "quality_missing_heart_rate",
    ]

    result[
        "quality_issue_count"
    ] = (
        result[
            quality_columns
        ]
        .astype(int)
        .sum(axis=1)
    )

    result[
        "quality_any_issue"
    ] = (
        result[
            "quality_issue_count"
        ] > 0
    )

    result[
        "quality_core_features_valid"
    ] = (
        ~result[
            "quality_low_spo2_validity"
        ]
        & ~result[
            "quality_missing_flow_rate"
        ]
        & ~result[
            "quality_missing_heart_rate"
        ]
    )

    result[
        "usable_for_core_"
        "respiratory_model"
    ] = result[
        "quality_core_features_valid"
    ]

    return result


def build_inference_dataset(
    patient_id: str,
    edf_file: str | Path,
    stages_file: str | Path,
) -> pd.DataFrame:
    mapper = ChannelMapper(
        CHANNEL_CONFIG
    )

    stage_builder = (
        StageFeatureBuilder(
            channel_mapper=mapper
        )
    )

    respiratory_builder = (
        RespiratoryFeatureBuilder(
            channel_mapper=mapper
        )
    )

    print("=" * 80)
    print(
        "建立新患者 Stage Features"
    )
    print("=" * 80)

    stage_features = (
        stage_builder
        .build_patient_features(
            patient_id=patient_id,
            edf_file=edf_file,
            stages_file=stages_file,
        )
    )

    print(
        f"Stage Features："
        f"{stage_features.shape}"
    )

    print()
    print("=" * 80)
    print(
        "建立新患者 Respiratory Features"
    )
    print("=" * 80)

    respiratory_features = (
        respiratory_builder
        .build_patient_features(
            patient_id=patient_id,
            edf_file=edf_file,
            stages_file=stages_file,
        )
    )

    print(
        "Respiratory Features："
        f"{respiratory_features.shape}"
    )

    respiratory_for_merge = (
        remove_duplicate_metadata(
            respiratory_features
        )
    )

    merged = stage_features.merge(
        respiratory_for_merge,
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=(
            "",
            "_resp",
        ),
    )

    if len(merged) != len(
        stage_features
    ):
        raise RuntimeError(
            "Stage 與 Respiratory Features "
            "合併後 Epoch 數量不一致："
            f"{len(stage_features)} → "
            f"{len(merged)}"
        )

    merged = apply_inference_cleaning(
        merged
    )

    merged = add_quality_columns(
        merged
    )

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "對已整理患者執行 "
            "Arousal Next 30s 推論"
        )
    )

    parser.add_argument(
        "--patient-id",
        required=True,
        help=(
            "患者資料夾名稱，例如："
            "20211015T231717 - 76b7d"
        ),
    )

    args = parser.parse_args()

    patient_id = str(
        args.patient_id
    )

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"找不到 Manifest："
            f"{MANIFEST_FILE}"
        )

    manifest = pd.read_csv(
        MANIFEST_FILE
    )

    patient_row = find_patient_row(
        manifest,
        patient_id,
    )

    edf_file = Path(
        str(
            patient_row[
                "edf_file"
            ]
        )
    )

    stages_file = (
        PROCESSED_ROOT
        / patient_id
        / "stages_aligned.csv"
    )

    if not edf_file.exists():
        raise FileNotFoundError(
            f"找不到 EDF："
            f"{edf_file}"
        )

    if not stages_file.exists():
        raise FileNotFoundError(
            "找不到 stages_aligned.csv："
            f"{stages_file}"
        )

    inference_data = (
        build_inference_dataset(
            patient_id=patient_id,
            edf_file=edf_file,
            stages_file=stages_file,
        )
    )

    print()
    print("=" * 80)
    print("推論資料建立完成")
    print("=" * 80)

    print(
        f"推論資料形狀："
        f"{inference_data.shape}"
    )

    print(
        "核心特徵品質合格："
        f"{int(inference_data['quality_core_features_valid'].sum())}"
        f"/{len(inference_data)}"
    )

    engine = ArousalInferenceEngine(
        model_file=MODEL_FILE,
        feature_config_file=(
            FEATURE_CONFIG_FILE
        ),
        threshold_file=(
            THRESHOLD_FILE
        ),
    )

    predictions = engine.predict(
        inference_data
    )

    output_folder = (
        INFERENCE_ROOT
        / patient_id
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_folder
        / "arousal_predictions.csv"
    )

    predictions.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    engine.print_summary(
        predictions
    )

    print(
        f"\n推論結果："
        f"{output_file}"
    )

    print(
        "\n最高風險的 20 個 Epoch："
    )

    display_columns = [
        column
        for column in [
            "epoch_index",
            "start_time",
            "stage",
            (
                "quality_core_"
                "features_valid"
            ),
            (
                "arousal_next_30s_"
                "probability"
            ),
            (
                "arousal_next_30s_"
                "alert"
            ),
            "arousal_risk_level",
        ]
        if column
        in predictions.columns
    ]

    print(
        predictions.sort_values(
            (
                "arousal_next_30s_"
                "probability"
            ),
            ascending=False,
        )[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
