from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

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
    / "patients.xlsx"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "arousal_next_30s"
    / "arousal_next_30s.joblib"
)

FEATURE_CONFIG_FILE = (
    PROJECT_ROOT
    / "models"
    / "arousal_next_30s"
    / "feature_columns.json"
)


def load_json(
    file_path: Path,
) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到 JSON：{file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"JSON 內容不是 object：{file_path}"
        )

    return data


def normalize_feature_name(
    feature_name: str,
) -> str:
    prefixes = [
        "numeric__",
        "categorical__",
        "remainder__",
    ]

    result = str(feature_name)

    for prefix in prefixes:
        if result.startswith(prefix):
            return result[
                len(prefix):
            ]

    return result


def extract_positive_class_shap(
    raw_shap_values: Any,
    row_count: int,
    feature_count: int,
) -> np.ndarray:
    """
    相容不同版本 SHAP 的二元分類輸出格式。

    最終回傳：
    shape = (row_count, feature_count)
    """
    if isinstance(
        raw_shap_values,
        list,
    ):
        if len(raw_shap_values) >= 2:
            values = np.asarray(
                raw_shap_values[1]
            )
        elif len(raw_shap_values) == 1:
            values = np.asarray(
                raw_shap_values[0]
            )
        else:
            raise RuntimeError(
                "SHAP values list 為空。"
            )

        if values.shape == (
            row_count,
            feature_count,
        ):
            return values

    values = np.asarray(
        raw_shap_values
    )

    if values.ndim == 2:
        if values.shape != (
            row_count,
            feature_count,
        ):
            raise RuntimeError(
                "SHAP 二維結果形狀不符："
                f"{values.shape}"
            )

        return values

    if values.ndim == 3:
        # SHAP 0.51 常見：
        # samples × features × classes
        if (
            values.shape[0]
            == row_count
            and values.shape[1]
            == feature_count
            and values.shape[2]
            >= 2
        ):
            return values[:, :, 1]

        # 另一種：
        # classes × samples × features
        if (
            values.shape[0] >= 2
            and values.shape[1]
            == row_count
            and values.shape[2]
            == feature_count
        ):
            return values[1, :, :]

    raise RuntimeError(
        "無法辨識 SHAP values 格式："
        f"{values.shape}"
    )


def classify_feature_group(
    feature_name: str,
) -> str:
    """
    將模型特徵歸入較容易解讀的生理系統。

    這是專案內部的模型特徵分類，
    不是臨床標準分類。
    """
    name = str(
        feature_name
    ).lower()

    if name.startswith(
        "stage_"
    ) or name == "stage":
        return "SLEEP_STAGE"

    if (
        "spo2" in name
        or "saturation" in name
        or "oxygen" in name
        or "desaturation" in name
    ):
        return "SPO2"

    if (
        "flow" in name
        or "nasal" in name
        or "thermistor" in name
    ):
        return "AIRFLOW"

    if (
        "thorax" in name
        or "abdomen" in name
        or "inductance" in name
    ):
        return "RESPIRATORY_EFFORT"

    if (
        "position" in name
        or "posangle" in name
        or "supine" in name
        or "prone" in name
        or "left_side" in name
        or "right_side" in name
    ):
        return "POSITION"

    if (
        "heart_rate" in name
        or name.startswith("hr_")
        or "pulse" in name
        or "ekg" in name
        or "ecg" in name
        or "ptt" in name
        or "pwa" in name
    ):
        return "CARDIOVASCULAR"

    if (
        "snore" in name
        or "audio" in name
    ):
        return "SNORE"

    if (
        "eeg" in name
        or name.startswith("c3")
        or name.startswith("c4")
        or name.startswith("f3")
        or name.startswith("f4")
        or name.startswith("o1")
        or name.startswith("o2")
    ):
        return "EEG"

    if (
        "eog" in name
        or name.startswith("e1")
        or name.startswith("e2")
    ):
        return "EOG"

    if (
        "leg" in name
        or "movement" in name
    ):
        return "MOVEMENT"

    if (
        "quality" in name
        or "missing" in name
        or "valid" in name
        or "backend" in name
        or "coverage" in name
    ):
        return "DATA_QUALITY"

    if (
        name in {
            "age",
            "bmi",
            "sex",
        }
        or "demographic" in name
    ):
        return "DEMOGRAPHICS"

    return "OTHER"


def to_bool_series(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "false": False,
                "0": False,
                "no": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def prepare_model_features(
    inference_data: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    feature_columns = (
        numeric_columns
        + categorical_columns
    )

    prepared = inference_data.reindex(
        columns=feature_columns
    ).copy()

    for column in numeric_columns:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    for column in categorical_columns:
        prepared[column] = (
            prepared[column]
            .astype("object")
        )

    prepared.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return prepared


def aggregate_features(
    feature_names: list[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    mask: np.ndarray,
    analysis_name: str,
) -> pd.DataFrame:
    selected_shap = shap_values[
        mask,
        :,
    ]

    selected_values = feature_values[
        mask,
        :,
    ]

    if selected_shap.shape[0] == 0:
        return pd.DataFrame(
            columns=[
                "analysis",
                "rank",
                "feature",
                "feature_group",
                "mean_absolute_shap",
                "mean_signed_shap",
                "positive_effect_fraction",
                "negative_effect_fraction",
                "mean_feature_value",
                "median_feature_value",
                "epoch_count",
            ]
        )

    mean_absolute_shap = np.nanmean(
        np.abs(selected_shap),
        axis=0,
    )

    mean_signed_shap = np.nanmean(
        selected_shap,
        axis=0,
    )

    positive_effect_fraction = (
        np.mean(
            selected_shap > 0,
            axis=0,
        )
    )

    negative_effect_fraction = (
        np.mean(
            selected_shap < 0,
            axis=0,
        )
    )

    mean_feature_value = np.nanmean(
        selected_values,
        axis=0,
    )

    median_feature_value = np.nanmedian(
        selected_values,
        axis=0,
    )

    frame = pd.DataFrame(
        {
            "analysis": analysis_name,
            "feature": feature_names,
            "feature_group": [
                classify_feature_group(
                    feature
                )
                for feature in feature_names
            ],
            "mean_absolute_shap": (
                mean_absolute_shap
            ),
            "mean_signed_shap": (
                mean_signed_shap
            ),
            "positive_effect_fraction": (
                positive_effect_fraction
            ),
            "negative_effect_fraction": (
                negative_effect_fraction
            ),
            "mean_feature_value": (
                mean_feature_value
            ),
            "median_feature_value": (
                median_feature_value
            ),
            "epoch_count": int(
                selected_shap.shape[0]
            ),
        }
    )

    frame = frame.sort_values(
        "mean_absolute_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )

    frame.insert(
        1,
        "rank",
        range(
            1,
            len(frame) + 1,
        ),
    )

    return frame


def aggregate_groups(
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    if feature_frame.empty:
        return pd.DataFrame(
            columns=[
                "analysis",
                "rank",
                "feature_group",
                "total_mean_absolute_shap",
                "mean_feature_absolute_shap",
                "total_mean_signed_shap",
                "feature_count",
                "epoch_count",
            ]
        )

    grouped = (
        feature_frame.groupby(
            [
                "analysis",
                "feature_group",
            ],
            as_index=False,
        )
        .agg(
            total_mean_absolute_shap=(
                "mean_absolute_shap",
                "sum",
            ),
            mean_feature_absolute_shap=(
                "mean_absolute_shap",
                "mean",
            ),
            total_mean_signed_shap=(
                "mean_signed_shap",
                "sum",
            ),
            feature_count=(
                "feature",
                "count",
            ),
            epoch_count=(
                "epoch_count",
                "max",
            ),
        )
    )

    result_frames: list[
        pd.DataFrame
    ] = []

    for analysis_name, part in (
        grouped.groupby(
            "analysis",
            sort=False,
        )
    ):
        part = part.sort_values(
            "total_mean_absolute_shap",
            ascending=False,
        ).reset_index(
            drop=True
        )

        part.insert(
            1,
            "rank",
            range(
                1,
                len(part) + 1,
            ),
        )

        result_frames.append(
            part
        )

    return pd.concat(
        result_frames,
        ignore_index=True,
    )


def create_feature_bar_chart(
    data: pd.DataFrame,
    output_file: Path,
    title: str,
    top_n: int,
) -> None:
    plot_data = (
        data.head(top_n)
        .sort_values(
            "mean_absolute_shap",
            ascending=True,
        )
    )

    if plot_data.empty:
        return

    plt.figure(
        figsize=(10, max(
            6,
            len(plot_data) * 0.35,
        ))
    )

    plt.barh(
        plot_data["feature"],
        plot_data[
            "mean_absolute_shap"
        ],
    )

    plt.xlabel(
        "Mean absolute SHAP value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def create_group_bar_chart(
    data: pd.DataFrame,
    output_file: Path,
    title: str,
) -> None:
    plot_data = data.sort_values(
        "total_mean_absolute_shap",
        ascending=True,
    )

    if plot_data.empty:
        return

    plt.figure(
        figsize=(9, max(
            5,
            len(plot_data) * 0.45,
        ))
    )

    plt.barh(
        plot_data[
            "feature_group"
        ],
        plot_data[
            "total_mean_absolute_shap"
        ],
    )

    plt.xlabel(
        "Total mean absolute SHAP"
    )

    plt.ylabel(
        "Physiological feature group"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


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
        if np.isnan(value):
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

    try:
        if pd.isna(value):
            return None
    except (
        TypeError,
        ValueError,
    ):
        pass

    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "彙整 incoming 新患者整晚的 "
            "SHAP 特徵與生理系統重要度。"
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

    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help=(
            "圖表與終端顯示的特徵數量，"
            "預設 25。"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help=(
            "SHAP 每批計算 Epoch 數，"
            "預設 100。"
        ),
    )

    args = parser.parse_args()

    patient_id = str(
        args.patient_id
    ).strip()

    top_n = max(
        1,
        int(args.top_n),
    )

    batch_size = max(
        1,
        int(args.batch_size),
    )

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

    predictions_file = (
        inference_folder
        / "arousal_predictions.csv"
    )

    required_files = [
        MODEL_FILE,
        FEATURE_CONFIG_FILE,
        stages_file,
        predictions_file,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"找不到必要檔案："
                f"{file_path}"
            )

    print("=" * 80)
    print("Patient-Level SHAP Aggregation")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    # ============================================================
    # 1. 尋找患者 EDF
    # ============================================================
    importer = NewPatientImporter(
        patient_folder=patient_folder,
        demographics_file=(
            DEMOGRAPHICS_FILE
        ),
    )

    patient_files = importer.inspect()

    # ============================================================
    # 2. 讀取預測
    # ============================================================
    predictions = pd.read_csv(
        predictions_file
    )

    required_prediction_columns = {
        "epoch_index",
        "arousal_next_30s_probability",
        "arousal_next_30s_alert",
    }

    missing_prediction_columns = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing_prediction_columns:
        raise RuntimeError(
            "預測檔缺少欄位："
            f"{sorted(missing_prediction_columns)}"
        )

    predictions[
        "epoch_index"
    ] = pd.to_numeric(
        predictions["epoch_index"],
        errors="coerce",
    )

    predictions[
        "arousal_next_30s_probability"
    ] = pd.to_numeric(
        predictions[
            "arousal_next_30s_probability"
        ],
        errors="coerce",
    )

    predictions[
        "arousal_next_30s_alert"
    ] = to_bool_series(
        predictions[
            "arousal_next_30s_alert"
        ]
    )

    if predictions[
        "epoch_index"
    ].isna().any():
        raise RuntimeError(
            "預測檔 epoch_index 含 NaN。"
        )

    # ============================================================
    # 3. 重建整晚推論資料
    # ============================================================
    print()
    print("=" * 80)
    print("重新建立整晚患者模型特徵")
    print("=" * 80)

    inference_data = (
        build_inference_dataset(
            patient_id=patient_id,
            edf_file=(
                patient_files.edf_file
            ),
            stages_file=stages_file,
        )
    )

    inference_data[
        "epoch_index"
    ] = pd.to_numeric(
        inference_data[
            "epoch_index"
        ],
        errors="coerce",
    )

    if inference_data[
        "epoch_index"
    ].isna().any():
        raise RuntimeError(
            "推論資料 epoch_index 含 NaN。"
        )

    inference_data = (
        inference_data.sort_values(
            "epoch_index"
        )
        .reset_index(
            drop=True
        )
    )

    predictions = (
        predictions.sort_values(
            "epoch_index"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(inference_data)
        != len(predictions)
    ):
        raise RuntimeError(
            "推論特徵與預測數量不同："
            f"{len(inference_data)} vs "
            f"{len(predictions)}"
        )

    if not np.array_equal(
        inference_data[
            "epoch_index"
        ].to_numpy(),
        predictions[
            "epoch_index"
        ].to_numpy(),
    ):
        raise RuntimeError(
            "推論特徵與預測的 Epoch "
            "順序不一致。"
        )

    # ============================================================
    # 4. 載入模型與前處理器
    # ============================================================
    feature_config = load_json(
        FEATURE_CONFIG_FILE
    )

    numeric_columns = list(
        feature_config.get(
            "numeric_columns",
            [],
        )
    )

    categorical_columns = list(
        feature_config.get(
            "categorical_columns",
            [],
        )
    )

    prepared = prepare_model_features(
        inference_data=inference_data,
        numeric_columns=numeric_columns,
        categorical_columns=(
            categorical_columns
        ),
    )

    model = joblib.load(
        MODEL_FILE
    )

    if not hasattr(
        model,
        "named_steps",
    ):
        raise RuntimeError(
            "模型不是 sklearn Pipeline。"
        )

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    transformed = (
        preprocessor.transform(
            prepared
        )
    )

    if hasattr(
        transformed,
        "toarray",
    ):
        transformed = (
            transformed.toarray()
        )

    transformed = np.asarray(
        transformed,
        dtype=np.float64,
    )

    transformed_feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    readable_feature_names = [
        normalize_feature_name(
            name
        )
        for name
        in transformed_feature_names
    ]

    if (
        transformed.shape[1]
        != len(readable_feature_names)
    ):
        raise RuntimeError(
            "轉換後特徵數與名稱數不一致。"
        )

    print(
        f"Epoch 數量："
        f"{transformed.shape[0]}"
    )

    print(
        f"轉換後特徵數："
        f"{transformed.shape[1]}"
    )

    # ============================================================
    # 5. 分批計算整晚 SHAP
    # ============================================================
    print()
    print("=" * 80)
    print("分批計算整晚 SHAP")
    print("=" * 80)

    explainer = shap.TreeExplainer(
        classifier
    )

    shap_batches: list[
        np.ndarray
    ] = []

    total_rows = transformed.shape[0]

    for start_index in range(
        0,
        total_rows,
        batch_size,
    ):
        end_index = min(
            start_index + batch_size,
            total_rows,
        )

        batch = transformed[
            start_index:end_index,
            :,
        ]

        raw_shap_values = (
            explainer.shap_values(
                batch
            )
        )

        positive_values = (
            extract_positive_class_shap(
                raw_shap_values=(
                    raw_shap_values
                ),
                row_count=len(batch),
                feature_count=(
                    transformed.shape[1]
                ),
            )
        )

        shap_batches.append(
            positive_values
        )

        print(
            f"SHAP："
            f"{end_index}/{total_rows}"
        )

    shap_values = np.vstack(
        shap_batches
    )

    if shap_values.shape != (
        transformed.shape[0],
        transformed.shape[1],
    ):
        raise RuntimeError(
            "SHAP 最終形狀不一致："
            f"{shap_values.shape}"
        )

    # ============================================================
    # 6. 建立分析群組
    # ============================================================
    alert_mask = (
        predictions[
            "arousal_next_30s_alert"
        ]
        .to_numpy(
            dtype=bool
        )
    )

    non_alert_mask = (
        ~alert_mask
    )

    high_probability_mask = (
        predictions[
            "arousal_next_30s_probability"
        ]
        .to_numpy(
            dtype=float
        )
        >= 0.50
    )

    all_mask = np.ones(
        len(predictions),
        dtype=bool,
    )

    feature_frames = [
        aggregate_features(
            feature_names=(
                readable_feature_names
            ),
            feature_values=(
                transformed
            ),
            shap_values=shap_values,
            mask=all_mask,
            analysis_name="ALL_EPOCHS",
        ),
        aggregate_features(
            feature_names=(
                readable_feature_names
            ),
            feature_values=(
                transformed
            ),
            shap_values=shap_values,
            mask=alert_mask,
            analysis_name="ALERT_EPOCHS",
        ),
        aggregate_features(
            feature_names=(
                readable_feature_names
            ),
            feature_values=(
                transformed
            ),
            shap_values=shap_values,
            mask=non_alert_mask,
            analysis_name=(
                "NON_ALERT_EPOCHS"
            ),
        ),
        aggregate_features(
            feature_names=(
                readable_feature_names
            ),
            feature_values=(
                transformed
            ),
            shap_values=shap_values,
            mask=(
                high_probability_mask
            ),
            analysis_name=(
                "PROBABILITY_GE_0_50"
            ),
        ),
    ]

    feature_importance = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    group_importance = (
        aggregate_groups(
            feature_importance
        )
    )

    # ============================================================
    # 7. 建立每個 Epoch 的群組 SHAP
    # ============================================================
    group_names = [
        classify_feature_group(
            feature
        )
        for feature
        in readable_feature_names
    ]

    unique_groups = sorted(
        set(group_names)
    )

    epoch_group_data = pd.DataFrame(
        {
            "patient_id": patient_id,
            "epoch_index": (
                inference_data[
                    "epoch_index"
                ].astype(int)
            ),
            "start_time": (
                inference_data.get(
                    "start_time",
                    pd.Series(
                        index=(
                            inference_data
                            .index
                        ),
                        dtype="object",
                    ),
                )
            ),
            "stage": (
                inference_data.get(
                    "stage",
                    pd.Series(
                        index=(
                            inference_data
                            .index
                        ),
                        dtype="object",
                    ),
                )
            ),
            "predicted_probability": (
                predictions[
                    "arousal_next_30s_probability"
                ]
            ),
            "alert": alert_mask,
        }
    )

    for group_name in unique_groups:
        column_indices = [
            index
            for index, current_group
            in enumerate(group_names)
            if current_group
            == group_name
        ]

        group_shap = shap_values[
            :,
            column_indices,
        ]

        epoch_group_data[
            f"{group_name}_signed_shap"
        ] = np.sum(
            group_shap,
            axis=1,
        )

        epoch_group_data[
            f"{group_name}_absolute_shap"
        ] = np.sum(
            np.abs(group_shap),
            axis=1,
        )

    # ============================================================
    # 8. 儲存 CSV
    # ============================================================
    output_folder = (
        inference_folder
        / "shap"
        / "patient_level"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_file = (
        output_folder
        / "patient_feature_importance.csv"
    )

    group_file = (
        output_folder
        / "patient_group_importance.csv"
    )

    epoch_group_file = (
        output_folder
        / "epoch_group_shap.csv"
    )

    feature_importance.to_csv(
        feature_file,
        index=False,
        encoding="utf-8-sig",
    )

    group_importance.to_csv(
        group_file,
        index=False,
        encoding="utf-8-sig",
    )

    epoch_group_data.to_csv(
        epoch_group_file,
        index=False,
        encoding="utf-8-sig",
    )

    # ============================================================
    # 9. 建立圖表
    # ============================================================
    all_features = (
        feature_importance[
            feature_importance[
                "analysis"
            ] == "ALL_EPOCHS"
        ]
    )

    alert_features = (
        feature_importance[
            feature_importance[
                "analysis"
            ] == "ALERT_EPOCHS"
        ]
    )

    all_groups = (
        group_importance[
            group_importance[
                "analysis"
            ] == "ALL_EPOCHS"
        ]
    )

    alert_groups = (
        group_importance[
            group_importance[
                "analysis"
            ] == "ALERT_EPOCHS"
        ]
    )

    all_feature_chart = (
        output_folder
        / "all_epochs_top_features.png"
    )

    alert_feature_chart = (
        output_folder
        / "alert_epochs_top_features.png"
    )

    all_group_chart = (
        output_folder
        / "all_epochs_feature_groups.png"
    )

    alert_group_chart = (
        output_folder
        / "alert_epochs_feature_groups.png"
    )

    create_feature_bar_chart(
        data=all_features,
        output_file=(
            all_feature_chart
        ),
        title=(
            "All Epochs: Top SHAP Features"
        ),
        top_n=top_n,
    )

    create_feature_bar_chart(
        data=alert_features,
        output_file=(
            alert_feature_chart
        ),
        title=(
            "Alert Epochs: Top SHAP Features"
        ),
        top_n=top_n,
    )

    create_group_bar_chart(
        data=all_groups,
        output_file=all_group_chart,
        title=(
            "All Epochs: Physiological "
            "Feature Groups"
        ),
    )

    create_group_bar_chart(
        data=alert_groups,
        output_file=alert_group_chart,
        title=(
            "Alert Epochs: Physiological "
            "Feature Groups"
        ),
    )

    # ============================================================
    # 10. 建立 JSON 摘要
    # ============================================================
    top_all_groups = (
        all_groups.head(10)
    )

    top_alert_groups = (
        alert_groups.head(10)
    )

    top_all_features = (
        all_features.head(top_n)
    )

    top_alert_features = (
        alert_features.head(top_n)
    )

    summary = {
        "patient_id": patient_id,
        "epoch_count": int(
            len(inference_data)
        ),
        "alert_epoch_count": int(
            alert_mask.sum()
        ),
        "non_alert_epoch_count": int(
            non_alert_mask.sum()
        ),
        "probability_ge_0_50_count": int(
            high_probability_mask.sum()
        ),
        "sex": safe_json_value(
            patient_files.sex
        ),
        "age": safe_json_value(
            patient_files.age
        ),
        "BMI": safe_json_value(
            patient_files.bmi
        ),
        "top_groups_all_epochs": (
            top_all_groups[
                [
                    "rank",
                    "feature_group",
                    (
                        "total_mean_"
                        "absolute_shap"
                    ),
                    (
                        "total_mean_"
                        "signed_shap"
                    ),
                    "feature_count",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "top_groups_alert_epochs": (
            top_alert_groups[
                [
                    "rank",
                    "feature_group",
                    (
                        "total_mean_"
                        "absolute_shap"
                    ),
                    (
                        "total_mean_"
                        "signed_shap"
                    ),
                    "feature_count",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "top_features_all_epochs": (
            top_all_features[
                [
                    "rank",
                    "feature",
                    "feature_group",
                    "mean_absolute_shap",
                    "mean_signed_shap",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "top_features_alert_epochs": (
            top_alert_features[
                [
                    "rank",
                    "feature",
                    "feature_group",
                    "mean_absolute_shap",
                    "mean_signed_shap",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "important_note": (
            "此彙整代表模型對患者資料的"
            "特徵敏感度與關聯性，"
            "不代表生理因素的因果效應。"
        ),
    }

    summary_file = (
        output_folder
        / "patient_shap_summary.json"
    )

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=safe_json_value,
        )

    # ============================================================
    # 11. 顯示結果
    # ============================================================
    print()
    print("=" * 80)
    print("Patient-Level SHAP 完成")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    print(
        f"全部 Epoch："
        f"{len(inference_data)}"
    )

    print(
        f"警報 Epoch："
        f"{int(alert_mask.sum())}"
    )

    print()
    print("整晚生理特徵群組排名：")

    print(
        top_all_groups[
            [
                "rank",
                "feature_group",
                (
                    "total_mean_"
                    "absolute_shap"
                ),
                (
                    "total_mean_"
                    "signed_shap"
                ),
                "feature_count",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("警報 Epoch 生理特徵群組排名：")

    print(
        top_alert_groups[
            [
                "rank",
                "feature_group",
                (
                    "total_mean_"
                    "absolute_shap"
                ),
                (
                    "total_mean_"
                    "signed_shap"
                ),
                "feature_count",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"警報 Epoch 前 {top_n} 個特徵："
    )

    print(
        top_alert_features[
            [
                "rank",
                "feature",
                "feature_group",
                "mean_absolute_shap",
                "mean_signed_shap",
                "positive_effect_fraction",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"特徵排名：{feature_file}"
    )

    print(
        f"群組排名：{group_file}"
    )

    print(
        f"Epoch 群組 SHAP："
        f"{epoch_group_file}"
    )

    print(
        f"摘要 JSON：{summary_file}"
    )

    print(
        f"整晚特徵圖："
        f"{all_feature_chart}"
    )

    print(
        f"警報特徵圖："
        f"{alert_feature_chart}"
    )

    print(
        f"整晚群組圖："
        f"{all_group_chart}"
    )

    print(
        f"警報群組圖："
        f"{alert_group_chart}"
    )

    print()
    print(
        "注意：SHAP 是模型關聯解釋，"
        "不是生理因果證明。"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
