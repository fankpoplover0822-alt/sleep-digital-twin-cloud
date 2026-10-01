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

from predict_new_patient import build_inference_dataset
from src.importers.new_patient_importer import NewPatientImporter


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
    """
    移除 ColumnTransformer 自動加入的前綴，
    讓輸出的欄位名稱較容易閱讀。

    例如：
    numeric__spo2_min -> spo2_min
    categorical__stage_N1 -> stage_N1
    """
    prefixes = [
        "numeric__",
        "categorical__",
        "remainder__",
    ]

    result = str(feature_name)

    for prefix in prefixes:
        if result.startswith(prefix):
            return result[len(prefix):]

    return result


def extract_positive_class_shap(
    raw_shap_values: Any,
    row_count: int,
    feature_count: int,
) -> np.ndarray:
    """
    相容不同 SHAP 版本對二元分類的輸出格式。

    最終回傳：
    shape = (row_count, feature_count)
    代表正類別，也就是 Arousal Next 30s。
    """
    if isinstance(raw_shap_values, list):
        if len(raw_shap_values) < 2:
            values = np.asarray(
                raw_shap_values[0]
            )
        else:
            values = np.asarray(
                raw_shap_values[1]
            )

        return values.reshape(
            row_count,
            feature_count,
        )

    values = np.asarray(
        raw_shap_values
    )

    if values.ndim == 2:
        return values.reshape(
            row_count,
            feature_count,
        )

    if values.ndim == 3:
        # 常見格式：
        # (samples, features, classes)
        if (
            values.shape[0] == row_count
            and values.shape[1] == feature_count
            and values.shape[2] >= 2
        ):
            return values[:, :, 1]

        # 另一種可能格式：
        # (classes, samples, features)
        if (
            values.shape[0] >= 2
            and values.shape[1] == row_count
            and values.shape[2] == feature_count
        ):
            return values[1, :, :]

    raise RuntimeError(
        "無法辨識 SHAP values 格式："
        f"shape={values.shape}"
    )


def extract_positive_expected_value(
    expected_value: Any,
) -> float:
    values = np.asarray(
        expected_value
    ).reshape(-1)

    if values.size >= 2:
        return float(values[1])

    if values.size == 1:
        return float(values[0])

    raise RuntimeError(
        "SHAP expected value 為空。"
    )


def find_target_epoch(
    predictions: pd.DataFrame,
    requested_epoch: int | None,
) -> pd.Series:
    if requested_epoch is None:
        probability = pd.to_numeric(
            predictions[
                "arousal_next_30s_probability"
            ],
            errors="coerce",
        )

        if probability.isna().all():
            raise RuntimeError(
                "預測檔沒有有效機率。"
            )

        row_index = probability.idxmax()

        return predictions.loc[
            row_index
        ]

    matched = predictions[
        pd.to_numeric(
            predictions["epoch_index"],
            errors="coerce",
        )
        == int(requested_epoch)
    ]

    if matched.empty:
        raise RuntimeError(
            f"找不到 epoch_index={requested_epoch}"
        )

    if len(matched) > 1:
        raise RuntimeError(
            f"epoch_index={requested_epoch} 不唯一。"
        )

    return matched.iloc[0]


def to_float_or_nan(
    value: Any,
) -> float:
    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return float("nan")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "使用 SHAP 解釋 incoming 新患者的 "
            "Arousal Next 30s 預測。"
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
        "--epoch-index",
        type=int,
        default=None,
        help=(
            "要解釋的 Epoch。"
            "未指定時，自動選風險最高 Epoch。"
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="輸出前幾個重要特徵，預設 20。",
    )

    args = parser.parse_args()

    patient_id = str(
        args.patient_id
    ).strip()

    top_n = max(
        int(args.top_n),
        1,
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
                f"找不到必要檔案：{file_path}"
            )

    print("=" * 80)
    print("Incoming Arousal SHAP 解釋")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    # ============================================================
    # 1. 尋找 EDF
    # ============================================================
    importer = NewPatientImporter(
        patient_folder=patient_folder,
        demographics_file=(
            DEMOGRAPHICS_FILE
        ),
    )

    patient_files = importer.inspect()

    edf_file = patient_files.edf_file

    # ============================================================
    # 2. 讀取預測，決定要解釋的 Epoch
    # ============================================================
    predictions = pd.read_csv(
        predictions_file
    )

    required_prediction_columns = {
        "epoch_index",
        "arousal_next_30s_probability",
    }

    missing_prediction_columns = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing_prediction_columns:
        raise RuntimeError(
            "arousal_predictions.csv 缺少欄位："
            f"{sorted(missing_prediction_columns)}"
        )

    target_prediction = find_target_epoch(
        predictions=predictions,
        requested_epoch=args.epoch_index,
    )

    target_epoch_index = int(
        target_prediction[
            "epoch_index"
        ]
    )

    target_probability = float(
        target_prediction[
            "arousal_next_30s_probability"
        ]
    )

    print(
        f"解釋 Epoch：{target_epoch_index}"
    )

    print(
        f"模型風險機率："
        f"{target_probability:.6f}"
    )

    if "start_time" in target_prediction.index:
        print(
            f"開始時間："
            f"{target_prediction['start_time']}"
        )

    if "stage" in target_prediction.index:
        print(
            f"睡眠階段："
            f"{target_prediction['stage']}"
        )

    # ============================================================
    # 3. 重建患者完整模型輸入資料
    # ============================================================
    print()
    print("=" * 80)
    print("重新建立患者模型特徵")
    print("=" * 80)

    inference_data = build_inference_dataset(
        patient_id=patient_id,
        edf_file=edf_file,
        stages_file=stages_file,
    )

    epoch_values = pd.to_numeric(
        inference_data[
            "epoch_index"
        ],
        errors="coerce",
    )

    matched_feature_row = inference_data[
        epoch_values
        == target_epoch_index
    ]

    if matched_feature_row.empty:
        raise RuntimeError(
            "特徵資料找不到指定 Epoch："
            f"{target_epoch_index}"
        )

    if len(matched_feature_row) > 1:
        raise RuntimeError(
            "特徵資料中的 Epoch 不唯一："
            f"{target_epoch_index}"
        )

    # ============================================================
    # 4. 載入訓練時特徵設定與模型
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

    model_feature_columns = (
        numeric_columns
        + categorical_columns
    )

    if not model_feature_columns:
        raise RuntimeError(
            "feature_columns.json 沒有特徵欄位。"
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

    if "preprocessor" not in model.named_steps:
        raise RuntimeError(
            "模型 Pipeline 找不到 preprocessor。"
        )

    if "classifier" not in model.named_steps:
        raise RuntimeError(
            "模型 Pipeline 找不到 classifier。"
        )

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    prepared = matched_feature_row.reindex(
        columns=model_feature_columns
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

    # ============================================================
    # 5. 經過模型前處理器轉換
    # ============================================================
    transformed = preprocessor.transform(
        prepared
    )

    if hasattr(
        transformed,
        "toarray",
    ):
        transformed = transformed.toarray()

    transformed = np.asarray(
        transformed,
        dtype=np.float64,
    )

    transformed_feature_names = (
        preprocessor.get_feature_names_out()
    )

    readable_feature_names = [
        normalize_feature_name(
            feature_name
        )
        for feature_name
        in transformed_feature_names
    ]

    if (
        transformed.shape[1]
        != len(readable_feature_names)
    ):
        raise RuntimeError(
            "轉換後特徵數量與特徵名稱數量不一致："
            f"{transformed.shape[1]} vs "
            f"{len(readable_feature_names)}"
        )

    # 確認直接使用 classifier 的機率
    classifier_probability = float(
        classifier.predict_proba(
            transformed
        )[0, 1]
    )

    print(
        f"Pipeline classifier 機率："
        f"{classifier_probability:.6f}"
    )

    # ============================================================
    # 6. 建立 Tree SHAP
    # ============================================================
    print()
    print("=" * 80)
    print("計算 SHAP values")
    print("=" * 80)

    explainer = shap.TreeExplainer(
        classifier
    )

    raw_shap_values = explainer.shap_values(
        transformed
    )

    positive_shap_values = (
        extract_positive_class_shap(
            raw_shap_values=(
                raw_shap_values
            ),
            row_count=1,
            feature_count=(
                transformed.shape[1]
            ),
        )
    )

    expected_value = (
        extract_positive_expected_value(
            explainer.expected_value
        )
    )

    row_shap_values = (
        positive_shap_values[0]
    )

    row_feature_values = (
        transformed[0]
    )

    explanation = shap.Explanation(
        values=row_shap_values,
        base_values=expected_value,
        data=row_feature_values,
        feature_names=(
            readable_feature_names
        ),
    )

    # ============================================================
    # 7. 建立重要特徵表
    # ============================================================
    factors = pd.DataFrame(
        {
            "feature": (
                readable_feature_names
            ),
            "feature_value": (
                row_feature_values
            ),
            "shap_value": (
                row_shap_values
            ),
        }
    )

    factors[
        "absolute_shap_value"
    ] = factors[
        "shap_value"
    ].abs()

    factors[
        "effect_direction"
    ] = np.where(
        factors["shap_value"] > 0,
        "INCREASE_RISK",
        np.where(
            factors["shap_value"] < 0,
            "DECREASE_RISK",
            "NEUTRAL",
        ),
    )

    factors = factors.sort_values(
        "absolute_shap_value",
        ascending=False,
    ).reset_index(
        drop=True
    )

    factors.insert(
        0,
        "rank",
        range(
            1,
            len(factors) + 1,
        ),
    )

    factors.insert(
        0,
        "patient_id",
        patient_id,
    )

    factors.insert(
        1,
        "epoch_index",
        target_epoch_index,
    )

    factors.insert(
        2,
        "predicted_probability",
        target_probability,
    )

    top_factors = factors.head(
        top_n
    ).copy()

    # ============================================================
    # 8. 儲存 SHAP 表格與圖
    # ============================================================
    output_folder = (
        inference_folder
        / "shap"
        / f"epoch_{target_epoch_index}"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    top_factors_file = (
        output_folder
        / "shap_top_factors.csv"
    )

    all_factors_file = (
        output_folder
        / "shap_all_factors.csv"
    )

    waterfall_file = (
        output_folder
        / "shap_waterfall.png"
    )

    bar_file = (
        output_folder
        / "shap_bar.png"
    )

    summary_file = (
        output_folder
        / "shap_summary.json"
    )

    top_factors.to_csv(
        top_factors_file,
        index=False,
        encoding="utf-8-sig",
    )

    factors.to_csv(
        all_factors_file,
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure()
    shap.plots.waterfall(
        explanation,
        max_display=top_n,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        waterfall_file,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    shap.plots.bar(
        explanation,
        max_display=top_n,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        bar_file,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    summary = {
        "patient_id": patient_id,
        "epoch_index": (
            target_epoch_index
        ),
        "start_time": (
            target_prediction.get(
                "start_time"
            )
        ),
        "stage": (
            target_prediction.get(
                "stage"
            )
        ),
        "predicted_probability": (
            target_probability
        ),
        "classifier_probability": (
            classifier_probability
        ),
        "shap_expected_value": (
            expected_value
        ),
        "top_factor_count": (
            len(top_factors)
        ),
        "top_factors": (
            top_factors[
                [
                    "rank",
                    "feature",
                    "feature_value",
                    "shap_value",
                    "effect_direction",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "important_note": (
            "SHAP 顯示模型特徵對此預測的影響，"
            "不代表已證明的生理因果關係。"
        ),
    }

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
            default=str,
        )

    # ============================================================
    # 9. 顯示摘要
    # ============================================================
    print()
    print("=" * 80)
    print("SHAP 解釋完成")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    print(
        f"Epoch：{target_epoch_index}"
    )

    print(
        f"模型機率："
        f"{target_probability:.6f}"
    )

    print(
        f"SHAP 基準值："
        f"{expected_value:.6f}"
    )

    print()
    print(
        f"前 {top_n} 個重要因素："
    )

    print(
        top_factors[
            [
                "rank",
                "feature",
                "feature_value",
                "shap_value",
                "effect_direction",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Top Factors："
        f"{top_factors_file}"
    )

    print(
        f"全部 Factors："
        f"{all_factors_file}"
    )

    print(
        f"Waterfall 圖："
        f"{waterfall_file}"
    )

    print(
        f"Bar 圖："
        f"{bar_file}"
    )

    print(
        f"摘要 JSON："
        f"{summary_file}"
    )

    print()
    print(
        "注意：SHAP 是模型解釋，"
        "不是生理因果證明。"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()