from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.event_label_builder import (
    EventLabelBuilder,
)


PROJECT_ROOT = Path(__file__).resolve().parent

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


TARGET_COLUMN = (
    "predict_arousal_next_30s"
)

PROBABILITY_COLUMN = (
    "arousal_next_30s_probability"
)

ALERT_COLUMN = (
    "arousal_next_30s_alert"
)


MERGE_KEYS = [
    "patient_id",
    "epoch_index",
]


def to_boolean_series(
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


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0:
        return float("nan")

    return float(
        numerator / denominator
    )


def to_json_safe(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32,
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

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat(
            sep=" "
        )

    return value


def save_json(
    output_path: Path,
    data: dict[str, Any],
) -> None:
    safe_data = {
        key: to_json_safe(value)
        for key, value
        in data.items()
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            safe_data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def validate_unique_keys(
    data: pd.DataFrame,
    name: str,
) -> None:
    duplicated = data.duplicated(
        MERGE_KEYS,
        keep=False,
    )

    if not duplicated.any():
        return

    examples = data.loc[
        duplicated,
        MERGE_KEYS,
    ].head(10)

    raise RuntimeError(
        f"{name} 的 patient_id + epoch_index "
        f"不是唯一鍵：\n"
        f"{examples.to_string(index=False)}"
    )


def build_true_labels(
    patient_id: str,
    processed_folder: Path,
) -> pd.DataFrame:
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

    builder = EventLabelBuilder()

    labels = (
        builder.build_patient_labels(
            patient_id=patient_id,
            stages_file=stages_file,
            events_file=events_file,
        )
    )

    if labels.empty:
        raise RuntimeError(
            "真實 Event Label 建立結果為空。"
        )

    if TARGET_COLUMN not in labels.columns:
        raise RuntimeError(
            "建立的 Event Label 缺少："
            f"{TARGET_COLUMN}"
        )

    return labels


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    negative_predictive_value = (
        safe_divide(
            tn,
            tn + fn,
        )
    )

    prevalence = float(
        np.mean(y_true)
    )

    predicted_positive_fraction = float(
        np.mean(predictions)
    )

    result = {
        "row_count": int(
            len(y_true)
        ),
        "positive_count": int(
            y_true.sum()
        ),
        "negative_count": int(
            (y_true == 0).sum()
        ),
        "prevalence": prevalence,
        "predicted_positive_count": int(
            predictions.sum()
        ),
        "predicted_positive_fraction": (
            predicted_positive_fraction
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": specificity,
        "negative_predictive_value": (
            negative_predictive_value
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return result


def build_event_episode_summary(
    merged: pd.DataFrame,
) -> pd.DataFrame:
    """
    將連續的正標籤 Epoch 合併為一個未來 Arousal episode，
    檢查每個 episode 是否至少被模型警報一次。

    這不是取代逐 Epoch 指標，而是補充事件層級結果。
    """
    data = merged.sort_values(
        "epoch_index"
    ).reset_index(
        drop=True
    ).copy()

    target = to_boolean_series(
        data[TARGET_COLUMN]
    )

    episode_id = np.full(
        len(data),
        np.nan,
        dtype=float,
    )

    current_episode = -1
    previous_positive_index: int | None = None

    for row_index, is_positive in enumerate(
        target.to_numpy()
    ):
        if not is_positive:
            previous_positive_index = None
            continue

        epoch_index = int(
            data.iloc[
                row_index
            ]["epoch_index"]
        )

        if (
            previous_positive_index
            is None
            or epoch_index
            != previous_positive_index + 1
        ):
            current_episode += 1

        episode_id[
            row_index
        ] = current_episode

        previous_positive_index = (
            epoch_index
        )

    data[
        "true_arousal_episode_id"
    ] = episode_id

    positive_data = data[
        data[
            "true_arousal_episode_id"
        ].notna()
    ].copy()

    if positive_data.empty:
        return pd.DataFrame(
            columns=[
                "episode_id",
                "start_epoch_index",
                "end_epoch_index",
                "epoch_count",
                "start_time",
                "end_time",
                "maximum_probability",
                "any_alert",
            ]
        )

    positive_data[
        "true_arousal_episode_id"
    ] = (
        positive_data[
            "true_arousal_episode_id"
        ]
        .astype(int)
    )

    grouped = (
        positive_data.groupby(
            "true_arousal_episode_id",
            sort=True,
        )
        .agg(
            start_epoch_index=(
                "epoch_index",
                "min",
            ),
            end_epoch_index=(
                "epoch_index",
                "max",
            ),
            epoch_count=(
                "epoch_index",
                "size",
            ),
            start_time=(
                "start_time",
                "min",
            ),
            end_time=(
                "end_time",
                "max",
            ),
            maximum_probability=(
                PROBABILITY_COLUMN,
                "max",
            ),
            any_alert=(
                ALERT_COLUMN,
                "max",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "true_arousal_episode_id": (
                    "episode_id"
                )
            }
        )
    )

    grouped[
        "any_alert"
    ] = to_boolean_series(
        grouped[
            "any_alert"
        ]
    )

    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "使用 Event Grid 真實標籤，"
            "評估 incoming 新患者的 "
            "Arousal Next 30s 預測表現。"
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

    processed_folder = (
        PROCESSED_ROOT
        / patient_id
    )

    inference_folder = (
        INFERENCE_ROOT
        / patient_id
    )

    predictions_file = (
        inference_folder
        / "arousal_predictions.csv"
    )

    if not predictions_file.exists():
        raise FileNotFoundError(
            "找不到模型預測檔："
            f"{predictions_file}\n"
            "請先執行 predict_incoming_patient.py。"
        )

    print("=" * 80)
    print("Incoming 新患者 Arousal 真實驗證")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    print(
        f"預測檔：{predictions_file}"
    )

    # ============================================================
    # 1. 建立真實未來 Arousal 標籤
    # ============================================================
    print()
    print("=" * 80)
    print("步驟 1：建立真實 Arousal 標籤")
    print("=" * 80)

    labels = build_true_labels(
        patient_id=patient_id,
        processed_folder=(
            processed_folder
        ),
    )

    print(
        f"真實標籤 Epoch："
        f"{len(labels)}"
    )

    print(
        "未來 30 秒內真實 Arousal："
        f"{int(to_boolean_series(labels[TARGET_COLUMN]).sum())}"
    )

    # ============================================================
    # 2. 讀取模型預測
    # ============================================================
    print()
    print("=" * 80)
    print("步驟 2：讀取模型預測")
    print("=" * 80)

    predictions = pd.read_csv(
        predictions_file
    )

    required_prediction_columns = {
        "patient_id",
        "epoch_index",
        PROBABILITY_COLUMN,
        ALERT_COLUMN,
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
        PROBABILITY_COLUMN
    ] = pd.to_numeric(
        predictions[
            PROBABILITY_COLUMN
        ],
        errors="coerce",
    )

    if predictions[
        PROBABILITY_COLUMN
    ].isna().any():
        raise RuntimeError(
            "模型機率欄位含有 NaN。"
        )

    predictions[
        ALERT_COLUMN
    ] = to_boolean_series(
        predictions[
            ALERT_COLUMN
        ]
    )

    validate_unique_keys(
        labels,
        "Event Labels",
    )

    validate_unique_keys(
        predictions,
        "Predictions",
    )

    # ============================================================
    # 3. 合併真實標籤與預測
    # ============================================================
    label_columns = [
        "patient_id",
        "epoch_index",
        "original_epoch",
        "start_time",
        "end_time",
        "stage",
        TARGET_COLUMN,
        "next_arousal_after_epoch_seconds",
        "has_arousal",
        "has_respiratory_arousal",
        "has_spontaneous_arousal",
    ]

    label_columns = [
        column
        for column in label_columns
        if column in labels.columns
    ]

    prediction_columns = [
        "patient_id",
        "epoch_index",
        PROBABILITY_COLUMN,
        ALERT_COLUMN,
        "arousal_risk_level",
        "quality_core_features_valid",
    ]

    prediction_columns = [
        column
        for column in prediction_columns
        if column in predictions.columns
    ]

    merged = labels[
        label_columns
    ].merge(
        predictions[
            prediction_columns
        ],
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(labels):
        raise RuntimeError(
            "真實標籤與預測合併後 Epoch 數量不一致："
            f"{len(labels)} → {len(merged)}"
        )

    merged[
        TARGET_COLUMN
    ] = to_boolean_series(
        merged[
            TARGET_COLUMN
        ]
    )

    merged[
        ALERT_COLUMN
    ] = to_boolean_series(
        merged[
            ALERT_COLUMN
        ]
    )

    y_true = (
        merged[
            TARGET_COLUMN
        ]
        .astype(int)
        .to_numpy()
    )

    probabilities = (
        merged[
            PROBABILITY_COLUMN
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    model_predictions = (
        merged[
            ALERT_COLUMN
        ]
        .astype(int)
        .to_numpy()
    )

    if np.unique(y_true).size < 2:
        raise RuntimeError(
            "此患者的真實標籤只有單一類別，"
            "無法計算 ROC-AUC。"
        )

    # ============================================================
    # 4. 計算逐 Epoch 指標
    # ============================================================
    metrics = calculate_metrics(
        y_true=y_true,
        probabilities=probabilities,
        predictions=model_predictions,
    )

    classification_text = (
        classification_report(
            y_true,
            model_predictions,
            digits=4,
            zero_division=0,
        )
    )

    # ============================================================
    # 5. 建立事件 Episode 補充分析
    # ============================================================
    episode_summary = (
        build_event_episode_summary(
            merged
        )
    )

    if episode_summary.empty:
        episode_count = 0
        detected_episode_count = 0
        episode_recall = float(
            "nan"
        )
    else:
        episode_count = len(
            episode_summary
        )

        detected_episode_count = int(
            episode_summary[
                "any_alert"
            ].sum()
        )

        episode_recall = safe_divide(
            detected_episode_count,
            episode_count,
        )

    metrics[
        "true_arousal_episode_count"
    ] = episode_count

    metrics[
        "detected_arousal_episode_count"
    ] = detected_episode_count

    metrics[
        "episode_recall"
    ] = episode_recall

    # ============================================================
    # 6. 儲存結果
    # ============================================================
    inference_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged_file = (
        inference_folder
        / "arousal_evaluation_rows.csv"
    )

    metrics_file = (
        inference_folder
        / "arousal_evaluation_metrics.csv"
    )

    metrics_json_file = (
        inference_folder
        / "arousal_evaluation_metrics.json"
    )

    episodes_file = (
        inference_folder
        / "arousal_evaluation_episodes.csv"
    )

    merged.to_csv(
        merged_file,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        metrics_file,
        index=False,
        encoding="utf-8-sig",
    )

    episode_summary.to_csv(
        episodes_file,
        index=False,
        encoding="utf-8-sig",
    )

    json_summary = {
        "patient_id": patient_id,
        **metrics,
        "classification_report": (
            classification_text
        ),
        "important_note": (
            "這是單一獨立患者的外部測試結果；"
            "不能單獨代表整體臨床效能。"
        ),
    }

    save_json(
        metrics_json_file,
        json_summary,
    )

    # ============================================================
    # 7. 顯示結果
    # ============================================================
    print()
    print("=" * 80)
    print("獨立新患者評估結果")
    print("=" * 80)

    print(
        f"Epoch："
        f"{metrics['row_count']}"
    )

    print(
        f"真實正樣本："
        f"{metrics['positive_count']}"
    )

    print(
        f"真實正樣本比例："
        f"{metrics['prevalence']:.2%}"
    )

    print(
        f"模型警報："
        f"{metrics['predicted_positive_count']}"
    )

    print(
        f"模型警報比例："
        f"{metrics['predicted_positive_fraction']:.2%}"
    )

    print()
    print(
        f"ROC-AUC："
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        "Average Precision："
        f"{metrics['average_precision']:.4f}"
    )

    print(
        "Balanced Accuracy："
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision："
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall："
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Specificity："
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"F1："
        f"{metrics['f1']:.4f}"
    )

    print()
    print("Confusion Matrix：")

    print(
        np.array(
            [
                [
                    metrics[
                        "true_negative"
                    ],
                    metrics[
                        "false_positive"
                    ],
                ],
                [
                    metrics[
                        "false_negative"
                    ],
                    metrics[
                        "true_positive"
                    ],
                ],
            ]
        )
    )

    print()
    print(
        f"真實 Arousal episodes："
        f"{episode_count}"
    )

    print(
        f"至少被警報一次的 episodes："
        f"{detected_episode_count}"
    )

    if np.isfinite(
        episode_recall
    ):
        print(
            f"Episode Recall："
            f"{episode_recall:.4f}"
        )
    else:
        print(
            "Episode Recall：無法計算"
        )

    print()
    print("Classification Report：")

    print(
        classification_text
    )

    print(
        f"逐 Epoch 結果："
        f"{merged_file}"
    )

    print(
        f"指標 CSV："
        f"{metrics_file}"
    )

    print(
        f"指標 JSON："
        f"{metrics_json_file}"
    )

    print(
        f"Episode 結果："
        f"{episodes_file}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()