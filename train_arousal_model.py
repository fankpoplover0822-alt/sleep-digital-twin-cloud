from __future__ import annotations

import json
from pathlib import Path

import joblib
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
from sklearn.model_selection import (
    StratifiedGroupKFold,
)

from src.models.arousal_model import (
    ArousalModelConfig,
    build_arousal_pipeline,
    sanitize_feature_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "arousal_next_30s"
)

MODEL_FILE = (
    MODEL_DIR
    / "arousal_next_30s.joblib"
)

METRICS_FILE = (
    MODEL_DIR
    / "cross_validation_metrics.csv"
)

PREDICTIONS_FILE = (
    MODEL_DIR
    / "out_of_fold_predictions.csv"
)

FEATURES_FILE = (
    MODEL_DIR
    / "feature_columns.json"
)

SUMMARY_FILE = (
    MODEL_DIR
    / "training_summary.json"
)


TARGET_COLUMN = (
    "predict_arousal_next_30s"
)

GROUP_COLUMN = "patient_id"


EXCLUDED_COLUMNS = {
    # 識別與時間
    "patient_id",
    "epoch_index",
    "original_epoch",
    "start_time",
    "end_time",
    "edf_start_seconds",
    "edf_end_seconds",

    # 所有事件標籤與事後資訊
    "has_hypopnea",
    "has_obstructive_apnea",
    "has_central_apnea",
    "has_mixed_apnea",
    "has_any_apnea",
    "has_apnea_or_hypopnea",
    "has_arousal",
    "has_respiratory_arousal",
    "has_spontaneous_arousal",
    "has_any_respiratory_event",
    "has_any_event",
    "event_count_in_epoch",
    "total_event_overlap_seconds",
    "respiratory_event_count",
    "respiratory_event_overlap_seconds",
    "arousal_count",
    "arousal_overlap_seconds",
    "next_arousal_seconds",
    "next_arousal_after_epoch_seconds",
    "arousal_within_15s",
    "arousal_within_30s",
    "arousal_within_60s",
    "predict_arousal_next_15s",
    "predict_arousal_next_30s",
    "predict_arousal_next_60s",

    # 不應讓模型學到讀檔方式
    "edf_backend",
    "respiratory_edf_backend",

    # source channel 名稱通常是設備資訊
    "flow_source",
    "thermistor_source",
    "spo2_source",
    "thorax_source",
    "abdomen_source",
    "heart_rate_source",
    "position_source",
    "snore_source",
    "eeg_c3_source",
    "eeg_c4_source",
    "eog_left_source",
    "eog_right_source",
}


def to_bool(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
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


def choose_feature_columns(
    data: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    candidate_columns = [
        column
        for column in data.columns
        if column not in EXCLUDED_COLUMNS
        and column != TARGET_COLUMN
    ]

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in candidate_columns:
        if pd.api.types.is_numeric_dtype(
            data[column]
        ):
            numeric_columns.append(column)

        elif pd.api.types.is_bool_dtype(
            data[column]
        ):
            numeric_columns.append(column)

        elif column == "stage":
            categorical_columns.append(
                column
            )

    return (
        numeric_columns,
        categorical_columns,
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"找不到模型資料：{INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE
    )

    if TARGET_COLUMN not in data.columns:
        raise RuntimeError(
            f"找不到目標欄位："
            f"{TARGET_COLUMN}"
        )

    if GROUP_COLUMN not in data.columns:
        raise RuntimeError(
            f"找不到患者分組欄位："
            f"{GROUP_COLUMN}"
        )

    target = to_bool(
        data[TARGET_COLUMN]
    ).astype(int)

    groups = (
        data[GROUP_COLUMN]
        .astype(str)
    )

    numeric_columns, categorical_columns = (
        choose_feature_columns(
            data
        )
    )

    feature_columns = (
        numeric_columns
        + categorical_columns
    )

    if not feature_columns:
        raise RuntimeError(
            "沒有可用的模型特徵。"
        )

    features = sanitize_feature_frame(
        data[feature_columns]
    )

    print("=" * 80)
    print("Arousal Next 30s 模型訓練")
    print("=" * 80)

    print(
        f"資料筆數：{len(data)}"
    )

    print(
        f"患者數量："
        f"{groups.nunique()}"
    )

    print(
        f"正樣本：{int(target.sum())}"
    )

    print(
        f"負樣本："
        f"{int((target == 0).sum())}"
    )

    print(
        f"數值特徵："
        f"{len(numeric_columns)}"
    )

    print(
        f"類別特徵："
        f"{len(categorical_columns)}"
    )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    config = ArousalModelConfig()

    fold_metrics: list[dict] = []

    out_of_fold_probability = np.full(
        len(data),
        np.nan,
        dtype=np.float64,
    )

    out_of_fold_prediction = np.full(
        len(data),
        -1,
        dtype=np.int64,
    )

    for fold_index, (
        train_indices,
        test_indices,
    ) in enumerate(
        splitter.split(
            features,
            target,
            groups,
        ),
        start=1,
    ):
        train_groups = set(
            groups.iloc[
                train_indices
            ]
        )

        test_groups = set(
            groups.iloc[
                test_indices
            ]
        )

        overlap = (
            train_groups
            & test_groups
        )

        if overlap:
            raise RuntimeError(
                "患者資料洩漏："
                f"{sorted(overlap)}"
            )

        model = build_arousal_pipeline(
            numeric_columns=(
                numeric_columns
            ),
            categorical_columns=(
                categorical_columns
            ),
            config=config,
        )

        model.fit(
            features.iloc[
                train_indices
            ],
            target.iloc[
                train_indices
            ],
        )

        probabilities = (
            model.predict_proba(
                features.iloc[
                    test_indices
                ]
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        y_true = target.iloc[
            test_indices
        ].to_numpy()

        out_of_fold_probability[
            test_indices
        ] = probabilities

        out_of_fold_prediction[
            test_indices
        ] = predictions

        fold_result = {
            "fold": fold_index,
            "train_rows": len(
                train_indices
            ),
            "test_rows": len(
                test_indices
            ),
            "train_patients": len(
                train_groups
            ),
            "test_patients": len(
                test_groups
            ),
            "positive_test": int(
                y_true.sum()
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
            "f1": float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
        }

        fold_metrics.append(
            fold_result
        )

        print()
        print(
            f"Fold {fold_index}"
        )

        print(
            f"測試患者："
            f"{sorted(test_groups)}"
        )

        print(
            f"ROC-AUC："
            f"{fold_result['roc_auc']:.4f}"
        )

        print(
            "Average Precision："
            f"{fold_result['average_precision']:.4f}"
        )

        print(
            "Balanced Accuracy："
            f"{fold_result['balanced_accuracy']:.4f}"
        )

        print(
            f"Recall："
            f"{fold_result['recall']:.4f}"
        )

        print(
            f"F1："
            f"{fold_result['f1']:.4f}"
        )

    if np.isnan(
        out_of_fold_probability
    ).any():
        raise RuntimeError(
            "部分資料沒有 OOF 預測。"
        )

    overall_predictions = (
        out_of_fold_probability
        >= 0.5
    ).astype(int)

    overall_metrics = {
        "roc_auc": float(
            roc_auc_score(
                target,
                out_of_fold_probability,
            )
        ),
        "average_precision": float(
            average_precision_score(
                target,
                out_of_fold_probability,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                target,
                overall_predictions,
            )
        ),
        "precision": float(
            precision_score(
                target,
                overall_predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                target,
                overall_predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                target,
                overall_predictions,
                zero_division=0,
            )
        ),
    }

    confusion = confusion_matrix(
        target,
        overall_predictions,
    )

    report_text = classification_report(
        target,
        overall_predictions,
        digits=4,
        zero_division=0,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_frame = pd.DataFrame(
        fold_metrics
    )

    metrics_frame.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    prediction_frame = data[
        [
            "patient_id",
            "epoch_index",
            "stage",
            TARGET_COLUMN,
        ]
    ].copy()

    prediction_frame[
        "predicted_probability"
    ] = out_of_fold_probability

    prediction_frame[
        "predicted_label"
    ] = overall_predictions

    prediction_frame.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    with FEATURES_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "numeric_columns": (
                    numeric_columns
                ),
                "categorical_columns": (
                    categorical_columns
                ),
                "target_column": (
                    TARGET_COLUMN
                ),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    final_model = build_arousal_pipeline(
        numeric_columns=(
            numeric_columns
        ),
        categorical_columns=(
            categorical_columns
        ),
        config=config,
    )

    final_model.fit(
        features,
        target,
    )

    joblib.dump(
        final_model,
        MODEL_FILE,
    )

    summary = {
        "target_column": TARGET_COLUMN,
        "row_count": len(data),
        "patient_count": int(
            groups.nunique()
        ),
        "positive_count": int(
            target.sum()
        ),
        "negative_count": int(
            (target == 0).sum()
        ),
        "numeric_feature_count": len(
            numeric_columns
        ),
        "categorical_feature_count": len(
            categorical_columns
        ),
        "overall_metrics": (
            overall_metrics
        ),
        "confusion_matrix": (
            confusion.tolist()
        ),
        "classification_report": (
            report_text
        ),
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("5-Fold Group Cross Validation 完成")
    print("=" * 80)

    for name, value in (
        overall_metrics.items()
    ):
        print(
            f"{name}：{value:.4f}"
        )

    print("\nConfusion Matrix：")
    print(confusion)

    print("\nClassification Report：")
    print(report_text)

    print(
        f"模型：{MODEL_FILE}"
    )

    print(
        f"Fold Metrics：{METRICS_FILE}"
    )

    print(
        f"OOF Predictions："
        f"{PREDICTIONS_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()