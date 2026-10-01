from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "models"
    / "arousal_next_30s"
    / "out_of_fold_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "models"
    / "arousal_next_30s"
    / "threshold_metrics.csv"
)

SELECTED_FILE = (
    PROJECT_ROOT
    / "models"
    / "arousal_next_30s"
    / "selected_thresholds.json"
)

TARGET_COLUMN = "predict_arousal_next_30s"
HIGH_SENSITIVITY_RECALL_TARGET = 0.90


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


def main() -> None:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"找不到 OOF 預測：{PREDICTIONS_FILE}"
        )

    data = pd.read_csv(PREDICTIONS_FILE)

    required_columns = {
        TARGET_COLUMN,
        "predicted_probability",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise RuntimeError(
            f"OOF 預測檔缺少欄位：{sorted(missing)}"
        )

    y_true = to_bool(
        data[TARGET_COLUMN]
    ).astype(int).to_numpy()

    probabilities = pd.to_numeric(
        data["predicted_probability"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    if np.isnan(probabilities).any():
        raise RuntimeError(
            "predicted_probability 含有 NaN。"
        )

    rows: list[dict] = []

    thresholds = np.arange(0.01, 0.991, 0.001)

    for threshold in thresholds:
        prediction = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1],
        ).ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        negative_predictive_value = (
            tn / (tn + fn)
            if (tn + fn) > 0
            else np.nan
        )

        rows.append(
            {
                "threshold": float(threshold),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "precision": float(
                    precision_score(
                        y_true,
                        prediction,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_true,
                        prediction,
                        zero_division=0,
                    )
                ),
                "specificity": float(specificity),
                "negative_predictive_value": float(
                    negative_predictive_value
                ),
                "f1": float(
                    f1_score(
                        y_true,
                        prediction,
                        zero_division=0,
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        y_true,
                        prediction,
                    )
                ),
                "predicted_positive_count": int(
                    prediction.sum()
                ),
            }
        )

    metrics = pd.DataFrame(rows)

    best_f1_row = metrics.loc[
        metrics["f1"].idxmax()
    ]

    best_balanced_row = metrics.loc[
        metrics[
            "balanced_accuracy"
        ].idxmax()
    ]

    recall_candidates = metrics[
        metrics["recall"] >= HIGH_SENSITIVITY_RECALL_TARGET
    ]

    if recall_candidates.empty:
        high_sensitivity_row = metrics.loc[
            metrics["recall"].idxmax()
        ]
    else:
        high_sensitivity_row = (
            recall_candidates
            .sort_values(
                [
                    "specificity",
                    "precision",
                ],
                ascending=False,
            )
            .iloc[0]
        )

    metrics.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    selected = {
        "best_f1": best_f1_row.to_dict(),
        "best_balanced_accuracy": (
            best_balanced_row.to_dict()
        ),
        "clinical_high_sensitivity": (
            high_sensitivity_row.to_dict()
        ),
        # Inference checks this key first. It deliberately favors fewer missed
        # events; the UI must disclose the associated false-alert burden.
        "continual_learning": high_sensitivity_row.to_dict(),
        "high_sensitivity_recall_target": HIGH_SENSITIVITY_RECALL_TARGET,
    }

    with SELECTED_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selected,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 80)
    print("Arousal 門檻選擇完成")
    print("=" * 80)

    print("\n最佳 F1：")
    print(
        best_f1_row[
            [
                "threshold",
                "precision",
                "recall",
                "specificity",
                "f1",
                "balanced_accuracy",
            ]
        ].to_string()
    )

    print("\n最佳 Balanced Accuracy：")
    print(
        best_balanced_row[
            [
                "threshold",
                "precision",
                "recall",
                "specificity",
                "f1",
                "balanced_accuracy",
            ]
        ].to_string()
    )

    print(f"\n高敏感度門檻（Recall ≥ {HIGH_SENSITIVITY_RECALL_TARGET:.0%}）：")
    print(
        high_sensitivity_row[
            [
                "threshold",
                "precision",
                "recall",
                "specificity",
                "f1",
                "balanced_accuracy",
            ]
        ].to_string()
    )

    print(
        f"\n完整門檻表：{OUTPUT_FILE}"
    )

    print(
        f"選定門檻：{SELECTED_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
