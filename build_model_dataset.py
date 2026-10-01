from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

STAGE_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage_features.csv"
)

RESPIRATORY_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "respiratory_features_clean.csv"
)

EVENT_LABELS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "event_labels.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset_report.csv"
)

PATIENT_DATA_ROOT = PROJECT_ROOT / "data" / "processed"


MERGE_KEYS = [
    "patient_id",
    "epoch_index",
]


# 三份檔案中可能重複存在，
# 只保留其中一份的欄位。
DUPLICATE_METADATA_COLUMNS = {
    "original_epoch",
    "start_time",
    "end_time",
    "stage",
    "edf_start_seconds",
    "edf_end_seconds",
    "epoch_signal_valid",
}


# 這些是標籤或事後資訊，不能當模型輸入。
LABEL_COLUMNS = {
    "has_hypopnea",
    "has_obstructive_apnea",
    "has_central_apnea",
    "has_mixed_apnea",
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
}


def assert_unique_keys(
    data: pd.DataFrame,
    name: str,
) -> None:
    duplicated = data.duplicated(
        MERGE_KEYS,
        keep=False,
    )

    if duplicated.any():
        examples = data.loc[
            duplicated,
            MERGE_KEYS,
        ].head(10)

        raise RuntimeError(
            f"{name} 的 patient_id + epoch_index "
            f"不是唯一鍵。\n{examples}"
        )


def remove_duplicate_metadata(
    data: pd.DataFrame,
    keep_keys: bool = True,
) -> pd.DataFrame:
    columns_to_drop = [
        column
        for column in DUPLICATE_METADATA_COLUMNS
        if column in data.columns
    ]

    if not keep_keys:
        columns_to_drop.extend(
            [
                column
                for column in MERGE_KEYS
                if column in data.columns
            ]
        )

    return data.drop(
        columns=columns_to_drop,
        errors="ignore",
    )


def main() -> None:
    required_files = [
        STAGE_FEATURES_FILE,
        RESPIRATORY_FEATURES_FILE,
        EVENT_LABELS_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"找不到必要檔案：{file_path}"
            )

    stage_data = pd.read_csv(
        STAGE_FEATURES_FILE
    )

    respiratory_data = pd.read_csv(
        RESPIRATORY_FEATURES_FILE
    )

    event_data = pd.read_csv(
        EVENT_LABELS_FILE
    )

    print("=" * 80)
    print("合併前資料")
    print("=" * 80)
    print(
        f"Stage Features："
        f"{stage_data.shape}"
    )
    print(
        f"Respiratory Features："
        f"{respiratory_data.shape}"
    )
    print(
        f"Event Labels："
        f"{event_data.shape}"
    )

    for name, data in [
        ("Stage Features", stage_data),
        (
            "Respiratory Features",
            respiratory_data,
        ),
        ("Event Labels", event_data),
    ]:
        missing_keys = (
            set(MERGE_KEYS)
            - set(data.columns)
        )

        if missing_keys:
            raise RuntimeError(
                f"{name} 缺少合併鍵："
                f"{sorted(missing_keys)}"
            )

        assert_unique_keys(
            data,
            name,
        )

    required_safe_labels = {
        "predict_arousal_next_15s",
        "predict_arousal_next_30s",
        "predict_arousal_next_60s",
    }

    missing_safe_labels = (
        required_safe_labels
        - set(event_data.columns)
    )

    if missing_safe_labels:
        raise RuntimeError(
            "event_labels.csv 尚未建立安全的 "
            "Arousal 未來標籤："
            f"{sorted(missing_safe_labels)}。"
            "請先重新執行 build_event_labels.py。"
        )

    # Stage Features 作為主要表，
    # 保留時間、Stage 與 EEG 特徵。
    respiratory_for_merge = (
        remove_duplicate_metadata(
            respiratory_data,
            keep_keys=True,
        )
    )

    event_for_merge = (
        remove_duplicate_metadata(
            event_data,
            keep_keys=True,
        )
    )

    merged = stage_data.merge(
        respiratory_for_merge,
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=(
            "",
            "_resp",
        ),
    )

    merged = merged.merge(
        event_for_merge,
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=(
            "",
            "_event",
        ),
    )

    # Patient-level demographics are repeated across that patient's epochs so
    # age and BMI are learned by the Arousal model instead of being display-only.
    demographic_rows: list[dict[str, object]] = []
    for metadata_file in sorted(PATIENT_DATA_ROOT.glob("*/patient_metadata.csv")):
        metadata = pd.read_csv(metadata_file)
        if metadata.empty or "patient_id" not in metadata.columns:
            continue
        row = metadata.iloc[0]
        demographic_rows.append(
            {
                "patient_id": str(row["patient_id"]).strip(),
                "age": pd.to_numeric(row.get("age"), errors="coerce"),
                "BMI": pd.to_numeric(row.get("BMI"), errors="coerce"),
            }
        )

    if not demographic_rows:
        raise RuntimeError("找不到可供 Arousal 模型使用的 age／BMI 病人資料。")

    demographics = (
        pd.DataFrame(demographic_rows)
        .drop_duplicates(subset=["patient_id"], keep="last")
    )
    merged = merged.drop(columns=["age", "BMI"], errors="ignore").merge(
        demographics,
        on="patient_id",
        how="left",
        validate="many_to_one",
    )

    missing_demographics = merged[["age", "BMI"]].isna().any(axis=1)
    if missing_demographics.any():
        missing_patients = sorted(
            merged.loc[missing_demographics, "patient_id"].astype(str).unique()
        )
        raise RuntimeError(
            "下列病人缺少 age 或 BMI，不能宣稱人口學特徵已完整參與訓練："
            f"{missing_patients}"
        )

    if len(merged) != len(stage_data):
        raise RuntimeError(
            "合併後 Epoch 數量與 Stage Features "
            "不一致："
            f"{len(stage_data)} → {len(merged)}"
        )

    # Inf 統一改成 NaN，之後由模型 Pipeline impute。
    merged.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    # 建立任一 Apnea 標籤。
    merged["has_any_apnea"] = (
        merged[
            [
                "has_obstructive_apnea",
                "has_central_apnea",
                "has_mixed_apnea",
            ]
        ]
        .astype(bool)
        .any(axis=1)
    )

    # 建立 Apnea 或 Hypopnea 合併標籤。
    merged[
        "has_apnea_or_hypopnea"
    ] = (
        merged["has_any_apnea"]
        | merged["has_hypopnea"].astype(bool)
    )

    # 標記可否用於核心呼吸模型。
    if (
        "quality_core_features_valid"
        in merged.columns
    ):
        merged[
            "usable_for_core_respiratory_model"
        ] = (
            merged[
                "quality_core_features_valid"
            ].astype(bool)
        )
    else:
        merged[
            "usable_for_core_respiratory_model"
        ] = True

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    label_summary = {}

    for column in sorted(
        LABEL_COLUMNS
        | {
            "has_any_apnea",
            "has_apnea_or_hypopnea",
        }
    ):
        if column in merged.columns:
            series = merged[column]

            if pd.api.types.is_bool_dtype(
                series
            ):
                label_summary[column] = int(
                    series.sum()
                )
            elif pd.api.types.is_numeric_dtype(
                series
            ):
                label_summary[column] = int(
                    series.fillna(0).astype(bool).sum()
                )

    report = pd.DataFrame(
        [
            {
                "row_count": len(
                    merged
                ),
                "column_count": len(
                    merged.columns
                ),
                "patient_count": int(
                    merged[
                        "patient_id"
                    ].nunique()
                ),
                "stage_feature_rows": len(
                    stage_data
                ),
                "respiratory_feature_rows": len(
                    respiratory_data
                ),
                "event_label_rows": len(
                    event_data
                ),
                "rows_with_core_quality": int(
                    merged[
                        "usable_for_core_respiratory_model"
                    ].sum()
                ),
                **{
                    f"positive_{key}": value
                    for key, value
                    in label_summary.items()
                },
            }
        ]
    )

    report.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("最終 Model Dataset 完成")
    print("=" * 80)

    print(
        f"資料形狀：{merged.shape}"
    )

    print(
        f"患者數量："
        f"{merged['patient_id'].nunique()}"
    )

    print(
        "核心呼吸品質合格 Epoch："
        f"{int(merged['usable_for_core_respiratory_model'].sum())}"
        f"/{len(merged)}"
    )

    print("\n主要 Label 數量：")

    display_labels = [
        "has_hypopnea",
        "has_obstructive_apnea",
        "has_any_apnea",
        "has_apnea_or_hypopnea",
        "has_arousal",
        "has_respiratory_arousal",
        "predict_arousal_next_15s",
        "predict_arousal_next_30s",
        "predict_arousal_next_60s",
    ]

    for column in display_labels:
        print(
            f"{column}："
            f"{int(merged[column].astype(bool).sum())}"
        )

    print(
        f"\n輸出檔案：{OUTPUT_FILE}"
    )

    print(
        f"資料集報告：{REPORT_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
