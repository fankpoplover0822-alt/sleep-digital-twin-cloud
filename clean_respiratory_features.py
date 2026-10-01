from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "respiratory_features.csv"
)

COLUMN_QUALITY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "respiratory_column_quality.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "respiratory_features_clean.csv"
)

CLEANING_REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "respiratory_cleaning_report.csv"
)


# 不應刪除的識別與標籤欄位
PROTECTED_COLUMNS = {
    "patient_id",
    "epoch_index",
    "original_epoch",
    "start_time",
    "end_time",
    "stage",
    "edf_start_seconds",
    "edf_end_seconds",
    "respiratory_edf_backend",
    "epoch_signal_valid",
}


def numeric(
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


def mark_outside_range_as_nan(
    data: pd.DataFrame,
    column: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> int:
    if column not in data.columns:
        return 0

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

    count = int(
        invalid.sum()
    )

    data.loc[
        invalid,
        column,
    ] = np.nan

    return count


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"找不到輸入檔案：{INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE
    )

    original_shape = data.shape

    data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    report_rows: list[dict] = []

    # ==========================================
    # 1. 呼吸頻率合理範圍
    # ==========================================
    respiratory_rate_columns = [
        "flow_respiratory_rate_bpm",
        "thermistor_respiratory_rate_bpm",
        "thorax_respiratory_rate_bpm",
        "abdomen_respiratory_rate_bpm",
    ]

    for column in respiratory_rate_columns:
        count = mark_outside_range_as_nan(
            data=data,
            column=column,
            minimum=4.0,
            maximum=45.0,
        )

        report_rows.append(
            {
                "action": "range_to_nan",
                "column": column,
                "affected_count": count,
                "rule": "4 <= respiratory rate <= 45 bpm",
            }
        )

    # ==========================================
    # 2. SpO₂ 合理範圍
    # ==========================================
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
        count = mark_outside_range_as_nan(
            data=data,
            column=column,
            minimum=50.0,
            maximum=100.5,
        )

        report_rows.append(
            {
                "action": "range_to_nan",
                "column": column,
                "affected_count": count,
                "rule": "50 <= SpO2 <= 100.5",
            }
        )

    # SpO₂ 變化量特徵應為合理範圍
    spo2_change_columns = [
        "spo2_drop_from_start",
        "spo2_largest_drop",
    ]

    for column in spo2_change_columns:
        count = mark_outside_range_as_nan(
            data=data,
            column=column,
            minimum=0.0,
            maximum=50.0,
        )

        report_rows.append(
            {
                "action": "range_to_nan",
                "column": column,
                "affected_count": count,
                "rule": "0 <= SpO2 drop <= 50",
            }
        )

    # 比例欄位必須在 0～1
    fraction_columns = [
        column
        for column in data.columns
        if (
            column.endswith("_fraction")
            or column.endswith("_coverage")
        )
    ]

    for column in fraction_columns:
        count = mark_outside_range_as_nan(
            data=data,
            column=column,
            minimum=0.0,
            maximum=1.0,
        )

        report_rows.append(
            {
                "action": "range_to_nan",
                "column": column,
                "affected_count": count,
                "rule": "0 <= fraction <= 1",
            }
        )

    # ==========================================
    # 3. 心率合理範圍
    # ==========================================
    heart_rate_level_columns = [
        "heart_rate_mean",
        "heart_rate_min",
        "heart_rate_max",
        "heart_rate_median",
        "heart_rate_p05",
        "heart_rate_p25",
        "heart_rate_p75",
        "heart_rate_p95",
    ]

    for column in heart_rate_level_columns:
        count = mark_outside_range_as_nan(
            data=data,
            column=column,
            minimum=25.0,
            maximum=220.0,
        )

        report_rows.append(
            {
                "action": "range_to_nan",
                "column": column,
                "affected_count": count,
                "rule": "25 <= heart rate <= 220 bpm",
            }
        )

    # ==========================================
    # 4. 胸腹相關與 lag
    # ==========================================
    count = mark_outside_range_as_nan(
        data=data,
        column="thorax_abdomen_correlation",
        minimum=-1.0,
        maximum=1.0,
    )

    report_rows.append(
        {
            "action": "range_to_nan",
            "column": "thorax_abdomen_correlation",
            "affected_count": count,
            "rule": "-1 <= correlation <= 1",
        }
    )

    count = mark_outside_range_as_nan(
        data=data,
        column="thorax_abdomen_lag_seconds",
        minimum=-5.0,
        maximum=5.0,
    )

    report_rows.append(
        {
            "action": "range_to_nan",
            "column": "thorax_abdomen_lag_seconds",
            "affected_count": count,
            "rule": "-5 <= lag <= 5 seconds",
        }
    )

    # ==========================================
    # 5. 建立品質旗標
    # ==========================================
    spo2_valid_fraction = numeric(
        data,
        "spo2_valid_fraction",
    )

    data["quality_low_spo2_validity"] = (
        spo2_valid_fraction.isna()
        | (spo2_valid_fraction < 0.80)
    )

    data["quality_missing_flow_rate"] = (
        numeric(
            data,
            "flow_respiratory_rate_bpm",
        ).isna()
    )

    data[
        "quality_missing_thorax_abdomen_correlation"
    ] = numeric(
        data,
        "thorax_abdomen_correlation",
    ).isna()

    heart_rate_mean = numeric(
        data,
        "heart_rate_mean",
    )

    data["quality_missing_heart_rate"] = (
        heart_rate_mean.isna()
    )

    quality_columns = [
        "quality_low_spo2_validity",
        "quality_missing_flow_rate",
        (
            "quality_missing_"
            "thorax_abdomen_correlation"
        ),
        "quality_missing_heart_rate",
    ]

    data["quality_issue_count"] = (
        data[
            quality_columns
        ]
        .astype(int)
        .sum(axis=1)
    )

    data["quality_any_issue"] = (
        data["quality_issue_count"] > 0
    )

    # 核心欄位均可用時才標記為高品質
    data["quality_core_features_valid"] = (
        ~data[
            "quality_low_spo2_validity"
        ]
        & ~data[
            "quality_missing_flow_rate"
        ]
        & ~data[
            "quality_missing_heart_rate"
        ]
    )

    # ==========================================
    # 6. 移除常數數值欄位
    # ==========================================
    constant_columns: list[str] = []

    if COLUMN_QUALITY_FILE.exists():
        quality = pd.read_csv(
            COLUMN_QUALITY_FILE
        )

        if {
            "column",
            "is_constant",
        }.issubset(
            quality.columns
        ):
            is_constant = (
                quality["is_constant"]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": True,
                        "1": True,
                        "false": False,
                        "0": False,
                    }
                )
                .fillna(False)
            )

            constant_columns = (
                quality.loc[
                    is_constant,
                    "column",
                ]
                .astype(str)
                .tolist()
            )

    # 不刪除 protected 欄位
    constant_columns = [
        column
        for column in constant_columns
        if (
            column in data.columns
            and column
            not in PROTECTED_COLUMNS
        )
    ]

    if constant_columns:
        data.drop(
            columns=constant_columns,
            inplace=True,
        )

    for column in constant_columns:
        report_rows.append(
            {
                "action": "drop_constant_column",
                "column": column,
                "affected_count": len(data),
                "rule": "numeric column has <= 1 finite unique value",
            }
        )

    # ==========================================
    # 7. 儲存
    # ==========================================
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    report = pd.DataFrame(
        report_rows
    )

    report.to_csv(
        CLEANING_REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("呼吸特徵清理完成")
    print("=" * 80)

    print(
        f"原始形狀：{original_shape}"
    )

    print(
        f"清理後形狀：{data.shape}"
    )

    print(
        f"移除常數欄位數量："
        f"{len(constant_columns)}"
    )

    if constant_columns:
        print("\n已移除常數欄位：")

        for column in constant_columns:
            print(f"- {column}")

    print(
        "\nSpO₂ 品質不足 Epoch："
        f"{int(data['quality_low_spo2_validity'].sum())}"
    )

    print(
        "Flow rate 缺值 Epoch："
        f"{int(data['quality_missing_flow_rate'].sum())}"
    )

    print(
        "胸腹相關缺值 Epoch："
        f"{int(data['quality_missing_thorax_abdomen_correlation'].sum())}"
    )

    print(
        "心率缺值 Epoch："
        f"{int(data['quality_missing_heart_rate'].sum())}"
    )

    print(
        "核心特徵品質合格 Epoch："
        f"{int(data['quality_core_features_valid'].sum())}"
        f"/{len(data)}"
    )

    print(
        f"\n輸出資料：{OUTPUT_FILE}"
    )

    print(
        f"清理報告：{CLEANING_REPORT_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()