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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

SUMMARY_FILE = (
    OUTPUT_ROOT
    / "respiratory_quality_summary.csv"
)

COLUMN_QUALITY_FILE = (
    OUTPUT_ROOT
    / "respiratory_column_quality.csv"
)

BACKEND_COMPARISON_FILE = (
    OUTPUT_ROOT
    / "respiratory_backend_comparison.csv"
)

INVALID_ROWS_FILE = (
    OUTPUT_ROOT
    / "respiratory_invalid_rows.csv"
)


ID_COLUMNS = {
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


IMPORTANT_FEATURES = [
    "flow_respiratory_rate_bpm",
    "thermistor_respiratory_rate_bpm",
    "thorax_respiratory_rate_bpm",
    "abdomen_respiratory_rate_bpm",
    "spo2_mean",
    "spo2_min",
    "spo2_valid_fraction",
    "spo2_below_90_fraction",
    "spo2_drop_from_start",
    "heart_rate_mean",
    "thorax_abdomen_correlation",
    "thorax_abdomen_lag_seconds",
    "snore_activity_fraction",
    "position_change_count",
]


def boolean_series(
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


def numeric_feature_columns(
    data: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in data.columns
        if column not in ID_COLUMNS
        and pd.api.types.is_numeric_dtype(
            data[column]
        )
    ]


def build_column_quality(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    numeric_columns = numeric_feature_columns(
        data
    )

    for column in numeric_columns:
        series = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        values = series.to_numpy(
            dtype=np.float64
        )

        finite_mask = np.isfinite(values)
        finite_values = values[
            finite_mask
        ]

        missing_count = int(
            series.isna().sum()
        )

        inf_count = int(
            np.isinf(values).sum()
        )

        unique_count = int(
            pd.Series(
                finite_values
            ).nunique()
        )

        row = {
            "column": column,
            "row_count": len(data),
            "missing_count": missing_count,
            "missing_fraction": (
                missing_count
                / max(len(data), 1)
            ),
            "inf_count": inf_count,
            "finite_count": int(
                finite_mask.sum()
            ),
            "unique_finite_count": (
                unique_count
            ),
            "is_constant": (
                unique_count <= 1
            ),
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

        if finite_values.size > 0:
            row.update(
                {
                    "mean": float(
                        np.mean(
                            finite_values
                        )
                    ),
                    "std": float(
                        np.std(
                            finite_values
                        )
                    ),
                    "min": float(
                        np.min(
                            finite_values
                        )
                    ),
                    "p01": float(
                        np.percentile(
                            finite_values,
                            1,
                        )
                    ),
                    "p05": float(
                        np.percentile(
                            finite_values,
                            5,
                        )
                    ),
                    "median": float(
                        np.median(
                            finite_values
                        )
                    ),
                    "p95": float(
                        np.percentile(
                            finite_values,
                            95,
                        )
                    ),
                    "p99": float(
                        np.percentile(
                            finite_values,
                            99,
                        )
                    ),
                    "max": float(
                        np.max(
                            finite_values
                        )
                    ),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_invalid_masks(
    data: pd.DataFrame,
) -> dict[str, pd.Series]:
    index = data.index

    masks: dict[str, pd.Series] = {}

    def numeric(
        column: str,
    ) -> pd.Series:
        if column not in data.columns:
            return pd.Series(
                np.nan,
                index=index,
                dtype="float64",
            )

        return pd.to_numeric(
            data[column],
            errors="coerce",
        )

    flow_rate = numeric(
        "flow_respiratory_rate_bpm"
    )

    thermistor_rate = numeric(
        "thermistor_respiratory_rate_bpm"
    )

    thorax_rate = numeric(
        "thorax_respiratory_rate_bpm"
    )

    abdomen_rate = numeric(
        "abdomen_respiratory_rate_bpm"
    )

    spo2_min = numeric(
        "spo2_min"
    )

    spo2_mean = numeric(
        "spo2_mean"
    )

    spo2_valid = numeric(
        "spo2_valid_fraction"
    )

    heart_rate = numeric(
        "heart_rate_mean"
    )

    correlation = numeric(
        "thorax_abdomen_correlation"
    )

    lag_seconds = numeric(
        "thorax_abdomen_lag_seconds"
    )

    masks[
        "invalid_flow_rate"
    ] = (
        flow_rate.notna()
        & (
            (flow_rate < 4.0)
            | (flow_rate > 45.0)
        )
    )

    masks[
        "invalid_thermistor_rate"
    ] = (
        thermistor_rate.notna()
        & (
            (thermistor_rate < 4.0)
            | (
                thermistor_rate
                > 45.0
            )
        )
    )

    masks[
        "invalid_thorax_rate"
    ] = (
        thorax_rate.notna()
        & (
            (thorax_rate < 4.0)
            | (thorax_rate > 45.0)
        )
    )

    masks[
        "invalid_abdomen_rate"
    ] = (
        abdomen_rate.notna()
        & (
            (abdomen_rate < 4.0)
            | (abdomen_rate > 45.0)
        )
    )

    masks[
        "invalid_spo2_min"
    ] = (
        spo2_min.notna()
        & (
            (spo2_min < 50.0)
            | (spo2_min > 100.5)
        )
    )

    masks[
        "invalid_spo2_mean"
    ] = (
        spo2_mean.notna()
        & (
            (spo2_mean < 50.0)
            | (spo2_mean > 100.5)
        )
    )

    masks[
        "low_spo2_validity"
    ] = (
        spo2_valid.isna()
        | (spo2_valid < 0.80)
    )

    masks[
        "invalid_heart_rate"
    ] = (
        heart_rate.notna()
        & (
            (heart_rate < 25.0)
            | (heart_rate > 220.0)
        )
    )

    masks[
        "invalid_thorax_abdomen_corr"
    ] = (
        correlation.notna()
        & (
            (correlation < -1.000001)
            | (
                correlation
                > 1.000001
            )
        )
    )

    masks[
        "invalid_thorax_abdomen_lag"
    ] = (
        lag_seconds.notna()
        & (
            np.abs(
                lag_seconds
            ) > 5.01
        )
    )

    if (
        "epoch_signal_valid"
        in data.columns
    ):
        masks[
            "epoch_signal_invalid"
        ] = ~boolean_series(
            data[
                "epoch_signal_valid"
            ]
        )

    return masks


def build_invalid_rows(
    data: pd.DataFrame,
    masks: dict[
        str,
        pd.Series,
    ],
) -> pd.DataFrame:
    invalid = pd.DataFrame(
        index=data.index
    )

    for name, mask in masks.items():
        invalid[name] = (
            mask.fillna(False)
            .astype(bool)
        )

    invalid[
        "invalid_reason_count"
    ] = invalid.sum(axis=1)

    invalid[
        "has_invalid_reason"
    ] = (
        invalid[
            "invalid_reason_count"
        ] > 0
    )

    reasons = []

    for row in invalid.itertuples():
        row_reasons = [
            column
            for column in masks
            if getattr(
                row,
                column,
            )
        ]

        reasons.append(
            " | ".join(
                row_reasons
            )
        )

    invalid[
        "invalid_reasons"
    ] = reasons

    keep_columns = [
        column
        for column in [
            "patient_id",
            "epoch_index",
            "original_epoch",
            "start_time",
            "stage",
            "respiratory_edf_backend",
            "flow_respiratory_rate_bpm",
            "thermistor_respiratory_rate_bpm",
            "thorax_respiratory_rate_bpm",
            "abdomen_respiratory_rate_bpm",
            "spo2_mean",
            "spo2_min",
            "spo2_valid_fraction",
            "heart_rate_mean",
            "thorax_abdomen_correlation",
            "thorax_abdomen_lag_seconds",
        ]
        if column in data.columns
    ]

    result = pd.concat(
        [
            data[
                keep_columns
            ].copy(),
            invalid,
        ],
        axis=1,
    )

    return result[
        result[
            "has_invalid_reason"
        ]
    ].reset_index(
        drop=True
    )


def build_backend_comparison(
    data: pd.DataFrame,
) -> pd.DataFrame:
    available_features = [
        feature
        for feature in IMPORTANT_FEATURES
        if feature in data.columns
    ]

    rows: list[dict] = []

    grouped = data.groupby(
        "respiratory_edf_backend",
        dropna=False,
    )

    for backend, group in grouped:
        for feature in available_features:
            series = pd.to_numeric(
                group[feature],
                errors="coerce",
            )

            finite = series[
                np.isfinite(
                    series.to_numpy(
                        dtype=np.float64
                    )
                )
            ]

            rows.append(
                {
                    "backend": backend,
                    "feature": feature,
                    "row_count": len(
                        group
                    ),
                    "valid_count": len(
                        finite
                    ),
                    "missing_fraction": (
                        1.0
                        - len(finite)
                        / max(
                            len(group),
                            1,
                        )
                    ),
                    "mean": (
                        float(
                            finite.mean()
                        )
                        if len(finite)
                        else np.nan
                    ),
                    "std": (
                        float(
                            finite.std(
                                ddof=0
                            )
                        )
                        if len(finite)
                        else np.nan
                    ),
                    "median": (
                        float(
                            finite.median()
                        )
                        if len(finite)
                        else np.nan
                    ),
                    "p05": (
                        float(
                            finite.quantile(
                                0.05
                            )
                        )
                        if len(finite)
                        else np.nan
                    ),
                    "p95": (
                        float(
                            finite.quantile(
                                0.95
                            )
                        )
                        if len(finite)
                        else np.nan
                    ),
                }
            )

    result = pd.DataFrame(rows)

    backend_names = sorted(
        data[
            "respiratory_edf_backend"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(backend_names) == 2:
        pivot = result.pivot(
            index="feature",
            columns="backend",
            values=[
                "mean",
                "std",
                "median",
                "missing_fraction",
            ],
        )

        pivot.columns = [
            f"{stat}_{backend}"
            for stat, backend
            in pivot.columns
        ]

        pivot = pivot.reset_index()

        backend_a = backend_names[0]
        backend_b = backend_names[1]

        mean_a = (
            f"mean_{backend_a}"
        )

        mean_b = (
            f"mean_{backend_b}"
        )

        std_a = (
            f"std_{backend_a}"
        )

        std_b = (
            f"std_{backend_b}"
        )

        pooled_std = np.sqrt(
            (
                np.square(
                    pivot[std_a]
                )
                + np.square(
                    pivot[std_b]
                )
            )
            / 2.0
        )

        pivot[
            "standardized_mean_difference"
        ] = (
            (
                pivot[mean_a]
                - pivot[mean_b]
            )
            / pooled_std.replace(
                0,
                np.nan,
            )
        )

        pivot[
            "backend_a"
        ] = backend_a

        pivot[
            "backend_b"
        ] = backend_b

        return pivot

    return result


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"找不到呼吸特徵檔："
            f"{INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE
    )

    data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    column_quality = (
        build_column_quality(
            data
        )
    )

    masks = build_invalid_masks(
        data
    )

    invalid_rows = (
        build_invalid_rows(
            data,
            masks,
        )
    )

    backend_comparison = (
        build_backend_comparison(
            data
        )
    )

    mask_counts = {
        name: int(
            mask.fillna(False).sum()
        )
        for name, mask in (
            masks.items()
        )
    }

    summary = pd.DataFrame(
        [
            {
                "row_count": len(data),
                "column_count": len(
                    data.columns
                ),
                "patient_count": int(
                    data[
                        "patient_id"
                    ].nunique()
                ),
                "backend_count": int(
                    data[
                        "respiratory_edf_backend"
                    ].nunique()
                ),
                "rows_with_any_invalid_reason": (
                    len(invalid_rows)
                ),
                "rows_with_any_invalid_fraction": (
                    len(invalid_rows)
                    / max(
                        len(data),
                        1,
                    )
                ),
                "numeric_feature_count": int(
                    len(
                        numeric_feature_columns(
                            data
                        )
                    )
                ),
                "constant_numeric_columns": int(
                    column_quality[
                        "is_constant"
                    ].sum()
                ),
                "columns_over_20pct_missing": int(
                    (
                        column_quality[
                            "missing_fraction"
                        ] > 0.20
                    ).sum()
                ),
                **mask_counts,
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    column_quality.to_csv(
        COLUMN_QUALITY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    backend_comparison.to_csv(
        BACKEND_COMPARISON_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    invalid_rows.to_csv(
        INVALID_ROWS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("Respiratory Feature 品質檢查")
    print("=" * 80)

    print(
        f"資料形狀：{data.shape}"
    )

    print(
        f"患者數量："
        f"{data['patient_id'].nunique()}"
    )

    print("\nBackend 筆數：")

    print(
        data[
            "respiratory_edf_backend"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n有任一異常理由的 Epoch："
        f"{len(invalid_rows)}"
        f"/{len(data)} "
        f"({len(invalid_rows) / max(len(data), 1):.2%})"
    )

    print("\n異常理由數量：")

    for name, count in (
        mask_counts.items()
    ):
        print(
            f"{name}：{count}"
        )

    high_missing = (
        column_quality[
            column_quality[
                "missing_fraction"
            ] > 0.20
        ]
        .sort_values(
            "missing_fraction",
            ascending=False,
        )
    )

    print(
        "\n缺值比例超過 20% 的欄位："
    )

    if high_missing.empty:
        print("無")
    else:
        print(
            high_missing[
                [
                    "column",
                    "missing_count",
                    "missing_fraction",
                ]
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    constant_columns = (
        column_quality[
            column_quality[
                "is_constant"
            ]
        ]
    )

    print("\n常數數值欄位：")

    if constant_columns.empty:
        print("無")
    else:
        print(
            constant_columns[
                [
                    "column",
                    "unique_finite_count",
                ]
            ]
            .to_string(
                index=False
            )
        )

    if (
        "standardized_mean_difference"
        in backend_comparison.columns
    ):
        largest_backend_differences = (
            backend_comparison.assign(
                absolute_smd=lambda frame: (
                    frame[
                        "standardized_mean_difference"
                    ].abs()
                )
            )
            .sort_values(
                "absolute_smd",
                ascending=False,
            )
        )

        print(
            "\nBackend 差異最大的特徵："
        )

        print(
            largest_backend_differences[
                [
                    "feature",
                    "backend_a",
                    "backend_b",
                    "standardized_mean_difference",
                ]
            ]
            .head(15)
            .round(4)
            .to_string(
                index=False
            )
        )

    print(
        f"\n摘要：{SUMMARY_FILE}"
    )

    print(
        f"欄位品質："
        f"{COLUMN_QUALITY_FILE}"
    )

    print(
        f"Backend 比較："
        f"{BACKEND_COMPARISON_FILE}"
    )

    print(
        f"異常 Epoch："
        f"{INVALID_ROWS_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
    