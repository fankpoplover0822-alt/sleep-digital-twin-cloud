from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EVENT_TYPE_TO_COLUMN = {
    "HYPOPNEA": "has_hypopnea",
    "OBSTRUCTIVE_APNEA": "has_obstructive_apnea",
    "CENTRAL_APNEA": "has_central_apnea",
    "MIXED_APNEA": "has_mixed_apnea",
    "AROUSAL": "has_arousal",
}


AROUSAL_SUBTYPE_COLUMNS = {
    "Respiratory Arousal": (
        "has_respiratory_arousal"
    ),
    "Spontaneous Arousal": (
        "has_spontaneous_arousal"
    ),
}


RESPIRATORY_EVENT_TYPES = {
    "HYPOPNEA",
    "OBSTRUCTIVE_APNEA",
    "CENTRAL_APNEA",
    "MIXED_APNEA",
}


FUTURE_AROUSAL_WINDOWS = [
    15,
    30,
    60,
]


def to_boolean_series(
    series: pd.Series,
) -> pd.Series:
    """
    將 CSV 中的 True、False、1、0、yes、no
    統一轉成布林值。
    """
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


class EventLabelBuilder:
    def build_patient_labels(
        self,
        patient_id: str,
        stages_file: str | Path,
        events_file: str | Path,
    ) -> pd.DataFrame:
        stages_path = Path(
            stages_file
        )

        events_path = Path(
            events_file
        )

        if not stages_path.exists():
            raise FileNotFoundError(
                f"找不到 Stage 檔案："
                f"{stages_path}"
            )

        if not events_path.exists():
            raise FileNotFoundError(
                f"找不到 Event 檔案："
                f"{events_path}"
            )

        stages = pd.read_csv(
            stages_path,
            parse_dates=[
                "start_time",
                "end_time",
            ],
        )

        events = pd.read_csv(
            events_path,
            parse_dates=[
                "start_time",
                "end_time",
            ],
        )

        required_stage_columns = {
            "epoch_index",
            "original_epoch",
            "start_time",
            "end_time",
            "stage",
            "edf_start_seconds",
            "edf_end_seconds",
            "usable_for_stage_training",
        }

        missing_stage_columns = (
            required_stage_columns
            - set(stages.columns)
        )

        if missing_stage_columns:
            raise RuntimeError(
                "stages_aligned.csv 缺少欄位："
                f"{sorted(missing_stage_columns)}"
            )

        usable_mask = to_boolean_series(
            stages[
                "usable_for_stage_training"
            ]
        )

        stages = stages[
            usable_mask
        ].copy()

        if stages.empty:
            raise RuntimeError(
                f"{patient_id} 沒有可用 Stage。"
            )

        required_event_columns = {
            "event_type",
            "subtype",
            "start_time",
            "end_time",
            "duration_seconds",
            "edf_start_seconds",
            "edf_end_seconds",
        }

        if events.empty:
            events = pd.DataFrame(
                columns=sorted(
                    required_event_columns
                )
            )

        missing_event_columns = (
            required_event_columns
            - set(events.columns)
        )

        if missing_event_columns:
            raise RuntimeError(
                "events_aligned.csv 缺少欄位："
                f"{sorted(missing_event_columns)}"
            )

        stages[
            "edf_start_seconds"
        ] = pd.to_numeric(
            stages[
                "edf_start_seconds"
            ],
            errors="coerce",
        )

        stages[
            "edf_end_seconds"
        ] = pd.to_numeric(
            stages[
                "edf_end_seconds"
            ],
            errors="coerce",
        )

        stages = stages.dropna(
            subset=[
                "edf_start_seconds",
                "edf_end_seconds",
            ]
        ).copy()

        events[
            "event_type"
        ] = (
            events["event_type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        events[
            "subtype"
        ] = (
            events["subtype"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        events[
            "duration_seconds"
        ] = pd.to_numeric(
            events[
                "duration_seconds"
            ],
            errors="coerce",
        )

        events[
            "edf_start_seconds"
        ] = pd.to_numeric(
            events[
                "edf_start_seconds"
            ],
            errors="coerce",
        )

        events[
            "edf_end_seconds"
        ] = pd.to_numeric(
            events[
                "edf_end_seconds"
            ],
            errors="coerce",
        )

        events = events.dropna(
            subset=[
                "edf_start_seconds",
                "edf_end_seconds",
            ]
        ).copy()

        events = events[
            events[
                "edf_end_seconds"
            ]
            > events[
                "edf_start_seconds"
            ]
        ].copy()

        events = events.sort_values(
            [
                "edf_start_seconds",
                "edf_end_seconds",
            ]
        ).reset_index(
            drop=True
        )

        arousal_events = events[
            events[
                "event_type"
            ] == "AROUSAL"
        ].copy()

        arousal_events = (
            arousal_events
            .sort_values(
                "edf_start_seconds"
            )
            .reset_index(
                drop=True
            )
        )

        arousal_start_seconds = (
            arousal_events[
                "edf_start_seconds"
            ].to_numpy(
                dtype=np.float64
            )
        )

        rows: list[dict] = []

        for stage_row in (
            stages.itertuples(
                index=False
            )
        ):
            epoch_start = float(
                stage_row
                .edf_start_seconds
            )

            epoch_end = float(
                stage_row
                .edf_end_seconds
            )

            overlapping = events[
                (
                    events[
                        "edf_start_seconds"
                    ] < epoch_end
                )
                & (
                    events[
                        "edf_end_seconds"
                    ] > epoch_start
                )
            ].copy()

            label_row = {
                "patient_id": (
                    patient_id
                ),
                "epoch_index": int(
                    stage_row.epoch_index
                ),
                "original_epoch": int(
                    stage_row.original_epoch
                ),
                "start_time": (
                    stage_row.start_time
                ),
                "end_time": (
                    stage_row.end_time
                ),
                "stage": (
                    stage_row.stage
                ),
                "edf_start_seconds": (
                    epoch_start
                ),
                "edf_end_seconds": (
                    epoch_end
                ),
                "event_count_in_epoch": 0,
                "total_event_overlap_seconds": (
                    0.0
                ),
                "respiratory_event_count": 0,
                "respiratory_event_overlap_seconds": (
                    0.0
                ),
                "arousal_count": 0,
                "arousal_overlap_seconds": (
                    0.0
                ),
            }

            for output_column in (
                EVENT_TYPE_TO_COLUMN
                .values()
            ):
                label_row[
                    output_column
                ] = False

            for output_column in (
                AROUSAL_SUBTYPE_COLUMNS
                .values()
            ):
                label_row[
                    output_column
                ] = False

            if not overlapping.empty:
                overlap_seconds = (
                    np.minimum(
                        overlapping[
                            "edf_end_seconds"
                        ].to_numpy(
                            dtype=np.float64
                        ),
                        epoch_end,
                    )
                    - np.maximum(
                        overlapping[
                            "edf_start_seconds"
                        ].to_numpy(
                            dtype=np.float64
                        ),
                        epoch_start,
                    )
                )

                overlap_seconds = (
                    np.maximum(
                        overlap_seconds,
                        0.0,
                    )
                )

                overlapping[
                    "overlap_seconds"
                ] = overlap_seconds

                label_row[
                    "event_count_in_epoch"
                ] = int(
                    len(overlapping)
                )

                label_row[
                    "total_event_overlap_seconds"
                ] = float(
                    overlapping[
                        "overlap_seconds"
                    ].sum()
                )

                for (
                    event_type,
                    output_column,
                ) in (
                    EVENT_TYPE_TO_COLUMN
                    .items()
                ):
                    event_mask = (
                        overlapping[
                            "event_type"
                        ] == event_type
                    )

                    label_row[
                        output_column
                    ] = bool(
                        event_mask.any()
                    )

                respiratory_mask = (
                    overlapping[
                        "event_type"
                    ].isin(
                        RESPIRATORY_EVENT_TYPES
                    )
                )

                label_row[
                    "respiratory_event_count"
                ] = int(
                    respiratory_mask.sum()
                )

                label_row[
                    "respiratory_event_overlap_seconds"
                ] = float(
                    overlapping.loc[
                        respiratory_mask,
                        "overlap_seconds",
                    ].sum()
                )

                arousal_mask = (
                    overlapping[
                        "event_type"
                    ] == "AROUSAL"
                )

                label_row[
                    "arousal_count"
                ] = int(
                    arousal_mask.sum()
                )

                label_row[
                    "arousal_overlap_seconds"
                ] = float(
                    overlapping.loc[
                        arousal_mask,
                        "overlap_seconds",
                    ].sum()
                )

                for (
                    subtype_name,
                    output_column,
                ) in (
                    AROUSAL_SUBTYPE_COLUMNS
                    .items()
                ):
                    subtype_mask = (
                        arousal_mask
                        & (
                            overlapping[
                                "subtype"
                            ] == subtype_name
                        )
                    )

                    label_row[
                        output_column
                    ] = bool(
                        subtype_mask.any()
                    )

            label_row[
                "has_any_respiratory_event"
            ] = bool(
                label_row[
                    "has_hypopnea"
                ]
                or label_row[
                    "has_obstructive_apnea"
                ]
                or label_row[
                    "has_central_apnea"
                ]
                or label_row[
                    "has_mixed_apnea"
                ]
            )

            label_row[
                "has_any_apnea"
            ] = bool(
                label_row[
                    "has_obstructive_apnea"
                ]
                or label_row[
                    "has_central_apnea"
                ]
                or label_row[
                    "has_mixed_apnea"
                ]
            )

            label_row[
                "has_apnea_or_hypopnea"
            ] = bool(
                label_row[
                    "has_any_apnea"
                ]
                or label_row[
                    "has_hypopnea"
                ]
            )

            label_row[
                "has_any_event"
            ] = bool(
                label_row[
                    "event_count_in_epoch"
                ] > 0
            )

            # ==========================================
            # 事件分析用標籤
            # 從 Epoch 開始時間計算下一次 Arousal
            # 這組標籤可能與 Epoch 內訊號重疊，
            # 不可直接當作提前預測模型標籤。
            # ==========================================
            next_arousal_seconds = (
                np.nan
            )

            if (
                arousal_start_seconds.size
                > 0
            ):
                start_index = int(
                    np.searchsorted(
                        arousal_start_seconds,
                        epoch_start,
                        side="left",
                    )
                )

                if (
                    start_index
                    < arousal_start_seconds.size
                ):
                    next_arousal_seconds = float(
                        arousal_start_seconds[
                            start_index
                        ]
                        - epoch_start
                    )

            label_row[
                "next_arousal_seconds"
            ] = next_arousal_seconds

            # ==========================================
            # 模型安全標籤
            # 使用目前完整 30 秒 Epoch 的訊號，
            # 預測 Epoch 結束後才開始的 Arousal。
            # 輸入與未來標籤在時間上不重疊。
            # ==========================================
            next_arousal_after_epoch_seconds = (
                np.nan
            )

            if (
                arousal_start_seconds.size
                > 0
            ):
                future_index = int(
                    np.searchsorted(
                        arousal_start_seconds,
                        epoch_end,
                        side="right",
                    )
                )

                if (
                    future_index
                    < arousal_start_seconds.size
                ):
                    next_arousal_after_epoch_seconds = (
                        float(
                            arousal_start_seconds[
                                future_index
                            ]
                            - epoch_end
                        )
                    )

            label_row[
                "next_arousal_after_epoch_seconds"
            ] = (
                next_arousal_after_epoch_seconds
            )

            for window_seconds in (
                FUTURE_AROUSAL_WINDOWS
            ):
                label_row[
                    f"arousal_within_"
                    f"{window_seconds}s"
                ] = bool(
                    np.isfinite(
                        next_arousal_seconds
                    )
                    and (
                        0.0
                        < next_arousal_seconds
                        <= window_seconds
                    )
                )

                label_row[
                    f"predict_arousal_next_"
                    f"{window_seconds}s"
                ] = bool(
                    np.isfinite(
                        next_arousal_after_epoch_seconds
                    )
                    and (
                        0.0
                        < (
                            next_arousal_after_epoch_seconds
                        )
                        <= window_seconds
                    )
                )

            rows.append(
                label_row
            )

        result = pd.DataFrame(
            rows
        )

        if result.empty:
            raise RuntimeError(
                f"{patient_id} 未建立任何 "
                "Event Label。"
            )

        result = result.sort_values(
            [
                "epoch_index",
            ]
        ).reset_index(
            drop=True
        )

        return result