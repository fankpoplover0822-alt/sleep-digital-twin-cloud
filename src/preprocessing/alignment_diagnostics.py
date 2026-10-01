from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_alignment_diagnostics(
    patient_id: str,
    edf_start,
    edf_end,
    stages: pd.DataFrame,
    events: pd.DataFrame,
) -> dict:
    edf_duration_seconds = (
        pd.Timestamp(edf_end)
        - pd.Timestamp(edf_start)
    ).total_seconds()

    stage_outside = stages[
        ~stages["inside_edf"]
    ].copy()

    event_outside = events[
        ~events["inside_edf"]
    ].copy()

    result = {
        "patient_id": patient_id,
        "edf_start": edf_start,
        "edf_end": edf_end,

        "stage_start": None,
        "stage_end": None,

        "event_start": None,
        "event_end": None,

        "stage_total": len(stages),
        "stage_outside_count": len(
            stage_outside
        ),
        "stage_before_edf_count": 0,
        "stage_after_edf_count": 0,

        "event_total": len(events),
        "event_outside_count": len(
            event_outside
        ),
        "event_before_edf_count": 0,
        "event_after_edf_count": 0,

        "minimum_stage_start_seconds": None,
        "maximum_stage_end_seconds": None,

        "minimum_event_start_seconds": None,
        "maximum_event_end_seconds": None,
    }

    if not stages.empty:
        result["stage_start"] = (
            stages["start_time"].min()
        )

        result["stage_end"] = (
            stages["end_time"].max()
        )

        result[
            "minimum_stage_start_seconds"
        ] = float(
            stages[
                "edf_start_seconds"
            ].min()
        )

        result[
            "maximum_stage_end_seconds"
        ] = float(
            stages[
                "edf_end_seconds"
            ].max()
        )

        result[
            "stage_before_edf_count"
        ] = int(
            (
                stages[
                    "edf_start_seconds"
                ] < 0
            ).sum()
        )

        result[
            "stage_after_edf_count"
        ] = int(
            (
                stages[
                    "edf_end_seconds"
                ]
                > edf_duration_seconds
            ).sum()
        )

    if not events.empty:
        result["event_start"] = (
            events["start_time"].min()
        )

        result["event_end"] = (
            events["end_time"].max()
        )

        result[
            "minimum_event_start_seconds"
        ] = float(
            events[
                "edf_start_seconds"
            ].min()
        )

        result[
            "maximum_event_end_seconds"
        ] = float(
            events[
                "edf_end_seconds"
            ].max()
        )

        result[
            "event_before_edf_count"
        ] = int(
            (
                events[
                    "edf_start_seconds"
                ] < 0
            ).sum()
        )

        result[
            "event_after_edf_count"
        ] = int(
            (
                events[
                    "edf_end_seconds"
                ]
                > edf_duration_seconds
            ).sum()
        )

    return result


def save_alignment_diagnostics(
    diagnostics: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"時間對齊診斷已儲存："
        f"{output_path}"
    )