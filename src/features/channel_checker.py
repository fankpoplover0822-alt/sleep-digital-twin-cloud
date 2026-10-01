from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.io_utils.edf_reader import EDFReader


CHANNEL_KEYWORDS = {
    "flow": [
        "flow",
        "nasal",
        "pressure",
    ],
    "spo2": [
        "saturation",
        "spo2",
        "sao2",
    ],
    "position": [
        "posangle",
        "position",
        "body position",
    ],
    "thorax": [
        "thor",
        "chest",
    ],
    "abdomen": [
        "abdom",
        "abdomen",
    ],
    "ecg": [
        "ekg",
        "ecg",
    ],
    "snore": [
        "snore",
    ],
    "thermistor": [
        "thermistor",
    ],
    "heart_rate": [
        "heart rate",
    ],
}


def find_channel(
    channel_names: list[str],
    keywords: list[str],
) -> str | None:
    for channel_name in channel_names:
        channel_lower = channel_name.lower()

        for keyword in keywords:
            if keyword.lower() in channel_lower:
                return channel_name

    return None


def check_all_patient_channels(
    raw_root: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    raw_root = Path(raw_root)
    output_path = Path(output_path)

    rows: list[dict] = []

    patient_folders = sorted(
        folder
        for folder in raw_root.iterdir()
        if folder.is_dir()
    )

    for index, folder in enumerate(
        patient_folders,
        start=1,
    ):
        print(
            f"[{index}/{len(patient_folders)}] "
            f"檢查：{folder.name}"
        )

        edf_files = list(
            folder.glob("*_EDF.edf")
        )

        if len(edf_files) != 1:
            rows.append(
                {
                    "patient_id": folder.name,
                    "edf_ok": False,
                    "error": (
                        f"EDF 數量異常："
                        f"{len(edf_files)}"
                    ),
                }
            )
            continue

        try:
            reader = EDFReader(
                edf_files[0]
            )
            reader.load()

            channel_names = list(
                reader.raw.ch_names
            )

            row = {
                "patient_id": folder.name,
                "edf_ok": True,
                "channel_count": len(
                    channel_names
                ),
                "all_channels": " | ".join(
                    channel_names
                ),
                "error": None,
            }

            for feature_name, keywords in (
                CHANNEL_KEYWORDS.items()
            ):
                matched_channel = find_channel(
                    channel_names,
                    keywords,
                )

                row[
                    f"{feature_name}_channel"
                ] = matched_channel

                row[
                    f"has_{feature_name}"
                ] = (
                    matched_channel is not None
                )

            rows.append(row)

        except Exception as exc:
            rows.append(
                {
                    "patient_id": folder.name,
                    "edf_ok": False,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

    result = pd.DataFrame(rows)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return result