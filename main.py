from pathlib import Path

from src.features.channel_checker import (
    check_all_patient_channels,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parent

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "channel_inventory.csv"
)


def main() -> None:
    result = check_all_patient_channels(
        raw_root=RAW_ROOT,
        output_path=OUTPUT_PATH,
    )

    print()
    print("=" * 80)
    print("Channel 檢查結果")
    print("=" * 80)

    print(
        f"患者數量：{len(result)}"
    )

    columns = [
        "has_flow",
        "has_spo2",
        "has_position",
        "has_thorax",
        "has_abdomen",
        "has_ecg",
        "has_snore",
        "has_thermistor",
        "has_heart_rate",
    ]

    for column in columns:
        if column in result.columns:
            print(
                f"{column}："
                f"{int(result[column].fillna(False).sum())}"
                f"/{len(result)}"
            )

    missing_rows = result[
        ~result[
            [
                column
                for column in columns
                if column in result.columns
            ]
        ]
        .fillna(False)
        .all(axis=1)
    ]

    if missing_rows.empty:
        print(
            "\n所有患者的重要 Channel "
            "都已找到。"
        )
    else:
        print(
            "\n有缺少重要 Channel 的患者："
        )

        display_columns = [
            "patient_id",
            "flow_channel",
            "spo2_channel",
            "position_channel",
            "thorax_channel",
            "abdomen_channel",
            "ecg_channel",
            "snore_channel",
            "thermistor_channel",
            "heart_rate_channel",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in missing_rows.columns
        ]

        print(
            missing_rows[
                display_columns
            ].to_string(
                index=False
            )
        )

    print(
        f"\n完整結果已儲存："
        f"{OUTPUT_PATH}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()