from pathlib import Path

import pandas as pd

from src.features.event_label_builder import (
    EventLabelBuilder,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parent

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manifest.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "event_labels.csv"
)

FAILURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "event_label_failures.csv"
)


def main() -> None:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"找不到 Manifest："
            f"{MANIFEST_FILE}"
        )

    manifest = pd.read_csv(
        MANIFEST_FILE
    )

    builder = EventLabelBuilder()

    all_labels: list[
        pd.DataFrame
    ] = []

    failures: list[dict] = []

    for index, row in (
        manifest.iterrows()
    ):
        patient_id = str(
            row["patient_id"]
        )

        print()
        print("=" * 80)
        print(
            f"[{index + 1}/{len(manifest)}] "
            f"建立 Event Labels："
            f"{patient_id}"
        )
        print("=" * 80)

        patient_folder = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / patient_id
        )

        stages_file = (
            patient_folder
            / "stages_aligned.csv"
        )

        events_file = (
            patient_folder
            / "events_aligned.csv"
        )

        try:
            labels = (
                builder
                .build_patient_labels(
                    patient_id=patient_id,
                    stages_file=stages_file,
                    events_file=events_file,
                )
            )

            if labels.empty:
                raise RuntimeError(
                    "Event Label 結果為空。"
                )

            all_labels.append(
                labels
            )

            print(
                f"完成："
                f"{len(labels)} epochs"
            )

        except Exception as exc:
            error_message = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            failures.append(
                {
                    "patient_id": (
                        patient_id
                    ),
                    "error": (
                        error_message
                    ),
                }
            )

            print(
                f"失敗：{error_message}"
            )

    if not all_labels:
        raise RuntimeError(
            "沒有任何患者成功建立"
            "Event Labels。"
        )

    dataset = pd.concat(
        all_labels,
        ignore_index=True,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    failure_df = pd.DataFrame(
        failures
    )

    failure_df.to_csv(
        FAILURE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("Event Label Dataset 完成")
    print("=" * 80)

    print(
        f"總 Epoch 數量："
        f"{len(dataset)}"
    )

    print(
        f"患者數量："
        f"{dataset['patient_id'].nunique()}"
    )

    label_columns = [
        "has_hypopnea",
        "has_obstructive_apnea",
        "has_central_apnea",
        "has_mixed_apnea",
        "has_arousal",
        "has_respiratory_arousal",
        "has_spontaneous_arousal",
        "has_any_respiratory_event",
        "arousal_within_15s",
        "arousal_within_30s",
        "arousal_within_60s",
    ]

    print("\nLabel 數量：")

    for column in label_columns:
        if column in dataset.columns:
            print(
                f"{column}："
                f"{int(dataset[column].sum())}"
            )

    print(
        f"\n輸出檔案："
        f"{OUTPUT_FILE}"
    )

    if failures:
        print("\n失敗患者：")

        print(
            failure_df.to_string(
                index=False
            )
        )
    else:
        print(
            "\n沒有失敗患者。"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()