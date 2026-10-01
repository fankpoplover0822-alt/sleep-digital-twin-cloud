from pathlib import Path

import pandas as pd

from src.features.channel_mapper import (
    ChannelMapper,
)
from src.features.stage_feature_builder import (
    StageFeatureBuilder,
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

CHANNEL_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "channel_map.yaml"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage_features.csv"
)


def main() -> None:
    manifest = pd.read_csv(
        MANIFEST_FILE
    )

    mapper = ChannelMapper(
        CHANNEL_CONFIG
    )

    builder = StageFeatureBuilder(
        mapper
    )

    patient_features: list[
        pd.DataFrame
    ] = []

    failures: list[dict] = []

    for index, row in manifest.iterrows():
        patient_id = row["patient_id"]

        print(
            f"[{index + 1}/{len(manifest)}] "
            f"建立 Stage Features："
            f"{patient_id}"
        )

        stages_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / patient_id
            / "stages_aligned.csv"
        )

        try:
            features = (
                builder
                .build_patient_features(
                    patient_id=patient_id,
                    edf_file=row["edf_file"],
                    stages_file=stages_file,
                    corrected_edf_start=row[
                        "corrected_edf_start"
                    ],
                )
            )

            patient_features.append(
                features
            )

            print(
                f"完成："
                f"{len(features)} epochs"
            )

        except Exception as exc:
            failures.append(
                {
                    "patient_id": patient_id,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

            print(
                f"失敗："
                f"{failures[-1]['error']}"
            )

    if not patient_features:
        raise RuntimeError(
            "沒有任何患者成功建立 Stage Features。"
        )

    dataset = pd.concat(
        patient_features,
        ignore_index=True,
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("Stage Feature Dataset 完成")
    print("=" * 80)
    print(
        f"總 Epoch 數量：{len(dataset)}"
    )
    print(
        f"患者數量："
        f"{dataset['patient_id'].nunique()}"
    )

    print("\nStage 數量：")
    print(
        dataset["stage"]
        .value_counts()
        .to_string()
    )

    print(
        f"\n輸出檔案：{OUTPUT_FILE}"
    )

    if failures:
        print("\n失敗患者：")

        print(
            pd.DataFrame(
                failures
            ).to_string(
                index=False
            )
        )

    print("=" * 80)


if __name__ == "__main__":
    main()