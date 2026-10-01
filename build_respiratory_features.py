from pathlib import Path

import pandas as pd

from src.features.channel_mapper import (
    ChannelMapper,
)
from src.features.respiratory_feature_builder import (
    RespiratoryFeatureBuilder,
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
    / "respiratory_features.csv"
)

FAILURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "respiratory_feature_failures.csv"
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

    mapper = ChannelMapper(
        CHANNEL_CONFIG
    )

    builder = RespiratoryFeatureBuilder(
        channel_mapper=mapper
    )

    all_features: list[
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
            f"建立 Respiratory Features："
            f"{patient_id}"
        )
        print("=" * 80)

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
                )
            )

            if features.empty:
                raise RuntimeError(
                    "呼吸特徵結果為空。"
                )

            all_features.append(
                features
            )

            print(
                f"完成："
                f"{len(features)} epochs，"
                f"{len(features.columns)} columns"
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

    if not all_features:
        raise RuntimeError(
            "沒有任何患者成功建立"
            "呼吸特徵。"
        )

    dataset = pd.concat(
        all_features,
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
    print(
        "Respiratory Feature Dataset 完成"
    )
    print("=" * 80)

    print(
        f"總 Epoch 數量："
        f"{len(dataset)}"
    )

    print(
        f"患者數量："
        f"{dataset['patient_id'].nunique()}"
    )

    print(
        f"總欄位數量："
        f"{len(dataset.columns)}"
    )

    print("\nStage 數量：")

    print(
        dataset["stage"]
        .value_counts()
        .to_string()
    )

    print("\nEDF backend：")

    print(
        dataset[
            "respiratory_edf_backend"
        ]
        .value_counts()
        .to_string()
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