from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predict_new_patient import (
    build_inference_dataset,
)
from src.importers.new_patient_importer import (
    NewPatientImporter,
)
from src.inference.arousal_inference import (
    ArousalInferenceEngine,
)
from src.continual_learning.service import (
    resolve_arousal_artifacts,
)



PROJECT_ROOT = Path(
    __file__
).resolve().parent

INCOMING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "incoming"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INFERENCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "inference"
)

DEMOGRAPHICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "demographics"
    / "patients.xlsx.xlsx"
)

AROUSAL_ARTIFACTS = resolve_arousal_artifacts(PROJECT_ROOT)
MODEL_FILE = AROUSAL_ARTIFACTS["model"]
FEATURE_CONFIG_FILE = AROUSAL_ARTIFACTS["features"]
THRESHOLD_FILE = AROUSAL_ARTIFACTS["thresholds"]


def to_json_safe(
    value: Any,
) -> Any:
    """
    將 pandas、numpy、Path 等物件轉成
    可以安全寫入 JSON 的格式。
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (
        TypeError,
        ValueError,
    ):
        pass

    if isinstance(value, Path):
        return str(value)

    if isinstance(
        value,
        (
            pd.Timestamp,
        ),
    ):
        return value.isoformat(
            sep=" "
        )

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()
        except (
            TypeError,
            ValueError,
        ):
            pass

    return value


def load_patient_metadata(
    metadata_file: Path,
) -> dict[str, Any]:
    """
    讀取 process_incoming_patient.py
    產生的 patient_metadata.json。
    """
    if not metadata_file.exists():
        return {}

    with metadata_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            "patient_metadata.json "
            "內容不是 JSON object。"
        )

    return data


def save_prediction_summary(
    output_file: Path,
    summary: dict[str, Any],
) -> None:
    safe_summary = {
        key: to_json_safe(value)
        for key, value
        in summary.items()
    }

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            safe_summary,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def validate_required_files(
    patient_folder: Path,
    stages_file: Path,
) -> None:
    if not patient_folder.exists():
        raise FileNotFoundError(
            "找不到 incoming 患者資料夾："
            f"{patient_folder}"
        )

    if not patient_folder.is_dir():
        raise NotADirectoryError(
            "incoming 患者路徑不是資料夾："
            f"{patient_folder}"
        )

    if not stages_file.exists():
        raise FileNotFoundError(
            "找不到已對齊 Stage："
            f"{stages_file}\n"
            "請先執行：\n"
            "python process_incoming_patient.py "
            '--patient-id "患者 ID"'
        )


def print_highest_risk_epochs(
    predictions: pd.DataFrame,
    top_n: int = 20,
) -> None:
    print()
    print(
        f"最高風險的 {top_n} 個 Epoch："
    )

    display_columns = [
        column
        for column in [
            "epoch_index",
            "start_time",
            "stage",
            "quality_core_features_valid",
            "arousal_next_30s_probability",
            "arousal_next_30s_alert",
            "arousal_risk_level",
        ]
        if column in predictions.columns
    ]

    highest = (
        predictions.sort_values(
            "arousal_next_30s_probability",
            ascending=False,
        )
        .head(top_n)
    )

    print(
        highest[
            display_columns
        ].to_string(
            index=False
        )
    )



def predict_patient(
    patient_id: str,
    top_n: int = 20,
) -> None:
    patient_id = str(
        patient_id
    ).strip()

    top_n = max(
        int(top_n),
        1,
    )

    print(
        f"準備執行 Arousal 推論："
        f"{patient_id}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "對 data/incoming 中已完成前處理的"
            "新患者執行 Arousal Next 30s 推論。"
        )
    )

    parser.add_argument(
        "--patient-id",
        required=True,
        help=(
            "data/incoming 中的患者資料夾名稱，"
            "例如：20201014T221256 - d25c6"
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="顯示最高風險 Epoch 數量，預設 20。",
    )
    parser.add_argument(
        "--research-preview",
        action="store_true",
        help="以最新 Challenger 產生隔離的研究測試輸出",
    )

    args = parser.parse_args()

    patient_id = str(
        args.patient_id
    ).strip()

    top_n = max(
        int(args.top_n),
        1,
    )

    patient_folder = (
        INCOMING_ROOT
        / patient_id
    )

    processed_folder = (
        PROCESSED_ROOT
        / patient_id
    )

    stages_file = (
        processed_folder
        / "stages_aligned.csv"
    )

    metadata_file = (
        processed_folder
        / "patient_metadata.json"
    )

    validate_required_files(
        patient_folder=patient_folder,
        stages_file=stages_file,
    )

    print("=" * 80)
    print("Incoming 新患者 Arousal 推論")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    print(
        f"Incoming 資料夾："
        f"{patient_folder}"
    )

    print(
        f"Stage 對齊檔："
        f"{stages_file}"
    )

    # ============================================================
    # 1. 尋找新患者 EDF 與基本資料
    # ============================================================
    importer = NewPatientImporter(
        patient_folder=patient_folder,
        demographics_file=(
            DEMOGRAPHICS_FILE
        ),
    )

    patient_files = importer.inspect()

    print()
    importer.print_summary(
        patient_files
    )

    edf_file = patient_files.edf_file

    # ============================================================
    # 2. 建立模型推論特徵
    # ============================================================
    print()
    print("=" * 80)
    print("建立 Incoming 患者模型特徵")
    print("=" * 80)

    inference_data = (
        build_inference_dataset(
            patient_id=patient_id,
            edf_file=edf_file,
            stages_file=stages_file,
        )
    )

    if inference_data.empty:
        raise RuntimeError(
            "建立出的推論資料為空。"
        )

    feature_output_folder = (
        INFERENCE_ROOT
        / patient_id
    )

    feature_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    inference_features_file = (
        feature_output_folder
        / "inference_features.csv"
    )

    inference_data.to_csv(
        inference_features_file,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("推論資料建立完成")
    print("=" * 80)

    print(
        f"資料形狀："
        f"{inference_data.shape}"
    )

    print(
        f"Epoch 數量："
        f"{len(inference_data)}"
    )

    print(
        f"推論特徵檔："
        f"{inference_features_file}"
    )

    if (
        "quality_core_features_valid"
        in inference_data.columns
    ):
        valid_count = int(
            inference_data[
                "quality_core_features_valid"
            ]
            .astype(bool)
            .sum()
        )

        print(
            "核心特徵品質合格："
            f"{valid_count}"
            f"/{len(inference_data)}"
        )
    else:
        valid_count = len(
            inference_data
        )

    # ============================================================
    # 3. 載入模型並推論
    # ============================================================
    print()
    print("=" * 80)
    print("載入 Arousal 模型並推論")
    print("=" * 80)

    active_artifacts = resolve_arousal_artifacts(
        PROJECT_ROOT,
        use_latest_challenger=bool(args.research_preview),
    )
    engine = ArousalInferenceEngine(
        model_file=active_artifacts["model"],
        feature_config_file=active_artifacts["features"],
        threshold_file=active_artifacts["thresholds"],
    )

    # age/BMI are model inputs, not merely columns appended after prediction.
    inference_data = inference_data.copy()
    inference_data["age"] = patient_files.age
    inference_data["BMI"] = patient_files.bmi

    predictions = engine.predict(
        inference_data
    )

    if predictions.empty:
        raise RuntimeError(
            "模型沒有產生任何推論結果。"
        )

    # ============================================================
    # 4. 將基本資料加入輸出
    # ============================================================
    predictions.insert(
        1,
        "sex",
        patient_files.sex,
    )

    predictions.insert(
        2,
        "age",
        patient_files.age,
    )

    predictions.insert(
        3,
        "BMI",
        patient_files.bmi,
    )

    predictions[
        "is_independent_incoming_patient"
    ] = True

    # ============================================================
    # 5. 儲存輸出
    # ============================================================
    output_folder = INFERENCE_ROOT / patient_id
    if args.research_preview:
        output_folder = output_folder / "research_preview" / "arousal_30s"

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_file = (
        output_folder
        / "arousal_predictions.csv"
    )

    predictions.to_csv(
        predictions_file,
        index=False,
        encoding="utf-8-sig",
    )

    engine.print_summary(
        predictions
    )

    # ============================================================
    # 6. 建立患者推論摘要
    # ============================================================
    alert_count = int(
        predictions[
            "arousal_next_30s_alert"
        ].astype(bool).sum()
    )

    total_epochs = len(
        predictions
    )

    average_probability = float(
        pd.to_numeric(
            predictions[
                "arousal_next_30s_probability"
            ],
            errors="coerce",
        ).mean()
    )

    maximum_probability = float(
        pd.to_numeric(
            predictions[
                "arousal_next_30s_probability"
            ],
            errors="coerce",
        ).max()
    )

    risk_counts = (
        predictions[
            "arousal_risk_level"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    patient_metadata = (
        load_patient_metadata(
            metadata_file
        )
    )

    summary = {
        "patient_id": patient_id,
        "output_mode": active_artifacts.get("active_mode"),
        "model_version": active_artifacts["model_dir"].name,
        "independent_incoming_patient": True,
        "edf_file": str(edf_file),
        "stages_file": str(
            stages_file
        ),
        "predictions_file": str(
            predictions_file
        ),
        "sex": patient_files.sex,
        "age": patient_files.age,
        "BMI": patient_files.bmi,
        "demographics_found": (
            patient_files.demographics_found
        ),
        "epoch_count": total_epochs,
        "core_quality_valid_count": (
            valid_count
        ),
        "core_quality_valid_fraction": (
            valid_count
            / total_epochs
            if total_epochs > 0
            else None
        ),
        "alert_threshold": (
            engine.alert_threshold
        ),
        "alert_count": alert_count,
        "alert_fraction": (
            alert_count
            / total_epochs
            if total_epochs > 0
            else None
        ),
        "average_probability": (
            average_probability
        ),
        "maximum_probability": (
            maximum_probability
        ),
        "low_risk_count": int(
            risk_counts.get(
                "LOW",
                0,
            )
        ),
        "moderate_risk_count": int(
            risk_counts.get(
                "MODERATE",
                0,
            )
        ),
        "high_risk_count": int(
            risk_counts.get(
                "HIGH",
                0,
            )
        ),
        "very_high_risk_count": int(
            risk_counts.get(
                "VERY_HIGH",
                0,
            )
        ),
        "original_edf_start": (
            patient_metadata.get(
                "original_edf_start"
            )
        ),
        "corrected_edf_start": (
            patient_metadata.get(
                "corrected_edf_start"
            )
        ),
        "edf_start_was_corrected": (
            patient_metadata.get(
                "edf_start_was_corrected"
            )
        ),
        "stage_coverage_fraction": (
            patient_metadata.get(
                "stage_coverage_fraction"
            )
        ),
        "event_coverage_fraction": (
            patient_metadata.get(
                "event_coverage_fraction"
            )
        ),
        "model_target": (
            engine.target_column
        ),
        "important_note": (
            "此結果為研究型模型輸出，"
            "不可單獨作為臨床診斷或治療決策。"
        ),
    }

    summary_file = (
        output_folder
        / "arousal_prediction_summary.json"
    )

    save_prediction_summary(
        output_file=summary_file,
        summary=summary,
    )

    summary_csv_file = (
        output_folder
        / "arousal_prediction_summary.csv"
    )

    pd.DataFrame(
        [
            {
                key: to_json_safe(
                    value
                )
                for key, value
                in summary.items()
            }
        ]
    ).to_csv(
        summary_csv_file,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("Incoming 新患者推論完成")
    print("=" * 80)

    print(
        f"Patient ID："
        f"{patient_id}"
    )

    print(
        f"基本資料："
        f"sex={patient_files.sex}, "
        f"age={patient_files.age}, "
        f"BMI={patient_files.bmi}"
    )

    print(
        f"Epoch："
        f"{total_epochs}"
    )

    print(
        f"警報門檻："
        f"{engine.alert_threshold:.3f}"
    )

    print(
        f"警報 Epoch："
        f"{alert_count}"
        f"/{total_epochs}"
    )

    print(
        f"警報比例："
        f"{alert_count / total_epochs:.2%}"
    )

    print(
        f"平均風險："
        f"{average_probability:.4f}"
    )

    print(
        f"最大風險："
        f"{maximum_probability:.4f}"
    )

    print(
        f"\n預測明細："
        f"{predictions_file}"
    )

    print(
        f"摘要 JSON："
        f"{summary_file}"
    )

    print(
        f"摘要 CSV："
        f"{summary_csv_file}"
    )

    print("=" * 80)

    print_highest_risk_epochs(
        predictions=predictions,
        top_n=top_n,
    )


if __name__ == "__main__":
    main()
