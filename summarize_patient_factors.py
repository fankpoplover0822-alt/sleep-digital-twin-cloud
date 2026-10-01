from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

INFERENCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "inference"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


FEATURE_IMPORTANCE_FILENAME = (
    "patient_feature_importance.csv"
)

GROUP_IMPORTANCE_FILENAME = (
    "patient_group_importance.csv"
)

SHAP_SUMMARY_FILENAME = (
    "patient_shap_summary.json"
)

PREDICTION_SUMMARY_FILENAME = (
    "arousal_prediction_summary.json"
)

EVALUATION_METRICS_FILENAME = (
    "arousal_evaluation_metrics.json"
)

PATIENT_METADATA_FILENAME = (
    "patient_metadata.json"
)


FEATURE_GROUP_LABELS = {
    "SPO2": "血氧與缺氧變化",
    "AIRFLOW": "鼻氣流與呼吸氣流",
    "RESPIRATORY_EFFORT": "胸腹呼吸努力",
    "EEG": "腦波活動",
    "EOG": "眼動訊號",
    "CARDIOVASCULAR": "心率與心血管訊號",
    "SNORE": "鼾聲訊號",
    "POSITION": "睡眠姿勢",
    "SLEEP_STAGE": "睡眠階段",
    "MOVEMENT": "肢體活動",
    "DEMOGRAPHICS": "基本身體資料",
    "DATA_QUALITY": "資料品質",
    "OTHER": "其他訊號",
}


def load_json(
    file_path: Path,
    required: bool = True,
) -> dict[str, Any]:
    if not file_path.exists():
        if required:
            raise FileNotFoundError(
                f"找不到 JSON：{file_path}"
            )

        return {}

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"JSON 內容不是 object：{file_path}"
        )

    return data


def safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(result):
        return None

    return result


def safe_int(
    value: Any,
) -> int | None:
    number = safe_float(value)

    if number is None:
        return None

    return int(round(number))


def safe_json_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (
            np.integer,
            np.int32,
            np.int64,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float32,
            np.float64,
        ),
    ):
        if not np.isfinite(value):
            return None

        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    try:
        if pd.isna(value):
            return None
    except (
        TypeError,
        ValueError,
    ):
        pass

    return value


def save_json(
    file_path: Path,
    data: dict[str, Any],
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=safe_json_value,
        )


def feature_group_label(
    group_name: str,
) -> str:
    return FEATURE_GROUP_LABELS.get(
        str(group_name),
        str(group_name),
    )


def describe_feature(
    feature_name: str,
) -> str:
    """
    將模型欄位名稱轉成較容易閱讀的中文描述。

    這些名稱是專案顯示用途，
    不代表正式臨床術語。
    """
    name = str(feature_name).lower()

    exact_labels = {
        "stage_n1": "處於 N1 淺睡期",
        "stage_n2": "處於 N2 睡眠期",
        "stage_n3": "處於 N3 深睡期",
        "stage_rem": "處於 REM 睡眠期",
        "stage_w": "處於清醒期",
        "spo2_mean": "平均血氧濃度",
        "spo2_min": "最低血氧濃度",
        "spo2_median": "血氧中位數",
        "spo2_std": "血氧波動程度",
        "spo2_std_resp": "呼吸分析中的血氧波動程度",
        "spo2_range": "血氧變化範圍",
        "spo2_range_resp": "呼吸分析中的血氧變化範圍",
        "spo2_iqr": "血氧四分位距",
        "spo2_iqr_resp": "呼吸分析中的血氧四分位距",
        "spo2_diff_std": "相鄰血氧變化的不穩定程度",
        "spo2_diff_std_resp": "呼吸分析中的相鄰血氧變化",
        "spo2_below_88_fraction": "血氧低於 88% 的時間比例",
        "spo2_below_90_fraction": "血氧低於 90% 的時間比例",
        "flow_p75": "鼻氣流振幅上四分位數",
        "flow_max": "鼻氣流最大振幅",
        "flow_max_resp": "呼吸分析中的鼻氣流最大振幅",
        "flow_iqr": "鼻氣流振幅四分位距",
        "flow_iqr_resp": "呼吸分析中的鼻氣流振幅四分位距",
        "flow_respiratory_rate_bpm": "鼻氣流估算呼吸頻率",
        "thermistor_std": "熱敏式氣流訊號波動",
        "thermistor_range": "熱敏式氣流訊號範圍",
        "thermistor_iqr": "熱敏式氣流訊號四分位距",
        "thermistor_amplitude_p90_p10": (
            "熱敏式氣流振幅 P90 與 P10 差異"
        ),
        "thorax_respiratory_rate_bpm": (
            "胸部呼吸訊號估算呼吸頻率"
        ),
        "abdomen_respiratory_rate_bpm": (
            "腹部呼吸訊號估算呼吸頻率"
        ),
        "thorax_abdomen_correlation": (
            "胸腹呼吸同步程度"
        ),
        "thorax_abdomen_lag_seconds": (
            "胸腹呼吸訊號時間差"
        ),
        "eeg_c3_beta_relative": (
            "C3 腦波 Beta 相對能量"
        ),
        "eeg_c4_beta_relative": (
            "C4 腦波 Beta 相對能量"
        ),
        "heart_rate_mean": "平均心率",
        "heart_rate_diff_std": (
            "心率變化的不穩定程度"
        ),
        "snore_range": "鼾聲振幅範圍",
        "snore_min": "鼾聲最低振幅",
        "position_change_count": "姿勢改變次數",
    }

    if name in exact_labels:
        return exact_labels[name]

    if name.startswith("stage_"):
        return (
            "睡眠階段："
            + name.replace(
                "stage_",
                "",
            ).upper()
        )

    if "spo2" in name:
        if "std" in name:
            return "血氧波動程度"

        if "range" in name:
            return "血氧變化範圍"

        if "iqr" in name:
            return "血氧分布變化"

        if "below" in name:
            return "低血氧時間比例"

        if "drop" in name:
            return "血氧下降程度"

        if "min" in name:
            return "最低血氧"

        if "mean" in name:
            return "平均血氧"

        return "血氧相關特徵"

    if (
        "flow" in name
        or "thermistor" in name
    ):
        if "rate" in name:
            return "呼吸頻率"

        if "amplitude" in name:
            return "氣流振幅"

        if "range" in name:
            return "氣流變化範圍"

        if "iqr" in name:
            return "氣流振幅分布"

        return "鼻氣流相關特徵"

    if (
        "thorax" in name
        or "abdomen" in name
    ):
        if "correlation" in name:
            return "胸腹呼吸同步程度"

        if "lag" in name:
            return "胸腹呼吸時間差"

        return "胸腹呼吸努力特徵"

    if "eeg" in name:
        return "腦波活動特徵"

    if (
        "heart" in name
        or "pulse" in name
        or "ecg" in name
        or "ekg" in name
    ):
        return "心率與心血管特徵"

    if "snore" in name:
        return "鼾聲相關特徵"

    if "position" in name:
        return "睡眠姿勢相關特徵"

    return str(feature_name)


def determine_direction(
    signed_shap: float,
    tolerance: float = 1e-8,
) -> str:
    if signed_shap > tolerance:
        return "INCREASE_RISK"

    if signed_shap < -tolerance:
        return "DECREASE_RISK"

    return "NEUTRAL"


def build_feature_records(
    data: pd.DataFrame,
    limit: int,
    direction: str,
) -> list[dict[str, Any]]:
    if data.empty:
        return []

    if direction == "INCREASE_RISK":
        selected = data[
            data["mean_signed_shap"] > 0
        ].copy()

        selected = selected.sort_values(
            [
                "mean_signed_shap",
                "mean_absolute_shap",
            ],
            ascending=False,
        )

    elif direction == "DECREASE_RISK":
        selected = data[
            data["mean_signed_shap"] < 0
        ].copy()

        selected["negative_magnitude"] = (
            -selected["mean_signed_shap"]
        )

        selected = selected.sort_values(
            [
                "negative_magnitude",
                "mean_absolute_shap",
            ],
            ascending=False,
        )

    else:
        selected = data.copy()

        selected = selected.sort_values(
            "mean_absolute_shap",
            ascending=False,
        )

    selected = selected.head(
        limit
    )

    records: list[
        dict[str, Any]
    ] = []

    for _, row in selected.iterrows():
        feature = str(
            row["feature"]
        )

        group = str(
            row["feature_group"]
        )

        mean_signed = safe_float(
            row["mean_signed_shap"]
        )

        mean_absolute = safe_float(
            row["mean_absolute_shap"]
        )

        positive_fraction = safe_float(
            row.get(
                "positive_effect_fraction"
            )
        )

        records.append(
            {
                "feature": feature,
                "description": (
                    describe_feature(
                        feature
                    )
                ),
                "feature_group": group,
                "feature_group_label": (
                    feature_group_label(
                        group
                    )
                ),
                "mean_absolute_shap": (
                    mean_absolute
                ),
                "mean_signed_shap": (
                    mean_signed
                ),
                "effect_direction": (
                    determine_direction(
                        mean_signed
                        if mean_signed
                        is not None
                        else 0.0
                    )
                ),
                "positive_effect_fraction": (
                    positive_fraction
                ),
            }
        )

    return records


def build_group_records(
    data: pd.DataFrame,
    limit: int,
) -> list[dict[str, Any]]:
    if data.empty:
        return []

    selected = (
        data.sort_values(
            "total_mean_absolute_shap",
            ascending=False,
        )
        .head(limit)
    )

    records: list[
        dict[str, Any]
    ] = []

    for _, row in selected.iterrows():
        group = str(
            row["feature_group"]
        )

        signed_value = safe_float(
            row[
                "total_mean_signed_shap"
            ]
        )

        records.append(
            {
                "rank": safe_int(
                    row.get("rank")
                ),
                "feature_group": group,
                "feature_group_label": (
                    feature_group_label(
                        group
                    )
                ),
                "total_mean_absolute_shap": (
                    safe_float(
                        row[
                            "total_mean_absolute_shap"
                        ]
                    )
                ),
                "total_mean_signed_shap": (
                    signed_value
                ),
                "effect_direction": (
                    determine_direction(
                        signed_value
                        if signed_value
                        is not None
                        else 0.0
                    )
                ),
                "feature_count": safe_int(
                    row["feature_count"]
                ),
            }
        )

    return records


def format_factor_sentence(
    factor: dict[str, Any],
) -> str:
    description = factor[
        "description"
    ]

    signed_shap = factor[
        "mean_signed_shap"
    ]

    positive_fraction = factor.get(
        "positive_effect_fraction"
    )

    sentence = (
        f"{description}為模型重要因素"
    )

    if signed_shap is not None:
        if signed_shap > 0:
            sentence += (
                "，平均將預測往較高風險方向推動"
            )
        elif signed_shap < 0:
            sentence += (
                "，平均將預測往較低風險方向推動"
            )

    if positive_fraction is not None:
        sentence += (
            f"；在警報 Epoch 中有 "
            f"{positive_fraction:.1%} "
            "呈現正向風險貢獻"
        )

    return sentence + "。"


def build_clinical_text(
    patient_id: str,
    patient_metadata: dict[str, Any],
    prediction_summary: dict[str, Any],
    evaluation_metrics: dict[str, Any],
    top_groups: list[dict[str, Any]],
    increasing_factors: list[dict[str, Any]],
    decreasing_factors: list[dict[str, Any]],
) -> str:
    age = patient_metadata.get(
        "age"
    )

    bmi = patient_metadata.get(
        "BMI"
    )

    sex = patient_metadata.get(
        "sex"
    )

    epoch_count = prediction_summary.get(
        "epoch_count"
    )

    alert_count = prediction_summary.get(
        "alert_count"
    )

    alert_fraction = prediction_summary.get(
        "alert_fraction"
    )

    average_probability = (
        prediction_summary.get(
            "average_probability"
        )
    )

    maximum_probability = (
        prediction_summary.get(
            "maximum_probability"
        )
    )

    lines: list[str] = []

    lines.append(
        f"患者 {patient_id} 的整晚 "
        "Arousal 模型因素摘要如下。"
    )

    demographic_parts: list[str] = []

    if sex is not None:
        demographic_parts.append(
            f"sex={sex}"
        )

    if age is not None:
        demographic_parts.append(
            f"年齡 {age} 歲"
        )

    if bmi is not None:
        demographic_parts.append(
            f"BMI {bmi}"
        )

    if demographic_parts:
        lines.append(
            "基本資料："
            + "、".join(
                demographic_parts
            )
            + "。"
        )

    if epoch_count is not None:
        risk_text = (
            f"共分析 {epoch_count} 個 "
            "30 秒 Epoch"
        )

        if (
            alert_count is not None
            and alert_fraction is not None
        ):
            risk_text += (
                f"，其中 {alert_count} 個 "
                f"達到警報門檻"
                f"（{float(alert_fraction):.1%}）"
            )

        if average_probability is not None:
            risk_text += (
                f"，平均預測風險為 "
                f"{float(average_probability):.3f}"
            )

        if maximum_probability is not None:
            risk_text += (
                f"，最高風險為 "
                f"{float(maximum_probability):.3f}"
            )

        lines.append(
            risk_text + "。"
        )

    if top_groups:
        group_labels = [
            record[
                "feature_group_label"
            ]
            for record in top_groups[:4]
        ]

        lines.append(
            "模型整體最關注的生理系統為："
            + "、".join(group_labels)
            + "。"
        )

    if increasing_factors:
        factor_descriptions = [
            factor[
                "description"
            ]
            for factor in increasing_factors[:5]
        ]

        lines.append(
            "警報期間主要提高模型風險的因素包括："
            + "、".join(
                factor_descriptions
            )
            + "。"
        )

    if decreasing_factors:
        factor_descriptions = [
            factor[
                "description"
            ]
            for factor in decreasing_factors[:4]
        ]

        lines.append(
            "警報期間部分降低模型風險的因素包括："
            + "、".join(
                factor_descriptions
            )
            + "。"
        )

    roc_auc = evaluation_metrics.get(
        "roc_auc"
    )

    recall = evaluation_metrics.get(
        "recall"
    )

    specificity = evaluation_metrics.get(
        "specificity"
    )

    precision = evaluation_metrics.get(
        "precision"
    )

    if roc_auc is not None:
        evaluation_text = (
            "此患者具真實 Event Grid，"
            f"獨立驗證 ROC-AUC 為 "
            f"{float(roc_auc):.3f}"
        )

        if recall is not None:
            evaluation_text += (
                f"，Recall 為 "
                f"{float(recall):.3f}"
            )

        if specificity is not None:
            evaluation_text += (
                f"，Specificity 為 "
                f"{float(specificity):.3f}"
            )

        if precision is not None:
            evaluation_text += (
                f"，Precision 為 "
                f"{float(precision):.3f}"
            )

        lines.append(
            evaluation_text + "。"
        )

    lines.append(
        "以上為模型特徵關聯性解釋，"
        "不能直接視為生理因果關係，"
        "亦不能單獨作為臨床診斷或治療決策。"
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "將 Patient-Level SHAP 結果整理成"
            "患者生理因素摘要。"
        )
    )

    parser.add_argument(
        "--patient-id",
        required=True,
        help=(
            "患者 ID，例如："
            "20201014T221256 - d25c6"
        ),
    )

    parser.add_argument(
        "--top-groups",
        type=int,
        default=8,
        help=(
            "輸出生理系統數量，預設 8。"
        ),
    )

    parser.add_argument(
        "--top-increasing",
        type=int,
        default=10,
        help=(
            "輸出提高風險特徵數量，預設 10。"
        ),
    )

    parser.add_argument(
        "--top-decreasing",
        type=int,
        default=10,
        help=(
            "輸出降低風險特徵數量，預設 10。"
        ),
    )

    args = parser.parse_args()

    patient_id = str(
        args.patient_id
    ).strip()

    top_group_count = max(
        int(args.top_groups),
        1,
    )

    top_increasing_count = max(
        int(args.top_increasing),
        1,
    )

    top_decreasing_count = max(
        int(args.top_decreasing),
        1,
    )

    inference_folder = (
        INFERENCE_ROOT
        / patient_id
    )

    shap_folder = (
        inference_folder
        / "shap"
        / "patient_level"
    )

    processed_folder = (
        PROCESSED_ROOT
        / patient_id
    )

    feature_file = (
        shap_folder
        / FEATURE_IMPORTANCE_FILENAME
    )

    group_file = (
        shap_folder
        / GROUP_IMPORTANCE_FILENAME
    )

    shap_summary_file = (
        shap_folder
        / SHAP_SUMMARY_FILENAME
    )

    prediction_summary_file = (
        inference_folder
        / PREDICTION_SUMMARY_FILENAME
    )

    evaluation_metrics_file = (
        inference_folder
        / EVALUATION_METRICS_FILENAME
    )

    patient_metadata_file = (
        processed_folder
        / PATIENT_METADATA_FILENAME
    )

    required_files = [
        feature_file,
        group_file,
        shap_summary_file,
        prediction_summary_file,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                "找不到必要檔案："
                f"{file_path}"
            )

    print("=" * 80)
    print("患者 SHAP 因素摘要")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    feature_data = pd.read_csv(
        feature_file
    )

    group_data = pd.read_csv(
        group_file
    )

    shap_summary = load_json(
        shap_summary_file
    )

    prediction_summary = load_json(
        prediction_summary_file
    )

    evaluation_metrics = load_json(
        evaluation_metrics_file,
        required=False,
    )

    patient_metadata = load_json(
        patient_metadata_file,
        required=False,
    )

    required_feature_columns = {
        "analysis",
        "feature",
        "feature_group",
        "mean_absolute_shap",
        "mean_signed_shap",
    }

    missing_feature_columns = (
        required_feature_columns
        - set(feature_data.columns)
    )

    if missing_feature_columns:
        raise RuntimeError(
            "特徵排名缺少欄位："
            f"{sorted(missing_feature_columns)}"
        )

    required_group_columns = {
        "analysis",
        "feature_group",
        "total_mean_absolute_shap",
        "total_mean_signed_shap",
        "feature_count",
    }

    missing_group_columns = (
        required_group_columns
        - set(group_data.columns)
    )

    if missing_group_columns:
        raise RuntimeError(
            "群組排名缺少欄位："
            f"{sorted(missing_group_columns)}"
        )

    numeric_feature_columns = [
        "mean_absolute_shap",
        "mean_signed_shap",
        "positive_effect_fraction",
        "negative_effect_fraction",
    ]

    for column in numeric_feature_columns:
        if column in feature_data.columns:
            feature_data[column] = (
                pd.to_numeric(
                    feature_data[column],
                    errors="coerce",
                )
            )

    numeric_group_columns = [
        "total_mean_absolute_shap",
        "total_mean_signed_shap",
        "feature_count",
    ]

    for column in numeric_group_columns:
        group_data[column] = (
            pd.to_numeric(
                group_data[column],
                errors="coerce",
            )
        )

    alert_features = feature_data[
        feature_data["analysis"]
        == "ALERT_EPOCHS"
    ].copy()

    all_features = feature_data[
        feature_data["analysis"]
        == "ALL_EPOCHS"
    ].copy()

    alert_groups = group_data[
        group_data["analysis"]
        == "ALERT_EPOCHS"
    ].copy()

    all_groups = group_data[
        group_data["analysis"]
        == "ALL_EPOCHS"
    ].copy()

    if alert_features.empty:
        raise RuntimeError(
            "找不到 ALERT_EPOCHS 特徵資料。"
        )

    if alert_groups.empty:
        raise RuntimeError(
            "找不到 ALERT_EPOCHS 群組資料。"
        )

    top_model_attention_groups = (
        build_group_records(
            data=alert_groups,
            limit=top_group_count,
        )
    )

    top_all_night_groups = (
        build_group_records(
            data=all_groups,
            limit=top_group_count,
        )
    )

    increasing_factors = (
        build_feature_records(
            data=alert_features,
            limit=top_increasing_count,
            direction="INCREASE_RISK",
        )
    )

    decreasing_factors = (
        build_feature_records(
            data=alert_features,
            limit=top_decreasing_count,
            direction="DECREASE_RISK",
        )
    )

    all_night_top_features = (
        build_feature_records(
            data=all_features,
            limit=15,
            direction="ALL",
        )
    )

    clinical_text = build_clinical_text(
        patient_id=patient_id,
        patient_metadata=patient_metadata,
        prediction_summary=(
            prediction_summary
        ),
        evaluation_metrics=(
            evaluation_metrics
        ),
        top_groups=(
            top_model_attention_groups
        ),
        increasing_factors=(
            increasing_factors
        ),
        decreasing_factors=(
            decreasing_factors
        ),
    )

    output_folder = (
        inference_folder
        / "factor_summary"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_json_file = (
        output_folder
        / "patient_factor_summary.json"
    )

    summary_csv_file = (
        output_folder
        / "patient_factor_summary.csv"
    )

    increasing_file = (
        output_folder
        / "risk_increasing_factors.csv"
    )

    decreasing_file = (
        output_folder
        / "risk_decreasing_factors.csv"
    )

    group_file_output = (
        output_folder
        / "physiological_group_summary.csv"
    )

    clinical_text_file = (
        output_folder
        / "patient_factor_summary.txt"
    )

    summary = {
        "patient_id": patient_id,
        "sex": patient_metadata.get(
            "sex",
            prediction_summary.get("sex"),
        ),
        "age": patient_metadata.get(
            "age",
            prediction_summary.get("age"),
        ),
        "BMI": patient_metadata.get(
            "BMI",
            prediction_summary.get("BMI"),
        ),
        "epoch_count": (
            prediction_summary.get(
                "epoch_count"
            )
        ),
        "alert_epoch_count": (
            prediction_summary.get(
                "alert_count"
            )
        ),
        "alert_fraction": (
            prediction_summary.get(
                "alert_fraction"
            )
        ),
        "average_probability": (
            prediction_summary.get(
                "average_probability"
            )
        ),
        "maximum_probability": (
            prediction_summary.get(
                "maximum_probability"
            )
        ),
        "top_model_attention_groups_alert_epochs": (
            top_model_attention_groups
        ),
        "top_model_attention_groups_all_night": (
            top_all_night_groups
        ),
        "main_risk_increasing_factors": (
            increasing_factors
        ),
        "main_risk_decreasing_factors": (
            decreasing_factors
        ),
        "all_night_top_features": (
            all_night_top_features
        ),
        "independent_evaluation": {
            "roc_auc": (
                evaluation_metrics.get(
                    "roc_auc"
                )
            ),
            "average_precision": (
                evaluation_metrics.get(
                    "average_precision"
                )
            ),
            "balanced_accuracy": (
                evaluation_metrics.get(
                    "balanced_accuracy"
                )
            ),
            "precision": (
                evaluation_metrics.get(
                    "precision"
                )
            ),
            "recall": (
                evaluation_metrics.get(
                    "recall"
                )
            ),
            "specificity": (
                evaluation_metrics.get(
                    "specificity"
                )
            ),
            "f1": (
                evaluation_metrics.get(
                    "f1"
                )
            ),
            "episode_recall": (
                evaluation_metrics.get(
                    "episode_recall"
                )
            ),
        },
        "clinical_readable_summary": (
            clinical_text
        ),
        "interpretation_note": (
            "此摘要反映模型對不同特徵的"
            "敏感度與統計關聯性，"
            "不代表生理因果效應。"
        ),
        "clinical_safety_note": (
            "本系統目前屬研究用途，"
            "不可單獨作為診斷、手術、"
            "CPAP、APAP 或藥物選擇依據。"
        ),
    }

    save_json(
        summary_json_file,
        summary,
    )

    summary_row = {
        "patient_id": patient_id,
        "sex": summary["sex"],
        "age": summary["age"],
        "BMI": summary["BMI"],
        "epoch_count": summary[
            "epoch_count"
        ],
        "alert_epoch_count": summary[
            "alert_epoch_count"
        ],
        "alert_fraction": summary[
            "alert_fraction"
        ],
        "average_probability": summary[
            "average_probability"
        ],
        "maximum_probability": summary[
            "maximum_probability"
        ],
        "top_group_1": (
            top_model_attention_groups[0][
                "feature_group_label"
            ]
            if len(
                top_model_attention_groups
            ) >= 1
            else None
        ),
        "top_group_2": (
            top_model_attention_groups[1][
                "feature_group_label"
            ]
            if len(
                top_model_attention_groups
            ) >= 2
            else None
        ),
        "top_group_3": (
            top_model_attention_groups[2][
                "feature_group_label"
            ]
            if len(
                top_model_attention_groups
            ) >= 3
            else None
        ),
        "top_increasing_factor_1": (
            increasing_factors[0][
                "description"
            ]
            if len(increasing_factors) >= 1
            else None
        ),
        "top_increasing_factor_2": (
            increasing_factors[1][
                "description"
            ]
            if len(increasing_factors) >= 2
            else None
        ),
        "top_increasing_factor_3": (
            increasing_factors[2][
                "description"
            ]
            if len(increasing_factors) >= 3
            else None
        ),
        "roc_auc": (
            evaluation_metrics.get(
                "roc_auc"
            )
        ),
        "recall": (
            evaluation_metrics.get(
                "recall"
            )
        ),
        "specificity": (
            evaluation_metrics.get(
                "specificity"
            )
        ),
        "precision": (
            evaluation_metrics.get(
                "precision"
            )
        ),
    }

    pd.DataFrame(
        [summary_row]
    ).to_csv(
        summary_csv_file,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        increasing_factors
    ).to_csv(
        increasing_file,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        decreasing_factors
    ).to_csv(
        decreasing_file,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        top_model_attention_groups
    ).to_csv(
        group_file_output,
        index=False,
        encoding="utf-8-sig",
    )

    with clinical_text_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            clinical_text
        )

    print()
    print("=" * 80)
    print("患者因素摘要完成")
    print("=" * 80)

    print(
        f"Patient ID：{patient_id}"
    )

    print()
    print("警報期間模型最關注的生理系統：")

    for index, group in enumerate(
        top_model_attention_groups,
        start=1,
    ):
        print(
            f"{index}. "
            f"{group['feature_group_label']} "
            f"(absolute SHAP="
            f"{group['total_mean_absolute_shap']:.6f}, "
            f"signed SHAP="
            f"{group['total_mean_signed_shap']:.6f})"
        )

    print()
    print("主要提高模型風險的因素：")

    for index, factor in enumerate(
        increasing_factors,
        start=1,
    ):
        print(
            f"{index}. "
            f"{factor['description']} "
            f"(mean signed SHAP="
            f"{factor['mean_signed_shap']:.6f})"
        )

    print()
    print("主要降低模型風險的因素：")

    for index, factor in enumerate(
        decreasing_factors,
        start=1,
    ):
        print(
            f"{index}. "
            f"{factor['description']} "
            f"(mean signed SHAP="
            f"{factor['mean_signed_shap']:.6f})"
        )

    print()
    print("=" * 80)
    print("自動產生的患者摘要")
    print("=" * 80)

    print(
        clinical_text
    )

    print()
    print("=" * 80)
    print("輸出檔案")
    print("=" * 80)

    print(
        f"完整摘要 JSON："
        f"{summary_json_file}"
    )

    print(
        f"摘要 CSV："
        f"{summary_csv_file}"
    )

    print(
        f"提高風險因素："
        f"{increasing_file}"
    )

    print(
        f"降低風險因素："
        f"{decreasing_file}"
    )

    print(
        f"生理群組摘要："
        f"{group_file_output}"
    )

    print(
        f"文字摘要："
        f"{clinical_text_file}"
    )

    print()
    print(
        "注意：模型因素不等於"
        "已證明的生理因果因素。"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()