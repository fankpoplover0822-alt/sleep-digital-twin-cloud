from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

from clinical_modules.parser_schemas import MODULE_SPECS


ROOT = Path.home() / "Downloads" / "clinical_test_bank" / "clinical_test_bank"
OUTPUT = ROOT / "網頁上傳測試_依選單分類"

LABELS = {
    "PAP": "01_PAP_治療成效與耐受度",
    "FOLLOW_UP": "02_治療回診與療效",
    "DEVICE_DATA": "03_穿戴式手錶與感測器同步",
    "ENT": "04_鼻內視鏡與耳鼻喉檢查",
    "DISE": "05_DISE_睡眠內視鏡",
    "IMAGING": "06_上呼吸道影像_CT_CBCT_MRI",
    "DENTAL_CRANIOFACIAL": "07_顎面與牙科評估",
    "SURGERY_HISTORY": "08_上呼吸道手術史",
    "MEDICATION": "09_目前用藥與失眠診斷",
    "SLEEP_QUESTIONNAIRE": "10_睡眠症狀與問卷",
    "PSYCHIATRIC": "11_心理與精神狀態",
    "PATIENT_PREFERENCE": "12_患者治療偏好與接受度",
    "HYPOXEMIA": "13_低血氧原因評估",
    "ABG": "14_血液氣體與低通氣",
    "PULMONARY": "15_肺功能與呼吸系統",
    "CARDIAC": "16_心臟與心血管評估",
    "ECG": "17_心電圖與心律",
    "ECHOCARDIOGRAPHY": "18_心臟超音波",
    "COMORBIDITY": "19_重要共病",
    "OXYGEN_THERAPY": "20_氧氣治療需求與反應",
    "ANTHROPOMETRY": "21_身體測量_BMI頸圍腰圍",
    "WEIGHT_MANAGEMENT": "22_體重變化與減重治療",
    "LIFESTYLE": "23_生活型態與睡眠作息",
    "LABORATORY": "24_實驗室檢查",
    "NEUROLOGICAL": "25_神經學病史",
    "OTHER": "26_其他經醫師確認資料",
}

# Clinically plausible normal/reference and abnormal/follow-up values.
VALUES = {
    "residual_ahi": (3.0, 18.0), "estimated_ahi": (4.0, 32.0),
    "minimum_spo2": (92.0, 78.0), "average_spo2": (96.0, 89.0),
    "resting_spo2": (96.0, 88.0), "exertional_spo2": (94.0, 82.0),
    "treatment_spo2": (95.0, 88.0), "sleep_duration_hours": (7.2, 4.1),
    "nightly_usage_hours": (6.5, 1.4), "pressure_setting": (10.0, 12.0),
    "mask_leak": (9.0, 42.0), "adherence_hours_per_night": (6.8, 1.2),
    "adherence_percentage": (95.0, 28.0), "tonsil_hypertrophy": (1, 4),
    "mallampati": (2, 4), "minimum_airway_area": (180.0, 58.0),
    "airway_volume": (18000.0, 7200.0), "overjet_mm": (2.0, 8.0),
    "ess_score": (5, 18), "stop_bang_score": (2, 7), "isi_score": (5, 22),
    "psqi_score": (4, 16), "paco2": (40.0, 58.0), "pao2": (90.0, 62.0),
    "ph": (7.4, 7.34), "hco3": (24.0, 31.0), "base_excess": (0.0, 5.0),
    "fev1_percent_predicted": (96.0, 55.0), "fvc_percent_predicted": (98.0, 82.0),
    "fev1_fvc_ratio": (80.0, 52.0), "dlco_percent_predicted": (92.0, 60.0),
    "heart_rate": (68.0, 112.0), "lvef": (62.0, 35.0),
    "pulmonary_artery_pressure": (25.0, 52.0), "oxygen_flow_lpm": (0.0, 2.0),
    "hours_per_day": (0.0, 8.0), "height_cm": (170.0, 170.0),
    "weight_kg": (65.0, 105.0), "bmi": (22.5, 36.3),
    "neck_circumference_cm": (36.0, 46.0), "waist_circumference_cm": (82.0, 118.0),
    "baseline_weight_kg": (100.0, 88.0), "current_weight_kg": (88.0, 98.0),
    "weight_change_kg": (-12.0, 10.0), "hemoglobin": (14.2, 17.8),
    "hematocrit": (43.0, 54.0), "tsh": (2.1, 12.0), "hba1c": (5.4, 8.2),
    "creatinine": (0.9, 1.8),
}

TEXT_VALUES = {
    "device_name": ("Synthetic SleepWatch Pro", "Synthetic SleepWatch Pro"),
    "monitoring_period": ("2026-08-01 night", "2026-08-02 night"),
    "tolerance": ("good", "poor"), "adherence": ("good", "poor"),
    "current_treatment": ("CPAP", "APAP"), "symptom_improvement": ("marked improvement", "minimal improvement"),
    "adverse_effects": ("none", "mask leak and insomnia"),
    "medication_list": ("amlodipine", "zolpidem and clonazepam"),
    "primary_treatment_goal": ("control apnea and improve oxygen", "avoid PAP mask"),
    "suspected_cause": ("sleep related obstruction", "possible pulmonary disease"),
    "rhythm": ("sinus rhythm", "atrial fibrillation"), "interpretation": ("normal ECG", "atrial fibrillation"),
    "symptoms": ("none", "exertional dyspnea"), "delivery_method": ("none", "nasal cannula"),
    "intervention": ("diet and exercise", "no active intervention"),
    "smoking": ("no", "daily"), "alcohol_use": ("none", "nightly"),
    "exercise": ("150 minutes per week", "none"), "sleep_schedule": ("23:00-07:00", "irregular"),
    "psychiatric_medications": ("none", "sertraline"),
    "neurological_findings": ("normal", "reduced respiratory muscle strength"),
    "procedure_name": ("none", "UPPP and tonsillectomy"), "procedure_date": ("not applicable", "2025-06-10"),
    "outcome": ("not applicable", "residual snoring and AHI 12"), "complications": ("none", "none"),
    "document_type": ("routine clinic note", "urgent clinic note"),
    "clinical_summary": ("stable OSA follow-up", "worsening nocturnal desaturation and sleepiness"),
    "assessment": ("continue monitoring", "possible treatment failure"),
    "plan": ("review in three months", "expedited sleep specialist review"),
}


def field_value(field, abnormal: bool):
    index = 1 if abnormal else 0
    if field.name in VALUES:
        return VALUES[field.name][index]
    if field.name in TEXT_VALUES:
        return TEXT_VALUES[field.name][index]
    if field.value_type == "boolean":
        return "yes" if abnormal else "no"
    if field.choices:
        return field.choices[-1] if abnormal else field.choices[0]
    if field.value_type in {"float", "int"}:
        low = field.minimum if field.minimum is not None else 0
        high = field.maximum if field.maximum is not None else low + 10
        value = low + (high - low) * (0.7 if abnormal else 0.3)
        return int(round(value)) if field.value_type == "int" else round(value, 2)
    if "date" in field.name:
        return "2026-08-01"
    return "abnormal finding" if abnormal else "normal finding"


def write_key_value(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "value"])
        writer.writerows(rows)


def write_wearable(path: Path, abnormal: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 8, 1, 23, 0)
    columns = ["timestamp", "device_name", "spo2", "heart_rate", "respiratory_rate", "sleep_stage", "position", "apnea_event"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(240):
            event = abnormal and index % 10 == 0
            writer.writerow({
                "timestamp": (start + timedelta(seconds=30 * index)).isoformat(),
                "device_name": "Synthetic SleepWatch Pro",
                "spo2": (86 + index % 3) if event else ((93 + index % 3) if abnormal else 96 + index % 2),
                "heart_rate": (84 + index % 8) if abnormal else (62 + index % 6),
                "respiratory_rate": (20 + index % 3) if abnormal else (14 + index % 2),
                "sleep_stage": "REM" if index % 5 == 0 else "N2",
                "position": "supine" if (abnormal or index % 4 == 0) else "left",
                "apnea_event": "yes" if event else "no",
            })


def main() -> None:
    manifest = []
    pap_cases = {
        "01_治療成功": [("therapy_type", "CPAP"), ("pressure_setting", 10), ("residual_ahi", 2.8), ("mask_leak", 9), ("treatment_spo2", 94), ("adherence_hours_per_night", 6.8), ("adherence_percentage", 95), ("tolerance", "good")],
        "02_治療效果不足": [("therapy_type", "CPAP"), ("pressure_setting", 12), ("residual_ahi", 18), ("mask_leak", 12), ("treatment_spo2", 88), ("adherence_hours_per_night", 6.1), ("adherence_percentage", 88), ("tolerance", "good")],
    }
    for name, rows in pap_cases.items():
        path = OUTPUT / LABELS["PAP"] / f"{name}.csv"
        write_key_value(path, rows)
        manifest.append(("PAP", LABELS["PAP"], path.name, "key_value"))

    for module_id, spec in MODULE_SPECS.items():
        folder = OUTPUT / LABELS[module_id]
        if module_id == "DEVICE_DATA":
            for name, abnormal in (("01_手錶同步_穩定睡眠", False), ("02_手錶同步_低血氧呼吸事件", True)):
                path = folder / f"{name}.csv"
                write_wearable(path, abnormal)
                manifest.append((module_id, LABELS[module_id], path.name, "wearable_time_series"))
            continue
        for name, abnormal in (("01_正常或穩定參考", False), ("02_異常或需注意情境", True)):
            rows = [(field.name, field_value(field, abnormal)) for field in spec.fields]
            path = folder / f"{name}.csv"
            write_key_value(path, rows)
            manifest.append((module_id, LABELS[module_id], path.name, "key_value"))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "00_測試檔對照表.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["module_id", "對應網頁選單", "測試檔名", "資料型態"])
        writer.writerows(manifest)
    (OUTPUT / "README_請先看.md").write_text(
        "# 網頁補充資料上傳測試庫\n\n"
        "所有資料都是合成測試資料，不可用於真實醫療決策。\n\n"
        "每個編號資料夾對應網頁的一種補充資料選單；01 為穩定參考，02 為異常情境。\n"
        "穿戴式資料為每 30 秒一筆的手錶感測時間序列。\n",
        encoding="utf-8",
    )
    print(f"generated={len(manifest)} output={OUTPUT}")


if __name__ == "__main__":
    main()
