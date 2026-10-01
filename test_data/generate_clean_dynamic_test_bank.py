from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_modules.parser_schemas import MODULE_SPECS


ROOT = Path.home() / "Downloads" / "醫院格式_完整補充臨床資料測試包_26類加穿戴式"

WEB_LABELS = {
    "PAP": "PAP治療成效與耐受度", "ENT": "耳鼻喉與鼻內視鏡", "DISE": "DISE睡眠內視鏡",
    "IMAGING": "上呼吸道影像_CT_CBCT_MRI", "HYPOXEMIA": "低氧來源評估", "ABG": "動脈血液氣體分析",
    "PULMONARY": "肺功能檢查", "CARDIAC": "心臟與心血管評估", "ECG": "十二導程心電圖",
    "ECHOCARDIOGRAPHY": "心臟超音波", "COMORBIDITY": "共病資料", "MEDICATION": "目前用藥與失眠診斷",
    "PATIENT_PREFERENCE": "患者治療偏好與共同決策", "FOLLOW_UP": "治療後回診追蹤",
    "ANTHROPOMETRY": "身體測量_BMI頸圍腰圍", "WEIGHT_MANAGEMENT": "體重與減重紀錄",
    "LIFESTYLE": "生活型態", "SLEEP_QUESTIONNAIRE": "睡眠量表_ESS_ISI_PSQI_STOPBANG",
    "DENTAL_CRANIOFACIAL": "牙科與顎顏面評估", "LABORATORY": "實驗室檢查",
    "NEUROLOGICAL": "神經學評估", "PSYCHIATRIC": "心理精神與睡眠相關診斷",
    "OXYGEN_THERAPY": "氧氣治療資料", "SURGERY_HISTORY": "手術與治療病史",
    "DEVICE_DATA": "穿戴裝置與居家監測", "OTHER": "其他臨床資訊",
}

SPECIAL = {
    "PAP": {"therapy_type": ("APAP", "CPAP"), "pressure_setting": (8, 16), "residual_ahi": (2.0, 32.0),
            "mask_leak": (5, 55), "treatment_spo2": (96, 82), "adherence_hours_per_night": (7.5, 0.5),
            "adherence_percentage": (98, 8), "tolerance": ("good", "intolerant")},
    "ANTHROPOMETRY": {"height_cm": (170, 170), "weight_kg": (62, 135), "bmi": (21.5, 46.7),
                       "neck_circumference_cm": (34, 54), "waist_circumference_cm": (76, 148)},
    "DISE": {"velum_collapse": ("none", "complete"), "oropharyngeal_lateral_wall_collapse": ("none", "complete"),
             "tongue_base_collapse": ("none", "complete"), "epiglottis_collapse": ("none", "complete"),
             "collapse_pattern": ("anteroposterior", "concentric"), "collapse_degree": ("partial", "complete"),
             "vote_classification": ("V0 O0 T0 E0", "V2C O2LAT T2AP E2AP"), "exam_date": ("2026-08-13", "2026-08-13")},
    "ENT": {"nasal_septal_deviation": ("no", "yes"), "inferior_turbinate_hypertrophy": ("no", "yes"),
            "tonsil_hypertrophy": (0, 4), "soft_palate_abnormality": ("no", "yes"), "mallampati": (1, 4),
            "nasal_obstruction": ("no", "yes"), "adenoid_hypertrophy": ("no", "yes"),
            "retrognathia": ("no", "yes"), "exam_date": ("2026-08-13", "2026-08-13")},
    "MEDICATION": {"medication_list": ("none", "clonazepam zolpidem opioid"), "sedative_hypnotics": ("no", "yes"),
                   "opioids": ("no", "yes"), "respiratory_depressants": ("no", "yes"), "insomnia_diagnosis": ("no", "yes")},
}


def value_for(module_id, field, high):
    if field.name in SPECIAL.get(module_id, {}):
        return SPECIAL[module_id][field.name][int(high)]
    if field.value_type == "boolean": return "yes" if high else "no"
    if field.choices: return field.choices[-1] if high else field.choices[0]
    if field.value_type in {"float", "int"}:
        low = field.minimum if field.minimum is not None else 0
        top = field.maximum if field.maximum is not None else low + 100
        value = low + (top - low) * (0.78 if high else 0.25)
        return int(round(value)) if field.value_type == "int" else round(value, 2)
    if "date" in field.name: return "2026-08-13"
    return "severe abnormal confirmed" if high else "normal or absent"


def write_hospital_report(path: Path, module_id: str, high: bool) -> None:
    spec = MODULE_SPECS[module_id]
    lines = [
        "臺灣睡眠醫學整合中心（合成測試資料，非真實病歷）", f"報告名稱：{spec.display_name}",
        "病歷號：SYN-OSA-20260813-001", "患者姓名：測試個案", "檢查日期：2026-08-13",
        f"儀器／方法：{spec.display_name} 標準化資料匯出", "報告狀態：FINAL", "資料品質：可判讀；完整性 98%",
        "情境：" + ("B－明顯異常／高風險合成情境" if high else "A－低風險／相對正常合成情境"), "",
        "【結構化檢測結果】", "欄位代碼 | 中文項目 | 結果 | 單位 | 參考／判讀 | 異常旗標",
    ]
    for field in spec.fields:
        value = value_for(module_id, field, high)
        label = field.aliases[-1] if field.aliases else field.name
        unit = field.unit or "—"
        reference = "依院內檢驗方法與臨床情境判讀"
        if field.minimum is not None or field.maximum is not None:
            reference = f"可接受資料範圍 {field.minimum if field.minimum is not None else '—'}–{field.maximum if field.maximum is not None else '—'}"
        lines.append(f"{field.name} | {label} | {value} | {unit} | {reference} | {'H' if high else 'N'}")
        lines.append(f"{field.name}: {value}")
    lines += ["", "【醫師判讀】", "本報告為系統驗證用合成資料；異常情境僅用於測試解析、模型重訓與輸出差異。",
              "正式臨床使用前須由合格醫師核對原始波形、影像、檢驗方法與病人狀況。", "電子簽章：SYNTHETIC-TEST-ONLY"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def write_pap(path: Path, high: bool) -> None:
    rows = [(key, values[int(high)]) for key, values in SPECIAL["PAP"].items()]
    lines = ["PAP裝置治療下載摘要（合成測試資料）", "病歷號：SYN-OSA-20260813-001", "裝置序號：SYN-PAP-001",
             "監測期間：2026-08-06 至 2026-08-13", "資料品質：有效夜數 7/7；訊號完整性 99%", ""]
    lines += [f"{key}: {value}" for key, value in rows]
    lines += ["", "醫師覆核：待確認", "用途：僅供軟體流程測試，非臨床治療依據"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def write_wearable(path: Path, high: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 8, 13, 23, 0)
    columns = ["timestamp", "patient_id", "device_name", "firmware", "signal_quality", "spo2", "heart_rate",
               "respiratory_rate", "sleep_stage", "position", "apnea_event"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for index in range(240):
            event = high and index % 4 == 0
            writer.writerow({"timestamp": (start + timedelta(seconds=30 * index)).isoformat(), "patient_id": "SYN-OSA-20260813-001",
                "device_name": "Hospital-Demo-Watch-X1", "firmware": "3.2.1", "signal_quality": 96 if not high else 91,
                "spo2": 76 + index % 5 if event else (90 if high else 97), "heart_rate": 108 + index % 12 if high else 62 + index % 5,
                "respiratory_rate": 25 if high else 14, "sleep_stage": "REM" if index % 3 == 0 else "N2",
                "position": "supine" if high else "left", "apnea_event": "yes" if event else "no"})


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    pap = ROOT / f"00_{WEB_LABELS['PAP']}"; write_pap(pap / "A_PAP治療成功_裝置下載報告.txt", False); write_pap(pap / "B_PAP治療失敗_高殘餘AHI低耐受報告.txt", True)
    manifest.append({"module_id": "PAP", "web_label": WEB_LABELS["PAP"], "folder": pap.name})
    for index, module_id in enumerate(MODULE_SPECS, start=1):
        folder = ROOT / f"{index:02d}_{WEB_LABELS[module_id]}"
        if module_id == "DEVICE_DATA":
            write_wearable(folder / "A_穿戴式低風險_完整夜間時序.csv", False); write_wearable(folder / "B_穿戴式高風險_低氧與呼吸事件時序.csv", True)
        else:
            write_hospital_report(folder / f"A_{WEB_LABELS[module_id]}_低風險醫院報告.txt", module_id, False)
            write_hospital_report(folder / f"B_{WEB_LABELS[module_id]}_高風險醫院報告.txt", module_id, True)
        manifest.append({"module_id": module_id, "web_label": WEB_LABELS[module_id], "folder": folder.name})
    (ROOT / "00_網頁欄位與資料夾對照.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "README_先看我.txt").write_text(
        "本資料包全部為合成測試資料，不是真實病歷，不可用於醫療決策。\n每個資料夾對應網頁的一個補充臨床資料類型。A為低風險，B為明顯異常情境。\n"
        "醫院報告包含病歷號、設備／方法、日期、單位、參考／判讀、異常旗標、品質與醫師判讀，並保留模型可解析的標準欄位代碼。\n"
        "相同檔案重複上傳會被去重；請交替使用A/B或不同類型測試。排名不一定每次改變，但Digital Twin特徵、訓練列、模型版本或高精度預測必須顯示更新證據。\n",
        encoding="utf-8-sig")
    print(ROOT)


if __name__ == "__main__": main()
