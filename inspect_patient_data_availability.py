from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

INCOMING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "incoming"
)

DEMOGRAPHICS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "demographics"
)


POSITION_KEYWORDS = [
    "position",
    "pos",
    "posangle",
    "body position",
    "bodyposition",
    "sleep position",
]

CO2_KEYWORDS = [
    "co2",
    "etco2",
    "etco₂",
    "end tidal co2",
    "endtidalco2",
    "tcco2",
    "tcpco2",
    "transcutaneous co2",
    "capnography",
    "capno",
]

PAP_PRESSURE_KEYWORDS = [
    "cpap",
    "apap",
    "pap pressure",
    "pap_pressure",
    "mask pressure",
    "maskpressure",
    "pressure",
    "cmh2o",
]

LEAK_KEYWORDS = [
    "leak",
    "mask leak",
    "maskleak",
    "total leak",
    "unintentional leak",
]

SPO2_KEYWORDS = [
    "spo2",
    "sao2",
    "saturation",
    "oxygen saturation",
    "oximetry",
    "oximeter",
]

FLOW_KEYWORDS = [
    "flow",
    "nasal",
    "pressure flow",
    "thermistor",
    "therm",
]

EFFORT_KEYWORDS = [
    "thorax",
    "thora",
    "thoracic",
    "chest",
    "abdomen",
    "abdom",
    "abdominal",
    "inductance",
    "effort",
]

ECG_KEYWORDS = [
    "ecg",
    "ekg",
    "heart",
    "pulse",
    "hr",
]

CLINICAL_KEYWORDS = {
    "hypertension": [
        "hypertension",
        "htn",
        "高血壓",
    ],
    "heart_failure": [
        "heart failure",
        "chf",
        "心衰竭",
        "心臟衰竭",
    ],
    "coronary_disease": [
        "coronary",
        "cad",
        "冠心病",
        "冠狀動脈",
    ],
    "stroke": [
        "stroke",
        "cva",
        "中風",
    ],
    "copd": [
        "copd",
        "慢性阻塞性肺病",
    ],
    "asthma": [
        "asthma",
        "氣喘",
    ],
    "diabetes": [
        "diabetes",
        "dm",
        "糖尿病",
    ],
    "kidney_disease": [
        "kidney",
        "renal",
        "ckd",
        "腎臟",
        "腎病",
    ],
    "smoking": [
        "smoking",
        "smoke",
        "tobacco",
        "抽菸",
        "吸菸",
    ],
    "alcohol": [
        "alcohol",
        "drinking",
        "酒精",
        "飲酒",
    ],
    "medication": [
        "medication",
        "medicine",
        "drug",
        "藥物",
        "用藥",
    ],
    "opioid": [
        "opioid",
        "opiate",
        "morphine",
        "fentanyl",
        "鴉片",
        "嗎啡",
    ],
    "sedative": [
        "sedative",
        "hypnotic",
        "benzodiazepine",
        "sleeping pill",
        "安眠",
        "鎮靜",
    ],
}


def normalize_text(
    value: Any,
) -> str:
    return str(value).strip().lower()


def find_matching_channels(
    channel_names: list[str],
    keywords: list[str],
) -> list[str]:
    matched: list[str] = []

    for channel_name in channel_names:
        normalized_channel = normalize_text(
            channel_name
        )

        for keyword in keywords:
            normalized_keyword = normalize_text(
                keyword
            )

            if normalized_keyword in normalized_channel:
                matched.append(
                    channel_name
                )
                break

    return sorted(
        set(matched)
    )


def find_edf_file(
    patient_folder: Path,
) -> Path:
    edf_files = sorted(
        patient_folder.glob("*.edf")
    )

    if not edf_files:
        raise FileNotFoundError(
            f"患者資料夾沒有 EDF："
            f"{patient_folder}"
        )

    if len(edf_files) == 1:
        return edf_files[0]

    preferred = [
        file_path
        for file_path in edf_files
        if "_EDF.edf" in file_path.name
    ]

    if len(preferred) == 1:
        return preferred[0]

    print()
    print(
        "警告：找到多個 EDF，"
        "目前使用第一個："
    )

    for file_path in edf_files:
        print(
            f"  - {file_path.name}"
        )

    return edf_files[0]


def inspect_channel_values(
    raw: mne.io.BaseRaw,
    channel_name: str,
    max_samples: int = 200000,
) -> dict[str, Any]:
    sampling_rate = float(
        raw.info["sfreq"]
    )

    total_samples = int(
        raw.n_times
    )

    if total_samples <= 0:
        return {
            "channel": channel_name,
            "sampling_rate": sampling_rate,
            "sample_count": 0,
            "finite_count": 0,
        }

    if total_samples <= max_samples:
        sample_indices = np.arange(
            total_samples,
            dtype=int,
        )
    else:
        sample_indices = np.linspace(
            0,
            total_samples - 1,
            max_samples,
            dtype=int,
        )

    channel_index = raw.ch_names.index(
        channel_name
    )

    data = raw.get_data(
        picks=[channel_index],
    )[0]

    sampled_data = data[
        sample_indices
    ]

    finite_data = sampled_data[
        np.isfinite(sampled_data)
    ]

    result: dict[str, Any] = {
        "channel": channel_name,
        "sampling_rate": sampling_rate,
        "sample_count": int(
            len(sampled_data)
        ),
        "finite_count": int(
            len(finite_data)
        ),
    }

    if len(finite_data) == 0:
        return result

    result.update(
        {
            "min": float(
                np.min(finite_data)
            ),
            "p01": float(
                np.percentile(
                    finite_data,
                    1,
                )
            ),
            "p05": float(
                np.percentile(
                    finite_data,
                    5,
                )
            ),
            "median": float(
                np.median(finite_data)
            ),
            "p95": float(
                np.percentile(
                    finite_data,
                    95,
                )
            ),
            "p99": float(
                np.percentile(
                    finite_data,
                    99,
                )
            ),
            "max": float(
                np.max(finite_data)
            ),
            "unique_rounded_count": int(
                len(
                    np.unique(
                        np.round(
                            finite_data,
                            3,
                        )
                    )
                )
            ),
        }
    )

    unique_values = np.unique(
        np.round(
            finite_data,
            3,
        )
    )

    if len(unique_values) <= 30:
        result[
            "unique_values"
        ] = unique_values.tolist()

    return result


def find_demographic_files() -> list[Path]:
    candidate_files: list[Path] = []

    search_roots = [
        PROJECT_ROOT,
        DEMOGRAPHICS_ROOT,
        PROJECT_ROOT / "data",
    ]

    seen: set[Path] = set()

    for search_root in search_roots:
        if not search_root.exists():
            continue

        for pattern in [
            "*.xlsx",
            "*.xls",
            "*.csv",
        ]:
            for file_path in search_root.rglob(
                pattern
            ):
                resolved = file_path.resolve()

                if resolved in seen:
                    continue

                seen.add(
                    resolved
                )

                normalized_name = (
                    file_path.name.lower()
                )

                if any(
                    keyword in normalized_name
                    for keyword in [
                        "patient",
                        "demographic",
                        "basic",
                        "subject",
                        "clinical",
                    ]
                ):
                    candidate_files.append(
                        file_path
                    )

    return sorted(
        candidate_files
    )


def inspect_tabular_file(
    file_path: Path,
) -> list[dict[str, Any]]:
    results: list[
        dict[str, Any]
    ] = []

    try:
        if file_path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(
                file_path
            )

            results.append(
                {
                    "sheet": "CSV",
                    "columns": (
                        dataframe.columns.tolist()
                    ),
                }
            )

            return results

        excel_file = pd.ExcelFile(
            file_path
        )

        for sheet_name in excel_file.sheet_names:
            dataframe = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                nrows=5,
            )

            results.append(
                {
                    "sheet": sheet_name,
                    "columns": (
                        dataframe.columns.tolist()
                    ),
                }
            )

    except Exception as error:
        results.append(
            {
                "sheet": "ERROR",
                "error": str(error),
                "columns": [],
            }
        )

    return results


def find_clinical_columns(
    columns: list[Any],
) -> dict[str, list[str]]:
    matched: dict[
        str,
        list[str],
    ] = {}

    for clinical_group, keywords in (
        CLINICAL_KEYWORDS.items()
    ):
        group_matches: list[str] = []

        for column in columns:
            normalized_column = normalize_text(
                column
            )

            for keyword in keywords:
                if (
                    normalize_text(keyword)
                    in normalized_column
                ):
                    group_matches.append(
                        str(column)
                    )
                    break

        if group_matches:
            matched[
                clinical_group
            ] = sorted(
                set(group_matches)
            )

    return matched


def print_channel_group(
    title: str,
    matched_channels: list[str],
) -> None:
    print()
    print(
        f"{title}："
    )

    if not matched_channels:
        print(
            "  [NOT FOUND]"
        )
        return

    for channel_name in matched_channels:
        print(
            f"  [FOUND] {channel_name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "檢查患者 EDF 與臨床表格中"
            "可供治療適配度分析使用的資料。"
        )
    )

    parser.add_argument(
        "--patient-id",
        required=True,
    )

    args = parser.parse_args()

    patient_id = str(
        args.patient_id
    ).strip()

    patient_folder = (
        INCOMING_ROOT
        / patient_id
    )

    if not patient_folder.exists():
        raise FileNotFoundError(
            f"找不到患者資料夾："
            f"{patient_folder}"
        )

    edf_file = find_edf_file(
        patient_folder
    )

    print("=" * 80)
    print("Patient Data Availability Inspector")
    print("=" * 80)
    print(
        f"Patient ID：{patient_id}"
    )
    print(
        f"EDF：{edf_file}"
    )

    print()
    print("=" * 80)
    print("步驟 1：讀取 EDF Channel")
    print("=" * 80)

    raw = mne.io.read_raw_edf(
        edf_file,
        preload=False,
        verbose=False,
    )

    channel_names = list(
        raw.ch_names
    )

    print(
        f"Channel 數量："
        f"{len(channel_names)}"
    )

    print()
    print("全部 EDF Channels：")

    for index, channel_name in enumerate(
        channel_names,
        start=1,
    ):
        print(
            f"{index:03d}. "
            f"{channel_name}"
        )

    position_channels = (
        find_matching_channels(
            channel_names,
            POSITION_KEYWORDS,
        )
    )

    co2_channels = (
        find_matching_channels(
            channel_names,
            CO2_KEYWORDS,
        )
    )

    pap_channels = (
        find_matching_channels(
            channel_names,
            PAP_PRESSURE_KEYWORDS,
        )
    )

    leak_channels = (
        find_matching_channels(
            channel_names,
            LEAK_KEYWORDS,
        )
    )

    spo2_channels = (
        find_matching_channels(
            channel_names,
            SPO2_KEYWORDS,
        )
    )

    flow_channels = (
        find_matching_channels(
            channel_names,
            FLOW_KEYWORDS,
        )
    )

    effort_channels = (
        find_matching_channels(
            channel_names,
            EFFORT_KEYWORDS,
        )
    )

    ecg_channels = (
        find_matching_channels(
            channel_names,
            ECG_KEYWORDS,
        )
    )

    print()
    print("=" * 80)
    print("步驟 2：EDF 資料可用性")
    print("=" * 80)

    print_channel_group(
        "姿勢 Channel",
        position_channels,
    )

    print_channel_group(
        "CO₂ Channel",
        co2_channels,
    )

    print_channel_group(
        "PAP Pressure Channel",
        pap_channels,
    )

    print_channel_group(
        "Leak Channel",
        leak_channels,
    )

    print_channel_group(
        "SpO₂ Channel",
        spo2_channels,
    )

    print_channel_group(
        "Airflow Channel",
        flow_channels,
    )

    print_channel_group(
        "Respiratory Effort Channel",
        effort_channels,
    )

    print_channel_group(
        "ECG / Heart Rate Channel",
        ecg_channels,
    )

    print()
    print("=" * 80)
    print("步驟 3：姿勢 Channel 值域")
    print("=" * 80)

    if position_channels:
        raw.load_data()

        for channel_name in position_channels:
            result = inspect_channel_values(
                raw,
                channel_name,
            )

            print()
            print(
                f"Channel：{channel_name}"
            )

            for key, value in result.items():
                if key == "channel":
                    continue

                print(
                    f"  {key}：{value}"
                )
    else:
        print(
            "沒有找到姿勢 Channel。"
        )

    print()
    print("=" * 80)
    print("步驟 4：CO₂ Channel 值域")
    print("=" * 80)

    if co2_channels:
        if not raw.preload:
            raw.load_data()

        for channel_name in co2_channels:
            result = inspect_channel_values(
                raw,
                channel_name,
            )

            print()
            print(
                f"Channel：{channel_name}"
            )

            for key, value in result.items():
                if key == "channel":
                    continue

                print(
                    f"  {key}：{value}"
                )
    else:
        print(
            "沒有找到 CO₂ Channel。"
        )

    print()
    print("=" * 80)
    print("步驟 5：臨床／基本資料欄位")
    print("=" * 80)

    demographic_files = (
        find_demographic_files()
    )

    if not demographic_files:
        print(
            "沒有找到可能的患者基本資料表。"
        )

    for file_path in demographic_files:
        print()
        print(
            f"檔案：{file_path}"
        )

        file_results = (
            inspect_tabular_file(
                file_path
            )
        )

        for file_result in file_results:
            print(
                f"  工作表："
                f"{file_result['sheet']}"
            )

            if "error" in file_result:
                print(
                    f"  ERROR："
                    f"{file_result['error']}"
                )
                continue

            columns = file_result[
                "columns"
            ]

            print(
                f"  欄位數："
                f"{len(columns)}"
            )

            for column in columns:
                print(
                    f"    - {column}"
                )

            clinical_matches = (
                find_clinical_columns(
                    columns
                )
            )

            if clinical_matches:
                print(
                    "  找到可能的臨床欄位："
                )

                for group, matches in (
                    clinical_matches.items()
                ):
                    print(
                        f"    {group}："
                        f"{matches}"
                    )
            else:
                print(
                    "  未找到明確共病／用藥欄位。"
                )

    print()
    print("=" * 80)
    print("資料可用性結論")
    print("=" * 80)

    print(
        "姿勢分類："
        + (
            "有原始 Channel，需建立映射／分類演算法"
            if position_channels
            else "目前 EDF 未找到明確姿勢 Channel"
        )
    )

    print(
        "低氧事件同步分析："
        + (
            "可進行"
            if spo2_channels
            and (
                flow_channels
                or effort_channels
            )
            else "需進一步確認呼吸與血氧 Channel"
        )
    )

    print(
        "清醒 PSG SpO₂："
        + (
            "SpO₂ Channel 已存在，可結合 Stage W 計算"
            if spo2_channels
            else "缺少 SpO₂ Channel"
        )
    )

    print(
        "CO₂："
        + (
            "找到 Channel，可進一步解析"
            if co2_channels
            else "EDF 未找到明確 CO₂ Channel"
        )
    )

    print(
        "PAP 滴定資料："
        + (
            "找到可能的 Pressure Channel，需確認是否為滴定 PSG"
            if pap_channels
            else "EDF 未找到明確 PAP Pressure Channel"
        )
    )

    print(
        "PAP Leak 資料："
        + (
            "找到可能的 Leak Channel"
            if leak_channels
            else "EDF 未找到明確 Leak Channel"
        )
    )

    print(
        "上呼吸道解剖："
        "EDF 無法直接提供，需 ENT／影像／DISE 等資料"
    )

    print(
        "完整共病與用藥："
        "請依上方臨床表格欄位結果確認"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()