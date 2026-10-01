from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCENARIOS = {
    "01_正常穩定_完整白天資料": (97, 10, 97.2, .35, 68, 4, 15, 1.2, "none", "資料品質良好，手錶修正接近 0。"),
    "02_輕微邊界低氧_完整白天資料": (97, 10, 94.5, 1, 74, 7, 17, 2, "mild", "較正常檔略升，但可能仍在同一層級。"),
    "03_間歇性低氧_完整白天資料": (97, 10, 95, .6, 78, 10, 18, 3, "intermittent", "最低 SpO₂ 與 T90 使風險明顯升高。"),
    "04_持續性低氧_完整白天資料": (97, 10, 88.5, 1.4, 82, 9, 19, 2.5, "sustained", "T90 很高，應有明顯向上修正。"),
    "05_心率高度波動_血氧正常": (97, 10, 96.5, .5, 82, 28, 17, 3, "none", "心率標準差偏高，應小幅增加。"),
    "06_低氧加心率波動_高風險壓力測試": (97, 10, 94, .5, 86, 24, 20, 5, "severe", "多項訊號異常，應比一般情境高。"),
    "07_資料不足_只有兩小時": (13, 10, 96.5, .5, 70, 5, 15, 1, "none", "品質應顯示 LIMITED。"),
    "08_稀疏監測_每小時一筆": (13, 60, 96, .8, 72, 6, 16, 1.5, "none", "時間長但筆數不足，品質應為 LIMITED。"),
    "09_缺少呼吸率_仍可分析": (97, 10, 96.8, .5, 69, 5, None, 0, "none", "仍可分析，但呼吸率不參與。"),
    "10_單次疑似血氧雜訊_其餘正常": (97, 10, 97, .3, 68, 4, 15, 1, "artifact", "單點雜訊應被 SpO₂ 第5百分位規則抑制，結果接近正常。"),
}


def write_case(path: Path, start: datetime, name: str, values: tuple) -> None:
    count, minutes, base_spo2, spo2_amp, base_hr, hr_amp, base_rr, rr_amp, event, _ = values
    rows = []
    for index in range(count):
        timestamp = start + timedelta(minutes=minutes * index)
        spo2 = base_spo2 + spo2_amp * math.sin(index / 7)
        hr = base_hr + hr_amp * math.sin(index / 2.8)
        rr = None if base_rr is None else base_rr + rr_amp * math.sin(index / 3.4)
        if event == "mild" and index % 15 in {0, 1}: spo2 -= 3
        elif event == "intermittent" and index % 10 in {0, 1, 2}: spo2 -= 8; hr += 8
        elif event == "sustained": spo2 -= 1.5 * abs(math.sin(index / 4))
        elif event == "severe" and index % 9 in {0, 1, 2, 3}: spo2 -= 12 if index % 9 < 3 else 8; hr += 10
        elif event == "artifact" and index == 48: spo2 = 78
        rows.append({
            "timestamp": timestamp.isoformat(), "spo2": round(max(78, min(spo2, 100)), 1),
            "heart_rate": round(max(35, min(hr, 190)), 1),
            "respiratory_rate": "" if rr is None else round(max(7, min(rr, 40)), 1),
            "movement": round(.15 + .25 * abs(math.sin(index / 5)), 3),
            "position": index % 4, "source": "synthetic_watch_test", "test_scenario": name,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "test_data" / "tonight_watch"
    tz = timezone(timedelta(hours=8))
    for offset, (name, values) in enumerate(SCENARIOS.items(), 1):
        write_case(output / f"{name}.csv", datetime(2026, 8, 1 + offset, 6, 0, tzinfo=tz), name, values)
    guide = ["今晚睡眠呼吸風險－穿戴式合成測試資料", "", "僅供功能驗證，不可當作真實臨床資料。", "每個情境建議使用不同 Patient ID。", ""]
    for name, values in SCENARIOS.items(): guide += [f"{name}.csv", f"  預期反應：{values[-1]}"]
    (output / "00_測試說明與預期反應.txt").write_text("\n".join(guide), encoding="utf-8-sig")
    print(output)
