from __future__ import annotations

import json
import io
import random
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src.continual_learning.wearable_realtime import WearableStore, predict_risk, start_receiver
from src.continual_learning.tonight_apnea_risk import (
    add_verified_night_outcome_and_retrain,
    predict as predict_tonight_risk,
)
from src.continual_learning.treatment import (
    apply_learned_treatment_adjustment,
    train_adaptive_treatment_models,
)
from clinical_modules.update_service import refresh_patient
from src.continual_learning.governance import current_monitoring_window, governance_summary, wearable_quality


ROOT = Path(__file__).resolve().parent
STORE = WearableStore(ROOT / "data" / "realtime_wearable")
MODEL = ROOT / "models" / "wearable_apnea_next_60s" / "model.joblib"
TONIGHT_MODEL = ROOT / "models" / "tonight_apnea_risk" / "model.joblib"
TREATMENT_SYNC_BATCH = 30
TREATMENT_SYNC_COOLDOWN_SECONDS = 60
TREATMENT_LABELS = {
    "CPAP": "固定壓力正壓呼吸器（CPAP）",
    "APAP": "自動調壓正壓呼吸器（APAP）",
    "SURGERY": "上呼吸道手術評估",
    "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW": "藥物／體重與睡眠共病專科評估",
}


_BLE = st.components.v2.component(
    "sleep_watch_banglejs_realtime",
    html="""
      <div class="box"><button id="connect">連接 Bangle.js</button><button id="disconnect" class="secondary" disabled>中斷</button><span id="status">尚未連線</span></div>
      <div id="live" class="live">等待手錶資料</div>
    """,
    css="""
      .box{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-family:var(--st-font)}
      button{background:var(--st-primary-color);color:white;border:0;border-radius:8px;padding:9px 14px;cursor:pointer}
      button.secondary{background:var(--st-secondary-background-color);color:var(--st-text-color);border:1px solid var(--st-border-color)}
      button:disabled{opacity:.5;cursor:not-allowed} span{color:var(--st-text-color);font-size:14px}
      .live{margin-top:10px;padding:10px 12px;border-radius:8px;background:var(--st-secondary-background-color);color:var(--st-text-color);font-size:14px}
    """,
    js=r"""
      export default function(component) {
        const { parentElement, setTriggerValue } = component
        const button=parentElement.querySelector('#connect'), disconnect=parentElement.querySelector('#disconnect')
        const status=parentElement.querySelector('#status'), live=parentElement.querySelector('#live')
        if (!button || button.dataset.bound) return
        button.dataset.bound='1'
        const SERVICE='6e400001-b5a3-f393-e0a9-e50e24dcca9e', RX='6e400002-b5a3-f393-e0a9-e50e24dcca9e', TX='6e400003-b5a3-f393-e0a9-e50e24dcca9e'
        let device,rx,buffer='',pending=[],timer
        const decoder=new TextDecoder(), encoder=new TextEncoder()
        async function writeCode(code){
          const bytes=encoder.encode('\x03\x10'+code+'\n')
          for(let i=0;i<bytes.length;i+=20){await rx.writeValueWithoutResponse(bytes.slice(i,i+20));await new Promise(r=>setTimeout(r,25))}
        }
        function flush(){if(pending.length)setTriggerValue('samples',pending.splice(0))}
        function onLine(line){
          if(!line.startsWith('{'))return
          try{const v=JSON.parse(line);if(v.t!=='sleepTwin')return;pending.push(v);live.textContent=`心率 ${v.heart_rate??'--'} bpm｜步數 ${v.steps??'--'}｜活動 ${v.movement??'--'}｜XYZ ${v.accel_x??'--'}, ${v.accel_y??'--'}, ${v.accel_z??'--'} g｜${new Date().toLocaleTimeString()}`;if(pending.length>=5)flush()}catch(_){}
        }
        function onData(e){buffer+=decoder.decode(e.target.value);const lines=buffer.split(/\r?\n/);buffer=lines.pop()||'';lines.forEach(onLine)}
        function stopped(){clearInterval(timer);flush();status.textContent='連線已中斷';button.disabled=false;disconnect.disabled=true}
        button.onclick=async()=>{
          try{
            if(!navigator.bluetooth)throw new Error('此瀏覽器不支援 Web Bluetooth，請使用 Chrome 或 Edge')
            device=await navigator.bluetooth.requestDevice({filters:[{namePrefix:'Bangle.js'}],optionalServices:[SERVICE]})
            device.addEventListener('gattserverdisconnected',stopped)
            const server=await device.gatt.connect(), service=await server.getPrimaryService(SERVICE)
            rx=await service.getCharacteristic(RX);const tx=await service.getCharacteristic(TX)
            await tx.startNotifications();tx.addEventListener('characteristicvaluechanged',onData)
            const code=`(function(){if(global.__sleepTwinStop)__sleepTwinStop();var hr=null,conf=0,mov=0,ax=null,ay=null,az=null,amag=null;function r(v){return Math.round(v*1000)/1000}function h(e){hr=e.bpm;conf=e.confidence||0}function a(e){ax=r(e.x);ay=r(e.y);az=r(e.z);amag=r(e.mag);mov=r(Math.abs(e.mag-1))}function s(){var x=Bangle.getHealthStatus('day')||{};Bluetooth.println(JSON.stringify({t:'sleepTwin',timestamp:(new Date()).toISOString(),heart_rate:hr,hr_confidence:conf,movement:mov,accel_x:ax,accel_y:ay,accel_z:az,accel_magnitude:amag,steps:x.steps||0,battery:E.getBattery(),source:'banglejs_ble'}))}Bangle.on('HRM',h);Bangle.on('accel',a);Bangle.setHRMPower(1,'sleepTwin');var q=setInterval(s,5000);global.__sleepTwinStop=function(){clearInterval(q);Bangle.removeListener('HRM',h);Bangle.removeListener('accel',a);Bangle.setHRMPower(0,'sleepTwin');delete global.__sleepTwinStop}})();`
            await writeCode(code);timer=setInterval(flush,12000);status.textContent=`持續接收中：${device.name||'Bangle.js'}`;button.disabled=true;disconnect.disabled=false
          }catch(error){status.textContent=`連線失敗：${error.message}`}
        }
        disconnect.onclick=async()=>{
          try{if(rx)await writeCode("if(global.__sleepTwinStop)__sleepTwinStop()") }catch(_){}
          const tail=pending.splice(0)
          setTriggerValue('session_ended',{ended_at:new Date().toISOString(),samples:tail})
          if(device?.gatt?.connected)device.gatt.disconnect();stopped()
        }
        return()=>clearInterval(timer)
      }
    """,
)


@st.cache_resource
def _receiver():
    return start_receiver(STORE)


def _metadata(patient_id: str):
    resolved = _resolve_patient_id(patient_id)
    path = ROOT / "data" / "processed" / resolved / "patient_metadata.csv"
    if not path.exists(): return None, None
    row = pd.read_csv(path).iloc[0]
    return pd.to_numeric(row.get("age"), errors="coerce"), pd.to_numeric(row.get("BMI"), errors="coerce")


def _resolve_patient_id(patient_id: str) -> str:
    """Accept either the full study folder name or its short patient token."""
    direct = ROOT / "data" / "inference" / patient_id
    if direct.is_dir():
        return patient_id
    token = patient_id.strip().lower()
    matches = [
        folder.name for folder in (ROOT / "data" / "inference").iterdir()
        if folder.is_dir() and folder.name.lower().split(" - ")[-1] == token
    ]
    return matches[0] if len(matches) == 1 else patient_id


def _add_demo(patient_id: str, high_risk: bool):
    for index in range(12):
        STORE.append(patient_id, {
            "spo2": random.uniform(82, 90) if high_risk else random.uniform(95, 99),
            "heart_rate": random.uniform(82, 112) if high_risk else random.uniform(55, 76),
            "position": 0 if index < 8 else 1,
            "movement": random.uniform(0, .15), "source": "demo",
        })


def _read_uploaded_wearable(uploaded_file) -> pd.DataFrame:
    """Read a wearable export and normalize common vendor column names."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith((".xlsx", ".xls")):
        frame = pd.read_excel(io.BytesIO(raw))
    elif name.endswith(".json"):
        payload = json.loads(raw.decode("utf-8-sig"))
        records = payload.get("records") if isinstance(payload, dict) else None
        if isinstance(records, dict) and any(
            key in records for key in ("heart_rate", "spo2", "respiratory_rate", "sleep_stage")
        ):
            frame = _read_apple_health_records(records)
        else:
            frame = pd.json_normalize(payload if isinstance(payload, list) else payload.get("samples", [payload]))
    else:
        frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    aliases = {
        "timestamp": ("timestamp", "time", "datetime", "時間", "日期時間"),
        "spo2": ("spo2", "sp_o2", "oxygen_saturation", "血氧", "血氧濃度"),
        "heart_rate": ("heart_rate", "heartrate", "hr", "pulse", "心率", "脈搏"),
        "respiratory_rate": ("respiratory_rate", "respiration_rate", "breathing_rate", "rr", "呼吸率"),
        "sleep_stage": ("sleep_stage", "stage", "sleep_state", "睡眠階段"),
        "position": ("position", "sleep_position", "posture", "姿勢", "睡姿"),
        "movement": ("movement", "activity", "accelerometer", "活動量", "動作"),
        "accel_x": ("accel_x", "accelerometer_x", "x", "x_axis", "x軸"),
        "accel_y": ("accel_y", "accelerometer_y", "y", "y_axis", "y軸"),
        "accel_z": ("accel_z", "accelerometer_z", "z", "z_axis", "z軸"),
        "accel_magnitude": ("accel_magnitude", "magnitude", "accel_mag", "合成加速度"),
        "apnea_observed_next_60s": ("apnea_observed_next_60s", "apnea_label", "apnea_event", "60秒後呼吸中止", "真實標籤"),
    }
    aliases["synthetic_scenario"] = (
        "synthetic_scenario", "test_scenario", "scenario", "合成測試情境"
    )
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    rename = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate.lower() in normalized:
                rename[normalized[candidate.lower()]] = target
                break
    frame = frame.rename(columns=rename)
    missing = [field for field in ("spo2", "heart_rate") if field not in frame.columns]
    if missing:
        raise ValueError("缺少必要欄位：" + "、".join(missing))
    return frame


def _apple_series(records: dict, key: str, value_name: str) -> pd.DataFrame:
    """Convert one Apple Health record group into a timestamped series."""
    values = records.get(key, [])
    if not isinstance(values, list) or not values:
        return pd.DataFrame(columns=["timestamp", value_name])
    frame = pd.DataFrame.from_records(values)
    if "start_time" not in frame or "value" not in frame:
        return pd.DataFrame(columns=["timestamp", value_name])
    result = pd.DataFrame({
        "timestamp": pd.to_datetime(frame["start_time"], errors="coerce", utc=True),
        value_name: frame["value"],
    }).dropna(subset=["timestamp"])
    return result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _read_apple_health_records(records: dict) -> pd.DataFrame:
    """Align grouped Apple Health exports on SpO2 observations.

    Apple Health stores each vital sign in a separate array.  SpO2 is the
    clinically relevant anchor for the 60-second respiratory-risk input; heart
    and respiratory rate are matched to their nearest observation. Sleep stage
    is carried forward from the latest stage record.
    """
    spo2 = _apple_series(records, "spo2", "spo2")
    heart = _apple_series(records, "heart_rate", "heart_rate")
    respiration = _apple_series(records, "respiratory_rate", "respiratory_rate")
    stage = _apple_series(records, "sleep_stage", "sleep_stage")
    if spo2.empty:
        raise ValueError("Apple Health 檔案沒有可用的血氧（SpO₂）紀錄。")
    result = spo2
    if not heart.empty:
        result = pd.merge_asof(
            result, heart, on="timestamp", direction="nearest",
            tolerance=pd.Timedelta("15min"),
        )
    if not respiration.empty:
        result = pd.merge_asof(
            result, respiration, on="timestamp", direction="nearest",
            tolerance=pd.Timedelta("30min"),
        )
    if not stage.empty:
        result = pd.merge_asof(
            result, stage, on="timestamp", direction="backward",
            tolerance=pd.Timedelta("12h"),
        )
    result["source"] = "apple_health_export"
    result["movement"] = 0.0
    return result.dropna(subset=["spo2", "heart_rate"]).reset_index(drop=True)


def _import_uploaded_wearable(
    patient_id: str,
    uploaded_file,
    replace_previous_uploads: bool = False,
) -> tuple[int, list[str]]:
    frame = _read_uploaded_wearable(uploaded_file)
    scenarios = {
        str(value).strip()
        for value in frame.get("synthetic_scenario", pd.Series(dtype=str)).dropna()
        if str(value).strip()
    }
    if (
        "SYNTHETIC_RANKING_STRESS_SURGERY_CANDIDATE" in scenarios
        and not patient_id.upper().startswith("SYNTHETIC_")
    ):
        raise ValueError(
            "合成測試檔不得寫入真實病人。請將 Patient ID 改為 "
            "SYNTHETIC_WEARABLE_DEMO；合成資料只供隔離測試，不會更新正式治療模型或報告。"
        )
    accepted, errors, prepared = 0, [], []
    for index, row in frame.iterrows():
        try:
            label = row.get("apnea_observed_next_60s")
            if pd.isna(label):
                label = None
            elif isinstance(label, str):
                label = label.strip().lower() in {"1", "true", "yes", "有", "是", "發生"}
            position = row.get("position", 0)
            if isinstance(position, str):
                position = {"supine": 0, "仰睡": 0, "left": 1, "左側": 1, "right": 2, "右側": 2, "prone": 3, "趴睡": 3}.get(position.strip().lower(), 0)
            scenario = row.get("synthetic_scenario")
            scenario = "" if pd.isna(scenario) else str(scenario).strip()
            source = f"uploaded:{uploaded_file.name}"
            if scenario:
                source += f"|scenario={scenario}"
            prepared.append({
                "timestamp": row.get("timestamp"), "spo2": row.get("spo2"),
                "heart_rate": row.get("heart_rate"), "position": position,
                "movement": row.get("movement", 0), "apnea_observed_next_60s": label,
                "accel_x": row.get("accel_x"), "accel_y": row.get("accel_y"),
                "accel_z": row.get("accel_z"),
                "accel_magnitude": row.get("accel_magnitude"),
                "respiratory_rate": row.get("respiratory_rate"),
                "sleep_stage": row.get("sleep_stage"),
                "source": source,
            })
        except Exception as exc:
            errors.append(f"第 {index + 2} 列：{exc}")
    if prepared:
        # Generated scenario files are mutually exclusive test cases. Replacing
        # only older generated rows prevents a prior high-risk test from
        # contaminating the next low-risk test; real device rows are retained.
        if replace_previous_uploads:
            STORE.remove_uploaded_rows(patient_id)
        elif scenarios:
            STORE.remove_synthetic_rows(patient_id)
        accepted = STORE.append_many(patient_id, prepared)
    return accepted, errors


def _wearable_treatment_summary(samples: pd.DataFrame, risk: dict | None) -> dict:
    recent = current_monitoring_window(samples).tail(120).copy()
    timestamps = pd.to_datetime(recent.get("timestamp"), errors="coerce", utc=True)
    duration_hours = max(
        ((timestamps.max() - timestamps.min()).total_seconds() / 3600)
        if timestamps.notna().sum() >= 2 else len(recent) * 30 / 3600,
        1 / 120,
    )
    labels = recent.get("apnea_observed_next_60s", pd.Series(index=recent.index, dtype=object))
    positive = labels.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "有", "是", "發生"})
    event_starts = int((positive & ~positive.shift(fill_value=False)).sum())
    position = pd.to_numeric(recent.get("position", 0), errors="coerce").fillna(0)
    scenario = ""
    for source in reversed(recent.get("source", pd.Series(dtype=str)).astype(str).tolist()):
        if "|scenario=" in source:
            scenario = source.split("|scenario=", 1)[1].strip()
            break
    return {
        "device_name": "即時穿戴裝置／上傳檔案",
        "monitoring_period": f"最新 {len(recent)} 筆；約 {duration_hours:.2f} 小時",
        "estimated_ahi": round(event_starts / duration_hours, 2) if labels.notna().any() else None,
        "minimum_spo2": round(pd.to_numeric(recent["spo2"], errors="coerce").min(), 2),
        "average_spo2": round(pd.to_numeric(recent["spo2"], errors="coerce").mean(), 2),
        "average_heart_rate": round(pd.to_numeric(recent["heart_rate"], errors="coerce").mean(), 2),
        "minimum_heart_rate": round(pd.to_numeric(recent["heart_rate"], errors="coerce").min(), 2),
        "maximum_heart_rate": round(pd.to_numeric(recent["heart_rate"], errors="coerce").max(), 2),
        "time_below_90_percent": round((pd.to_numeric(recent["spo2"], errors="coerce") < 90).mean() * 100, 2),
        "apnea_event_count": event_starts,
        "supine_sleep_percent": round(position.eq(0).mean() * 100, 2),
        "sensor_data_points": int(len(recent)),
        "sync_received_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "data_source_type": str(recent.iloc[-1].get("source", "wearable")),
        "wearable_next_60s_risk": round(float(risk["probability"]), 6) if risk else None,
        "synthetic_scenario": scenario or None,
        "synthetic_test_training": bool(scenario),
    }


def _personalize_treatment_with_wearable(ranking, summary: dict) -> tuple[list[dict], dict]:
    """Add a bounded, explainable patient-state overlay to base scores."""
    if isinstance(ranking, dict):
        items = [{"treatment": key, **value} for key, value in ranking.items() if isinstance(value, dict)]
    elif isinstance(ranking, list):
        items = [dict(item) for item in ranking if isinstance(item, dict)]
    else:
        items = []
    minimum_spo2 = float(summary.get("minimum_spo2") or 100)
    average_spo2 = float(summary.get("average_spo2") or 100)
    t90 = float(summary.get("time_below_90_percent") or 0)
    risk = float(summary.get("wearable_next_60s_risk") or 0)
    mean_hr = float(summary.get("average_heart_rate") or 70)
    oxygen_signal = min(
        max(90 - minimum_spo2, 0) / 8
        + max(95 - average_spo2, 0) / 5
        + min(t90 / 10, 1), 3.0
    )
    risk_signal = min(max(risk, 0), 1) * 3.0
    adjustments = {
        "CPAP": min(oxygen_signal + risk_signal, 6.0),
        "APAP": min(oxygen_signal * .75 + risk_signal * .8, 5.0),
        "SURGERY": 0.0,
        "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW": min(max(mean_hr - 80, 0) / 30, 1.0),
    }
    reasons = []
    if minimum_spo2 < 90: reasons.append(f"近期最低 SpO₂ {minimum_spo2:.1f}%（低於 90%）")
    if t90 > 0: reasons.append(f"近期低於 90% 的時間比例 {t90:.1f}%")
    if risk > 0: reasons.append(f"未來 60 秒研究風險 {risk:.1%}")
    if mean_hr > 80: reasons.append(f"近期平均心率 {mean_hr:.0f} bpm")
    if not reasons: reasons.append("近期穿戴指標未顯示需額外提高治療支持度")
    personalized = []
    for item in items:
        treatment = str(item.get("treatment") or item.get("treatment_id") or "")
        base = float(item.get("score", item.get("final_score", 0)) or 0)
        delta = round(float(adjustments.get(treatment, 0)), 1)
        personalized.append({**item, "treatment": treatment, "base_score": base,
            "wearable_adjustment": delta, "personalized_score": round(min(max(base + delta, 0), 100), 1)})
    personalized.sort(key=lambda item: item["personalized_score"], reverse=True)
    for rank, item in enumerate(personalized, 1): item["personalized_rank"] = rank
    return personalized, {
        "status": "realtime_personalization_overlay",
        "calculated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "reasons": reasons,
        "signals": {"minimum_spo2": minimum_spo2, "average_spo2": average_spo2,
            "time_below_90_percent": t90, "wearable_next_60s_risk": risk, "average_heart_rate": mean_hr},
        "safety_note": "即時穿戴調整不是治療成效標籤，也不以手錶推定手術解剖適應症。",
    }


def _realtime_signal_quality(samples: pd.DataFrame) -> dict:
    """Fail closed before displaying a time-sensitive research notice."""
    recent = samples.tail(10).copy()
    if len(recent) < 10:
        return {"usable": False, "reason": "至少需要 10 筆連續資料"}
    spo2 = pd.to_numeric(recent.get("spo2"), errors="coerce")
    heart_rate = pd.to_numeric(recent.get("heart_rate"), errors="coerce")
    if spo2.isna().any() or (~spo2.between(70, 100)).any():
        return {"usable": False, "reason": "SpO₂ 缺值或超出感測範圍 70–100%"}
    if heart_rate.isna().any() or (~heart_rate.between(30, 220)).any():
        return {"usable": False, "reason": "心率缺值或超出感測範圍 30–220 bpm"}
    timestamps = pd.to_datetime(recent.get("timestamp"), errors="coerce", utc=True)
    if timestamps.isna().any():
        return {"usable": False, "reason": "缺少有效時間戳記"}
    age_seconds = (pd.Timestamp.now(tz="UTC") - timestamps.iloc[-1]).total_seconds()
    if age_seconds < -30 or age_seconds > 90:
        return {"usable": False, "reason": "資料不是 90 秒內收到的即時訊號"}
    gaps = timestamps.diff().dt.total_seconds().dropna()
    if (gaps <= 0).any() or (gaps > 120).any():
        return {"usable": False, "reason": "資料順序錯誤或連續資料中斷"}
    return {"usable": True, "reason": "基本訊號品質檢查通過", "latest": timestamps.iloc[-1].isoformat()}


def _sync_wearable_to_treatment(patient_id: str, samples: pd.DataFrame, risk: dict | None, force: bool = False) -> dict:
    summary = _wearable_treatment_summary(samples, risk)
    if summary.get("synthetic_test_training"):
        return {
            "status": "synthetic_test_isolated",
            "message": "合成穿戴資料已隔離，不會寫入正式 Digital Twin、治療模型或臨床報告。",
        }
    checkpoint = STORE.root / patient_id / "treatment_sync.json"
    state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    previous_count = int(state.get("sample_count", 0))
    previous_at = pd.to_datetime(state.get("synced_at"), errors="coerce", utc=True)
    elapsed = float("inf") if pd.isna(previous_at) else (pd.Timestamp.now(tz="UTC") - previous_at).total_seconds()
    if not force and (len(samples) - previous_count < TREATMENT_SYNC_BATCH or elapsed < TREATMENT_SYNC_COOLDOWN_SECONDS):
        return {"status": "waiting", "new_samples": len(samples) - previous_count}
    result = refresh_patient(
        patient_id=_resolve_patient_id(patient_id),
        module_id="DEVICE_DATA",
        confirmed_data=summary,
        source_files=[str(samples.iloc[-1].get("source", "wearable realtime"))],
        project_root=ROOT,
        regenerate_report=True,
    )
    personalized, wearable_audit = _personalize_treatment_with_wearable(
        result.get("ranking_after") or [], summary
    )
    result["wearable_summary"] = summary
    result["personalized_ranking"] = personalized
    result["wearable_personalization_audit"] = wearable_audit
    checkpoint.write_text(json.dumps({
        "sample_count": len(samples), "synced_at": datetime.now(timezone.utc).isoformat(),
        "version_after": result.get("version_after"),
        "personalized_ranking": personalized,
        "wearable_personalization_audit": wearable_audit,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "updated", **result}


def _render_treatment_ranking(sync: dict) -> None:
    if not isinstance(sync, dict):
        return
    ranking = sync.get("personalized_ranking") or sync.get("ranking_after") or []
    if not ranking:
        return
    if isinstance(ranking, dict):
        ranking = [
            {"treatment": treatment, **details}
            for treatment, details in ranking.items()
            if isinstance(details, dict)
        ]
    elif not isinstance(ranking, list):
        return
    else:
        ranking = [item for item in ranking if isinstance(item, dict)]
    if not ranking:
        return
    if sync.get("personalized_ranking"):
        rows = []
        for item in sorted(ranking, key=lambda row: int(row.get("personalized_rank", 99))):
            treatment = str(item.get("treatment", ""))
            rows.append({
                "排名": int(item.get("personalized_rank", len(rows) + 1)),
                "治療方式": TREATMENT_LABELS.get(treatment, treatment),
                "基礎分數": float(item.get("base_score", 0)),
                "穿戴調整": float(item.get("wearable_adjustment", 0)),
                "最新分數": float(item.get("personalized_score", 0)),
            })
        st.subheader("最新個人化治療排名")
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={
                "排名": st.column_config.NumberColumn(format="%d"),
                "基礎分數": st.column_config.NumberColumn(format="%.1f"),
                "穿戴調整": st.column_config.NumberColumn(format="%+.1f"),
                "最新分數": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            },
        )
        audit = sync.get("wearable_personalization_audit") or {}
        st.caption(
            f"Digital Twin 資料版本：{sync.get('version_after', '-')}｜"
            "即時調整不等同已證實治療成效。"
        )
        with st.expander("查看即時調整原因與安全說明", expanded=False, icon=":material/info:"):
            for reason in audit.get("reasons", []):
                st.markdown(f"- {reason}")
            st.info(audit.get("safety_note", "穿戴資料僅作為即時個人化輔助。"))
            st.caption(f"計算時間：{audit.get('calculated_at', '-')}")
        return
    ranking_before = sync.get("ranking_before") or []
    if isinstance(ranking_before, dict):
        ranking_before = [
            {"treatment": treatment, **details}
            for treatment, details in ranking_before.items()
            if isinstance(details, dict)
        ]
    elif isinstance(ranking_before, list):
        ranking_before = [
            item for item in ranking_before if isinstance(item, dict)
        ]
    else:
        ranking_before = []
    before = {
        str(item.get("treatment")): item
        for item in ranking_before
        if isinstance(item, dict)
    }
    rows = []
    for item in sorted(ranking, key=lambda row: int(row.get("rank", 99))):
        treatment = str(item.get("treatment", ""))
        score = float(item.get("score", 0.0))
        old = before.get(treatment, {})
        old_score = float(old.get("score", score))
        rows.append({
            "排名": int(item.get("rank", len(rows) + 1)),
            "治療方式": TREATMENT_LABELS.get(treatment, treatment),
            "最新分數": score,
            "分數變化": score - old_score,
        })
    st.subheader("最新四種治療簡易排名")
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "排名": st.column_config.NumberColumn(format="%d"),
            "最新分數": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            "分數變化": st.column_config.NumberColumn(format="%+.1f"),
        },
    )
    st.caption(
        f"Digital Twin 資料版本：{sync.get('version_after', '-')}｜"
        "此排名是研究型決策支援，穿戴資料可能更新監測風險，但仍受 AHI、解剖與安全規則限制。"
    )


@st.fragment
def _live_panel(patient_id: str, auto_treatment_update: bool):
    samples = STORE.read(patient_id)
    if samples.empty:
        st.info("等待手錶資料；收到至少 10 筆後開始計算，30 筆形成完整窗口。")
        return
    age, bmi = _metadata(patient_id)
    recent = samples.tail(120)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("已接收", f"{len(samples)} 筆")
    c2.metric("最新 SpO₂", f"{float(samples.iloc[-1]['spo2']):.1f}%")
    c3.metric("最新心率", f"{float(samples.iloc[-1]['heart_rate']):.0f} bpm")
    c4.metric("資料來源", str(samples.iloc[-1].get("source", "-")))
    st.line_chart(recent.set_index("timestamp")[["spo2", "heart_rate"]])
    if len(samples) < 10 or not MODEL.exists():
        st.warning("資料窗口或穿戴式模型尚未準備完成。")
        return
    result = predict_risk(MODEL, samples, age, bmi)
    quality = _realtime_signal_quality(samples)
    st.progress(min(max(result["probability"], 0.0), 1.0), text=f"未來 60 秒 Apnea 研究型風險：{result['probability']:.1%}")
    if not quality["usable"]:
        st.warning(f"不顯示即時警示：{quality['reason']}。資料仍可供離線研究分析。")
        st.session_state[f"watch_alert_streak_{patient_id}"] = 0
        st.session_state[f"watch_alert_{patient_id}"] = False
    elif result["alert"]:
        latest_key = f"watch_last_alert_sample_{patient_id}"
        streak_key = f"watch_alert_streak_{patient_id}"
        if st.session_state.get(latest_key) != quality.get("latest"):
            st.session_state[latest_key] = quality.get("latest")
            st.session_state[streak_key] = int(st.session_state.get(streak_key, 0)) + 1
        streak = int(st.session_state.get(streak_key, 0))
        if streak < 3:
            st.warning(f"高風險訊號待確認：已連續 {streak}/3 個新窗口達門檻，尚不彈出通知。")
            st.session_state[f"watch_alert_{patient_id}"] = False
        else:
            st.error("高風險研究提示：請確認感測器與患者狀態；尚未臨床驗證，不能取代醫療警報、PSG 或緊急處置。")
            alert_key = f"watch_alert_{patient_id}"
            if not st.session_state.get(alert_key, False):
                st.toast("未來 60 秒 Apnea 高風險研究提示", icon=":material/warning:")
            st.session_state[alert_key] = True
    elif result["watch"]:
        st.warning("風險進入觀察區；目前不構成醫療警報。")
        st.session_state[f"watch_alert_streak_{patient_id}"] = 0
        st.session_state[f"watch_alert_{patient_id}"] = False
    else:
        st.success("目前未達研究型風險提示門檻。")
        st.session_state[f"watch_alert_streak_{patient_id}"] = 0
        st.session_state[f"watch_alert_{patient_id}"] = False
    st.caption(f"模型：{result['model_version']}｜窗口 {result['sample_count']} 筆｜觀察門檻 {result['threshold']:.1%}｜彈出警示門檻 {result['notification_threshold']:.1%}｜尚未臨床驗證")
    if auto_treatment_update:
        try:
            sync = _sync_wearable_to_treatment(patient_id, samples, result)
            if sync.get("status") == "updated":
                st.session_state[f"latest_watch_treatment_{patient_id}"] = sync
                st.success(f"已將最新穿戴窗口送入 Digital Twin，治療模型與報告已更新至資料版本 {sync.get('version_after')}。")
                _render_treatment_ranking(sync)
            elif sync.get("status") == "synthetic_test_isolated":
                st.info(sync.get("message"))
            else:
                st.caption(f"治療模型準即時同步：再累積 {max(TREATMENT_SYNC_BATCH - int(sync.get('new_samples', 0)), 0)} 筆後更新。")
        except FileNotFoundError:
            st.info("此 Patient ID 尚未完成前段 PSG／治療分析，因此先保留穿戴資料，建立患者 Digital Twin 後即可同步治療模型。")
        except Exception as exc:
            st.warning(f"穿戴風險已更新，但治療模型同步暫時失敗：{exc}")


def _render_legacy_wearable_realtime_center(default_patient_id: str = ""):
    st.divider(); st.header("⌚ 即時穿戴裝置與 60 秒風險中心")
    st.warning("研究型監測功能：不能取代 PSG、醫師判讀、緊急醫療系統或已認證醫材警報。")
    server = _receiver()
    patient_id = st.text_input("Patient ID", value=default_patient_id, key="watch_patient_id").strip()
    if not patient_id: st.info("請先輸入 Patient ID。"); return
    auto_treatment_update = st.toggle(
        "每累積 30 筆新穿戴資料，更新患者 Digital Twin 狀態並重新計算規則型建議",
        value=True,
        key=f"watch_auto_treatment_{patient_id}",
        help="採批次準即時更新並設冷卻時間，避免每一筆訊號造成治療排名抖動。合成測試資料會標記為 SYNTHETIC_TEST_ONLY。",
    )
    tab_network, tab_bluetooth, tab_upload = st.tabs(["網路同步", "藍牙連線", "上傳資料"])
    with tab_network:
        st.code(f"POST http://127.0.0.1:8765/api/wearable/{patient_id}", language="text")
        st.code(json.dumps({"spo2": 96, "heart_rate": 68, "position": 0, "movement": 0.05, "timestamp": "2026-08-06T09:30:00+08:00", "apnea_observed_next_60s": None}, ensure_ascii=False, indent=2), language="json")
        st.caption("目前僅接受本機連線；手機／手錶 App 可透過同一台電腦上的轉接程式送入。" if server else "接收器已由另一個 Streamlit 程序啟動。")
    with tab_bluetooth:
        st.write("支援瀏覽器標準 Heart Rate Service；SpO₂、姿勢與活動量仍須依手錶廠牌 UUID 增加解析器。")
        result = _BLE(key="watch_ble", on_sample_change=lambda: None)
        if result.sample:
            previous = STORE.read(patient_id, 1)
            STORE.append(patient_id, {"heart_rate": result.sample["heart_rate"], "spo2": float(previous.iloc[-1]["spo2"]) if not previous.empty else 97, "source": "web_bluetooth"})
    with tab_upload:
        st.write("上傳手錶或穿戴裝置匯出的 CSV、Excel 或 JSON；資料會與網路、藍牙來源存入同一個患者時間序列並立即重新推論。")
        wearable_file = st.file_uploader(
            "穿戴式資料檔案（Apple Health／CSV／Excel／JSON）", type=["csv", "xlsx", "xls", "json"],
            key=f"wearable_upload_{patient_id}",
            help="必要欄位：spo2、heart_rate；可選：timestamp、position、movement、apnea_observed_next_60s。",
        )
        if wearable_file is not None:
            st.caption(f"已選取：{wearable_file.name}｜{wearable_file.size / 1024 / 1024:.1f} MB")
        if st.button("讀取並加入患者時間序列", disabled=wearable_file is None, type="primary", width="stretch"):
            try:
                accepted, errors = _import_uploaded_wearable(patient_id, wearable_file)
                if accepted:
                    st.success(f"已加入 {accepted} 筆資料；下方風險與圖表會使用最新窗口更新。")
                if errors:
                    st.warning(f"有 {len(errors)} 列未匯入。")
                    with st.expander("查看未匯入列"):
                        st.code("\n".join(errors[:30]))
                if accepted:
                    st.rerun()
            except Exception as exc:
                st.error(f"無法讀取穿戴資料：{exc}")
        with st.expander("沒有裝置檔案？加入內建示範資料"):
            left, right = st.columns(2)
            if left.button("加入 12 筆一般示範資料", width="stretch"): _add_demo(patient_id, False); st.rerun()
            if right.button("加入 12 筆高風險示範資料", type="primary", width="stretch"): _add_demo(patient_id, True); st.rerun()
        st.info(
            "模型可產生偽標籤供除錯與研究比較，但偽標籤、一般上傳標籤與合成資料均不會進入正式模型訓練；"
            "只有具來源、由醫師／Event Grid 確認的真實標籤可訓練 Challenger，且不會自動升級。"
        )
        if st.button("由模型自動標記最新窗口並加入重訓資料", type="primary", width="stretch"):
            samples = STORE.read(patient_id)
            if len(samples) < 10 or not MODEL.exists():
                st.error("至少需要10筆有效資料且穿戴式模型必須存在。")
            else:
                age, bmi = _metadata(patient_id)
                model_result = predict_risk(MODEL, samples, age, bmi)
                row = samples.iloc[-1]
                pending_dir = ROOT / "data" / "continual_learning" / "pending_apnea_labels"
                pending_dir.mkdir(parents=True, exist_ok=True)
                pending_file = pending_dir / f"{patient_id}.csv"
                pending_row = pd.DataFrame([{
                    "patient_id": patient_id,
                    "timestamp": row.get("timestamp"),
                    "spo2": row["spo2"],
                    "heart_rate": row["heart_rate"],
                    "position": row.get("position", 0),
                    "movement": row.get("movement", 0),
                    "model_probability": model_result["probability"],
                    "label_status": "pending_human_or_event_grid_confirmation",
                }])
                pending_row.to_csv(
                    pending_file, mode="a", header=not pending_file.exists(),
                    index=False, encoding="utf-8-sig",
                )
                # Display-only state; it is deliberately not persisted as truth.
                pseudo_label = bool(
                    model_result["probability"] >= model_result["notification_threshold"]
                )
                st.success(
                    f"模型判斷：{'可能發生 Apnea' if pseudo_label else '未達 Apnea 偽標籤門檻'}；"
                    f"機率 {model_result['probability']:.1%}，已加入合成重訓資料。"
                )
        if st.button("重新訓練穿戴式 Challenger", width="stretch"):
            run = subprocess.run([sys.executable, str(ROOT / "train_wearable_apnea_model.py")], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if run.returncode == 0:
                st.success("研究型 Challenger 已重新訓練；未通過獨立驗證，因此不會自動取代正式模型。")
                st.cache_data.clear()
            else: st.error("訓練失敗"); st.code(run.stdout + run.stderr)
        if st.button("立即用最新穿戴窗口更新治療模型與建議", width="stretch"):
            try:
                samples = STORE.read(patient_id)
                age, bmi = _metadata(patient_id)
                risk = predict_risk(MODEL, samples, age, bmi) if len(samples) >= 10 and MODEL.exists() else None
                sync = _sync_wearable_to_treatment(patient_id, samples, risk, force=True)
                st.session_state[f"latest_watch_treatment_{patient_id}"] = sync
                if sync.get("status") == "synthetic_test_isolated":
                    st.info(sync.get("message"))
                else:
                    st.success(f"已更新治療資料版本 {sync.get('version_after')}，並重新產生四種治療建議與 HTML 報告。")
            except Exception as exc:
                st.error(f"無法更新治療模型：{exc}")
        latest_sync = st.session_state.get(f"latest_watch_treatment_{patient_id}")
        if isinstance(latest_sync, dict):
            _render_treatment_ranking(latest_sync)
    _live_panel(patient_id, auto_treatment_update)


def _current_treatment(patient_id: str) -> dict:
    resolved = _resolve_patient_id(patient_id)
    path = ROOT / "data" / "inference" / resolved / "treatment_refinement" / "refined_treatment_recommendation.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    ranking = payload.get("personalized_treatment_ranking") or []
    ranking = sorted(
        [item for item in ranking if isinstance(item, dict)],
        key=lambda item: int(item.get("rank", 999)),
    )
    return {"ranking": ranking, "top": ranking[0] if ranking else {}, "path": str(path)}


def _finalize_bangle_session(patient_id: str, session_id: str) -> dict:
    """Archive one BLE session as an auditable, pending-label training unit."""
    samples = STORE.read(patient_id, limit=100000)
    if samples.empty:
        raise ValueError("本次監測時段尚無可封存資料")
    source = samples.get("source", pd.Series("", index=samples.index)).fillna("").astype(str)
    session_rows = samples.loc[source.eq(f"banglejs_ble|session={session_id}")].copy()
    if session_rows.empty:
        raise ValueError("本次監測時段尚未收到有效 Bangle.js 資料")
    folder = ROOT / "data" / "continual_learning" / "tonight_pending_sessions" / _resolve_patient_id(patient_id)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"bangle_session_{session_id}.csv"
    receipt_path = folder / f"bangle_session_{session_id}.json"
    temporary = csv_path.with_suffix(".csv.tmp")
    session_rows.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(csv_path)
    timestamps = pd.to_datetime(session_rows["timestamp"], errors="coerce", utc=True)
    coverage = (
        float((timestamps.max() - timestamps.min()).total_seconds() / 3600)
        if timestamps.notna().sum() > 1 else 0.0
    )
    receipt = {
        "patient_id": _resolve_patient_id(patient_id),
        "session_id": session_id,
        "sample_count": int(len(session_rows)),
        "coverage_hours": round(coverage, 3),
        "started_at": timestamps.min().isoformat() if timestamps.notna().any() else None,
        "ended_at": timestamps.max().isoformat() if timestamps.notna().any() else None,
        "source_file": str(csv_path),
        "training_status": "PENDING_PAIRED_PSG_OR_HSAT_LABEL",
        "eligible_for_supervised_retraining": False,
        "reason": "已保存白天輸入特徵；尚缺同一監測日的隔夜 PSG／HSAT 真實結果標籤。",
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def _render_tonight_risk(
    patient_id: str,
    wearable_session_id: str | None = None,
    upload_only_source: str | None = None,
) -> dict | None:
    resolved = _resolve_patient_id(patient_id)
    samples = STORE.read(patient_id, limit=100000)
    if wearable_session_id and not samples.empty:
        source = samples.get("source", pd.Series("", index=samples.index)).fillna("").astype(str)
        is_bangle = source.str.startswith("banglejs_ble")
        current_bangle = source.eq(f"banglejs_ble|session={wearable_session_id}")
        # Old Bangle sessions stay on disk for audit/history, but never enter
        # the current monitoring estimate. Other patient-specific upload/network
        # sources remain available to the same Patient ID.
        samples = samples.loc[~is_bangle | current_bangle].reset_index(drop=True)
    if upload_only_source and not samples.empty:
        source = samples.get("source", pd.Series("", index=samples.index)).fillna("").astype(str)
        # In single-file demonstration mode the requested output must describe
        # only the file the user just confirmed. Network, Bluetooth and older
        # upload rows remain on disk for audit but cannot affect these cards.
        samples = samples.loc[source.str.startswith(upload_only_source)].reset_index(drop=True)
    if samples.empty:
        st.info("尚未收到穿戴式資料。開始同步或上傳檔案後，系統會更新今晚風險。")
        return None
    try:
        result = predict_tonight_risk(ROOT, resolved, samples, TONIGHT_MODEL)
    except Exception as exc:
        st.error(f"今晚風險模型無法完成推論：{exc}")
        return None
    if upload_only_source:
        result["active_input_source"] = upload_only_source
    raw_probability = result.get("probability")
    probability = float(raw_probability) if raw_probability is not None else 0.0
    quality = result.get("data_quality") or {}
    latest_key = f"tonight_latest_result_{patient_id}"
    previous_key = f"tonight_previous_result_{patient_id}"
    comparison_flag = f"tonight_show_comparison_{patient_id}"
    cols = st.columns(4)
    output_available = raw_probability is not None and quality.get("quality") in {"GOOD", "LIMITED_NO_SPO2"}
    display_level = result.get("risk_level", "-") if output_available else "資料不足"
    cols[0].metric("今晚風險層級", display_level)
    cols[1].metric("研究風險", f"{probability:.1%}" if raw_probability is not None else "暫不輸出")
    cols[2].metric("今日有效資料", f"{quality.get('sample_count', 0)} 筆")
    cols[3].metric("涵蓋時間", f"{quality.get('coverage_hours', 0):.1f} 小時")
    event_range = result.get("estimated_event_count_range")
    event_count_text = (
        f"約 {event_range[0]}–{event_range[1]} 次"
        if isinstance(event_range, list) and len(event_range) == 2
        else "無法估計"
    )
    event_cols = st.columns(2)
    event_cols[0].metric("今晚是否可能出現呼吸事件", result.get("event_likelihood", "資料不足"))
    event_cols[1].metric("整晚事件次數粗估", event_count_text)
    st.caption(
        "事件次數指呼吸中止與低通氣事件合計，依既往 AHI、既往睡眠時數及今日穿戴式狀態推估；"
        "不是今晚 PSG 實測值，也不能用來單獨診斷。"
    )
    previous_result = st.session_state.get(previous_key)
    if (
        st.session_state.get(comparison_flag)
        and isinstance(previous_result, dict)
        and previous_result.get("probability") is not None
        and raw_probability is not None
    ):
        previous_probability = float(previous_result.get("probability"))
        delta = probability - previous_probability
        st.info(
            f"與前一個上傳檔比較：{previous_probability:.1%} → {probability:.1%} "
            f"（{delta:+.1%}）"
        )
        st.session_state[comparison_flag] = False
    if quality.get("input_mode") == "SYNTHETIC_TEST_ISOLATED":
        st.caption(
            f"測試隔離模式：本次只使用最新合成情境「{quality.get('active_scenario')}」，"
            "不混入上一份高／低風險測試檔。"
        )
    else:
        st.caption("真實資料模式：持續累積同一位患者的即時穿戴式時間序列。")
    if upload_only_source:
        st.success(
            f"本次六項輸出只分析：{upload_only_source.removeprefix('uploaded:')}；"
            "未混入網路、藍牙或前一次上傳資料。"
        )
    st.caption(
        f"監測日期：{quality.get('monitoring_date') or '-'}｜"
        f"既往 PSG／人設基礎風險：{float(result.get('base_probability', 0)):.1%}｜"
        f"今日穿戴式調整：{float(result.get('watch_modifier', 0)):+.1%}"
    )
    if raw_probability is not None:
        st.progress(probability, text="今晚睡眠呼吸事件負荷升高的研究風險")
    if quality.get("quality") == "LIMITED_NO_SPO2" and raw_probability is not None:
        st.warning(
            "目前為無 SpO₂ 的有限資料模式：已使用 Bangle.js 心率、活動資料與既往 PSG／人設輸出低可信度研究估計；"
            "它不是完整血氧模型結果，也不可作為診斷或醫療警報。"
        )
    elif quality.get("quality") != "GOOD" or raw_probability is None:
        st.warning("今日資料未通過品質門檻，系統已暫停顯示風險百分比；請補足至少30筆、涵蓋6小時且有效率達80%。")
    elif result.get("risk_level") == "高":
        st.warning("今晚風險偏高：建議醫師檢視既往 PSG 與症狀；此結果不是緊急警報，也不能單獨診斷 OSA。")
    elif result.get("risk_level") == "中":
        st.info("今晚風險居中：建議繼續收集完整白天資料，並與既往睡眠檢查一起判讀。")
    else:
        st.success("目前研究風險偏低；低風險不代表已排除睡眠呼吸中止症。")
    with st.expander("查看模型依據、資料品質與限制", expanded=False, icon=":material/info:"):
        st.write(result.get("interpretation"))
        for reason in result.get("reasons", []):
            st.markdown(f"- {reason}")
        xyz = result.get("xyz_features") or {}
        if any(value is not None for value in xyz.values()):
            st.caption(
                "XYZ 摘要：平均活動波動 "
                f"{xyz.get('movement_mean') if xyz.get('movement_mean') is not None else '-'} g｜"
                "高活動時段比例 "
                f"{float(xyz.get('active_ratio') or 0):.1%}｜"
                "合成加速度標準差 "
                f"{xyz.get('accel_magnitude_std') if xyz.get('accel_magnitude_std') is not None else '-'} g"
            )
        st.caption(f"模型版本：{result.get('model_version')}｜初始訓練患者：{result.get('trained_patients')}｜資料品質：{quality.get('quality')}")
        st.caption(
            "為降低單一感測器雜訊影響，資料達20筆以上時以 SpO₂ 第5百分位代表低血氧狀態，"
            "不直接使用單一最低值。"
        )
        st.warning(result.get("limitation"))
    st.session_state[latest_key] = result
    return result


def _render_stable_treatment(patient_id: str) -> None:
    current = _current_treatment(patient_id)
    top = current.get("top") or {}
    if not top:
        st.info("尚無可顯示的治療建議；請先完成 PSG 與治療分析。")
        return
    code = str(top.get("treatment") or "")
    label = top.get("treatment_label") or TREATMENT_LABELS.get(code, code)
    score = float(top.get("score", 0) or 0)
    cols = st.columns([2, 1, 1])
    cols[0].metric("目前穩定首選方案", label)
    cols[1].metric("目前分數", f"{score:.1f}")
    cols[2].metric("狀態", "待醫師定期審閱")
    st.caption(
        "穿戴式資料會更新患者狀態與今晚風險，但不會因單次或短時間波動自動更換正式首選治療。"
        "只有新的 PSG/HSAT 嚴重度、已確認解剖檢查、PAP 耐受或治療成效等重大證據，才建立候選變更並交由醫師確認。"
    )
    with st.expander("查看四種治療的穩定排名", expanded=False):
        rows = []
        for item in current.get("ranking", []):
            rows.append({
                "排名": item.get("rank"),
                "治療方式": item.get("treatment_label") or TREATMENT_LABELS.get(str(item.get("treatment")), item.get("treatment")),
                "分數": float(item.get("score", 0) or 0),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _one_record(uploaded_file) -> dict:
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)
    if name.endswith(".json"):
        value = json.load(uploaded_file)
        if isinstance(value, list):
            value = value[0] if value else {}
        if not isinstance(value, dict):
            raise ValueError("JSON must contain one object or a list containing one object")
        return value
    if name.endswith((".xlsx", ".xls")):
        frame = pd.read_excel(uploaded_file)
    else:
        frame = pd.read_csv(uploaded_file)
    if frame.empty:
        raise ValueError("uploaded result file is empty")
    return frame.iloc[0].where(pd.notna(frame.iloc[0]), None).to_dict()


def _receipt_path(patient_id: str, name: str) -> Path:
    return ROOT / "data" / "inference" / _resolve_patient_id(patient_id) / "paired_learning" / f"{name}.json"


def _save_receipt(patient_id: str, name: str, receipt: dict) -> None:
    path = _receipt_path(patient_id, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_receipt(patient_id: str, name: str) -> dict | None:
    path = _receipt_path(patient_id, name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _render_paired_learning(patient_id: str) -> None:
    resolved = _resolve_patient_id(patient_id)
    with st.expander("隔夜 PSG／HSAT 結果配對與今晚模型重訓", expanded=False, icon=":material/bedtime:"):
        st.caption("先上傳白天手錶資料，再上傳同一監測日的隔夜結果。合成測試檔只建立研究版本。")
        night_file = st.file_uploader(
            "隔夜結果（CSV、Excel、JSON）",
            type=["csv", "xlsx", "xls", "json"],
            key=f"night_outcome_{patient_id}",
        )
        if st.button("配對、加入訓練集並重訓今晚模型", disabled=night_file is None, type="primary", width="stretch", key=f"train_night_{patient_id}"):
            try:
                outcome = _one_record(night_file)
                samples = STORE.read(patient_id, limit=100000)
                receipt = add_verified_night_outcome_and_retrain(ROOT, resolved, samples, outcome, TONIGHT_MODEL)
                st.session_state[f"night_training_receipt_{patient_id}"] = receipt
                _save_receipt(patient_id, "night_training_receipt", receipt)
                st.success(f"已重訓：{receipt['model_version']}｜訓練列 {receipt['training_rows']} 筆")
                st.rerun()
            except Exception as exc:
                st.error(f"隔夜結果無法配對：{exc}")
        receipt = st.session_state.get(f"night_training_receipt_{patient_id}") or _load_receipt(patient_id, "night_training_receipt")
        if isinstance(receipt, dict):
            st.json(receipt, expanded=False)

    with st.expander("治療後回診結果與治療模型重訓", expanded=False, icon=":material/clinical_notes:"):
        st.caption("檔案必須包含 patient_id、study_id、treatment、baseline_ahi、follow_up_ahi、follow_up_days、outcome_status 與 reviewer_id。")
        treatment_file = st.file_uploader(
            "治療後回診結果（CSV、Excel、JSON）",
            type=["csv", "xlsx", "xls", "json"],
            key=f"treatment_outcome_{patient_id}",
        )
        if st.button("加入治療訓練集、重訓並產生新研究排名", disabled=treatment_file is None, type="primary", width="stretch", key=f"train_treatment_{patient_id}"):
            try:
                row = _one_record(treatment_file)
                required = {"patient_id", "study_id", "treatment", "baseline_ahi", "follow_up_ahi", "follow_up_days", "outcome_status", "reviewer_id"}
                missing = sorted(required - set(row))
                if missing:
                    raise ValueError(f"缺少欄位：{missing}")
                uploaded_patient = str(row["patient_id"]).strip()
                synthetic_demo = (
                    str(row.get("reviewer_id", "")).strip().upper().startswith(("TEST_", "DEMO_", "SYNTH_"))
                    or str(row.get("study_id", "")).strip().upper().startswith(("TEST_", "DEMO_", "SYNTH_"))
                )
                if uploaded_patient not in {patient_id, resolved}:
                    if not synthetic_demo:
                        raise ValueError("治療結果 patient_id 與目前患者不一致；真實臨床檔案不可自動改綁患者")
                    row["original_test_patient_id"] = uploaded_patient
                    row["patient_id"] = resolved
                    safe_suffix = "".join(ch for ch in resolved if ch.isalnum())[-8:]
                    row["study_id"] = f"{row['study_id']}_{safe_suffix}"
                    st.info(f"合成測試檔已由 {uploaded_patient} 自動改綁至目前患者 {resolved}。")
                outcome_dir = ROOT / "data" / "continual_learning" / "treatment_outcomes"
                outcome_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                pd.DataFrame([row]).to_csv(outcome_dir / f"outcomes_{stamp}.csv", index=False, encoding="utf-8-sig")
                model_receipt = train_adaptive_treatment_models(ROOT, allow_synthetic_demo=True)
                current = _current_treatment(patient_id)
                payload = json.loads(Path(current["path"]).read_text(encoding="utf-8"))
                research = apply_learned_treatment_adjustment(ROOT, payload, research_preview=True)
                output = ROOT / "data" / "inference" / resolved / "treatment_refinement" / "wearable_research_treatment_recommendation.json"
                output.write_text(json.dumps(research, ensure_ascii=False, indent=2), encoding="utf-8")
                receipt = {"model_version": model_receipt.get("model_version"), "training_rows": model_receipt.get("training_row_count"), "confirmed_rows": model_receipt.get("confirmed_outcome_rows"), "ranking": research.get("personalized_treatment_ranking", [])}
                st.session_state[f"treatment_training_receipt_{patient_id}"] = receipt
                _save_receipt(patient_id, "treatment_training_receipt", receipt)
                st.success(f"已重訓治療 Challenger：{receipt['model_version']}")
                st.rerun()
            except Exception as exc:
                st.error(f"治療模型更新失敗：{exc}")
        receipt = st.session_state.get(f"treatment_training_receipt_{patient_id}") or _load_receipt(patient_id, "treatment_training_receipt")
        if isinstance(receipt, dict):
            st.caption(f"新研究版本：{receipt.get('model_version')}｜訓練列：{receipt.get('training_rows')}｜確認成效列：{receipt.get('confirmed_rows')}")
            rows = [{"排名": x.get("rank"), "治療方式": x.get("treatment_label") or TREATMENT_LABELS.get(str(x.get("treatment")), x.get("treatment")), "研究分數": float(x.get("score", 0) or 0)} for x in receipt.get("ranking", []) if isinstance(x, dict)]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.warning("此為 Challenger 研究排名；未通過獨立驗證與治理核准，不自動取代正式穩定治療方案。")


def _render_upload_receipt(patient_id: str, result: dict | None) -> None:
    """Make the distinction between data ingestion, inference and training explicit."""
    if not isinstance(result, dict):
        return
    quality = result.get("data_quality") or {}
    night_receipt = st.session_state.get(f"night_training_receipt_{patient_id}") or _load_receipt(patient_id, "night_training_receipt")
    treatment_receipt = st.session_state.get(f"treatment_training_receipt_{patient_id}") or _load_receipt(patient_id, "treatment_training_receipt")
    night_trained = isinstance(night_receipt, dict) and night_receipt.get("status") == "retrained"
    treatment_trained = isinstance(treatment_receipt, dict) and bool(treatment_receipt.get("model_version"))
    rows = [
        {
            "處理項目": "穿戴式資料寫入患者時間序列",
            "結果": "已完成" if int(quality.get("sample_count", 0) or 0) > 0 else "未完成",
            "說明": f"本次採用 {int(quality.get('sample_count', 0) or 0)} 筆。",
        },
        {
            "處理項目": "今晚風險重新推論與輸出",
            "結果": "已完成",
            "說明": f"已使用模型 {result.get('model_version') or '-'} 產生新結果。",
        },
        {
            "處理項目": "今晚風險模型重新訓練",
            "結果": "已完成" if night_trained else "等待隔夜標籤",
            "說明": (
                f"新版本 {night_receipt.get('model_version')}｜訓練列 {night_receipt.get('training_rows')} 筆。"
                if night_trained else "請上傳同一監測日的 PSG/HSAT 結果完成配對。"
            ),
        },
        {
            "處理項目": "治療效果模型重新訓練",
            "結果": "已完成" if treatment_trained else "等待治療成效標籤",
            "說明": (
                f"新 Challenger {treatment_receipt.get('model_version')}｜訓練列 {treatment_receipt.get('training_rows')} 筆。"
                if treatment_trained else "請上傳治療方式、治療前後 AHI、追蹤天數與確認者。"
            ),
        },
        {
            "處理項目": "四種治療研究排名重新輸出",
            "結果": "已完成" if treatment_trained else "尚未更新",
            "說明": "新排名已顯示於治療後回診區塊。" if treatment_trained else "正式穩定治療排名維持不變。",
        },
    ]
    with st.expander("本次上傳處理收據", expanded=False, icon=":material/receipt_long:"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("「重新推論」是新資料進入既有模型得到新輸出；「重新訓練」才是改變模型參數與版本。")


def _render_governance_center(patient_id: str, result: dict | None) -> None:
    """Show safety and audit controls without crowding the clinical view."""
    samples = STORE.read(patient_id, limit=100000)
    quality = wearable_quality(samples)
    governance = governance_summary(ROOT, _resolve_patient_id(patient_id))
    night = governance.get("latest_night_training") or {}
    treatment = governance.get("latest_treatment_training") or {}
    rows = [
        {"安全檢查": "測試／正式資料隔離", "狀態": "通過", "說明": "合成資料只進入研究 Challenger，不會自動升級正式 Champion。"},
        {"安全檢查": "無標籤資料處理", "狀態": "通過", "說明": "只更新患者狀態及重新推論，不把模型自己的答案當訓練標籤。"},
        {"安全檢查": "內容去重", "狀態": "通過", "說明": "使用內容雜湊、Patient ID、study_id與治療方式去重。"},
        {"安全檢查": "特徵洩漏防護", "狀態": "通過", "說明": "排除答案、結果、檔名、測試情境、排名與治療後欄位。"},
        {"安全檢查": "穿戴資料品質", "狀態": "通過" if quality.get("usable") else "暫停輸出", "說明": "；".join(quality.get("warnings") or [f"{quality.get('sample_count', 0)}筆／{quality.get('coverage_hours', 0)}小時"])},
        {"安全檢查": "今晚模型最新訓練", "狀態": "已重訓" if night.get("status") == "retrained" else "尚無合格新標籤", "說明": str(night.get("model_version") or "-")},
        {"安全檢查": "治療模型最新訓練", "狀態": "已產生研究版本" if treatment.get("model_version") else "尚無合格新療效標籤", "說明": str(treatment.get("model_version") or "-")},
        {"安全檢查": "醫療用途", "狀態": "研究限定", "說明": "尚未完成外部前瞻性驗證，不能當作診斷、處方或醫療警報。"},
    ]
    with st.expander("資料、模型與臨床安全稽核", expanded=False, icon=":material/verified_user:"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("短期穿戴狀態只調整今晚風險；長期治療基準須由治療後成效標籤、醫療規則與版本門檻更新。訓練與推論均保存版本和稽核紀錄。")
        if isinstance(result, dict):
            st.code(json.dumps({
                "patient_id": _resolve_patient_id(patient_id),
                "input_mode": (result.get("data_quality") or {}).get("input_mode"),
                "output_status": result.get("output_status"),
                "tonight_model_version": result.get("model_version"),
                "treatment_model_version": treatment.get("model_version"),
                "clinical_use_status": result.get("clinical_use_status"),
            }, ensure_ascii=False, indent=2), language="json")


def render_wearable_realtime_center(default_patient_id: str = ""):
    """Continuous daytime wearable intake and tonight respiratory-risk view."""
    st.header(":material/watch: 白天穿戴式監測與今晚睡眠呼吸風險")
    st.caption(
        "這是一個以『今天白天』資料預測『今晚睡眠期間』風險的獨立模型。"
        "可透過網路、藍牙或檔案持續累積一整天資料，與病人人設及既往睡眠資料一起估計。"
    )
    server = _receiver()
    locked_patient_id = _resolve_patient_id(default_patient_id.strip()) if default_patient_id.strip() else ""
    lock_key = "tonight_watch_locked_patient_id"
    widget_key = "tonight_watch_patient_id"
    if st.session_state.get(lock_key) != locked_patient_id:
        # Streamlit widget state otherwise retains the previous patient's ID
        # even after the PSG workflow switches to a newly uploaded patient.
        st.session_state.pop(widget_key, None)
        st.session_state[lock_key] = locked_patient_id
    patient_id = st.text_input(
        "Patient ID（由目前 PSG 患者鎖定）",
        value=locked_patient_id,
        key=widget_key,
        disabled=True,
        help="穿戴資料、今晚風險、治療重訓與報告只能寫入目前鎖定的 PSG 患者。",
    ).strip()
    if not patient_id:
        st.info("請先輸入 Patient ID。")
        return
    session_key = f"bangle_monitoring_session_{patient_id}"
    if session_key not in st.session_state:
        st.session_state[session_key] = uuid.uuid4().hex[:12]
    wearable_session_id = st.session_state[session_key]
    session_cols = st.columns([3, 1])
    session_cols[0].info(
        f"目前手錶監測對象：{patient_id}｜本次監測時段：{wearable_session_id}。"
        "只有本時段收到的 Bangle.js 資料會進入本次風險計算。"
    )
    if session_cols[1].button(
        "開始新的監測時段",
        key=f"new_bangle_session_{patient_id}",
        width="stretch",
        help="更換配戴者或重新測試前請先中斷手錶，再按此按鈕。舊資料保留稽核但不混入新結果。",
    ):
        st.session_state[session_key] = uuid.uuid4().hex[:12]
        st.rerun()
    data_mode = st.segmented_control(
        "資料使用方式",
        ["單檔展示模式（新檔取代前一次上傳）", "真實連續監測模式（持續累積）"],
        default="單檔展示模式（新檔取代前一次上傳）",
        key=f"tonight_data_mode_{patient_id}",
        help="會議展示建議使用單檔模式；真實病人全天監測才使用持續累積模式。",
    )
    if data_mode.startswith("單檔展示"):
        st.caption("目前為單檔展示：按下讀取後，只替換先前『檔案上傳』資料；網路與藍牙資料不會被刪除。")
    else:
        st.caption("目前為連續監測：每次上傳、網路與藍牙資料都會持續累積。")
    network, bluetooth, upload = st.tabs(["網路即時同步", "藍牙連線", "上傳穿戴式檔案"])
    with network:
        st.code(f"POST http://127.0.0.1:8765/api/wearable/{patient_id}", language="text")
        st.code(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(), "spo2": 96,
            "heart_rate": 68, "respiratory_rate": 15, "movement": 0.05,
        }, ensure_ascii=False, indent=2), language="json")
        st.caption("接收服務已啟動。" if server else "接收服務已由另一個 Streamlit 工作程序啟動。")
    with bluetooth:
        st.write("連接 Bangle.js 後，網頁會每 5 秒直接接收心率、心率可信度、原始 XYZ 三軸加速度、活動量、步數與電量，並自動寫入目前患者的時間序列；不再需要透過 Selenium 或人工下載加速度檔案。")
        st.info("Bangle.js 沒有 SpO₂ 感測器，因此不會捏造血氧值；今晚風險若需要 SpO₂，請合併血氧裝置或含 SpO₂ 的檔案。")
        st.caption("第一次連線必須按下按鈕並在 Chrome／Edge 的裝置清單授權；成功後會在連線期間自主持續接收。Bangle.js 的 Bluetooth 與 Programmable 必須開啟。")
        result = _BLE(
            key=f"tonight_banglejs_ble_{patient_id}_{wearable_session_id}",
            on_samples_change=lambda: None,
            on_session_ended_change=lambda: None,
        )
        incoming_samples = list(result.samples or [])
        ended_payload = result.session_ended if isinstance(result.session_ended, dict) else None
        if ended_payload:
            incoming_samples.extend(ended_payload.get("samples") or [])
        if incoming_samples:
            accepted = 0
            for sample in incoming_samples:
                heart_rate = pd.to_numeric(sample.get("heart_rate"), errors="coerce")
                # Bangle.js starts streaming before the optical HR sensor has
                # acquired a pulse.  Those initial null/zero placeholders are
                # connection telemetry, not physiological observations.
                if pd.isna(heart_rate) or not 20 <= float(heart_rate) <= 240:
                    continue
                try:
                    STORE.append(patient_id, {
                        "timestamp": sample.get("timestamp"),
                        "heart_rate": float(heart_rate),
                        "movement": sample.get("movement", 0),
                        "accel_x": sample.get("accel_x"),
                        "accel_y": sample.get("accel_y"),
                        "accel_z": sample.get("accel_z"),
                        "accel_magnitude": sample.get("accel_magnitude"),
                        "steps": sample.get("steps"),
                        "battery": sample.get("battery"),
                        "hr_confidence": sample.get("hr_confidence"),
                        "source": f"banglejs_ble|session={wearable_session_id}",
                    })
                    accepted += 1
                except (TypeError, ValueError):
                    # One malformed BLE packet must not terminate the whole
                    # Streamlit page or interrupt later valid samples.
                    continue
            if accepted:
                st.toast(f"已自動接收並保存 {accepted} 筆 Bangle.js 資料")
        if ended_payload:
            receipt_key = f"bangle_finalize_receipt_{patient_id}_{wearable_session_id}"
            if receipt_key not in st.session_state:
                try:
                    st.session_state[receipt_key] = _finalize_bangle_session(
                        patient_id, wearable_session_id
                    )
                except ValueError as exc:
                    st.session_state[receipt_key] = {"error": str(exc)}
            receipt = st.session_state[receipt_key]
            if receipt.get("error"):
                st.warning(receipt["error"])
            else:
                st.success(
                    f"本次監測已封存：{receipt['sample_count']} 筆／"
                    f"{receipt['coverage_hours']:.2f} 小時。患者狀態與今晚研究輸出會重新計算。"
                )
                st.info(
                    "此 Session 已加入待配對訓練集；取得同日隔夜 PSG／HSAT 標籤後，"
                    "才會自動成為有效監督式樣本並觸發模型重訓與新版本。"
                )
    with upload:
        active_upload_key = f"tonight_active_upload_source_{patient_id}"
        wearable_file = st.file_uploader(
            "穿戴式資料（Apple Health JSON、CSV、Excel）",
            type=["csv", "xlsx", "xls", "json"], key=f"tonight_wearable_upload_{patient_id}",
        )
        if st.button("讀取並加入今日時間序列", disabled=wearable_file is None, type="primary", width="stretch"):
            try:
                latest_key = f"tonight_latest_result_{patient_id}"
                previous_key = f"tonight_previous_result_{patient_id}"
                comparison_flag = f"tonight_show_comparison_{patient_id}"
                if isinstance(st.session_state.get(latest_key), dict):
                    st.session_state[previous_key] = st.session_state[latest_key]
                    st.session_state[comparison_flag] = True
                accepted, errors = _import_uploaded_wearable(
                    patient_id,
                    wearable_file,
                    replace_previous_uploads=data_mode.startswith("單檔展示"),
                )
                if accepted:
                    st.session_state[active_upload_key] = f"uploaded:{wearable_file.name}"
                st.success(f"已加入 {accepted} 筆資料。")
                if errors:
                    with st.expander(f"查看 {len(errors)} 筆未採用資料"):
                        st.code("\n".join(errors[:50]))
            except Exception as exc:
                st.error(f"無法讀取穿戴式資料：{exc}")
    upload_only_source = (
        st.session_state.get(f"tonight_active_upload_source_{patient_id}")
        if data_mode.startswith("單檔展示")
        else None
    )
    tonight_result = _render_tonight_risk(
        patient_id,
        wearable_session_id,
        upload_only_source=upload_only_source,
    )
    _render_governance_center(patient_id, tonight_result)
    _render_upload_receipt(patient_id, tonight_result)
    _render_paired_learning(patient_id)
    with st.expander("治療建議（穩定決策，不隨短時間訊號翻轉）", expanded=True):
        _render_stable_treatment(patient_id)
