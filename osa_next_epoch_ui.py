from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def render_osa_next_epoch_model(project_root: Path, patient_id: str) -> None:
    """Render the independent causal next-epoch OSA research module."""
    if not patient_id:
        return
    model_summary_path = project_root / "models" / "osa_next_epoch_lstm" / "training_summary.json"
    result_root = project_root / "data" / "inference" / patient_id / "osa_next_epoch_lstm"
    patient_summary_path = result_root / "summary.json"
    prediction_path = result_root / "osa_next_epoch_predictions.csv"
    prediction_ready = patient_summary_path.exists() and prediction_path.exists()
    st.markdown(
        '<span class="dt-step">步驟 5 · 下一個 30 秒 OSA 預測</span>',
        unsafe_allow_html=True,
    )
    with st.expander(
        "本次上傳患者：前 N 個 epoch 預測下一個 30 秒 OSA 事件（單向 LSTM）",
        expanded=prediction_ready,
        icon=":material/timeline:",
    ):
        st.caption(
            "這裡顯示的是本次上傳患者的預測，不是訓練集或測試集的成績。"
            "模型以該患者前7個30秒epoch（3.5分鐘）預測緊接的下一個30秒是否出現阻塞型apnea或hypopnea。"
        )
        if not model_summary_path.exists():
            st.warning("尚未訓練單向LSTM研究模型。")
            return
        if not patient_summary_path.exists() or not prediction_path.exists():
            st.warning("這位患者尚未執行新版Pipeline；重新分析後才會產生逐epoch結果。")
            return
        patient_summary = json.loads(patient_summary_path.read_text(encoding="utf-8"))
        predictions_all = pd.read_csv(prediction_path).sort_values("predicted_epoch_index").reset_index(drop=True)
        if predictions_all.empty:
            st.warning("此患者沒有足夠的連續 epoch 可產生下一個 30 秒預測。")
            return
        latest = predictions_all.iloc[-1]
        threshold = float(patient_summary.get("threshold") or 0.5)
        probability = float(latest["osa_next_30s_probability"])
        latest_start = pd.to_datetime(latest.get("end_time"), errors="coerce")
        next_window = "—" if pd.isna(latest_start) else (
            f"{latest_start.strftime('%Y-%m-%d %H:%M:%S')} ～ "
            f"{(latest_start + pd.Timedelta(seconds=30)).strftime('%H:%M:%S')}"
        )
        st.subheader("最新可預測的下一個 30 秒")
        columns = st.columns(4)
        columns[0].metric("OSA事件機率", f"{probability:.1%}")
        columns[1].metric("風險", "高風險：達研究門檻" if probability >= threshold else "未達研究警示門檻")
        columns[2].metric("預測目標 epoch", int(latest["predicted_epoch_index"]))
        columns[3].metric("使用歷史 epoch", f"{int(latest['history_start_epoch'])}–{int(latest['epoch_index'])}")
        st.caption(f"預測時間窗：{next_window}｜研究警示門檻：{threshold:.1%}")
        columns = st.columns(3)
        columns[0].metric("本夜最高下一epoch風險", f"{float(patient_summary.get('maximum_probability') or 0):.1%}")
        columns[1].metric("本夜平均風險", f"{float(patient_summary.get('mean_probability') or 0):.1%}")
        columns[2].metric("超過研究門檻", int(patient_summary.get("alert_count", 0)))
        st.subheader("本患者風險最高的20個下一 epoch 預測")
        predictions = predictions_all.sort_values("osa_next_30s_probability", ascending=False).head(20)
        st.dataframe(
            predictions,
            hide_index=True,
            width="stretch",
            column_config={
                "osa_next_30s_probability": st.column_config.ProgressColumn(
                    "下一個30秒OSA事件機率", min_value=0.0, max_value=1.0, format="percent"
                )
            },
        )
        st.error(
            "此結果是單中心回溯性研究輸出，尚未完成外部與前瞻性驗證；"
            "不可用來自動診斷、喚醒患者或取代醫師判讀。"
        )
