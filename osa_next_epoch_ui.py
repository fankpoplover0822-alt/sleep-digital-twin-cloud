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
    with st.expander(
        "前 N 個 epoch 預測下一個 epoch 的 OSA 事件（單向 LSTM）",
        expanded=False,
        icon=":material/timeline:",
    ):
        st.caption(
            "研究模組：只使用目前及過去資料；前7個30秒epoch（3.5分鐘）預測下一個30秒epoch。"
            "陽性限定為阻塞型apnea或hypopnea，中央型與混合型apnea不作為OSA陽性標籤。"
        )
        if not model_summary_path.exists():
            st.warning("尚未訓練單向LSTM研究模型。")
            return
        model_summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
        test = model_summary.get("locked_test_metrics") or {}
        columns = st.columns(4)
        columns[0].metric("獨立測試患者", len((model_summary.get("patient_split") or {}).get("test", [])))
        columns[1].metric("Test AUROC", f"{float(test.get('roc_auc', 0)):.3f}")
        columns[2].metric("Test AUPRC", f"{float(test.get('average_precision', 0)):.3f}")
        columns[3].metric("Test Recall", f"{float(test.get('recall', 0)):.1%}")
        st.info(
            "訓練、驗證及鎖定測試按患者分開；警示門檻只由驗證患者選擇，"
            "測試患者不參與特徵標準化、訓練、早停或門檻調整。"
        )
        if not patient_summary_path.exists() or not prediction_path.exists():
            st.warning("這位患者尚未執行新版Pipeline；重新分析後才會產生逐epoch結果。")
            return
        patient_summary = json.loads(patient_summary_path.read_text(encoding="utf-8"))
        columns = st.columns(4)
        columns[0].metric("最高下一epoch風險", f"{float(patient_summary.get('maximum_probability') or 0):.1%}")
        columns[1].metric("平均風險", f"{float(patient_summary.get('mean_probability') or 0):.1%}")
        columns[2].metric("超過研究門檻", int(patient_summary.get("alert_count", 0)))
        columns[3].metric("可預測epoch", int(patient_summary.get("prediction_count", 0)))
        predictions = pd.read_csv(prediction_path).sort_values("osa_next_30s_probability", ascending=False).head(20)
        st.dataframe(
            predictions,
            hide_index=True,
            width="stretch",
            column_config={
                "osa_next_30s_probability": st.column_config.ProgressColumn(
                    "下一個30秒OSA風險", min_value=0.0, max_value=1.0, format="percent"
                )
            },
        )
        st.error(
            "此結果是單中心回溯性研究輸出，尚未完成外部與前瞻性驗證；"
            "不可用來自動診斷、喚醒患者或取代醫師判讀。"
        )
