from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.continual_learning.service import ContinualLearningService
from src.continual_learning.treatment import train_adaptive_treatment_models
from src.continual_learning.apnea60 import Apnea60Service


PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_resource
def get_learning_service() -> ContinualLearningService:
    return ContinualLearningService(PROJECT_ROOT)


def _save_upload(upload: Any, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.name).suffix.lower()
    path = folder / f"upload{suffix}"
    path.write_bytes(upload.getvalue())
    return path


def _render_status(service: ContinualLearningService) -> None:
    status = service.status()
    columns = st.columns(5)
    columns[0].metric("正式 Arousal 模型", status["champion_version"] or "尚未登錄")
    columns[1].metric("已核准 Arousal epoch", status["approved_arousal_rows"])
    columns[2].metric("新增已核准病患", status["approved_new_patients"])
    columns[3].metric("待審核 epoch", status["pending_arousal_rows"])
    columns[4].metric("治療追蹤結果", status["treatment_outcome_rows"])
    if status["arousal_retraining_ready"]:
        st.success("已達 Arousal Challenger 再訓練條件。")
    else:
        st.info("尚未達到再訓練門檻；資料仍會更新病患 Digital Twin，但不會改寫正式模型。")


def _render_apnea60_status() -> None:
    """Show the independently managed 60-second apnea learning model."""
    apnea_service = Apnea60Service(PROJECT_ROOT)
    registry = apnea_service.registry()
    champion_version = registry.get("champion")
    champion_record = next(
        (
            model
            for model in reversed(registry.get("models", []))
            if model.get("model_version") == champion_version
        ),
        {},
    )

    st.markdown("#### 未來 60 秒 Apnea 持續學習狀態")
    columns = st.columns(4)
    columns[0].metric("正式 60 秒模型", champion_version or "尚未建立")
    columns[1].metric("目前訓練患者", champion_record.get("patient_count", 0))
    columns[2].metric("目前訓練 epoch", champion_record.get("row_count", 0))
    columns[3].metric("已學習新增檢查", len(registry.get("processed_studies", [])))

    if champion_record:
        metrics = champion_record.get("metrics", {})
        st.success(
            "60 秒模型會在新的 EDF／Stage／Event Grid 產生可驗證標籤後，"
            "自動去重、加入訓練集、重新訓練並建立版本。"
        )
        st.caption(
            f"目前 ROC-AUC：{metrics.get('roc_auc', '-')}"
            f"｜Recall：{metrics.get('recall', '-')}"
            f"｜Specificity：{metrics.get('specificity', '-')}"
        )
    else:
        st.info("尚未建立未來 60 秒 Apnea 模型。")


def _render_arousal_feedback(service: ContinualLearningService) -> None:
    st.markdown("#### Arousal 真實標籤回饋")
    st.caption(
        "監測特徵與人工確認標籤會以 patient_id、epoch_index 對齊；"
        "study_id 建議提供，缺少時會以本次標籤檔名稱建立。"
        "模型自己的預測不能當成 true_label。"
    )
    feature_upload = st.file_uploader(
        "監測器 epoch 特徵檔",
        type=["csv", "xlsx", "xls", "parquet"],
        key="cl_arousal_features",
    )
    label_upload = st.file_uploader(
        "人工確認 Arousal 標籤檔",
        type=["csv", "xlsx", "xls", "parquet"],
        key="cl_arousal_labels",
    )
    reviewer = st.text_input("審核者 ID", key="cl_arousal_reviewer")
    approve = st.checkbox(
        "我確認 true_label 已由合格人員或經驗證標註流程確認",
        key="cl_arousal_approve",
    )
    if st.button(
        "匯入 Arousal 學習資料",
        type="primary",
        disabled=not (feature_upload and label_upload and reviewer and approve),
    ):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            feature_path = _save_upload(feature_upload, folder / "features")
            label_path = _save_upload(label_upload, folder / "labels")
            try:
                result = service.ingest_arousal_feedback(
                    feature_path,
                    label_path,
                    reviewer_id=reviewer,
                    approve=True,
                )
            except Exception as error:
                st.error(f"匯入失敗：{error}")
            else:
                st.success(
                    f"已核准 {result['row_count']} 個 epoch、"
                    f"{result['patient_count']} 位病患。"
                )
                st.json(result, expanded=False)


def _render_treatment_outcomes(service: ContinualLearningService) -> None:
    sample_path = PROJECT_ROOT / "test_data" / "完整合成治療成效測試_80位患者.xlsx"
    if sample_path.exists():
        st.download_button(
            "下載四種治療結果測試 CSV",
            data=sample_path.read_bytes(),
            file_name=sample_path.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="content",
            key="download_treatment_test_csv",
        )
        st.caption(
            "此檔全部使用 TEST_ 患者，只用於驗證匯入、重訓與版本更新。"
        )
    st.markdown("#### 治療後結果回饋")
    st.caption(
        "必要欄位：patient_id、study_id、treatment、baseline_ahi、"
        "follow_up_ahi、follow_up_days、outcome_status。"
        "outcome_status 必須是 confirmed。"
    )
    outcome_upload = st.file_uploader(
        "治療追蹤結果",
        type=["csv", "xlsx", "xls", "parquet"],
        key="cl_treatment_outcomes",
    )
    reviewer = st.text_input("臨床審核者 ID", key="cl_outcome_reviewer")
    confirmed = st.checkbox(
        "我確認結果已完成臨床審核",
        key="cl_outcome_confirmed",
    )
    if st.button(
        "匯入治療結果",
        disabled=not (outcome_upload and reviewer and confirmed),
    ):
        with tempfile.TemporaryDirectory() as temporary:
            path = _save_upload(outcome_upload, Path(temporary))
            try:
                result = service.ingest_treatment_outcomes(
                    path,
                    reviewer_id=reviewer,
                )
            except Exception as error:
                st.error(f"匯入失敗：{error}")
            else:
                st.success(f"已匯入 {result['row_count']} 筆確認結果。")
                st.json(result, expanded=False)


def _render_treatment_learning_status() -> None:
    registry_path = (
        PROJECT_ROOT
        / "models"
        / "registry"
        / "treatment"
        / "registry.json"
    )
    if not registry_path.exists():
        st.info("尚未建立四種治療的自適應學習模型。")
        return
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    champion = registry.get("champion")
    model_record = next(
        (
            item
            for item in reversed(registry.get("models", []))
            if item.get("model_version") == champion
        ),
        {},
    )
    columns = st.columns(4)
    columns[0].metric("正式治療模型", champion or "尚未建立")
    columns[1].metric("訓練資料", model_record.get("training_row_count", 0))
    columns[2].metric("排除的弱標籤", model_record.get("weak_label_rows", 0))
    columns[3].metric("真實療效標籤", model_record.get("confirmed_outcome_rows", 0))
    st.caption(
        "推薦快照只供稽核，不是療效真值，也不參與訓練。只有經確認的治療後結果可建立研究型 Challenger；"
        "未完成獨立驗證與治理核准前，不會成為正式模型或影響病人排名。"
    )


def _render_model_management(service: ContinualLearningService) -> None:
    st.info(
        "治療模型現在只學習經確認的治療後療效。一般補充資料只更新患者特徵與規則輸出；"
        "推薦快照、測試資料與合成資料不會被當成臨床療效真值。"
    )
    st.warning(
        "若匯入 TEST_／SYNTH_／DEMO_患者，系統會建立可改變研究分數的合成示範模型；"
        "該版本不可升級為正式臨床 Champion，也不代表臨床有效。"
    )
    st.markdown("#### 模型訓練與安全升級")
    st.warning(
        "訓練只會建立 Challenger。必須通過驗證門檻，並由使用者再次確認，"
        "才能成為正式 Champion。"
    )
    left, right = st.columns(2)
    if left.button("訓練 Arousal Challenger"):
        try:
            result = service.train_arousal_challenger()
        except Exception as error:
            st.error(f"訓練未執行：{error}")
        else:
            st.session_state["last_arousal_challenger"] = result["model_version"]
            st.success(f"已建立 Challenger：{result['model_version']}")
            st.json(result, expanded=False)
    if right.button("訓練治療效果模型"):
        try:
            result = train_adaptive_treatment_models(PROJECT_ROOT)
        except Exception as error:
            st.error(f"治療模型訓練未執行：{error}")
        else:
            st.success(f"已建立研究型治療 Challenger：{result['model_version']}；尚不可部署。")
            st.json(result, expanded=False)

    version = st.text_input(
        "Arousal Challenger 版本",
        value=st.session_state.get("last_arousal_challenger", ""),
        key="cl_challenger_version",
    )
    if st.button("執行 Champion–Challenger 驗證", disabled=not version):
        try:
            evaluation = service.evaluate_arousal_challenger(version)
        except Exception as error:
            st.error(f"驗證失敗：{error}")
        else:
            st.session_state["last_arousal_evaluation"] = evaluation
            if evaluation["passed"]:
                st.success("候選模型通過所有升級門檻。")
            else:
                st.error("候選模型未通過升級門檻，不可部署。")
            st.json(evaluation, expanded=True)
    evaluation = st.session_state.get("last_arousal_evaluation", {})
    allow_promotion = bool(
        evaluation.get("passed") and evaluation.get("version") == version
    )
    promotion_confirmed = st.checkbox(
        "我確認要將通過驗證的 Challenger 升級為正式模型",
        key="cl_promotion_confirmed",
        disabled=not allow_promotion,
    )
    if st.button(
        "升級正式 Arousal 模型",
        disabled=not (allow_promotion and promotion_confirmed),
    ):
        try:
            promoted = service.promote_arousal_challenger(version)
        except Exception as error:
            st.error(f"升級失敗：{error}")
        else:
            st.success(f"正式模型已更新為 {promoted['champion']}。")
            st.json(promoted, expanded=False)


def render_continual_learning_center() -> None:
    st.divider()
    st.header("持續學習中心")
    st.caption(
        "每次上傳都可更新病患分析；只有完成真實標籤／治療結果確認的資料，"
        "才會進入版本化模型學習。"
    )
    service = get_learning_service()
    _render_apnea60_status()
    st.divider()
    st.markdown("#### Arousal 與治療效果模型")
    _render_status(service)
    _render_treatment_learning_status()
    arousal_tab, treatment_tab, model_tab = st.tabs(
        ["Arousal 回饋", "治療結果", "模型管理"]
    )
    with arousal_tab:
        _render_arousal_feedback(service)
    with treatment_tab:
        _render_treatment_outcomes(service)
    with model_tab:
        _render_model_management(service)
