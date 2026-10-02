from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from osa_next_epoch_ui import render_osa_next_epoch_model


# Keep optional, heavyweight PSG and wearable modules out of app startup.
# Streamlit Cloud hot-reloads code while a previous session may still be
# finishing; importing these modules lazily prevents partial-import errors.
def build_uploaded_data_review(*args, **kwargs):
    from data_processing_review import build_uploaded_data_review as implementation
    return implementation(*args, **kwargs)


def build_processed_data_review(*args, **kwargs):
    from data_processing_review import build_processed_data_review as implementation
    return implementation(*args, **kwargs)


def build_training_dataset_overview(*args, **kwargs):
    from data_processing_review import build_training_dataset_overview as implementation
    return implementation(*args, **kwargs)


def build_all_patient_processing_index(*args, **kwargs):
    from data_processing_review import build_all_patient_processing_index as implementation
    return implementation(*args, **kwargs)


def render_wearable_realtime_center(default_patient_id: str = ""):
    if not default_patient_id:
        return None
    from wearable_realtime_ui import render_wearable_realtime_center as implementation
    return implementation(default_patient_id)


# Clinical-data tools are optional after upload.  Lazy imports keep a partial
# Streamlit Cloud hot reload from taking down the complete PSG pipeline before
# a user opens the clinical-data section.
def get_parser(*args, **kwargs):
    from clinical_modules.registry import get_parser as implementation
    return implementation(*args, **kwargs)


def has_parser(*args, **kwargs):
    from clinical_modules.registry import has_parser as implementation
    return implementation(*args, **kwargs)


def parse_clinical_update(*args, **kwargs):
    from clinical_modules.update_parser import parse_clinical_update as implementation
    return implementation(*args, **kwargs)


def refresh_patient(*args, **kwargs):
    from clinical_modules.update_service import refresh_patient as implementation
    return implementation(*args, **kwargs)


def refresh_patient_batch(*args, **kwargs):
    from clinical_modules.update_service import refresh_patient_batch as implementation
    return implementation(*args, **kwargs)


def prepare_psg_follow_up(*args, **kwargs):
    from clinical_modules.psg_follow_up_service import prepare_psg_follow_up as implementation
    return implementation(*args, **kwargs)


def finalize_psg_follow_up(*args, **kwargs):
    from clinical_modules.psg_follow_up_service import finalize_psg_follow_up as implementation
    return implementation(*args, **kwargs)


def rollback_psg_follow_up(*args, **kwargs):
    from clinical_modules.psg_follow_up_service import rollback_psg_follow_up as implementation
    return implementation(*args, **kwargs)


def validate_follow_up_files(*args, **kwargs):
    from clinical_modules.psg_follow_up_service import validate_follow_up_files as implementation
    return implementation(*args, **kwargs)


def get_module_spec(*args, **kwargs):
    from clinical_modules.parser_schemas import get_module_spec as implementation
    return implementation(*args, **kwargs)


def get_module_key_map() -> dict[str, str]:
    from clinical_modules.update_merger import MODULE_KEY_MAP
    return dict(MODULE_KEY_MAP)


# ============================================================
# 專案路徑
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
# Community Cloud stores the previous visitor's files in the same short-lived
# container.  Never restore that shared pointer into a new browser session:
# doing so makes every visitor re-read a large EDF before they can upload.
IS_CLOUD_RUNTIME = PROJECT_ROOT.as_posix().startswith("/mount/src/")

INCOMING_ROOT = PROJECT_ROOT / "data" / "incoming"
INFERENCE_ROOT = PROJECT_ROOT / "data" / "inference"
ACTIVE_PATIENT_STATE_FILE = PROJECT_ROOT / "data" / "active_patient_state.json"

PIPELINE_SCRIPT = PROJECT_ROOT / "run_patient_pipeline.py"
TREATMENT_REFINEMENT_SCRIPT = (
    PROJECT_ROOT / "build_treatment_refinement.py"
)

REPORT_FILENAMES = {
    "html": "patient_digital_twin_report.html",
    "json": "patient_digital_twin_report.json",
    "csv": "patient_digital_twin_report.csv",
    "txt": "patient_digital_twin_report.txt",
}
TREATMENT_INTERFACE_FILENAME = (
    "treatment_refinement_interface.json"
)
CLINICAL_DATA_TYPES = {
    "PSG_FOLLOW_UP": "最新 PSG／睡眠檢查（EDF、Stage、Event）",
    "PAP": "PAP 治療成效與耐受度（壓力、漏氣、殘餘 AHI、依從性）",
    "FOLLOW_UP": "治療回診與療效（症狀、殘餘 AHI、副作用、耐受度）",
    "DEVICE_DATA": "穿戴裝置／居家監測（SpO₂、心率、睡眠、呼吸事件）",
    "ENT": "鼻內視鏡／耳鼻喉檢查（鼻阻塞、扁桃腺分級、軟顎）",
    "DISE": "DISE 睡眠內視鏡（軟顎、側咽壁、舌根、會厭塌陷）",
    "IMAGING": "上呼吸道影像（CT／CBCT／MRI：結構狹窄）",
    "DENTAL_CRANIOFACIAL": "顎面／牙科評估（下顎後縮、小顎症、咬合）",
    "SURGERY_HISTORY": "上呼吸道手術史與術後效果",
    "MEDICATION": "目前用藥與失眠診斷（安眠藥、鴉片類、呼吸抑制藥）",
    "SLEEP_QUESTIONNAIRE": "睡眠症狀與問卷（ESS、ISI、PSQI、STOP-BANG）",
    "PSYCHIATRIC": "心理、精神狀態與睡眠相關診斷",
    "PATIENT_PREFERENCE": "患者治療偏好、接受度與共同決策",
    "HYPOXEMIA": "低血氧原因評估（白天／活動 SpO₂、是否需氧）",
    "ABG": "血液氣體與低通氣（PaCO₂、PaO₂、HCO₃⁻）",
    "PULMONARY": "肺功能、呼吸系統疾病與低通氣風險",
    "CARDIAC": "心臟與心血管共病評估",
    "ECG": "心電圖／心律不整",
    "ECHOCARDIOGRAPHY": "心臟超音波／心臟功能",
    "COMORBIDITY": "重要共病（心肺、代謝、腎病、肥胖）",
    "OXYGEN_THERAPY": "氧氣治療需求與治療反應",
    "ANTHROPOMETRY": "身體測量（BMI、頸圍、腰圍）",
    "WEIGHT_MANAGEMENT": "體重變化與減重治療紀錄",
    "LIFESTYLE": "生活型態（酒精、吸菸、睡眠作息、活動量）",
    "LABORATORY": "相關實驗室檢查",
    "NEUROLOGICAL": "神經學病史（中風、神經肌肉疾病）",
    "OTHER": "其他經醫師確認的臨床資訊",
}

CLINICAL_DATA_GROUPS = {
    "核心睡眠、PAP 與追蹤": [
        "PAP", "FOLLOW_UP", "DEVICE_DATA",
    ],
    "上呼吸道與手術評估": [
        "ENT", "DISE", "IMAGING", "DENTAL_CRANIOFACIAL",
        "SURGERY_HISTORY",
    ],
    "用藥、失眠、症狀與偏好": [
        "MEDICATION", "SLEEP_QUESTIONNAIRE", "PSYCHIATRIC",
        "PATIENT_PREFERENCE",
    ],
    "低氧、心肺與重要共病": [
        "HYPOXEMIA", "ABG", "PULMONARY", "CARDIAC", "ECG",
        "ECHOCARDIOGRAPHY", "COMORBIDITY", "OXYGEN_THERAPY",
    ],
    "身體、生活、神經與其他": [
        "ANTHROPOMETRY", "WEIGHT_MANAGEMENT", "LIFESTYLE",
        "LABORATORY", "NEUROLOGICAL", "OTHER",
    ],
}


# ============================================================
# Streamlit 頁面設定
# ============================================================

st.set_page_config(
    page_title="Sleep Digital Twin",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 工具函式
# ============================================================

def sanitize_filename(filename: str) -> str:
    """
    清理上傳檔案名稱，避免 Windows 不允許的字元。
    """
    safe_name = Path(filename).name.strip()

    safe_name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        safe_name,
    )

    safe_name = safe_name.rstrip(". ")

    if not safe_name:
        raise ValueError("上傳檔案名稱為空白。")

    return safe_name


def sanitize_patient_id(patient_id: str) -> str:
    """
    清理 Patient ID，但保留空白與連字號。

    範例：
    20201014T221256 - d25c6
    """
    patient_id = patient_id.strip()

    patient_id = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        patient_id,
    )

    patient_id = re.sub(
        r"\s+",
        " ",
        patient_id,
    )

    patient_id = patient_id.rstrip(". ")

    if not patient_id:
        raise ValueError("Patient ID 為空白。")

    return patient_id


def extract_patient_id_from_edf(filename: str) -> str:
    """
    從 EDF 檔名擷取 Patient ID。

    支援：

    20201014T221256 - d25c6_EDF.edf
    20201014T221256 - d25c6-EDF.edf
    20201014T221256 - d25c6 EDF.edf
    20201014T221256 - d25c6.edf

    結果：

    20201014T221256 - d25c6
    """
    original_name = Path(filename).name
    stem = Path(original_name).stem.strip()

    if not stem:
        raise ValueError(
            "EDF 檔名無效，無法取得 Patient ID。"
        )

    patient_id = re.sub(
        r"(?i)(?:[\s_-]+EDF)$",
        "",
        stem,
    ).strip()

    patient_id = sanitize_patient_id(patient_id)

    if not patient_id:
        raise ValueError(
            f"無法從 EDF 檔名取得 Patient ID：{original_name}"
        )

    return patient_id


def extract_patient_id_from_companion(filename: str) -> str | None:
    """Extract a patient prefix from a Stage/Event filename when present."""
    stem = Path(filename).stem.strip()
    match = re.match(r"^(\d{8}T\d{6}\s*-\s*[A-Za-z0-9]+)", stem)
    return sanitize_patient_id(match.group(1)) if match else None


def save_uploaded_file(
    uploaded_file: BinaryIO,
    destination: Path,
) -> None:
    """
    將 Streamlit 上傳檔案寫入磁碟。
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{datetime.now().strftime('%H%M%S%f')}.part"
    )
    try:
        uploaded_file.seek(0)
        with temporary.open("xb") as output_file:
            shutil.copyfileobj(uploaded_file, output_file, length=1024 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass


class DownloadedEDFFile:
    """Small file-like adapter so a streamed Drive download uses the normal save path."""

    def __init__(self, path: Path, original_name: str) -> None:
        self.name = original_name
        self._handle = path.open("rb")

    def seek(self, *args, **kwargs):
        return self._handle.seek(*args, **kwargs)

    def read(self, *args, **kwargs):
        return self._handle.read(*args, **kwargs)

    def close(self) -> None:
        self._handle.close()


def extract_google_drive_file_id(share_url: str) -> str:
    """Accept only Google Drive URLs; never fetch an arbitrary remote URL."""
    parsed = urlparse(share_url.strip())
    host = parsed.netloc.lower()
    allowed_hosts = {
        "drive.google.com",
        "www.drive.google.com",
        "drive.usercontent.google.com",
    }
    if parsed.scheme != "https" or host not in allowed_hosts:
        raise ValueError("只接受 Google Drive 的 HTTPS 分享連結。")

    match = re.search(r"/file/d/([A-Za-z0-9_-]+)", parsed.path)
    if match:
        return match.group(1)
    file_id = parse_qs(parsed.query).get("id", [""])[0].strip()
    if re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
        return file_id
    raise ValueError("無法從 Google Drive 連結讀取檔案 ID。請使用「取得連結」產生的分享網址。")


def download_google_drive_edf(share_url: str, original_name: str) -> Path:
    """Stream an explicitly shared EDF from Drive to a private temporary file."""
    if Path(original_name).suffix.lower() != ".edf":
        raise ValueError("請填入原始 EDF 檔名，且副檔名必須是 .edf。")

    file_id = extract_google_drive_file_id(share_url)
    download_url = (
        "https://drive.usercontent.google.com/download?"
        f"id={file_id}&export=download&confirm=t"
    )
    temp_dir = PROJECT_ROOT / "data" / "remote_downloads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temporary = temp_dir / (
        f".{sanitize_filename(original_name)}.{os.getpid()}."
        f"{datetime.now().strftime('%Y%m%dT%H%M%S%f')}.part"
    )

    try:
        request = Request(download_url, headers={"User-Agent": "SleepDigitalTwin/1.0"})
        with urlopen(request, timeout=60) as response, temporary.open("xb") as output:
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "text/html" in content_type:
                raise ValueError(
                    "Google Drive 未提供檔案下載。請確認分享權限設為「知道連結的使用者可檢視」。"
                )
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > 2 * 1024 * 1024 * 1024:
                raise ValueError("EDF 檔案超過 2 GB，請改用本機或正式雲端儲存流程。")
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if temporary.stat().st_size == 0:
            raise ValueError("Google Drive 下載到空白檔案，請確認分享連結與權限。")
        final_path = temporary.with_suffix(".edf")
        os.replace(temporary, final_path)
        return final_path
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def remove_folder(folder: Path) -> None:
    """
    刪除指定資料夾。
    """
    if folder.exists():
        shutil.rmtree(folder)


def clear_analysis_session() -> None:
    """
    清除上一位患者在 Streamlit Session 中留下的分析狀態。
    """
    keys_to_remove = [
        "saved_patient_id",
        "saved_patient_folder",
        "saved_files",
        "analysis_completed",
        "analysis_failed",
        "pipeline_return_code",
        "pipeline_log",
        "report_paths",
        "selected_clinical_data_type",
        "clinical_files_ready",
        "clinical_data_parsed",
        "clinical_data_confirmed",
        "parsed_clinical_data",
        "parsed_clinical_source_files",
        "parsed_clinical_warnings",
        "parsed_clinical_metadata",
        "reevaluation_result",
        "reevaluation_error",
        "psg_follow_up_result",
        "psg_follow_up_error",
        "pipeline_autorun_patient_id",
        "data_processing_review_confirmed_patient_id",
    ]

    for key in keys_to_remove:
        st.session_state.pop(key, None)


def save_patient_uploads(
    patient_id: str,
    edf_file,
    stage_file,
    event_file,
    overwrite: bool,
) -> tuple[Path, dict[str, Path]]:
    """
    儲存三個患者上傳檔案。

    回傳：
    - 患者資料夾
    - 三個檔案的完整路徑
    """
    patient_folder = INCOMING_ROOT / patient_id
    folder_already_existed = patient_folder.exists()

    if folder_already_existed:
        if not overwrite:
            raise FileExistsError(
                f"患者資料夾已存在：{patient_folder}"
            )

    # Windows 可能在分析後繼續鎖住大型 EDF。覆蓋時不先刪除整個患者
    # 資料夾，而是保存成新的版本檔；後續 Pipeline 會使用本次回傳路徑。
    patient_folder.mkdir(parents=True, exist_ok=True)
    version_token = datetime.now().strftime("%Y%m%dT%H%M%S%f")

    def versioned_name(original_name: str) -> str:
        safe_name = sanitize_filename(original_name)
        if not folder_already_existed:
            return safe_name
        source = Path(safe_name)
        return f"{source.stem}__upload_{version_token}{source.suffix}"

    filenames = {
        "EDF": versioned_name(edf_file.name),
        "Stage": versioned_name(stage_file.name),
        "Event Grid": versioned_name(event_file.name),
    }

    file_paths = {
        label: patient_folder / filename
        for label, filename in filenames.items()
    }

    try:
        save_uploaded_file(
            uploaded_file=edf_file,
            destination=file_paths["EDF"],
        )

        save_uploaded_file(
            uploaded_file=stage_file,
            destination=file_paths["Stage"],
        )

        save_uploaded_file(
            uploaded_file=event_file,
            destination=file_paths["Event Grid"],
        )

    except Exception:
        # 只清理由本次嘗試建立的檔案，不遞迴刪除可能含鎖定 EDF 的舊資料。
        for path in file_paths.values():
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                pass
        if not folder_already_existed:
            try:
                patient_folder.rmdir()
            except OSError:
                pass
        raise

    return patient_folder, file_paths


def _uploaded_review_cache_key(saved_files: dict[str, str]) -> tuple[tuple[str, str, int, int], ...]:
    """Return a content-sensitive, bounded cache key for the upload preview.

    Opening an EDF repeatedly is expensive on Streamlit Community Cloud.  The
    file metadata makes a replacement upload invalidate the preview without
    retaining a separate cache entry for every rerun.
    """
    entries: list[tuple[str, str, int, int]] = []
    for label, value in sorted(saved_files.items()):
        path = Path(value)
        try:
            stat = path.stat()
            entries.append((label, str(path), int(stat.st_mtime_ns), int(stat.st_size)))
        except OSError:
            entries.append((label, str(path), -1, -1))
    return tuple(entries)


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def _load_uploaded_data_review_cached(
    cache_key: tuple[tuple[str, str, int, int], ...],
) -> dict[str, Any]:
    """Parse a saved upload once per file version rather than once per rerun."""
    paths = {label: path for label, path, _, _ in cache_key}
    return build_uploaded_data_review(paths)


def show_data_processing_review(patient_id: str, saved_files: dict[str, str]) -> bool:
    """Display raw and post-pipeline data lineage in clinician-readable terms."""
    st.markdown('<span class="dt-step">步驟 3 · 資料處理與醫師檢視</span>', unsafe_allow_html=True)
    st.header("資料處理摘要（送入模型前先檢視）")
    st.info(
        "這一頁讓您先確認原始PSG資料、睡眠分期、呼吸事件與品質檢查。"
        "確認後才會啟動模型分析；系統不會把單次新患者PSG直接當成治療效果訓練答案。"
    )
    review = _load_uploaded_data_review_cached(
        _uploaded_review_cache_key(saved_files)
    )
    st.subheader("A. 原始檔案與用途")
    st.dataframe(pd.DataFrame(review.get("files", [])), hide_index=True, width="stretch")

    edf = review.get("edf", {})
    sleep = review.get("sleep", {})
    events = review.get("events", {})
    if edf and sleep:
        cols = st.columns(5)
        cols[0].metric("EDF訊號通道", edf.get("channel_count", "—"))
        cols[1].metric("取樣率", f"{edf.get('sampling_rate_hz', '—')} Hz")
        cols[2].metric("EDF時長", f"{edf.get('duration_minutes', '—')} 分")
        cols[3].metric("睡眠分期epoch", sleep.get("epoch_count", "—"))
        cols[4].metric("估計睡眠時間", f"{sleep.get('estimated_sleep_minutes', '—')} 分")
        with st.expander("查看EDF可用訊號通道", expanded=False):
            st.write(", ".join(edf.get("channels", [])) or "未讀到通道資訊")

    st.subheader("B. 轉成人可讀的睡眠與呼吸摘要")
    left, right = st.columns(2)
    with left:
        st.caption("睡眠分期：每個epoch為30秒；W=清醒，N1/N2/N3=非REM，REM=快速動眼期。")
        stage_counts = review.get("stage_counts", {})
        stage_frame = pd.DataFrame([
            {"睡眠期": key, "epoch數": value, "分鐘": round(value * 0.5, 1)}
            for key, value in stage_counts.items()
        ])
        st.dataframe(stage_frame, hide_index=True, width="stretch")
    with right:
        st.caption("事件由Event Grid技師標記；OSA-only事件=阻塞型apnea與hypopnea。")
        event_counts = review.get("event_counts", {})
        event_frame = pd.DataFrame([
            {"事件類型": key, "事件數": value}
            for key, value in event_counts.items()
        ])
        st.dataframe(event_frame, hide_index=True, width="stretch")
        if events:
            st.caption(
                f"OSA-only標籤事件：{events.get('osa_label_events', 0)}；"
                f"中央／混合型事件：{events.get('central_or_mixed', 0)}（不納入OSA-only目標）。"
            )

    alignment = review.get("alignment", {})
    st.subheader("C. 品質與時間對齊檢查")
    if alignment:
        st.success(
            f"狀態：{alignment.get('status', '—')}。{alignment.get('explanation', '')}"
        )
        st.caption(
            f"EDF與Stage重疊：{alignment.get('overlap_minutes', '—')}分鐘；"
            f"理論最大Stage覆蓋率：{alignment.get('maximum_stage_coverage', '—')}%。"
        )
    for warning in review.get("warnings", []):
        st.warning(warning)

    st.subheader("D. 這份資料將怎麼進入系統")
    st.markdown(
        "- **正式患者分析：**清理檔名與患者ID、讀取EDF、解析每30秒睡眠分期與呼吸事件、進行時間對齊，"
        "再產生呼吸／睡眠結構／姿勢等特徵，供本次Digital Twin與治療評估使用。\n"
        "- **下一epoch OSA研究模型：**只有對齊成功且Event Grid完整的資料，才會以『前7個30秒epoch→下一個30秒是否為阻塞型apnea或hypopnea』建立研究序列。\n"
        "- **治療推薦訓練：**不會因單次PSG自動訓練。只有已確認治療方式、治療前後結果及醫師確認的資料，才可作為治療效果標籤。"
    )
    st.subheader("E. 目前逐epoch研究模型的訓練資料庫總覽")
    corpus, corpus_meta = build_training_dataset_overview(PROJECT_ROOT)
    if not corpus.empty:
        st.dataframe(corpus, hide_index=True, width="stretch")
        st.caption(
            f"候選PSG：{corpus_meta.get('candidate_studies', 0)}份；"
            f"成功轉換：{corpus_meta.get('completed_studies', 0)}份；"
            f"未通過品質／檔案檢查：{corpus_meta.get('failed_studies', 0)}份。"
        )
        st.caption(
            f"目前版本：{corpus_meta.get('model_version', '尚未訓練')}；"
            f"患者數：{corpus_meta.get('patient_count', '—')}；"
            f"可用序列：{corpus_meta.get('sequence_count', '—')}。"
        )
    show_all_model_data_review()
    return st.button(
        "我已檢視資料處理摘要，開始模型分析",
        type="primary",
        key="confirm_data_review_and_run_pipeline",
        width="stretch",
    )


def show_processed_data_review(patient_id: str) -> None:
    """Show the actual aligned data lineage after processing completes."""
    review = build_processed_data_review(PROJECT_ROOT, patient_id)
    if not review.get("available"):
        return
    with st.expander("查看本次實際處理後資料與模型使用紀錄", expanded=False):
        st.success(
            f"已對齊睡眠分期：{review.get('aligned_epochs', 0)}個30秒epoch；"
            f"已產生模型特徵：{review.get('feature_epochs', 0)}個epoch。"
        )
        left, right = st.columns(2)
        with left:
            st.caption("處理後睡眠分期")
            st.dataframe(pd.DataFrame([
                {"睡眠期": key, "epoch數": value} for key, value in review.get("stage_counts", {}).items()
            ]), hide_index=True, width="stretch")
        with right:
            st.caption("處理後對齊事件")
            st.dataframe(pd.DataFrame([
                {"事件類型": key, "事件數": value} for key, value in review.get("event_counts", {}).items()
            ]), hide_index=True, width="stretch")
        st.caption("模型特徵欄位：" + "、".join(review.get("feature_columns", [])[:24]))
        st.dataframe(pd.DataFrame(review.get("training_status", [])), hide_index=True, width="stretch")


def show_all_model_data_review() -> None:
    """Allow a meeting participant to inspect all processing records, not only one patient."""
    show_index = st.toggle(
        "顯示模型資料庫的全部患者處理紀錄",
        value=False,
        key="show_all_model_processing_index",
    )
    if not show_index:
        st.caption("需要時再展開全部患者處理紀錄，避免分析期間重複讀取大型資料。")
        return
    with st.container(border=True):
        st.info(
            "此清單顯示每份PSG被處理成多少個30秒epoch、事件統計、是否已有模型特徵，"
            "以及它在下一epoch研究模型中的訓練／驗證／鎖定測試用途。"
        )
        index = build_all_patient_processing_index(PROJECT_ROOT)
        if index.empty:
            st.warning("目前尚無可檢視的處理後PSG資料。")
            return
        st.dataframe(index, hide_index=True, width="stretch", height=360)
        st.download_button(
            "下載全部資料處理清單（CSV）",
            data=index.to_csv(index=False, encoding="utf-8-sig"),
            file_name="psg_data_processing_inventory.csv",
            mime="text/csv",
            key="download_all_processing_inventory",
        )
        selected_id = st.selectbox(
            "選擇一位患者／一份PSG，查看處理後詳細資料",
            options=index["Patient／Study ID"].tolist(),
            key="all_data_review_patient_id",
        )
        detail = build_processed_data_review(PROJECT_ROOT, selected_id)
        if detail.get("available"):
            st.caption(
                f"{selected_id}：對齊{detail.get('aligned_epochs', 0)}個epoch；"
                f"模型特徵{detail.get('feature_epochs', 0)}個epoch。"
            )
            left, right = st.columns(2)
            with left:
                st.dataframe(pd.DataFrame([
                    {"睡眠期": key, "epoch數": value} for key, value in detail.get("stage_counts", {}).items()
                ]), hide_index=True, width="stretch")
            with right:
                st.dataframe(pd.DataFrame([
                    {"事件類型": key, "事件數": value} for key, value in detail.get("event_counts", {}).items()
                ]), hide_index=True, width="stretch")


def find_report_file(
    patient_id: str,
    filename: str,
) -> Path | None:
    """
    在患者 inference 資料夾中遞迴尋找指定報告。

    可同時支援：

    data/inference/<patient_id>/patient_digital_twin_report.html

    或：

    data/inference/<patient_id>/digital_twin_report/
        patient_digital_twin_report.html
    """
    patient_inference_folder = INFERENCE_ROOT / patient_id

    if not patient_inference_folder.exists():
        return None

    direct_path = patient_inference_folder / filename

    if direct_path.is_file():
        return direct_path

    matches = list(
        patient_inference_folder.rglob(filename)
    )

    if not matches:
        return None

    # 如果找到多份，優先取最後修改的檔案
    return max(
        matches,
        key=lambda path: path.stat().st_mtime,
    )


def collect_report_paths(
    patient_id: str,
) -> dict[str, Path]:
    """
    搜尋 Digital Twin Report 的 HTML、JSON、CSV、TXT。
    """
    result: dict[str, Path] = {}

    for extension, filename in REPORT_FILENAMES.items():
        report_path = find_report_file(
            patient_id=patient_id,
            filename=filename,
        )

        if report_path is not None:
            result[extension] = report_path

    return result

def find_treatment_interface_file(
    patient_id: str,
) -> Path | None:
    """
    尋找患者的 Treatment Refinement Interface JSON。

    預期位置：

    data/inference/<patient_id>/treatment_refinement/
        treatment_refinement_interface.json

    同時保留遞迴搜尋，避免未來資料夾結構調整後找不到。
    """
    patient_inference_folder = (
        INFERENCE_ROOT / patient_id
    )

    if not patient_inference_folder.exists():
        return None

    expected_path = (
        patient_inference_folder
        / "treatment_refinement"
        / TREATMENT_INTERFACE_FILENAME
    )

    if expected_path.is_file():
        return expected_path

    matches = list(
        patient_inference_folder.rglob(
            TREATMENT_INTERFACE_FILENAME
        )
    )

    if not matches:
        return None

    return max(
        matches,
        key=lambda path: path.stat().st_mtime,
    )


def load_json_file(
    json_path: Path,
) -> dict:
    """
    安全讀取 UTF-8 JSON 檔案。
    """
    if not json_path.is_file():
        raise FileNotFoundError(
            f"找不到 JSON 檔案：{json_path}"
        )

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        data = json.load(input_file)

    if not isinstance(data, dict):
        raise ValueError(
            "Treatment Refinement Interface "
            "最上層必須是 JSON object。"
        )

    return data


def persist_active_patient(patient_id: str) -> None:
    """Persist the last analysed patient so browser refresh does not lose context."""
    payload = {
        "patient_id": sanitize_patient_id(patient_id),
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    temporary = ACTIVE_PATIENT_STATE_FILE.with_suffix(".json.part")
    ACTIVE_PATIENT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, ACTIVE_PATIENT_STATE_FILE)


def restore_saved_source_files(patient_id: str) -> tuple[Path, dict[str, str]] | None:
    """Restore the most recent complete source triplet for a persisted patient."""
    folder = INCOMING_ROOT / patient_id
    if not folder.is_dir():
        return None
    files = sorted(folder.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)
    edf = next((item for item in files if item.is_file() and item.suffix.lower() == ".edf"), None)
    stage = next((item for item in files if item.is_file() and re.search(r"signal grid|stage", item.name, re.I)), None)
    event = next((item for item in files if item.is_file() and re.search(r"event grid", item.name, re.I)), None)
    if not all((edf, stage, event)):
        return None
    return folder, {"EDF": str(edf), "Stage": str(stage), "Event Grid": str(event)}


def restore_active_patient() -> str | None:
    """Restore a persisted patient only when its latest Interface still exists."""
    try:
        payload = load_json_file(ACTIVE_PATIENT_STATE_FILE)
        patient_id = sanitize_patient_id(str(payload.get("patient_id") or ""))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return patient_id if find_treatment_interface_file(patient_id) is not None else None

def format_score_change(
    score_change: float | int | None,
) -> str:
    """
    將分數變化格式化為 +62、-4 或 0。
    """
    if score_change is None:
        return "0"

    try:
        numeric_value = float(score_change)
    except (TypeError, ValueError):
        return "0"

    if numeric_value > 0:
        return f"+{numeric_value:g}"

    return f"{numeric_value:g}"


def get_suitability_label(
    suitability_level: str | None,
) -> str:
    """
    將英文適配度轉為中文顯示。
    """
    label_map = {
        "HIGH": "較強規則支持",
        "MODERATE": "中等規則支持",
        "LOW": "有限規則支持",
        "VERY_LOW": "目前支持不足",
    }

    if not suitability_level:
        return "未分類"

    return label_map.get(
        suitability_level,
        suitability_level,
    )


def get_confidence_label(
    confidence_level: str | None,
) -> str:
    """
    將英文信心等級轉為中文顯示。
    """
    label_map = {
        "HIGH": "高",
        "MODERATE": "中等",
        "LOW": "低",
    }

    if not confidence_level:
        return "未提供"

    return label_map.get(
        confidence_level,
        confidence_level,
    )


def show_factor_list(
    title: str,
    factors: list,
    factor_type: str,
    show_weights: bool = True,
) -> None:
    """
    顯示支持因素或限制因素。
    """
    st.markdown(f"#### {title}")

    if not factors:
        st.caption("目前沒有資料。")
        return

    for factor in factors:
        if not isinstance(factor, dict):
            continue

        description = factor.get(
            "description",
            "未提供說明",
        )

        weight = factor.get(
            "weight",
            0,
        )

        code = factor.get(
            "code",
            "",
        )

        weight_text = format_score_change(weight)
        prefix = f"**{weight_text} 分**　" if show_weights else ""

        if factor_type == "supporting":
            st.success(f"{prefix}{description}")
        else:
            if float(weight or 0) < 0:
                st.error(f"{prefix}{description}")
            else:
                st.warning(f"{prefix}{description}")

        if code:
            st.caption(f"規則代碼：{code}")


def show_trace_timeline(
    trace_timeline: list,
) -> None:
    """
    顯示 Stage 2 分數調整時間軸。
    """
    st.markdown("#### 分數調整歷程")

    if not trace_timeline:
        st.info(
            "此治療未發生實際分數調整。"
        )
        return

    for trace_item in trace_timeline:
        if not isinstance(trace_item, dict):
            continue

        step = trace_item.get("step")
        module = trace_item.get(
            "module",
            "UNKNOWN",
        )

        before_score = trace_item.get(
            "before_score",
        )

        score_change = trace_item.get(
            "score_change",
            0,
        )

        after_score = trace_item.get(
            "after_score",
        )

        note = trace_item.get(
            "note",
            "",
        )

        st.markdown(
            f"""
**步驟 {step}｜{module}**

`{before_score:g}` → **{format_score_change(score_change)}** → `{after_score:g}`
"""
        )

        if note:
            st.caption(note)


def get_treatment_evaluation_profile(treatment_id: str) -> tuple[str, str]:
    """Return the outcome target and evidence boundary for each pathway.

    The available datasets do not provide a common post-treatment outcome for
    every option.  Showing one numerical score across them would imply a
    comparison that the data cannot support.
    """
    profiles = {
        "SURGERY": (
            "預測目標：手術前後 PSG 的 AHI 改善",
            "證據：具治療前後 PSG 的手術資料；僅供手術專科評估，不選定術式。",
        ),
        "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW": (
            "預測目標：試驗介入相對安慰劑的群體 AHI 改善訊號",
            "證據：未指定藥名的隨機試驗；不可解讀為特定藥物的個人療效或處方。",
        ),
        "CPAP": (
            "預測目標：PAP 適用性與後續依從性風險",
            "證據：PAP 使用、壓力與漏氣追蹤；目前不是 CPAP 後 AHI 改善模型。",
        ),
        "APAP": (
            "預測目標：PAP 適用性與後續依從性風險",
            "證據：PAP 使用、壓力與漏氣追蹤；目前不是 APAP 後 AHI 改善模型。",
        ),
    }
    return profiles.get(
        treatment_id,
        ("預測目標：專科評估適用性", "證據：目前資料可觀察到的 PSG 與臨床條件。"),
    )


def show_treatment_card(
    treatment: dict,
) -> None:
    """
    顯示單一治療候選的完整資料。
    """
    treatment_id = treatment.get(
        "treatment_id",
        "UNKNOWN",
    )

    treatment_label = treatment.get(
        "treatment_label",
        treatment_id,
    )

    # This index is preserved for longitudinal review of the *same* pathway.
    # It is intentionally not used to rank different treatment mechanisms.
    pathway_index = treatment.get(
        "final_score",
        treatment.get("rule_based_score", 0),
    )
    model_prediction = treatment.get("model_research_prediction", {})
    if not isinstance(model_prediction, dict):
        model_prediction = {}
    predicted_utility = model_prediction.get("predicted_treatment_utility")
    learned_status = str(model_prediction.get("learned_status") or "")
    has_individual_outcome_prediction = (
        predicted_utility is not None
        and learned_status == "confidence_weighted_blend"
    )
    hybrid_score = pathway_index
    hybrid_source = "規則適用性與安全條件（尚無可用個人療效模型）"
    if has_individual_outcome_prediction:
        hybrid_score = model_prediction.get("score", pathway_index)
        hybrid_source = "模型預測與規則安全條件的有限度混合"
    model_validation = treatment.get("model_validation", {})
    if not isinstance(model_validation, dict):
        model_validation = {}

    suitability_level = treatment.get(
        "suitability_level",
    )

    recommendation_status = treatment.get(
        "recommendation_status",
        "UNKNOWN",
    )

    confidence = treatment.get(
        "confidence",
        {},
    )

    if not isinstance(confidence, dict):
        confidence = {}

    confidence_level = confidence.get(
        "level",
    )

    completeness = confidence.get(
        "completeness",
        0,
    )

    try:
        completeness_percentage = (
            float(completeness) * 100
        )
    except (TypeError, ValueError):
        completeness_percentage = 0.0

    target_text, evidence_text = get_treatment_evaluation_profile(treatment_id)
    expander_title = f"{treatment_label}（{treatment_id}）"

    with st.expander(
        expander_title,
        expanded=False,
    ):
        metric_columns = st.columns(3)

        with metric_columns[0]:
            st.metric(
                "目前適用性",
                get_suitability_label(suitability_level),
            )

        with metric_columns[1]:
            st.metric(
                "混合決策支持指數",
                f"{float(hybrid_score or 0):.1f} / 100",
            )

        with metric_columns[2]:
            st.metric(
                "證據信心",
                get_confidence_label(confidence_level),
            )

        st.caption(
            f"介入角色：{treatment.get('evaluation_role', 'OSA治療或專科介入評估')}｜"
            f"評估狀態：{recommendation_status}"
        )

        st.info(target_text)
        st.caption(evidence_text)
        if has_individual_outcome_prediction:
            st.caption(
                f"模型原始治療效益預測：{float(predicted_utility) * 100:.1f} / 100"
            )
            st.caption("模型狀態：已由目前模型推論；仍屬研究用途，尚未完成外部前瞻驗證。")
        else:
            st.caption(
                "此路徑目前沒有可直接對應的個人療效模型輸出；"
                "系統不會把未驗證的探索性數值或規則分數偽裝成模型預測。"
            )
        if model_validation:
            st.caption(
                "模型驗證："
                f"已確認治療後結果 {model_validation.get('confirmed_outcome_rows', 0)} 筆｜"
                f"MAE {model_validation.get('mae', '—')}｜"
                f"R² {model_validation.get('r2', '—')}"
            )
        st.caption(
            f"混合來源：{hybrid_source}｜規則適用性摘要 {float(pathway_index or 0):g}/100。"
            "此指數不是成功率、AHI 改善百分比或跨治療排名。"
        )

        st.markdown("#### 證據完整度")

        safe_completeness = max(
            0.0,
            min(
                float(completeness or 0),
                1.0,
            ),
        )

        st.progress(
            safe_completeness,
            text=(
                f"{completeness_percentage:.1f}%"
            ),
        )

        interpretation = treatment.get(
            "interpretation"
        )

        if interpretation:
            st.info(interpretation)

        factor_columns = st.columns(2)

        with factor_columns[0]:
            show_factor_list(
                title="支持因素",
                factors=treatment.get(
                    "supporting_factors",
                    [],
                ),
                factor_type="supporting",
                show_weights=False,
            )

        with factor_columns[1]:
            show_factor_list(
                title="限制因素",
                factors=treatment.get(
                    "limiting_factors",
                    [],
                ),
                factor_type="limiting",
                show_weights=False,
            )

        required_confirmation = treatment.get(
            "required_confirmation",
            [],
        )

        if required_confirmation:
            st.markdown("#### 尚需確認")

            for item in required_confirmation:
                st.markdown(f"- {item}")

        candidate_procedures = treatment.get(
            "candidate_procedures",
            [],
        )

        if candidate_procedures:
            st.markdown("#### 候選處置")

            for item in candidate_procedures:
                st.markdown(f"- {item}")

        projected_score_changes = treatment.get(
            "projected_score_changes",
            [],
        )

        if projected_score_changes:
            st.markdown("#### 後續評估方向")

            for item in projected_score_changes:
                st.markdown(f"- {item}")

        available_evidence = confidence.get(
            "available_evidence",
            [],
        )

        missing_evidence = confidence.get(
            "missing_evidence",
            [],
        )

        evidence_columns = st.columns(2)

        with evidence_columns[0]:
            st.markdown("#### 已取得證據")

            if available_evidence:
                for item in available_evidence:
                    st.markdown(f"- {item}")
            else:
                st.caption("無")

        with evidence_columns[1]:
            st.markdown("#### 缺少證據")

            if missing_evidence:
                for item in missing_evidence:
                    st.markdown(f"- {item}")
            else:
                st.success("目前沒有缺少證據。")


def show_model_update_audit(patient_id: str) -> None:
    """顯示本次資料是否真的進入訓練、驗證與正式模型。"""
    receipt_path = (
        INFERENCE_ROOT / patient_id / "model_update_audit" / "latest_model_update_receipt.json"
    )
    st.subheader("模型更新稽核")
    if not receipt_path.exists():
        st.info("本次分析尚未產生模型更新收據；請重新執行完整 Pipeline。")
        return
    try:
        receipt = load_json_file(receipt_path)
    except Exception as exc:
        st.error(f"模型更新收據無法讀取：{type(exc).__name__}: {exc}")
        return

    arousal = receipt.get("arousal_30s") or {}
    apnea_wrapper = receipt.get("apnea_60s") or {}
    latest_apnea_receipt = (
        INFERENCE_ROOT
        / patient_id
        / "apnea_next_60s"
        / "continual_learning_receipt.json"
    )
    if latest_apnea_receipt.exists():
        apnea_wrapper = load_json_file(latest_apnea_receipt)
    apnea_ingestion = apnea_wrapper.get("learning_ingestion") or {}
    apnea_update = apnea_wrapper.get("model_update") or {}
    treatment = receipt.get("treatment") or {}

    def row(name: str, ingestion: str, update: dict, fallback_reason: str) -> dict:
        version = update.get("model_version") or update.get("version") or "—"
        trained = bool(update) or ingestion == "challenger_trained"
        champion = bool(update.get("champion_updated", False))
        return {
            "模型": name,
            "資料加入": "是" if ingestion in {"approved", "challenger_trained", "clinical_update_retrained"} else ("沿用去重資料" if ingestion == "retrained_deduplicated_dataset" else "否"),
            "重新訓練": "是" if trained else "否",
            "新版本": str(version),
            "正式模型更新": "是" if champion else "否",
            "說明": fallback_reason,
        }

    rows = [
        row(
            "30 秒 Arousal",
            str(arousal.get("status", "")),
            arousal.get("challenger") or {"model_version": arousal.get("model_version")},
            str(arousal.get("reason", "—")),
        ),
        row(
            "60 秒 Apnea",
            str(apnea_ingestion.get("status", "")),
            apnea_update,
            str(
                apnea_ingestion.get("reason")
                or ("已建立並驗證新 Challenger" if apnea_update else "沒有建立新版本")
            ),
        ),
        row(
            "四種治療效果",
            str(treatment.get("status", "")),
            treatment.get("challenger") or {"model_version": treatment.get("model_version")},
            str(treatment.get("reason", "—")),
        ),
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    research_root = INFERENCE_ROOT / patient_id / "research_preview"
    arousal_preview_path = (
        research_root / "arousal_30s" / "arousal_prediction_summary.json"
    )
    apnea_preview_path = (
        research_root / "apnea_60s" / "apnea_next_60s_summary.json"
    )
    arousal_preview = (
        load_json_file(arousal_preview_path) if arousal_preview_path.exists() else {}
    )
    apnea_preview = (
        load_json_file(apnea_preview_path) if apnea_preview_path.exists() else {}
    )
    if arousal_preview or apnea_preview:
        st.markdown("#### 重新訓練後的短期風險研究輸出")
        st.error(
            "以下使用最新 Challenger，只用來驗證持續學習流程；"
            "尚未完成外部及前瞻性驗證，不可作為醫療警報。",
            icon=":material/science:",
        )
        preview_rows = []
        if arousal_preview:
            preview_rows.append({
                "研究模型": "30 秒 Arousal",
                "版本": arousal_preview.get("model_version", "—"),
                "最高風險": arousal_preview.get("maximum_probability"),
                "平均風險": arousal_preview.get("average_probability"),
                "警示數": arousal_preview.get("alert_count"),
            })
        if apnea_preview:
            preview_rows.append({
                "研究模型": "60 秒 Apnea",
                "版本": apnea_preview.get("model_version", "—"),
                "最高風險": apnea_preview.get("maximum_probability"),
                "平均風險": apnea_preview.get("mean_probability"),
                "警示數": apnea_preview.get("alert_count"),
            })
        st.dataframe(pd.DataFrame(preview_rows), hide_index=True, width="stretch")
    st.caption(
        "資料加入、重新訓練、新版本與正式 Champion 是四個不同階段。"
        "重複資料不會重複加入；沒有真實標籤的資料可更新 Digital Twin 與推論，"
        "但不能偽裝成監督式訓練標籤。"
    )


def show_apnea_60_prediction(patient_id: str) -> None:
    """顯示獨立的未來60秒 apnea 預測與學習版本。"""
    result_folder = (
        INFERENCE_ROOT
        / patient_id
        / "apnea_next_60s"
    )
    summary_path = result_folder / "apnea_next_60s_summary.json"
    prediction_path = result_folder / "apnea_next_60s_predictions.csv"
    receipt_path = result_folder / "continual_learning_receipt.json"

    st.subheader("未來 60 秒睡眠呼吸中止預測")
    st.caption(
        "預測定義：目前30秒 epoch 結束後，未來60秒內是否出現 apnea。"
        "此項不是 Arousal，也不是正式診斷。"
    )

    if not summary_path.exists():
        st.info("本次資料尚未產生 apnea-next-60s 預測。")
        return

    summary = load_json_file(summary_path)
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "最高風險",
        f"{float(summary.get('maximum_probability', 0)):.1%}",
    )
    metric_columns[1].metric(
        "平均風險",
        f"{float(summary.get('mean_probability', 0)):.1%}",
    )
    metric_columns[2].metric(
        "警報 epoch",
        str(summary.get("alert_count", 0)),
    )
    metric_columns[3].metric(
        "模型版本",
        str(summary.get("model_version", "-")),
    )

    st.info(
        "目前採高敏感度警示模式：門檻由病人分組交叉驗證選擇，"
        "優先將 Recall 提高至約 90%。警示代表需要查看原始生理訊號，"
        "不是確診，也不代表一定會發生呼吸中止。"
    )
    if summary.get("distribution_guard_fallback"):
        st.warning(
            "兩階段 XGBoost＋BiLSTM Challenger 在本次資料出現機率分布漂移，"
            "系統已自動回退至既有 Champion，避免產生幾乎全程警報。"
            "此狀態代表候選模型需要更多不同病人的外部資料再驗證。"
        )
    sensor_check_count = int(summary.get("sensor_check_alert_count", 0) or 0)
    if sensor_check_count:
        st.error(
            f"有 {sensor_check_count} 個 epoch 的核心訊號品質不足；"
            "這些時段不會顯示為安全，請檢查鼻氣流、SpO₂、胸腹帶或感測器連線。"
        )

    alert_rate = float(summary.get("alert_rate", 0))
    st.progress(
        max(0.0, min(alert_rate, 1.0)),
        text=f"整晚警報比例：{alert_rate:.1%}",
    )

    if prediction_path.exists():
        predictions = pd.read_csv(prediction_path)
        alert_values = predictions["apnea_next_60s_alert"]
        if alert_values.dtype == object:
            alert_mask = (
                alert_values.astype(str).str.strip().str.lower()
                .isin({"true", "1", "yes"})
            )
        else:
            alert_mask = alert_values.fillna(False).astype(bool)

        alerts = predictions[alert_mask].sort_values(
            "apnea_next_60s_probability",
            ascending=False,
        )
        highest_risk = predictions.sort_values(
            "apnea_next_60s_probability",
            ascending=False,
        ).head(20)

        if alerts.empty:
            st.success(
                "本次 249 個可分析時間點均未超過 "
                f"{float(summary.get('threshold', 0.35)):.0%} 警示門檻。"
                "這代表模型仍有逐時點預測，但沒有達到需要警示的程度。"
            )
        else:
            st.warning(
                f"共有 {len(alerts)} 個時間點超過 "
                f"{float(summary.get('threshold', 0.35)):.0%} 警示門檻。"
            )

        with st.expander("查看風險最高的 20 個時間點", expanded=False):
            st.dataframe(
                highest_risk,
                hide_index=True,
                column_config={
                    "apnea_next_60s_probability": st.column_config.ProgressColumn(
                        "未來 60 秒 apnea 風險",
                        min_value=0.0,
                        max_value=1.0,
                        format="percent",
                    )
                },
            )

    if receipt_path.exists():
        receipt = load_json_file(receipt_path)
        ingestion = receipt.get("learning_ingestion", {})
        update = receipt.get("model_update")
        if ingestion.get("status") == "approved":
            st.success(
                "本次 Event Grid 已產生真實 apnea 標籤並加入持續學習資料。"
            )
        elif ingestion.get("status") == "already_learned":
            st.info("本次檢查先前已納入學習，不會重複加入。")
        else:
            st.warning(
                "本次資料尚未取得可用的 apnea 真實標籤，"
                "已更新病患結果但沒有修改正式模型。"
            )
        if isinstance(update, dict):
            st.caption(
                "模型更新："
                f"{update.get('model_version')}｜"
                f"病患數 {update.get('patient_count')}｜"
                f"ROC-AUC {update.get('metrics', {}).get('roc_auc', '-')}"
            )
            expected_count = int(update.get("expected_patient_count", update.get("patient_count", 0)) or 0)
            included_count = int(update.get("patient_count", 0) or 0)
            excluded = update.get("excluded_patients", [])
            if expected_count and included_count < expected_count:
                st.warning(
                    f"Apnea有效訓練患者為 {included_count}/{expected_count} 位；"
                    "未納入者不會默默消失，原因如下。"
                )
                if isinstance(excluded, list) and excluded:
                    st.dataframe(excluded, hide_index=True)


def _show_clinical_prerequisites_content(
    active_prerequisites: list,
    resolved_prerequisites: list,
) -> None:
    """
    顯示待完成與已完成的臨床前置事項。
    """
    active_column, resolved_column = st.columns(2)

    with active_column:
        st.markdown("### 待處理")

        if not active_prerequisites:
            st.success(
                "目前沒有待處理的臨床前置事項。"
            )

        for prerequisite in active_prerequisites:
            if not isinstance(
                prerequisite,
                dict,
            ):
                continue

            priority = prerequisite.get(
                "priority",
                "-",
            )

            title = prerequisite.get(
                "title",
                "未命名前置事項",
            )

            reason = prerequisite.get(
                "reason",
                "",
            )

            code = prerequisite.get(
                "code",
                "",
            )

            with st.container(border=True):
                st.warning(
                    f"優先順序 {priority}｜{title}"
                )

                if reason:
                    st.write(reason)

                if code:
                    st.caption(
                        f"代碼：{code}"
                    )

                required_information = (
                    prerequisite.get(
                        "required_information",
                        [],
                    )
                )

                if required_information:
                    st.markdown("需要補充：")

                    for item in required_information:
                        st.markdown(f"- {item}")

    with resolved_column:
        st.markdown("### 已完成")

        if not resolved_prerequisites:
            st.info(
                "目前沒有已完成的臨床前置事項。"
            )

        for prerequisite in resolved_prerequisites:
            if not isinstance(
                prerequisite,
                dict,
            ):
                continue

            title = prerequisite.get(
                "title",
                "未命名前置事項",
            )

            resolution_reason = (
                prerequisite.get(
                    "resolution_reason",
                    "",
                )
            )

            resolved_by_module = (
                prerequisite.get(
                    "resolved_by_module",
                    "",
                )
            )

            observed_value = (
                prerequisite.get(
                    "observed_value"
                )
            )

            with st.container(border=True):
                st.success(
                    f"已完成｜{title}"
                )

                if resolution_reason:
                    st.write(
                        resolution_reason
                    )

                if resolved_by_module:
                    st.caption(
                        "完成來源："
                        f"{resolved_by_module}"
                    )

                if isinstance(
                    observed_value,
                    dict,
                ):
                    st.json(
                        observed_value,
                        expanded=False,
                    )


def show_clinical_prerequisites(
    active_prerequisites: list,
    resolved_prerequisites: list,
) -> None:
    """Keep the clinical checklist available without crowding the main view."""
    active_count = len(active_prerequisites) if isinstance(active_prerequisites, list) else 0
    resolved_count = len(resolved_prerequisites) if isinstance(resolved_prerequisites, list) else 0
    with st.expander(
        f"臨床前置事項｜待處理 {active_count}・已完成 {resolved_count}",
        expanded=False,
        icon=":material/clinical_notes:",
    ):
        _show_clinical_prerequisites_content(active_prerequisites, resolved_prerequisites)


def show_treatment_refinement(
    interface_data: dict,
    interface_path: Path,
) -> None:
    """
    顯示完整 Treatment Refinement Interface。
    """
    st.header("5. 個人化治療與專科介入評估")

    # A clinical update writes a new Interface and then triggers a Streamlit
    # rerun. Always reload the persisted file here so this section cannot show
    # the pre-update object that was created earlier in the same app run.
    if interface_path.is_file():
        try:
            persisted_interface = load_json_file(interface_path)
            if isinstance(persisted_interface, dict):
                interface_data = persisted_interface
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    # Old stored outputs must not remain on a superseded scoring definition.
    # Rebuild once on access, then reload the new interface from disk.
    current_engine_version = 16
    try:
        stored_engine_version = int(interface_data.get("engine_version", 0))
    except (TypeError, ValueError):
        stored_engine_version = 0
    if stored_engine_version < current_engine_version:
        patient_for_upgrade = str(interface_data.get("patient_id") or "").strip()
        if patient_for_upgrade:
            upgrade = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(PROJECT_ROOT / "build_treatment_refinement.py"),
                    "--patient-id",
                    patient_for_upgrade,
                    "--skip-training-snapshot",
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if upgrade.returncode == 0 and interface_path.is_file():
                interface_data = load_json_file(interface_path)
                st.toast("已更新患者輸出；治療頁改為依各自預測目標分開顯示", icon=":material/update:")
            else:
                st.error(
                    "舊版治療輸出自動更新失敗；本頁不會將舊分數冒充為新版結果。",
                    icon=":material/error:",
                )

    patient_id = interface_data.get("patient_id", "UNKNOWN")

    # The update pipeline persists the clinical context and Interface before
    # Streamlit finishes the current rerun. Rebuild this display block from the
    # patient's latest disk state, rather than trusting a stale in-memory dict.
    persisted_clinical_path = (
        INFERENCE_ROOT / str(patient_id) / "treatment_refinement" /
        "clinical_decision_data.json"
    )
    if persisted_clinical_path.is_file():
        try:
            persisted_clinical = load_json_file(persisted_clinical_path)
            feature_rows = []
            modules_received = []
            ignored_top_level = {
                "schema_version", "patient_id", "data_version",
                "last_updated_at", "last_updated_module",
            }
            for module_key, module_payload in persisted_clinical.items():
                if module_key in ignored_top_level or not isinstance(module_payload, dict):
                    continue
                findings = module_payload.get("findings", {})
                if not isinstance(findings, dict):
                    continue
                non_empty = {
                    key: value for key, value in findings.items()
                    if value not in (None, "", [], {})
                }
                if not non_empty:
                    continue
                modules_received.append(module_key)
                feature_rows.extend({
                    "module": module_key,
                    "source_field": key,
                    "model_feature": f"clinical_{module_key}_{key}",
                    "value": value,
                    "status": "included_in_treatment_context",
                } for key, value in non_empty.items())
            interface_data["clinical_feature_ingestion"] = {
                "status": "connected_to_treatment_context",
                "data_version": persisted_clinical.get("data_version", 0),
                "last_updated_module": persisted_clinical.get("last_updated_module"),
                "feature_count": len(feature_rows),
                "modules_received": modules_received,
                "features": feature_rows,
                "all_non_empty_fields_retained": True,
                "display_source": "persisted_clinical_decision_data",
            }
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    schema_version = interface_data.get("schema_version", "-")

    summary = interface_data.get(
        "summary",
        {},
    )

    if not isinstance(summary, dict):
        summary = {}

    st.info(
        "此頁將 CPAP／APAP、手術與藥物研究分開呈現。"
        "它們使用的真實世界結果標記不同，因此不顯示共同療效分數或跨治療名次。",
        icon=":material/info:",
    )

    recommendation_summary = (
        interface_data.get(
            "recommendation_summary",
            {},
        )
    )

    if not isinstance(
        recommendation_summary,
        dict,
    ):
        recommendation_summary = {}

    header_columns = st.columns(3)

    with header_columns[0]:
        st.metric(
            "Patient ID",
            patient_id,
        )

    with header_columns[1]:
        st.metric(
            "治療路徑數",
            summary.get("treatment_count", 0),
        )

    with header_columns[2]:
        st.metric(
            "跨治療比較",
            "不顯示共同分數",
        )

    st.caption(
        f"Interface schema version：{schema_version}"
    )

    ingestion = interface_data.get(
        "clinical_feature_ingestion",
        {},
    )
    if isinstance(ingestion, dict):
        feature_count = int(ingestion.get("feature_count", 0) or 0)
        modules_received = ingestion.get("modules_received", [])
        with st.expander(
            f"目前患者補充資料（{patient_id}｜累積 {feature_count} 個特徵）",
            expanded=feature_count > 0,
        ):
            st.write(
                "以下是本次 Digital Twin 治療 context 已接收的醫師補充資料。"
                "即使某欄位本次分數影響為 0，也會保留在訓練快照並參與新版模型訓練。"
            )
            st.caption(
                "已接入模組："
                + (", ".join(str(item) for item in modules_received) or "尚無")
            )
            feature_rows = ingestion.get("features", [])
            if isinstance(feature_rows, list) and feature_rows:
                st.dataframe(
                    pd.DataFrame(feature_rows),
                    hide_index=True,
                )
            elif feature_count == 0:
                st.info(
                    f"患者 {patient_id} 目前尚未收到成功解析且確認的補充欄位。"
                    "這不代表全域治療模型或其他患者資料歸零。"
                )

            registry_path = PROJECT_ROOT / "models" / "registry" / "treatment" / "registry.json"
            snapshot_dir = PROJECT_ROOT / "data" / "continual_learning" / "treatment_snapshots"
            registry = load_json_file(registry_path) if registry_path.is_file() else {}
            snapshot_files = list(snapshot_dir.glob("*.csv")) if snapshot_dir.is_dir() else []
            training_rows = 0
            training_patients: set[str] = set()
            for snapshot_path in snapshot_files:
                try:
                    snapshot_frame = pd.read_csv(snapshot_path)
                except (OSError, ValueError, pd.errors.ParserError):
                    continue
                training_rows += len(snapshot_frame)
                if "patient_id" in snapshot_frame.columns:
                    training_patients.update(
                        snapshot_frame["patient_id"].dropna().astype(str)
                    )
            st.markdown("**全域持續學習模型（跨患者累積，不因切換患者歸零）**")
            global_columns = st.columns(4)
            global_columns[0].metric("累積訓練列", training_rows)
            global_columns[1].metric("累積患者", len(training_patients))
            global_columns[2].metric("歷史模型版本", len(registry.get("models", [])))
            global_columns[3].metric(
                "最新模型",
                str(registry.get("latest_challenger") or "尚無").replace("treatment_adaptive_", "…"),
            )
            st.caption(
                "目前患者特徵屬於個人Digital Twin；全域訓練列與模型版本才是所有患者的持續學習累積。"
            )

    learning_snapshot = interface_data.get(
        "treatment_learning_snapshot",
        {},
    )
    if isinstance(learning_snapshot, dict) and learning_snapshot:
        update = learning_snapshot.get("model_update", {})
        if isinstance(update, dict) and update.get("model_version"):
            st.success(
                "本次補充資料已建立治療學習快照並完成模型更新："
                f"{update.get('model_version')}"
            )

    first_confidence = (
        recommendation_summary.get(
            "first_candidate_confidence"
        )
    )

    active_count = (
        recommendation_summary.get(
            "active_prerequisite_count",
            0,
        )
    )

    summary_columns = st.columns(3)

    with summary_columns[0]:
        st.metric(
            "比較原則",
            "依各自結果標記",
        )

    with summary_columns[1]:
        st.metric(
            "第一候選信心",
            get_confidence_label(
                first_confidence
            ),
        )

    with summary_columns[2]:
        st.metric(
            "待處理前置事項",
            active_count,
        )

    summary_interpretation = (
        recommendation_summary.get(
            "interpretation"
        )
    )

    if summary_interpretation:
        st.info(summary_interpretation)

    treatments = interface_data.get(
        "treatments",
        [],
    )

    if not isinstance(treatments, list):
        treatments = []

    st.subheader("治療與專科介入：分開的預測目標與適用性")

    st.success(
        "CPAP／APAP 顯示 PAP 適用性與依從性風險；"
        "手術顯示治療前後 PSG 的 AHI 改善模型；"
        "藥物顯示試驗介入相對安慰劑的群體研究訊號。"
        "它們不再被當成同一種療效分數。",
        icon=":material/verified:",
    )

    st.warning(
        "不會因資料量多就假裝四種治療已有相同的療效標籤。"
        "每張卡都會顯示模型／研究實際預測的目標、可用證據與尚須確認項目；"
        "結果僅供醫師討論，不可自動取代診斷與處方。",
        icon=":material/clinical_notes:",
    )

    if not treatments:
        st.warning(
            "Treatment Refinement Interface "
            "沒有治療候選資料。"
        )

    # Keep the research model output separate from the clinical-rule index.
    # This lets clinicians inspect what the trained model actually produced
    # without treating incomparable treatment targets as one leaderboard.
    research_preview = interface_data.get("research_treatment_preview", {})
    research_ranking = (
        research_preview.get("ranking", [])
        if isinstance(research_preview, dict)
        else []
    )
    research_by_treatment = {
        str(item.get("treatment")): item
        for item in research_ranking
        if isinstance(item, dict) and item.get("treatment")
    }
    model_info = (
        research_preview.get("model", {})
        if isinstance(research_preview, dict)
        else {}
    )
    model_selection = (
        model_info.get("model_selection", {})
        if isinstance(model_info, dict)
        else {}
    )

    prepared_treatments = []
    for treatment in treatments:
        if not isinstance(treatment, dict):
            continue
        current = dict(treatment)
        treatment_id = str(
            current.get("treatment_id") or current.get("treatment") or ""
        )
        research_prediction = research_by_treatment.get(treatment_id, {})
        current["model_research_prediction"] = research_prediction
        current["model_validation"] = (
            model_selection.get(treatment_id, {})
            if isinstance(model_selection, dict)
            else {}
        )
        model_score = research_prediction.get("score") if isinstance(research_prediction, dict) else None
        model_is_usable = (
            isinstance(research_prediction, dict)
            and research_prediction.get("learned_status") == "confidence_weighted_blend"
            and model_score is not None
        )
        current["display_hybrid_score"] = float(
            model_score if model_is_usable else current.get("final_score", 0)
        )
        prepared_treatments.append(current)

    sorted_treatments = sorted(
        prepared_treatments,
        key=lambda treatment: -float(treatment.get("display_hybrid_score", 0)),
    )

    st.caption(
        "下列卡片依畫面顯示的混合決策支持指數由高至低排列；"
        "它是評估順序，不代表四種治療具有可直接比較的療效機率。"
    )

    for treatment in sorted_treatments:
        show_treatment_card(treatment)

    if research_ranking:
        st.subheader("Challenger 研究模型驗證（不提供病人排名）")
        st.warning(
            "此區只用於檢查重新訓練與模型驗證流程。"
            "不會對目前患者輸出共同分數、排名或處方建議。",
            icon=":material/science:",
        )
        st.error(
            research_preview.get(
                "warning",
                "研究測試輸出，不可用於臨床決策。",
            ),
            icon=":material/science:",
        )
        research_rows = []
        for item in sorted(
            (row for row in research_ranking if isinstance(row, dict)),
            key=lambda row: int(row.get("rank", 999)),
        ):
            research_rows.append(
                {
                    "治療方式": item.get("treatment_label", item.get("treatment")),
                    "病人端使用方式": "僅顯示該路徑的預測目標與證據，不納入共同排名",
                    "研究模型狀態": "候選驗證中",
                }
            )
        st.dataframe(pd.DataFrame(research_rows), hide_index=True, width="stretch")
        model_info = research_preview.get("model", {})
        st.caption(
            "整合治療模型版本："
            + str(model_info.get("model_version", "—"))
            + "｜此區可用來確認新資料、重新訓練與重新輸出的流程。"
        )
        selection = model_info.get("model_selection", {})
        if isinstance(selection, dict) and selection:
            validation_rows = []
            for treatment_name, details in selection.items():
                if not isinstance(details, dict):
                    continue
                fold_mae = [
                    float(value) for value in details.get("selected_fold_mae", [])
                    if value is not None
                ]
                r2_value = details.get("r2")
                mae_passed = details.get("outperforms_mean_baseline") is True
                stable_passed = (
                    r2_value is not None
                    and float(r2_value) >= 0.0
                    and (
                        not fold_mae
                        or max(fold_mae) - min(fold_mae) <= 0.10
                    )
                )
                validation_rows.append({
                    "治療模型": treatment_name,
                    "入選演算法": details.get("selected_algorithm", "—"),
                    "患者分組折數": details.get("fold_count", "—"),
                    "驗證 MAE": details.get("mae"),
                    "平均值基準 MAE": details.get("mean_baseline_mae"),
                    "R²": r2_value,
                    "各折 MAE 範圍": (
                        f"{min(fold_mae):.3f}–{max(fold_mae):.3f}"
                        if fold_mae else "—"
                    ),
                    "驗證結論": (
                        "完整通過"
                        if mae_passed and stable_passed
                        else "MAE通過，但穩定性未通過"
                        if mae_passed
                        else "未通過"
                    ),
                    "允許調整分數": "是" if details.get("score_adjustment_allowed") else "否",
                })
            with st.expander("查看四種治療模型的選模與驗證結果"):
                st.dataframe(
                    pd.DataFrame(validation_rows), hide_index=True, width="stretch"
                )
                st.caption(
                    "每種治療分別比較多個候選演算法，以患者分組交叉驗證 MAE 最低者入選；"
                    "MAE、R²與各折波動會分開呈現。MAE略優於基準不等於模型已具完整穩定性。"
                )
        drug_trial = model_info.get("drug_trial_challenger", {})
        if isinstance(drug_trial, dict) and drug_trial:
            st.info(
                "藥物研究 Challenger：完整配對 "
                f"{drug_trial.get('complete_pair_count', '—')} 人（藥物組 "
                f"{drug_trial.get('drug_count', '—')}、安慰劑組 "
                f"{drug_trial.get('placebo_count', '—')}）；藥物研究分數調整 "
                f"{float(drug_trial.get('research_score_adjustment_points', 0)):+.1f} 分。"
                "此結果只影響藥物研究排名，不影響今晚風險模型，也不是正式臨床模型。"
            )

    st.divider()

    show_clinical_prerequisites(
        active_prerequisites=(
            interface_data.get(
                "clinical_prerequisites",
                [],
            )
        ),
        resolved_prerequisites=(
            interface_data.get(
                "resolved_clinical_prerequisites",
                [],
            )
        ),
    )

    safety_note = interface_data.get(
        "safety_note"
    )

    if safety_note:
        st.divider()
        st.warning(safety_note)

    with interface_path.open(
        "rb",
    ) as input_file:
        interface_bytes = input_file.read()

    st.download_button(
        label=(
            "下載 Treatment Refinement "
            "Interface JSON"
        ),
        data=interface_bytes,
        file_name=interface_path.name,
        mime="application/json",
        width="stretch",
    )

    st.caption(
        f"資料來源：{interface_path}"
    )

def safe_float(
    value,
    default: float = 0.0,
) -> float:
    """
    將輸入安全轉換為 float。
    None、空字串或無效值會回傳 default。
    """
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def get_clinical_field_label(field_spec) -> str:
    """Prefer a readable Traditional-Chinese label from the schema aliases."""
    preferred = {
        "nasal_septal_deviation": "鼻中膈偏曲",
        "inferior_turbinate_hypertrophy": "下鼻甲肥厚",
        "tonsil_hypertrophy": "扁桃腺分級",
        "soft_palate_abnormality": "軟顎異常",
        "mallampati": "Mallampati 分級",
        "nasal_obstruction": "鼻腔阻塞／鼻塞",
        "adenoid_hypertrophy": "腺樣體肥厚",
        "exam_date": "檢查日期",
        "velum_collapse": "軟顎塌陷",
        "oropharyngeal_lateral_wall_collapse": "口咽側壁塌陷",
        "tongue_base_collapse": "舌根塌陷",
        "epiglottis_collapse": "會厭塌陷",
        "collapse_pattern": "塌陷型態",
        "collapse_degree": "塌陷程度",
        "vote_classification": "VOTE 分類",
        "modality": "影像檢查類型",
        "nasal_airway_narrowing": "鼻腔氣道狹窄",
        "retropalatal_airway_narrowing": "顎後氣道狹窄",
        "retroglossal_airway_narrowing": "舌後氣道狹窄",
        "craniofacial_abnormality": "顱顏面異常",
        "other_anatomical_obstruction": "其他解剖阻塞／影像所見",
        "minimum_airway_area": "最小氣道截面積",
        "airway_volume": "氣道容積",
        "document_type": "文件類型",
        "clinical_summary": "臨床摘要",
        "assessment": "評估／診斷印象",
        "plan": "治療計畫／後續處置",
        "rhythm": "心律",
        "heart_rate": "心率",
        "atrial_fibrillation": "心房顫動",
        "conduction_abnormality": "傳導異常",
        "interpretation": "判讀／結論",
    }
    if field_spec.name in preferred:
        return preferred[field_spec.name]
    for alias in field_spec.aliases:
        if any("\u4e00" <= char <= "\u9fff" for char in alias):
            return alias
    return field_spec.name.replace("_", " ").title()


def render_schema_confirmation_form(
    module_id: str,
    parsed_data: dict,
) -> dict:
    """Render physician-editable fields from the parser schema."""
    try:
        module_spec = get_module_spec(module_id)
    except KeyError:
        st.json(parsed_data, expanded=True)
        return dict(parsed_data)

    confirmed: dict = {}

    for field_spec in module_spec.fields:
        field_name = field_spec.name
        current_value = parsed_data.get(field_name)
        label = get_clinical_field_label(field_spec)
        widget_key = (
            f"confirmed_{module_id.lower()}_{field_name}"
        )

        if field_spec.value_type == "boolean":
            option_map = {
                "尚未提供": None,
                "是／有／陽性": True,
                "否／無／陰性": False,
            }
            default_label = "尚未提供"
            if current_value is True:
                default_label = "是／有／陽性"
            elif current_value is False:
                default_label = "否／無／陰性"

            selected_label = st.selectbox(
                label,
                options=list(option_map),
                index=list(option_map).index(default_label),
                key=widget_key,
            )
            value = option_map[selected_label]

        elif field_spec.value_type in {"float", "int"}:
            numeric_text = st.text_input(
                (
                    f"{label}（{field_spec.unit}）"
                    if field_spec.unit
                    else label
                ),
                value=(
                    ""
                    if current_value is None
                    else str(current_value)
                ),
                key=widget_key,
            ).strip()

            value = None
            if numeric_text:
                try:
                    numeric_value = float(numeric_text)
                except ValueError:
                    st.warning(
                        f"`{field_name}` 不是有效數字，"
                        "本次不會寫入此欄位。"
                    )
                else:
                    in_range = True
                    if (
                        field_spec.minimum is not None
                        and numeric_value < field_spec.minimum
                    ):
                        in_range = False
                    if (
                        field_spec.maximum is not None
                        and numeric_value > field_spec.maximum
                    ):
                        in_range = False
                    if not in_range:
                        st.warning(
                            f"`{field_name}` 超出合理範圍，"
                            "本次不會寫入此欄位。"
                        )
                    else:
                        value = (
                            int(round(numeric_value))
                            if field_spec.value_type == "int"
                            else numeric_value
                        )

        elif field_spec.choices:
            choices = ["尚未提供", *field_spec.choices]
            default = (
                str(current_value)
                if current_value in field_spec.choices
                else "尚未提供"
            )
            selected = st.selectbox(
                label,
                options=choices,
                index=choices.index(default),
                key=widget_key,
            )
            value = None if selected == "尚未提供" else selected

        else:
            value = st.text_area(
                label,
                value=(
                    ""
                    if current_value is None
                    else str(current_value)
                ),
                height=80,
                key=widget_key,
            ).strip()
            if not value:
                value = None

        if value is not None and value != "":
            confirmed[field_name] = value

    extra_fields = {
        key: value
        for key, value in parsed_data.items()
        if key not in {field.name for field in module_spec.fields}
        and value is not None
        and value != ""
    }
    confirmed.update(extra_fields)
    return confirmed


def get_pipeline_step_from_line(line: str) -> str | None:
    """
    根據 Pipeline 輸出文字推測目前執行步驟。

    即使你的 Pipeline 顯示方式稍有不同，
    沒有匹配到也不會影響實際執行。
    """
    normalized = line.lower()

    step_keywords = [
        (
            [
                "process_patient",
                "前處理",
                "preprocess",
            ],
            "1/10　患者資料前處理",
        ),
        (
            [
                "predict_incoming_patient",
                "arousal prediction",
                "arousal 推論",
            ],
            "2/10　睡眠訊號特徵分析",
        ),
        (
            [
                "analyze_arousal_early_warning",
                "early warning",
                "提前預警",
            ],
            "2/10　睡眠訊號特徵分析",
        ),
        (
            [
                "build_patient_respiratory_profile",
                "respiratory profile",
            ],
            "3/10　呼吸事件與睡眠分期摘要",
        ),
        (
            [
                "build_patient_position_profile",
                "position profile",
            ],
            "4/10　睡眠姿勢分析",
        ),
        (
            [
                "analyze_oxygen_event_coupling",
                "oxygen event coupling",
                "oxygen-event coupling",
            ],
            "6/10　血氧與呼吸事件關聯",
        ),
        (
            [
                "build_treatment_suitability",
                "treatment suitability",
            ],
            "7/10　治療適用性分析",
        ),
        (
            [
                "build_patient_digital_twin_report",
                "digital twin report",
            ],
            "9/10　Digital Twin 報告",
        ),
        (
            ["build_spo2_quality_mask", "spo2 quality mask"],
            "5/10　SpO₂ 資料品質檢查",
        ),
        (
            ["build_treatment_refinement", "treatment refinement"],
            "8/10　個人化治療建議",
        ),
        (
            ["predict_tonight_apnea_risk", "tonight risk prediction"],
            "10/10　今晚睡眠呼吸風險",
        ),
    ]

    for keywords, step_name in step_keywords:
        if any(
            keyword in normalized
            for keyword in keywords
        ):
            return step_name

    return None


def run_pipeline(
    patient_id: str,
    follow_up: bool = False,
) -> tuple[int, str]:
    """
    執行 run_patient_pipeline.py，並在網頁即時顯示輸出。

    回傳：
    - return code
    - 完整執行紀錄
    """
    if not PIPELINE_SCRIPT.is_file():
        raise FileNotFoundError(
            f"找不到 Pipeline 程式：{PIPELINE_SCRIPT}"
        )

    command = [
        sys.executable,
        "-u",
        str(PIPELINE_SCRIPT),
        "--patient-id",
        patient_id,
        "--allow-partial-edf",
    ]
    if follow_up:
        command.append(
            "--follow-up"
        )

    log_placeholder = st.empty()
    step_placeholder = st.empty()
    progress_bar = st.progress(
        0,
        text="準備啟動分析 Pipeline...",
    )

    all_lines: list[str] = []
    detected_steps: list[str] = []

    process_environment = os.environ.copy()

    # 讓 Python 子程序盡量立即輸出，不要大量緩衝
    process_environment["PYTHONUNBUFFERED"] = "1"
    process_environment["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=process_environment,
    )

    if process.stdout is None:
        raise RuntimeError(
            "無法取得 Pipeline 的標準輸出。"
        )

    for raw_line in iter(
        process.stdout.readline,
        "",
    ):
        line = raw_line.rstrip("\r\n")

        if not line and process.poll() is not None:
            break

        all_lines.append(line)

        step_name = get_pipeline_step_from_line(line)

        if (
            step_name is not None
            and step_name not in detected_steps
        ):
            detected_steps.append(step_name)

            progress_value = min(
                len(detected_steps) / 10,
                0.95,
            )

            progress_bar.progress(
                progress_value,
                text=step_name,
            )

            step_placeholder.info(
                f"目前步驟：{step_name}"
            )

        # 只顯示最後 250 行，避免網頁過度龐大
        visible_log = "\n".join(
            all_lines[-250:]
        )

        log_placeholder.code(
            visible_log,
            language="text",
        )

    process.stdout.close()

    return_code = process.wait()

    complete_log = "\n".join(all_lines)

    if return_code == 0:
        progress_bar.progress(
            1.0,
            text="分析 Pipeline 執行完成",
        )
        step_placeholder.success(
            "所有分析步驟執行完成。"
        )
    else:
        progress_bar.progress(
            1.0,
            text="分析 Pipeline 執行失敗",
        )
        step_placeholder.error(
            f"Pipeline 執行失敗，Return Code：{return_code}"
        )

    return return_code, complete_log


def get_persisted_pipeline_log_path(patient_id: str) -> Path:
    """Return the durable log path for one patient's latest full Pipeline run."""
    return INFERENCE_ROOT / patient_id / "pipeline_execution.log"


def persist_pipeline_log(patient_id: str, pipeline_log: str) -> None:
    """Persist the complete Pipeline output so it survives Streamlit reruns/refreshes."""
    log_path = get_persisted_pipeline_log_path(patient_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(pipeline_log, encoding="utf-8")


def load_persisted_pipeline_log(patient_id: str | None) -> str:
    """Load the latest durable Pipeline output for the active patient, if present."""
    if not patient_id:
        return ""
    log_path = get_persisted_pipeline_log_path(patient_id)
    if not log_path.is_file():
        return ""
    try:
        return log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def explain_pipeline_failure(pipeline_log: str) -> tuple[str, list[str]]:
    """Convert common pipeline failures into actionable UI guidance."""
    if "停止分析：EDF 記錄不完整" in pipeline_log:
        details = []
        for label in (
            "EDF 時長：",
            "Stage 時間範圍：",
            "理論最大 Stage：",
        ):
            matching_lines = [
                line.strip()
                for line in pipeline_log.splitlines()
                if line.strip().startswith(label)
            ]
            if matching_lines:
                details.append(matching_lines[-1])

        return (
            "EDF 與睡眠分期資料的時間長度不一致。"
            "為避免根據不完整訊號產生錯誤的醫療預測，系統已停止分析。",
            details
            + [
                "請確認 EDF、Stage、Event Grid 是否來自同一次、同一位患者的完整睡眠檢查。",
                "這次上傳的 EDF 只有約 132.67 分鐘，但 Stage 約有 431.50 分鐘。",
                "請改用完整 EDF；不要只上傳被截短或匯出不完整的片段。",
            ],
        )

    if "No module named 'openpyxl'" in pipeline_log:
        return (
            "缺少讀取 .xlsx 所需的 openpyxl 套件。",
            ["目前專案已補齊此套件，請重新執行分析。"],
        )

    if "No module named 'xlrd'" in pipeline_log:
        return (
            "缺少讀取舊版 .xls 所需的 xlrd 套件。",
            ["目前專案已補齊此套件，請重新執行分析。"],
        )

    return (
        "Pipeline 執行失敗。請查看下方的錯誤紀錄以確認失敗步驟。",
        [],
    )


def show_pipeline_failure_details(pipeline_log: str) -> None:
    """Show a concise diagnosis first, with the technical trace available."""
    summary, actions = explain_pipeline_failure(pipeline_log)
    st.error(summary)
    for action in actions:
        st.write(f"- {action}")

    log_lines = pipeline_log.splitlines()
    with st.expander("查看完整 Pipeline 錯誤紀錄", expanded=False):
        st.code("\n".join(log_lines[-120:]), language="text")


def show_saved_patient_information() -> None:
    """
    顯示最近成功儲存的患者資訊。
    """
    patient_id = st.session_state.get(
        "saved_patient_id"
    )

    patient_folder = st.session_state.get(
        "saved_patient_folder"
    )

    saved_files = st.session_state.get(
        "saved_files",
        {},
    )

    if not patient_id:
        return

    st.subheader("患者資料")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Patient ID",
            value=patient_id,
        )

    with col2:
        st.metric(
            label="已儲存檔案",
            value=len(saved_files),
        )

    if patient_folder:
        st.write("患者上傳資料夾：")
        st.code(
            patient_folder,
            language=None,
        )

    if saved_files:
        st.write("已儲存檔案：")

        for label, path_text in saved_files.items():
            path = Path(path_text)
            st.write(
                f"{label}：`{path.name}`"
            )


def show_download_buttons(
    report_paths: dict[str, Path],
) -> None:
    """
    顯示報告下載按鈕。
    """
    st.subheader("下載 Digital Twin 報告")

    if not report_paths:
        st.warning(
            "Pipeline 已完成，但尚未找到 Digital Twin 報告檔案。"
        )
        return

    extensions = [
        extension
        for extension in [
            "html",
            "json",
            "csv",
            "txt",
        ]
        if extension in report_paths
    ]

    columns = st.columns(
        len(extensions)
    )

    mime_types = {
        "html": "text/html",
        "json": "application/json",
        "csv": "text/csv",
        "txt": "text/plain",
    }

    labels = {
        "html": "下載 HTML 報告",
        "json": "下載 JSON 報告",
        "csv": "下載 CSV 報告",
        "txt": "下載 TXT 報告",
    }

    for column, extension in zip(
        columns,
        extensions,
    ):
        report_path = report_paths[extension]

        with report_path.open("rb") as input_file:
            report_bytes = input_file.read()

        with column:
            st.download_button(
                label=labels[extension],
                data=report_bytes,
                file_name=report_path.name,
                mime=mime_types[extension],
                width="stretch",
            )

    st.write("報告位置：")

    for extension in extensions:
        st.code(
            str(report_paths[extension]),
            language=None,
        )


def get_treatment_refinement_paths(
    patient_id: str,
) -> dict[str, Path]:
    """
    回傳患者 Treatment Refinement 相關檔案路徑。
    """
    output_dir = (
        INFERENCE_ROOT
        / patient_id
        / "treatment_refinement"
    )

    return {
        "output_dir": output_dir,
        "clinical_data": (
            output_dir
            / "clinical_decision_data.json"
        ),
        "refined_result": (
            output_dir
            / "refined_treatment_recommendation.json"
        ),
        "interface": (
            output_dir
            / TREATMENT_INTERFACE_FILENAME
        ),
        "history_dir": (
            output_dir
            / "history"
        ),
    }


def save_json_file(
    path: Path,
    data: dict,
) -> None:
    """
    以 UTF-8 與嚴格 JSON 格式安全寫入檔案。
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            data,
            output_file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    temporary_path.replace(path)


def make_timestamp() -> str:
    """
    建立適合檔名使用的時間戳記。
    """
    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def snapshot_file(
    source_path: Path,
    history_dir: Path,
    label: str,
    timestamp: str,
) -> Path | None:
    """
    將現有檔案複製到 history，供重新評估前後比較。
    """
    if not source_path.is_file():
        return None

    history_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_path = (
        history_dir
        / f"{timestamp}_{label}_{source_path.name}"
    )

    shutil.copy2(
        source_path,
        snapshot_path,
    )

    return snapshot_path


def normalize_confirmed_clinical_data(
    module_id: str,
    confirmed_data: dict,
) -> dict:
    """
    將 UI 確認資料整理成可寫入 ClinicalDecisionData 的 findings。

    None 與空字串不會覆蓋既有有效資料。
    """
    if not isinstance(
        confirmed_data,
        dict,
    ):
        raise TypeError(
            "確認後的臨床資料必須是 dict。"
        )

    normalized_data = {
        key: value
        for key, value in confirmed_data.items()
        if value is not None
        and value != ""
    }

    if module_id == "PAP":
        # 目前 PAP 規則必須知道治療模式。
        normalized_data.setdefault(
            "therapy_type",
            "CPAP",
        )

    return normalized_data


def update_clinical_decision_data(
    patient_id: str,
    module_id: str,
    confirmed_data: dict,
    source_files: list[str],
) -> tuple[Path, dict, dict]:
    """
    把醫師確認資料正式合併進 clinical_decision_data.json。

    回傳：
    - ClinicalDecisionData 路徑
    - 更新前資料
    - 更新後資料
    """
    paths = get_treatment_refinement_paths(
        patient_id
    )

    clinical_data_path = paths[
        "clinical_data"
    ]

    if not clinical_data_path.is_file():
        raise FileNotFoundError(
            "找不到 ClinicalDecisionData："
            f"{clinical_data_path}"
        )

    original_data = load_json_file(
        clinical_data_path
    )

    updated_data = json.loads(
        json.dumps(
            original_data,
            ensure_ascii=False,
        )
    )

    module_key_map = get_module_key_map()

    module_key = module_key_map.get(
        module_id
    )

    if module_key is None:
        raise ValueError(
            "目前尚未建立此資料類型與 "
            "ClinicalDecisionData 的映射："
            f"{module_id}"
        )

    module_record = updated_data.setdefault(
        module_key,
        {},
    )

    if not isinstance(module_record, dict):
        module_record = {}
        updated_data[module_key] = (
            module_record
        )

    existing_findings = module_record.get(
        "findings",
        {},
    )

    if not isinstance(
        existing_findings,
        dict,
    ):
        existing_findings = {}

    normalized_findings = (
        normalize_confirmed_clinical_data(
            module_id=module_id,
            confirmed_data=confirmed_data,
        )
    )

    merged_findings = {
        **existing_findings,
        **normalized_findings,
    }

    module_record["available"] = True
    module_record["findings"] = (
        merged_findings
    )
    module_record["source"] = (
        ", ".join(source_files)
        if source_files
        else "Streamlit clinical update"
    )
    module_record["notes"] = (
        "Confirmed through Sleep Digital "
        "Twin clinical update workflow."
    )

    current_version = updated_data.get(
        "data_version",
        0,
    )

    try:
        current_version = int(
            current_version
        )
    except (TypeError, ValueError):
        current_version = 0

    updated_data["data_version"] = (
        current_version + 1
    )
    updated_data["patient_id"] = patient_id
    updated_data[
        "last_updated_at"
    ] = datetime.now().isoformat(
        timespec="seconds"
    )
    updated_data[
        "last_updated_module"
    ] = module_id

    save_json_file(
        clinical_data_path,
        updated_data,
    )

    return (
        clinical_data_path,
        original_data,
        updated_data,
    )


def run_treatment_refinement(
    patient_id: str,
) -> tuple[int, str]:
    """
    真正執行 build_treatment_refinement.py。
    """
    if not TREATMENT_REFINEMENT_SCRIPT.is_file():
        raise FileNotFoundError(
            "找不到 Treatment Refinement 程式："
            f"{TREATMENT_REFINEMENT_SCRIPT}"
        )

    command = [
        sys.executable,
        "-u",
        str(
            TREATMENT_REFINEMENT_SCRIPT
        ),
        "--patient-id",
        patient_id,
    ]

    process_environment = os.environ.copy()
    process_environment[
        "PYTHONUNBUFFERED"
    ] = "1"
    process_environment[
        "PYTHONIOENCODING"
    ] = "utf-8"

    completed_process = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=process_environment,
        check=False,
    )

    combined_log = "\n".join(
        part
        for part in [
            completed_process.stdout,
            completed_process.stderr,
        ]
        if part
    )

    return (
        completed_process.returncode,
        combined_log,
    )


def index_treatments(
    interface_data: dict,
) -> dict[str, dict]:
    """
    將 Interface treatments 轉成 treatment_id 索引。
    """
    treatments = interface_data.get(
        "treatments",
        [],
    )

    if not isinstance(treatments, list):
        return {}

    return {
        treatment.get(
            "treatment_id"
        ): treatment
        for treatment in treatments
        if isinstance(treatment, dict)
        and treatment.get("treatment_id")
    }


def index_research_treatments(interface_data: dict) -> dict[str, dict]:
    """Index the latest research-Challenger ranking by treatment code."""
    preview = interface_data.get("research_treatment_preview", {})
    ranking = preview.get("ranking", []) if isinstance(preview, dict) else []
    if not isinstance(ranking, list):
        return {}
    return {
        str(item.get("treatment")): item
        for item in ranking
        if isinstance(item, dict) and item.get("treatment")
    }


def compare_treatment_interfaces(
    before_data: dict,
    after_data: dict,
) -> dict:
    """
    比較重新評估前後的治療排名與分數。
    """
    before_index = index_treatments(
        before_data
    )
    after_index = index_treatments(
        after_data
    )

    treatment_ids = sorted(
        set(before_index)
        | set(after_index)
    )

    before_research = index_research_treatments(before_data)
    after_research = index_research_treatments(after_data)
    treatment_changes = []

    for treatment_id in treatment_ids:
        before_item = before_index.get(
            treatment_id,
            {},
        )
        after_item = after_index.get(
            treatment_id,
            {},
        )

        before_score = safe_float(
            before_item.get("final_score"),
            default=0.0,
        )
        after_score = safe_float(
            after_item.get("final_score"),
            default=0.0,
        )

        research_before = before_research.get(treatment_id, {})
        research_after = after_research.get(treatment_id, {})
        model_adjustment_before = safe_float(
            research_before.get("learned_adjustment"), default=0.0
        )
        model_adjustment_after = safe_float(
            research_after.get("learned_adjustment"), default=0.0
        )
        research_score_before = safe_float(
            research_before.get("score"), default=before_score
        )
        research_score_after = safe_float(
            research_after.get("score"), default=after_score
        )

        treatment_changes.append(
            {
                "treatment_id":
                    treatment_id,
                "treatment_label": (
                    after_item.get(
                        "treatment_label"
                    )
                    or before_item.get(
                        "treatment_label"
                    )
                    or treatment_id
                ),
                "rank_before":
                    before_item.get("rank"),
                "rank_after":
                    after_item.get("rank"),
                "score_before":
                    before_score,
                "score_after":
                    after_score,
                "score_change":
                    after_score
                    - before_score,
                "research_score_before": research_score_before,
                "research_score_after": research_score_after,
                "research_score_change": research_score_after - research_score_before,
                "model_adjustment_before": model_adjustment_before,
                "model_adjustment_after": model_adjustment_after,
                "model_adjustment_change": model_adjustment_after - model_adjustment_before,
                "guardrail_applied": abs(research_score_after - after_score) > 0.001,
            }
        )

    before_summary = before_data.get(
        "recommendation_summary",
        {},
    )
    after_summary = after_data.get(
        "recommendation_summary",
        {},
    )

    if not isinstance(
        before_summary,
        dict,
    ):
        before_summary = {}

    if not isinstance(
        after_summary,
        dict,
    ):
        after_summary = {}

    top_before = before_summary.get(
        "recommended_first_candidate"
    )
    top_after = after_summary.get(
        "recommended_first_candidate"
    )

    return {
        "top_treatment_before":
            top_before,
        "top_treatment_after":
            top_after,
        "ranking_changed": (
            top_before != top_after
        ),
        "treatment_changes":
            treatment_changes,
        "model_version_before": (
            before_data.get("research_treatment_preview", {}).get("model", {}).get("version")
            if isinstance(before_data.get("research_treatment_preview"), dict) else None
        ),
        "model_version_after": (
            after_data.get("research_treatment_preview", {}).get("model", {}).get("version")
            if isinstance(after_data.get("research_treatment_preview"), dict) else None
        ),
    }


def show_reevaluation_result() -> None:
    """
    顯示最近一次重新評估執行結果。
    """
    result = st.session_state.get(
        "reevaluation_result"
    )

    if not isinstance(result, dict):
        return

    st.divider()
    st.subheader("重新評估執行結果")

    st.success(
        "補充臨床資料已寫入 "
        "ClinicalDecisionData，且 "
        "Treatment Refinement 已重新執行。"
    )

    status_columns = st.columns(3)

    with status_columns[0]:
        st.metric(
            "已套用模組",
            result.get(
                "module",
                "-",
            ),
        )

    with status_columns[1]:
        st.metric(
            "資料版本",
            result.get(
                "data_version",
                "-",
            ),
        )

    comparison = result.get(
        "comparison",
        {},
    )

    with status_columns[2]:
        ranking_changed = bool(
            comparison.get(
                "ranking_changed"
            )
        )

        st.metric(
            "第一候選是否改變",
            (
                "是"
                if ranking_changed
                else "否"
            ),
        )

    applied_fields = result.get(
        "applied_fields",
        [],
    )

    if applied_fields:
        st.markdown("#### 已納入模型的欄位")

        for field_name in applied_fields:
            st.markdown(
                f"- `{field_name}`"
            )

    treatment_changes = comparison.get(
        "treatment_changes",
        [],
    )

    learning_rows = [
        {
            "治療方式": (
                "上呼吸道手術專科轉介評估優先度"
                if item.get("treatment_id") == "SURGERY"
                else item.get("treatment_label")
            ),
            "本次模型調整": round(safe_float(item.get("model_adjustment_after")), 2),
            "相較前版模型": round(safe_float(item.get("model_adjustment_change")), 2),
            "研究分數（前）": round(safe_float(item.get("research_score_before")), 4),
            "研究分數（後）": round(safe_float(item.get("research_score_after")), 4),
            "高精度變化": round(safe_float(item.get("research_score_change")), 4),
            "安全規則限制": "是" if item.get("guardrail_applied") else "否",
        }
        for item in treatment_changes
    ]

    model_before = comparison.get("model_version_before") or "未記錄"
    model_after = comparison.get("model_version_after") or "未記錄"
    model_changed = model_before != model_after and model_after != "未記錄"
    with st.container(border=True):
        st.markdown("**本次模型更新證據**")
        st.write(f"模型版本：`{model_before}` → `{model_after}`")
        st.write("新模型版本：" + ("已產生" if model_changed else "未偵測到版本變更"))
        receipt_path = (
            INFERENCE_ROOT / str(result.get("patient_id", "")) /
            "model_update_audit" / "latest_clinical_treatment_retrain_receipt.json"
        )
        receipt = load_json_file(receipt_path) if receipt_path.is_file() else {}
        if receipt:
            st.write(
                "訓練資料："
                f"{receipt.get('training_row_count', 0)} 列（本次上傳快照："
                f"{'已加入' if receipt.get('new_upload_snapshots_included') else '未加入'}；"
                f"真實療效標籤 {receipt.get('confirmed_outcome_rows', 0)} 列；"
                f"弱標籤 {receipt.get('weak_label_rows', 0)} 列）"
            )
        st.caption(
            "最終臨床顯示分數可能受醫療安全規則限制；下方另列模型原始調整，"
            "避免把安全限制後的 0 分差誤認為模型沒有學習。"
        )

    if treatment_changes:
        st.markdown("#### 模型學習調整（安全規則套用前）")
        st.dataframe(learning_rows, hide_index=True)
        st.markdown("#### 本次上傳後的實際模型分數變化")
        st.caption(
            "以下優先顯示重新訓練後的研究Challenger分數前後差異；"
            "醫療規則基準未變不代表模型沒有更新。"
        )

        for item in sorted(
            treatment_changes,
            key=lambda value: (
                value.get(
                    "rank_after"
                )
                if value.get(
                    "rank_after"
                ) is not None
                else 999
            ),
        ):
            score_before_display = safe_float(item.get("research_score_before"))
            score_after_display = safe_float(item.get("research_score_after"))
            score_change = score_after_display - score_before_display

            st.markdown(
                "**"
                f"{item.get('treatment_label')}"
                "**："
                f"排名 {item.get('rank_before')} "
                f"→ {item.get('rank_after')}；"
                f"分數 "
                f"{score_before_display:.4f} "
                f"→ "
                f"{score_after_display:.4f} "
                f"（{format_score_change(score_change)}）"
            )

    with st.expander(
        "查看重新評估執行紀錄",
        expanded=False,
    ):
        st.code(
            result.get(
                "execution_log",
                "",
            ),
            language="text",
        )

def _show_clinical_data_update_section_content(
    interface_data: dict,
    patient_id: str,
) -> None:
    """
    顯示補充臨床資料與治療重新評估流程。

    流程：
    1. 顯示系統建議優先補充的資料。
    2. 選擇臨床資料類型。
    3. 上傳補充資料檔案。
    4. 解析檔案。
    5. 醫師確認解析結果。
    6. 重新評估治療建議。
    """
    st.write(
        "上傳患者後續取得的臨床資料，"
        "系統會解析內容，經確認後更新 Digital Twin，"
        "並重新評估適合的治療方法。"
    )

    # ========================================================
    # AI 建議優先補充
    # ========================================================

    st.subheader("AI 建議優先補充")

    active_prerequisites = interface_data.get(
        "clinical_prerequisites",
        [],
    )

    if not isinstance(active_prerequisites, list):
        active_prerequisites = []

    valid_prerequisites = [
        prerequisite
        for prerequisite in active_prerequisites
        if isinstance(prerequisite, dict)
    ]

    valid_prerequisites = sorted(
        valid_prerequisites,
        key=lambda prerequisite: int(
            prerequisite.get("priority", 999)
        ),
    )

    if not valid_prerequisites:
        st.success(
            "目前沒有尚未完成的臨床前置事項。"
        )

    for prerequisite in valid_prerequisites:
        priority = prerequisite.get(
            "priority",
            "-",
        )

        title = prerequisite.get(
            "title",
            "未命名前置事項",
        )

        reason = prerequisite.get(
            "reason",
            "",
        )

        required_information = prerequisite.get(
            "required_information",
            [],
        )

        if priority == 1:
            priority_text = "★★★★★ Priority 1"
        elif priority == 2:
            priority_text = "★★★★☆ Priority 2"
        elif priority == 3:
            priority_text = "★★★☆☆ Priority 3"
        elif priority == 4:
            priority_text = "★★☆☆☆ Priority 4"
        else:
            priority_text = f"★☆☆☆☆ Priority {priority}"

        with st.container(border=True):
            st.markdown(
                f"### {priority_text}"
            )

            st.markdown(
                f"**{title}**"
            )

            if reason:
                st.write(reason)

            if required_information:
                st.markdown(
                    "**建議補充內容：**"
                )

                for item in required_information:
                    st.markdown(f"- {item}")

    st.divider()

    # ========================================================
    # 新增臨床資料
    # ========================================================

    st.subheader("補充臨床資料設定")

    st.info(
        "下列每一種資料在醫師確認後，都會更新患者 Digital Twin、"
        "轉成可稽核的治療特徵並更新 HTML 報告；只有具基準值、追蹤值、治療方式與醫師確認的真實療效資料，才可訓練研究型 Challenger。"
    )

    with st.expander(
        "醫師快速補登｜PAP耐受、失眠與用藥安全",
        expanded=False,
        icon=":material/edit_note:",
    ):
        st.caption(
            "僅填寫已由患者訪談、病歷或醫師確認的內容。未填寫不會被當成陰性資料；"
            "儲存後會更新此患者Digital Twin並重新計算治療與專科介入評估優先度。"
        )
        with st.form("medication_pathway_quick_entry", clear_on_submit=False):
            pap_tolerance_entry = st.selectbox(
                "PAP耐受或接受情況",
                ["未填寫", "良好", "部分耐受", "不耐受", "拒絕"],
            )
            isi_available = st.checkbox("已有ISI失眠嚴重度指數")
            isi_entry = st.number_input(
                "ISI分數（0–28）",
                min_value=0,
                max_value=28,
                value=15,
                disabled=not isi_available,
            )
            insomnia_entry = st.selectbox(
                "是否已有失眠診斷",
                ["未填寫", "是", "否"],
            )
            medication_list_entry = st.text_area(
                "目前用藥（可填藥名，以逗號分隔）"
            )
            sedative_entry = st.selectbox(
                "是否使用鎮靜安眠藥",
                ["未填寫", "是", "否"],
            )
            opioid_entry = st.selectbox(
                "是否使用鴉片類藥物",
                ["未填寫", "是", "否"],
            )
            depressant_entry = st.selectbox(
                "是否使用其他可能抑制呼吸的藥物",
                ["未填寫", "是", "否"],
            )
            quick_entry_confirmed = st.checkbox(
                "我確認以上內容來自患者、病歷或醫師審核資料"
            )
            save_quick_entry = st.form_submit_button(
                "儲存並重新評估",
                type="primary",
                disabled=not quick_entry_confirmed,
                width="stretch",
            )

        if save_quick_entry:
            yes_no = {"是": True, "否": False}
            updates = []
            if pap_tolerance_entry != "未填寫":
                updates.append({
                    "module_id": "FOLLOW_UP",
                    "confirmed_data": {"tolerance": pap_tolerance_entry},
                    "source_files": ["manual_clinician_entry"],
                })
            questionnaire_data = {}
            if isi_available:
                questionnaire_data["isi_score"] = int(isi_entry)
            if insomnia_entry != "未填寫":
                questionnaire_data["insomnia_diagnosis"] = yes_no[insomnia_entry]
            if questionnaire_data:
                updates.append({
                    "module_id": "SLEEP_QUESTIONNAIRE",
                    "confirmed_data": questionnaire_data,
                    "source_files": ["manual_clinician_entry"],
                })
            medication_data = {}
            if medication_list_entry.strip():
                medication_data["medication_list"] = medication_list_entry.strip()
            for field_name, field_value in (
                ("sedative_hypnotics", sedative_entry),
                ("opioids", opioid_entry),
                ("respiratory_depressants", depressant_entry),
            ):
                if field_value != "未填寫":
                    medication_data[field_name] = yes_no[field_value]
            if medication_data:
                updates.append({
                    "module_id": "MEDICATION",
                    "confirmed_data": medication_data,
                    "source_files": ["manual_clinician_entry"],
                })

            if not updates:
                st.warning("尚未填寫任何可儲存的臨床資料。")
            else:
                try:
                    with st.spinner("正在更新Digital Twin並重新計算…"):
                        refresh_patient_batch(
                            patient_id=patient_id,
                            updates=updates,
                            project_root=PROJECT_ROOT,
                            regenerate_report=True,
                        )
                    st.success("臨床資料已儲存，新藥物評估路徑已重新計算。")
                    st.rerun()
                except Exception as exc:
                    st.error("臨床資料更新失敗，原有患者資料已保留。")
                    st.exception(exc)

    with st.expander("一次上傳多類補充資料（批次更新）", expanded=False):
        st.caption(
            "可同時選擇多個臨床類型，每類可上傳多份檔案。"
            "系統會使用各類型的專用讀取器，先全部解析，"
            "再一次合併 Digital Twin、新增訓練快照、重訓模型與產生輸出。"
        )
        batch_module_ids = st.multiselect(
            "選擇本次要同時上傳的資料類型",
            options=[
                module_id for module_id in CLINICAL_DATA_TYPES
                if module_id != "PSG_FOLLOW_UP" and has_parser(module_id)
            ],
            format_func=lambda module_id: CLINICAL_DATA_TYPES.get(module_id, module_id),
            key="batch_clinical_module_ids",
        )
        batch_uploads = {}
        for module_id in batch_module_ids:
            batch_uploads[module_id] = st.file_uploader(
                CLINICAL_DATA_TYPES.get(module_id, module_id),
                type=["pdf", "xls", "xlsx", "doc", "docx", "csv", "txt", "jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"batch_clinical_upload_{module_id}",
            )

        batch_ready = bool(batch_module_ids) and all(batch_uploads.get(module_id) for module_id in batch_module_ids)
        if st.button(
            "解析全部資料並批次更新模型",
            type="primary",
            disabled=not batch_ready,
            width="stretch",
            key="run_batch_clinical_update",
        ):
            try:
                parsed_updates = []
                parse_summary = []
                for module_id in batch_module_ids:
                    files = batch_uploads[module_id]
                    parsed = parse_clinical_update(
                        module_id=module_id,
                        parser=get_parser(module_id),
                        uploaded_files=files,
                    )
                    extracted_count = sum(
                        value not in (None, "", [], {}) for value in parsed.data.values()
                    )
                    if extracted_count == 0:
                        raise ValueError(
                            f"{CLINICAL_DATA_TYPES.get(module_id, module_id)} 未讀取到可用欄位"
                        )
                    confirmed_data = dict(parsed.data)
                    confirmed_data["synthetic_test_training"] = True
                    parsed_updates.append({
                        "module_id": module_id,
                        "confirmed_data": confirmed_data,
                        "source_files": list(parsed.source_files),
                    })
                    parse_summary.append({
                        "資料類型": CLINICAL_DATA_TYPES.get(module_id, module_id),
                        "檔案數": len(files),
                        "讀取欄位": extracted_count,
                    })
                with st.spinner("正在合併多類資料、新增訓練快照、重訓並更新輸出…"):
                    batch_result = refresh_patient_batch(
                        patient_id=patient_id,
                        updates=parsed_updates,
                        project_root=PROJECT_ROOT,
                        regenerate_report=True,
                    )
                st.session_state["batch_clinical_result"] = {
                    **batch_result,
                    "parse_summary": parse_summary,
                }
                st.success(
                    f"批次更新完成：{len(parsed_updates)} 種資料已合併，"
                    "治療模型僅重訓一次並已重新輸出。"
                )
                st.rerun()
            except Exception as exc:
                st.error("批次更新失敗；原始 Digital Twin 已保留或回復。")
                st.exception(exc)
        batch_result = st.session_state.get("batch_clinical_result")
        if isinstance(batch_result, dict):
            st.success(
                f"最近批次：{len(batch_result.get('module_ids', []))} 種資料，"
                f"Digital Twin 版本 {batch_result.get('version_before')} → {batch_result.get('version_after')}"
            )
            if batch_result.get("parse_summary"):
                st.dataframe(batch_result["parse_summary"], width="stretch", hide_index=True)

    selected_group = st.selectbox(
        "先選擇臨床領域",
        options=list(CLINICAL_DATA_GROUPS.keys()),
        key="clinical_data_group_selector",
    )

    type_options = CLINICAL_DATA_GROUPS[selected_group]

    selected_data_type = st.selectbox(
        "再選擇要補充的檢查或資料",
        options=type_options,
        format_func=lambda module_id: (
            CLINICAL_DATA_TYPES.get(
                module_id,
                module_id,
            )
        ),
        key="clinical_data_type_selector",
    )

    previous_selected_type = st.session_state.get("selected_clinical_data_type")
    if previous_selected_type and previous_selected_type != selected_data_type:
        for state_key in (
            "clinical_data_parsed",
            "clinical_data_confirmed",
            "parsed_clinical_data",
            "parsed_clinical_source_files",
            "parsed_clinical_warnings",
            "parsed_clinical_metadata",
        ):
            st.session_state.pop(state_key, None)

    st.session_state[
        "selected_clinical_data_type"
    ] = selected_data_type

    selected_display_name = (
        CLINICAL_DATA_TYPES.get(
            selected_data_type,
            selected_data_type,
        )
    )

    st.caption(
        f"目前選擇：{selected_display_name}"
    )

    # 最新 PSG 是獨立更新管線，必須永遠與其他補充臨床資料同時可用。
    # 因此不可再用 selected_data_type 控制其顯示。
    show_independent_psg_upload = True
    if show_independent_psg_upload:
        st.markdown("### 最新 PSG／睡眠檢查更新")
        st.info(
            "此更新會先備份目前患者的原始輸入與分析結果，"
            "再以新的 EDF、Stage、Event Grid 執行完整分析 Pipeline。"
        )

        study_type = st.selectbox(
            "檢查類型",
            options=[
                "治療後追蹤檢查",
                "PAP 治療中檢查",
                "術後追蹤檢查",
                "一般追蹤檢查",
                "其他",
            ],
            key="psg_follow_up_study_type",
        )
        study_date = st.date_input(
            "檢查日期",
            value=datetime.now().date(),
            key="psg_follow_up_study_date",
        )
        follow_up_edf = st.file_uploader(
            "最新 EDF", type=["edf"], key="psg_follow_up_edf"
        )
        follow_up_stage = st.file_uploader(
            "最新 Stage Excel", type=["xls", "xlsx"], key="psg_follow_up_stage"
        )
        follow_up_event = st.file_uploader(
            "最新 Event Grid Excel", type=["xls", "xlsx"], key="psg_follow_up_event"
        )
        follow_up_notes = st.text_area(
            "備註（選填）", key="psg_follow_up_notes"
        )

        all_follow_up_files = all(
            file is not None
            for file in (follow_up_edf, follow_up_stage, follow_up_event)
        )
        follow_up_validation_error = None
        if all_follow_up_files:
            try:
                validation_preview = validate_follow_up_files(
                    follow_up_edf,
                    follow_up_stage,
                    follow_up_event,
                )
            except ValueError as exc:
                follow_up_validation_error = str(exc)
                st.error(follow_up_validation_error)
            else:
                st.success(
                    "三個檔案已確認為同一次睡眠檢查："
                    f"{validation_preview['study_key']}"
                )
        run_follow_up = st.button(
            "儲存並執行最新睡眠分析",
            type="primary",
            disabled=(
                not all_follow_up_files
                or follow_up_validation_error is not None
            ),
            width="stretch",
            key="run_psg_follow_up",
        )

        if run_follow_up:
            manifest = None
            try:
                manifest = prepare_psg_follow_up(
                    patient_id=patient_id,
                    edf_file=follow_up_edf,
                    stage_file=follow_up_stage,
                    event_file=follow_up_event,
                    incoming_root=INCOMING_ROOT,
                    inference_root=INFERENCE_ROOT,
                    project_root=PROJECT_ROOT,
                    study_date=study_date.isoformat(),
                    study_type=study_type,
                    notes=follow_up_notes,
                )

                st.subheader("最新睡眠資料 Pipeline 執行紀錄")
                return_code, pipeline_log = run_pipeline(
                    patient_id=patient_id,
                    follow_up=True,
                )

                # Preserve the complete child-process output even when the
                # follow-up Pipeline fails.  Previously the return-code check
                # raised first, leaving the UI with only a generic wrapper
                # RuntimeError and hiding the actionable root cause.
                st.session_state["pipeline_log"] = pipeline_log
                st.session_state["pipeline_return_code"] = return_code
                persist_pipeline_log(patient_id, pipeline_log)

                if return_code != 0:
                    raise RuntimeError(
                        f"最新睡眠資料 Pipeline 執行失敗，Return Code：{return_code}"
                    )

                manifest = finalize_psg_follow_up(
                    manifest=manifest, inference_root=INFERENCE_ROOT
                )
                refreshed_report_paths = collect_report_paths(patient_id=patient_id)
                st.session_state["analysis_completed"] = True
                st.session_state["analysis_failed"] = False
                st.session_state["report_paths"] = {
                    extension: str(path)
                    for extension, path in refreshed_report_paths.items()
                }
                st.session_state["psg_follow_up_result"] = {
                    "success": True,
                    "study_id": manifest["study_id"],
                    "study_key": manifest["study_key"],
                    "study_date": manifest["study_date"],
                    "study_type": manifest["study_type"],
                    "history_path": str(
                        PROJECT_ROOT / "data" / "study_history" / patient_id / manifest["study_id"]
                    ),
                }
                st.session_state.pop("psg_follow_up_error", None)
                st.rerun()
            except Exception as exc:
                if manifest is not None:
                    try:
                        rollback_psg_follow_up(
                            manifest=manifest,
                            incoming_root=INCOMING_ROOT,
                            inference_root=INFERENCE_ROOT,
                        )
                    except Exception as rollback_exc:
                        st.error(f"自動回復原資料失敗：{rollback_exc}")
                st.session_state["psg_follow_up_error"] = str(exc)
                st.session_state["analysis_failed"] = True
                st.error("最新睡眠資料更新失敗；系統已嘗試回復更新前資料。")
                st.exception(exc)
                failed_log = st.session_state.get("pipeline_log", "")
                if failed_log:
                    show_pipeline_failure_details(failed_log)

        follow_up_result = st.session_state.get(
            "psg_follow_up_result"
        )

        if isinstance(follow_up_result, dict):
            st.divider()
            st.subheader("最新睡眠資料更新結果")

            st.success(
                "最新睡眠資料已完成完整 Pipeline 分析，"
                "並更新 Digital Twin。"
            )

            result_columns = st.columns(3)

            with result_columns[0]:
                st.metric(
                    "檢查日期",
                    follow_up_result.get(
                        "study_date",
                        "-",
                    ),
                )

            with result_columns[1]:
                st.metric(
                    "檢查類型",
                    follow_up_result.get(
                        "study_type",
                        "-",
                    ),
                )

            with result_columns[2]:
                st.metric(
                    "檢查識別碼",
                    follow_up_result.get(
                        "study_key",
                        "-",
                    ),
                )

            st.caption(
                "歷史備份位置："
                f"{follow_up_result.get('history_path', '-')}"
            )

            st.divider()
            st.subheader("更新後治療評估結果")

            refreshed_interface_path = (
                find_treatment_interface_file(
                    patient_id=patient_id,
                )
            )

            if refreshed_interface_path is None:
                st.warning(
                    "Pipeline 已完成，但找不到更新後的 "
                    "Treatment Refinement Interface JSON。"
                )

            else:
                try:
                    refreshed_interface_data = (
                        load_json_file(
                            refreshed_interface_path
                        )
                    )

                except Exception as exc:
                    st.error(
                        "讀取更新後治療評估結果失敗。"
                    )
                    st.exception(exc)

                else:
                    st.success(
                        "更新後的治療評估結果已重新載入。"
                    )

                    st.info(
                        "完整的最新治療排名、評分、"
                        "Clinical Prerequisites 與 Follow-up Comparison，"
                        "請查看本頁上方的「Treatment Refinement」結果區。"
                    )

                    top_candidate = refreshed_interface_data.get(
                        "top_candidate",
                        {},
                    )

                    if not isinstance(top_candidate, dict):
                        top_candidate = {}

                    if not top_candidate:
                        refreshed_candidates = (
                            refreshed_interface_data.get("treatments")
                            or []
                        )
                        if (
                            isinstance(refreshed_candidates, list)
                            and refreshed_candidates
                            and isinstance(refreshed_candidates[0], dict)
                        ):
                            top_candidate = refreshed_candidates[0]

                    top_name = (
                        top_candidate.get("treatment_name")
                        or top_candidate.get("treatment_label")
                        or top_candidate.get("name")
                        or top_candidate.get("treatment")
                        or top_candidate.get("treatment_id")
                        or "-"
                    )

                    top_score = top_candidate.get("final_score")
                    if top_score is None:
                        top_score = top_candidate.get("score")
                    if top_score is None:
                        top_score = "-"

                    summary_columns = st.columns(2)

                    with summary_columns[0]:
                        st.metric(
                            "更新後第一順位治療",
                            top_name,
                        )

                    with summary_columns[1]:
                        st.metric(
                            "更新後評分",
                            top_score,
                        )
                    # 不 return：PSG 更新結果下方仍要保留其他臨床資料上傳區。

    # ========================================================
    # 上傳檔案
    # ========================================================

    st.markdown(
        f"### 補充資料上傳｜{selected_display_name}"
    )
    if selected_data_type == "DEVICE_DATA":
        st.info(
            "穿戴式同步模式：可上傳手錶或感測器匯出的 CSV／Excel，"
            "欄位可包含 timestamp、SpO₂、heart_rate、sleep_stage、"
            "position、respiratory_rate、apnea_event。系統會自動彙整後"
            "視為新的監測資料更新 Digital Twin；不會由手錶資料推斷解剖結構、PAP 耐受度或直接重訓正式治療模型。",
            icon=":material/watch:",
        )
    st.caption(
        "支援 PDF、DOCX、Excel、CSV、TXT、JPG、PNG；"
        "可一次選擇多個檔案。DICOM 尚未支援。"
    )

    uploaded_clinical_files = st.file_uploader(
        "拖曳檔案到此，或按 Browse files 選擇檔案",
        type=[
            "pdf",
            "xls",
            "xlsx",
            "doc",
            "docx",
            "csv",
            "txt",
            "jpg",
            "jpeg",
            "png",
        ],
        accept_multiple_files=True,
        key=(
            "clinical_supplement_upload_"
            f"{selected_data_type}"
        ),
        help=(
            "可一次上傳多個檔案。"
            "DICOM 檔案將在後續版本支援。"
        ),
    )

    if uploaded_clinical_files:
        for uploaded_file in uploaded_clinical_files:
            file_size_kb = (
                uploaded_file.size / 1024
            )

            st.write(
                f"📄 `{uploaded_file.name}` "
                f"（{file_size_kb:.1f} KB）"
            )

        st.session_state[
            "clinical_files_ready"
        ] = True

    else:
        st.session_state[
            "clinical_files_ready"
        ] = False

        st.info(
            "尚未選擇補充臨床資料檔案。"
        )

    # ========================================================
    # 解析資料
    # ========================================================

    parser_available = has_parser(
        selected_data_type
    )

    if not parser_available:
        st.info(
            "此資料類型尚未建立自動解析器。"
            "目前可先選擇 PAP 治療資料測試流程。"
        )

    parse_button = st.button(
        (
            "同步並讀取感測器資料"
            if selected_data_type == "DEVICE_DATA"
            else "讀取補充資料"
        ),
        type="primary",
        disabled=(
            not bool(uploaded_clinical_files)
            or not parser_available
        ),
        width="stretch",
        key="parse_clinical_data_button",
    )

    if parse_button:
        st.session_state[
            "clinical_data_parsed"
        ] = False

        st.session_state[
            "clinical_data_confirmed"
        ] = False

        st.session_state[
            "parsed_clinical_data"
        ] = {}

        st.session_state[
            "parsed_clinical_source_files"
        ] = []

        st.session_state[
            "parsed_clinical_warnings"
        ] = []

        st.session_state[
            "parsed_clinical_metadata"
        ] = {}

        if not has_parser(selected_data_type):
            st.error(
                "目前尚未建立此資料類型的解析器："
                f"{selected_display_name}"
            )

        else:
            try:
                parser = get_parser(
                    selected_data_type
                )

                parsed_result = parse_clinical_update(
                    module_id=selected_data_type,
                    parser=parser,
                    uploaded_files=uploaded_clinical_files,
                )

            except Exception as exc:
                st.session_state["clinical_data_parsed"] = True
                st.session_state["parsed_clinical_module_id"] = selected_data_type
                st.session_state["parsed_clinical_data"] = {}
                st.session_state["parsed_clinical_source_files"] = [
                    uploaded_file.name for uploaded_file in uploaded_clinical_files
                ]
                st.session_state["parsed_clinical_warnings"] = [
                    "自動讀取未能完成；此檔案暫時不會送入模型。"
                    f"（{type(exc).__name__}）"
                ]
                st.session_state["parsed_clinical_metadata"] = {}
                st.warning(
                    "檔案已保留，但尚未辨識到可供模型使用的資料。"
                )

            else:
                st.session_state[
                    "clinical_data_parsed"
                ] = True
                st.session_state["parsed_clinical_module_id"] = selected_data_type

                st.session_state[
                    "parsed_clinical_data"
                ] = parsed_result.data

                st.session_state[
                    "parsed_clinical_source_files"
                ] = parsed_result.source_files

                st.session_state[
                    "parsed_clinical_warnings"
                ] = parsed_result.warnings

                st.session_state[
                    "parsed_clinical_metadata"
                ] = parsed_result.metadata

                st.success(
                    f"{parser.display_name}解析流程已完成。"
                )

    # ========================================================
    # 精簡確認
    # ========================================================

    if (
        st.session_state.get("clinical_data_parsed", False)
        and st.session_state.get("parsed_clinical_module_id") == selected_data_type
    ):
        source_files = st.session_state.get(
            "parsed_clinical_source_files",
            [], 
        )

        parser_warnings = st.session_state.get(
            "parsed_clinical_warnings",
            [],
        )

        parser_metadata = st.session_state.get(
            "parsed_clinical_metadata",
            {},
        )

        if parser_warnings:
            for warning_message in parser_warnings:
                st.warning(warning_message)

        parsed_data = st.session_state.get(
            "parsed_clinical_data",
            {},
        )

        extracted_count = sum(
            value not in (None, "", [], {})
            for value in parsed_data.values()
        )
        if extracted_count:
            st.success(
                f"已從檔案辨識 {extracted_count} 個可用臨床欄位。"
            )
        else:
            st.warning(
                "此檔案尚未辨識到可用的結構化臨床欄位，"
                "不會將空白資料送入模型。請確認檔案內容或改用可讀取格式。"
            )

        synthetic_training_mode = st.toggle(
            "合成資料測試模式：本次資料也加入監督式 Challenger 重訓",
            value=True,
            key="synthetic_clinical_training_mode",
            help=(
                "以目前重新計算出的治療分數作為弱／合成標籤，"
                "用來驗證重訓與版本更新流程；不可視為真實臨床療效。"
            ),
        )
        if synthetic_training_mode:
            st.warning(
                "目前為合成測試模式：會建立新模型版本並更新輸出，"
                "但版本將標記 SYNTHETIC_TEST_ONLY，不能升級為臨床模型。"
            )

        confirm_button = st.button(
            "確認更新 Digital Twin 與重新計算建議",
            type="primary",
            disabled=not bool(extracted_count),
            width="stretch",
            key="confirm_clinical_update_button",
        )

        if confirm_button:
            st.session_state[
                "clinical_data_confirmed"
            ] = True

            st.success(
                "資料已確認，可以更新模型與治療建議。"
            )

    # ========================================================
    # 重新評估治療
    # ========================================================

    reevaluate_button = st.button(
        "更新模型並重新產生治療建議",
        type="primary",
        disabled=not st.session_state.get(
            "clinical_data_confirmed",
            False,
        ),
        width="stretch",
        key="reevaluate_treatment_button",
    )

    if reevaluate_button:
        confirmed_data = dict(st.session_state.get(
            "parsed_clinical_data",
            {},
        ))
        confirmed_data["synthetic_test_training"] = bool(
            st.session_state.get("synthetic_clinical_training_mode", True)
        )

        source_files = st.session_state.get(
            "parsed_clinical_source_files",
            [],
        )

        paths = get_treatment_refinement_paths(
            patient_id
        )

        try:
            before_interface = load_json_file(
                paths["interface"]
            )

            with st.spinner(
                "正在更新 ClinicalDecisionData、"
                "加入治療訓練集、重新訓練模型、"
                "重新執行 Treatment Refinement，"
                "並產生新的 Digital Twin 報告..."
            ):
                service_result = refresh_patient(
                    patient_id=patient_id,
                    module_id=selected_data_type,
                    confirmed_data=confirmed_data,
                    source_files=source_files,
                    project_root=PROJECT_ROOT,
                    regenerate_report=True,
                )

            if not paths["interface"].is_file():
                raise FileNotFoundError(
                    "重新評估完成後找不到新的 "
                    "Treatment Refinement Interface："
                    f"{paths['interface']}"
                )

            after_interface = load_json_file(
                paths["interface"]
            )

            comparison = compare_treatment_interfaces(
                before_data=before_interface,
                after_data=after_interface,
            )

            script_results = service_result.get(
                "script_results",
                [],
            )

            execution_log_parts = []

            for script_result in script_results:
                if not isinstance(
                    script_result,
                    dict,
                ):
                    continue

                script_name = script_result.get(
                    "script",
                    "unknown script",
                )

                execution_log_parts.append(
                    f"===== {script_name} ====="
                )

                stdout = script_result.get(
                    "stdout",
                    "",
                )

                stderr = script_result.get(
                    "stderr",
                    "",
                )

                if stdout:
                    execution_log_parts.append(
                        str(stdout).rstrip()
                    )

                if stderr:
                    execution_log_parts.append(
                        "[stderr]\n"
                        + str(stderr).rstrip()
                    )

            st.session_state[
                "reevaluation_result"
            ] = {
                "success": True,
                "patient_id": patient_id,
                "module": service_result.get(
                    "module_id",
                    selected_data_type,
                ),
                "clinical_data_path":
                    service_result.get(
                        "clinical_data_path",
                        "",
                    ),
                "data_version":
                    service_result.get(
                        "version_after"
                    ),
                "version_before":
                    service_result.get(
                        "version_before"
                    ),
                "changed_fields":
                    service_result.get(
                        "changed_fields",
                        {},
                    ),
                "applied_fields": sorted(
                    service_result.get(
                        "changed_fields",
                        {},
                    ).keys()
                ),
                "comparison": comparison,
                "recommendation_changes":
                    service_result.get(
                        "recommendation_changes",
                        [],
                    ),
                "execution_log": "\n\n".join(
                    execution_log_parts
                ),
                "completed_at":
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),
            }

            st.session_state[
                "clinical_data_confirmed"
            ] = False

            st.session_state.pop(
                "reevaluation_error",
                None,
            )

            # 重新收集報告，確保下載區使用新產生的版本。
            refreshed_report_paths = (
                collect_report_paths(
                    patient_id=patient_id,
                )
            )

            st.session_state[
                "report_paths"
            ] = {
                extension: str(path)
                for extension, path
                in refreshed_report_paths.items()
            }

            st.rerun()

        except Exception as exc:
            st.session_state[
                "reevaluation_error"
            ] = str(exc)

            st.error(
                "重新評估治療建議失敗。"
            )
            st.exception(exc)

    show_reevaluation_result()


def show_clinical_data_update_section(
    interface_data: dict,
    patient_id: str,
) -> None:
    """Render the full update workflow behind one concise section title."""
    st.header("6. 補充臨床資料重新評估")
    with st.expander(
        "展開補充資料上傳、醫師確認與模型重新評估",
        expanded=False,
        icon=":material/upload_file:",
    ):
        _show_clinical_data_update_section_content(interface_data, patient_id)

# ============================================================
# Session 預設值
# ============================================================

if "analysis_completed" not in st.session_state:
    st.session_state["analysis_completed"] = False

if "analysis_failed" not in st.session_state:
    st.session_state["analysis_failed"] = False

if "pipeline_log" not in st.session_state:
    st.session_state["pipeline_log"] = ""

if "report_paths" not in st.session_state:
    st.session_state["report_paths"] = {}

# Browser refresh clears Streamlit session memory. Reload the last completed
# patient from a small disk pointer; patient data, learned models and reports
# themselves are already persisted on disk.
if (
    not IS_CLOUD_RUNTIME
    and not st.session_state.get("saved_patient_id")
):
    restored_patient_id = restore_active_patient()
    if restored_patient_id:
        st.session_state["saved_patient_id"] = restored_patient_id
        restored_sources = restore_saved_source_files(restored_patient_id)
        if restored_sources:
            restored_folder, restored_files = restored_sources
            st.session_state["saved_patient_folder"] = str(restored_folder)
            st.session_state["saved_files"] = restored_files
        st.session_state["analysis_completed"] = True
        st.session_state["analysis_failed"] = False
        st.session_state["report_paths"] = {
            extension: str(path)
            for extension, path in collect_report_paths(restored_patient_id).items()
        }


# ============================================================
# 網頁標題
# ============================================================

st.markdown(
    '<div class="dt-hero"><h1>Sleep Digital Twin</h1>'
    '<p>個人化睡眠分析、持續學習與治療決策支援</p></div>',
    unsafe_allow_html=True,
)
st.caption("依序完成患者資料上傳、模型分析與臨床補充；所有更新皆保留可稽核紀錄。")

st.divider()


# ============================================================
# 上傳區域
# ============================================================

st.markdown('<span class="dt-step">步驟 1 · 原始資料</span>', unsafe_allow_html=True)
st.header("上傳患者資料")

edf_source_mode = st.radio(
    "EDF 來源",
    options=["直接上傳 EDF", "從 Google Drive 下載 EDF（適用大於 100 MB）"],
    horizontal=True,
    key="edf_source_mode",
)

edf_file = None
drive_edf_link = ""
drive_edf_original_name = ""
if edf_source_mode == "直接上傳 EDF":
    edf_file = st.file_uploader(
        "1. PSG 生理訊號（EDF）",
        type=["edf"],
        key="edf_upload",
        help=(
            "例如：20201014T221256 - d25c6_EDF.edf。"
            "系統會從 EDF 檔名取得 Patient ID。"
        ),
    )
else:
    st.info(
        "適用於公開臨時網址無法直接上傳的大型 EDF。請先將 EDF 上傳至 Google Drive，"
        "設定為「知道連結的使用者可檢視」，再貼上分享連結。網站只會從 Google Drive 下載該 EDF；"
        "請不要貼其他網站的網址。"
    )
    drive_edf_link = st.text_input(
        "1. Google Drive EDF 分享連結",
        placeholder="https://drive.google.com/file/d/.../view?usp=sharing",
        key="drive_edf_link",
    ).strip()
    drive_edf_original_name = st.text_input(
        "EDF 原始檔名",
        placeholder="例如：20201014T221256 - d25c6_EDF.edf",
        key="drive_edf_original_name",
        help="檔名用來辨識 Patient ID；必須與 Drive 中的 EDF 原始檔名一致。",
    ).strip()

stage_file = st.file_uploader(
    "2. 睡眠分期（Stage Excel）",
    type=["xls", "xlsx"],
    key="stage_upload",
)

event_file = st.file_uploader(
    "3. 呼吸事件標記（Event Grid Excel）",
    type=["xls", "xlsx"],
    key="event_upload",
)

def get_uploaded_file_signature(uploaded_file):
    if uploaded_file is None:
        return None

    return {
        "name": uploaded_file.name,
        # UploadedFile.size is metadata supplied by Streamlit.  Do not call
        # getvalue() here: EDF files can be very large, and reading the whole
        # payload merely to detect a widget change can duplicate hundreds of
        # megabytes on every rerun before the other two files are uploaded.
        "size": uploaded_file.size,
    }

current_upload_signature = {
    "edf": (
        get_uploaded_file_signature(edf_file)
        if edf_source_mode == "直接上傳 EDF"
        else {
            "source": "google_drive",
            "link": drive_edf_link,
            "original_name": drive_edf_original_name,
        }
    ),
    "stage": get_uploaded_file_signature(stage_file),
    "event": get_uploaded_file_signature(event_file),
}

previous_upload_signature = st.session_state.get(
    "upload_signature"
)

if (
    previous_upload_signature is not None
    and current_upload_signature != previous_upload_signature
):
    # A file-uploader widget can briefly become empty during a Streamlit
    # rerun.  That UI-only transition must not hide a completed patient's
    # original result dashboard.  The saved result state is cleared later,
    # only when a new patient's three source files are actually saved.
    st.session_state.pop("reevaluation_result", None)
    st.session_state.pop("reevaluation_error", None)

st.session_state["upload_signature"] = (
    current_upload_signature
)

all_files_uploaded = (
    (
        edf_file is not None
        if edf_source_mode == "直接上傳 EDF"
        else bool(drive_edf_link and drive_edf_original_name)
    )
    and stage_file is not None
    and event_file is not None
)

detected_patient_id: str | None = None
patient_id_error: str | None = None
companion_id_mismatch: str | None = None

edf_name_for_patient_id = (
    edf_file.name
    if edf_file is not None
    else drive_edf_original_name
)
if edf_name_for_patient_id:
    try:
        detected_patient_id = extract_patient_id_from_edf(
            edf_name_for_patient_id
        )
    except Exception as exc:
        patient_id_error = str(exc)

if patient_id_error:
    st.error(patient_id_error)

elif detected_patient_id:
    st.success(
        "已從 EDF 檔名取得 Patient ID。"
    )

    st.code(
        detected_patient_id,
        language=None,
    )

if detected_patient_id:
    companion_ids = {
        "Stage Excel": extract_patient_id_from_companion(stage_file.name) if stage_file else None,
        "Event Grid Excel": extract_patient_id_from_companion(event_file.name) if event_file else None,
    }
    mismatches = {
        label: value for label, value in companion_ids.items()
        if value and value != detected_patient_id
    }
    if mismatches:
        details = "；".join(f"{label}={value}" for label, value in mismatches.items())
        companion_id_mismatch = (
            f"三個檔案的患者ID不一致：EDF={detected_patient_id}；{details}。"
            "請選擇同一位患者的EDF、Stage與Event Grid，避免錯誤標籤進入模型。"
        )
        st.error(companion_id_mismatch)

if all_files_uploaded:
    st.success(
        "EDF、Stage Excel、Event Grid Excel 均已準備完成。"
    )
else:
    st.info(
        "請準備 EDF、Stage Excel 與 Event Grid Excel 三個必要檔案。"
    )

st.divider()


# ============================================================
# 儲存上傳資料
# ============================================================

st.markdown('<span class="dt-step">步驟 2 · 安全保存</span>', unsafe_allow_html=True)
st.header("儲存上傳資料")

patient_folder: Path | None = None
folder_exists = False

if detected_patient_id:
    patient_folder = (
        INCOMING_ROOT / detected_patient_id
    )
    folder_exists = patient_folder.exists()

overwrite_existing = False

if folder_exists and patient_folder is not None:
    st.warning(
        "此 Patient ID 的上傳資料夾已存在：\n\n"
        f"`{patient_folder}`"
    )

    overwrite_existing = st.checkbox(
        "為此患者儲存新的上傳版本",
        value=False,
        key="overwrite_existing_upload",
    )
    st.caption(
        "新版 EDF、Stage 與 Event Grid 會使用版本化檔名；"
        "舊檔若仍被 Windows 鎖定會安全保留，不會再造成儲存失敗。"
    )

save_disabled = not (
    all_files_uploaded
    and detected_patient_id
    and not patient_id_error
    and not companion_id_mismatch
)

st.markdown(
    """
    <style>
    :root { --dt-navy:#24364b; --dt-blue:#1769aa; --dt-border:#dfe6ed; --dt-muted:#647486; }
    .stApp { background:#f8fafc; }
    .main .block-container { max-width:1120px; padding-top:1.5rem; padding-bottom:4rem; }
    h1,h2,h3 { color:var(--dt-navy); letter-spacing:-.025em; }
    h1 { font-weight:760!important; }
    [data-testid="stFileUploader"] { background:#fff; border:1px solid var(--dt-border); border-radius:10px; padding:.55rem .8rem .15rem; box-shadow:none; }
    [data-testid="stFileUploaderDropzone"] { border-radius:10px; background:#f7fbff; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--dt-border); border-radius:10px; padding:.8rem .95rem; box-shadow:none; }
    [data-testid="stMetricLabel"] { color:var(--dt-muted); }
    [data-testid="stExpander"] { background:#fff; border:1px solid var(--dt-border)!important; border-radius:10px!important; box-shadow:none; overflow:hidden; }
    [data-testid="stAlert"] { border-radius:12px; border-width:1px; }
    .stButton>button,.stDownloadButton>button { border-radius:10px; min-height:2.7rem; font-weight:650; }
    .stButton>button[kind="primary"] { background:#1769aa; border:0; box-shadow:none; }
    [data-testid="stDataFrame"] { border:1px solid var(--dt-border); border-radius:12px; overflow:hidden; }
    hr { border-color:#e8eef4!important; margin:1.8rem 0!important; }
    .dt-hero { padding:1.1rem 1.25rem; margin-bottom:1rem; border-radius:10px; color:#fff; background:#243f5d; box-shadow:none; border-left:5px solid #2f91d1; }
    .dt-hero h1 { color:#fff; margin:0 0 .2rem; font-size:1.75rem; }
    .dt-hero p { margin:0; color:#dce9f4; font-size:.95rem; }
    .dt-step { display:inline-flex; padding:.24rem .58rem; color:#185c96; background:#eaf3fa; border-radius:6px; font-size:.78rem; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

if folder_exists and not overwrite_existing:
    save_disabled = True

save_button = st.button(
    "儲存上傳資料",
    type="primary",
    disabled=save_disabled,
    width="stretch",
)

if save_button:
    assert detected_patient_id is not None
    assert stage_file is not None
    assert event_file is not None

    clear_analysis_session()

    try:
        downloaded_path: Path | None = None
        downloaded_edf: DownloadedEDFFile | None = None
        try:
            if edf_source_mode == "從 Google Drive 下載 EDF（適用大於 100 MB）":
                with st.spinner("正在由 Google Drive 下載 EDF 到分析電腦，下載完成後才會儲存與分析…"):
                    downloaded_path = download_google_drive_edf(
                        drive_edf_link,
                        drive_edf_original_name,
                    )
                downloaded_edf = DownloadedEDFFile(
                    downloaded_path,
                    drive_edf_original_name,
                )
                edf_source = downloaded_edf
            else:
                assert edf_file is not None
                edf_source = edf_file

            patient_folder, saved_paths = save_patient_uploads(
                patient_id=detected_patient_id,
                edf_file=edf_source,
                stage_file=stage_file,
                event_file=event_file,
                overwrite=overwrite_existing,
            )
        finally:
            if downloaded_edf is not None:
                downloaded_edf.close()
            if downloaded_path is not None:
                downloaded_path.unlink(missing_ok=True)

    except Exception as exc:
        st.error(
            "儲存患者資料失敗。"
        )
        st.exception(exc)

    else:
        st.session_state[
            "saved_patient_id"
        ] = detected_patient_id

        persist_active_patient(detected_patient_id)

        st.session_state[
            "saved_patient_folder"
        ] = str(patient_folder)

        st.session_state[
            "saved_files"
        ] = {
            label: str(path)
            for label, path in saved_paths.items()
        }

        st.success(
            "患者上傳資料已成功儲存。請先查看下方的資料處理摘要，再開始模型分析。"
        )

show_saved_patient_information()

st.divider()


# ============================================================
# 資料處理與醫師檢視
# ============================================================
saved_patient_id = st.session_state.get("saved_patient_id")
saved_files = st.session_state.get("saved_files", {})
review_confirmed = False
if saved_patient_id and isinstance(saved_files, dict) and saved_files:
    review_confirmed = show_data_processing_review(saved_patient_id, saved_files)
    if review_confirmed:
        st.session_state["data_processing_review_confirmed_patient_id"] = saved_patient_id
        st.session_state["pipeline_autorun_patient_id"] = saved_patient_id
elif saved_patient_id:
    st.warning("找不到本次保存的三個來源檔案，請重新上傳後再進行資料處理檢視。")

st.divider()


# ============================================================
# 執行分析 Pipeline
# ============================================================

st.markdown('<span class="dt-step">步驟 4 · 模型運算</span>', unsafe_allow_html=True)
st.header("執行模型分析")

if not saved_patient_id:
    st.info(
        "請先上傳並儲存患者資料。"
    )

if not PIPELINE_SCRIPT.is_file():
    st.error(
        "找不到 run_patient_pipeline.py：\n\n"
        f"`{PIPELINE_SCRIPT}`"
    )

run_disabled = (
    not saved_patient_id
    or not PIPELINE_SCRIPT.is_file()
    or st.session_state.get("data_processing_review_confirmed_patient_id") != saved_patient_id
)

run_button = st.button(
    "開始分析",
    type="primary",
    disabled=run_disabled,
    width="stretch",
)

autorun_patient_id = st.session_state.get("pipeline_autorun_patient_id")
run_requested = bool(run_button or (
    saved_patient_id and autorun_patient_id == saved_patient_id
))

if run_requested:
    assert saved_patient_id is not None

    # Consume before starting.  A failure stays visible and cannot create an
    # infinite rerun loop; the user can explicitly press Start Analysis again.
    st.session_state.pop("pipeline_autorun_patient_id", None)

    st.session_state[
        "analysis_failed"
    ] = False

    st.session_state[
        "pipeline_log"
    ] = ""

    st.session_state["analysis_running"] = True

    st.caption("Pipeline 正在執行；詳細紀錄完成後可由下方箭頭展開。")

    try:
        return_code, pipeline_log = run_pipeline(
            patient_id=saved_patient_id,
        )

    except Exception as exc:
        st.session_state["analysis_running"] = False
        st.session_state[
            "analysis_failed"
        ] = True

        st.error(
            "無法啟動分析 Pipeline。"
        )

        st.exception(exc)

    else:
        st.session_state["analysis_running"] = False
        st.session_state[
            "pipeline_return_code"
        ] = return_code

        st.session_state[
            "pipeline_log"
        ] = pipeline_log

        # The live console is rendered in a temporary placeholder and is
        # destroyed by the clean rerun below.  Keep an on-disk copy so the
        # same complete record remains available after reruns and refreshes.
        persist_pipeline_log(saved_patient_id, pipeline_log)

        if return_code == 0:
            report_paths = collect_report_paths(
                patient_id=saved_patient_id,
            )

            st.session_state[
                "report_paths"
            ] = {
                extension: str(path)
                for extension, path
                in report_paths.items()
            }

            st.session_state[
                "analysis_completed"
            ] = True

            st.session_state[
                "analysis_failed"
            ] = False

            st.success(
                "Sleep Digital Twin 分析完成。"
            )

            # Start a clean render pass after the blocking Pipeline returns.
            # This restores the complete original result area (reports,
            # treatment ranking, clinical details and wearable section)
            # instead of leaving the page at the button's execution frame.
            st.session_state["analysis_just_completed"] = True
            st.rerun()

        else:
            st.session_state[
                "analysis_failed"
            ] = True

            # Do not discard the last successful report merely because the
            # replacement run failed. The failure is recorded separately.
            st.session_state["analysis_completed"] = bool(
                st.session_state.get("report_paths")
                or collect_report_paths(patient_id=saved_patient_id)
            )

            show_pipeline_failure_details(pipeline_log)


# ============================================================
# 顯示上一次執行結果
# ============================================================

# Keep the next-epoch result beside the analysis controls.  It used to be at
# the bottom of the page after unrelated sections, which made the patient-
# specific causal forecast difficult to find.
render_osa_next_epoch_model(
    PROJECT_ROOT,
    str(st.session_state.get("saved_patient_id") or detected_patient_id or ""),
)

st.divider()

# Session state is per browser tab and may be lost on refresh. Restore the
# patient's latest durable execution log before rendering the result area.
if saved_patient_id and not st.session_state.get("pipeline_log"):
    st.session_state["pipeline_log"] = load_persisted_pipeline_log(
        saved_patient_id
    )

if (
    st.session_state.get("pipeline_log")
    and not run_button
):
    with st.expander(
        "最近一次 Pipeline 執行紀錄（分析完成後仍會保留）",
        expanded=True,
    ):
        st.caption(
            "以下為目前患者最近一次完整分析紀錄；可使用箭頭收合，"
            "重新整理網頁後仍可再次查看。"
        )
        st.code(
            st.session_state["pipeline_log"],
            language="text",
        )

if st.session_state.get(
    "analysis_failed"
):
    return_code = st.session_state.get(
        "pipeline_return_code"
    )

    st.error(
        "最近一次分析失敗。"
        + (
            f" Return Code：{return_code}"
            if return_code is not None
            else ""
        )
    )

    previous_pipeline_log = st.session_state.get(
        "pipeline_log",
        "",
    )
    if previous_pipeline_log and not run_button:
        show_pipeline_failure_details(previous_pipeline_log)


# Reconcile display state with persisted output files.  Session state is only
# browser memory; generated model outputs on disk are the durable source of
# truth.  A refresh, uploader reset or server rerun must therefore restore the
# complete original result dashboard automatically.
if saved_patient_id:
    persisted_report_paths = collect_report_paths(
        patient_id=saved_patient_id,
    )
    if persisted_report_paths:
        st.session_state["analysis_completed"] = True
        st.session_state["report_paths"] = {
            extension: str(path)
            for extension, path in persisted_report_paths.items()
        }
        # Clear a stale UI failure left by an earlier run only when the full
        # durable output set now exists, including the final tonight-risk
        # step and treatment interface.  This lets an externally verified
        # successful rerun restore the original result dashboard immediately.
        tonight_summary = (
            INFERENCE_ROOT
            / saved_patient_id
            / "tonight_apnea_risk"
            / "tonight_risk_summary.json"
        )
        treatment_interface = find_treatment_interface_file(
            patient_id=saved_patient_id,
        )
        if tonight_summary.is_file() and treatment_interface is not None:
            st.session_state["analysis_failed"] = False
            st.session_state["pipeline_return_code"] = 0

if st.session_state.pop("analysis_just_completed", False):
    st.success("Sleep Digital Twin 分析完成，以下為完整分析結果。")


# ============================================================
# 報告下載與原版完整結果
# ============================================================

if st.session_state.get(
    "analysis_completed"
):
    st.divider()
    st.markdown('<span class="dt-step">步驟 5 · 結果與報告</span>', unsafe_allow_html=True)
    st.header("分析結果")

    if saved_patient_id:
        show_processed_data_review(saved_patient_id)

    saved_report_paths = {
        extension: Path(path_text)
        for extension, path_text
        in st.session_state.get(
            "report_paths",
            {},
        ).items()
        if Path(path_text).is_file()
    }

    if not saved_report_paths and saved_patient_id:
        saved_report_paths = collect_report_paths(
            patient_id=saved_patient_id,
        )

    show_download_buttons(
        report_paths=saved_report_paths,
    )

    if saved_patient_id:
        sleepfm_path = (
            INFERENCE_ROOT / saved_patient_id / "sleepfm" / "sleepfm_disease_risk.json"
        )
        with st.expander(
            "SleepFM 多模態疾病研究風險",
            expanded=False,
            icon=":material/neurology:",
        ):
            if not sleepfm_path.is_file():
                st.info("本次尚未產生 SleepFM 結果；重新執行完整分析後會在此顯示。")
            else:
                sleepfm_data = load_json_file(sleepfm_path)
                sleepfm_status = sleepfm_data.get("status")
                if sleepfm_status == "completed":
                    st.success(
                        f"已分析 {sleepfm_data.get('five_second_tokens', 0)} 個 5 秒片段，"
                        f"約 {sleepfm_data.get('duration_hours', 0)} 小時。"
                    )
                    st.warning(
                        "下表是 SleepFM CoxPH 模型的相對風險排序，不是患病機率、確診結果或處方；"
                        "需由醫師結合病史、檢查與外部驗證判讀。"
                    )
                    sleepfm_rows = sleepfm_data.get("ranking") or []
                    if sleepfm_rows:
                        st.dataframe(sleepfm_rows, width="stretch", hide_index=True)
                elif sleepfm_status == "skipped_missing_modalities":
                    missing_text = "、".join(sleepfm_data.get("missing_modalities") or [])
                    st.warning(
                        f"未執行疾病排序：此 PSG 缺少 SleepFM 必要模態（{missing_text}）。"
                        "系統不會用假訊號補值。"
                    )
                else:
                    st.error(
                        "SleepFM 執行未完成："
                        + str(sleepfm_data.get("error_message") or "未知錯誤")
                    )
                st.caption(
                    "模型：SleepFM（BAS／RESP／EKG／EMG）｜研究用途｜"
                    "授權：CC BY-NC 4.0；醫院正式部署前須確認授權與臨床驗證。"
                )

        positional_endotype_path = (
            INFERENCE_ROOT / saved_patient_id / "positional_endotype_research"
            / "positional_endotype_summary.json"
        )
        with st.expander(
            "ERJ 2024 姿勢型 OSA 與生理 Endotype",
            expanded=False,
            icon=":material/airline_seat_flat:",
        ):
            if not positional_endotype_path.is_file():
                st.info("重新執行完整分析後，會在此顯示姿勢型 OSA 與 Endotype 資料完整度。")
            else:
                positional_data = load_json_file(positional_endotype_path)
                phenotype_labels = {
                    "SUPINE_PREDOMINANT_OSA": "仰睡優勢型 OSA（spOSA）",
                    "NON_POSITIONAL_OSA": "非姿勢型 OSA（npOSA）",
                    "INDETERMINATE": "目前資料不足，無法判定",
                }
                metrics = positional_data.get("metrics") or {}
                phenotype = positional_data.get("positional_phenotype", "INDETERMINATE")
                st.subheader(phenotype_labels.get(phenotype, phenotype))
                st.write(positional_data.get("interpretation", ""))
                metric_columns = st.columns(4)
                values = [
                    ("仰睡 AHI", metrics.get("supine_ahi")),
                    ("側睡 AHI", metrics.get("lateral_ahi")),
                    ("仰睡／側睡比值", metrics.get("supine_to_lateral_ahi_ratio")),
                    ("總睡眠時間", metrics.get("total_sleep_minutes")),
                ]
                for column, (label, value) in zip(metric_columns, values):
                    column.metric(label, "無資料" if value is None else str(round(value, 2)))
                st.caption(
                    "論文研究定義：仰睡 AHI／非仰睡 AHI > 2；"
                    "仰睡與側睡必須各至少 30 分鐘，且姿勢感測器須完成解剖方向校正。"
                )
                failed_gates = positional_data.get("failed_quality_gates") or []
                pending_gates = positional_data.get("not_evaluated_quality_gates") or []
                if failed_gates or pending_gates:
                    st.warning(
                        "尚未完全符合研究計算條件。缺少／未通過："
                        + "、".join(failed_gates + pending_gates)
                    )
                st.markdown("**四種 PUP 生理 Endotype 狀態**")
                endotype_rows = []
                endotype_names = {
                    "collapsibility": "上呼吸道塌陷性",
                    "upper_airway_muscle_compensation": "上呼吸道肌肉代償",
                    "arousal_threshold": "覺醒閾值",
                    "loop_gain": "Loop gain",
                }
                for key, item in (positional_data.get("pup_endotypes") or {}).items():
                    endotype_rows.append({
                        "生理特徵": endotype_names.get(key, key),
                        "目前狀態": item.get("status"),
                        "研究型 Proxy": item.get("proxy_score"),
                        "仍需方法": item.get("required_method") or item.get("warning"),
                    })
                if endotype_rows:
                    st.dataframe(endotype_rows, width="stretch", hide_index=True)
                st.error(
                    "本區是研究型決策支援，不是診斷。系統不會把既有 proxy 假裝成 PUPpy 的正式 Endotype。"
                )
                st.download_button(
                    "下載本次姿勢／Endotype JSON",
                    data=positional_endotype_path.read_bytes(),
                    file_name=f"{saved_patient_id}_positional_endotype.json",
                    mime="application/json",
                    key=f"download_positional_endotype_{saved_patient_id}",
                )

        jsr_pap_path = (
            INFERENCE_ROOT / saved_patient_id / "jsr2024_pap_endotype_research"
            / "jsr2024_pap_endotype_summary.json"
        )
        with st.expander(
            "JSR 2024 PAP 依從性與生理內型證據",
            expanded=False,
            icon=":material/respiratory_rate:",
        ):
            if not jsr_pap_path.is_file():
                st.info("重新執行完整分析後，會在此顯示 PAP 論文可用特徵與 PUP 資料完整度。")
            else:
                jsr_data = load_json_file(jsr_pap_path)
                observed = jsr_data.get("observable_psg_features") or {}
                eligibility = jsr_data.get("study_cohort_eligibility") or {}
                st.write(jsr_data.get("interpretation", ""))
                metric_columns = st.columns(4)
                metric_values = [
                    ("AHI", observed.get("ahi")),
                    ("中樞型指數", observed.get("central_apnea_index")),
                    ("阻塞型指數", observed.get("obstructive_apnea_index")),
                    ("覺醒指數", observed.get("arousal_index")),
                ]
                for column, (label, value) in zip(metric_columns, metric_values):
                    column.metric(label, "無資料" if value is None else f"{float(value):.2f}")
                if eligibility.get("matches_key_psg_criteria"):
                    st.success("符合該研究的主要 PSG 篩選條件：AHI ≥ 15，且中樞型指數 < 5。")
                else:
                    st.warning("不符合或尚無足夠資料確認該研究的主要 PSG 篩選條件。")
                if jsr_data.get("exact_pup_endotypes_available"):
                    st.success("已具備並驗證正式 PUP/PUPpy 生理內型。")
                else:
                    st.warning(
                        "尚未具備正式 PUP/PUPpy 四種生理內型；目前只顯示可觀察 PSG 指標，"
                        "不會用代理值直接改動治療分數。"
                    )
                st.caption(
                    "Cheng 等人，Journal of Sleep Research 2024；DOI 10.1111/jsr.13999。"
                    "論文用於特徵設計與驗證框架，不等同個別患者診斷。"
                )
                st.download_button(
                    "下載本次 JSR 2024 PAP 證據 JSON",
                    data=jsr_pap_path.read_bytes(),
                    file_name=f"{saved_patient_id}_jsr2024_pap_evidence.json",
                    mime="application/json",
                    key=f"download_jsr2024_pap_{saved_patient_id}",
                )

    if saved_patient_id:
        treatment_interface_path = (
            find_treatment_interface_file(
                patient_id=saved_patient_id,
            )
        )

        st.divider()

        if treatment_interface_path is None:
            st.warning(
                "尚未找到 Treatment Refinement "
                "Interface JSON。\n\n"
                "預期檔案：\n\n"
                "`data/inference/"
                f"{saved_patient_id}/"
                "treatment_refinement/"
                "treatment_refinement_interface.json`"
            )

        else:
            try:
                treatment_interface_data = (
                    load_json_file(
                        treatment_interface_path
                    )
                )

            except Exception as exc:
                st.error(
                    "讀取 Treatment Refinement "
                    "Interface 失敗。"
                )
                st.exception(exc)

            else:
                st.success(
                    "更新後的治療評估結果已重新載入。"
                )

                show_treatment_refinement(
                    interface_data=treatment_interface_data,
                    interface_path=treatment_interface_path,
                )

                candidate_list = (
                    treatment_interface_data.get("treatments")
                    or []
                )

                if not isinstance(candidate_list, list):
                    candidate_list = []

                if not isinstance(candidate_list, list):
                    candidate_list = []

                top_candidate = (
                    candidate_list[0]
                    if candidate_list
                    else {}
                )

                top_name = top_candidate.get(
                    "treatment_label",
                    "-"
                )

                top_score = top_candidate.get(
                    "final_score",
                    "-"
                )

                summary_columns = st.columns(2)

                with summary_columns[0]:
                    st.metric(
                        "更新後第一順位治療",
                        str(top_name),
                    )

                with summary_columns[1]:
                    st.metric(
                        "更新後評分",
                        str(top_score),
                    )

                follow_up_comparison = (
                    treatment_interface_data.get(
                        "follow_up_comparison",
                        {},
                    )
                )

                if isinstance(follow_up_comparison, dict):
                    comparison_summary = (
                        follow_up_comparison.get("summary")
                        or follow_up_comparison.get(
                            "comparison_summary"
                        )
                    )

                    with st.expander(
                        "與前次結果比較",
                        expanded=False,
                        icon=":material/compare_arrows:",
                    ):
                        if comparison_summary:
                            st.write(comparison_summary)
                        st.json(
                            follow_up_comparison,
                            expanded=False,
                        )

                st.info(
                    "完整治療排名、前置條件與報告下載，"
                    "請查看本頁上方的分析結果區。"
                )

                st.divider()

                show_clinical_data_update_section(
                    interface_data=treatment_interface_data,
                    patient_id=saved_patient_id,
                )

        st.divider()
        st.info(
            "『今晚睡眠呼吸事件負荷風險』會在下方穿戴式中心，依白天累積資料、病人人設與既往 PSG 更新。"
        )


# ============================================================
# 重新開始
# ============================================================

st.divider()

render_wearable_realtime_center(
    str(st.session_state.get("saved_patient_id") or detected_patient_id or "")
)
