"""Readable and auditable PSG data-processing summaries for the web UI.

This module deliberately reports *what was used* and *what was not used*.
It does not calculate a diagnosis or silently admit an uploaded patient to a
treatment-outcome training set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import json

from src.io_utils.edf_reader import EDFReader
from src.preprocessing.time_alignment import TimeAligner
from src.respiratory.event_reader import EventReader
from src.staging.stage_reader import StageReader


def _file_row(label: str, path: Path) -> dict[str, Any]:
    return {
        "資料檔": label,
        "檔名": path.name,
        "大小(MB)": round(path.stat().st_size / 1024 / 1024, 2) if path.exists() else None,
        "用途": {
            "EDF": "原始PSG生理波形；擷取呼吸、血氧、心率、睡眠等逐epoch特徵",
            "Stage": "每30秒睡眠分期；計算睡眠結構，並與PSG時間軸對齊",
            "Event Grid": "技師標記的呼吸事件；用於事件統計與OSA事件標籤",
        }.get(label, "來源檔"),
    }


def build_uploaded_data_review(paths: dict[str, str | Path]) -> dict[str, Any]:
    """Read the saved source triplet and create a pre-analysis clinical preview."""
    resolved = {name: Path(value) for name, value in paths.items()}
    result: dict[str, Any] = {
        "files": [_file_row(name, path) for name, path in resolved.items()],
        "warnings": [],
        "stage_counts": {},
        "event_counts": {},
        "alignment": {},
        "edf": {},
    }
    try:
        edf = EDFReader(resolved["EDF"])
        edf.load()
        result["edf"] = {
            "channel_count": len(edf.raw.ch_names),
            "sampling_rate_hz": round(edf.sampling_rate, 2),
            "duration_minutes": round(edf.duration_seconds / 60, 1),
            "start_time": str(edf.original_measurement_start),
            "channels": list(edf.raw.ch_names),
        }
        stages = StageReader(resolved["Stage"]).load()
        events = EventReader(resolved["Event Grid"]).load()
        stage_counts = stages["stage"].value_counts().reindex(
            ["W", "N1", "N2", "N3", "REM"], fill_value=0
        )
        result["stage_counts"] = {str(key): int(value) for key, value in stage_counts.items()}
        sleep_epochs = int(len(stages) - stage_counts.get("W", 0))
        result["sleep"] = {
            "epoch_count": int(len(stages)),
            "recording_minutes": round(len(stages) * 0.5, 1),
            "estimated_sleep_minutes": round(sleep_epochs * 0.5, 1),
            "rem_minutes": round(int(stage_counts.get("REM", 0)) * 0.5, 1),
        }
        event_counts = events["event_type"].value_counts()
        result["event_counts"] = {str(key): int(value) for key, value in event_counts.items()}
        osa_events = int(event_counts.get("OBSTRUCTIVE_APNEA", 0) + event_counts.get("HYPOPNEA", 0))
        result["events"] = {
            "total": int(len(events)),
            "osa_label_events": osa_events,
            "central_or_mixed": int(event_counts.get("CENTRAL_APNEA", 0) + event_counts.get("MIXED_APNEA", 0)),
            "arousal": int(event_counts.get("AROUSAL", 0)),
        }
        aligned_stages = TimeAligner(edf).align_stages(stages)
        usable_epochs = int(aligned_stages["usable_for_stage_training"].sum())
        overlap_epochs = int(aligned_stages["overlaps_edf"].sum())
        if usable_epochs == len(stages):
            alignment_status = "TIME_RANGE_COMPATIBLE"
            explanation = "所有30秒睡眠分期皆完整落在EDF訊號範圍內，可進入後續時間對齊。"
        elif overlap_epochs:
            alignment_status = "PARTIAL_STAGE_COVERAGE"
            explanation = "部分睡眠分期位於EDF訊號外；正式分析只會使用完整重疊的30秒epoch。"
        else:
            alignment_status = "NO_STAGE_OVERLAP"
            explanation = "Stage與EDF沒有可用重疊區段，正式分析會停止，避免錯誤標籤進入模型。"
        result["alignment"] = {
            "status": alignment_status,
            "explanation": explanation,
            "overlap_minutes": round(overlap_epochs * 0.5, 1),
            "maximum_stage_coverage": round(usable_epochs / len(stages) * 100, 1) if len(stages) else 0.0,
        }
        if alignment_status != "TIME_RANGE_COMPATIBLE":
            result["warnings"].append("EDF、Stage時間軸需在正式分析時進一步對齊；未通過品質檢查不會送入模型。")
        if not osa_events:
            result["warnings"].append("Event Grid未找到阻塞型apnea或hypopnea；可完成檔案檢查，但不能建立OSA事件標籤。")
    except Exception as exc:  # visible review failure, never hidden
        result["warnings"].append(f"原始資料預覽尚未完整解析：{exc}")
    return result


def build_processed_data_review(project_root: str | Path, patient_id: str) -> dict[str, Any]:
    """Summarise actual outputs after the pipeline, not an assumed workflow."""
    root = Path(project_root)
    processed = root / "data" / "processed" / patient_id
    inference = root / "data" / "inference" / patient_id
    stages_path = processed / "stages_aligned.csv"
    events_path = processed / "events_aligned.csv"
    features_path = inference / "inference_features.csv"
    result: dict[str, Any] = {"available": False, "warnings": [], "training_status": []}
    try:
        stages = pd.read_csv(stages_path)
        events = pd.read_csv(events_path)
        features = pd.read_csv(features_path) if features_path.exists() else pd.DataFrame()
        event_counts = events.get("event_type", pd.Series(dtype=str)).value_counts()
        stage_counts = stages.get("stage", pd.Series(dtype=str)).value_counts()
        result.update({
            "available": True,
            "aligned_epochs": int(len(stages)),
            "feature_epochs": int(len(features)),
            "stage_counts": {str(key): int(value) for key, value in stage_counts.items()},
            "event_counts": {str(key): int(value) for key, value in event_counts.items()},
            "feature_columns": list(features.columns),
            "training_status": [
                {
                    "模型／資料集": "單向LSTM：前7個epoch預測下一個30秒OSA事件",
                    "本次資料用途": "若對齊後特徵及Event Grid完整，可作為新增研究資料；患者層級切分後才可進訓練／驗證／鎖定測試。",
                    "不使用的資料": "中央型、混合型apnea不作為OSA-only目標；未對齊或品質失敗資料不納入。",
                },
                {
                    "模型／資料集": "治療推薦模型",
                    "本次資料用途": "僅更新此患者的治療評估特徵與規則解釋。",
                    "不使用的資料": "新患者單次PSG不是治療療效真值；必須有治療前後結果、治療方式與確認標籤才可進治療模型訓練。",
                },
            ],
        })
    except Exception as exc:
        result["warnings"].append(f"找不到完整處理後資料：{exc}")
    return result


def build_training_dataset_overview(project_root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a concise, clinician-readable inventory of the epoch training corpus."""
    root = Path(project_root)
    manifest_root = root / "data" / "continual_learning" / "external_epoch_psg"
    manifests = sorted(manifest_root.glob("manifest*.csv"))
    if not manifests:
        return pd.DataFrame(), {}
    rows = pd.concat([pd.read_csv(path) for path in manifests], ignore_index=True)
    rows = rows.drop_duplicates("study_id", keep="last")
    completed = rows[rows.get("status", "").astype(str).eq("COMPLETED")].copy()
    overview = (
        completed.groupby(["cohort", "split"], dropna=False)
        .size()
        .reset_index(name="PSG份數")
        .rename(columns={"cohort": "資料來源", "split": "資料用途"})
    )
    summary_path = root / "models" / "osa_next_epoch_lstm" / "training_summary.json"
    model = {}
    if summary_path.exists():
        model = pd.read_json(summary_path, typ="series").to_dict()
    return overview, {
        "candidate_studies": int(len(rows)),
        "completed_studies": int(len(completed)),
        "failed_studies": int((rows.get("status", "").astype(str) == "FAILED").sum()),
        "model_version": model.get("model_version"),
        "sequence_count": model.get("sequence_count"),
        "patient_count": model.get("patient_count"),
    }


def build_all_patient_processing_index(project_root: str | Path) -> pd.DataFrame:
    """List each processed PSG and its actual model-data role without exposing raw waveforms."""
    root = Path(project_root)
    processed_root = root / "data" / "processed"
    inference_root = root / "data" / "inference"
    split_path = root / "models" / "osa_next_epoch_lstm" / "patient_split.json"
    split_map: dict[str, str] = {}
    if split_path.exists():
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        labels = {"train": "訓練", "validation": "驗證", "test": "鎖定內部測試", "external_test": "鎖定外部測試"}
        for split, patient_ids in split_data.items():
            for patient_id in patient_ids:
                split_map[str(patient_id)] = labels.get(str(split), str(split))
    cohort_map: dict[str, str] = {}
    manifest_root = root / "data" / "continual_learning" / "external_epoch_psg"
    manifests = sorted(manifest_root.glob("manifest*.csv"))
    if manifests:
        manifest = pd.concat([pd.read_csv(path) for path in manifests], ignore_index=True).drop_duplicates("study_id", keep="last")
        cohort_map = dict(zip(manifest["study_id"].astype(str), manifest["cohort"].astype(str)))
        external_split_labels = {
            "train": "訓練",
            "validation": "驗證",
            "external_test": "鎖定外部測試",
        }
        for _, row in manifest.iterrows():
            if str(row.get("status", "")) == "COMPLETED":
                split_map[str(row["study_id"])] = external_split_labels.get(
                    str(row.get("split", "")), "未納入／待品質確認"
                )
    rows: list[dict[str, Any]] = []
    for folder in sorted(path for path in processed_root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        patient_id = folder.name
        stages_path = folder / "stages_aligned.csv"
        events_path = folder / "events_aligned.csv"
        feature_path = inference_root / patient_id / "inference_features.csv"
        if not (stages_path.exists() and events_path.exists()):
            continue
        try:
            stages = pd.read_csv(stages_path, usecols=lambda name: name in {"stage", "epoch_index"})
            events = pd.read_csv(events_path, usecols=lambda name: name in {"event_type"})
            event_counts = events.get("event_type", pd.Series(dtype=str)).value_counts()
            rows.append({
                "Patient／Study ID": patient_id,
                "資料來源": cohort_map.get(patient_id, "原始20位／一般患者資料"),
                "30秒對齊epoch": int(len(stages)),
                "OSA-only事件": int(event_counts.get("OBSTRUCTIVE_APNEA", 0) + event_counts.get("HYPOPNEA", 0)),
                "中央／混合型事件": int(event_counts.get("CENTRAL_APNEA", 0) + event_counts.get("MIXED_APNEA", 0)),
                "模型特徵已產生": "是" if feature_path.exists() else "否",
                "下一epoch模型資料用途": split_map.get(patient_id, "未納入／待品質確認"),
                "治療模型用途": "僅在有治療前後確認結果時可作為療效標籤",
            })
        except Exception:
            continue
    return pd.DataFrame(rows)
