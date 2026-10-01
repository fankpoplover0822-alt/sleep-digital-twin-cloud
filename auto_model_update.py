from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.continual_learning.service import ContinualLearningService
from src.continual_learning.treatment import train_adaptive_treatment_models
from src.features.event_label_builder import EventLabelBuilder


ROOT = Path(__file__).resolve().parent
TARGET = "predict_arousal_next_30s"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _arousal_update(patient_id: str) -> dict[str, Any]:
    features_path = ROOT / "data" / "inference" / patient_id / "inference_features.csv"
    patient_folder = ROOT / "data" / "processed" / patient_id
    stages_path = patient_folder / "stages_aligned.csv"
    events_path = patient_folder / "events_aligned.csv"
    missing = [str(path) for path in (features_path, stages_path, events_path) if not path.exists()]
    if missing:
        return {
            "status": "not_eligible",
            "reason": "缺少可產生真實 Arousal 標籤的 PSG/Event Grid 檔案",
            "missing_files": missing,
        }

    study_hash = hashlib.sha256(
        (_sha256(features_path) + _sha256(events_path)).encode("utf-8")
    ).hexdigest()
    service = ContinualLearningService(ROOT)
    destination = service.paths.approved / f"arousal_{patient_id}_{study_hash[:12]}.csv"
    trained_marker = destination.with_suffix(".trained.json")
    reused_existing_rows = destination.exists() and trained_marker.exists()

    if not destination.exists():
        labels = EventLabelBuilder().build_patient_labels(
            patient_id=patient_id,
            stages_file=stages_path,
            events_file=events_path,
        )
        features = pd.read_csv(features_path)
        merge_keys = [key for key in ("patient_id", "epoch_index") if key in labels.columns and key in features.columns]
        if len(merge_keys) < 2 or TARGET not in labels.columns:
            return {
                "status": "not_eligible",
                "reason": "Event Grid 未能產生可對齊的 30 秒 Arousal 真實標籤",
            }
        labelled = features.merge(labels[merge_keys + [TARGET]], on=merge_keys, how="inner").copy()
        if labelled.empty or labelled[TARGET].nunique(dropna=True) < 2:
            return {
                "status": "not_eligible",
                "reason": "Arousal 標籤為空或只有單一類別，無法進行監督式訓練",
            }
        labelled = labelled.assign(
            study_id=study_hash[:16],
            label_status="approved",
            label_source="event_grid_observed",
            reviewer_id="SYSTEM_EVENT_GRID",
            reviewed_at=_now(),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        labelled.to_csv(destination, index=False, encoding="utf-8-sig")
    else:
        labelled = pd.read_csv(destination, low_memory=False)
    challenger = service.train_arousal_challenger(force=True)
    evaluation = service.evaluate_arousal_challenger(challenger["model_version"])
    _write_json(
        trained_marker,
        {
            "study_hash": study_hash,
            "model_version": challenger["model_version"],
            "trained_at": _now(),
        },
    )
    return {
        "status": (
            "retrained_deduplicated_dataset"
            if reused_existing_rows
            else "challenger_trained"
        ),
        "study_hash": study_hash,
        "label_rows_added": 0 if reused_existing_rows else int(len(labelled)),
        "label_rows_reused": int(len(labelled)) if reused_existing_rows else 0,
        "positive_rows_added": int(labelled[TARGET].sum()),
        "challenger": challenger,
        "evaluation": evaluation,
        "champion_updated": False,
        "reason": (
            "已使用去重後完整訓練集重新訓練；相同檔案不會複製成新的病人樣本。正式升級仍需通過所有驗證與治理門檻"
            if reused_existing_rows
            else "已重新訓練並驗證；正式升級仍需通過所有驗證與治理門檻"
            if not evaluation.get("passed")
            else "已通過模型指標；仍須完成外部驗證與治理核准"
        ),
    }


def _treatment_update() -> dict[str, Any]:
    outcome_dir = ROOT / "data" / "continual_learning" / "treatment_outcomes"
    outcome_files = sorted(outcome_dir.glob("outcomes_*.csv"))
    if not outcome_files:
        return {
            "status": "not_eligible",
            "reason": "沒有經確認的治療後結果可進行監督式重新訓練",
            "champion_updated": False,
        }
    dataset_hash = hashlib.sha256(
        "".join(_sha256(path) for path in outcome_files).encode("utf-8")
    ).hexdigest()
    marker = outcome_dir / ".latest_trained_dataset.json"
    reused_existing_rows = False
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("dataset_hash") == dataset_hash:
            reused_existing_rows = True
    try:
        challenger = train_adaptive_treatment_models(ROOT)
    except Exception as exc:
        return {
            "status": "not_eligible",
            "reason": f"沒有足夠且經確認的治療後結果可重新訓練：{type(exc).__name__}: {exc}",
            "champion_updated": False,
        }
    _write_json(
        marker,
        {
            "dataset_hash": dataset_hash,
            "model_version": challenger.get("model_version"),
            "trained_at": _now(),
        },
    )
    return {
        "status": (
            "retrained_deduplicated_dataset"
            if reused_existing_rows
            else "challenger_trained"
        ),
        "challenger": challenger,
        "champion_updated": False,
        "reason": (
            "已使用去重後完整治療結果訓練集重新訓練；相同結果不重複計權。未通過獨立驗證前不會取代正式模型"
            if reused_existing_rows
            else "已用經確認療效重新訓練；未通過獨立驗證前不會取代正式模型"
        ),
    }


def run(patient_id: str) -> dict[str, Any]:
    output_dir = ROOT / "data" / "inference" / patient_id / "model_update_audit"
    receipt = {
        "schema_version": 1,
        "patient_id": patient_id,
        "started_at": _now(),
        "arousal_30s": _arousal_update(patient_id),
        "treatment": _treatment_update(),
        "apnea_60s": {},
    }
    apnea_receipt = ROOT / "data" / "inference" / patient_id / "apnea_next_60s" / "continual_learning_receipt.json"
    if apnea_receipt.exists():
        receipt["apnea_60s"] = json.loads(apnea_receipt.read_text(encoding="utf-8"))
    else:
        receipt["apnea_60s"] = {"status": "not_run", "reason": "找不到 60 秒模型更新收據"}
    receipt["completed_at"] = _now()
    _write_json(output_dir / "latest_model_update_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()
    result = run(str(args.patient_id).strip())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
