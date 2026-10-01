from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.continual_learning.treatment import train_adaptive_treatment_models


ROOT = Path(__file__).resolve().parent


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def run(patient_id: str) -> dict[str, Any]:
    try:
        metadata = train_adaptive_treatment_models(
            ROOT,
            allow_synthetic_demo=True,
        )
    except RuntimeError as error:
        metadata = {
            "model_version": None,
            "training_row_count": 0,
            "confirmed_outcome_rows": 0,
            "weak_label_rows": 0,
            "synthetic_test_rows": 0,
            "status": "retraining_skipped_no_eligible_outcomes",
            "reason": str(error),
        }
    training_policy = str(metadata.get("training_policy") or "")
    default_status = (
        "clinical_outcome_model_retrained"
        if training_policy == "confirmed_post_treatment_outcomes_only"
        else "research_weak_supervision_model_retrained"
    )
    receipt = {
        "status": metadata.get("status", default_status),
        "patient_id": patient_id,
        "completed_at": _now(),
        "model_version": metadata.get("model_version"),
        "training_row_count": metadata.get("training_row_count", 0),
        "confirmed_outcome_rows": metadata.get("confirmed_outcome_rows", 0),
        "weak_label_rows": metadata.get("weak_label_rows", 0),
        "synthetic_test_rows": metadata.get("synthetic_test_rows", 0),
        "training_policy": training_policy,
        "new_upload_snapshots_included": bool(
            metadata.get("weak_recommendation_snapshots_used")
        ),
        "output_mode": (
            "research_preview_latest_challenger"
            if metadata.get("model_version")
            else "clinical_features_updated_rules_output_refreshed"
        ),
        "reason": metadata.get("reason"),
        "warning": (
            "補充資料與弱監督列僅供測試持續學習流程；"
            "未經外部驗證，不得作為正式臨床模型。"
        ),
    }
    output = (
        ROOT
        / "data"
        / "inference"
        / patient_id
        / "model_update_audit"
        / "latest_clinical_treatment_retrain_receipt.json"
    )
    _write_json(output, receipt)

    unified = output.parent / "latest_model_update_receipt.json"
    if unified.exists():
        try:
            payload = json.loads(unified.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {"patient_id": patient_id}
        policy = str(metadata.get("training_policy") or "")
        policy_reason = {
            "synthetic_demo_only": "合成測試資料已建立 Demo Challenger；僅更新研究預覽，未升級正式 Champion。",
            "weak_snapshot_research_only": "本次只使用推薦快照弱標籤重新訓練研究 Challenger；沒有真實治療成效標籤。",
            "mixed_confirmed_synthetic_and_weak_snapshot_research": "本次使用混合研究資料重新訓練 Challenger；請依收據確認各類標籤列數。",
            "confirmed_post_treatment_outcomes_only": "已使用經確認的真實治療後療效建立 Challenger；正式 Champion 尚未自動升級。",
        }.get(policy, "已重新訓練研究 Challenger；正式 Champion 尚未自動升級。")
        payload["treatment"] = {
            "status": receipt["status"],
            "challenger": metadata,
            "champion_updated": False,
            "research_output_updated": bool(metadata.get("model_version")),
            "clinical_feature_output_updated": True,
            "policy_reason": policy_reason,
            "reason": metadata.get("reason") or (
                "已使用合成治療成效建立 Demo Challenger；只更新研究輸出，禁止升級為正式臨床 Champion。"
                if metadata.get("training_policy") == "synthetic_demo_only"
                else "已使用經確認的真實治療後療效建立 Challenger；正式 Champion 尚未自動升級。"
            ),
        }
        if not metadata.get("reason"):
            payload["treatment"]["reason"] = policy_reason
        payload["completed_at"] = _now()
        _write_json(unified, payload)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(str(args.patient_id).strip()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
