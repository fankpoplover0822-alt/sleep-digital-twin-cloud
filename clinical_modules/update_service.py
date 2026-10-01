from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from clinical_modules.update_merger import (
    save_json_atomic,
    update_clinical_decision_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ClinicalUpdateError(RuntimeError):
    """Raised when a clinical update cannot be completed safely."""


def _restore_backup(backup_path: Path, destination: Path) -> None:
    """Restore through the same atomic writer, tolerating short Windows locks."""
    payload = _load_json(backup_path)
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            save_json_atomic(destination, payload)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.1 * (attempt + 1))
    raise ClinicalUpdateError(f"無法還原 {destination}：{last_error}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, dict):
        raise ClinicalUpdateError(f"JSON 根節點必須是物件：{path}")

    return payload


def _score_snapshot(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return rank/score/level keyed by treatment name."""
    snapshot: dict[str, dict[str, Any]] = {}

    research_preview = payload.get("research_treatment_preview", {})
    ranking = (
        research_preview.get("ranking", [])
        if isinstance(research_preview, Mapping)
        else []
    )
    if not isinstance(ranking, list) or not ranking:
        ranking = payload.get("personalized_treatment_ranking", [])
    if not isinstance(ranking, list):
        return snapshot

    for item in ranking:
        if not isinstance(item, Mapping):
            continue

        treatment = str(item.get("treatment", "")).strip()
        if not treatment:
            continue

        snapshot[treatment] = {
            "rank": item.get("rank"),
            "score": item.get("score"),
            "suitability_level": item.get("suitability_level"),
        }

    return snapshot


def _compare_snapshots(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    names = list(dict.fromkeys([*before.keys(), *after.keys()]))
    differences: list[dict[str, Any]] = []

    for name in names:
        old = dict(before.get(name, {}))
        new = dict(after.get(name, {}))

        try:
            old_score = float(old.get("score"))
        except (TypeError, ValueError):
            old_score = None

        try:
            new_score = float(new.get("score"))
        except (TypeError, ValueError):
            new_score = None

        score_delta = (
            new_score - old_score
            if old_score is not None and new_score is not None
            else None
        )

        differences.append(
            {
                "treatment": name,
                "rank_before": old.get("rank"),
                "rank_after": new.get("rank"),
                "score_before": old_score,
                "score_after": new_score,
                "score_delta": score_delta,
                "level_before": old.get("suitability_level"),
                "level_after": new.get("suitability_level"),
            }
        )

    differences.sort(
        key=lambda item: (
            item.get("rank_after") is None,
            item.get("rank_after") or 10**9,
        )
    )
    return differences


def _run_patient_script(
    project_root: Path,
    script_name: str,
    patient_id: str,
    extra_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    script_path = project_root / script_name
    if not script_path.is_file():
        raise ClinicalUpdateError(f"找不到重新評估程式：{script_path}")

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"

    command = [
            sys.executable,
            str(script_path),
            "--patient-id",
            patient_id,
    ]
    if extra_args:
        command.extend(str(argument) for argument in extra_args)

    process = subprocess.run(
        command,
        cwd=str(project_root),
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    result = {
        "script": script_name,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }

    if process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip()
        raise ClinicalUpdateError(
            f"{script_name} 執行失敗（exit={process.returncode}）：\n{message}"
        )

    return result


def _append_history(
    history_path: Path,
    record: Mapping[str, Any],
) -> None:
    history_payload = _load_json(history_path)
    records = history_payload.get("updates")
    if not isinstance(records, list):
        records = []

    records.append(dict(record))
    history_payload["updates"] = records
    history_payload["latest_version"] = record.get("version_after")
    history_payload["last_updated_at"] = record.get("updated_at")

    save_json_atomic(history_path, history_payload)


def refresh_patient_clinical_data(
    *,
    patient_id: str,
    module_id: str,
    confirmed_data: Mapping[str, Any],
    source_files: Sequence[str] | None = None,
    project_root: str | Path | None = None,
    regenerate_report: bool = True,
) -> dict[str, Any]:
    """
    Merge physician-confirmed clinical data and rerun only downstream steps.

    Execution order:
      1. Backup current clinical decision data.
      2. Merge confirmed fields.
      3. Build a new treatment snapshot with the confirmed clinical fields.
      4. Add it to the research training set and retrain Treatment Challenger.
      5. Rebuild treatment output using the new Challenger without duplicating a snapshot.
      6. Optionally rebuild the Digital Twin HTML report.
      7. Save update history and before/after ranking differences.

    If any downstream step fails, the original clinical decision file is
    restored so the patient state is not left half-updated.
    """
    normalized_patient_id = str(patient_id).strip()
    if not normalized_patient_id:
        raise ValueError("patient_id 不可為空。")

    if not isinstance(confirmed_data, Mapping):
        raise TypeError("confirmed_data 必須是 mapping/dict。")

    root = Path(project_root).resolve() if project_root else PROJECT_ROOT

    refinement_dir = (
        root
        / "data"
        / "inference"
        / normalized_patient_id
        / "treatment_refinement"
    )
    clinical_data_path = refinement_dir / "clinical_decision_data.json"
    refined_output_path = refinement_dir / "refined_treatment_recommendation.json"
    history_path = refinement_dir / "clinical_update_history.json"

    if not clinical_data_path.is_file():
        save_json_atomic(clinical_data_path, {
            "schema_version": 1,
            "patient_id": normalized_patient_id,
            "data_version": 0,
            "last_updated_at": None,
            "last_updated_module": None,
        })

    before_recommendation = _load_json(refined_output_path)
    ranking_before = _score_snapshot(before_recommendation)

    backup_dir = refinement_dir / "update_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = backup_dir / f"clinical_decision_data_{timestamp}.json"
    shutil.copy2(clinical_data_path, backup_path)

    script_results: list[dict[str, Any]] = []

    try:
        original_data, updated_data, change_record = (
            update_clinical_decision_file(
                clinical_data_path,
                patient_id=normalized_patient_id,
                module_id=module_id,
                confirmed_data=confirmed_data,
                source_files=source_files,
            )
        )

        script_results.append(
            _run_patient_script(
                root,
                "build_treatment_refinement.py",
                normalized_patient_id,
            )
        )

        script_results.append(
            _run_patient_script(
                root,
                "retrain_treatment_after_clinical_update.py",
                normalized_patient_id,
            )
        )

        script_results.append(
            _run_patient_script(
                root,
                "build_treatment_refinement.py",
                normalized_patient_id,
                extra_args=["--skip-training-snapshot"],
            )
        )

        if regenerate_report:
            script_results.append(
                _run_patient_script(
                    root,
                    "build_patient_digital_twin_report.py",
                    normalized_patient_id,
                )
            )

        after_recommendation = _load_json(refined_output_path)
        ranking_after = _score_snapshot(after_recommendation)
        recommendation_changes = _compare_snapshots(
            ranking_before,
            ranking_after,
        )

        history_record = {
            **change_record,
            "comparison_output_mode": "clinical_rules_plus_last_validated_research_challenger",
            "backup_file": str(backup_path.relative_to(root)),
            "ranking_before": ranking_before,
            "ranking_after": ranking_after,
            "recommendation_changes": recommendation_changes,
            "source_files": list(source_files or []),
        }
        _append_history(history_path, history_record)

        return {
            "success": True,
            "message": (
                "臨床特徵與報告已更新。只有新增且經確認的治療後療效資料，"
                "才會加入監督式訓練並建立新的治療 Challenger。"
            ),
            "patient_id": normalized_patient_id,
            "module_id": str(module_id).strip().upper(),
            "version_before": change_record["version_before"],
            "version_after": change_record["version_after"],
            "changed_fields": change_record["changed_fields"],
            "ranking_before": ranking_before,
            "ranking_after": ranking_after,
            "recommendation_changes": recommendation_changes,
            "comparison_output_mode": "clinical_rules_plus_last_validated_research_challenger",
            "clinical_data_path": str(clinical_data_path),
            "refined_output_path": str(refined_output_path),
            "history_path": str(history_path),
            "backup_path": str(backup_path),
            "script_results": script_results,
            "original_data": original_data,
            "updated_data": updated_data,
        }

    except Exception as exc:
        _restore_backup(backup_path, clinical_data_path)
        raise ClinicalUpdateError(
            "重新評估失敗，已還原更新前的 clinical_decision_data.json。"
            f" 原因：{exc}"
        ) from exc


def refresh_patient_clinical_batch(
    *,
    patient_id: str,
    updates: Sequence[Mapping[str, Any]],
    project_root: str | Path | None = None,
    regenerate_report: bool = True,
) -> dict[str, Any]:
    """Merge multiple parsed clinical modules, then retrain and render once.

    Every item requires ``module_id`` and ``confirmed_data``. The original
    patient file is restored if any merge or downstream command fails.
    """
    normalized_patient_id = str(patient_id).strip()
    if not normalized_patient_id:
        raise ValueError("patient_id is required")
    if not updates:
        raise ValueError("At least one clinical update is required")

    root = Path(project_root).resolve() if project_root else PROJECT_ROOT
    refinement_dir = root / "data" / "inference" / normalized_patient_id / "treatment_refinement"
    clinical_data_path = refinement_dir / "clinical_decision_data.json"
    refined_output_path = refinement_dir / "refined_treatment_recommendation.json"
    history_path = refinement_dir / "clinical_update_history.json"
    if not clinical_data_path.is_file():
        save_json_atomic(clinical_data_path, {
            "schema_version": 1,
            "patient_id": normalized_patient_id,
            "data_version": 0,
            "last_updated_at": None,
            "last_updated_module": None,
        })

    before_recommendation = _load_json(refined_output_path)
    ranking_before = _score_snapshot(before_recommendation)
    backup_dir = refinement_dir / "update_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = backup_dir / f"clinical_decision_data_batch_{timestamp}.json"
    shutil.copy2(clinical_data_path, backup_path)

    change_records: list[dict[str, Any]] = []
    script_results: list[dict[str, Any]] = []
    registry_dir = root / "models" / "registry" / "treatment"
    model_dirs_before = {path.name for path in registry_dir.iterdir() if path.is_dir()} if registry_dir.exists() else set()
    snapshot_dir = root / "data" / "continual_learning" / "treatment_snapshots"
    snapshots_before = {path.name for path in snapshot_dir.glob("*.csv")} if snapshot_dir.exists() else set()
    try:
        for item in updates:
            module_id = str(item.get("module_id", "")).strip().upper()
            confirmed_data = item.get("confirmed_data")
            if not module_id or not isinstance(confirmed_data, Mapping):
                raise ValueError("Each batch item needs module_id and confirmed_data")
            _, _, record = update_clinical_decision_file(
                clinical_data_path,
                patient_id=normalized_patient_id,
                module_id=module_id,
                confirmed_data=confirmed_data,
                source_files=list(item.get("source_files") or []),
            )
            change_records.append(record)

        # A batch is one Digital Twin transaction even though module-specific
        # merge validation runs independently.
        merged_payload = _load_json(clinical_data_path)
        initial_version = int(change_records[0].get("version_before") or 0)
        any_change = any(record.get("changed_fields") for record in change_records)
        merged_payload["data_version"] = initial_version + (1 if any_change else 0)
        merged_payload["last_updated_module"] = "BATCH" if any_change else merged_payload.get("last_updated_module")
        save_json_atomic(clinical_data_path, merged_payload)
        for record in change_records:
            record["version_before"] = initial_version
            record["version_after"] = merged_payload["data_version"]

        # Create one post-merge snapshot, train one Challenger on the accumulated
        # dataset, and regenerate one final output from that new Challenger.
        script_results.append(_run_patient_script(root, "build_treatment_refinement.py", normalized_patient_id))
        script_results.append(_run_patient_script(root, "retrain_treatment_after_clinical_update.py", normalized_patient_id))
        script_results.append(_run_patient_script(
            root, "build_treatment_refinement.py", normalized_patient_id,
            extra_args=["--skip-training-snapshot"],
        ))
        if regenerate_report:
            script_results.append(_run_patient_script(root, "build_patient_digital_twin_report.py", normalized_patient_id))

        ranking_after = _score_snapshot(_load_json(refined_output_path))
        recommendation_changes = _compare_snapshots(ranking_before, ranking_after)
        changed_fields = {
            f"{record.get('module_id')}.{field}": change
            for record in change_records
            for field, change in dict(record.get("changed_fields") or {}).items()
        }
        history_record = {
            "patient_id": normalized_patient_id,
            "module_id": "BATCH",
            "module_ids": [record.get("module_id") for record in change_records],
            "version_before": change_records[0].get("version_before"),
            "version_after": change_records[-1].get("version_after"),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "changed_fields": changed_fields,
            "batch_updates": change_records,
            "backup_file": str(backup_path.relative_to(root)),
            "ranking_before": ranking_before,
            "ranking_after": ranking_after,
            "recommendation_changes": recommendation_changes,
            "comparison_output_mode": "research_preview_latest_challenger",
        }
        _append_history(history_path, history_record)
        return {
            "success": True,
            "patient_id": normalized_patient_id,
            "module_id": "BATCH",
            "module_ids": history_record["module_ids"],
            "version_before": history_record["version_before"],
            "version_after": history_record["version_after"],
            "changed_fields": changed_fields,
            "ranking_before": ranking_before,
            "ranking_after": ranking_after,
            "recommendation_changes": recommendation_changes,
            "script_results": script_results,
            "history_path": str(history_path),
        }
    except Exception as exc:
        _restore_backup(backup_path, clinical_data_path)
        # Roll back derivative artifacts created by this failed transaction.
        if registry_dir.exists():
            for path in registry_dir.iterdir():
                if path.is_dir() and path.name not in model_dirs_before:
                    shutil.rmtree(path, ignore_errors=True)
        if snapshot_dir.exists():
            for path in snapshot_dir.glob("*.csv"):
                if path.name not in snapshots_before:
                    path.unlink(missing_ok=True)
        raise ClinicalUpdateError(f"Batch clinical update failed and was rolled back: {exc}") from exc


# A shorter alias for Streamlit/UI code.
refresh_patient = refresh_patient_clinical_data
refresh_patient_batch = refresh_patient_clinical_batch
