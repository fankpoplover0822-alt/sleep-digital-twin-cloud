from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.continual_learning.drug_trial import publish_combined_treatment_challenger
from src.continual_learning.treatment import train_adaptive_treatment_models


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _adaptive_versions(root: Path) -> list[dict[str, Any]]:
    registry = root / "models" / "registry" / "treatment"
    versions: list[dict[str, Any]] = []
    for path in registry.glob("treatment_adaptive_*/metadata.json"):
        try:
            versions.append(_read_json(path))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(versions, key=lambda row: str(row.get("created_at", "")))


def _metric_rows(metadata: dict[str, Any], previous: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_treatments = (previous or {}).get("treatments", {})
    for treatment, details in metadata.get("treatments", {}).items():
        metrics = details.get("metrics", {})
        previous_metrics = previous_treatments.get(treatment, {}).get("metrics", {})
        mae = metrics.get("mae")
        previous_mae = previous_metrics.get("mae")
        delta = None
        if mae is not None and previous_mae is not None:
            delta = float(mae) - float(previous_mae)
        rows.append({
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "model_version": metadata.get("model_version"),
            "previous_model_version": (previous or {}).get("model_version"),
            "treatment": treatment,
            "selected_algorithm": details.get("selected_algorithm"),
            "training_rows": details.get("row_count"),
            "unique_patients": details.get("patient_count"),
            "confirmed_outcome_rows": details.get("confirmed_outcome_rows"),
            "weak_label_rows": details.get("weak_label_rows"),
            "validation_strategy": metrics.get("validation_status"),
            "fold_count": metrics.get("fold_count"),
            "mae": mae,
            "previous_mae": previous_mae,
            "mae_change": delta,
            "mae_improved": (delta < -1e-12) if delta is not None else None,
            "mean_baseline_mae": metrics.get("mean_baseline_mae"),
            "outperforms_mean_baseline": metrics.get("outperforms_mean_baseline"),
            "r2": metrics.get("r2"),
            "score_adjustment_allowed": details.get("score_adjustment_allowed"),
            "promotion_allowed": metadata.get("promotion_allowed", False),
        })
    return rows


def _append_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_versions: set[tuple[str, str]] = set()
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                existing_versions.add((row.get("model_version", ""), row.get("treatment", "")))
    rows = [
        row for row in rows
        if (str(row["model_version"]), str(row["treatment"])) not in existing_versions
    ]
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--record-existing", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    history_file = root / "data" / "continual_learning" / "model_metrics" / "treatment_training_history.csv"

    versions = _adaptive_versions(root)
    if args.record_existing:
        for index, version in enumerate(versions):
            previous = versions[index - 1] if index else None
            _append_history(history_file, _metric_rows(version, previous))

    previous = versions[-1] if versions else None
    challenger = train_adaptive_treatment_models(root, allow_synthetic_demo=False)
    combined = publish_combined_treatment_challenger(root)
    rows = _metric_rows(challenger, previous)
    _append_history(history_file, rows)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "challenger_model_version": challenger.get("model_version"),
        "combined_model_version": combined.get("model_version"),
        "previous_model_version": (previous or {}).get("model_version"),
        "history_file": str(history_file),
        "automatic_champion_replacement": False,
        "reason": "Only unseen-patient validation plus external/temporal validation and clinical approval may promote a Champion.",
        "treatments": rows,
    }
    report_file = history_file.with_name("latest_treatment_training_report.json")
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
