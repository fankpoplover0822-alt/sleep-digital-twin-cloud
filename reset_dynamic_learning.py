from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAMP = datetime.now().strftime("%Y%m%dT%H%M%S")
ARCHIVE = ROOT / "data" / "reset_archives" / f"pre_reset_{STAMP}"


def move_if_exists(path: Path) -> None:
    if not path.exists():
        return
    relative = path.relative_to(ROOT)
    destination = ARCHIVE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))


def main() -> None:
    processed = ROOT / "data" / "processed"
    patients = sorted(
        path.name
        for path in processed.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )
    if len(patients) != 20:
        raise RuntimeError(f"Expected exactly 20 baseline patients, found {len(patients)}")

    # Dynamic learning stores. Original raw/processed PSG data and patient
    # metadata are deliberately outside this list and therefore retained.
    dynamic_targets = [
        ROOT / "data" / "incoming",
        ROOT / "data" / "realtime_wearable",
        ROOT / "data" / "study_history",
        ROOT / "data" / "continual_learning",
        ROOT / "models" / "registry" / "treatment",
        ROOT / "models" / "registry" / "challengers",
        ROOT / "models" / "apnea_next_60s" / "candidates",
    ]
    for target in dynamic_targets:
        move_if_exists(target)

    # Remove patient-specific supplemental learning state while preserving the
    # original PSG-derived inference products needed as the clean baseline.
    inference_root = ROOT / "data" / "inference"
    supplemental_names = {
        "model_update_audit",
        "research_preview",
        "wearable_realtime",
    }
    for patient_dir in inference_root.iterdir() if inference_root.exists() else []:
        if not patient_dir.is_dir():
            continue
        for name in supplemental_names:
            move_if_exists(patient_dir / name)
        refinement = patient_dir / "treatment_refinement"
        for filename in (
            "clinical_decision_data.json",
            "clinical_update_history.json",
        ):
            move_if_exists(refinement / filename)

    # Recreate empty runtime directories. Baseline models will be rebuilt from
    # the retained 20-patient processed data by the normal training pipeline.
    for path in (
        ROOT / "data" / "continual_learning" / "treatment_snapshots",
        ROOT / "data" / "continual_learning" / "treatment_outcomes",
        ROOT / "data" / "incoming",
        ROOT / "data" / "realtime_wearable",
        ROOT / "data" / "study_history",
        ROOT / "models" / "registry" / "treatment",
        ROOT / "models" / "registry" / "challengers",
        ROOT / "models" / "apnea_next_60s" / "candidates",
    ):
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "reset_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "archive": str(ARCHIVE),
        "baseline_patient_count": len(patients),
        "baseline_patients": patients,
        "retained": [
            "data/processed/<20 patients>",
            "original PSG-derived inference products",
            "original patient metadata/persona inputs",
        ],
        "removed_from_active_learning": [str(path.relative_to(ROOT)) for path in dynamic_targets],
    }
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "reset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
