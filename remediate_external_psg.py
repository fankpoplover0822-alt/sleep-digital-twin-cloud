"""Reprocess explicitly supplied replacement PSG ZIPs and update their audit record."""
from __future__ import annotations

import gc
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from predict_new_patient import build_inference_dataset
from prepare_external_epoch_psg import _split, _study_specs
from process_incoming_patient import process_patient


ROOT = Path(__file__).resolve().parent
REPLACEMENTS = {
    "20200616T225805 - 0945c": Path(r"C:\Users\fan\Downloads\20200616T225805 - 0945c-20260930T100149Z-1-001.zip"),
    "20201222T223228 - 740e3": Path(r"C:\Users\fan\Downloads\20201222T223228 - 740e3-20260930T100126Z-1-001.zip"),
    "20200806T223033 - 8d99c": Path(r"C:\Users\fan\Downloads\20200806T223033 - 8d99c-20260930T100119Z-1-001.zip"),
    "20250203T225705 - e0d27": Path(r"C:\Users\fan\Downloads\20250203T225705 - e0d27-20260930T095858Z-1-001.zip"),
}


def extract_triplet(archive_path: Path, patient_id: str, destination: Path) -> Path:
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        edf = next(name for name in names if name.lower().endswith(".edf"))
        stage = next(name for name in names if "signal grid" in name.lower())
        event = next(name for name in names if "event grid" in name.lower())
        destination.mkdir(parents=True, exist_ok=True)
        for source, target in (
            (edf, destination / f"{patient_id}_EDF.edf"),
            (stage, destination / f"{patient_id}_stage.xls"),
            (event, destination / f"{patient_id}_Event Grid.xls"),
        ):
            with archive.open(source) as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
    return destination / f"{patient_id}_EDF.edf"


def backup_existing(patient_id: str) -> None:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_root = ROOT / "data" / "remediation_backups" / stamp / patient_id
    for source in (ROOT / "data" / "processed" / patient_id, ROOT / "data" / "inference" / patient_id):
        if source.exists():
            target = backup_root / source.parent.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, dirs_exist_ok=True)


def main() -> None:
    specs = {item["study_id"]: item for item in _study_specs()}
    rows: dict[str, dict] = {}
    manifest_root = ROOT / "data" / "continual_learning" / "external_epoch_psg"
    for manifest_path in sorted(manifest_root.glob("manifest*.csv")):
        for row in pd.read_csv(manifest_path).to_dict("records"):
            rows[str(row["study_id"])] = row
    outcomes = []
    for patient_id, archive_path in REPLACEMENTS.items():
        spec = specs[patient_id]
        incoming = ROOT / "data" / "incoming" / patient_id
        try:
            if not archive_path.exists():
                raise FileNotFoundError(archive_path)
            backup_existing(patient_id)
            shutil.rmtree(incoming, ignore_errors=True)
            edf_path = extract_triplet(archive_path, patient_id, incoming)
            process_patient(patient_id, allow_partial_edf=True)
            stages = ROOT / "data" / "processed" / patient_id / "stages_aligned.csv"
            features = build_inference_dataset(patient_id, edf_path, stages)
            features["age"] = pd.to_numeric(spec.get("age"), errors="coerce")
            features["BMI"] = pd.to_numeric(spec.get("BMI"), errors="coerce")
            feature_path = ROOT / "data" / "inference" / patient_id / "inference_features.csv"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(feature_path, index=False, encoding="utf-8-sig")
            status, error = "COMPLETED", ""
        except Exception as exc:
            status, error = "FAILED", f"{type(exc).__name__}: {exc}"
        finally:
            shutil.rmtree(incoming, ignore_errors=True)
            gc.collect()
        rows[patient_id] = {
            "study_id": patient_id,
            "cohort": spec["cohort"],
            "split": _split(patient_id, spec["cohort"]),
            "status": status,
            "feature_file": str(ROOT / "data" / "inference" / patient_id / "inference_features.csv") if status == "COMPLETED" else "",
            "event_file": str(ROOT / "data" / "processed" / patient_id / "events_aligned.csv") if status == "COMPLETED" else "",
            "error": error,
            "remediation_source": str(archive_path),
        }
        outcomes.append({"study_id": patient_id, "status": status, "error": error})
        print(json.dumps(outcomes[-1], ensure_ascii=False), flush=True)
    output = manifest_root / "manifest_zzz_remediated.csv"
    pd.DataFrame(rows.values()).sort_values(["cohort", "study_id"]).to_csv(output, index=False, encoding="utf-8-sig")
    print(json.dumps({"manifest": str(output), "outcomes": outcomes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
