from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from predict_new_patient import build_inference_dataset
from process_incoming_patient import process_patient


ROOT = Path(__file__).resolve().parent
ID_PATTERN = re.compile(r"(\d{8}T\d{6}\s*-\s*[0-9a-fA-F]+)")


def _id(value: Any) -> str:
    match = ID_PATTERN.search(str(value or ""))
    return re.sub(r"\s+", " ", match.group(1)) if match else ""


def _split(study_id: str, cohort: str) -> str:
    # Immutable deterministic 65/15/20 split within each cohort. External test
    # is never used for fitting, early stopping, thresholding, or model choice.
    value = int(hashlib.sha256(f"{cohort}|{study_id}".encode()).hexdigest()[:8], 16)
    bucket = value % 20
    if bucket < 4:
        return "external_test"
    if bucket < 7:
        return "validation"
    return "train"


def _study_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    ent = pd.read_csv(ROOT / "data/continual_learning/external_clinical_cohorts/ent/ent_paired_psg_normalized.csv")
    for _, row in ent.iterrows():
        specs.append({"study_id": _id(row["baseline_study_id"]), "cohort": "ENT_PRE_SURGERY", "zip": r"C:\Users\fan\Desktop\ENT.zip", "age": row.get("age"), "BMI": row.get("BMI")})
    cpap = pd.read_csv(ROOT / "data/continual_learning/external_clinical_cohorts/pap/cpap_baseline_psg_with_pap_followup.csv")
    for _, row in cpap.iterrows():
        specs.append({"study_id": _id(row["patient_id"]), "cohort": "PAP_BASELINE", "zip": r"C:\Users\fan\Desktop\CPAP.zip", "age": row.get("age_at_baseline"), "BMI": row.get("BMI")})
    roster = pd.read_excel(r"C:\Users\fan\Downloads\Roster for Scott (1).xlsx")
    roster.columns = [str(column).strip().lower().replace(" ", "_") for column in roster.columns]
    psg1 = "psg1" if "psg1" in roster else "psg1_id"
    for _, row in roster.iterrows():
        specs.append({"study_id": _id(row.get(psg1)), "cohort": "DRUG_BASELINE", "zip": r"C:\Users\fan\Downloads\新藥物檔案.zip", "age": row.get("age"), "BMI": row.get("bmi")})
    # A study present in several cohorts is processed once with a stable first
    # provenance. Existing 20-patient studies are already complete and skipped.
    unique: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec["study_id"]:
            unique.setdefault(spec["study_id"], spec)
    return list(unique.values())


def _member_map(outer: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    result: dict[str, zipfile.ZipInfo] = {}
    for info in outer.infolist():
        if not info.filename.lower().endswith(".zip"):
            continue
        study_id = _id(Path(info.filename).name)
        if study_id:
            result.setdefault(study_id, info)
    return result


def _extract_study(outer: zipfile.ZipFile, member: zipfile.ZipInfo, study_id: str, destination: Path) -> tuple[Path, Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
        names = [name for name in inner.namelist() if not name.endswith("/")]
        edf_name = next(name for name in names if name.lower().endswith(".edf"))
        stage_name = next(name for name in names if "signal grid" in name.lower())
        event_name = next(name for name in names if "event grid" in name.lower())
        edf = destination / f"{study_id}_EDF.edf"
        stage = destination / f"{study_id}_stage.xls"
        event = destination / f"{study_id}_Event Grid.xls"
        for source, target in ((edf_name, edf), (stage_name, stage), (event_name, event)):
            with inner.open(source) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    return edf, stage, event


def _save_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["cohort", "study_id"]).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["all", "ENT_PRE_SURGERY", "PAP_BASELINE", "DRUG_BASELINE"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--repartition-only",
        action="store_true",
        help="依目前雜湊規則重建既有完成個案的train/validation/external_test分組，不重讀EDF。",
    )
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index必須介於0與shard-count-1")
    manifest_root = ROOT / "data/continual_learning/external_epoch_psg"
    manifest_path = manifest_root / f"manifest_shard_{args.shard_index:02d}.csv"
    previous: list[dict[str, Any]] = []
    for existing in sorted(manifest_root.glob("manifest*.csv")):
        previous.extend(pd.read_csv(existing).to_dict("records"))
    by_id = {str(row["study_id"]): row for row in previous}
    if args.repartition_only:
        rows: list[dict[str, Any]] = []
        for study_id, row in sorted(by_id.items()):
            updated = dict(row)
            updated["split"] = _split(study_id, str(updated.get("cohort", "")))
            rows.append(updated)
        output = manifest_root / "manifest_zz_repartitioned.csv"
        _save_manifest(rows, output)
        counts = pd.DataFrame(rows).query("status == 'COMPLETED'")["split"].value_counts().to_dict()
        print(json.dumps({"manifest": str(output), "completed_split_counts": counts}, ensure_ascii=False, indent=2))
        return
    specs = [spec for spec in _study_specs() if args.cohort == "all" or spec["cohort"] == args.cohort]
    specs = [spec for index, spec in enumerate(specs) if index % args.shard_count == args.shard_index]
    if args.limit:
        specs = specs[: max(args.limit, 0)]
    outer_cache: dict[str, tuple[zipfile.ZipFile, dict[str, zipfile.ZipInfo]]] = {}
    try:
        for index, spec in enumerate(specs, start=1):
            study_id = spec["study_id"]
            feature_path = ROOT / "data/inference" / study_id / "inference_features.csv"
            event_path = ROOT / "data/processed" / study_id / "events_aligned.csv"
            if feature_path.exists() and event_path.exists():
                status = "COMPLETED"
            else:
                status = "PENDING"
                incoming = ROOT / "data/incoming" / study_id
                try:
                    archive = spec["zip"]
                    if archive not in outer_cache:
                        outer = zipfile.ZipFile(archive)
                        outer_cache[archive] = (outer, _member_map(outer))
                    outer, members = outer_cache[archive]
                    if study_id not in members:
                        raise FileNotFoundError(f"壓縮檔找不到 {study_id}")
                    edf, _, _ = _extract_study(outer, members[study_id], study_id, incoming)
                    process_patient(study_id, allow_partial_edf=True)
                    stages = ROOT / "data/processed" / study_id / "stages_aligned.csv"
                    features = build_inference_dataset(study_id, edf, stages)
                    features["age"] = pd.to_numeric(spec.get("age"), errors="coerce")
                    features["BMI"] = pd.to_numeric(spec.get("BMI"), errors="coerce")
                    feature_path.parent.mkdir(parents=True, exist_ok=True)
                    features.to_csv(feature_path, index=False, encoding="utf-8-sig")
                    status = "COMPLETED"
                except Exception as exc:
                    status = "FAILED"
                    spec["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    shutil.rmtree(incoming, ignore_errors=True)
                    gc.collect()
            by_id[study_id] = {
                "study_id": study_id, "cohort": spec["cohort"],
                "split": _split(study_id, spec["cohort"]), "status": status,
                "feature_file": str(feature_path) if status == "COMPLETED" else "",
                "event_file": str(event_path) if status == "COMPLETED" else "",
                "error": spec.get("error", ""),
            }
            _save_manifest(list(by_id.values()), manifest_path)
            print(f"[{index}/{len(specs)}] {spec['cohort']} {study_id}: {status}", flush=True)
    finally:
        for outer, _ in outer_cache.values():
            outer.close()
    print(json.dumps({"manifest": str(manifest_path), "rows": len(by_id)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
