"""Run the released SleepFM research checkpoints on one patient's PSG.

This adapter deliberately writes research hazard rankings, not probabilities or
diagnoses.  Missing modalities are reported as a skipped result and never filled
with synthetic signals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "sleepfm-clinical"
SLEEPFM = VENDOR / "sleepfm"
OUTPUT_NAME = "sleepfm_disease_risk.json"
MODALITIES = ("BAS", "RESP", "EKG", "EMG")


def _normalise_channel(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper().replace("ECG", "EKG"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(patient_id: str, payload: dict) -> Path:
    target = ROOT / "data" / "inference" / patient_id / "sleepfm"
    target.mkdir(parents=True, exist_ok=True)
    path = target / OUTPUT_NAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _metadata(patient_id: str) -> dict:
    path = ROOT / "data" / "processed" / patient_id / "patient_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _find_edf(patient_id: str, metadata: dict) -> Path | None:
    stated = Path(str(metadata.get("edf_file", "")))
    if stated.is_file():
        return stated
    candidates = sorted((ROOT / "data" / "incoming" / patient_id).glob("*.edf"))
    return candidates[-1] if candidates else None


def _load_state(model, checkpoint: Path, key: str | None = None):
    import torch

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = loaded.get(key, loaded) if isinstance(loaded, dict) else loaded
    state = {name.removeprefix("module."): value for name, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _channel_assignment(channel_names: list[str], groups: dict) -> dict[str, list[int]]:
    lookup = {
        modality: {_normalise_channel(alias) for alias in groups[modality]}
        for modality in MODALITIES
    }
    assigned = {modality: [] for modality in MODALITIES}
    for index, name in enumerate(channel_names):
        normalized = _normalise_channel(name)
        for modality in MODALITIES:
            if normalized in lookup[modality]:
                assigned[modality].append(index)
                break
    return assigned


def run(patient_id: str) -> Path:
    base = {
        "schema_version": "sleepfm-research-v1",
        "patient_id": patient_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "SleepFM disease prediction",
        "model_source": "https://github.com/zou-group/sleepfm-clinical",
        "interpretation": (
            "CoxPH 相對風險排序；不是發病機率、確診結果或醫療處方。"
        ),
        "clinical_use": "research_only",
        "license": "CC BY-NC 4.0; 醫院正式部署前須另行確認授權與驗證。",
    }
    try:
        import mne
        import torch

        sys.path.insert(0, str(SLEEPFM))
        from models.models import DiagnosisFinetuneFullLSTMCOXPHWithDemo, SetTransformer

        metadata = _metadata(patient_id)
        edf = _find_edf(patient_id, metadata)
        if edf is None:
            raise ValueError("找不到患者 EDF")

        config_dir = SLEEPFM / "configs"
        groups = json.loads((config_dir / "channel_groups.json").read_text(encoding="utf-8"))
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        if int(round(raw.info["sfreq"])) != 128:
            raw.resample(128, verbose="ERROR")
        assigned = _channel_assignment(raw.ch_names, groups)
        missing = [name for name in MODALITIES if not assigned[name]]
        coverage = {
            name: [raw.ch_names[index] for index in assigned[name]] for name in MODALITIES
        }
        if missing:
            base.update(status="skipped_missing_modalities", missing_modalities=missing,
                        channel_coverage=coverage, source_edf=str(edf))
            return _write(patient_id, base)

        base_cfg = json.loads((SLEEPFM / "checkpoints/model_base/config.json").read_text())
        diagnosis_cfg = json.loads((SLEEPFM / "checkpoints/model_diagnosis/config.json").read_text())
        base_model = SetTransformer(
            in_channels=1, patch_size=base_cfg["patch_size"],
            embed_dim=base_cfg["embed_dim"], num_heads=base_cfg["num_heads"],
            num_layers=base_cfg["num_layers"], pooling_head=base_cfg["pooling_head"],
            dropout=base_cfg["dropout"], max_seq_length=128,
        )
        base_checkpoint = SLEEPFM / "checkpoints/model_base/best.pt"
        base_model = _load_state(base_model, base_checkpoint, "state_dict")

        signal = raw.get_data().astype(np.float32, copy=False)
        tokens_by_modality = []
        usable_tokens = None
        with torch.inference_mode():
            for modality in MODALITIES:
                x = signal[assigned[modality]]
                mean = x.mean(axis=1, keepdims=True)
                std = x.std(axis=1, keepdims=True)
                x = np.nan_to_num((x - mean) / np.maximum(std, 1e-6))
                token_count = x.shape[1] // 640
                if token_count < 1:
                    raise ValueError("EDF 長度不足 5 秒")
                token_count = min(token_count, diagnosis_cfg["model_params"]["max_seq_length"])
                x = x[:, : token_count * 640]
                chunks = []
                for start in range(0, token_count, 60):
                    stop = min(start + 60, token_count)
                    samples = x[:, start * 640 : stop * 640]
                    tensor = torch.from_numpy(samples).unsqueeze(0)
                    channel_mask = torch.zeros((1, tensor.shape[1]), dtype=torch.bool)
                    _, granular = base_model(tensor, channel_mask)
                    chunks.append(granular.squeeze(0).cpu())
                embedded = torch.cat(chunks, dim=0)
                usable_tokens = len(embedded) if usable_tokens is None else min(usable_tokens, len(embedded))
                tokens_by_modality.append(embedded)

            embeddings = torch.stack([value[:usable_tokens] for value in tokens_by_modality], dim=0).unsqueeze(0)
            mask = torch.zeros((1, len(MODALITIES), usable_tokens), dtype=torch.bool)
            params = diagnosis_cfg["model_params"]
            disease_model = DiagnosisFinetuneFullLSTMCOXPHWithDemo(**params)
            disease_checkpoint = SLEEPFM / "checkpoints/model_diagnosis/best.pth"
            disease_model = _load_state(disease_model, disease_checkpoint)
            age = float(metadata.get("age") or 0.0)
            sex = float(metadata.get("sex") or 0.0)
            # Released demo stores age on a 0..1 scale and sex as 0/1.
            demographics = torch.tensor([[max(0.0, min(age / 98.0, 1.0)), sex]], dtype=torch.float32)
            hazards = disease_model(embeddings, mask, demographics).squeeze(0).cpu().numpy()

        labels = {}
        with (config_dir / "label_mapping.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                labels[int(row["label_idx"])] = row
        top_indices = np.argsort(hazards)[::-1][:15]
        ranking = [
            {
                "rank": rank,
                "output_index": int(index),
                "phecode": labels.get(int(index), {}).get("phecode", ""),
                "phenotype": labels.get(int(index), {}).get("phenotype", "Unknown"),
                "relative_log_hazard": round(float(hazards[index]), 6),
            }
            for rank, index in enumerate(top_indices, start=1)
        ]
        base.update(
            status="completed", source_edf=str(edf), channel_coverage=coverage,
            five_second_tokens=int(usable_tokens), duration_hours=round(usable_tokens * 5 / 3600, 3),
            demographics={"age_years": age, "sex_code": sex}, ranking=ranking,
            checkpoints={"base_sha256": _sha256(base_checkpoint),
                         "diagnosis_sha256": _sha256(disease_checkpoint)},
        )
    except Exception as exc:
        base.update(status="error", error_type=type(exc).__name__, error_message=str(exc))
    return _write(patient_id, base)


def main() -> None:
    parser = argparse.ArgumentParser()
    # The shared patient pipeline calls every child with ``--patient-id``.
    # Keep the positional form as a backward-compatible CLI shortcut.
    parser.add_argument("patient_id_positional", nargs="?")
    parser.add_argument("--patient-id", dest="patient_id_option")
    args = parser.parse_args()
    patient_id = args.patient_id_option or args.patient_id_positional
    if not patient_id:
        parser.error("必須提供 patient_id 或 --patient-id")
    path = run(patient_id)
    result = json.loads(path.read_text(encoding="utf-8"))
    print(f"SleepFM：{result['status']}｜{path}")


if __name__ == "__main__":
    main()
