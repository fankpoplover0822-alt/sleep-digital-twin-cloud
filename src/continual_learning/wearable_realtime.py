from __future__ import annotations

import json
import csv
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import joblib
import numpy as np
import pandas as pd


WATCH_FEATURES = [
    "spo2_mean", "spo2_std", "spo2_min", "spo2_range",
    "spo2_drop_from_start", "spo2_largest_drop",
    "heart_rate_mean", "heart_rate_std", "heart_rate_min", "heart_rate_max",
    "position_mean", "position_std", "age", "BMI",
]

STORE_COLUMNS = [
    "timestamp", "spo2", "heart_rate", "position", "movement",
    "accel_x", "accel_y", "accel_z", "accel_magnitude",
    "respiratory_rate", "sleep_stage", "steps", "battery", "hr_confidence",
    "apnea_observed_next_60s", "source",
]
LEGACY_COLUMNS = [
    "timestamp", "spo2", "heart_rate", "position", "movement",
    "apnea_observed_next_60s", "source",
]


def _safe_patient_id(value: str) -> str:
    return "".join(c for c in value.strip() if c.isalnum() or c in " -_")[:100]


class WearableStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.lock = threading.RLock()

    def _read_compatible(self, path: Path) -> pd.DataFrame:
        """Read both the former 7-column and current 9-column store safely."""
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            header = next(reader, [])
            for values in reader:
                if not values:
                    continue
                if len(values) == len(LEGACY_COLUMNS):
                    row = dict(zip(LEGACY_COLUMNS, values))
                    row["respiratory_rate"] = None
                    row["sleep_stage"] = None
                elif len(values) == len(STORE_COLUMNS):
                    # Mixed files can have a legacy header followed by new rows.
                    row = dict(zip(STORE_COLUMNS, values))
                elif len(values) == len(header):
                    row = dict(zip(header, values))
                else:
                    continue
                records.append({column: row.get(column) for column in STORE_COLUMNS})
        return pd.DataFrame(records, columns=STORE_COLUMNS)

    def _upgrade_store(self, path: Path) -> None:
        """Atomically normalize a legacy/mixed file before another append."""
        if not path.exists():
            return
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            header = next(csv.reader(source), [])
        if header == STORE_COLUMNS:
            return
        frame = self._read_compatible(path)
        temporary = path.with_suffix(".csv.upgrading")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)

    def remove_synthetic_rows(self, patient_id: str) -> int:
        """Remove only prior generated test rows; preserve real device data."""
        path = self.root / _safe_patient_id(patient_id) / "samples.csv"
        if not path.exists():
            return 0
        with self.lock:
            frame = self._read_compatible(path)
            source = frame["source"].fillna("").astype(str)
            synthetic = source.str.contains("|scenario=", regex=False) | source.str.contains(
                "synthetic_watch_test", regex=False
            )
            removed = int(synthetic.sum())
            if not removed:
                return 0
            remaining = frame.loc[~synthetic, STORE_COLUMNS]
            temporary = path.with_suffix(".csv.synthetic-cleanup")
            remaining.to_csv(temporary, index=False, encoding="utf-8-sig")
            os.replace(temporary, path)
            return removed

    def remove_uploaded_rows(self, patient_id: str) -> int:
        """Remove earlier file-upload batches while preserving network/BLE rows."""
        path = self.root / _safe_patient_id(patient_id) / "samples.csv"
        if not path.exists():
            return 0
        with self.lock:
            frame = self._read_compatible(path)
            uploaded = frame["source"].fillna("").astype(str).str.startswith("uploaded:")
            removed = int(uploaded.sum())
            if not removed:
                return 0
            temporary = path.with_suffix(".csv.upload-cleanup")
            frame.loc[~uploaded, STORE_COLUMNS].to_csv(
                temporary, index=False, encoding="utf-8-sig"
            )
            os.replace(temporary, path)
            return removed

    def append(self, patient_id: str, sample: dict[str, Any]) -> dict[str, Any]:
        patient_id = _safe_patient_id(patient_id)
        if not patient_id:
            raise ValueError("patient_id 不可空白")
        row = {
            "timestamp": str(sample.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            "spo2": float(sample.get("spo2", float("nan"))),
            "heart_rate": float(sample["heart_rate"]),
            "position": float(sample.get("position", 0)),
            "movement": float(sample.get("movement", 0)),
            "accel_x": sample.get("accel_x"),
            "accel_y": sample.get("accel_y"),
            "accel_z": sample.get("accel_z"),
            "accel_magnitude": sample.get("accel_magnitude"),
            "respiratory_rate": sample.get("respiratory_rate"),
            "sleep_stage": sample.get("sleep_stage"),
            "steps": sample.get("steps"),
            "battery": sample.get("battery"),
            "hr_confidence": sample.get("hr_confidence"),
            "apnea_observed_next_60s": sample.get("apnea_observed_next_60s"),
            "source": str(sample.get("source", "network")),
        }
        if not ((pd.isna(row["spo2"]) or 50 <= row["spo2"] <= 100) and 20 <= row["heart_rate"] <= 240):
            raise ValueError("SpO₂ 或心率超出可接受輸入範圍")
        folder = self.root / patient_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "samples.csv"
        with self.lock:
            self._upgrade_store(path)
            pd.DataFrame([row]).to_csv(
                path, mode="a", header=not path.exists(), index=False,
                columns=STORE_COLUMNS, encoding="utf-8-sig"
            )
        return row

    def append_many(self, patient_id: str, samples: list[dict[str, Any]]) -> int:
        """Validate and persist one device export with a single disk write."""
        patient_id = _safe_patient_id(patient_id)
        if not patient_id:
            raise ValueError("patient_id 不可為空")
        rows = []
        for sample in samples:
            row = {
                "timestamp": str(sample.get("timestamp") or datetime.now(timezone.utc).isoformat()),
                "spo2": float(sample.get("spo2", float("nan"))),
                "heart_rate": float(sample["heart_rate"]),
                "position": float(sample.get("position", 0)),
                "movement": float(sample.get("movement", 0)),
                "accel_x": sample.get("accel_x"),
                "accel_y": sample.get("accel_y"),
                "accel_z": sample.get("accel_z"),
                "accel_magnitude": sample.get("accel_magnitude"),
                "respiratory_rate": sample.get("respiratory_rate"),
                "sleep_stage": sample.get("sleep_stage"),
                "steps": sample.get("steps"),
                "battery": sample.get("battery"),
                "hr_confidence": sample.get("hr_confidence"),
                "apnea_observed_next_60s": sample.get("apnea_observed_next_60s"),
                "source": str(sample.get("source", "uploaded")),
            }
            if not ((pd.isna(row["spo2"]) or 50 <= row["spo2"] <= 100) and 20 <= row["heart_rate"] <= 240):
                raise ValueError("SpO₂ 或心率超出可接受範圍")
            rows.append(row)
        if not rows:
            return 0
        folder = self.root / patient_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "samples.csv"
        with self.lock:
            self._upgrade_store(path)
            pd.DataFrame(rows).to_csv(
                path, mode="a", header=not path.exists(), index=False,
                columns=STORE_COLUMNS, encoding="utf-8-sig"
            )
        return len(rows)

    def read(self, patient_id: str, limit: int = 600) -> pd.DataFrame:
        path = self.root / _safe_patient_id(patient_id) / "samples.csv"
        if not path.exists():
            return pd.DataFrame()
        with self.lock:
            frame = self._read_compatible(path)
            return frame.tail(limit).reset_index(drop=True)


def feature_window(samples: pd.DataFrame, age: float | None, bmi: float | None) -> pd.DataFrame:
    if samples.empty:
        raise ValueError("尚無手錶資料")
    spo2 = pd.to_numeric(samples["spo2"], errors="coerce").dropna().to_numpy()
    hr = pd.to_numeric(samples["heart_rate"], errors="coerce").dropna().to_numpy()
    pos = pd.to_numeric(samples.get("position", 0), errors="coerce").dropna().to_numpy()
    if len(spo2) < 10 or len(hr) < 10:
        raise ValueError("至少需要 10 筆有效 SpO₂ 與心率資料")
    values = {
        "spo2_mean": np.mean(spo2), "spo2_std": np.std(spo2), "spo2_min": np.min(spo2),
        "spo2_range": np.ptp(spo2), "spo2_drop_from_start": spo2[-1] - spo2[0],
        "spo2_largest_drop": np.max(np.maximum.accumulate(spo2) - spo2),
        "heart_rate_mean": np.mean(hr), "heart_rate_std": np.std(hr),
        "heart_rate_min": np.min(hr), "heart_rate_max": np.max(hr),
        "position_mean": np.mean(pos) if len(pos) else 0,
        "position_std": np.std(pos) if len(pos) else 0,
        "age": age, "BMI": bmi,
    }
    return pd.DataFrame([values], columns=WATCH_FEATURES)


def predict_risk(model_path: Path, samples: pd.DataFrame, age: float | None, bmi: float | None) -> dict[str, Any]:
    bundle = joblib.load(model_path)
    frame = feature_window(samples.tail(int(bundle.get("window_samples", 30))), age, bmi)
    raw = bundle["model"].predict_proba(frame)[:, 1]
    if "final_train_raw_sorted" in bundle and "oof_raw_sorted" in bundle:
        raw = np.interp(raw, bundle["final_train_raw_sorted"], bundle["oof_raw_sorted"])
    probability = float(bundle["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1][0])
    threshold = float(bundle["threshold"])
    notification_threshold = float(bundle.get("notification_threshold", threshold))
    return {
        "probability": probability,
        "threshold": threshold,
        "watch": probability >= threshold,
        "alert": probability >= notification_threshold,
        "notification_threshold": notification_threshold,
        "sample_count": int(min(len(samples), bundle.get("window_samples", 30))),
        "model_version": bundle.get("version"),
        "clinical_validation": False,
    }


def start_receiver(store: WearableStore, host: str = "127.0.0.1", port: int = 8765):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, payload: dict[str, Any]):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            try:
                prefix = "/api/wearable/"
                if not self.path.startswith(prefix):
                    return self._reply(404, {"ok": False, "error": "not found"})
                patient_id = unquote(self.path[len(prefix):])
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                row = store.append(patient_id, payload)
                self._reply(200, {"ok": True, "sample": row})
            except Exception as exc:
                self._reply(400, {"ok": False, "error": str(exc)})

        def log_message(self, *_):
            return

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError:
        return None
    threading.Thread(target=server.serve_forever, daemon=True, name="wearable-http-receiver").start()
    return server
