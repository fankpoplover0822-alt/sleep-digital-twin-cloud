from __future__ import annotations

import argparse
from pathlib import Path

from src.continual_learning.tonight_apnea_risk import predict
from src.continual_learning.wearable_realtime import WearableStore

ROOT = Path(__file__).resolve().parent

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    patient_id = str(parser.parse_args().patient_id).strip()
    samples = WearableStore(ROOT / "data" / "realtime_wearable").read(patient_id, limit=100000)
    result = predict(ROOT, patient_id, samples, ROOT / "models" / "tonight_apnea_risk" / "model.joblib")
    probability = result.get("probability")
    probability_text = (
        f"{float(probability):.1%}"
        if probability is not None
        else "資料不足，暫不顯示"
    )
    data_quality = result.get("data_quality") or {}
    print(
        "Tonight Risk Prediction: "
        f"level={result.get('risk_level', '資料不足')}, "
        f"probability={probability_text}, "
        f"samples={data_quality.get('sample_count', 0)}"
    )

if __name__ == "__main__":
    main()
