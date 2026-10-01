from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.continual_learning.apnea60 import build_baseline_target
from src.models.two_stage_sequence import AlertPolicy, train_bundle
from train_arousal_model import choose_feature_columns


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "processed" / "model_dataset.csv"


def main() -> None:
    frame = pd.read_csv(DATASET)
    numeric, _ = choose_feature_columns(frame)

    arousal = train_bundle(
        frame=frame,
        target_column="predict_arousal_next_30s",
        candidate_features=numeric,
        output=ROOT / "models" / "arousal_next_30s" / "two_stage_sequence.joblib",
        policy=AlertPolicy(recall_floor=0.90, window_epochs=7, persistence_epochs=2, refractory_epochs=2),
        horizon_epochs=1,
    )

    frame["predict_apnea_next_60s"] = build_baseline_target(frame).astype(int)
    apnea = train_bundle(
        frame=frame,
        target_column="predict_apnea_next_60s",
        candidate_features=numeric,
        output=ROOT / "models" / "apnea_next_60s" / "two_stage_sequence.joblib",
        policy=AlertPolicy(recall_floor=0.90, window_epochs=9, persistence_epochs=2, refractory_epochs=3),
        horizon_epochs=2,
    )

    output = ROOT / "models" / "two_stage_alert_training_summary.json"
    output.write_text(json.dumps({"arousal": arousal, "apnea": apnea}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"arousal": arousal, "apnea": apnea}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
