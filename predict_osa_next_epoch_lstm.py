from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.osa_next_epoch_lstm import predict_patient


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()
    print(json.dumps(predict_patient(Path(__file__).resolve().parent, args.patient_id), ensure_ascii=False, indent=2))
