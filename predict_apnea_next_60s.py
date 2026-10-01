from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.continual_learning.apnea60 import Apnea60Service


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="預測未來60秒 apnea，並將有 Event Grid 的資料納入持續學習"
    )
    parser.add_argument("--patient-id", required=True)
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="只預測，不加入訓練資料",
    )
    parser.add_argument(
        "--research-preview",
        action="store_true",
        help="以最新 Challenger 產生隔離的研究測試輸出",
    )
    args = parser.parse_args()
    service = Apnea60Service(PROJECT_ROOT)
    result = (
        service.predict_patient(
            args.patient_id,
            research_preview=bool(args.research_preview),
        )
        if args.predict_only
        else service.process_and_learn(args.patient_id)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
