from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.continual_learning.service import ContinualLearningService
from src.continual_learning.treatment import train_treatment_models


PROJECT_ROOT = Path(__file__).resolve().parent


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sleep Digital Twin 持續學習管理工具"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="顯示持續學習狀態")

    arousal_import = subparsers.add_parser(
        "import-arousal", help="匯入人工確認的 Arousal 標籤"
    )
    arousal_import.add_argument("--features", required=True)
    arousal_import.add_argument("--labels", required=True)
    arousal_import.add_argument("--reviewer-id", required=True)
    arousal_import.add_argument("--approve", action="store_true")
    arousal_import.add_argument(
        "--label-source", default="human_psg_review"
    )

    outcome_import = subparsers.add_parser(
        "import-outcomes", help="匯入已確認的治療追蹤結果"
    )
    outcome_import.add_argument("--file", required=True)
    outcome_import.add_argument("--reviewer-id", required=True)

    train_arousal = subparsers.add_parser(
        "train-arousal", help="訓練 Arousal Challenger"
    )
    train_arousal.add_argument("--force", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate-arousal", help="評估 Arousal Challenger"
    )
    evaluate.add_argument("--version", required=True)

    promote = subparsers.add_parser(
        "promote-arousal", help="將通過驗證的 Challenger 升級"
    )
    promote.add_argument("--version", required=True)

    subparsers.add_parser(
        "train-treatment", help="從確認的追蹤結果訓練治療效果模型"
    )

    args = parser.parse_args()
    service = ContinualLearningService(PROJECT_ROOT)
    if args.command == "status":
        print_json(service.status())
    elif args.command == "import-arousal":
        print_json(
            service.ingest_arousal_feedback(
                args.features,
                args.labels,
                reviewer_id=args.reviewer_id,
                approve=args.approve,
                label_source=args.label_source,
            )
        )
    elif args.command == "import-outcomes":
        print_json(
            service.ingest_treatment_outcomes(
                args.file,
                reviewer_id=args.reviewer_id,
            )
        )
    elif args.command == "train-arousal":
        print_json(service.train_arousal_challenger(force=args.force))
    elif args.command == "evaluate-arousal":
        print_json(service.evaluate_arousal_challenger(args.version))
    elif args.command == "promote-arousal":
        print_json(service.promote_arousal_challenger(args.version))
    elif args.command == "train-treatment":
        print_json(train_treatment_models(PROJECT_ROOT))


if __name__ == "__main__":
    main()
