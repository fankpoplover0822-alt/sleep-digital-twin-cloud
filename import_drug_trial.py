from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.continual_learning.drug_trial import train_drug_trial_challenger


def main() -> None:
    parser = argparse.ArgumentParser(description="建立藥物相對安慰劑研究 Challenger")
    parser.add_argument("--roster", required=True)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    result = train_drug_trial_challenger(Path(__file__).parent, args.roster, args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
