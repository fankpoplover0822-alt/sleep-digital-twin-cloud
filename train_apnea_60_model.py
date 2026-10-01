from __future__ import annotations

import json
from pathlib import Path

from src.continual_learning.apnea60 import Apnea60Service


if __name__ == "__main__":
    result = Apnea60Service(Path(__file__).resolve().parent).train()
    print(json.dumps(result, ensure_ascii=False, indent=2))
