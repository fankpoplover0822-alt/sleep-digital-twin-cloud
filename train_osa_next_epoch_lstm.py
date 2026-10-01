from __future__ import annotations

import json
from pathlib import Path

from src.models.osa_next_epoch_lstm import train


if __name__ == "__main__":
    print(json.dumps(train(Path(__file__).resolve().parent), ensure_ascii=False, indent=2))
