from __future__ import annotations

import time
from pathlib import Path

from src.continual_learning.wearable_realtime import WearableStore, start_receiver


ROOT = Path(__file__).resolve().parent


def main():
    server = start_receiver(WearableStore(ROOT / "data" / "realtime_wearable"))
    if server is None:
        print("Wearable receiver already running on 127.0.0.1:8765")
        return
    print("Wearable receiver: http://127.0.0.1:8765/api/wearable/<patient_id>")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
