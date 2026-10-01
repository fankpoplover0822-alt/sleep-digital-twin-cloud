from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from clinical_modules.registry import get_parser
from test_data.generate_clean_dynamic_test_bank import ROOT


class Upload:
    def __init__(self, path: Path):
        self.name = path.name
        self._stream = path.open("rb")

    def read(self, *args): return self._stream.read(*args)
    def seek(self, *args): return self._stream.seek(*args)
    def tell(self): return self._stream.tell()


manifest = json.loads((ROOT / "00_網頁欄位與資料夾對照.json").read_text(encoding="utf-8"))
results = []
for item in manifest:
    module_id = item["module_id"]
    for path in sorted((ROOT / item["folder"]).iterdir()):
        if not path.is_file():
            continue
        upload = Upload(path)
        try:
            parsed = get_parser(module_id).parse([upload])
            results.append({"module": module_id, "file": path.name, "fields": len(parsed.data), "keys": sorted(parsed.data)})
        except Exception as exc:
            results.append({"module": module_id, "file": path.name, "fields": -1, "error": f"{type(exc).__name__}: {exc}"})

failed = [row for row in results if row["fields"] <= 0]
print(json.dumps({"files": len(results), "passed": len(results) - len(failed), "failed": failed}, ensure_ascii=False, indent=2))
raise SystemExit(1 if failed else 0)
