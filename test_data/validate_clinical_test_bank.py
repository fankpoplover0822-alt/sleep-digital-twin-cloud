from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path

from clinical_modules import get_parser
from clinical_modules.update_parser import parse_clinical_update


def parse_one(module: str, path: Path):
    stream = BytesIO(path.read_bytes())
    stream.name = path.name
    stream.size = len(stream.getvalue())
    return parse_clinical_update(
        module_id=module,
        parser=get_parser(module),
        uploaded_files=[stream],
    )


def main() -> None:
    root = Path.home() / "Downloads" / "clinical_test_bank" / "clinical_test_bank"
    expanded = root / "網頁上傳測試_依選單分類"
    checks = []
    with (expanded / "00_測試檔對照表.csv").open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            checks.append(("new", row["module_id"], expanded / row["對應網頁選單"] / row["測試檔名"]))
    with (root / "manifest.csv").open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            checks.append(("existing", row["Module"], root / row["Module"] / row["Filename"]))

    results = []
    for kind, module, path in checks:
        try:
            parsed = parse_one(module, path)
            results.append((kind, module, path.name, len(parsed.data), " | ".join(parsed.warnings)))
        except Exception as exc:
            results.append((kind, module, path.name, -1, f"{type(exc).__name__}: {exc}"))

    failed = [item for item in results if item[3] <= 0]
    print(f"TOTAL={len(results)} PASSED={len(results)-len(failed)} FAILED={len(failed)}")
    for item in failed:
        print("FAIL", item)
    for item in results:
        if item[1] == "DEVICE_DATA":
            print("WEARABLE", item)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
