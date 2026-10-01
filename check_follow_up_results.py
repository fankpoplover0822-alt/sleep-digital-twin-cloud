from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PATIENT_ID = "20201014T221256 - d25c6"

patient_history_root = (
    PROJECT_ROOT
    / "data"
    / "study_history"
    / PATIENT_ID
)

patient_inference_root = (
    PROJECT_ROOT
    / "data"
    / "inference"
    / PATIENT_ID
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def find_latest_history() -> Path:
    manifests = sorted(
        patient_history_root.glob("*/manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not manifests:
        raise FileNotFoundError(
            f"找不到 Follow-up manifest：{patient_history_root}"
        )

    return manifests[0].parent


latest_history = find_latest_history()

before_interface = (
    latest_history
    / "before"
    / "inference"
    / "treatment_refinement"
    / "treatment_refinement_interface.json"
)

after_interface = (
    patient_inference_root
    / "treatment_refinement"
    / "treatment_refinement_interface.json"
)

before_data = load_json(before_interface)
after_data = load_json(after_interface)

print("=" * 70)
print("Follow-up 前後結果比較")
print("=" * 70)

print(f"History：{latest_history}")
print(f"Before：{before_interface}")
print(f"After ：{after_interface}")

print()
print("Before JSON 是否存在：", bool(before_data))
print("After JSON 是否存在 ：", bool(after_data))
print("完整 JSON 是否相同   ：", before_data == after_data)

before_candidates = before_data.get("treatments", [])
after_candidates = after_data.get("treatments", [])

if not isinstance(before_candidates, list):
    before_candidates = []

if not isinstance(after_candidates, list):
    after_candidates = []

print()
print("Before 候選數：", len(before_candidates))
print("After 候選數 ：", len(after_candidates))

print()
print("Before 第一順位：")
print(
    json.dumps(
        before_candidates[0] if before_candidates else {},
        ensure_ascii=False,
        indent=2,
    )
)

print()
print("After 第一順位：")
print(
    json.dumps(
        after_candidates[0] if after_candidates else {},
        ensure_ascii=False,
        indent=2,
    )
)

print()
print()
print("Before summary：")
print(
    json.dumps(
        before_data.get("summary", {}),
        ensure_ascii=False,
        indent=2,
    )
)

print()
print("After summary：")
print(
    json.dumps(
        after_data.get("summary", {}),
        ensure_ascii=False,
        indent=2,
    )
)

print()
print("Before recommendation_summary：")
print(
    json.dumps(
        before_data.get("recommendation_summary", {}),
        ensure_ascii=False,
        indent=2,
    )
)

print()
print("After recommendation_summary：")
print(
    json.dumps(
        after_data.get("recommendation_summary", {}),
        ensure_ascii=False,
        indent=2,
    )
)