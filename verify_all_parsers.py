from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_modules import PARSER_REGISTRY, get_parser, has_parser
from clinical_modules.parser_schemas import MODULE_SPECS
from clinical_modules.update_merger import MODULE_KEY_MAP, merge_clinical_decision_data

EXPECTED_MODULES = {
    "PAP", "ENT", "DISE", "IMAGING", "HYPOXEMIA", "ABG",
    "PULMONARY", "CARDIAC", "ECG", "ECHOCARDIOGRAPHY",
    "COMORBIDITY", "MEDICATION", "PATIENT_PREFERENCE", "FOLLOW_UP",
    "ANTHROPOMETRY", "WEIGHT_MANAGEMENT", "LIFESTYLE",
    "SLEEP_QUESTIONNAIRE", "DENTAL_CRANIOFACIAL", "LABORATORY",
    "NEUROLOGICAL", "PSYCHIATRIC", "OXYGEN_THERAPY",
    "SURGERY_HISTORY", "DEVICE_DATA", "OTHER",
}


class MemoryUpload(BytesIO):
    def __init__(self, name: str, text: str) -> None:
        super().__init__(text.encode("utf-8"))
        self.name = name


def check(condition: bool, title: str, details: str = "") -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {title}")
    if details:
        print(f"       {details}")
    return condition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-registry", action="store_true")
    args = parser.parse_args()

    failures = 0
    registered = set(PARSER_REGISTRY)
    failures += not check(
        registered == EXPECTED_MODULES,
        "所有 Clinical Data Types 均已註冊",
        f"registered={len(registered)} expected={len(EXPECTED_MODULES)}",
    )

    failures += not check(
        set(MODULE_KEY_MAP) == EXPECTED_MODULES,
        "Update merger 支援所有 Parser 模組",
    )

    failures += not check(
        set(MODULE_SPECS) == EXPECTED_MODULES - {"PAP"},
        "非 PAP Parser schema 完整",
    )

    for module_id in sorted(EXPECTED_MODULES):
        failures += not check(has_parser(module_id), f"Parser registry：{module_id}")
        parser_instance = get_parser(module_id)
        failures += not check(
            callable(getattr(parser_instance, "parse", None)),
            f"Parser parse()：{module_id}",
            parser_instance.__class__.__name__,
        )

    smoke_samples = {
        "PAP": "Pressure: 10\nResidual AHI: 4.5\nMask leak: 15\nUsage: 1.2",
        "ENT": "鼻中膈偏曲：有\n下鼻甲肥厚：無\n扁桃腺分級：3\nMallampati: 4",
        "DISE": "Velum collapse: complete\nTongue base collapse: partial\nCollapse pattern: concentric",
        "IMAGING": "Modality: CBCT\nRetropalatal airway narrowing: positive\nMinimum airway area: 82.5",
        "ABG": "pH: 7.39\nPaCO2: 48\nPaO2: 72\nHCO3: 29",
        "FOLLOW_UP": "Current treatment: CPAP\nResidual AHI: 3.2\nNightly usage hours: 5.5\nTreatment continued: yes",
    }

    for module_id, sample in smoke_samples.items():
        try:
            result = get_parser(module_id).parse(
                [MemoryUpload(f"{module_id.lower()}_sample.txt", sample)]
            )
            json.dumps(result.data, ensure_ascii=False, allow_nan=False)
            condition = bool(result.data)
            detail = repr(result.data)
        except Exception as exc:
            condition = False
            detail = repr(exc)
        failures += not check(condition, f"Smoke test：{module_id}", detail)

    try:
        merged, change = merge_clinical_decision_data(
            {"patient_id": "TEST", "data_version": 1},
            patient_id="TEST",
            module_id="ABG",
            confirmed_data={"paco2": 48.0},
            source_files=["abg.txt"],
        )
        condition = (
            merged["abg"]["findings"]["paco2"] == 48.0
            and change["version_after"] == 2
        )
        detail = repr(merged["abg"])
    except Exception as exc:
        condition = False
        detail = repr(exc)
    failures += not check(condition, "非核心模組 Merge smoke test", detail)

    app_path = PROJECT_ROOT / "app.py"
    try:
        compile(app_path.read_text(encoding="utf-8"), str(app_path), "exec")
        app_ok = True
        detail = str(app_path)
    except Exception as exc:
        app_ok = False
        detail = repr(exc)
    failures += not check(app_ok, "app.py 語法檢查", detail)

    if args.show_registry:
        print("\nRegistered parsers:")
        for module_id, parser_instance in sorted(PARSER_REGISTRY.items()):
            print(f"- {module_id}: {parser_instance.display_name}")

    print("=" * 72)
    if failures:
        print(f"檢查完成：{failures} 個問題。")
        return 1
    print("檢查完成：所有 Parser 基礎整合均通過。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
