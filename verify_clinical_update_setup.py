from __future__ import annotations

"""
Sleep Digital Twin clinical-update integration checker.

Run from the project root:

    python verify_clinical_update_setup.py

Optional patient-level checks:

    python verify_clinical_update_setup.py --patient-id "PATIENT_ID"

This script performs read-only checks. It does not modify patient data and does
not execute the treatment refinement or report generation scripts.
"""

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent

REQUIRED_PROJECT_FILES = (
    "app.py",
    "build_treatment_refinement.py",
    "build_patient_digital_twin_report.py",
)

REQUIRED_CLINICAL_MODULE_FILES = (
    "__init__.py",
    "base_parser.py",
    "registry.py",
    "update_parser.py",
    "update_merger.py",
    "update_service.py",
)


def print_check(
    passed: bool,
    title: str,
    detail: str = "",
) -> None:
    icon = "PASS" if passed else "FAIL"
    print(f"[{icon}] {title}")

    if detail:
        for line in str(detail).splitlines():
            print(f"       {line}")


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, dict):
        raise TypeError(
            f"JSON 根節點不是 object：{path}"
        )

    return payload


def check_project_files() -> list[str]:
    errors: list[str] = []

    for filename in REQUIRED_PROJECT_FILES:
        path = PROJECT_ROOT / filename
        passed = path.is_file()
        print_check(
            passed,
            f"專案檔案：{filename}",
            str(path),
        )

        if not passed:
            errors.append(
                f"缺少專案檔案：{path}"
            )

    module_dir = PROJECT_ROOT / "clinical_modules"

    print_check(
        module_dir.is_dir(),
        "clinical_modules 資料夾",
        str(module_dir),
    )

    if not module_dir.is_dir():
        errors.append(
            f"缺少資料夾：{module_dir}"
        )
        return errors

    for filename in REQUIRED_CLINICAL_MODULE_FILES:
        path = module_dir / filename
        passed = path.is_file()
        print_check(
            passed,
            f"Clinical module：{filename}",
            str(path),
        )

        if not passed:
            errors.append(
                f"缺少 Clinical module：{path}"
            )

    return errors


def check_imports() -> list[str]:
    errors: list[str] = []

    modules = (
        "clinical_modules",
        "clinical_modules.update_parser",
        "clinical_modules.update_merger",
        "clinical_modules.update_service",
    )

    for module_name in modules:
        try:
            imported_module = importlib.import_module(
                module_name
            )
        except Exception as exc:
            print_check(
                False,
                f"匯入：{module_name}",
                repr(exc),
            )
            errors.append(
                f"無法匯入 {module_name}：{exc}"
            )
        else:
            print_check(
                True,
                f"匯入：{module_name}",
                getattr(
                    imported_module,
                    "__file__",
                    "",
                ),
            )

    return errors


def check_public_api() -> list[str]:
    errors: list[str] = []

    try:
        from clinical_modules_backup.update_parser import (
            ClinicalUpdateParseResult,
            parse_clinical_update,
        )
        from clinical_modules_backup.update_merger import (
            update_clinical_decision_file,
        )
        from clinical_modules_backup.update_service import (
            refresh_patient,
        )
    except Exception as exc:
        print_check(
            False,
            "讀取 Clinical Update API",
            repr(exc),
        )
        return [
            f"無法讀取 Clinical Update API：{exc}"
        ]

    required_callables = {
        "parse_clinical_update":
            parse_clinical_update,
        "update_clinical_decision_file":
            update_clinical_decision_file,
        "refresh_patient":
            refresh_patient,
    }

    for name, function in required_callables.items():
        passed = callable(function)

        detail = ""
        if passed:
            try:
                detail = str(
                    inspect.signature(function)
                )
            except Exception:
                detail = "無法取得 signature"

        print_check(
            passed,
            f"API：{name}",
            detail,
        )

        if not passed:
            errors.append(
                f"{name} 不是 callable"
            )

    try:
        sample_result = ClinicalUpdateParseResult(
            module_id="PAP",
            data={
                "pressure_setting": 10.0,
                "invalid_nan": float("nan"),
            },
            source_files=["sample.csv"],
        )

        sample_dict = sample_result.to_dict()

        json.dumps(
            sample_dict,
            ensure_ascii=False,
            allow_nan=False,
        )

        nan_converted = (
            sample_dict["data"]["invalid_nan"]
            is None
        )

        print_check(
            nan_converted,
            "嚴格 JSON：NaN 轉換為 None",
            repr(sample_dict),
        )

        if not nan_converted:
            errors.append(
                "update_parser 未將 NaN 轉為 None"
            )

    except Exception as exc:
        print_check(
            False,
            "ClinicalUpdateParseResult JSON 測試",
            repr(exc),
        )
        errors.append(
            f"ParseResult JSON 測試失敗：{exc}"
        )

    return errors


def check_registry() -> list[str]:
    errors: list[str] = []

    try:
        from clinical_modules_backup import (
            get_parser,
            has_parser,
        )
    except Exception as exc:
        print_check(
            False,
            "Registry API",
            repr(exc),
        )
        return [
            f"Registry API 無法匯入：{exc}"
        ]

    pap_available = bool(
        has_parser("PAP")
    )

    print_check(
        pap_available,
        "PAP Parser 已註冊",
    )

    if not pap_available:
        errors.append(
            "PAP Parser 尚未註冊"
        )
        return errors

    try:
        pap_parser = get_parser("PAP")
    except Exception as exc:
        print_check(
            False,
            "取得 PAP Parser",
            repr(exc),
        )
        errors.append(
            f"無法取得 PAP Parser：{exc}"
        )
    else:
        parse_method = getattr(
            pap_parser,
            "parse",
            None,
        )

        print_check(
            callable(parse_method),
            "PAP Parser 提供 parse()",
            pap_parser.__class__.__name__,
        )

        if not callable(parse_method):
            errors.append(
                "PAP Parser 沒有可呼叫的 parse()"
            )

    return errors


def check_app_integration() -> list[str]:
    errors: list[str] = []
    app_path = PROJECT_ROOT / "app.py"

    if not app_path.is_file():
        return [
            f"找不到 app.py：{app_path}"
        ]

    app_text = app_path.read_text(
        encoding="utf-8"
    )

    expected_fragments = {
        "匯入 update_parser":
            "from clinical_modules.update_parser import",
        "使用 parse_clinical_update":
            "parsed_result = parse_clinical_update(",
        "匯入 update_service":
            "from clinical_modules.update_service import",
        "使用 refresh_patient":
            "service_result = refresh_patient(",
    }

    for title, fragment in expected_fragments.items():
        passed = fragment in app_text

        print_check(
            passed,
            f"app.py：{title}",
        )

        if not passed:
            errors.append(
                f"app.py 缺少：{fragment}"
            )

    obsolete_fragment = (
        "from clinical_update.ui import "
        "show_clinical_update_section"
    )

    obsolete_removed = (
        obsolete_fragment not in app_text
    )

    print_check(
        obsolete_removed,
        "app.py：已移除舊 clinical_update.ui",
    )

    if not obsolete_removed:
        errors.append(
            "app.py 仍包含舊 clinical_update.ui import"
        )

    try:
        compile(
            app_text,
            str(app_path),
            "exec",
        )
    except SyntaxError as exc:
        print_check(
            False,
            "app.py 語法檢查",
            str(exc),
        )
        errors.append(
            f"app.py 語法錯誤：{exc}"
        )
    else:
        print_check(
            True,
            "app.py 語法檢查",
        )

    return errors


def check_patient_files(
    patient_id: str,
) -> list[str]:
    errors: list[str] = []

    patient_id = patient_id.strip()
    if not patient_id:
        return errors

    refinement_dir = (
        PROJECT_ROOT
        / "data"
        / "inference"
        / patient_id
        / "treatment_refinement"
    )

    paths = {
        "ClinicalDecisionData":
            refinement_dir
            / "clinical_decision_data.json",
        "Refined recommendation":
            refinement_dir
            / "refined_treatment_recommendation.json",
        "Treatment interface":
            refinement_dir
            / "treatment_refinement_interface.json",
    }

    print()
    print(
        f"Patient-level read-only checks：{patient_id}"
    )

    for label, path in paths.items():
        exists = path.is_file()

        print_check(
            exists,
            label,
            str(path),
        )

        if not exists:
            errors.append(
                f"患者檔案不存在：{path}"
            )
            continue

        try:
            payload = load_json_object(path)
        except Exception as exc:
            print_check(
                False,
                f"{label} JSON",
                repr(exc),
            )
            errors.append(
                f"{label} JSON 無法讀取：{exc}"
            )
        else:
            print_check(
                True,
                f"{label} JSON",
                f"keys={len(payload)}",
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "檢查 Sleep Digital Twin "
            "Clinical Update 整合狀態。"
        )
    )

    parser.add_argument(
        "--patient-id",
        default="",
        help=(
            "選填；檢查指定患者的現有輸出檔案，"
            "不會修改資料。"
        ),
    )

    args = parser.parse_args()

    # Ensure project imports resolve when this script is
    # launched using an absolute path.
    project_root_text = str(PROJECT_ROOT)
    if project_root_text not in sys.path:
        sys.path.insert(
            0,
            project_root_text,
        )

    print("=" * 72)
    print("Sleep Digital Twin Clinical Update Setup Check")
    print("=" * 72)

    errors: list[str] = []

    errors.extend(
        check_project_files()
    )

    print()
    errors.extend(
        check_imports()
    )

    print()
    errors.extend(
        check_public_api()
    )

    print()
    errors.extend(
        check_registry()
    )

    print()
    errors.extend(
        check_app_integration()
    )

    if args.patient_id:
        errors.extend(
            check_patient_files(
                args.patient_id
            )
        )

    print()
    print("=" * 72)

    if errors:
        print(
            f"檢查完成：發現 {len(errors)} 個問題。"
        )

        for index, error in enumerate(
            errors,
            start=1,
        ):
            print(f"{index}. {error}")

        return 1

    print(
        "檢查完成：Clinical Update 基礎整合全部通過。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())