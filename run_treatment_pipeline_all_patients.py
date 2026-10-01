from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

INCOMING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "incoming"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INFERENCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "inference"
)

PIPELINE_REPORT_ROOT = (
    INFERENCE_ROOT
    / "_treatment_pipeline"
)

LOG_ROOT = (
    PIPELINE_REPORT_ROOT
    / "logs"
)


@dataclass(frozen=True)
class PipelineStep:
    name: str
    script_name: str
    expected_output_parts: tuple[str, ...]


PIPELINE_STEPS = [
    PipelineStep(
        name="RESPIRATORY_PROFILE",
        script_name=(
            "build_patient_respiratory_profile.py"
        ),
        expected_output_parts=(
            "respiratory_profile",
            "patient_respiratory_profile.json",
        ),
    ),
    PipelineStep(
        name="SPO2_QUALITY",
        script_name=(
            "build_spo2_quality_mask.py"
        ),
        expected_output_parts=(
            "spo2_quality",
            "spo2_quality_mask.csv",
        ),
    ),
    PipelineStep(
        name="OXYGEN_EVENT_COUPLING",
        script_name=(
            "analyze_oxygen_event_coupling.py"
        ),
        expected_output_parts=(
            "oxygen_event_coupling",
            "oxygen_event_coupling_summary.json",
        ),
    ),
    PipelineStep(
        name="POSITION_PROFILE",
        script_name=(
            "build_patient_position_profile.py"
        ),
        expected_output_parts=(
            "position_profile",
            "patient_position_profile.json",
        ),
    ),
    PipelineStep(
        name="TREATMENT_RECOMMENDATION",
        script_name=(
            "build_treatment_suitability.py"
        ),
        expected_output_parts=(
            "treatment_suitability",
            "personalized_treatment_recommendation.json",
        ),
    ),
]


def safe_json_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): safe_json_value(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            safe_json_value(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    return str(value)


def save_json(
    file_path: Path,
    data: dict[str, Any],
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            safe_json_value(data),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def is_windows_reparse_point(
    path: Path,
) -> bool:
    if not path.exists():
        return False

    try:
        stat_result = path.stat()
    except OSError:
        return False

    file_attributes = getattr(
        stat_result,
        "st_file_attributes",
        0,
    )

    reparse_flag = getattr(
        os,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        0x400,
    )

    return bool(
        file_attributes
        & reparse_flag
    )


def remove_directory_junction(
    junction_path: Path,
) -> None:
    if not junction_path.exists():
        return

    if os.name != "nt":
        raise RuntimeError(
            "Junction 清理只支援 Windows。"
        )

    result = subprocess.run(
        [
            "cmd",
            "/c",
            "rmdir",
            str(junction_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "無法移除 Junction："
            f"{junction_path}\n"
            f"{result.stderr.strip()}"
        )


def create_directory_junction(
    junction_path: Path,
    target_path: Path,
) -> None:
    if os.name != "nt":
        raise RuntimeError(
            "本批次程式的 Junction 功能只支援 Windows。"
        )

    junction_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = subprocess.run(
        [
            "cmd",
            "/c",
            "mklink",
            "/J",
            str(junction_path),
            str(target_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "建立 Junction 失敗。\n"
            f"Junction：{junction_path}\n"
            f"Target：{target_path}\n"
            f"STDOUT：{result.stdout.strip()}\n"
            f"STDERR：{result.stderr.strip()}"
        )


def ensure_patient_incoming_access(
    patient_id: str,
) -> dict[str, Any]:
    incoming_folder = (
        INCOMING_ROOT
        / patient_id
    )

    raw_folder = (
        RAW_ROOT
        / patient_id
    )

    if incoming_folder.exists():
        return {
            "status": "EXISTING_INCOMING",
            "incoming_folder": (
                incoming_folder
            ),
            "source_folder": (
                incoming_folder
            ),
            "junction_created": False,
        }

    if not raw_folder.exists():
        raise FileNotFoundError(
            "找不到患者資料來源資料夾："
            f"{patient_id}\n"
            f"Incoming：{incoming_folder}\n"
            f"Raw：{raw_folder}"
        )

    create_directory_junction(
        junction_path=incoming_folder,
        target_path=raw_folder,
    )

    if not incoming_folder.exists():
        raise RuntimeError(
            "Junction 建立後仍無法存取："
            f"{incoming_folder}"
        )

    return {
        "status": "CREATED_RAW_JUNCTION",
        "incoming_folder": (
            incoming_folder
        ),
        "source_folder": (
            raw_folder
        ),
        "junction_created": True,
    }


def discover_patient_ids() -> list[str]:
    if not PROCESSED_ROOT.exists():
        raise FileNotFoundError(
            f"找不到 processed 根目錄：{PROCESSED_ROOT}"
        )

    patient_ids = sorted(
        path.name
        for path in PROCESSED_ROOT.iterdir()
        if (
            path.is_dir()
            and not path.name.startswith("_")
        )
    )

    return patient_ids


def validate_required_scripts() -> None:
    required_scripts = [
        step.script_name
        for step in PIPELINE_STEPS
    ]

    required_scripts.append(
        "validate_treatment_recommendations.py"
    )

    missing_scripts = [
        script_name
        for script_name in required_scripts
        if not (
            PROJECT_ROOT
            / script_name
        ).exists()
    ]

    if missing_scripts:
        raise FileNotFoundError(
            "缺少必要程式："
            f"{missing_scripts}"
        )


def expected_output_path(
    patient_id: str,
    step: PipelineStep,
) -> Path:
    return (
        INFERENCE_ROOT
        / patient_id
        / Path(
            *step.expected_output_parts
        )
    )


def output_is_complete(
    patient_id: str,
    step: PipelineStep,
) -> bool:
    output_path = expected_output_path(
        patient_id,
        step,
    )

    if not output_path.exists():
        return False

    try:
        return (
            output_path.stat().st_size
            > 0
        )
    except OSError:
        return False


def sanitize_filename(
    text: str,
) -> str:
    invalid_characters = (
        '<>:"/\\|?*'
    )

    result = str(text)

    for character in invalid_characters:
        result = result.replace(
            character,
            "_",
        )

    return result.strip()


def write_step_log(
    patient_id: str,
    step_name: str,
    stdout_text: str,
    stderr_text: str,
    command: list[str],
    return_code: int | None,
    elapsed_seconds: float,
) -> Path:
    patient_log_folder = (
        LOG_ROOT
        / sanitize_filename(
            patient_id
        )
    )

    patient_log_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = (
        patient_log_folder
        / (
            sanitize_filename(
                step_name
            )
            + ".log"
        )
    )

    command_text = subprocess.list2cmdline(
        command
    )

    with log_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "=" * 80
        )
        file.write("\n")

        file.write(
            f"Patient ID：{patient_id}\n"
        )

        file.write(
            f"Step：{step_name}\n"
        )

        file.write(
            f"Command：{command_text}\n"
        )

        file.write(
            f"Return Code：{return_code}\n"
        )

        file.write(
            f"Elapsed Seconds："
            f"{elapsed_seconds:.3f}\n"
        )

        file.write(
            "=" * 80
        )
        file.write("\n\n")

        file.write(
            "STDOUT\n"
        )
        file.write(
            "-" * 80
        )
        file.write("\n")
        file.write(
            stdout_text
        )

        if (
            stdout_text
            and not stdout_text.endswith(
                "\n"
            )
        ):
            file.write("\n")

        file.write("\n")
        file.write(
            "STDERR\n"
        )
        file.write(
            "-" * 80
        )
        file.write("\n")
        file.write(
            stderr_text
        )

        if (
            stderr_text
            and not stderr_text.endswith(
                "\n"
            )
        ):
            file.write("\n")

    return log_file


def run_command(
    command: list[str],
    timeout_seconds: int | None,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    child_environment = (
        os.environ.copy()
    )

    child_environment[
        "PYTHONIOENCODING"
    ] = "utf-8"

    child_environment[
        "PYTHONUTF8"
    ] = "1"

    child_environment[
        "PYTHONLEGACYWINDOWSSTDIO"
    ] = "0"

    effective_command = list(
        command
    )

    if (
        effective_command
        and Path(
            effective_command[0]
        ).resolve()
        == Path(
            sys.executable
        ).resolve()
        and "-X" not in effective_command[1:3]
    ):
        effective_command = [
            effective_command[0],
            "-X",
            "utf8",
            *effective_command[1:],
        ]

    try:
        completed = subprocess.run(
            effective_command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            env=child_environment,
        )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        return {
            "return_code": (
                completed.returncode
            ),
            "stdout": (
                completed.stdout
                or ""
            ),
            "stderr": (
                completed.stderr
                or ""
            ),
            "elapsed_seconds": (
                elapsed_seconds
            ),
            "timed_out": False,
            "exception_type": None,
            "exception_message": None,
            "executed_command": (
                effective_command
            ),
        }

    except subprocess.TimeoutExpired as error:
        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        stdout_text = (
            error.stdout
            if isinstance(
                error.stdout,
                str,
            )
            else ""
        )

        stderr_text = (
            error.stderr
            if isinstance(
                error.stderr,
                str,
            )
            else ""
        )

        return {
            "return_code": None,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "elapsed_seconds": (
                elapsed_seconds
            ),
            "timed_out": True,
            "exception_type": (
                type(error).__name__
            ),
            "exception_message": (
                str(error)
            ),
            "executed_command": (
                effective_command
            ),
        }

    except Exception as error:
        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        return {
            "return_code": None,
            "stdout": "",
            "stderr": "",
            "elapsed_seconds": (
                elapsed_seconds
            ),
            "timed_out": False,
            "exception_type": (
                type(error).__name__
            ),
            "exception_message": (
                str(error)
            ),
            "executed_command": (
                effective_command
            ),
        }


def run_patient_step(
    patient_id: str,
    step: PipelineStep,
    force: bool,
    timeout_seconds: int | None,
) -> dict[str, Any]:
    output_path = expected_output_path(
        patient_id,
        step,
    )

    if (
        not force
        and output_is_complete(
            patient_id,
            step,
        )
    ):
        return {
            "patient_id": patient_id,
            "step": step.name,
            "script": step.script_name,
            "status": (
                "SKIPPED_OUTPUT_EXISTS"
            ),
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "expected_output": (
                str(output_path)
            ),
            "output_exists": True,
            "log_file": None,
            "error_type": None,
            "error_message": None,
        }

    command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / step.script_name
        ),
        "--patient-id",
        patient_id,
    ]

    execution = run_command(
        command=command,
        timeout_seconds=(
            timeout_seconds
        ),
    )

    output_exists = (
        output_is_complete(
            patient_id,
            step,
        )
    )

    return_code = execution[
        "return_code"
    ]

    exception_type = execution[
        "exception_type"
    ]

    exception_message = execution[
        "exception_message"
    ]

    if execution["timed_out"]:
        status = "FAILED_TIMEOUT"

    elif exception_type is not None:
        status = "FAILED_EXCEPTION"

    elif (
        return_code == 0
        and output_exists
    ):
        status = "SUCCESS"

    elif return_code == 0:
        status = (
            "FAILED_OUTPUT_MISSING"
        )

        exception_type = (
            "MissingOutputError"
        )

        exception_message = (
            "程式 return code 為 0，"
            "但預期輸出不存在："
            f"{output_path}"
        )

    else:
        status = (
            "FAILED_RETURN_CODE"
        )

        exception_type = (
            "SubprocessError"
        )

        exception_message = (
            f"程式 return code："
            f"{return_code}"
        )

    stderr_text = execution[
        "stderr"
    ]

    if exception_message:
        stderr_text = (
            stderr_text
            + "\n"
            + exception_message
        ).strip()

    log_file = write_step_log(
        patient_id=patient_id,
        step_name=step.name,
        stdout_text=(
            execution["stdout"]
        ),
        stderr_text=(
            stderr_text
        ),
        command=execution.get(
            "executed_command",
            command,
        ),
        return_code=return_code,
        elapsed_seconds=(
            execution[
                "elapsed_seconds"
            ]
        ),
    )

    return {
        "patient_id": patient_id,
        "step": step.name,
        "script": step.script_name,
        "status": status,
        "return_code": return_code,
        "elapsed_seconds": (
            execution[
                "elapsed_seconds"
            ]
        ),
        "expected_output": (
            str(output_path)
        ),
        "output_exists": (
            output_exists
        ),
        "log_file": (
            str(log_file)
        ),
        "error_type": (
            exception_type
        ),
        "error_message": (
            exception_message
        ),
    }


def run_validation(
    timeout_seconds: int | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / "validate_treatment_recommendations.py"
        ),
    ]

    execution = run_command(
        command=command,
        timeout_seconds=(
            timeout_seconds
        ),
    )

    expected_output = (
        INFERENCE_ROOT
        / "_treatment_validation"
        / "treatment_validation_summary.json"
    )

    output_exists = (
        expected_output.exists()
        and expected_output.stat().st_size
        > 0
    )

    if (
        execution["return_code"] == 0
        and output_exists
    ):
        status = "SUCCESS"

    elif execution["timed_out"]:
        status = "FAILED_TIMEOUT"

    else:
        status = "FAILED"

    log_file = write_step_log(
        patient_id="_ALL_PATIENTS",
        step_name=(
            "VALIDATE_TREATMENT_RECOMMENDATIONS"
        ),
        stdout_text=(
            execution["stdout"]
        ),
        stderr_text=(
            execution["stderr"]
        ),
        command=execution.get(
            "executed_command",
            command,
        ),
        return_code=(
            execution["return_code"]
        ),
        elapsed_seconds=(
            execution[
                "elapsed_seconds"
            ]
        ),
    )

    return {
        "patient_id": (
            "_ALL_PATIENTS"
        ),
        "step": (
            "VALIDATE_TREATMENT_RECOMMENDATIONS"
        ),
        "script": (
            "validate_treatment_recommendations.py"
        ),
        "status": status,
        "return_code": (
            execution["return_code"]
        ),
        "elapsed_seconds": (
            execution[
                "elapsed_seconds"
            ]
        ),
        "expected_output": (
            str(expected_output)
        ),
        "output_exists": (
            output_exists
        ),
        "log_file": (
            str(log_file)
        ),
        "error_type": (
            execution[
                "exception_type"
            ]
        ),
        "error_message": (
            execution[
                "exception_message"
            ]
        ),
    }


def write_csv(
    file_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        file_path.write_text(
            "",
            encoding="utf-8-sig",
        )
        return

    field_names: list[str] = []

    seen_fields: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen_fields:
                seen_fields.add(key)
                field_names.append(key)

    with file_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=field_names,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: safe_json_value(
                        value
                    )
                    for key, value in row.items()
                }
            )


def print_step_result(
    result: dict[str, Any],
) -> None:
    status = result.get(
        "status"
    )

    step = result.get(
        "step"
    )

    elapsed_seconds = result.get(
        "elapsed_seconds",
        0.0,
    )

    if status == "SUCCESS":
        marker = "[OK]"

    elif status == (
        "SKIPPED_OUTPUT_EXISTS"
    ):
        marker = "[SKIP]"

    else:
        marker = "[FAILED]"

    print(
        f"  {marker} {step}："
        f"{status} "
        f"({float(elapsed_seconds):.1f} 秒)"
    )

    if result.get(
        "error_message"
    ):
        print(
            "         "
            f"{result['error_message']}"
        )

    if (
        status.startswith(
            "FAILED"
        )
        and result.get(
            "log_file"
        )
    ):
        print(
            "         Log："
            f"{result['log_file']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "逐位患者執行呼吸 Profile、SpO₂ 品質、"
            "低氧事件耦合、姿勢 Profile 與"
            "個人化治療推薦，最後執行整體驗證。"
        )
    )

    parser.add_argument(
        "--patient-id",
        default=None,
        help=(
            "只處理指定患者；"
            "未指定時處理全部患者。"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "即使預期輸出已存在，"
            "仍重新執行全部步驟。"
        ),
    )

    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help=(
            "同一患者某一步失敗後，"
            "仍嘗試執行後續步驟。"
        ),
    )

    parser.add_argument(
        "--no-validation",
        action="store_true",
        help=(
            "完成患者流程後不執行"
            "整體推薦驗證。"
        ),
    )

    parser.add_argument(
        "--cleanup-created-junctions",
        action="store_true",
        help=(
            "執行完成後移除本次建立的"
            "data/incoming Junction。"
        ),
    )

    parser.add_argument(
        "--step-timeout-minutes",
        type=float,
        default=60.0,
        help=(
            "每個患者每一步的逾時分鐘數；"
            "預設 60 分鐘。"
        ),
    )

    args = parser.parse_args()

    validate_required_scripts()

    PIPELINE_REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_ids = (
        discover_patient_ids()
    )

    if args.patient_id is not None:
        requested_patient_id = str(
            args.patient_id
        ).strip()

        patient_ids = [
            patient_id
            for patient_id in patient_ids
            if patient_id
            == requested_patient_id
        ]

        if not patient_ids:
            raise FileNotFoundError(
                "data/processed 中找不到患者："
                f"{requested_patient_id}"
            )

    timeout_seconds: int | None

    if (
        args.step_timeout_minutes
        <= 0
    ):
        timeout_seconds = None
    else:
        timeout_seconds = int(
            round(
                args.step_timeout_minutes
                * 60.0
            )
        )

    print("=" * 80)
    print("All-Patient Treatment Pipeline")
    print("=" * 80)

    print(
        f"Python：{sys.executable}"
    )

    print(
        f"患者數量：{len(patient_ids)}"
    )

    print(
        f"強制重跑：{args.force}"
    )

    print(
        "單一步驟失敗後繼續："
        f"{args.continue_after_failure}"
    )

    print(
        "每步逾時："
        f"{args.step_timeout_minutes} 分鐘"
    )

    print(
        "完成後執行整體驗證："
        f"{not args.no_validation}"
    )

    all_step_rows: list[
        dict[str, Any]
    ] = []

    patient_summary_rows: list[
        dict[str, Any]
    ] = []

    created_junctions: list[
        Path
    ] = []

    pipeline_start = (
        time.perf_counter()
    )

    for patient_index, patient_id in enumerate(
        patient_ids,
        start=1,
    ):
        print()
        print("=" * 80)

        print(
            f"患者 {patient_index}/"
            f"{len(patient_ids)}："
            f"{patient_id}"
        )

        print("=" * 80)

        patient_start = (
            time.perf_counter()
        )

        patient_failed = False

        completed_steps = 0
        skipped_steps = 0
        failed_steps = 0

        try:
            incoming_access = (
                ensure_patient_incoming_access(
                    patient_id
                )
            )

            print(
                "資料來源："
                f"{incoming_access['status']}"
            )

            if incoming_access[
                "junction_created"
            ]:
                junction_path = Path(
                    incoming_access[
                        "incoming_folder"
                    ]
                )

                created_junctions.append(
                    junction_path
                )

                print(
                    "已建立 Junction："
                    f"{junction_path}"
                )

        except Exception as error:
            patient_failed = True
            failed_steps += 1

            setup_result = {
                "patient_id": patient_id,
                "step": (
                    "PREPARE_INCOMING_ACCESS"
                ),
                "script": None,
                "status": (
                    "FAILED_SETUP"
                ),
                "return_code": None,
                "elapsed_seconds": 0.0,
                "expected_output": None,
                "output_exists": False,
                "log_file": None,
                "error_type": (
                    type(error).__name__
                ),
                "error_message": str(
                    error
                ),
            }

            all_step_rows.append(
                setup_result
            )

            print_step_result(
                setup_result
            )

        if not patient_failed:
            for step in PIPELINE_STEPS:
                result = run_patient_step(
                    patient_id=patient_id,
                    step=step,
                    force=args.force,
                    timeout_seconds=(
                        timeout_seconds
                    ),
                )

                all_step_rows.append(
                    result
                )

                print_step_result(
                    result
                )

                status = str(
                    result["status"]
                )

                if status == "SUCCESS":
                    completed_steps += 1

                elif status == (
                    "SKIPPED_OUTPUT_EXISTS"
                ):
                    skipped_steps += 1

                else:
                    failed_steps += 1
                    patient_failed = True

                    if not (
                        args.continue_after_failure
                    ):
                        print(
                            "  後續步驟略過："
                            "目前患者已有步驟失敗。"
                        )
                        break

        patient_elapsed = (
            time.perf_counter()
            - patient_start
        )

        final_recommendation = (
            INFERENCE_ROOT
            / patient_id
            / "treatment_suitability"
            / "personalized_treatment_recommendation.json"
        )

        final_output_exists = bool(
            final_recommendation.exists()
            and final_recommendation.stat().st_size
            > 0
        )

        if (
            not patient_failed
            and final_output_exists
        ):
            patient_status = "SUCCESS"

        elif final_output_exists:
            patient_status = (
                "PARTIAL_WITH_EXISTING_OUTPUT"
            )

        else:
            patient_status = "FAILED"

        patient_summary_rows.append(
            {
                "patient_id": (
                    patient_id
                ),
                "status": (
                    patient_status
                ),
                "completed_steps": (
                    completed_steps
                ),
                "skipped_steps": (
                    skipped_steps
                ),
                "failed_steps": (
                    failed_steps
                ),
                "elapsed_seconds": (
                    patient_elapsed
                ),
                "recommendation_file": (
                    str(
                        final_recommendation
                    )
                ),
                "recommendation_exists": (
                    final_output_exists
                ),
            }
        )

        print(
            "患者結果："
            f"{patient_status} "
            f"({patient_elapsed:.1f} 秒)"
        )

    validation_result: (
        dict[str, Any]
        | None
    ) = None

    if not args.no_validation:
        print()
        print("=" * 80)
        print("執行整體推薦驗證")
        print("=" * 80)

        validation_result = (
            run_validation(
                timeout_seconds=(
                    timeout_seconds
                ),
            )
        )

        all_step_rows.append(
            validation_result
        )

        print_step_result(
            validation_result
        )

    cleanup_rows: list[
        dict[str, Any]
    ] = []

    if args.cleanup_created_junctions:
        print()
        print("=" * 80)
        print("清理本次建立的 Junction")
        print("=" * 80)

        for junction_path in reversed(
            created_junctions
        ):
            try:
                if (
                    junction_path.exists()
                    and is_windows_reparse_point(
                        junction_path
                    )
                ):
                    remove_directory_junction(
                        junction_path
                    )

                    status = "REMOVED"

                else:
                    status = (
                        "SKIPPED_NOT_JUNCTION"
                    )

                print(
                    f"[{status}] "
                    f"{junction_path}"
                )

                cleanup_rows.append(
                    {
                        "junction_path": str(
                            junction_path
                        ),
                        "status": status,
                        "error_type": None,
                        "error_message": None,
                    }
                )

            except Exception as error:
                print(
                    "[FAILED] "
                    f"{junction_path}："
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                cleanup_rows.append(
                    {
                        "junction_path": str(
                            junction_path
                        ),
                        "status": "FAILED",
                        "error_type": (
                            type(error).__name__
                        ),
                        "error_message": (
                            str(error)
                        ),
                    }
                )

    pipeline_elapsed = (
        time.perf_counter()
        - pipeline_start
    )

    patient_report_csv = (
        PIPELINE_REPORT_ROOT
        / "patient_pipeline_summary.csv"
    )

    step_report_csv = (
        PIPELINE_REPORT_ROOT
        / "pipeline_step_results.csv"
    )

    cleanup_report_csv = (
        PIPELINE_REPORT_ROOT
        / "junction_cleanup_results.csv"
    )

    summary_json = (
        PIPELINE_REPORT_ROOT
        / "pipeline_summary.json"
    )

    write_csv(
        patient_report_csv,
        patient_summary_rows,
    )

    write_csv(
        step_report_csv,
        all_step_rows,
    )

    write_csv(
        cleanup_report_csv,
        cleanup_rows,
    )

    patient_status_counts: (
        dict[str, int]
    ) = {}

    for row in patient_summary_rows:
        status = str(
            row["status"]
        )

        patient_status_counts[
            status
        ] = (
            patient_status_counts.get(
                status,
                0,
            )
            + 1
        )

    step_status_counts: (
        dict[str, int]
    ) = {}

    for row in all_step_rows:
        status = str(
            row["status"]
        )

        step_status_counts[
            status
        ] = (
            step_status_counts.get(
                status,
                0,
            )
            + 1
        )

    summary = {
        "project_root": (
            PROJECT_ROOT
        ),
        "python_executable": (
            sys.executable
        ),
        "patient_count": (
            len(patient_ids)
        ),
        "force": (
            args.force
        ),
        "continue_after_failure": (
            args.continue_after_failure
        ),
        "step_timeout_minutes": (
            args.step_timeout_minutes
        ),
        "pipeline_elapsed_seconds": (
            pipeline_elapsed
        ),
        "patient_status_counts": (
            patient_status_counts
        ),
        "step_status_counts": (
            step_status_counts
        ),
        "created_junction_count": (
            len(created_junctions)
        ),
        "cleanup_requested": (
            args.cleanup_created_junctions
        ),
        "validation_result": (
            validation_result
        ),
        "patient_results": (
            patient_summary_rows
        ),
        "step_results": (
            all_step_rows
        ),
        "junction_cleanup_results": (
            cleanup_rows
        ),
    }

    save_json(
        summary_json,
        summary,
    )

    print()
    print("=" * 80)
    print("全患者 Pipeline 完成")
    print("=" * 80)

    print(
        f"患者數量：{len(patient_ids)}"
    )

    print(
        "患者狀態："
        f"{patient_status_counts}"
    )

    print(
        "步驟狀態："
        f"{step_status_counts}"
    )

    print(
        "總執行時間："
        f"{pipeline_elapsed / 60.0:.2f} 分鐘"
    )

    print(
        "本次建立 Junction："
        f"{len(created_junctions)}"
    )

    print(
        f"患者摘要：{patient_report_csv}"
    )

    print(
        f"步驟明細：{step_report_csv}"
    )

    print(
        f"完整摘要：{summary_json}"
    )

    print(
        f"Log 資料夾：{LOG_ROOT}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()