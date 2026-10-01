from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from process_incoming_patient import process_patient
from src.continual_learning.drug_trial import refresh_patient_representation_model


PROJECT_ROOT = Path(__file__).resolve().parent


def run_script(
    script_name: str,
    patient_id: str,
    follow_up: bool = False,
    extra_args: list[str] | None = None,
) -> None:
    """
    以目前 Python 環境執行單一患者分析程式。

    子程式若失敗，subprocess.run(check=True) 會立即拋出例外，
    讓上層 Streamlit Follow-up 流程執行 rollback。
    """
    script_path = PROJECT_ROOT / script_name

    if not script_path.is_file():
        raise FileNotFoundError(
            f"找不到 Pipeline 程式：{script_path}"
        )

    process_environment = os.environ.copy()
    process_environment["PYTHONUNBUFFERED"] = "1"
    process_environment["PYTHONIOENCODING"] = "utf-8"

    command = [
        sys.executable,
        "-u",
        str(script_path),
        "--patient-id",
        patient_id,
    ]
    if follow_up:
        command.append("--follow-up")
    if extra_args:
        command.extend(str(argument) for argument in extra_args)

    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=process_environment,
        check=True,
    )


def print_step(
    step_number: int,
    title: str,
) -> None:
    print()
    print("=" * 80)
    print(f"Pipeline Step {step_number}：{title}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sleep Digital Twin Pipeline"
    )

    parser.add_argument(
        "--patient-id",
        required=True,
        help="患者 ID",
    )
    parser.add_argument(
        "--allow-partial-edf",
        action="store_true",
        help="允許使用不完整 EDF 繼續執行片段分析。",
    )
    parser.add_argument(
        "--follow-up",
        action="store_true",
        help=(
            "使用 PSG Follow-up 模式。"
            "此模式略過新患者建立，但仍執行後續完整分析。"
        ),
    )

    args = parser.parse_args()

    patient_id = str(args.patient_id).strip()
    follow_up_mode = bool(args.follow_up)

    if not patient_id:
        parser.error("--patient-id 不可為空白。")

    print("=" * 80)
    print("Sleep Digital Twin Pipeline")
    print("=" * 80)
    print(f"Patient ID：{patient_id}")
    print(
        "Pipeline Mode："
        + (
            "Follow-up PSG"
            if follow_up_mode
            else "New Patient"
        )
    )

    # ============================================================
    # Pipeline Step 1：患者前處理／Follow-up 模式切換
    # ============================================================
    print_step(
        1,
        (
            "Follow-up PSG 前處理"
            if follow_up_mode
            else "新患者前處理"
        ),
    )

    if follow_up_mode:
        print("Follow-up PSG 模式")
        print("略過新患者建立流程")
        print("使用 prepare_psg_follow_up() 已放入的最新 PSG 輸入檔案")
    else:
        process_patient(
            patient_id=patient_id,
            allow_partial_edf=bool(
                args.allow_partial_edf
            ),
        )

    # ============================================================
    # Pipeline Step 2：Arousal 推論
    # ============================================================
    print_step(2, "Arousal 推論")
    run_script(
        "predict_incoming_patient.py",
        patient_id,
    )

    # ============================================================
    # Pipeline Step 3：Respiratory Profile
    # ============================================================
    print_step(3, "Respiratory Profile")
    run_script(
        "build_patient_respiratory_profile.py",
        patient_id,
    )

    # ============================================================
    # Pipeline Step 4：Position Profile
    # ============================================================
    print_step(4, "Position Profile")
    run_script(
        "build_patient_position_profile.py",
        patient_id,
    )

    # ERJ 2024：姿勢型 OSA 與 PUP endotype 資料完整度研究模組。
    # 僅計算論文可直接重現的姿勢 AHI 定義；不以 proxy 冒充 PUP 結果。
    run_script(
        "build_positional_endotype_research.py",
        patient_id,
    )

    # JSR 2024：CPAP 與生理內型證據層。只輸出現有 PSG 可觀察指標；
    # 未取得並驗證 PUP/PUPpy 呼吸驅動模型前，不產生四種 endotype 假值。
    run_script(
        "build_jsr2024_pap_endotype_research.py",
        patient_id,
    )

    # ============================================================
    # Pipeline Step 5：SpO2 Quality Mask
    # ============================================================
    print_step(5, "SpO2 Quality Mask")
    run_script(
        "build_spo2_quality_mask.py",
        patient_id,
    )

    # ============================================================
    # Pipeline Step 6：Oxygen Event Coupling
    # ============================================================
    print_step(6, "Oxygen Event Coupling")
    run_script(
        "analyze_oxygen_event_coupling.py",
        patient_id,
    )

    # ============================================================
    # Pipeline Step 7：The only official treatment ranking source
    #
    # Do not generate the retired ``treatment_suitability`` score here.
    # It used a different, earlier rule set and could leave a second score
    # (for example APAP=65 versus the final APAP=45) for one patient.
    # ``build_treatment_refinement.py`` is self-contained and is the sole
    # patient-facing treatment-ranking authority.
    # ============================================================
    print_step(7, "Final Treatment Ranking (only official output)")
    run_script(
        "build_treatment_refinement.py",
        patient_id,
        follow_up=follow_up_mode,
    )
    representation_receipt = refresh_patient_representation_model(PROJECT_ROOT)
    print(
        "Treatment patient representation："
        f"{representation_receipt.get('status')}｜"
        f"patients={representation_receipt.get('patient_count', 0)}"
    )

    # ============================================================
    # Pipeline Step 8：Auditable observed-factor summary
    # This formal report source is deterministic and lightweight. Full-night
    # SHAP remains a separate research analysis and cannot silently block the
    # patient pipeline or be confused with rule-based clinical evidence.
    # ============================================================
    print_step(8, "Observed Clinical Factor Summary")
    run_script(
        "build_patient_factor_summary.py",
        patient_id,
    )

    # ============================================================
    # Pipeline Step 9：SleepFM multimodal disease research ranking
    # This is intentionally non-blocking: missing modalities or unsupported
    # PSG channels are recorded in JSON and must not break the clinical pipeline.
    # ============================================================
    print_step(9, "SleepFM Multimodal Disease Research Ranking")
    run_script(
        "run_sleepfm_disease_prediction.py",
        patient_id,
    )

    # ============================================================
    # Pipeline Step 10：Digital Twin Report
    # ============================================================
    print_step(10, "Digital Twin Report")
    run_script(
        "build_patient_digital_twin_report.py",
        patient_id,
        follow_up=follow_up_mode,
    )

    # ============================================================
    # Pipeline Step 11：Tonight risk prediction
    # 先以目前正式模型輸出，再以 Event Grid 真實事件更新下一版模型。
    # ============================================================
    print_step(11, "Tonight Sleep Respiratory Risk")
    run_script(
        "predict_tonight_apnea_risk.py",
        patient_id,
    )

    # Independent research output: a causal single-direction LSTM uses only
    # the preceding seven 30-second epochs to predict an OSA event in the next
    # 30-second epoch. It is deliberately separate from tonight-risk scoring.
    print_step(12, "Next 30-second OSA Event (Causal UniLSTM)")
    run_script(
        "predict_osa_next_epoch_lstm.py",
        patient_id,
    )

    # ============================================================
    # Pipeline Step 13：統一模型更新與可稽核收據
    # 每份新的、有真實標籤的資料都建立 Challenger；重複或無標籤資料
    # 會留下明確原因，不會被誤稱為已完成正式模型學習。
    # ============================================================
    # Retired: former short-horizon Arousal/Apnea model-update audit.

    # ============================================================
    # Pipeline Step 14：以最新正式模型重新輸出
    # Challenger 會完整留存；只有通過驗證並升級的版本才會影響正式輸出。
    # ============================================================
    # Retired: duplicate prediction/report refresh for former short alerts.

    print()
    print("=" * 80)
    print("Sleep Digital Twin Pipeline 完成")
    print("=" * 80)
    print(f"Patient ID：{patient_id}")
    print(
        "Pipeline Mode："
        + (
            "Follow-up PSG"
            if follow_up_mode
            else "New Patient"
        )
    )


if __name__ == "__main__":
    main()
