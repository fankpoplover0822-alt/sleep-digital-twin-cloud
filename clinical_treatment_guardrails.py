"""Clinically grounded guardrails for the four treatment candidates.

These rules are deliberately applied after any learned score adjustment.  A
small or weakly labelled machine-learning dataset must never override a
clinical prerequisite such as a documented anatomical target before surgery.
The output remains decision support and is not a prescription.
"""

from __future__ import annotations

from typing import Any


GUIDELINE_REFERENCES = [
    {
        "title": "AASM PAP treatment guideline (2019)",
        "url": "https://doi.org/10.5664/jcsm.7640",
        "use": "PAP is recommended therapy for adult obstructive sleep apnea; CPAP or APAP selection requires clinical context.",
    },
    {
        "title": "AASM surgical referral guideline (2021)",
        "url": "https://doi.org/10.5664/jcsm.9592",
        "use": "Surgery is a specialist-referral decision; PAP is generally considered first even when a major anatomical abnormality is present.",
    },
    {
        "title": "AASM adult OSA severity definitions",
        "url": "https://aasm.org/resources/factsheets/sleepapnea.pdf",
        "use": "Mild AHI 5–15, moderate 15–30, severe >30 events/hour.",
    },
    {
        "title": "Loop gain predicts upper-airway surgical response",
        "url": "https://pubmed.ncbi.nlm.nih.gov/28531336/",
        "use": "Elevated loop gain is a non-anatomical mechanism associated with poorer upper-airway surgical response.",
    },
    {
        "title": "FDA approval: tirzepatide for OSA with obesity (2024)",
        "url": "https://www.fda.gov/news-events/press-announcements/fda-approves-first-medication-obstructive-sleep-apnea",
        "use": "Approved only for adults with obesity and moderate-to-severe OSA, together with reduced-calorie diet and increased physical activity.",
    },
    {
        "title": "SURMOUNT-OSA randomized phase 3 trials (2024)",
        "url": "https://pubmed.ncbi.nlm.nih.gov/38912654/",
        "use": "Tirzepatide reduced AHI, hypoxic burden and body weight in adults with obesity and moderate-to-severe OSA.",
    },
    {
        "title": "AD109 phase 3 trial (2026; investigational)",
        "url": "https://pubmed.ncbi.nlm.nih.gov/42148495/",
        "use": "Aroxybutynin plus atomoxetine improved AHI in PAP-intolerant/refusing adults, but remains investigational and had substantial adverse-event discontinuation.",
    },
]


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {
        "1", "true", "yes", "y", "present", "abnormal", "confirmed",
        "有", "是", "異常", "已確認",
    }


def _text(context: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = context.get(key)
        if value not in (None, "", "UNKNOWN"):
            return str(value)
    return ""


def _has_structural_target(context: dict[str, Any]) -> bool:
    direct_keys = (
        "upper_airway_structural_abnormality",
        "anatomical_obstruction_confirmed",
        "surgical_anatomical_target_confirmed",
        "dise_target_confirmed",
    )
    if any(_flag(context.get(key)) for key in direct_keys):
        return True
    description = _text(
        context,
        "upper_airway_anatomy",
        "anatomical_findings",
        "ent_findings",
        "dise_findings",
    ).lower()
    positive_terms = (
        "tonsil", "retrognath", "maxill", "palatal collapse",
        "tongue base", "nasal obstruction", "structural abnormal",
        "扁桃腺", "下顎後縮", "顎骨", "腭", "舌根", "鼻阻塞", "結構異常",
    )
    return any(term in description for term in positive_terms)


def _append_factor(
    item: dict[str, Any],
    bucket: str,
    code: str,
    text: str,
    score_change: float = 0.0,
) -> None:
    factors = item.setdefault(bucket, [])
    if not isinstance(factors, list):
        factors = []
        item[bucket] = factors
    if any(isinstance(row, dict) and row.get("code") == code for row in factors):
        return
    factors.append(
        {
            "code": code,
            "description": text,
            "score_change": score_change,
            "weight": score_change,
        }
    )


def _set_score(item: dict[str, Any], score: float) -> None:
    item["score"] = round(max(0.0, min(100.0, score)), 1)
    item["suitability_level"] = (
        "HIGH" if item["score"] >= 70
        else "MODERATE" if item["score"] >= 45
        else "LOW"
    )


def apply_medical_guardrails(
    treatments: list[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply AHI/airflow/position/SpO2/anatomy/loop-gain guardrails."""
    ahi = _number(context.get("ahi"))
    severity = str(context.get("ahi_severity") or "UNKNOWN").upper()
    min_spo2 = _number(
        context.get("min_spo2")
        or context.get("minimum_spo2")
        or context.get("spo2_nadir")
    )
    oxygen_burden = str(context.get("oxygen_burden") or "UNKNOWN").upper()
    central_fraction = _number(context.get("central_event_fraction"))
    obstructive_fraction = _number(context.get("obstructive_event_fraction"))
    loop_gain = str(context.get("loop_gain_level") or "UNKNOWN").upper()
    position_relevance = str(
        context.get("position_relevance") or "UNKNOWN"
    ).upper()
    anatomy_confirmed = _has_structural_target(context)
    pap_tolerance = str(context.get("pap_tolerance") or "").strip().lower()
    poor_pap_tolerance = any(
        term in pap_tolerance
        for term in (
            "poor", "intolerant", "cannot tolerate", "unable",
            "差", "不耐受", "無法耐受", "拒絕",
        )
    )
    bmi = _number(context.get("bmi") or context.get("body_mass_index"))
    obesity = bmi is not None and bmi >= 30.0
    patient_preference_text = " ".join(
        str(context.get(key) or "")
        for key in (
            "clinical_patient_preference_accept_cpap",
            "clinical_patient_preference_accept_apap",
            "clinical_patient_preference_cpap_preference",
            "clinical_patient_preference_apap_preference",
        )
    ).lower()
    pap_refusal = poor_pap_tolerance or any(
        term in patient_preference_text
        for term in ("false", "refused", "decline", "reject", "拒絕", "不接受")
    )
    insomnia = _flag(context.get("insomnia_diagnosis"))
    isi_score = _number(context.get("isi_score"))
    ess_score = _number(context.get("ess_score"))
    sedative_use = _flag(context.get("sedative_use"))
    opioid_use = _flag(context.get("opioid_use"))
    respiratory_depressant_use = _flag(
        context.get("respiratory_depressant_use")
    )

    lookup = {
        str(item.get("treatment", "")).upper(): item
        for item in treatments
        if isinstance(item, dict)
    }
    cpap = lookup.get("CPAP")
    apap = lookup.get("APAP")
    surgery = lookup.get("SURGERY") or lookup.get("SURGERY_EVALUATION")
    medication = lookup.get(
        "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW"
    )

    hypoxemia = (
        oxygen_burden in {"HIGH", "HIGH_BURDEN", "MODERATE_BURDEN"}
        or (min_spo2 is not None and min_spo2 < 90)
    )

    if cpap:
        cpap_score = _number(cpap.get("score")) or 0.0
        if loop_gain == "HIGH":
            _append_factor(
                cpap, "supporting_factors", "HIGH_LOOP_GAIN_FAVORS_PAP_OVER_SURGERY",
                "Loop gain 偏高屬非解剖性呼吸控制不穩定；本系統優先保留 PAP 評估，避免僅靠上呼吸道手術。",
            )
        if poor_pap_tolerance:
            cpap_score -= 15.0
            _append_factor(
                cpap, "limiting_factors", "DOCUMENTED_PAP_INTOLERANCE",
                "補充資料記錄 PAP 耐受度不佳；不代表 PAP 無效，但需改善面罩、壓力、加濕、失眠或焦慮問題，並重新討論替代方案。",
                -15.0,
            )
        _set_score(cpap, cpap_score)
        _append_factor(
            cpap, "supporting_factors", "PAP_AHI_SPO2_BASIS",
            "CPAP 以正壓氣流維持上呼吸道開放；AHI 越高或事件伴隨明顯 SpO₂ 下降時，PAP 優先度提高。CPAP 本身不等同額外供氧。",
        )

    if apap:
        score = _number(apap.get("score")) or 0.0
        if loop_gain == "HIGH" or (central_fraction is not None and central_fraction >= 0.20):
            previous_score = score
            score = min(score, 68.0)
            _append_factor(
                apap, "limiting_factors", "VENTILATORY_INSTABILITY_REQUIRES_TITRATION",
                "Loop gain 偏高或中樞事件比例偏高時，不應只靠自動壓力演算法決定治療；需睡眠專科評估與壓力滴定。",
                score - previous_score,
            )
        if poor_pap_tolerance:
            score -= 15.0
            _append_factor(
                apap, "limiting_factors", "DOCUMENTED_PAP_INTOLERANCE",
                "醫師補充資料記錄 PAP 耐受度不佳；需先釐清壓力、漏氣、面罩、鼻阻力或共病失眠。",
                -15.0,
            )
        _set_score(apap, score)
        _append_factor(
            apap, "supporting_factors", "APAP_UNCOMPLICATED_OSA_OPTION",
            "APAP 可作為部分成人 OSA 的起始或持續 PAP 方式；仍須依殘餘 AHI、漏氣、耐受度、共病與血氧追蹤調整。",
        )

    surgical_referral_eligible = bool(
        ahi is not None
        and ahi >= 5
        and (bmi is None or bmi < 40)
        and pap_refusal
        and not (central_fraction is not None and central_fraction >= 0.20)
    )
    procedure_target_supported = bool(
        anatomy_confirmed
        and loop_gain != "HIGH"
        and not (central_fraction is not None and central_fraction >= 0.20)
    )
    if surgery:
        score = _number(surgery.get("score")) or 0.0
        if surgical_referral_eligible:
            score += 35.0
            _append_factor(
                surgery, "supporting_factors", "AASM_SURGICAL_REFERRAL_DISCUSSION",
                "患者有 OSA 且已記錄無法耐受或不接受 PAP；依 AASM 指引，睡眠外科轉介已成為可執行的替代治療路徑。此分數代表專科評估優先度，不代表已選定或保證接受特定手術。",
                35.0,
            )
            if not anatomy_confirmed:
                _append_factor(
                    surgery, "limiting_factors", "ANATOMY_REQUIRED_BEFORE_PROCEDURE_SELECTION",
                    "尚無鼻內視鏡、DISE、影像或顱顏評估確認阻塞標的；不得選定手術方式。",
                )
        elif ahi is not None and ahi >= 5 and anatomy_confirmed:
            _append_factor(
                surgery, "limiting_factors", "PAP_FIRST_OR_SHARED_DECISION_REQUIRED",
                "已有解剖異常，但未確認 PAP 不耐受或拒絕；應先完成 PAP 討論、教育與問題排除，再共同決策是否轉介。",
            )
        else:
            previous_score = score
            reasons = []
            observed_safety_reasons = []
            if ahi is None:
                reasons.append("缺少 AHI")
            elif ahi < 5:
                observed_safety_reasons.append("尚未達 OSA 診斷範圍")
            if not anatomy_confirmed:
                reasons.append("缺少可供手術選擇的解剖資料")
            if bmi is not None and bmi >= 40:
                observed_safety_reasons.append("BMI ≥40，應優先討論減重／肥胖專科並個別評估手術風險")
            if not pap_refusal:
                reasons.append("未確認 PAP 無法耐受或不接受")
            if loop_gain == "HIGH":
                observed_safety_reasons.append("loop gain 偏高，單純結構手術反應可能較差")
            if central_fraction is not None and central_fraction >= 0.20:
                observed_safety_reasons.append("中樞型事件比例偏高")
            # Missing examinations reduce confidence and block procedure
            # selection, but must not change the numerical treatment score.
            # Only observed clinical contraindications may constrain a score.
            if observed_safety_reasons:
                score = min(score, 25.0)
            reasons.extend(observed_safety_reasons)
            _append_factor(
                surgery, "limiting_factors", "SURGERY_HARD_SAFETY_GATE",
                "；".join(reasons) + "。因此不得由機器學習分數提升為主要治療。",
                score - previous_score,
            )
            surgery["recommendation_status"] = "NOT_ELIGIBLE_PENDING_SPECIALIST_EVALUATION"
        if procedure_target_supported:
            score += 20.0
            _append_factor(
                surgery, "supporting_factors", "CONFIRMED_ANATOMICAL_TARGET",
                "耳鼻喉、影像或 DISE 已確認可評估的上呼吸道阻塞標的；提高外科專科評估優先度，但仍須由醫師決定術式。",
                20.0,
            )
        _set_score(surgery, score)
        surgery["clinical_eligibility"] = procedure_target_supported
        surgery["surgical_referral_eligible"] = surgical_referral_eligible
        surgery["procedure_selection_requires_specialist"] = True

    if medication:
        # This is a triage score for physician review, not a prescription and
        # not a probability of successful treatment.
        score = _number(medication.get("score")) or 0.0
        medication_review_reasons = []

        if ahi is not None and 5 <= ahi < 15:
            _append_factor(
                medication, "supporting_factors", "MILD_OSA_PHARMACOTHERAPY_RESEARCH_REVIEW",
                "AHI 為輕度範圍；若患者無法耐受或拒絕 PAP，可由睡眠專科討論研究中的藥物機轉。目前不應當作一般已核准的輕度 OSA 處方。",
            )
            if pap_refusal:
                _append_factor(
                    medication, "supporting_factors", "PAP_INTOLERANT_INVESTIGATIONAL_OPTION_REVIEW",
                    "已記錄 PAP 耐受不佳或拒絕；這只提高專科／臨床試驗評估的優先度，不等於推薦特定藥物。",
                )

        if ahi is not None and ahi >= 15 and obesity:
            medication_review_reasons.append(
                "成人肥胖且為中重度 OSA，符合 tirzepatide 核准適應症的初步篩檢條件"
            )
            _append_factor(
                medication, "supporting_factors", "FDA_APPROVED_TIRZEPATIDE_SCREEN",
                "FDA 已核准 tirzepatide 用於成人肥胖合併中重度 OSA，並需同時配合低熱量飲食與增加活動。此處僅表示應轉介醫師評估適應症、禁忌與不良反應。",
            )
        if insomnia:
            medication_review_reasons.append("已有失眠診斷或 ISI ≥15")
        if isi_score is not None and isi_score >= 15:
            medication_review_reasons.append(f"ISI={isi_score:g}，達中度以上失眠症狀篩檢範圍")
        if ess_score is not None and ess_score > 10:
            medication_review_reasons.append(f"ESS={ess_score:g}，有日間嗜睡，需檢視治療充分性與致嗜睡藥物")
        if sedative_use:
            medication_review_reasons.append("目前使用鎮靜安眠藥")
        if opioid_use or respiratory_depressant_use:
            medication_review_reasons.append("目前使用鴉片類或呼吸抑制藥物")
        if medication_review_reasons:
            _append_factor(
                medication, "supporting_factors", "PHYSICIAN_MEDICATION_REVIEW_TRIGGER",
                "；".join(medication_review_reasons)
                + "。此分數代表需要醫師進行藥物安全與共病失眠評估，不代表建議用藥治療 OSA。",
            )
        _set_score(medication, score)
        medication["treatment_label"] = "藥物／體重與睡眠共病專科評估"
        medication["evidence_classification"] = {
            "approved_path": (
                "FDA-approved tirzepatide evaluation"
                if ahi is not None and ahi >= 15 and obesity
                else None
            ),
            "investigational_path": (
                "OSA pharmacotherapy research/clinical-trial review"
                if ahi is not None and 5 <= ahi < 15
                else None
            ),
            "comorbidity_path": "insomnia and medication-safety review",
        }
        _append_factor(
            medication, "limiting_factors", "NOT_PRIMARY_OSA_THERAPY",
            "除成人肥胖合併中重度 OSA 的 tirzepatide 核准路徑外，其他直接用於 OSA 的藥物多仍屬研究中。鎮靜或影響呼吸的藥物可能惡化呼吸事件，必須由醫師審查。",
        )

    # Preference is a shared-decision modifier, never a claim of physiologic
    # efficacy.  Limit its magnitude so it cannot override medical gates.
    preference_fields = {
        "CPAP": (cpap, "clinical_patient_preference_accept_cpap", "clinical_patient_preference_cpap_preference"),
        "APAP": (apap, "clinical_patient_preference_accept_apap", "clinical_patient_preference_apap_preference"),
        "SURGERY": (surgery, "clinical_patient_preference_accept_surgery", "clinical_patient_preference_surgery_preference"),
        "MEDICATION": (medication, None, "clinical_patient_preference_medication_preference"),
    }
    for label, (item, accept_key, text_key) in preference_fields.items():
        if not item:
            continue
        preference_value = str(context.get(text_key) or "").strip().lower()
        accept_value = context.get(accept_key) if accept_key else None
        positive = accept_value is True or any(term in preference_value for term in ("prefer", "accept", "願意", "偏好", "接受"))
        negative = accept_value is False or any(term in preference_value for term in ("refuse", "decline", "reject", "不接受", "拒絕"))
        if positive:
            _set_score(item, (_number(item.get("score")) or 0.0) + 3.0)
            _append_factor(item, "supporting_factors", "SHARED_DECISION_ACCEPTANCE", f"患者表示可接受／偏好 {label}；僅作共同決策的小幅調整，不代表療效證據。", 3.0)
        elif negative:
            _set_score(item, (_number(item.get("score")) or 0.0) - 5.0)
            _append_factor(item, "limiting_factors", "SHARED_DECISION_DECLINED", f"患者目前不接受 {label}；需說明選項並共同決策，不代表醫療適應症消失。", -5.0)

    tie_priority = {
        "CPAP": 4,
        "APAP": 3,
        "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW": 2,
        "SURGERY": 1,
        "SURGERY_EVALUATION": 1,
    }
    treatments.sort(
        key=lambda row: (
            _number(row.get("score")) or 0.0,
            _number((row.get("confidence") or {}).get("completeness")) or 0.0,
            tie_priority.get(str(row.get("treatment", "")).upper(), 0),
        ),
        reverse=True,
    )
    score_counts = {}
    for item in treatments:
        score_key = round(_number(item.get("score")) or 0.0, 8)
        score_counts[score_key] = score_counts.get(score_key, 0) + 1
    previous_score = None
    dense_rank = 0
    for position, item in enumerate(treatments, 1):
        score_key = round(_number(item.get("score")) or 0.0, 8)
        if previous_score is None or score_key != previous_score:
            dense_rank = position
            previous_score = score_key
        item["rank"] = dense_rank
        item["is_tied"] = score_counts[score_key] > 1
        item["tie_break_note"] = (
            "同分治療顯示相同名次；卡片順序僅依證據完整度排列，"
            "不代表治療分數較高。"
            if item["is_tied"]
            else None
        )
        item["score_semantics"] = (
            "此數值是由0分起算的共同評估優先度淨分；四種候選皆以強適應條件"
            "約35–45分、重要路徑支持約15–25分、小幅修正約2–10分的同級幅度計算，"
            "供四種候選方向相對排序；"
            "0–100是顯示邊界，不是百分比、成功率或疾病機率，"
            "且尚未經外部前瞻性臨床驗證。"
        )
        treatment_code = str(item.get("treatment", "")).upper()
        item["evaluation_role"] = {
            "CPAP": "OSA主要呼吸支持治療評估",
            "APAP": "OSA主要呼吸支持治療評估",
            "SURGERY": "OSA上呼吸道手術介入評估",
            "SURGERY_EVALUATION": "OSA上呼吸道手術介入評估",
            "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW": (
                "OSA相關藥物、體重與睡眠共病專科介入評估"
            ),
        }.get(treatment_code, "OSA治療或專科介入評估")
        item["externally_clinically_validated_score"] = False
        item["clinical_use_status"] = "RESEARCH_DECISION_SUPPORT_ONLY"

    missing_data = []
    if ahi is None:
        missing_data.append("完整 PSG 的 AHI")
    if obstructive_fraction is None:
        missing_data.append("Nasal Airflow 搭配胸腹動作所判定的阻塞／中樞事件型態")
    if min_spo2 is None and oxygen_burden == "UNKNOWN":
        missing_data.append("SpO₂ 最低值、T90 與事件後下降幅度")
    if position_relevance == "UNKNOWN":
        missing_data.append("XYZ codebook 轉換後的仰臥／側臥／俯臥事件率")
    if not anatomy_confirmed:
        missing_data.append("耳鼻喉檢查、影像或 DISE 確認的上呼吸道結構異常")

    basis = {
        "status": "clinical_guardrails_applied_after_machine_learning",
        "decision_support_only": True,
        "score_validation": {
            "meaning": "relative treatment-fit/referral-priority index",
            "not_a_probability": True,
            "not_a_treatment_success_rate": True,
            "externally_prospectively_validated": False,
            "weights_are_local_transparent_heuristics": True,
        },
        "ahi_events_per_hour": ahi,
        "ahi_severity": severity,
        "signals_used": [
            "Nasal Airflow／呼吸事件型態",
            "XYZ codebook 推算的睡眠姿勢",
            "SpO₂ 缺氧程度",
            "AHI",
            "上呼吸道解剖結構",
            "loop gain 與中樞事件比例",
            "PAP 耐受度及追蹤療效（若有）",
        ],
        "surgery_referral": {
            "rule": "依 AASM 指引，成人 OSA、BMI<40 且 PAP 無法耐受或不接受時可討論轉介睡眠外科；特定術式仍需解剖、共病及手術風險評估。",
            "referral_eligible": surgical_referral_eligible,
            "procedure_target_supported": procedure_target_supported,
        },
        "other_options_not_in_four_way_rank": [
            {
                "option": "口腔矯治器／下顎前移裝置評估",
                "consider_when": "成人 OSA 不接受或無法耐受 CPAP，或偏好替代療法；需由合格牙科睡眠專業人員評估。",
            },
            {
                "option": "體位治療評估",
                "consider_when": "仰臥 AHI 明顯高於非仰臥 AHI，且非仰臥事件負荷較低。",
            },
            {
                "option": "生活型態與體重管理",
                "consider_when": "所有患者均應評估；過重或肥胖者尤其重要，但不應任意取代有適應症的 PAP。",
            },
        ],
        "missing_data": missing_data,
        "guideline_references": GUIDELINE_REFERENCES,
        "important_correction": "CPAP/APAP 是正壓氣流治療，不等同於補充氧氣；是否另需氧氣必須由醫師依持續低血氧原因決定。",
        "supplemental_data_connection": {
            "received_feature_count": int(
                _number(context.get("clinical_feature_count")) or 0
            ),
            "last_updated_module": context.get(
                "clinical_last_updated_module"
            ),
            "used_now_by_rules": {
                "upper_airway_anatomy": anatomy_confirmed,
                "pap_tolerance": bool(pap_tolerance),
                "insomnia_diagnosis": insomnia,
                "sedative_use": sedative_use,
                "opioid_or_respiratory_depressant_use": (
                    opioid_use or respiratory_depressant_use
                ),
            },
            "retained_for_model_learning_even_if_score_change_is_zero": True,
        },
    }
    return treatments, basis
