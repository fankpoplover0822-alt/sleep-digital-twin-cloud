"""Merge physician-confirmed supplemental data into treatment model features."""

from __future__ import annotations

from typing import Any, Mapping


def _present(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {
        "", "0", "false", "no", "none", "normal", "absent",
        "無", "否", "正常", "未見", "unknown",
    }


def _findings(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    record = data.get(key, {})
    if not isinstance(record, Mapping) or record.get("available") is not True:
        return {}
    findings = record.get("findings", {})
    return dict(findings) if isinstance(findings, Mapping) else {}


def merge_clinical_context(
    base_context: Mapping[str, Any],
    clinical_data: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain every confirmed field and derive canonical clinical features."""
    context = dict(base_context)
    data: Mapping[str, Any] = (
        clinical_data if isinstance(clinical_data, Mapping) else {}
    )
    module_keys = (
        "pap", "ent", "dise", "imaging", "hypoxemia", "abg", "pulmonary",
        "cardiac", "ecg", "echocardiography", "comorbidities", "medication",
        "patient_preference", "follow_up", "anthropometry",
        "weight_management", "lifestyle", "sleep_questionnaire",
        "dental_craniofacial", "laboratory", "neurological", "psychiatric",
        "oxygen_therapy", "surgery_history", "device_data", "other",
    )
    received: list[dict[str, Any]] = []
    modules: dict[str, dict[str, Any]] = {}
    trusted_modules: dict[str, dict[str, Any]] = {}
    synthetic_modules: list[str] = []
    for module in module_keys:
        findings = _findings(data, module)
        if not findings:
            continue
        modules[module] = findings
        module_is_synthetic = findings.get("synthetic_test_training") is True
        # Wrist-watch/device exports are longitudinal follow-up observations.
        # They can be retained for audit and displayed with their own sync time,
        # but must not be silently treated as contemporaneous diagnostic PSG
        # evidence or as a direct treatment-ranking rule input.
        module_is_follow_up_only = module == "device_data"
        if module_is_synthetic:
            synthetic_modules.append(module)
        elif not module_is_follow_up_only:
            trusted_modules[module] = findings
        for field, value in findings.items():
            if value is None or value == "":
                continue
            model_feature = f"clinical_{module}_{field}"
            context[model_feature] = value
            received.append(
                {
                    "module": module,
                    "source_field": str(field),
                    "model_feature": model_feature,
                    "value": value,
                    "status": (
                        "audit_only_synthetic_excluded_from_clinical_inference"
                        if module_is_synthetic
                        else (
                            "follow_up_only_not_used_by_treatment_rules"
                            if module_is_follow_up_only
                            else "included_in_treatment_context"
                        )
                    ),
                }
            )

    # Synthetic demonstration modules remain visible in the audit trail but
    # must never override diagnostic PSG values or clinical treatment rules.
    ent = trusted_modules.get("ent", {})
    dise = trusted_modules.get("dise", {})
    imaging = trusted_modules.get("imaging", {})
    dental = trusted_modules.get("dental_craniofacial", {})
    anatomy = {
        "nasal_septal_deviation": ent.get("nasal_septal_deviation"),
        "inferior_turbinate_hypertrophy": ent.get("inferior_turbinate_hypertrophy"),
        "tonsil_hypertrophy": ent.get("tonsil_hypertrophy"),
        "soft_palate_abnormality": ent.get("soft_palate_abnormality"),
        "nasal_obstruction": ent.get("nasal_obstruction"),
        "adenoid_hypertrophy": ent.get("adenoid_hypertrophy"),
        "velum_collapse": dise.get("velum_collapse"),
        "oropharyngeal_lateral_wall_collapse": dise.get(
            "oropharyngeal_lateral_wall_collapse"
        ),
        "tongue_base_collapse": dise.get("tongue_base_collapse"),
        "epiglottis_collapse": dise.get("epiglottis_collapse"),
        "nasal_airway_narrowing": imaging.get("nasal_airway_narrowing"),
        "retropalatal_airway_narrowing": imaging.get(
            "retropalatal_airway_narrowing"
        ),
        "retroglossal_airway_narrowing": imaging.get(
            "retroglossal_airway_narrowing"
        ),
        "craniofacial_abnormality": imaging.get("craniofacial_abnormality"),
        "other_anatomical_obstruction": imaging.get(
            "other_anatomical_obstruction"
        ),
        "retrognathia": dental.get("retrognathia"),
        "micrognathia": dental.get("micrognathia"),
    }
    positive_anatomy = [
        field for field, value in anatomy.items() if _present(value)
    ]
    context["upper_airway_structural_abnormality"] = bool(positive_anatomy)
    context["anatomical_obstruction_confirmed"] = bool(positive_anatomy)
    context["surgical_anatomical_target_confirmed"] = bool(positive_anatomy)
    context["anatomical_findings"] = ", ".join(positive_anatomy)
    context["anatomical_target_count"] = len(positive_anatomy)
    for field, value in anatomy.items():
        if value is not None and value != "":
            context[field] = value

    medication = trusted_modules.get("medication", {})
    questionnaire = trusted_modules.get("sleep_questionnaire", {})
    follow_up = trusted_modules.get("follow_up", {})
    pap = trusted_modules.get("pap", {})
    context["current_medications"] = medication.get("medication_list")
    context["sedative_use"] = _present(medication.get("sedative_hypnotics"))
    context["opioid_use"] = _present(medication.get("opioids"))
    context["respiratory_depressant_use"] = _present(
        medication.get("respiratory_depressants")
    )
    context["isi_score"] = questionnaire.get("isi_score")
    try:
        isi_positive = float(questionnaire.get("isi_score")) >= 15
    except (TypeError, ValueError):
        isi_positive = False
    context["insomnia_diagnosis"] = bool(
        _present(questionnaire.get("insomnia_diagnosis"))
        or _present(medication.get("insomnia_diagnosis"))
        or isi_positive
    )
    context["pap_tolerance"] = follow_up.get("tolerance") or pap.get("tolerance")
    context["residual_ahi"] = (
        follow_up.get("residual_ahi")
        if follow_up.get("residual_ahi") is not None
        else pap.get("residual_ahi")
    )
    context["pap_adherence_hours_per_night"] = (
        follow_up.get("nightly_usage_hours")
        if follow_up.get("nightly_usage_hours") is not None
        else pap.get("adherence_hours_per_night")
    )
    context["pap_adherence_percentage"] = pap.get("adherence_percentage")
    context["pap_pressure_setting"] = pap.get("pressure_setting")
    context["pap_mask_leak"] = pap.get("mask_leak")
    context["pap_treatment_spo2"] = pap.get("treatment_spo2")

    # Canonical values below are consumed by the safety guardrails.  Keeping
    # only the clinical_* audit copy made uploads visible in the audit trail
    # but unable to change the four treatment scores.
    anthropometry = trusted_modules.get("anthropometry", {})
    if anthropometry.get("bmi") is not None:
        # The treatment rules consume the canonical upper-case BMI field.
        # Preserve aliases for model compatibility, but make the newest
        # clinician-confirmed anthropometry value authoritative.
        context["BMI"] = anthropometry.get("bmi")
        context["bmi"] = anthropometry.get("bmi")
        context["body_mass_index"] = anthropometry.get("bmi")
    context["ess_score"] = questionnaire.get("ess_score")
    context["stop_bang_score"] = questionnaire.get("stop_bang_score")
    context["psqi_score"] = questionnaire.get("psqi_score")

    hypoxemia = trusted_modules.get("hypoxemia", {})
    if hypoxemia.get("minimum_spo2") is not None:
        context["minimum_spo2"] = hypoxemia.get("minimum_spo2")
        context["min_spo2"] = hypoxemia.get("minimum_spo2")

    preference = trusted_modules.get("patient_preference", {})
    for field in (
        "accept_cpap", "accept_apap", "accept_surgery",
        "cpap_preference", "apap_preference", "surgery_preference",
        "medication_preference", "main_concern", "treatment_goal",
        "primary_treatment_goal",
    ):
        if preference.get(field) is not None:
            context[f"clinical_patient_preference_{field}"] = preference.get(field)

    # Wearable/watch exports are follow-up signals, not a replacement for a
    # diagnostic PSG. Keep their provenance explicit, but never allow them to
    # change the PSG-based treatment score merely because an old upload remains
    # in the patient's clinical-data record.
    device = modules.get("device_data", {})
    context["wearable_device_name"] = device.get("device_name")
    context["wearable_estimated_ahi"] = device.get("estimated_ahi")
    context["wearable_average_spo2"] = device.get("average_spo2")
    context["wearable_minimum_spo2"] = device.get("minimum_spo2")
    context["wearable_time_below_90_percent"] = device.get(
        "time_below_90_percent"
    )
    context["wearable_average_heart_rate"] = device.get(
        "average_heart_rate"
    )
    context["wearable_supine_sleep_percent"] = device.get(
        "supine_sleep_percent"
    )
    context["wearable_apnea_event_count"] = device.get(
        "apnea_event_count"
    )
    context["wearable_sensor_data_points"] = device.get(
        "sensor_data_points"
    )
    # Synthetic scenarios are retained for audit only. A wearable signal can
    # never establish PAP intolerance, patient preference, or airway anatomy.
    synthetic_scenario = str(device.get("synthetic_scenario") or "").strip()
    context["synthetic_wearable_scenario_present"] = bool(synthetic_scenario)
    context["synthetic_wearable_scenario_applied"] = False
    context["synthetic_data_excluded_from_clinical_inference"] = bool(
        synthetic_modules or synthetic_scenario
    )
    context["synthetic_clinical_modules_excluded"] = sorted(synthetic_modules)
    context["wearable_data_role"] = (
        "historical_follow_up_only_not_used_by_treatment_rules"
        if device
        else "not_available"
    )
    try:
        wearable_ahi_alert = float(device.get("estimated_ahi")) >= 15
    except (TypeError, ValueError):
        wearable_ahi_alert = False
    try:
        wearable_spo2_alert = float(device.get("minimum_spo2")) < 90
    except (TypeError, ValueError):
        wearable_spo2_alert = False
    context["wearable_follow_up_alert"] = bool(
        wearable_ahi_alert or wearable_spo2_alert
    )
    context["clinical_data_version"] = data.get("data_version")
    context["clinical_last_updated_module"] = data.get("last_updated_module")
    context["clinical_feature_count"] = len(received)

    return context, {
        "status": "connected_to_treatment_context",
        "data_version": data.get("data_version"),
        "last_updated_module": data.get("last_updated_module"),
        "feature_count": len(received),
        "modules_received": sorted(modules),
        "trusted_modules_used_by_clinical_rules": sorted(trusted_modules),
        "follow_up_only_modules_not_used_by_treatment_rules": (
            ["device_data"] if device else []
        ),
        "wearable_data_role": context["wearable_data_role"],
        "synthetic_modules_excluded_from_clinical_rules": sorted(synthetic_modules),
        "features": received,
        "all_non_empty_fields_retained": True,
    }
