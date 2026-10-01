from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RuleResult:
    """
    Result produced by one clinical rule.
    """
    status: str
    level: str
    reason: str
    value: Any = None
    threshold: Any = None
    evidence: Dict[str, Any] = field(default_factory=dict)


class RuleEngine:
    """
    Evaluate structured sleep-clinical data produced by ClinicalParser.

    Notes
    -----
    - This module performs descriptive rule evaluation.
    - It does not make diagnoses or final treatment decisions.
    - Missing inputs produce "not_evaluable" rather than guessing.
    """

    ENGINE_VERSION = "1.0.0"

    def evaluate(
        self,
        parsed_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a ClinicalParser result.

        Parameters
        ----------
        parsed_result:
            Dictionary returned by ClinicalParser.parse() or
            ClinicalParser.parse_document().
        """
        if not isinstance(parsed_result, dict):
            parsed_result = {}

        data = parsed_result.get("data")
        if not isinstance(data, dict):
            data = {}

        derived = parsed_result.get("derived")
        if not isinstance(derived, dict):
            derived = {}

        parser_metadata = parsed_result.get("metadata")
        if not isinstance(parser_metadata, dict):
            parser_metadata = {}

        rules: Dict[str, RuleResult] = {
            "osa_severity": self._evaluate_osa_severity(data, derived),
            "hypoxemia_risk": self._evaluate_hypoxemia(data, derived),
            "positional_component": self._evaluate_positional(data, derived),
            "bmi_risk": self._evaluate_bmi(data),
            "sleep_efficiency_status": self._evaluate_sleep_efficiency(data),
            "rem_dominance": self._evaluate_rem_dominance(data),
            "data_completeness": self._evaluate_data_completeness(data),
            "parser_reliability": self._evaluate_parser_reliability(
                parsed_result
            ),
        }

        priority_flags = self._build_priority_flags(rules)
        triggered_count = sum(
            rule.status == "triggered"
            for rule in rules.values()
        )
        not_evaluable_count = sum(
            rule.status == "not_evaluable"
            for rule in rules.values()
        )

        result: Dict[str, Any] = {
            "patient_id": data.get("patient_id"),
            "rules": {
                name: asdict(rule)
                for name, rule in rules.items()
            },
            "priority_flags": priority_flags,
            "summary": {
                "highest_priority": self._highest_priority(priority_flags),
                "triggered_rule_count": triggered_count,
                "not_evaluable_rule_count": not_evaluable_count,
                "requires_clinical_review": bool(priority_flags),
            },
            "metadata": {
                "engine": self.__class__.__name__,
                "engine_version": self.ENGINE_VERSION,
                "source_parser": parser_metadata.get("parser"),
                "source_parser_version": parser_metadata.get(
                    "parser_version"
                ),
            },
            "warnings": [],
        }

        source_warnings = parsed_result.get("warnings")
        if isinstance(source_warnings, list) and source_warnings:
            result["warnings"].append(
                f"ClinicalParser 回傳 {len(source_warnings)} 個警告；"
                "規則結果應配合原始欄位與信心分數審閱。"
            )

        return self._make_json_compatible(result)

    def _evaluate_osa_severity(
        self,
        data: Dict[str, Any],
        derived: Dict[str, Any],
    ) -> RuleResult:
        ahi = self._number(data.get("ahi"))
        if ahi is None:
            return self._not_evaluable("缺少 AHI，無法判定 OSA 嚴重度。")

        if ahi < 5:
            return RuleResult(
                status="not_triggered",
                level="none",
                reason=f"AHI {ahi:g} < 5。",
                value=ahi,
                threshold={"normal": "<5"},
                evidence={"ahi": ahi},
            )

        if ahi < 15:
            level = "low"
            label = "mild"
            threshold = "5–14.9"
        elif ahi < 30:
            level = "moderate"
            label = "moderate"
            threshold = "15–29.9"
        else:
            level = "high"
            label = "severe"
            threshold = "≥30"

        return RuleResult(
            status="triggered",
            level=level,
            reason=f"AHI {ahi:g}，落在 {label} 範圍（{threshold}）。",
            value=ahi,
            threshold=threshold,
            evidence={
                "ahi": ahi,
                "derived_ahi_severity": derived.get("ahi_severity"),
            },
        )

    def _evaluate_hypoxemia(
        self,
        data: Dict[str, Any],
        derived: Dict[str, Any],
    ) -> RuleResult:
        minimum_spo2 = self._number(data.get("minimum_spo2"))
        mean_spo2 = self._number(data.get("mean_spo2"))

        if minimum_spo2 is None:
            return self._not_evaluable(
                "缺少最低 SpO2，無法判定低血氧風險。"
            )

        if minimum_spo2 < 80:
            level = "high"
            reason = f"最低 SpO2 為 {minimum_spo2:g}%，低於 80%。"
            threshold = "<80%"
        elif minimum_spo2 < 90:
            level = "moderate"
            reason = (
                f"最低 SpO2 為 {minimum_spo2:g}%，介於 80% 至 89.9%。"
            )
            threshold = "80–89.9%"
        else:
            return RuleResult(
                status="not_triggered",
                level="none",
                reason=f"最低 SpO2 為 {minimum_spo2:g}%，未低於 90%。",
                value=minimum_spo2,
                threshold="≥90%",
                evidence={
                    "minimum_spo2": minimum_spo2,
                    "mean_spo2": mean_spo2,
                },
            )

        return RuleResult(
            status="triggered",
            level=level,
            reason=reason,
            value=minimum_spo2,
            threshold=threshold,
            evidence={
                "minimum_spo2": minimum_spo2,
                "mean_spo2": mean_spo2,
                "derived_hypoxemia_level": derived.get(
                    "hypoxemia_level"
                ),
            },
        )

    def _evaluate_positional(
        self,
        data: Dict[str, Any],
        derived: Dict[str, Any],
    ) -> RuleResult:
        supine_ahi = self._number(data.get("supine_ahi"))
        non_supine_ahi = self._number(data.get("non_supine_ahi"))

        if supine_ahi is None or non_supine_ahi is None:
            return self._not_evaluable(
                "缺少仰臥或非仰臥 AHI，無法判定體位相關性。"
            )

        if non_supine_ahi == 0:
            if supine_ahi > 0:
                return RuleResult(
                    status="triggered",
                    level="high",
                    reason=(
                        "非仰臥 AHI 為 0 且仰臥 AHI 大於 0，"
                        "呈現明顯體位差異。"
                    ),
                    value=None,
                    threshold="supine/non-supine ≥2",
                    evidence={
                        "supine_ahi": supine_ahi,
                        "non_supine_ahi": non_supine_ahi,
                    },
                )
            return RuleResult(
                status="not_triggered",
                level="none",
                reason="仰臥與非仰臥 AHI 均為 0。",
                evidence={
                    "supine_ahi": supine_ahi,
                    "non_supine_ahi": non_supine_ahi,
                },
            )

        ratio = round(supine_ahi / non_supine_ahi, 3)

        if ratio >= 2:
            level = "high" if ratio >= 4 else "moderate"
            return RuleResult(
                status="triggered",
                level=level,
                reason=(
                    f"仰臥 AHI / 非仰臥 AHI = {ratio:g}，"
                    "達到體位相關門檻 2.0。"
                ),
                value=ratio,
                threshold="≥2.0",
                evidence={
                    "supine_ahi": supine_ahi,
                    "non_supine_ahi": non_supine_ahi,
                    "derived_position_relevance": derived.get(
                        "position_relevance"
                    ),
                },
            )

        return RuleResult(
            status="not_triggered",
            level="none",
            reason=(
                f"仰臥 AHI / 非仰臥 AHI = {ratio:g}，低於 2.0。"
            ),
            value=ratio,
            threshold="<2.0",
            evidence={
                "supine_ahi": supine_ahi,
                "non_supine_ahi": non_supine_ahi,
            },
        )

    def _evaluate_bmi(self, data: Dict[str, Any]) -> RuleResult:
        bmi = self._number(data.get("bmi"))
        if bmi is None:
            return self._not_evaluable("缺少 BMI，無法評估體重相關風險。")

        if bmi >= 30:
            return RuleResult(
                status="triggered",
                level="high",
                reason=f"BMI {bmi:g} ≥ 30。",
                value=bmi,
                threshold="≥30",
                evidence={"bmi": bmi},
            )

        if bmi >= 25:
            return RuleResult(
                status="triggered",
                level="moderate",
                reason=f"BMI {bmi:g} 介於 25 至 29.9。",
                value=bmi,
                threshold="25–29.9",
                evidence={"bmi": bmi},
            )

        return RuleResult(
            status="not_triggered",
            level="none",
            reason=f"BMI {bmi:g} < 25。",
            value=bmi,
            threshold="<25",
            evidence={"bmi": bmi},
        )

    def _evaluate_sleep_efficiency(
        self,
        data: Dict[str, Any],
    ) -> RuleResult:
        efficiency = self._number(data.get("sleep_efficiency"))
        if efficiency is None:
            return self._not_evaluable(
                "缺少睡眠效率，無法評估睡眠品質指標。"
            )

        if efficiency < 75:
            level = "high"
        elif efficiency < 85:
            level = "moderate"
        else:
            return RuleResult(
                status="not_triggered",
                level="none",
                reason=f"睡眠效率 {efficiency:g}% ≥ 85%。",
                value=efficiency,
                threshold="≥85%",
                evidence={"sleep_efficiency": efficiency},
            )

        return RuleResult(
            status="triggered",
            level=level,
            reason=f"睡眠效率 {efficiency:g}% 低於 85%。",
            value=efficiency,
            threshold="<85%",
            evidence={"sleep_efficiency": efficiency},
        )

    def _evaluate_rem_dominance(
        self,
        data: Dict[str, Any],
    ) -> RuleResult:
        rem_ahi = self._number(data.get("rem_ahi"))
        nrem_ahi = self._number(data.get("nrem_ahi"))

        if rem_ahi is None or nrem_ahi is None:
            return self._not_evaluable(
                "缺少 REM 或 NREM AHI，無法判定 REM 優勢。"
            )

        if nrem_ahi == 0:
            if rem_ahi > 0:
                return RuleResult(
                    status="triggered",
                    level="high",
                    reason="NREM AHI 為 0 且 REM AHI 大於 0。",
                    value=None,
                    threshold="REM/NREM ≥2",
                    evidence={
                        "rem_ahi": rem_ahi,
                        "nrem_ahi": nrem_ahi,
                    },
                )
            return RuleResult(
                status="not_triggered",
                level="none",
                reason="REM 與 NREM AHI 均為 0。",
                evidence={
                    "rem_ahi": rem_ahi,
                    "nrem_ahi": nrem_ahi,
                },
            )

        ratio = round(rem_ahi / nrem_ahi, 3)

        if ratio >= 2:
            return RuleResult(
                status="triggered",
                level="moderate",
                reason=f"REM AHI / NREM AHI = {ratio:g}，達到 2.0。",
                value=ratio,
                threshold="≥2.0",
                evidence={
                    "rem_ahi": rem_ahi,
                    "nrem_ahi": nrem_ahi,
                },
            )

        return RuleResult(
            status="not_triggered",
            level="none",
            reason=f"REM AHI / NREM AHI = {ratio:g}，低於 2.0。",
            value=ratio,
            threshold="<2.0",
            evidence={
                "rem_ahi": rem_ahi,
                "nrem_ahi": nrem_ahi,
            },
        )

    def _evaluate_data_completeness(
        self,
        data: Dict[str, Any],
    ) -> RuleResult:
        essential_fields = (
            "ahi",
            "minimum_spo2",
            "bmi",
            "sleep_efficiency",
        )
        missing = [
            field_name
            for field_name in essential_fields
            if data.get(field_name) is None
        ]

        if not missing:
            return RuleResult(
                status="not_triggered",
                level="none",
                reason="核心規則欄位完整。",
                value=1.0,
                threshold="4/4 essential fields",
                evidence={
                    "essential_fields": list(essential_fields),
                    "missing_fields": [],
                },
            )

        completeness = round(
            (len(essential_fields) - len(missing))
            / len(essential_fields),
            3,
        )

        return RuleResult(
            status="triggered",
            level="high" if len(missing) >= 2 else "moderate",
            reason="缺少核心欄位：" + "、".join(missing) + "。",
            value=completeness,
            threshold="all essential fields present",
            evidence={
                "essential_fields": list(essential_fields),
                "missing_fields": missing,
            },
        )

    def _evaluate_parser_reliability(
        self,
        parsed_result: Dict[str, Any],
    ) -> RuleResult:
        metadata = parsed_result.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        confidence = self._number(metadata.get("overall_confidence"))
        warnings = parsed_result.get("warnings")
        warning_count = len(warnings) if isinstance(warnings, list) else 0

        if confidence is None:
            return self._not_evaluable(
                "缺少 parser overall_confidence。"
            )

        if confidence < 0.75:
            return RuleResult(
                status="triggered",
                level="high",
                reason=f"解析整體信心分數僅 {confidence:g}。",
                value=confidence,
                threshold="<0.75",
                evidence={"warning_count": warning_count},
            )

        if confidence < 0.9 or warning_count > 0:
            return RuleResult(
                status="triggered",
                level="moderate",
                reason=(
                    f"解析信心分數為 {confidence:g}，"
                    f"且共有 {warning_count} 個解析警告。"
                ),
                value=confidence,
                threshold="confidence ≥0.9 and no warnings",
                evidence={"warning_count": warning_count},
            )

        return RuleResult(
            status="not_triggered",
            level="none",
            reason=f"解析信心分數為 {confidence:g}，且無解析警告。",
            value=confidence,
            threshold="≥0.9",
            evidence={"warning_count": warning_count},
        )

    def _build_priority_flags(
        self,
        rules: Dict[str, RuleResult],
    ) -> List[str]:
        flags: List[str] = []

        mapping = {
            "osa_severity": {
                "high": "severe_osa",
                "moderate": "moderate_osa",
                "low": "mild_osa",
            },
            "hypoxemia_risk": {
                "high": "severe_hypoxemia",
                "moderate": "moderate_hypoxemia",
            },
            "positional_component": {
                "high": "strong_position_related_osa",
                "moderate": "position_related_osa",
            },
            "bmi_risk": {
                "high": "obesity_related_risk",
                "moderate": "overweight_related_risk",
            },
            "sleep_efficiency_status": {
                "high": "markedly_low_sleep_efficiency",
                "moderate": "low_sleep_efficiency",
            },
            "rem_dominance": {
                "high": "strong_rem_dominance",
                "moderate": "rem_dominance",
            },
            "data_completeness": {
                "high": "insufficient_core_data",
                "moderate": "incomplete_core_data",
            },
            "parser_reliability": {
                "high": "low_parser_reliability",
                "moderate": "parser_review_needed",
            },
        }

        for rule_name, level_map in mapping.items():
            rule = rules.get(rule_name)
            if rule is None or rule.status != "triggered":
                continue

            flag = level_map.get(rule.level)
            if flag and flag not in flags:
                flags.append(flag)

        return flags

    def _highest_priority(self, flags: List[str]) -> Optional[str]:
        priority_order = (
            "severe_hypoxemia",
            "severe_osa",
            "low_parser_reliability",
            "insufficient_core_data",
            "obesity_related_risk",
            "strong_position_related_osa",
            "markedly_low_sleep_efficiency",
            "moderate_hypoxemia",
            "moderate_osa",
            "parser_review_needed",
            "incomplete_core_data",
            "position_related_osa",
            "overweight_related_risk",
            "low_sleep_efficiency",
            "strong_rem_dominance",
            "rem_dominance",
            "mild_osa",
        )

        for flag in priority_order:
            if flag in flags:
                return flag

        return None

    def _not_evaluable(self, reason: str) -> RuleResult:
        return RuleResult(
            status="not_evaluable",
            level="unknown",
            reason=reason,
        )

    def _number(self, value: Any) -> Optional[float]:
        if isinstance(value, bool) or value is None:
            return None

        if isinstance(value, (int, float)):
            numeric = float(value)
            if math.isnan(numeric) or math.isinf(numeric):
                return None
            return numeric

        try:
            numeric = float(str(value).strip())
        except (TypeError, ValueError):
            return None

        if math.isnan(numeric) or math.isinf(numeric):
            return None

        return numeric

    def _make_json_compatible(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, str)):
            return value

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        if isinstance(value, dict):
            return {
                str(key): self._make_json_compatible(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                self._make_json_compatible(item)
                for item in value
            ]

        if hasattr(value, "item"):
            try:
                return self._make_json_compatible(value.item())
            except Exception:
                pass

        return str(value)


if __name__ == "__main__":
    import json

    sample_parsed_result = {
        "data": {
            "patient_id": "P001",
            "ahi": 32.5,
            "odi": 28.1,
            "minimum_spo2": 78.0,
            "mean_spo2": 93.2,
            "bmi": 29.4,
            "age": 52,
            "sex": "male",
            "total_sleep_time": 356.5,
            "sleep_efficiency": 81.6,
            "supine_ahi": 48.2,
            "non_supine_ahi": 17.1,
            "rem_ahi": 41.3,
            "nrem_ahi": 29.8,
        },
        "derived": {
            "ahi_severity": "severe",
            "hypoxemia_level": "severe",
            "positional_ratio": 2.819,
            "position_relevance": True,
        },
        "metadata": {
            "parser": "ClinicalParser",
            "parser_version": "1.0.0",
            "overall_confidence": 0.93,
        },
        "warnings": [
            "ahi：找到多個不同候選值，已採用優先順序最高者。",
            "supine_ahi：找到多個不同候選值，已採用優先順序最高者。",
            "rem_ahi：找到多個不同候選值，已採用優先順序最高者。",
        ],
    }

    engine = RuleEngine()
    result = engine.evaluate(sample_parsed_result)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
