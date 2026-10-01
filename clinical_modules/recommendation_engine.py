from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Recommendation:
    """
    One structured recommendation produced by RecommendationEngine.
    """
    code: str
    category: str
    title: str
    priority: str
    rationale: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    cautions: List[str] = field(default_factory=list)


class RecommendationEngine:
    """
    Convert ClinicalParser and RuleEngine outputs into structured,
    reviewable sleep-clinical recommendations.

    Important
    ---------
    - This module is decision-support software, not a diagnostic system.
    - Recommendations must be reviewed by a qualified clinician.
    - Missing or unreliable source data lowers recommendation confidence.
    """

    ENGINE_VERSION = "1.0.0"

    PRIORITY_RANK = {
        "urgent": 0,
        "high": 1,
        "moderate": 2,
        "low": 3,
        "informational": 4,
    }

    def recommend(
        self,
        parsed_result: Dict[str, Any],
        rule_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate structured recommendations.

        Parameters
        ----------
        parsed_result:
            Output from ClinicalParser.parse() or parse_document().
        rule_result:
            Output from RuleEngine.evaluate().
        """
        if not isinstance(parsed_result, dict):
            parsed_result = {}

        if not isinstance(rule_result, dict):
            rule_result = {}

        data = parsed_result.get("data")
        if not isinstance(data, dict):
            data = {}

        rules = rule_result.get("rules")
        if not isinstance(rules, dict):
            rules = {}

        flags = rule_result.get("priority_flags")
        if not isinstance(flags, list):
            flags = []

        recommendations: List[Recommendation] = []

        self._add_osa_management(
            recommendations,
            data=data,
            rules=rules,
            flags=flags,
        )
        self._add_hypoxemia_management(
            recommendations,
            data=data,
            rules=rules,
            flags=flags,
        )
        self._add_positional_management(
            recommendations,
            data=data,
            rules=rules,
            flags=flags,
        )
        self._add_weight_management(
            recommendations,
            data=data,
            rules=rules,
            flags=flags,
        )
        self._add_sleep_efficiency_management(
            recommendations,
            data=data,
            rules=rules,
            flags=flags,
        )
        self._add_data_quality_actions(
            recommendations,
            parsed_result=parsed_result,
            rule_result=rule_result,
            flags=flags,
        )
        self._add_follow_up(
            recommendations,
            data=data,
            rules=rules,
            flags=flags,
        )

        recommendations = self._deduplicate(recommendations)
        recommendations.sort(
            key=lambda item: (
                self.PRIORITY_RANK.get(item.priority, 99),
                item.category,
                item.code,
            )
        )

        primary = recommendations[0] if recommendations else None
        alternatives = [
            item
            for item in recommendations[1:]
            if item.category in {
                "therapy",
                "adjunctive_therapy",
                "lifestyle",
            }
        ]
        follow_up = [
            item
            for item in recommendations
            if item.category in {
                "follow_up",
                "assessment",
                "data_quality",
            }
        ]

        confidence = self._recommendation_confidence(
            parsed_result=parsed_result,
            rule_result=rule_result,
        )

        result = {
            "patient_id": data.get("patient_id"),
            "primary_recommendation": (
                asdict(primary)
                if primary is not None
                else None
            ),
            "recommendations": [
                asdict(item)
                for item in recommendations
            ],
            "alternative_recommendations": [
                asdict(item)
                for item in alternatives
            ],
            "follow_up_actions": [
                asdict(item)
                for item in follow_up
            ],
            "summary": {
                "recommendation_count": len(recommendations),
                "highest_priority": (
                    primary.priority
                    if primary is not None
                    else None
                ),
                "requires_clinician_review": True,
                "recommendation_confidence": confidence,
            },
            "metadata": {
                "engine": self.__class__.__name__,
                "engine_version": self.ENGINE_VERSION,
                "source_rule_engine": (
                    rule_result.get("metadata", {}).get("engine")
                    if isinstance(rule_result.get("metadata"), dict)
                    else None
                ),
                "source_rule_engine_version": (
                    rule_result.get("metadata", {}).get("engine_version")
                    if isinstance(rule_result.get("metadata"), dict)
                    else None
                ),
                "source_parser": (
                    parsed_result.get("metadata", {}).get("parser")
                    if isinstance(parsed_result.get("metadata"), dict)
                    else None
                ),
                "source_parser_version": (
                    parsed_result.get("metadata", {}).get("parser_version")
                    if isinstance(parsed_result.get("metadata"), dict)
                    else None
                ),
            },
            "warnings": self._build_warnings(
                parsed_result=parsed_result,
                rule_result=rule_result,
            ),
            "disclaimer": (
                "本結果僅供臨床決策支援與流程測試，"
                "不得取代醫師診斷、處方或正式治療計畫。"
            ),
        }

        return self._make_json_compatible(result)

    def _add_osa_management(
        self,
        recommendations: List[Recommendation],
        *,
        data: Dict[str, Any],
        rules: Dict[str, Any],
        flags: List[str],
    ) -> None:
        ahi = self._number(data.get("ahi"))
        osa_rule = self._rule(rules, "osa_severity")

        if "severe_osa" in flags:
            recommendations.append(
                Recommendation(
                    code="sleep_specialist_review",
                    category="assessment",
                    title="優先安排睡眠專科或相關臨床評估",
                    priority="high",
                    rationale=(
                        f"AHI {self._fmt(ahi)} 顯示重度阻塞型睡眠呼吸中止"
                        "風險，需要由臨床人員確認治療策略。"
                    ),
                    evidence={
                        "ahi": ahi,
                        "rule": osa_rule,
                    },
                    prerequisites=[
                        "確認睡眠檢查品質與有效睡眠時間",
                        "評估症狀、共病與白天嗜睡情形",
                    ],
                )
            )
            recommendations.append(
                Recommendation(
                    code="pap_therapy_evaluation",
                    category="therapy",
                    title="評估正壓呼吸器治療適用性",
                    priority="high",
                    rationale=(
                        "重度 OSA 通常需要優先評估能穩定上呼吸道的"
                        "治療方式；實際模式與壓力設定應由臨床流程決定。"
                    ),
                    evidence={
                        "ahi": ahi,
                        "osa_level": osa_rule.get("level"),
                    },
                    prerequisites=[
                        "完成臨床診斷確認",
                        "評估鼻腔阻塞、面罩適配與依從性風險",
                    ],
                    cautions=[
                        "不得由本系統自動決定處方壓力",
                        "需依醫師判斷與正式檢查結果執行",
                    ],
                )
            )
            return

        if "moderate_osa" in flags:
            recommendations.append(
                Recommendation(
                    code="osa_treatment_discussion",
                    category="therapy",
                    title="討論中度 OSA 治療選項",
                    priority="moderate",
                    rationale=(
                        f"AHI {self._fmt(ahi)} 落在中度範圍，"
                        "可依症狀、共病、解剖條件與偏好比較不同治療方案。"
                    ),
                    evidence={"ahi": ahi, "rule": osa_rule},
                    prerequisites=[
                        "臨床確認 OSA 診斷",
                        "評估治療偏好與依從性",
                    ],
                )
            )
            return

        if "mild_osa" in flags:
            recommendations.append(
                Recommendation(
                    code="mild_osa_conservative_plan",
                    category="therapy",
                    title="規劃輕度 OSA 的保守治療與追蹤",
                    priority="low",
                    rationale=(
                        f"AHI {self._fmt(ahi)} 落在輕度範圍，"
                        "治療強度宜結合症狀、共病與生活型態因素。"
                    ),
                    evidence={"ahi": ahi, "rule": osa_rule},
                )
            )

    def _add_hypoxemia_management(
        self,
        recommendations: List[Recommendation],
        *,
        data: Dict[str, Any],
        rules: Dict[str, Any],
        flags: List[str],
    ) -> None:
        minimum_spo2 = self._number(data.get("minimum_spo2"))
        mean_spo2 = self._number(data.get("mean_spo2"))
        rule = self._rule(rules, "hypoxemia_risk")

        if "severe_hypoxemia" in flags:
            recommendations.append(
                Recommendation(
                    code="urgent_hypoxemia_review",
                    category="assessment",
                    title="優先確認顯著夜間低血氧",
                    priority="urgent",
                    rationale=(
                        f"最低 SpO2 為 {self._fmt(minimum_spo2)}%，"
                        "應確認訊號品質、低血氧持續時間與可能共病。"
                    ),
                    evidence={
                        "minimum_spo2": minimum_spo2,
                        "mean_spo2": mean_spo2,
                        "rule": rule,
                    },
                    prerequisites=[
                        "檢查血氧探頭訊號品質與偽影",
                        "確認低血氧事件的持續時間與分布",
                    ],
                    cautions=[
                        "單一最低值不能獨立決定氧氣治療",
                        "需排除肺部、心臟或低通氣相關因素",
                    ],
                )
            )
            return

        if "moderate_hypoxemia" in flags:
            recommendations.append(
                Recommendation(
                    code="hypoxemia_review",
                    category="assessment",
                    title="評估夜間低血氧程度與原因",
                    priority="high",
                    rationale=(
                        f"最低 SpO2 為 {self._fmt(minimum_spo2)}%，"
                        "建議配合事件時間軸與臨床背景進一步判讀。"
                    ),
                    evidence={
                        "minimum_spo2": minimum_spo2,
                        "mean_spo2": mean_spo2,
                        "rule": rule,
                    },
                )
            )

    def _add_positional_management(
        self,
        recommendations: List[Recommendation],
        *,
        data: Dict[str, Any],
        rules: Dict[str, Any],
        flags: List[str],
    ) -> None:
        if (
            "position_related_osa" not in flags
            and "strong_position_related_osa" not in flags
        ):
            return

        supine_ahi = self._number(data.get("supine_ahi"))
        non_supine_ahi = self._number(data.get("non_supine_ahi"))
        rule = self._rule(rules, "positional_component")

        priority = (
            "moderate"
            if "strong_position_related_osa" in flags
            else "low"
        )

        recommendations.append(
            Recommendation(
                code="positional_therapy_evaluation",
                category="adjunctive_therapy",
                title="評估體位治療作為輔助方案",
                priority=priority,
                rationale=(
                    "仰臥與非仰臥 AHI 存在明顯差異，"
                    "可評估降低仰臥睡眠比例是否具有附加效益。"
                ),
                evidence={
                    "supine_ahi": supine_ahi,
                    "non_supine_ahi": non_supine_ahi,
                    "ratio": rule.get("value"),
                },
                prerequisites=[
                    "確認各睡姿記錄時間足夠",
                    "評估病人能否長期遵從體位介入",
                ],
                cautions=[
                    "不應在重度 OSA 或顯著低血氧時單獨取代主要治療",
                ],
            )
        )

    def _add_weight_management(
        self,
        recommendations: List[Recommendation],
        *,
        data: Dict[str, Any],
        rules: Dict[str, Any],
        flags: List[str],
    ) -> None:
        if (
            "overweight_related_risk" not in flags
            and "obesity_related_risk" not in flags
        ):
            return

        bmi = self._number(data.get("bmi"))
        rule = self._rule(rules, "bmi_risk")
        priority = (
            "moderate"
            if "obesity_related_risk" in flags
            else "low"
        )

        recommendations.append(
            Recommendation(
                code="weight_management_support",
                category="lifestyle",
                title="納入體重與代謝風險管理",
                priority=priority,
                rationale=(
                    f"BMI 為 {self._fmt(bmi)}，體重管理可作為"
                    "整體睡眠呼吸風險控制的一部分。"
                ),
                evidence={"bmi": bmi, "rule": rule},
                prerequisites=[
                    "依個人狀況設定可執行目標",
                    "必要時轉介營養或代謝相關專業人員",
                ],
                cautions=[
                    "體重管理不應延後必要的 OSA 主要治療",
                ],
            )
        )

    def _add_sleep_efficiency_management(
        self,
        recommendations: List[Recommendation],
        *,
        data: Dict[str, Any],
        rules: Dict[str, Any],
        flags: List[str],
    ) -> None:
        if (
            "low_sleep_efficiency" not in flags
            and "markedly_low_sleep_efficiency" not in flags
        ):
            return

        efficiency = self._number(data.get("sleep_efficiency"))
        rule = self._rule(rules, "sleep_efficiency_status")

        recommendations.append(
            Recommendation(
                code="sleep_efficiency_review",
                category="lifestyle",
                title="評估睡眠效率偏低的原因",
                priority=(
                    "moderate"
                    if "markedly_low_sleep_efficiency" in flags
                    else "low"
                ),
                rationale=(
                    f"睡眠效率為 {self._fmt(efficiency)}%，"
                    "建議確認失眠症狀、檢查環境與睡眠習慣。"
                ),
                evidence={
                    "sleep_efficiency": efficiency,
                    "rule": rule,
                },
                prerequisites=[
                    "確認總記錄時間與總睡眠時間",
                    "詢問入睡困難、夜間清醒與早醒",
                ],
            )
        )

    def _add_data_quality_actions(
        self,
        recommendations: List[Recommendation],
        *,
        parsed_result: Dict[str, Any],
        rule_result: Dict[str, Any],
        flags: List[str],
    ) -> None:
        if (
            "parser_review_needed" in flags
            or "low_parser_reliability" in flags
        ):
            parsed_warnings = parsed_result.get("warnings")
            warning_count = (
                len(parsed_warnings)
                if isinstance(parsed_warnings, list)
                else 0
            )
            confidence = self._number(
                parsed_result.get("metadata", {}).get(
                    "overall_confidence"
                )
                if isinstance(parsed_result.get("metadata"), dict)
                else None
            )

            recommendations.append(
                Recommendation(
                    code="verify_extracted_fields",
                    category="data_quality",
                    title="人工核對自動擷取欄位",
                    priority=(
                        "high"
                        if "low_parser_reliability" in flags
                        else "moderate"
                    ),
                    rationale=(
                        f"Parser 信心分數為 {self._fmt(confidence)}，"
                        f"並回傳 {warning_count} 個警告。"
                    ),
                    evidence={
                        "overall_confidence": confidence,
                        "warning_count": warning_count,
                    },
                    prerequisites=[
                        "核對 AHI、最低 SpO2、BMI 與睡眠效率",
                        "比對原始報告與自動擷取結果",
                    ],
                )
            )

        if (
            "incomplete_core_data" in flags
            or "insufficient_core_data" in flags
        ):
            rule = self._rule(
                rule_result.get("rules", {})
                if isinstance(rule_result.get("rules"), dict)
                else {},
                "data_completeness",
            )
            missing = (
                rule.get("evidence", {}).get("missing_fields", [])
                if isinstance(rule.get("evidence"), dict)
                else []
            )

            recommendations.append(
                Recommendation(
                    code="complete_core_data",
                    category="data_quality",
                    title="補齊核心臨床欄位",
                    priority=(
                        "high"
                        if "insufficient_core_data" in flags
                        else "moderate"
                    ),
                    rationale=(
                        "核心欄位不完整，可能降低規則與建議可靠性。"
                    ),
                    evidence={"missing_fields": missing},
                )
            )

    def _add_follow_up(
        self,
        recommendations: List[Recommendation],
        *,
        data: Dict[str, Any],
        rules: Dict[str, Any],
        flags: List[str],
    ) -> None:
        clinically_relevant = any(
            flag in flags
            for flag in {
                "severe_osa",
                "moderate_osa",
                "mild_osa",
                "severe_hypoxemia",
                "moderate_hypoxemia",
                "position_related_osa",
                "strong_position_related_osa",
            }
        )

        if not clinically_relevant:
            return

        recommendations.append(
            Recommendation(
                code="treatment_response_follow_up",
                category="follow_up",
                title="建立治療後追蹤與反應評估",
                priority="moderate",
                rationale=(
                    "治療計畫確立後，應追蹤症狀、依從性、"
                    "殘餘事件與夜間血氧變化。"
                ),
                evidence={
                    "ahi": self._number(data.get("ahi")),
                    "minimum_spo2": self._number(
                        data.get("minimum_spo2")
                    ),
                },
                prerequisites=[
                    "先完成正式治療計畫",
                    "設定可比較的追蹤指標",
                ],
            )
        )

    def _recommendation_confidence(
        self,
        *,
        parsed_result: Dict[str, Any],
        rule_result: Dict[str, Any],
    ) -> float:
        parser_confidence = self._number(
            parsed_result.get("metadata", {}).get("overall_confidence")
            if isinstance(parsed_result.get("metadata"), dict)
            else None
        )

        rules = rule_result.get("rules")
        if not isinstance(rules, dict):
            rules = {}

        evaluable_rules = [
            rule
            for rule in rules.values()
            if isinstance(rule, dict)
            and rule.get("status") != "not_evaluable"
        ]

        total_rules = len(rules)
        evaluable_ratio = (
            len(evaluable_rules) / total_rules
            if total_rules
            else 0.0
        )

        if parser_confidence is None:
            parser_confidence = 0.5

        warning_count = 0
        parser_warnings = parsed_result.get("warnings")
        if isinstance(parser_warnings, list):
            warning_count += len(parser_warnings)

        rule_warnings = rule_result.get("warnings")
        if isinstance(rule_warnings, list):
            warning_count += len(rule_warnings)

        confidence = (
            parser_confidence * 0.7
            + evaluable_ratio * 0.3
            - min(0.2, warning_count * 0.02)
        )

        return round(max(0.0, min(1.0, confidence)), 3)

    def _build_warnings(
        self,
        *,
        parsed_result: Dict[str, Any],
        rule_result: Dict[str, Any],
    ) -> List[str]:
        warnings: List[str] = []

        parser_warnings = parsed_result.get("warnings")
        if isinstance(parser_warnings, list) and parser_warnings:
            warnings.append(
                f"ClinicalParser 有 {len(parser_warnings)} 個警告。"
            )

        rule_warnings = rule_result.get("warnings")
        if isinstance(rule_warnings, list) and rule_warnings:
            warnings.append(
                f"RuleEngine 有 {len(rule_warnings)} 個警告。"
            )

        if not isinstance(parsed_result.get("data"), dict):
            warnings.append("ClinicalParser data 結構缺失。")

        if not isinstance(rule_result.get("rules"), dict):
            warnings.append("RuleEngine rules 結構缺失。")

        return warnings

    def _rule(
        self,
        rules: Dict[str, Any],
        name: str,
    ) -> Dict[str, Any]:
        value = rules.get(name)
        return value if isinstance(value, dict) else {}

    def _deduplicate(
        self,
        recommendations: List[Recommendation],
    ) -> List[Recommendation]:
        seen = set()
        output: List[Recommendation] = []

        for item in recommendations:
            if item.code in seen:
                continue
            seen.add(item.code)
            output.append(item)

        return output

    def _number(self, value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            numeric = float(value)
        else:
            try:
                numeric = float(str(value).strip())
            except (TypeError, ValueError):
                return None

        if math.isnan(numeric) or math.isinf(numeric):
            return None

        return numeric

    def _fmt(self, value: Optional[float]) -> str:
        if value is None:
            return "未知"
        return f"{value:g}"

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

    sample_rule_result = {
        "rules": {
            "osa_severity": {
                "status": "triggered",
                "level": "high",
                "value": 32.5,
            },
            "hypoxemia_risk": {
                "status": "triggered",
                "level": "high",
                "value": 78.0,
            },
            "positional_component": {
                "status": "triggered",
                "level": "moderate",
                "value": 2.819,
            },
            "bmi_risk": {
                "status": "triggered",
                "level": "moderate",
                "value": 29.4,
            },
            "sleep_efficiency_status": {
                "status": "triggered",
                "level": "moderate",
                "value": 81.6,
            },
            "rem_dominance": {
                "status": "not_triggered",
                "level": "none",
                "value": 1.386,
            },
            "data_completeness": {
                "status": "not_triggered",
                "level": "none",
                "value": 1.0,
            },
            "parser_reliability": {
                "status": "triggered",
                "level": "moderate",
                "value": 0.93,
            },
        },
        "priority_flags": [
            "severe_osa",
            "severe_hypoxemia",
            "position_related_osa",
            "overweight_related_risk",
            "low_sleep_efficiency",
            "parser_review_needed",
        ],
        "metadata": {
            "engine": "RuleEngine",
            "engine_version": "1.0.0",
        },
        "warnings": [
            "ClinicalParser 回傳 3 個警告；規則結果應配合原始欄位與信心分數審閱。"
        ],
    }

    engine = RecommendationEngine()
    result = engine.recommend(
        sample_parsed_result,
        sample_rule_result,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
