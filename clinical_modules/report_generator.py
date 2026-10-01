from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ReportGenerator:
    """
    Merge ClinicalParser, RuleEngine, and RecommendationEngine outputs
    into one structured report and one readable Traditional Chinese report.

    This module does not diagnose or prescribe treatment.
    """

    GENERATOR_VERSION = "1.0.0"

    PRIORITY_LABELS = {
        "urgent": "緊急優先",
        "high": "高優先",
        "moderate": "中等優先",
        "low": "低優先",
        "informational": "資訊性",
    }

    LEVEL_LABELS = {
        "high": "高",
        "moderate": "中",
        "low": "低",
        "none": "無",
        "unknown": "未知",
    }

    FIELD_LABELS = {
        "patient_id": "患者編號",
        "age": "年齡",
        "sex": "性別",
        "ahi": "AHI",
        "odi": "ODI",
        "minimum_spo2": "最低 SpO2",
        "mean_spo2": "平均 SpO2",
        "bmi": "BMI",
        "total_sleep_time": "總睡眠時間",
        "sleep_efficiency": "睡眠效率",
        "supine_ahi": "仰臥 AHI",
        "non_supine_ahi": "非仰臥 AHI",
        "rem_ahi": "REM AHI",
        "nrem_ahi": "NREM AHI",
    }

    FIELD_UNITS = {
        "ahi": "次/小時",
        "odi": "次/小時",
        "minimum_spo2": "%",
        "mean_spo2": "%",
        "bmi": "kg/m²",
        "age": "歲",
        "total_sleep_time": "分鐘",
        "sleep_efficiency": "%",
        "supine_ahi": "次/小時",
        "non_supine_ahi": "次/小時",
        "rem_ahi": "次/小時",
        "nrem_ahi": "次/小時",
    }

    def generate(
        self,
        parsed_result: Dict[str, Any],
        rule_result: Dict[str, Any],
        recommendation_result: Dict[str, Any],
        *,
        source_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        parsed_result = parsed_result if isinstance(parsed_result, dict) else {}
        rule_result = rule_result if isinstance(rule_result, dict) else {}
        recommendation_result = (
            recommendation_result
            if isinstance(recommendation_result, dict)
            else {}
        )

        data = parsed_result.get("data")
        if not isinstance(data, dict):
            data = {}

        patient_summary = self._build_patient_summary(data)
        clinical_findings = self._build_clinical_findings(
            data=data,
            rule_result=rule_result,
        )
        risk_summary = self._build_risk_summary(rule_result)
        data_quality = self._build_data_quality(
            parsed_result=parsed_result,
            rule_result=rule_result,
            recommendation_result=recommendation_result,
        )

        primary_recommendation = recommendation_result.get(
            "primary_recommendation"
        )
        if not isinstance(primary_recommendation, dict):
            primary_recommendation = None

        all_recommendations = recommendation_result.get("recommendations")
        if not isinstance(all_recommendations, list):
            all_recommendations = []

        warnings = self._collect_warnings(
            parsed_result,
            rule_result,
            recommendation_result,
        )

        report = {
            "report_id": self._build_report_id(data.get("patient_id")),
            "source_filename": (
                source_filename
                or parsed_result.get("source_filename")
            ),
            "patient_summary": patient_summary,
            "clinical_findings": clinical_findings,
            "risk_summary": risk_summary,
            "primary_recommendation": primary_recommendation,
            "all_recommendations": all_recommendations,
            "follow_up_actions": self._safe_list(
                recommendation_result.get("follow_up_actions")
            ),
            "data_quality": data_quality,
            "warnings": warnings,
            "plain_text_report": "",
            "metadata": {
                "generator": self.__class__.__name__,
                "generator_version": self.GENERATOR_VERSION,
                "generated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "source_parser": self._metadata_value(
                    parsed_result,
                    "parser",
                ),
                "source_parser_version": self._metadata_value(
                    parsed_result,
                    "parser_version",
                ),
                "source_rule_engine": self._metadata_value(
                    rule_result,
                    "engine",
                ),
                "source_rule_engine_version": self._metadata_value(
                    rule_result,
                    "engine_version",
                ),
                "source_recommendation_engine": self._metadata_value(
                    recommendation_result,
                    "engine",
                ),
                "source_recommendation_engine_version": self._metadata_value(
                    recommendation_result,
                    "engine_version",
                ),
            },
            "disclaimer": (
                recommendation_result.get("disclaimer")
                or "本報告僅供臨床決策支援與系統流程測試，"
                "不得取代醫師診斷、處方或正式治療計畫。"
            ),
        }

        report["plain_text_report"] = self._render_plain_text(report)
        return self._make_json_compatible(report)

    def save_json(
        self,
        report: Dict[str, Any],
        output_path: str,
    ) -> str:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                report,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        return output_path

    def save_text(
        self,
        report: Dict[str, Any],
        output_path: str,
    ) -> str:
        text = report.get("plain_text_report", "")
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(str(text))
        return output_path

    def _build_patient_summary(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        ordered_fields = (
            "patient_id",
            "age",
            "sex",
            "ahi",
            "odi",
            "minimum_spo2",
            "mean_spo2",
            "bmi",
            "total_sleep_time",
            "sleep_efficiency",
            "supine_ahi",
            "non_supine_ahi",
            "rem_ahi",
            "nrem_ahi",
        )

        summary: Dict[str, Any] = {}
        for field_name in ordered_fields:
            value = data.get(field_name)
            if value is not None:
                summary[field_name] = value

        return summary

    def _build_clinical_findings(
        self,
        *,
        data: Dict[str, Any],
        rule_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rules = rule_result.get("rules")
        if not isinstance(rules, dict):
            return []

        findings: List[Dict[str, Any]] = []
        for rule_name, rule in rules.items():
            if not isinstance(rule, dict):
                continue

            findings.append(
                {
                    "rule": rule_name,
                    "status": rule.get("status"),
                    "level": rule.get("level"),
                    "reason": rule.get("reason"),
                    "value": rule.get("value"),
                    "threshold": rule.get("threshold"),
                    "evidence": (
                        rule.get("evidence")
                        if isinstance(rule.get("evidence"), dict)
                        else {}
                    ),
                }
            )

        return findings

    def _build_risk_summary(
        self,
        rule_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary = rule_result.get("summary")
        if not isinstance(summary, dict):
            summary = {}

        flags = rule_result.get("priority_flags")
        if not isinstance(flags, list):
            flags = []

        return {
            "highest_priority": summary.get("highest_priority"),
            "priority_flags": flags,
            "triggered_rule_count": summary.get(
                "triggered_rule_count",
                0,
            ),
            "not_evaluable_rule_count": summary.get(
                "not_evaluable_rule_count",
                0,
            ),
            "requires_clinical_review": summary.get(
                "requires_clinical_review",
                bool(flags),
            ),
        }

    def _build_data_quality(
        self,
        *,
        parsed_result: Dict[str, Any],
        rule_result: Dict[str, Any],
        recommendation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        parser_metadata = parsed_result.get("metadata")
        if not isinstance(parser_metadata, dict):
            parser_metadata = {}

        parser_fields = parsed_result.get("fields")
        if not isinstance(parser_fields, dict):
            parser_fields = {}

        low_confidence_fields: List[Dict[str, Any]] = []
        for field_name, field_info in parser_fields.items():
            if not isinstance(field_info, dict):
                continue

            confidence = self._number(field_info.get("confidence"))
            warnings = self._safe_list(field_info.get("warnings"))

            if (
                confidence is not None
                and confidence < 0.9
            ) or warnings:
                low_confidence_fields.append(
                    {
                        "field": field_name,
                        "confidence": confidence,
                        "warnings": warnings,
                    }
                )

        recommendation_summary = recommendation_result.get("summary")
        if not isinstance(recommendation_summary, dict):
            recommendation_summary = {}

        return {
            "parser_overall_confidence": self._number(
                parser_metadata.get("overall_confidence")
            ),
            "parsed_field_count": parser_metadata.get(
                "parsed_field_count"
            ),
            "missing_field_count": parser_metadata.get(
                "missing_field_count"
            ),
            "low_confidence_fields": low_confidence_fields,
            "parser_warning_count": len(
                self._safe_list(parsed_result.get("warnings"))
            ),
            "rule_warning_count": len(
                self._safe_list(rule_result.get("warnings"))
            ),
            "recommendation_warning_count": len(
                self._safe_list(
                    recommendation_result.get("warnings")
                )
            ),
            "recommendation_confidence": self._number(
                recommendation_summary.get(
                    "recommendation_confidence"
                )
            ),
        }

    def _collect_warnings(
        self,
        *results: Dict[str, Any],
    ) -> List[str]:
        output: List[str] = []
        for result in results:
            warnings = result.get("warnings")
            if not isinstance(warnings, list):
                continue

            for warning in warnings:
                text = str(warning)
                if text not in output:
                    output.append(text)

        return output

    def _render_plain_text(
        self,
        report: Dict[str, Any],
    ) -> str:
        lines: List[str] = []
        separator = "=" * 72

        lines.append(separator)
        lines.append("睡眠數位孿生臨床決策支援報告")
        lines.append(separator)

        report_id = report.get("report_id")
        if report_id:
            lines.append(f"報告編號：{report_id}")

        source_filename = report.get("source_filename")
        if source_filename:
            lines.append(f"來源檔案：{source_filename}")

        lines.append("")

        lines.append("一、患者與睡眠檢查摘要")
        patient_summary = report.get("patient_summary")
        if isinstance(patient_summary, dict) and patient_summary:
            for field_name, value in patient_summary.items():
                label = self.FIELD_LABELS.get(
                    field_name,
                    field_name,
                )
                unit = self.FIELD_UNITS.get(field_name, "")
                display_value = self._format_field_value(
                    field_name,
                    value,
                )
                suffix = f" {unit}" if unit else ""
                lines.append(f"- {label}：{display_value}{suffix}")
        else:
            lines.append("- 無可用患者摘要資料")

        lines.append("")
        lines.append("二、臨床規則判讀")

        findings = report.get("clinical_findings")
        if isinstance(findings, list) and findings:
            for finding in findings:
                if not isinstance(finding, dict):
                    continue

                level = self.LEVEL_LABELS.get(
                    str(finding.get("level")),
                    str(finding.get("level") or "未知"),
                )
                status = str(
                    finding.get("status") or "unknown"
                )
                reason = str(
                    finding.get("reason") or "無說明"
                )
                lines.append(
                    f"- {finding.get('rule')} "
                    f"[{status}／{level}]：{reason}"
                )
        else:
            lines.append("- 無可用規則結果")

        lines.append("")
        lines.append("三、風險摘要")

        risk_summary = report.get("risk_summary")
        if isinstance(risk_summary, dict):
            lines.append(
                "- 最高優先旗標："
                + str(
                    risk_summary.get("highest_priority")
                    or "無"
                )
            )
            flags = risk_summary.get("priority_flags")
            if isinstance(flags, list) and flags:
                lines.append(
                    "- 風險旗標："
                    + "、".join(str(flag) for flag in flags)
                )
            else:
                lines.append("- 風險旗標：無")
            lines.append(
                "- 已觸發規則數："
                + str(
                    risk_summary.get(
                        "triggered_rule_count",
                        0,
                    )
                )
            )
            lines.append(
                "- 無法評估規則數："
                + str(
                    risk_summary.get(
                        "not_evaluable_rule_count",
                        0,
                    )
                )
            )

        lines.append("")
        lines.append("四、主要建議")

        primary = report.get("primary_recommendation")
        if isinstance(primary, dict):
            priority = self.PRIORITY_LABELS.get(
                str(primary.get("priority")),
                str(primary.get("priority") or "未分級"),
            )
            lines.append(
                f"- {primary.get('title', '未命名建議')}（{priority}）"
            )
            lines.append(
                f"  理由：{primary.get('rationale', '無說明')}"
            )

            prerequisites = primary.get("prerequisites")
            if isinstance(prerequisites, list) and prerequisites:
                lines.append(
                    "  前置確認："
                    + "；".join(str(item) for item in prerequisites)
                )

            cautions = primary.get("cautions")
            if isinstance(cautions, list) and cautions:
                lines.append(
                    "  注意事項："
                    + "；".join(str(item) for item in cautions)
                )
        else:
            lines.append("- 尚無主要建議")

        lines.append("")
        lines.append("五、完整建議清單")

        recommendations = report.get("all_recommendations")
        if isinstance(recommendations, list) and recommendations:
            for index, item in enumerate(recommendations, start=1):
                if not isinstance(item, dict):
                    continue

                priority = self.PRIORITY_LABELS.get(
                    str(item.get("priority")),
                    str(item.get("priority") or "未分級"),
                )
                lines.append(
                    f"{index}. {item.get('title', '未命名建議')} "
                    f"[{priority}]"
                )
                lines.append(
                    f"   {item.get('rationale', '無說明')}"
                )
        else:
            lines.append("- 無可用建議")

        lines.append("")
        lines.append("六、資料品質")

        quality = report.get("data_quality")
        if isinstance(quality, dict):
            lines.append(
                "- Parser 整體信心："
                + self._fmt(
                    quality.get("parser_overall_confidence")
                )
            )
            lines.append(
                "- Recommendation 信心："
                + self._fmt(
                    quality.get("recommendation_confidence")
                )
            )
            lines.append(
                "- Parser 警告數："
                + str(quality.get("parser_warning_count", 0))
            )
            lines.append(
                "- RuleEngine 警告數："
                + str(quality.get("rule_warning_count", 0))
            )

            low_confidence_fields = quality.get(
                "low_confidence_fields"
            )
            if (
                isinstance(low_confidence_fields, list)
                and low_confidence_fields
            ):
                field_names = [
                    str(item.get("field"))
                    for item in low_confidence_fields
                    if isinstance(item, dict)
                ]
                lines.append(
                    "- 建議人工核對欄位："
                    + "、".join(field_names)
                )

        warnings = report.get("warnings")
        if isinstance(warnings, list) and warnings:
            lines.append("")
            lines.append("七、系統警告")
            for warning in warnings:
                lines.append(f"- {warning}")

        lines.append("")
        lines.append("聲明")
        lines.append(str(report.get("disclaimer") or ""))

        return "\n".join(lines)

    def _build_report_id(
        self,
        patient_id: Any,
    ) -> str:
        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
        identifier = (
            str(patient_id).strip()
            if patient_id is not None
            else "UNKNOWN"
        )
        identifier = identifier or "UNKNOWN"
        return f"SDT-{identifier}-{timestamp}"

    def _metadata_value(
        self,
        result: Dict[str, Any],
        key: str,
    ) -> Any:
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            return None
        return metadata.get(key)

    def _format_field_value(
        self,
        field_name: str,
        value: Any,
    ) -> str:
        if field_name == "sex":
            mapping = {
                "male": "男",
                "female": "女",
                "other": "其他",
                "unknown": "未知",
            }
            return mapping.get(str(value).lower(), str(value))

        if isinstance(value, float):
            return f"{value:g}"

        return str(value)

    def _safe_list(
        self,
        value: Any,
    ) -> List[Any]:
        return value if isinstance(value, list) else []

    def _number(
        self,
        value: Any,
    ) -> Optional[float]:
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

    def _fmt(
        self,
        value: Any,
    ) -> str:
        numeric = self._number(value)
        if numeric is None:
            return "未知"
        return f"{numeric:g}"

    def _make_json_compatible(
        self,
        value: Any,
    ) -> Any:
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
    sample_parsed_result = {
        "source_filename": "sample_report.txt",
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
        "fields": {
            "ahi": {
                "confidence": 0.76,
                "warnings": [
                    "找到多個不同候選值，已採用優先順序最高者。"
                ],
            },
            "supine_ahi": {
                "confidence": 0.85,
                "warnings": [
                    "找到多個不同候選值，已採用優先順序最高者。"
                ],
            },
            "rem_ahi": {
                "confidence": 0.85,
                "warnings": [
                    "找到多個不同候選值，已採用優先順序最高者。"
                ],
            },
        },
        "metadata": {
            "parser": "ClinicalParser",
            "parser_version": "1.0.0",
            "parsed_field_count": 14,
            "missing_field_count": 0,
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
                "reason": "AHI 32.5，落在 severe 範圍（≥30）。",
                "value": 32.5,
                "threshold": "≥30",
                "evidence": {"ahi": 32.5},
            },
            "hypoxemia_risk": {
                "status": "triggered",
                "level": "high",
                "reason": "最低 SpO2 為 78%，低於 80%。",
                "value": 78.0,
                "threshold": "<80%",
                "evidence": {"minimum_spo2": 78.0},
            },
            "positional_component": {
                "status": "triggered",
                "level": "moderate",
                "reason": (
                    "仰臥 AHI / 非仰臥 AHI = 2.819，"
                    "達到體位相關門檻 2.0。"
                ),
                "value": 2.819,
                "threshold": "≥2.0",
                "evidence": {
                    "supine_ahi": 48.2,
                    "non_supine_ahi": 17.1,
                },
            },
        },
        "priority_flags": [
            "severe_osa",
            "severe_hypoxemia",
            "position_related_osa",
        ],
        "summary": {
            "highest_priority": "severe_hypoxemia",
            "triggered_rule_count": 3,
            "not_evaluable_rule_count": 0,
            "requires_clinical_review": True,
        },
        "metadata": {
            "engine": "RuleEngine",
            "engine_version": "1.0.0",
        },
        "warnings": [
            "ClinicalParser 回傳 3 個警告；"
            "規則結果應配合原始欄位與信心分數審閱。"
        ],
    }

    sample_recommendation_result = {
        "primary_recommendation": {
            "code": "urgent_hypoxemia_review",
            "category": "assessment",
            "title": "優先確認顯著夜間低血氧",
            "priority": "urgent",
            "rationale": (
                "最低 SpO2 為 78%，應確認訊號品質、"
                "低血氧持續時間與可能共病。"
            ),
            "evidence": {
                "minimum_spo2": 78.0,
                "mean_spo2": 93.2,
            },
            "prerequisites": [
                "檢查血氧探頭訊號品質與偽影",
                "確認低血氧事件的持續時間與分布",
            ],
            "cautions": [
                "單一最低值不能獨立決定氧氣治療",
            ],
        },
        "recommendations": [
            {
                "code": "urgent_hypoxemia_review",
                "category": "assessment",
                "title": "優先確認顯著夜間低血氧",
                "priority": "urgent",
                "rationale": (
                    "最低 SpO2 為 78%，應確認訊號品質、"
                    "低血氧持續時間與可能共病。"
                ),
            },
            {
                "code": "pap_therapy_evaluation",
                "category": "therapy",
                "title": "評估正壓呼吸器治療適用性",
                "priority": "high",
                "rationale": (
                    "重度 OSA 應優先評估主要治療方式。"
                ),
            },
        ],
        "follow_up_actions": [],
        "summary": {
            "recommendation_count": 2,
            "highest_priority": "urgent",
            "requires_clinician_review": True,
            "recommendation_confidence": 0.871,
        },
        "metadata": {
            "engine": "RecommendationEngine",
            "engine_version": "1.0.0",
        },
        "warnings": [
            "ClinicalParser 有 3 個警告。",
            "RuleEngine 有 1 個警告。",
        ],
        "disclaimer": (
            "本結果僅供臨床決策支援與流程測試，"
            "不得取代醫師診斷、處方或正式治療計畫。"
        ),
    }

    generator = ReportGenerator()
    report = generator.generate(
        sample_parsed_result,
        sample_rule_result,
        sample_recommendation_result,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
