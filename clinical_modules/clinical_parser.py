from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Pattern, Tuple


@dataclass(frozen=True)
class FieldDefinition:
    """
    Definition for one clinical field.

    Attributes
    ----------
    name:
        Canonical output field name.
    patterns:
        Regular expressions used to locate the field.
    value_type:
        "float", "int", "string", or "sex".
    unit:
        Canonical unit used in the output metadata.
    min_value / max_value:
        Plausibility range. Values outside this range are retained as None
        and produce a warning.
    """
    name: str
    patterns: Tuple[str, ...]
    value_type: str
    unit: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class ParsedField:
    """
    Detailed parsing result for one field.
    """
    value: Any = None
    confidence: float = 0.0
    raw_match: Optional[str] = None
    normalized_value: Any = None
    unit: Optional[str] = None
    pattern_index: Optional[int] = None
    warnings: List[str] = field(default_factory=list)


class ClinicalParser:
    """
    Parse sleep-clinical report text into structured data.

    Main features
    -------------
    - Chinese and English label support
    - OCR-tolerant normalization
    - Numeric and percentage extraction
    - Plausibility range validation
    - Missing values represented as None
    - Per-field confidence and raw match tracking
    - JSON-compatible output
    - DocumentReader result integration
    """

    FIELD_DEFINITIONS: Tuple[FieldDefinition, ...] = (
        FieldDefinition(
            name="patient_id",
            patterns=(
                r"(?:患者|病人|個案)\s*(?:編號|代號|ID)\s*[:：]?\s*([A-Za-z0-9][A-Za-z0-9._\-/]{0,63})",
                r"(?:Patient\s*(?:ID|No\.?|Number))\s*[:：]?\s*([A-Za-z0-9][A-Za-z0-9._\-/]{0,63})",
                r"\bID\s*[:：]\s*([A-Za-z0-9][A-Za-z0-9._\-/]{0,63})",
            ),
            value_type="string",
        ),
        FieldDefinition(
            name="ahi",
            patterns=(
                r"(?<![A-Za-z])AHI(?:\s*\([^)]*\))?\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
                r"(?:呼吸中止低通氣指數|呼吸暫停低通氣指數|呼吸中止及低通氣指數)\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
                r"Apnea[-\s]*Hypopnea\s*Index\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
            ),
            value_type="float",
            unit="events/hour",
            min_value=0.0,
            max_value=200.0,
        ),
        FieldDefinition(
            name="odi",
            patterns=(
                r"(?<![A-Za-z])ODI(?:\s*\d+%?)?\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
                r"(?:氧減飽和指數|血氧下降指數|血氧去飽和指數)\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
                r"Oxygen\s*Desaturation\s*Index\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
            ),
            value_type="float",
            unit="events/hour",
            min_value=0.0,
            max_value=200.0,
        ),
        FieldDefinition(
            name="minimum_spo2",
            patterns=(
                r"(?:最低|最小|最低值|Minimum|Min\.?)\s*(?:SpO\s*2|SpO₂|SaO\s*2|SaO₂|血氧(?:飽和度)?)\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)\s*%?",
                r"(?:SpO\s*2|SpO₂|SaO\s*2|SaO₂)\s*(?:最低|最小|Minimum|Min\.?)\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)\s*%?",
                r"(?:Nadir\s*(?:SpO\s*2|SpO₂|SaO\s*2|SaO₂))\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)\s*%?",
            ),
            value_type="float",
            unit="%",
            min_value=20.0,
            max_value=100.0,
        ),
        FieldDefinition(
            name="mean_spo2",
            patterns=(
                r"(?:平均|Mean|Average|Avg\.?)\s*(?:SpO\s*2|SpO₂|SaO\s*2|SaO₂|血氧(?:飽和度)?)\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)\s*%?",
                r"(?:SpO\s*2|SpO₂|SaO\s*2|SaO₂)\s*(?:平均|Mean|Average|Avg\.?)\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)\s*%?",
            ),
            value_type="float",
            unit="%",
            min_value=20.0,
            max_value=100.0,
        ),
        FieldDefinition(
            name="bmi",
            patterns=(
                r"(?<![A-Za-z])BMI\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
                r"(?:身體質量指數|身體質量指数|體質量指數)\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
                r"Body\s*Mass\s*Index\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
            ),
            value_type="float",
            unit="kg/m²",
            min_value=8.0,
            max_value=100.0,
        ),
        FieldDefinition(
            name="age",
            patterns=(
                r"(?:年齡|年龄|Age)\s*[:：=]?\s*(\d{1,3})(?:\s*(?:歲|岁|years?|y/o))?",
            ),
            value_type="int",
            unit="years",
            min_value=0.0,
            max_value=130.0,
        ),
        FieldDefinition(
            name="sex",
            patterns=(
                r"(?:性別|性别|Sex|Gender)\s*[:：=]?\s*(男性|女性|男|女|Male|Female|M|F)\b",
            ),
            value_type="sex",
        ),
        FieldDefinition(
            name="total_sleep_time",
            patterns=(
                r"(?:總睡眠時間|总睡眠时间|Total\s*Sleep\s*Time|TST)\s*[:：=]?\s*(\d+(?:[.,]\d+)?)\s*(?:分鐘|分|min(?:utes?)?|小時|小时|hr(?:s)?|hours?)?",
            ),
            value_type="float",
            unit="minutes",
            min_value=0.0,
            max_value=1440.0,
        ),
        FieldDefinition(
            name="sleep_efficiency",
            patterns=(
                r"(?:睡眠效率|Sleep\s*Efficiency|SE)\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)\s*%?",
            ),
            value_type="float",
            unit="%",
            min_value=0.0,
            max_value=100.0,
        ),
        FieldDefinition(
            name="supine_ahi",
            patterns=(
                r"(?:仰臥|仰卧|Supine)\s*(?:AHI)?\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
                r"(?:AHI)\s*(?:仰臥|仰卧|Supine)\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
            ),
            value_type="float",
            unit="events/hour",
            min_value=0.0,
            max_value=200.0,
        ),
        FieldDefinition(
            name="non_supine_ahi",
            patterns=(
                r"(?:非仰臥|非仰卧|Non[-\s]*Supine)\s*(?:AHI)?\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
                r"(?:AHI)\s*(?:非仰臥|非仰卧|Non[-\s]*Supine)\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
            ),
            value_type="float",
            unit="events/hour",
            min_value=0.0,
            max_value=200.0,
        ),
        FieldDefinition(
            name="rem_ahi",
            patterns=(
                r"(?:REM)\s*(?:AHI)?\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
                r"(?:AHI)\s*(?:REM)\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
            ),
            value_type="float",
            unit="events/hour",
            min_value=0.0,
            max_value=200.0,
        ),
        FieldDefinition(
            name="nrem_ahi",
            patterns=(
                r"(?:NREM|Non[-\s]*REM)\s*(?:AHI)?\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
                r"(?:AHI)\s*(?:NREM|Non[-\s]*REM)\s*[:：=]?\s*([<>]?\s*-?\d+(?:[.,]\d+)?)",
            ),
            value_type="float",
            unit="events/hour",
            min_value=0.0,
            max_value=200.0,
        ),
    )

    def __init__(self) -> None:
        self._compiled_patterns: Dict[str, Tuple[Pattern[str], ...]] = {
            definition.name: tuple(
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in definition.patterns
            )
            for definition in self.FIELD_DEFINITIONS
        }

    def parse(
        self,
        text: str,
        *,
        source_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse raw clinical text.

        Parameters
        ----------
        text:
            Text extracted from a clinical report.
        source_filename:
            Optional source filename for traceability.
        """
        warnings: List[str] = []
        normalized_text = self._normalize_text(text)

        result: Dict[str, Any] = {
            "source_filename": source_filename,
            "data": {},
            "fields": {},
            "derived": {},
            "metadata": {
                "parser": self.__class__.__name__,
                "parser_version": "1.0.0",
                "input_character_count": len(text or ""),
                "normalized_character_count": len(normalized_text),
            },
            "warnings": warnings,
        }

        if not normalized_text:
            warnings.append("沒有可供解析的臨床文字。")
            for definition in self.FIELD_DEFINITIONS:
                empty_field = ParsedField(unit=definition.unit)
                result["data"][definition.name] = None
                result["fields"][definition.name] = asdict(empty_field)
            result["metadata"]["parsed_field_count"] = 0
            result["metadata"]["missing_field_count"] = len(
                self.FIELD_DEFINITIONS
            )
            return result

        for definition in self.FIELD_DEFINITIONS:
            parsed_field = self._parse_field(
                normalized_text,
                definition,
            )
            result["data"][definition.name] = parsed_field.value
            result["fields"][definition.name] = asdict(parsed_field)

            for warning in parsed_field.warnings:
                warnings.append(f"{definition.name}：{warning}")

        result["derived"] = self._derive_values(result["data"], warnings)

        parsed_count = sum(
            value is not None
            for value in result["data"].values()
        )
        total_count = len(self.FIELD_DEFINITIONS)

        result["metadata"]["parsed_field_count"] = parsed_count
        result["metadata"]["missing_field_count"] = total_count - parsed_count
        result["metadata"]["overall_confidence"] = self._overall_confidence(
            result["fields"]
        )

        return self._make_json_compatible(result)

    def parse_document(
        self,
        document_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Parse the dictionary returned by DocumentReader.read().
        """
        if not isinstance(document_result, dict):
            return self.parse("")

        parsed = self.parse(
            str(document_result.get("text") or ""),
            source_filename=document_result.get("filename"),
        )

        reader_warnings = document_result.get("warnings") or []
        if reader_warnings:
            parsed["metadata"]["document_reader_warning_count"] = len(
                reader_warnings
            )
            parsed["document_reader_warnings"] = [
                str(item)
                for item in reader_warnings
            ]

        reader_metadata = document_result.get("metadata")
        if isinstance(reader_metadata, dict):
            parsed["source_metadata"] = self._make_json_compatible(
                reader_metadata
            )

        return parsed

    def _parse_field(
        self,
        text: str,
        definition: FieldDefinition,
    ) -> ParsedField:
        """
        Parse one field by trying its patterns in priority order.
        """
        candidates: List[Tuple[int, re.Match[str]]] = []

        for pattern_index, pattern in enumerate(
            self._compiled_patterns[definition.name]
        ):
            for match in pattern.finditer(text):
                candidates.append((pattern_index, match))

        if not candidates:
            return ParsedField(unit=definition.unit)

        # Prefer earlier patterns; then earlier occurrence in the text.
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1].start(),
            )
        )

        pattern_index, match = candidates[0]
        raw_value = match.group(1).strip()
        raw_match = match.group(0).strip()

        value, conversion_warning = self._convert_value(
            raw_value,
            definition.value_type,
            raw_match=raw_match,
            field_name=definition.name,
        )

        field_warnings: List[str] = []
        if conversion_warning:
            field_warnings.append(conversion_warning)

        normalized_value = value

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not self._is_within_range(
                float(value),
                definition.min_value,
                definition.max_value,
            ):
                field_warnings.append(
                    f"解析值 {value} 超出合理範圍"
                    f"（{definition.min_value}–{definition.max_value}），"
                    "已設為 None。"
                )
                value = None

        if len(candidates) > 1:
            distinct_values = {
                candidate_match.group(1).strip()
                for _, candidate_match in candidates
            }
            if len(distinct_values) > 1:
                field_warnings.append(
                    "找到多個不同候選值，已採用優先順序最高者。"
                )

        confidence = self._calculate_confidence(
            value=value,
            pattern_index=pattern_index,
            candidate_count=len(candidates),
            raw_match=raw_match,
            warning_count=len(field_warnings),
        )

        return ParsedField(
            value=value,
            confidence=confidence,
            raw_match=raw_match,
            normalized_value=normalized_value,
            unit=definition.unit,
            pattern_index=pattern_index,
            warnings=field_warnings,
        )

    def _convert_value(
        self,
        raw_value: str,
        value_type: str,
        *,
        raw_match: str,
        field_name: str,
    ) -> Tuple[Any, Optional[str]]:
        """
        Convert a captured string into the requested canonical type.
        """
        if value_type == "string":
            value = raw_value.strip(" \t\r\n:：,，;；")
            return (value or None), None

        if value_type == "sex":
            normalized = raw_value.strip().lower()
            male_values = {"男", "男性", "male", "m"}
            female_values = {"女", "女性", "female", "f"}

            if normalized in male_values:
                return "male", None
            if normalized in female_values:
                return "female", None
            return None, f"無法辨識性別值：{raw_value}"

        number_text = self._normalize_number_text(raw_value)
        number_match = re.search(
            r"-?\d+(?:\.\d+)?",
            number_text,
        )

        if not number_match:
            return None, f"無法轉換數值：{raw_value}"

        try:
            numeric_value = float(number_match.group(0))
        except ValueError:
            return None, f"無法轉換數值：{raw_value}"

        # Total sleep time expressed in hours is converted to minutes.
        if field_name == "total_sleep_time":
            if re.search(
                r"(?:小時|小时|hr(?:s)?|hours?)",
                raw_match,
                re.IGNORECASE,
            ):
                numeric_value *= 60.0

        if value_type == "int":
            if not numeric_value.is_integer():
                return int(round(numeric_value)), (
                    f"原始值 {numeric_value} 不是整數，已四捨五入。"
                )
            return int(numeric_value), None

        return numeric_value, None

    def _derive_values(
        self,
        data: Dict[str, Any],
        warnings: List[str],
    ) -> Dict[str, Any]:
        """
        Derive simple clinically useful classifications.

        These are descriptive classifications, not treatment decisions.
        """
        derived: Dict[str, Any] = {
            "ahi_severity": None,
            "hypoxemia_level": None,
            "positional_ratio": None,
            "position_relevance": None,
        }

        ahi = data.get("ahi")
        if isinstance(ahi, (int, float)):
            if ahi < 5:
                derived["ahi_severity"] = "normal"
            elif ahi < 15:
                derived["ahi_severity"] = "mild"
            elif ahi < 30:
                derived["ahi_severity"] = "moderate"
            else:
                derived["ahi_severity"] = "severe"

        minimum_spo2 = data.get("minimum_spo2")
        if isinstance(minimum_spo2, (int, float)):
            if minimum_spo2 >= 90:
                derived["hypoxemia_level"] = "none_or_mild"
            elif minimum_spo2 >= 80:
                derived["hypoxemia_level"] = "moderate"
            else:
                derived["hypoxemia_level"] = "severe"

        supine_ahi = data.get("supine_ahi")
        non_supine_ahi = data.get("non_supine_ahi")

        if (
            isinstance(supine_ahi, (int, float))
            and isinstance(non_supine_ahi, (int, float))
        ):
            if non_supine_ahi > 0:
                ratio = round(supine_ahi / non_supine_ahi, 3)
                derived["positional_ratio"] = ratio
                derived["position_relevance"] = ratio >= 2.0
            elif supine_ahi > 0:
                derived["positional_ratio"] = None
                derived["position_relevance"] = True
                warnings.append(
                    "non_supine_ahi 為 0，無法計算有限的仰臥比值；"
                    "position_relevance 暫設為 True。"
                )

        return derived

    def _normalize_text(self, text: Any) -> str:
        """
        Normalize OCR and report text without aggressively guessing content.
        """
        if text is None:
            return ""

        normalized = str(text)

        replacements = {
            "\u00a0": " ",
            "\u3000": " ",
            "﹕": "：",
            "∶": "：",
            "；": "；",
            "ＳｐＯ２": "SpO2",
            "ＳｐＯ₂": "SpO2",
            "Sp02": "SpO2",
            "SP02": "SpO2",
            "spo2": "SpO2",
            "ＳａＯ２": "SaO2",
            "Sa02": "SaO2",
            "ＡＨＩ": "AHI",
            "ＯＤＩ": "ODI",
            "ＢＭＩ": "BMI",
        }

        for old, new in replacements.items():
            normalized = normalized.replace(old, new)

        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        # Normalize common OCR spacing in clinical abbreviations.
        normalized = re.sub(
            r"\bA\s*H\s*I\b",
            "AHI",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\bO\s*D\s*I\b",
            "ODI",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\bB\s*M\s*I\b",
            "BMI",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\bSpO\s*[₂2]\b",
            "SpO2",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\bSaO\s*[₂2]\b",
            "SaO2",
            normalized,
            flags=re.IGNORECASE,
        )

        return normalized.strip()

    def _normalize_number_text(self, value: str) -> str:
        """
        Normalize a numeric token conservatively.
        """
        normalized = value.strip()
        normalized = normalized.replace("，", ",")
        normalized = normalized.replace("。", ".")
        normalized = normalized.replace("％", "%")
        normalized = normalized.replace("−", "-")
        normalized = normalized.replace("–", "-")
        normalized = normalized.replace("—", "-")
        normalized = normalized.replace("＜", "<")
        normalized = normalized.replace("＞", ">")
        normalized = re.sub(r"\s+", "", normalized)

        # A single comma between digits is treated as a decimal separator.
        if re.fullmatch(r"[<>]?-?\d+,\d+", normalized):
            normalized = normalized.replace(",", ".")
        else:
            normalized = normalized.replace(",", "")

        return normalized

    def _calculate_confidence(
        self,
        *,
        value: Any,
        pattern_index: int,
        candidate_count: int,
        raw_match: str,
        warning_count: int,
    ) -> float:
        """
        Produce a transparent heuristic confidence score in [0, 1].
        """
        if value is None:
            return 0.0

        confidence = 0.96 - (pattern_index * 0.05)

        if candidate_count > 1:
            confidence -= min(0.15, 0.03 * (candidate_count - 1))

        if warning_count:
            confidence -= min(0.25, 0.08 * warning_count)

        if ":" not in raw_match and "：" not in raw_match and "=" not in raw_match:
            confidence -= 0.03

        return round(max(0.0, min(1.0, confidence)), 3)

    def _overall_confidence(
        self,
        fields: Dict[str, Dict[str, Any]],
    ) -> float:
        """
        Average confidence across fields that were actually parsed.
        """
        scores = [
            float(field_result.get("confidence", 0.0))
            for field_result in fields.values()
            if field_result.get("value") is not None
        ]

        if not scores:
            return 0.0

        return round(sum(scores) / len(scores), 3)

    def _is_within_range(
        self,
        value: float,
        min_value: Optional[float],
        max_value: Optional[float],
    ) -> bool:
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True

    def _make_json_compatible(self, value: Any) -> Any:
        """
        Recursively convert non-JSON-safe values and NaN/Infinity.
        """
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        if isinstance(value, str):
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

    sample_text = """
    睡眠檢查報告
    患者編號：P001
    年齡：52 歲
    性別：男
    BMI：29.4
    AHI：32.5 次/小時
    ODI：28.1 次/小時
    最低 SpO₂：78%
    平均 SpO₂：93.2%
    總睡眠時間：356.5 分鐘
    睡眠效率：81.6%
    仰臥 AHI：48.2
    非仰臥 AHI：17.1
    REM AHI：41.3
    NREM AHI：29.8
    """

    parser = ClinicalParser()
    result = parser.parse(
        sample_text,
        source_filename="sample_report.txt",
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
