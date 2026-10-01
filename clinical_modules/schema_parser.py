from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

from clinical_modules.base_parser import BaseClinicalParser, ParsedClinicalData
from clinical_modules.document_reader import DocumentReader


@dataclass(frozen=True)
class ClinicalFieldSpec:
    name: str
    aliases: tuple[str, ...]
    value_type: str = "string"
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClinicalModuleSpec:
    module_id: str
    display_name: str
    fields: tuple[ClinicalFieldSpec, ...]
    supported_extensions: frozenset[str] = frozenset({
        ".pdf", ".xls", ".xlsx", ".csv", ".txt", ".docx",
        ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp",
    })


class SchemaClinicalParser(BaseClinicalParser):
    """Schema-driven parser shared by non-PAP clinical modules."""

    def __init__(self, spec: ClinicalModuleSpec) -> None:
        self.spec = spec
        self.module_id = spec.module_id
        self.display_name = spec.display_name
        self.supported_extensions = set(spec.supported_extensions)
        self.reader = DocumentReader()
        self._alias_map = self._build_alias_map(spec.fields)

    def parse(self, uploaded_files: list[BinaryIO]) -> ParsedClinicalData:
        self.validate_files(uploaded_files)

        merged: dict[str, Any] = {}
        source_files: list[str] = []
        warnings: list[str] = []
        field_sources: dict[str, str] = {}
        document_metadata: list[dict[str, Any]] = []
        extracted_text_parts: list[str] = []

        for uploaded_file in uploaded_files:
            filename = str(getattr(uploaded_file, "name", "unknown_file"))
            source_files.append(filename)

            try:
                result = self.reader.read(uploaded_file)
            except Exception as exc:
                warnings.append(f"無法讀取 `{filename}`：{type(exc).__name__}: {exc}")
                continue

            warnings.extend(str(item) for item in result.get("warnings", []) if item)
            document_metadata.append({
                "filename": filename,
                "extension": result.get("extension"),
                **(result.get("metadata") if isinstance(result.get("metadata"), dict) else {}),
            })

            candidates: dict[str, Any] = {}
            text = str(result.get("text") or "")
            if text:
                extracted_text_parts.append(text)
                candidates.update(self._extract_from_text(text))

            tables = result.get("tables")
            if isinstance(tables, list):
                candidates.update(self._extract_from_tables(tables))

            candidates.update(self._module_specific_fallback(text, filename, candidates))

            for field_name, value in candidates.items():
                if value is None or value == "":
                    continue
                if field_name in merged and merged[field_name] != value:
                    warnings.append(
                        f"欄位 `{field_name}` 在多個檔案中出現不同值；"
                        f"已採用 `{filename}` 的較後值。"
                    )
                merged[field_name] = value
                field_sources[field_name] = filename

        missing = [field.name for field in self.spec.fields if field.name not in merged]
        if not merged:
            warnings.append(
                f"未從上傳檔案辨識出 {self.display_name} 的結構化欄位；"
                "請由醫師查看原始文件並手動確認。"
            )

        return ParsedClinicalData(
            module_id=self.module_id,
            data=merged,
            source_files=source_files,
            warnings=self._deduplicate(warnings),
            metadata={
                "parser": self.__class__.__name__,
                "parser_version": "2.1.0",
                "module_id": self.module_id,
                "display_name": self.display_name,
                "field_sources": field_sources,
                "recognized_fields": sorted(merged),
                "missing_fields": missing,
                "documents": document_metadata,
                "raw_text_excerpt": "\n\n".join(extracted_text_parts)[:4000],
                "requires_physician_confirmation": True,
            },
        )

    def _module_specific_fallback(
        self,
        text: str,
        filename: str,
        existing: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Handle common report layouts not written as exact key-value pairs."""
        normalized = self._normalize_text(text or "")
        lower = normalized.lower()
        output: dict[str, Any] = {}

        if self.module_id == "IMAGING":
            if "modality" not in existing:
                for modality, patterns in {
                    "CBCT": ("cbct", "cone beam"),
                    "CT": ("computed tomography", " ct ", "電腦斷層", "电脑断层"),
                    "MRI": ("mri", "magnetic resonance", "磁振造影", "磁共振"),
                    "X-ray": ("x-ray", "x ray", "radiograph", "x光"),
                    "cephalometry": ("cephalometric", "頭影測量", "头影测量"),
                }.items():
                    if any(item in f" {lower} {filename.lower()} " for item in patterns):
                        output["modality"] = modality
                        break
            impression = self._extract_section(normalized, ("impression", "conclusion", "findings", "影像結論", "影像所見", "結論", "所見"))
            if impression and "other_anatomical_obstruction" not in existing:
                output["other_anatomical_obstruction"] = impression[:500]

        elif self.module_id == "DISE":
            # Common VOTE shorthand, e.g. V2AP O1LAT T2AP E0.
            vote_tokens = re.findall(r"\b([VOTE])\s*([012])\s*([A-Z]{0,4})", normalized, flags=re.I)
            if vote_tokens:
                output.setdefault("vote_classification", " ".join("".join(token) for token in vote_tokens))
                mapping = {"V": "velum_collapse", "O": "oropharyngeal_lateral_wall_collapse", "T": "tongue_base_collapse", "E": "epiglottis_collapse"}
                degrees = []
                patterns = []
                for site, degree, pattern in vote_tokens:
                    field = mapping[site.upper()]
                    output.setdefault(field, {"0": "none", "1": "partial", "2": "complete"}[degree])
                    degrees.append(int(degree))
                    if pattern:
                        patterns.append(pattern.upper())
                output.setdefault("collapse_degree", "complete" if 2 in degrees else "partial" if 1 in degrees else "none")
                joined = " ".join(patterns)
                if "AP" in joined: output.setdefault("collapse_pattern", "anteroposterior")
                elif "LAT" in joined or "L" in joined: output.setdefault("collapse_pattern", "lateral")
                elif "C" in joined: output.setdefault("collapse_pattern", "concentric")

        elif self.module_id == "OTHER":
            output.setdefault("document_type", Path(filename).suffix.lstrip(".").upper() + " 臨床文件")
            if normalized.strip():
                output.setdefault("clinical_summary", normalized.strip()[:1500])
                assessment = self._extract_section(normalized, ("assessment", "impression", "診斷", "評估", "診斷印象"))
                plan = self._extract_section(normalized, ("plan", "treatment plan", "計畫", "治療計畫", "處置"))
                if assessment: output.setdefault("assessment", assessment[:800])
                if plan: output.setdefault("plan", plan[:800])

        return {key: value for key, value in output.items() if key not in existing and value not in (None, "")}

    @staticmethod
    def _extract_section(text: str, headings: Sequence[str]) -> str | None:
        for heading in headings:
            match = re.search(
                rf"(?:^|\n)\s*{re.escape(heading)}\s*[:：]?\s*(.+?)(?=\n\s*[A-Za-z\u4e00-\u9fff][^\n]{{0,30}}[:：]\s*|\Z)",
                text,
                flags=re.I | re.S,
            )
            if match:
                value = re.sub(r"\s+", " ", match.group(1)).strip()
                if value:
                    return value
        return None

    def _extract_from_text(self, text: str) -> dict[str, Any]:
        normalized = self._normalize_text(text)
        output: dict[str, Any] = {}

        for field in self.spec.fields:
            value = self._search_field_in_text(field, normalized)
            if value is not None:
                output[field.name] = value

        return output

    def _search_field_in_text(self, field: ClinicalFieldSpec, text: str) -> Any:
        # Include the canonical field name because exported PDF/Word reports
        # commonly use machine-readable labels such as ``accept_cpap``.
        aliases = (field.name, *field.aliases)
        alias_group = "|".join(
            sorted((re.escape(alias) for alias in aliases), key=len, reverse=True)
        )
        if not alias_group:
            return None

        value_pattern = self._value_capture_pattern(field)
        patterns = (
            rf"(?:{alias_group})\s*(?:[:：=]|is|為|为)?\s*{value_pattern}",
            rf"{value_pattern}\s*(?:[,，;；]|\s)+(?:{alias_group})",
        )

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                raw = match.group("value")
                return self._normalize_value(field, raw)

        if field.value_type == "boolean":
            return self._infer_boolean(field, text)

        return None

    def _value_capture_pattern(self, field: ClinicalFieldSpec) -> str:
        if field.value_type in {"float", "int"}:
            return r"(?P<value>[<>]?-?\d+(?:\.\d+)?)"
        if field.value_type == "boolean":
            return (
                r"(?P<value>有|無|无|是|否|陽性|阳性|陰性|阴性|"
                r"present|absent|positive|negative|yes|no|true|false)"
            )
        if field.choices:
            choices = "|".join(
                sorted((re.escape(choice) for choice in field.choices), key=len, reverse=True)
            )
            return rf"(?P<value>{choices}|[^\n\r,，;；]{1,80})"
        return r"(?P<value>[^\n\r,，;；]{1,120})"

    def _extract_from_tables(self, tables: Sequence[Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}

        for table in tables:
            rows = self._table_rows(table)
            for row in rows:
                if not isinstance(row, Mapping):
                    continue

                for raw_key, raw_value in row.items():
                    field = self._field_for_label(raw_key)
                    if field is not None:
                        value = self._normalize_value(field, raw_value)
                        if value is not None:
                            output[field.name] = value

                values = list(row.values())
                if len(values) >= 2:
                    for index in range(len(values) - 1):
                        field = self._field_for_label(values[index])
                        if field is None:
                            continue
                        value = self._normalize_value(field, values[index + 1])
                        if value is not None:
                            output[field.name] = value

        return output

    @staticmethod
    def _table_rows(table: Any) -> list[Mapping[str, Any]]:
        if isinstance(table, list):
            output = []
            for row in table:
                if isinstance(row, Mapping):
                    output.append(row)
                elif isinstance(row, (list, tuple)):
                    output.append({str(i): value for i, value in enumerate(row)})
            return output
        if isinstance(table, Mapping):
            records = table.get("records")
            if not isinstance(records, list):
                records = table.get("rows")
            if isinstance(records, list):
                output = []
                for row in records:
                    if isinstance(row, Mapping):
                        output.append(row)
                    elif isinstance(row, (list, tuple)):
                        output.append({str(i): value for i, value in enumerate(row)})
                return output
            return [table]
        return []

    def _field_for_label(self, label: Any) -> ClinicalFieldSpec | None:
        normalized = self._normalize_label(label)
        field_name = self._alias_map.get(normalized)
        if field_name is None:
            return None
        return next((item for item in self.spec.fields if item.name == field_name), None)

    def _normalize_value(self, field: ClinicalFieldSpec, raw_value: Any) -> Any:
        if raw_value is None:
            return None
        if isinstance(raw_value, float) and not math.isfinite(raw_value):
            return None

        if field.value_type in {"float", "int"}:
            match = re.search(r"[<>]?\s*(-?\d+(?:\.\d+)?)", str(raw_value).replace(",", ""))
            if not match:
                return None
            value = float(match.group(1))
            if field.minimum is not None and value < field.minimum:
                return None
            if field.maximum is not None and value > field.maximum:
                return None
            return int(round(value)) if field.value_type == "int" else value

        if field.value_type == "boolean":
            return self._parse_boolean(raw_value)

        text = str(raw_value).strip().strip("：:=")
        if not text:
            return None

        if field.choices:
            lowered = text.lower()
            for choice in sorted(field.choices, key=len, reverse=True):
                if choice.lower() == lowered or choice.lower() in lowered:
                    return choice

        return re.sub(r"\s+", " ", text)[:500]

    def _infer_boolean(self, field: ClinicalFieldSpec, text: str) -> bool | None:
        for alias in sorted(field.aliases, key=len, reverse=True):
            escaped = re.escape(alias)
            negative = re.search(
                rf"(?:無|无|未見|未见|沒有|没有|否認|否认|no|without|absent|negative)"
                rf"[^\n\r]{{0,20}}{escaped}|{escaped}[^\n\r]{{0,20}}"
                rf"(?:無|无|未見|未见|沒有|没有|否|absent|negative)",
                text,
                flags=re.IGNORECASE,
            )
            if negative:
                return False
            positive = re.search(
                rf"(?:有|可見|可见|顯示|显示|陽性|阳性|present|positive)"
                rf"[^\n\r]{{0,20}}{escaped}|{escaped}[^\n\r]{{0,20}}"
                rf"(?:有|可見|可见|陽性|阳性|present|positive|yes)",
                text,
                flags=re.IGNORECASE,
            )
            if positive:
                return True
        return None

    @staticmethod
    def _parse_boolean(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        truthy = {"有", "是", "陽性", "阳性", "present", "positive", "yes", "true", "1"}
        falsy = {"無", "无", "否", "陰性", "阴性", "absent", "negative", "no", "false", "0"}
        if text in truthy:
            return True
        if text in falsy:
            return False
        return None

    @staticmethod
    def _build_alias_map(fields: Iterable[ClinicalFieldSpec]) -> dict[str, str]:
        result: dict[str, str] = {}
        for field in fields:
            for alias in (field.name, *field.aliases):
                result[SchemaClinicalParser._normalize_label(alias)] = field.name
        return result

    @staticmethod
    def _normalize_label(value: Any) -> str:
        text = str(value or "").strip().lower().replace("₂", "2").replace("％", "%")
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\u3000", " ").replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("₂", "2").replace("％", "%")
        return re.sub(r"[ \t]+", " ", text)

    @staticmethod
    def _deduplicate(values: Iterable[str]) -> list[str]:
        output: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in output:
                output.append(text)
        return output
