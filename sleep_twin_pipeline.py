from __future__ import annotations

import re
import argparse
import csv
import io
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from clinical_modules_backup.document_reader import DocumentReader
from clinical_modules_backup.clinical_parser import ClinicalParser
from clinical_modules_backup.rule_engine import RuleEngine
from clinical_modules_backup.recommendation_engine import RecommendationEngine
from clinical_modules_backup.report_generator import ReportGenerator


class SleepTwinPipeline:
    """
    End-to-end pipeline:

        DocumentReader
            -> ClinicalParser
            -> RuleEngine
            -> RecommendationEngine
            -> ReportGenerator

    The pipeline accepts a report file and exports JSON and TXT results.
    """

    PIPELINE_VERSION = "1.2.2"

    def __init__(
        self,
        output_dir: str | Path = "output",
    ) -> None:
        self.output_dir = Path(output_dir)

        self.document_reader = DocumentReader()
        self.clinical_parser = ClinicalParser()
        self.rule_engine = RuleEngine()
        self.recommendation_engine = RecommendationEngine()
        self.report_generator = ReportGenerator()

    def run(
        self,
        input_path: str | Path,
        *,
        patient_id: Optional[str] = None,
        output_dir: str | Path | None = None,
        save_intermediate: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the complete workflow.

        CSV/XLSX files containing multiple rows are evaluated one patient
        at a time. Other file types remain single-report evaluations.
        """
        source_path = Path(input_path).expanduser().resolve()

        if not source_path.exists():
            raise FileNotFoundError(
                f"找不到輸入檔案：{source_path}"
            )

        if not source_path.is_file():
            raise ValueError(
                f"輸入路徑不是檔案：{source_path}"
            )

        active_output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else self.output_dir.expanduser().resolve()
        )
        active_output_dir.mkdir(parents=True, exist_ok=True)

        raw_text, reader_result = self._read_document(source_path)
        records = self._extract_structured_records(
            source_path,
            reader_result,
            raw_text,
        )

        if records:
            results = []

            for row_index, record in enumerate(
                records,
                start=1,
            ):
                record_text = self._record_to_clinical_text(
                    record
                )
                record_patient_id = (
                    patient_id
                    or self._record_patient_id(record)
                )
                suffix = (
                    record_patient_id
                    or f"row_{row_index:03d}"
                )

                result = self._process_single_text(
                    raw_text=record_text,
                    reader_result=reader_result,
                    source_path=source_path,
                    active_output_dir=active_output_dir,
                    patient_id=record_patient_id,
                    save_intermediate=save_intermediate,
                    output_suffix=suffix,
                    source_row_index=row_index,
                    source_record=record,
                )
                results.append(result)

            return {
                "multi_patient": True,
                "source_file": str(source_path),
                "patient_count": len(results),
                "results": results,
            }

        return self._process_single_text(
            raw_text=raw_text,
            reader_result=reader_result,
            source_path=source_path,
            active_output_dir=active_output_dir,
            patient_id=patient_id,
            save_intermediate=save_intermediate,
        )

    def _process_single_text(
        self,
        *,
        raw_text: str,
        reader_result: Dict[str, Any],
        source_path: Path,
        active_output_dir: Path,
        patient_id: Optional[str],
        save_intermediate: bool,
        output_suffix: Optional[str] = None,
        source_row_index: Optional[int] = None,
        source_record: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        parsed_result = self._parse_text(
            raw_text,
            source_path=source_path,
        )

        if patient_id:
            self._inject_patient_id(
                parsed_result,
                patient_id,
            )

        rule_result = self.rule_engine.evaluate(
            parsed_result
        )
        rule_result = self._normalize_rule_priority(
            rule_result
        )

        recommendation_result = (
            self.recommendation_engine.recommend(
                parsed_result,
                rule_result,
            )
        )

        report = self.report_generator.generate(
            parsed_result,
            rule_result,
            recommendation_result,
            source_filename=source_path.name,
        )

        extension_name = (
            source_path.suffix.lower().lstrip(".")
            or "no_extension"
        )
        base_value = (
            f"{source_path.stem}_{extension_name}"
        )

        if output_suffix:
            base_value += f"_{output_suffix}"

        base_name = self._safe_stem(base_value)
        output_paths = self._build_output_paths(
            active_output_dir,
            base_name,
        )

        self.report_generator.save_json(
            report,
            str(output_paths["report_json"]),
        )
        self.report_generator.save_text(
            report,
            str(output_paths["report_text"]),
        )

        if save_intermediate:
            patient_reader_result = dict(reader_result)

            if source_row_index is not None:
                patient_reader_result[
                    "selected_row_index"
                ] = source_row_index
                patient_reader_result[
                    "selected_record"
                ] = source_record
                patient_reader_result[
                    "text"
                ] = raw_text

            self._save_json(
                patient_reader_result,
                output_paths["reader_json"],
            )
            self._save_json(
                parsed_result,
                output_paths["parsed_json"],
            )
            self._save_json(
                rule_result,
                output_paths["rules_json"],
            )
            self._save_json(
                recommendation_result,
                output_paths["recommendations_json"],
            )

        manifest = {
            "status": "success",
            "pipeline": "SleepTwinPipeline",
            "pipeline_version": self.PIPELINE_VERSION,
            "input_file": str(source_path),
            "source_row_index": source_row_index,
            "source_record": source_record,
            "output_directory": str(active_output_dir),
            "outputs": {
                key: str(value)
                for key, value in output_paths.items()
                if save_intermediate
                or key in {
                    "report_json",
                    "report_text",
                    "manifest_json",
                }
            },
            "summary": {
                "patient_id": (
                    report.get("patient_summary", {}).get(
                        "patient_id"
                    )
                    if isinstance(
                        report.get("patient_summary"),
                        dict,
                    )
                    else None
                ),
                "highest_priority": (
                    report.get("risk_summary", {}).get(
                        "highest_priority"
                    )
                    if isinstance(
                        report.get("risk_summary"),
                        dict,
                    )
                    else None
                ),
                "primary_recommendation": (
                    self._extract_primary_recommendation(
                        report,
                        recommendation_result,
                    )
                ),
                "warning_count": len(
                    report.get("warnings", [])
                    if isinstance(
                        report.get("warnings"),
                        list,
                    )
                    else []
                ),
            },
            "completed_at": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        self._save_json(
            manifest,
            output_paths["manifest_json"],
        )

        return {
            "manifest": manifest,
            "reader_result": reader_result,
            "parsed_result": parsed_result,
            "rule_result": rule_result,
            "recommendation_result": recommendation_result,
            "report": report,
        }

    def _normalize_rule_priority(
        self,
        rule_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Keep data-completeness as an important flag, but do not let it hide
        an already evaluable clinical risk.

        Example:
        moderate_hypoxemia + moderate_osa + insufficient_core_data
        -> highest_priority = moderate_hypoxemia

        The original priority_flags remain unchanged.
        """
        if not isinstance(rule_result, dict):
            return rule_result

        flags = rule_result.get("priority_flags")
        summary = rule_result.get("summary")

        if not isinstance(flags, list):
            return rule_result

        if not isinstance(summary, dict):
            return rule_result

        current = summary.get("highest_priority")

        if current != "insufficient_core_data":
            return rule_result

        clinical_order = (
            "critical_hypoxemia",
            "severe_hypoxemia",
            "severe_osa",
            "moderate_hypoxemia",
            "moderate_osa",
            "mild_hypoxemia",
            "mild_osa",
        )

        selected = next(
            (
                flag
                for flag in clinical_order
                if flag in flags
            ),
            None,
        )

        if selected is None:
            return rule_result

        summary["highest_priority"] = selected
        summary[
            "data_completeness_flag"
        ] = "insufficient_core_data"
        summary[
            "priority_adjustment_reason"
        ] = (
            "Clinical risk takes precedence over missing "
            "non-blocking core fields; data completeness "
            "remains in priority_flags."
        )

        return rule_result

    def _extract_primary_recommendation(
        self,
        report: Dict[str, Any],
        recommendation_result: Dict[str, Any],
    ) -> Optional[str]:
        """
        Read the primary recommendation across compatible result schemas.

        RecommendationEngine / ReportGenerator versions may expose the
        recommendation under slightly different keys. This helper only
        searches recommendation-related containers, avoiding unrelated IDs.
        """
        direct_keys = (
            "primary_recommendation",
            "primary_recommendation_id",
            "recommendation",
            "recommendation_id",
            "recommendation_code",
            "primary_action",
            "action",
            "code",
        )
        container_keys = (
            "recommendation_summary",
            "recommendation_result",
            "recommendations",
            "primary",
            "selected_recommendation",
            "top_recommendation",
        )

        def clean(value: Any) -> Optional[str]:
            if isinstance(value, str):
                text = value.strip()
                return text or None

            if isinstance(value, (int, float)):
                return str(value)

            return None

        def search(value: Any, depth: int = 0) -> Optional[str]:
            if depth > 6:
                return None

            if isinstance(value, dict):
                for key in direct_keys:
                    if key not in value:
                        continue

                    candidate = value.get(key)
                    cleaned = clean(candidate)

                    if cleaned:
                        return cleaned

                    if isinstance(candidate, (dict, list)):
                        nested = search(candidate, depth + 1)

                        if nested:
                            return nested

                for key in container_keys:
                    if key not in value:
                        continue

                    nested = search(
                        value.get(key),
                        depth + 1,
                    )

                    if nested:
                        return nested

                # Some engines return a single recommendation object
                # with descriptive identifiers under these keys.
                for key in (
                    "name",
                    "type",
                    "category",
                    "title",
                ):
                    cleaned = clean(value.get(key))

                    if cleaned:
                        return cleaned

            elif isinstance(value, list):
                for item in value:
                    nested = search(item, depth + 1)

                    if nested:
                        return nested

            return None

        for source in (
            report.get("recommendation_summary"),
            report.get("recommendations"),
            recommendation_result,
        ):
            result = search(source)

            if result:
                return result

        return None

    def _extract_structured_records(
        self,
        source_path: Path,
        reader_result: Dict[str, Any],
        raw_text: str,
    ) -> list[Dict[str, Any]]:
        extension = source_path.suffix.lower()

        if extension == ".csv":
            try:
                rows = list(
                    csv.DictReader(
                        io.StringIO(raw_text)
                    )
                )
            except Exception:
                return []

            return [
                {
                    str(key).strip(): value
                    for key, value in row.items()
                    if key is not None
                }
                for row in rows
                if any(
                    str(value).strip()
                    for value in row.values()
                    if value is not None
                )
            ]

        if extension in {".xlsx", ".xls"}:
            tables = reader_result.get("tables", [])

            if not isinstance(tables, list):
                return []

            records = []

            for table in tables:
                if not isinstance(table, dict):
                    continue

                rows = table.get("rows", [])

                if not isinstance(rows, list):
                    continue

                for row in rows:
                    if isinstance(row, dict) and row:
                        records.append(dict(row))

            return records

        return []

    def _record_patient_id(
        self,
        record: Dict[str, Any],
    ) -> Optional[str]:
        aliases = {
            "patient_id",
            "patient id",
            "patientid",
            "病歷號",
            "病歷號碼",
            "患者編號",
        }

        for key, value in record.items():
            normalized_key = str(key).strip().lower()

            if normalized_key not in aliases:
                continue

            if value is None:
                continue

            text = str(value).strip()

            if text:
                return text

        return None

    def _record_to_clinical_text(
        self,
        record: Dict[str, Any],
    ) -> str:
        aliases = {
            "patient_id": "patient_id",
            "patient id": "patient_id",
            "patientid": "patient_id",
            "病歷號": "patient_id",
            "病歷號碼": "patient_id",
            "患者編號": "patient_id",
            "ahi": "AHI",
            "min_spo2": "minimum SpO2",
            "minimum_spo2": "minimum SpO2",
            "minimum spo2": "minimum SpO2",
            "最低血氧": "最低血氧",
            "odi": "ODI",
            "bmi": "BMI",
            "age": "age",
            "sex": "sex",
            "note": "note",
            "備註": "備註",
        }

        lines = []

        for key, value in record.items():
            if value is None:
                continue

            text = str(value).strip()

            if not text:
                continue

            normalized_key = str(key).strip().lower()
            output_key = aliases.get(
                normalized_key,
                str(key).strip(),
            )

            if output_key in {
                "minimum SpO2",
                "最低血氧",
            } and not text.endswith("%"):
                text += "%"

            lines.append(f"{output_key}: {text}")

        return "\n".join(lines)

    def _read_document(
        self,
        source_path: Path,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Read plain-text files directly with explicit encoding handling.
        Other formats are delegated to DocumentReader.

        Some readers accept a filesystem path, while Streamlit-style
        readers expect an uploaded-file-like object that implements
        read(), seek(), and name.
        """
        if source_path.suffix.lower() in {".txt", ".csv"}:
            return self._read_plain_text(source_path)

        reader = self.document_reader
        candidates = (
            "read",
            "read_document",
            "extract_text",
            "load",
        )

        errors: list[str] = []

        for method_name in candidates:
            method = getattr(reader, method_name, None)
            if not callable(method):
                continue

            attempts = (
                ("path_string", lambda: str(source_path)),
                ("path_object", lambda: source_path),
                (
                    "binary_file",
                    lambda: source_path.open("rb"),
                ),
                (
                    "named_bytes_buffer",
                    lambda: self._make_uploaded_file_like(
                        source_path
                    ),
                ),
            )

            for input_kind, input_factory in attempts:
                input_value = None

                try:
                    input_value = input_factory()
                    result = method(input_value)

                    text, normalized = (
                        self._normalize_reader_result(
                            result,
                            source_path,
                            method_name,
                        )
                    )

                    if not text.strip():
                        raise ValueError(
                            "DocumentReader 沒有擷取到文字。"
                        )

                    normalized["reader_input_kind"] = (
                        input_kind
                    )
                    return text, normalized

                except Exception as error:
                    errors.append(
                        f"{method_name}/{input_kind}: "
                        f"{type(error).__name__}: {error}"
                    )

                finally:
                    if (
                        input_value is not None
                        and hasattr(input_value, "close")
                        and input_kind == "binary_file"
                    ):
                        try:
                            input_value.close()
                        except Exception:
                            pass

        detail = "\n- ".join(errors[-12:])
        raise RuntimeError(
            "無法呼叫 DocumentReader，或 Reader 未回傳有效文字。"
            + (f"\n- {detail}" if detail else "")
        )

    def _read_plain_text(
        self,
        source_path: Path,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Decode TXT/CSV deterministically.

        UTF-8 is attempted first because it is the project output format.
        Traditional-Chinese legacy encodings are used as fallbacks.
        """
        raw = source_path.read_bytes()

        if not raw:
            raise ValueError(
                f"輸入文字檔是空的：{source_path}"
            )

        attempts = (
            "utf-8-sig",
            "utf-8",
            "cp950",
            "big5",
        )
        decode_errors: list[str] = []

        for encoding in attempts:
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError as error:
                decode_errors.append(
                    f"{encoding}: {error}"
                )
                continue

            if not text.strip():
                continue

            return text, {
                "filename": source_path.name,
                "extension": source_path.suffix.lower(),
                "text": text,
                "tables": [],
                "metadata": {
                    "encoding": encoding,
                    "byte_count": len(raw),
                    "character_count": len(text),
                },
                "warnings": [],
                "source_filename": source_path.name,
                "source_path": str(source_path),
                "reader_method": "direct_text_decode",
                "reader_input_kind": "raw_bytes",
            }

        raise UnicodeError(
            "無法解碼文字檔。嘗試結果："
            + " | ".join(decode_errors)
        )

    def _make_uploaded_file_like(
        self,
        source_path: Path,
    ) -> Any:
        """
        Create a Streamlit UploadedFile-like in-memory object.
        """
        from io import BytesIO

        class NamedBytesIO(BytesIO):
            def __init__(
                self,
                data: bytes,
                name: str,
            ) -> None:
                super().__init__(data)
                self.name = name
                self.type = (
                    "text/plain"
                    if source_path.suffix.lower() == ".txt"
                    else "application/octet-stream"
                )
                self.size = len(data)

        return NamedBytesIO(
            source_path.read_bytes(),
            source_path.name,
        )

    def _normalize_reader_result(
        self,
        result: Any,
        source_path: Path,
        method_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        if isinstance(result, str):
            return result, {
                "text": result,
                "source_filename": source_path.name,
                "source_path": str(source_path),
                "reader_method": method_name,
            }

        if isinstance(result, dict):
            normalized = dict(result)
        elif hasattr(result, "__dict__"):
            normalized = dict(vars(result))
        else:
            raise TypeError(
                "DocumentReader 回傳值必須是字串、dict，"
                "或具有屬性的結果物件；"
                f"目前為 {type(result).__name__}。"
            )

        text = self._find_text_content(normalized)

        normalized.setdefault(
            "source_filename",
            source_path.name,
        )
        normalized.setdefault(
            "source_path",
            str(source_path),
        )
        normalized.setdefault(
            "reader_method",
            method_name,
        )

        if not text.strip():
            warning_text = normalized.get("warnings")
            available_keys = ", ".join(
                str(key) for key in normalized.keys()
            ) or "(無)"
            raise ValueError(
                "DocumentReader 呼叫完成，但沒有有效文字。"
                f"頂層欄位：{available_keys}；"
                f"Reader warnings：{warning_text}"
            )

        normalized["text"] = text
        return text, normalized

    def _find_text_content(
        self,
        value: Any,
        *,
        depth: int = 0,
    ) -> str:
        """
        Locate document body only from known content fields.

        Do not scan arbitrary strings because filenames, MIME types,
        status messages, and warnings are not report content.
        """
        if depth > 8:
            return ""

        if isinstance(value, str):
            return value if value.strip() else ""

        if isinstance(value, dict):
            preferred_keys = (
                "text",
                "content",
                "raw_text",
                "extracted_text",
                "document_text",
                "full_text",
                "plain_text",
                "body",
            )

            for key in preferred_keys:
                if key not in value:
                    continue

                text = self._find_text_content(
                    value[key],
                    depth=depth + 1,
                )
                if text.strip():
                    return text

            nested_keys = (
                "data",
                "result",
                "document",
                "payload",
                "output",
                "reader_result",
            )

            for key in nested_keys:
                nested = value.get(key)

                if not isinstance(
                    nested,
                    (dict, list, tuple),
                ):
                    continue

                text = self._find_text_content(
                    nested,
                    depth=depth + 1,
                )
                if text.strip():
                    return text

            return ""

        if isinstance(value, (list, tuple)):
            parts = []

            for item in value:
                if not isinstance(
                    item,
                    (str, dict, list, tuple),
                ):
                    continue

                text = self._find_text_content(
                    item,
                    depth=depth + 1,
                )

                if text.strip():
                    parts.append(text.strip())

            return "\n".join(parts)

        if hasattr(value, "__dict__"):
            return self._find_text_content(
                vars(value),
                depth=depth + 1,
            )

        return ""

    def _normalize_patient_id(
        self,
        parsed_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Correct common OCR confusion only for the strict pattern:
        P + exactly three numeric-like characters.

        Examples:
        POOL -> P001
        POO1 -> P001

        Values outside this narrow pattern are left unchanged to avoid
        corrupting legitimate identifiers.
        """
        data = parsed_result.get("data")

        if not isinstance(data, dict):
            return parsed_result

        original = data.get("patient_id")

        if not isinstance(original, str):
            return parsed_result

        candidate = original.strip().upper()

        if not re.fullmatch(r"P[0-9OIL]{3}", candidate):
            return parsed_result

        normalized = (
            "P"
            + candidate[1:]
            .replace("O", "0")
            .replace("I", "1")
            .replace("L", "1")
        )

        if not re.fullmatch(r"P\d{3}", normalized):
            return parsed_result

        if normalized == candidate:
            return parsed_result

        data["patient_id"] = normalized

        fields = parsed_result.get("fields")
        if isinstance(fields, dict):
            patient_field = fields.get("patient_id")

            if isinstance(patient_field, dict):
                patient_field["value"] = normalized
                patient_field["normalized_value"] = normalized

                warnings = patient_field.setdefault(
                    "warnings",
                    [],
                )
                warning_message = (
                    "OCR patient_id normalized: "
                    f"{candidate} -> {normalized}"
                )

                if warning_message not in warnings:
                    warnings.append(warning_message)

        warnings = parsed_result.setdefault(
            "warnings",
            [],
        )
        warning_message = (
            "OCR patient_id normalized: "
            f"{candidate} -> {normalized}"
        )

        if warning_message not in warnings:
            warnings.append(warning_message)

        return parsed_result

    def _parse_text(
        self,
        raw_text: str,
        *,
        source_path: Path,
    ) -> Dict[str, Any]:
        parser = self.clinical_parser

        parse_method = getattr(parser, "parse", None)
        if not callable(parse_method):
            raise AttributeError(
                "ClinicalParser 缺少 parse() 方法。"
            )

        parsed_result = parse_method(raw_text)

        if not isinstance(parsed_result, dict):
            raise TypeError(
                "ClinicalParser.parse() 必須回傳 dict。"
            )

        parsed_result = self._normalize_patient_id(
            parsed_result
        )

        if not parsed_result.get("source_filename"):
            parsed_result["source_filename"] = (
                source_path.name
            )

        if not parsed_result.get("source_path"):
            parsed_result["source_path"] = (
                str(source_path)
            )

        return parsed_result

    def _inject_patient_id(
        self,
        parsed_result: Dict[str, Any],
        patient_id: str,
    ) -> None:
        data = parsed_result.get("data")
        if not isinstance(data, dict):
            data = {}
            parsed_result["data"] = data

        if not data.get("patient_id"):
            data["patient_id"] = patient_id

    def _build_output_paths(
        self,
        output_dir: Path,
        base_name: str,
    ) -> Dict[str, Path]:
        return {
            "report_json": (
                output_dir / f"{base_name}_report.json"
            ),
            "report_text": (
                output_dir / f"{base_name}_report.txt"
            ),
            "reader_json": (
                output_dir / f"{base_name}_reader.json"
            ),
            "parsed_json": (
                output_dir / f"{base_name}_parsed.json"
            ),
            "rules_json": (
                output_dir / f"{base_name}_rules.json"
            ),
            "recommendations_json": (
                output_dir
                / f"{base_name}_recommendations.json"
            ),
            "manifest_json": (
                output_dir / f"{base_name}_manifest.json"
            ),
        }

    def _save_json(
        self,
        value: Any,
        output_path: Path,
    ) -> None:
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self._make_json_compatible(value),
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

    def _make_json_compatible(
        self,
        value: Any,
    ) -> Any:
        if value is None or isinstance(
            value,
            (bool, int, str),
        ):
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
                return self._make_json_compatible(
                    value.item()
                )
            except Exception:
                pass

        return str(value)

    def _safe_stem(
        self,
        value: str,
    ) -> str:
        safe = "".join(
            character
            if character.isalnum()
            or character in {"-", "_"}
            else "_"
            for character in value
        )
        return safe.strip("_") or "sleep_report"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "執行 Sleep Digital Twin 完整臨床流程，"
            "並輸出 JSON 與文字報告。"
        )
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help=(
            "一個或多個睡眠檢查報告路徑；"
            "也可傳入資料夾以批次處理其中的支援檔案。"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="傳入資料夾時，遞迴搜尋子資料夾。",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output",
        help="輸出資料夾，預設為 output。",
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help=(
            "當原始報告沒有患者編號時，"
            "使用此值補入 patient_id。"
        ),
    )
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help=(
            "只輸出最終報告與 manifest，"
            "不儲存各階段 JSON。"
        ),
    )

    return parser



SUPPORTED_EXTENSIONS = {
    ".txt",
    ".csv",
    ".pdf",
    ".docx",
    ".xlsx",
    ".xls",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


def collect_input_files(
    input_values: list[str],
    *,
    recursive: bool = False,
) -> list[Path]:
    """
    Collect report files safely.

    Directory mode ignores known project/support files and obvious
    negative-test fixtures. If an empty `input_reports` directory is
    supplied, valid clinical test files are discovered from its parent
    automatically, so the user does not need to move files manually.
    """
    collected: list[Path] = []
    seen: set[Path] = set()

    excluded_names = {
        "requirements.txt",
        "readme.txt",
        "license.txt",
        "test_empty.txt",
        "test_large.txt",
    }
    excluded_prefixes = (
        "test_broken",
    )

    def is_candidate(path: Path) -> bool:
        name_lower = path.name.lower()

        if not path.is_file():
            return False

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False

        if name_lower in excluded_names:
            return False

        if name_lower.startswith(excluded_prefixes):
            return False

        try:
            if path.stat().st_size == 0:
                return False
        except OSError:
            return False

        return True

    for raw_value in input_values:
        path = Path(raw_value).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"找不到輸入路徑：{path}"
            )

        if path.is_file():
            candidates = (
                [path] if is_candidate(path) else []
            )
        else:
            pattern = "**/*" if recursive else "*"
            candidates = [
                candidate
                for candidate in path.glob(pattern)
                if is_candidate(candidate)
            ]

            # Convenience fallback:
            # an empty input_reports folder uses valid clinical files
            # already present in the project root.
            if (
                not candidates
                and path.name.lower() == "input_reports"
            ):
                parent_pattern = (
                    "**/*" if recursive else "*"
                )
                candidates = [
                    candidate
                    for candidate in path.parent.glob(
                        parent_pattern
                    )
                    if is_candidate(candidate)
                    and (
                        candidate.stem.lower().startswith(
                            "test_clinical"
                        )
                        or candidate.stem.lower().startswith(
                            "test_scanned_clinical"
                        )
                        or candidate.stem.lower().startswith(
                            "clinical"
                        )
                        or candidate.stem.lower().startswith(
                            "patient"
                        )
                        or candidate.stem.lower().startswith(
                            "report"
                        )
                    )
                ]

                if candidates:
                    print(
                        "[INFO] input_reports 目前沒有檔案，"
                        "已自動改用專案根目錄中的臨床測試檔。"
                    )

        for candidate in sorted(candidates):
            resolved = candidate.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)
            collected.append(resolved)

    if not collected:
        raise ValueError(
            "沒有找到可處理的報告檔案。"
        )

    return collected


def save_batch_summary(
    batch_records: list[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_at = datetime.now().isoformat(
        timespec="seconds"
    )
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    output_path = (
        output_dir
        / f"batch_evaluation_{timestamp}.json"
    )

    successful = [
        record
        for record in batch_records
        if record.get("status") == "success"
    ]
    failed = [
        record
        for record in batch_records
        if record.get("status") == "failed"
    ]

    payload = {
        "pipeline": "SleepTwinPipeline",
        "pipeline_version": (
            SleepTwinPipeline.PIPELINE_VERSION
        ),
        "completed_at": completed_at,
        "total_files": len(batch_records),
        "success_count": len(successful),
        "failure_count": len(failed),
        "results": batch_records,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    return output_path

def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        input_files = collect_input_files(
            args.input_files,
            recursive=args.recursive,
        )
    except Exception as error:
        print(
            f"[ERROR] 無法建立批次清單：{error}",
            file=sys.stderr,
        )
        return 1

    pipeline = SleepTwinPipeline(
        output_dir=args.output_dir,
    )
    batch_records: list[Dict[str, Any]] = []

    print("=" * 72)
    print(
        "Sleep Digital Twin 批次評估開始"
    )
    print("=" * 72)
    print(f"待處理檔案數：{len(input_files)}")
    print("")

    for index, input_file in enumerate(
        input_files,
        start=1,
    ):
        print(
            f"[{index}/{len(input_files)}] "
            f"{input_file.name}"
        )

        try:
            result = pipeline.run(
                input_file,
                patient_id=args.patient_id,
                save_intermediate=(
                    not args.no_intermediate
                ),
            )

            patient_results = (
                result.get("results", [])
                if result.get("multi_patient")
                else [result]
            )

            for patient_result in patient_results:
                manifest = patient_result["manifest"]
                summary = manifest.get("summary", {})
                report = patient_result.get(
                    "report",
                    {},
                )
                parsed_result = patient_result.get(
                    "parsed_result",
                    {},
                )
                parser_metadata = parsed_result.get(
                    "metadata",
                    {},
                )

                batch_records.append(
                    {
                        "status": "success",
                        "input_file": str(input_file),
                        "source_row_index": (
                            manifest.get(
                                "source_row_index"
                            )
                        ),
                        "patient_id": summary.get(
                            "patient_id"
                        ),
                        "highest_priority": summary.get(
                            "highest_priority"
                        ),
                        "primary_recommendation": (
                            summary.get(
                                "primary_recommendation"
                            )
                        ),
                        "warning_count": summary.get(
                            "warning_count",
                            0,
                        ),
                        "parsed_field_count": (
                            parser_metadata.get(
                                "parsed_field_count"
                            )
                        ),
                        "missing_field_count": (
                            parser_metadata.get(
                                "missing_field_count"
                            )
                        ),
                        "overall_confidence": (
                            parser_metadata.get(
                                "overall_confidence"
                            )
                        ),
                        "ahi": (
                            parsed_result.get(
                                "data",
                                {},
                            ).get("ahi")
                        ),
                        "minimum_spo2": (
                            parsed_result.get(
                                "data",
                                {},
                            ).get("minimum_spo2")
                        ),
                        "report_id": report.get(
                            "report_id"
                        ),
                        "outputs": manifest.get(
                            "outputs",
                            {},
                        ),
                    }
                )

                row_label = (
                    f"row={manifest.get('source_row_index')}, "
                    if manifest.get(
                        "source_row_index"
                    ) is not None
                    else ""
                )
                print(
                    "  成功："
                    f"{row_label}"
                    f"patient="
                    f"{summary.get('patient_id') or '未知'}, "
                    f"priority="
                    f"{summary.get('highest_priority') or '無'}, "
                    f"recommendation="
                    f"{summary.get('primary_recommendation') or '無'}"
                )

        except Exception as error:
            batch_records.append(
                {
                    "status": "failed",
                    "input_file": str(input_file),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(f"  失敗：{error}")

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    try:
        batch_summary_path = save_batch_summary(
            batch_records,
            output_dir,
        )
    except Exception as error:
        print(
            f"[ERROR] 無法儲存批次摘要：{error}",
            file=sys.stderr,
        )
        return 1

    success_count = sum(
        record.get("status") == "success"
        for record in batch_records
    )
    failure_count = (
        len(batch_records) - success_count
    )

    print("")
    print("=" * 72)
    print("Sleep Digital Twin 批次評估完成")
    print("=" * 72)
    print(f"總檔案數：{len(batch_records)}")
    print(f"成功：{success_count}")
    print(f"失敗：{failure_count}")
    print(f"批次摘要：{batch_summary_path}")

    return 0 if failure_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())