from __future__ import annotations

"""
Clinical update parser utilities.

This module provides a small compatibility layer for normalizing the output of
existing clinical parsers before the data is shown in Streamlit or merged into
ClinicalDecisionData.

It does not replace existing parsers such as pap_parser.py.  Instead, it gives
all parsers one consistent output contract.
"""

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import math


SUPPORTED_MODULES = {
    "PAP",
    "ENT",
    "DISE",
    "IMAGING",
    "HYPOXEMIA",
    "ABG",
    "PULMONARY",
    "CARDIAC",
    "ECG",
    "ECHOCARDIOGRAPHY",
    "COMORBIDITY",
    "MEDICATION",
    "PATIENT_PREFERENCE",
    "FOLLOW_UP",
    "ANTHROPOMETRY",
    "WEIGHT_MANAGEMENT",
    "LIFESTYLE",
    "SLEEP_QUESTIONNAIRE",
    "DENTAL_CRANIOFACIAL",
    "LABORATORY",
    "NEUROLOGICAL",
    "PSYCHIATRIC",
    "OXYGEN_THERAPY",
    "SURGERY_HISTORY",
    "DEVICE_DATA",
    "OTHER",
}


@dataclass(slots=True)
class ClinicalUpdateParseResult:
    """
    Unified parser result used by the clinical update workflow.

    Attributes
    ----------
    module_id:
        Clinical module identifier, for example ``PAP`` or ``ENT``.
    data:
        Extracted clinical fields.
    source_files:
        Original uploaded file names.
    warnings:
        Non-fatal parser warnings.
    metadata:
        Parser name, confidence, timestamps, or other trace information.
    """

    module_id: str
    data: dict[str, Any] = field(default_factory=dict)
    source_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return {
            "module_id": self.module_id,
            "data": make_json_safe(self.data),
            "source_files": list(self.source_files),
            "warnings": list(self.warnings),
            "metadata": make_json_safe(self.metadata),
        }


def normalize_module_id(module_id: Any) -> str:
    """Normalize and validate a clinical module identifier."""
    normalized = str(module_id or "").strip().upper()

    if not normalized:
        raise ValueError("module_id 不可為空白。")

    if normalized not in SUPPORTED_MODULES:
        raise ValueError(
            "不支援的 clinical module_id："
            f"{normalized}"
        )

    return normalized


def make_json_safe(value: Any) -> Any:
    """
    Convert nested values into strict JSON-compatible values.

    - NaN and Infinity become ``None``.
    - Path objects become strings.
    - tuples and sets become lists.
    - dataclasses become dictionaries.
    """

    if value is None:
        return None

    if is_dataclass(value):
        return make_json_safe(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    # Support NumPy scalar values without requiring NumPy as a dependency.
    if hasattr(value, "item") and callable(value.item):
        try:
            return make_json_safe(value.item())
        except Exception:
            pass

    if isinstance(value, (str, int, bool)):
        return value

    return str(value)


def _clean_mapping(
    data: Mapping[str, Any],
    *,
    drop_empty: bool,
) -> dict[str, Any]:
    """Normalize one parser data mapping."""
    cleaned: dict[str, Any] = {}

    for raw_key, raw_value in data.items():
        key = str(raw_key).strip()

        if not key:
            continue

        value = make_json_safe(raw_value)

        if drop_empty and (
            value is None
            or value == ""
            or value == []
            or value == {}
        ):
            continue

        cleaned[key] = value

    return cleaned


def _normalize_source_files(
    source_files: Iterable[Any] | None,
) -> list[str]:
    """Convert uploaded files, Paths, or file names to unique names."""
    if source_files is None:
        return []

    normalized: list[str] = []

    for source in source_files:
        if source is None:
            continue

        if hasattr(source, "name"):
            name = str(source.name)
        else:
            name = str(source)

        name = Path(name).name.strip()

        if name and name not in normalized:
            normalized.append(name)

    return normalized


def _normalize_warnings(
    warnings: Iterable[Any] | Any | None,
) -> list[str]:
    """Normalize warnings into a unique string list."""
    if warnings is None:
        return []

    if isinstance(warnings, str):
        warning_values = [warnings]
    elif isinstance(warnings, Iterable):
        warning_values = list(warnings)
    else:
        warning_values = [warnings]

    normalized: list[str] = []

    for warning in warning_values:
        text = str(warning).strip()
        if text and text not in normalized:
            normalized.append(text)

    return normalized


def normalize_parse_result(
    *,
    module_id: str,
    parser_result: Any,
    source_files: Iterable[Any] | None = None,
    parser_name: str | None = None,
    drop_empty: bool = True,
) -> ClinicalUpdateParseResult:
    """
    Normalize output from an existing parser.

    Accepted parser result forms
    ----------------------------
    1. ``ClinicalUpdateParseResult``
    2. object with ``data``, ``source_files``, ``warnings``, ``metadata``
    3. dictionary containing those same keys
    4. plain dictionary of extracted clinical fields

    This makes the function compatible with the current ``parser.parse(...)``
    pattern already used in ``app.py``.
    """
    normalized_module = normalize_module_id(module_id)

    if isinstance(
        parser_result,
        ClinicalUpdateParseResult,
    ):
        result = parser_result
        result.module_id = normalized_module
        result.data = _clean_mapping(
            result.data,
            drop_empty=drop_empty,
        )
        result.source_files = _normalize_source_files(
            result.source_files or source_files
        )
        result.warnings = _normalize_warnings(
            result.warnings
        )
        result.metadata = _clean_mapping(
            result.metadata,
            drop_empty=False,
        )
        return result

    parsed_data: Any = {}
    parsed_sources: Any = source_files
    parsed_warnings: Any = []
    parsed_metadata: Any = {}

    if isinstance(parser_result, Mapping):
        envelope_keys = {
            "module_id",
            "data",
            "source_files",
            "warnings",
            "metadata",
        }

        if envelope_keys.intersection(
            parser_result.keys()
        ):
            parsed_data = parser_result.get(
                "data",
                {},
            )
            parsed_sources = parser_result.get(
                "source_files",
                source_files,
            )
            parsed_warnings = parser_result.get(
                "warnings",
                [],
            )
            parsed_metadata = parser_result.get(
                "metadata",
                {},
            )
        else:
            parsed_data = parser_result

    else:
        parsed_data = getattr(
            parser_result,
            "data",
            {},
        )
        parsed_sources = getattr(
            parser_result,
            "source_files",
            source_files,
        )
        parsed_warnings = getattr(
            parser_result,
            "warnings",
            [],
        )
        parsed_metadata = getattr(
            parser_result,
            "metadata",
            {},
        )

    if not isinstance(parsed_data, Mapping):
        raise TypeError(
            "Parser 的 data 必須是 dict 或 Mapping。"
        )

    if not isinstance(parsed_metadata, Mapping):
        parsed_metadata = {
            "raw_metadata": make_json_safe(
                parsed_metadata
            )
        }

    metadata = _clean_mapping(
        parsed_metadata,
        drop_empty=False,
    )

    if parser_name:
        metadata.setdefault(
            "parser_name",
            parser_name,
        )

    metadata.setdefault(
        "normalized_by",
        "clinical_modules.update_parser",
    )

    return ClinicalUpdateParseResult(
        module_id=normalized_module,
        data=_clean_mapping(
            parsed_data,
            drop_empty=drop_empty,
        ),
        source_files=_normalize_source_files(
            parsed_sources
        ),
        warnings=_normalize_warnings(
            parsed_warnings
        ),
        metadata=metadata,
    )


def parse_clinical_update(
    *,
    module_id: str,
    parser: Any,
    uploaded_files: Iterable[Any],
    drop_empty: bool = True,
) -> ClinicalUpdateParseResult:
    """
    Execute an existing parser and normalize its output.

    Example
    -------
    result = parse_clinical_update(
        module_id="PAP",
        parser=get_parser("PAP"),
        uploaded_files=uploaded_clinical_files,
    )
    """
    if parser is None:
        raise ValueError("parser 不可為 None。")

    parse_method = getattr(
        parser,
        "parse",
        None,
    )

    if not callable(parse_method):
        raise TypeError(
            "parser 必須提供可呼叫的 parse() 方法。"
        )

    uploaded_files = list(
        uploaded_files or []
    )

    if not uploaded_files:
        raise ValueError(
            "至少需要一個補充臨床資料檔案。"
        )

    raw_result = parse_method(
        uploaded_files
    )

    parser_name = (
        getattr(parser, "display_name", None)
        or parser.__class__.__name__
    )

    return normalize_parse_result(
        module_id=module_id,
        parser_result=raw_result,
        source_files=uploaded_files,
        parser_name=str(parser_name),
        drop_empty=drop_empty,
    )


__all__ = [
    "ClinicalUpdateParseResult",
    "SUPPORTED_MODULES",
    "make_json_safe",
    "normalize_module_id",
    "normalize_parse_result",
    "parse_clinical_update",
]
