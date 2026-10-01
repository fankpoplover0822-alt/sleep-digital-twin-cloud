from __future__ import annotations

from clinical_modules.parser_schemas import get_module_spec
from clinical_modules.schema_parser import SchemaClinicalParser


class ClinicalReportParser(SchemaClinicalParser):
    """Generic clinical-note/report parser registered as OTHER."""

    def __init__(self) -> None:
        super().__init__(get_module_spec("OTHER"))
