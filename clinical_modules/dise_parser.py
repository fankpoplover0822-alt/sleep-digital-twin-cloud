from clinical_modules.parser_schemas import get_module_spec
from clinical_modules.schema_parser import SchemaClinicalParser


class DISEParser(SchemaClinicalParser):
    def __init__(self) -> None:
        super().__init__(get_module_spec("DISE"))
