from clinical_modules.parser_schemas import get_module_spec
from clinical_modules.schema_parser import SchemaClinicalParser


class GenericClinicalParser(SchemaClinicalParser):
    def __init__(self, module_id: str) -> None:
        super().__init__(get_module_spec(module_id))
