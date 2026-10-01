from __future__ import annotations

from clinical_modules.base_parser import BaseClinicalParser
from clinical_modules.clinical_report_parser import ClinicalReportParser
from clinical_modules.dise_parser import DISEParser
from clinical_modules.ent_parser import ENTParser
from clinical_modules.generic_parser import GenericClinicalParser
from clinical_modules.imaging_parser import ImagingParser
from clinical_modules.pap_parser import PAPParser
from clinical_modules.parser_schemas import MODULE_SPECS
from clinical_modules.wearable_sensor_parser import WearableSensorParser


PARSER_REGISTRY: dict[str, BaseClinicalParser] = {
    "PAP": PAPParser(),
    "ENT": ENTParser(),
    "DISE": DISEParser(),
    "IMAGING": ImagingParser(),
    "OTHER": ClinicalReportParser(),
    "DEVICE_DATA": WearableSensorParser(),
}

for _module_id in MODULE_SPECS:
    PARSER_REGISTRY.setdefault(
        _module_id,
        GenericClinicalParser(_module_id),
    )


def get_parser(module_id: str) -> BaseClinicalParser:
    normalized_module_id = str(module_id).strip().upper()
    parser = PARSER_REGISTRY.get(normalized_module_id)
    if parser is None:
        available_modules = ", ".join(sorted(PARSER_REGISTRY))
        raise KeyError(
            "尚未建立此臨床資料類型的解析器："
            f"{normalized_module_id}。目前可用：{available_modules}"
        )
    return parser


def has_parser(module_id: str) -> bool:
    return str(module_id).strip().upper() in PARSER_REGISTRY


def available_parsers() -> dict[str, str]:
    return {
        module_id: parser.display_name
        for module_id, parser in sorted(PARSER_REGISTRY.items())
    }
