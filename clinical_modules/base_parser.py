from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO


@dataclass
class ParsedClinicalData:
    """
    臨床資料解析器的統一輸出格式。
    """

    module_id: str
    data: dict[str, Any]
    source_files: list[str] = field(
        default_factory=list
    )
    warnings: list[str] = field(
        default_factory=list
    )
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class BaseClinicalParser(ABC):
    """
    所有臨床資料解析器的共同介面。
    """

    module_id: str = "UNKNOWN"
    display_name: str = "Unknown Clinical Module"

    supported_extensions: set[str] = set()

    def supports_filename(
        self,
        filename: str,
    ) -> bool:
        """
        判斷解析器是否支援該檔案副檔名。
        """
        suffix = Path(filename).suffix.lower()

        return suffix in self.supported_extensions

    def validate_files(
        self,
        uploaded_files: list[BinaryIO],
    ) -> None:
        """
        驗證是否有檔案，以及副檔名是否受支援。
        """
        if not uploaded_files:
            raise ValueError(
                "尚未提供任何補充臨床資料檔案。"
            )

        unsupported_files: list[str] = []

        for uploaded_file in uploaded_files:
            filename = getattr(
                uploaded_file,
                "name",
                "",
            )

            if not filename:
                unsupported_files.append(
                    "未命名檔案"
                )
                continue

            if not self.supports_filename(
                filename
            ):
                unsupported_files.append(
                    filename
                )

        if unsupported_files:
            supported_text = ", ".join(
                sorted(self.supported_extensions)
            )

            raise ValueError(
                "下列檔案格式不受此解析器支援："
                f"{', '.join(unsupported_files)}。"
                f"支援格式：{supported_text}"
            )

    @abstractmethod
    def parse(
        self,
        uploaded_files: list[BinaryIO],
    ) -> ParsedClinicalData:
        """
        解析檔案並回傳標準化資料。
        """
        raise NotImplementedError
    
