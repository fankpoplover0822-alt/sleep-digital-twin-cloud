from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from clinical_modules.base_parser import (
    BaseClinicalParser,
    ParsedClinicalData,
)
from clinical_modules.document_reader import DocumentReader


class PAPParser(BaseClinicalParser):
    """
    PAP 治療資料解析器。

    Version 1.0 支援：
    - Excel：.xlsx、.xls
    - CSV：.csv
    - 純文字：.txt

    目前會接受但尚未自動解析：
    - PDF
    - Word
    - JPG／JPEG／PNG

    標準輸出欄位：
    - therapy_type
    - pressure_setting
    - residual_ahi
    - mask_leak
    - treatment_spo2
    - adherence_hours_per_night
    - adherence_percentage
    - tolerance
    """

    module_id = "PAP"
    display_name = "PAP 治療資料"

    supported_extensions = {
        ".pdf",
        ".xls",
        ".xlsx",
        ".csv",
        ".txt",
        ".doc",
        ".docx",
        ".jpg",
        ".jpeg",
        ".png",
    }

    automatically_parsed_extensions = {
        ".pdf",
        ".xls",
        ".xlsx",
        ".csv",
        ".txt",
        ".docx",
        ".jpg",
        ".jpeg",
        ".png",
    }

    STANDARD_FIELDS = (
        "therapy_type",
        "pressure_setting",
        "residual_ahi",
        "mask_leak",
        "treatment_spo2",
        "adherence_hours_per_night",
        "adherence_percentage",
        "tolerance",
    )

    FIELD_ALIASES = {
        "therapy_type": {
            "therapytype",
            "therapy",
            "paptype",
            "devicetype",
            "modality",
            "treatmenttype",
            "papmode",
            "mode",
            "治療模式",
            "治療類型",
            "裝置類型",
        },
        "pressure_setting": {
            "pressuresetting",
            "pressure",
            "cpappressure",
            "papressure",
            "papsetting",
            "setpressure",
            "therapypressure",
            "pressurecmh2o",
            "pressurecmh20",
            "壓力",
            "治療壓力",
            "設定壓力",
        },
        "residual_ahi": {
            "residualahi",
            "treatedahi",
            "therapyahi",
            "papahi",
            "ahi",
            "residualapneahypopneaindex",
            "殘餘ahi",
            "治療後ahi",
            "殘存ahi",
        },
        "mask_leak": {
            "maskleak",
            "leak",
            "leakrate",
            "medianleak",
            "95leak",
            "95thpercentileleak",
            "unintentionalleak",
            "漏氣",
            "面罩漏氣",
            "漏氣量",
        },
        "treatment_spo2": {
            "treatmentspo2",
            "therapyspo2",
            "treatedspo2",
            "spo2",
            "meanspo2",
            "averagespo2",
            "oxygensaturation",
            "治療後spo2",
            "血氧",
            "平均血氧",
        },
        "adherence_hours_per_night": {
            "adherencehourspernight",
            "usagehourspernight",
            "averageusage",
            "avgusage",
            "dailyusage",
            "usage",
            "hourspernight",
            "averagehourspernight",
            "compliancehours",
            "使用時數",
            "每晚使用時數",
            "平均使用時數",
            "配戴時數",
        },
        "adherence_percentage": {
            "adherencepercentage",
            "compliancepercentage",
            "compliance",
            "usagepercentage",
            "percentdaysused",
            "daysused4hours",
            "daysusedover4hours",
            "使用率",
            "依從率",
            "順從率",
            "配戴率",
        },
        "tolerance": {
            "tolerance",
            "paptolerance",
            "masktolerance",
            "comfort",
            "acceptance",
            "耐受度",
            "耐受性",
            "接受度",
            "舒適度",
        },
    }

    def parse(
        self,
        uploaded_files: list[BinaryIO],
    ) -> ParsedClinicalData:
        """
        解析 PAP 補充資料並回傳標準化結果。

        多檔案處理原則：
        - 依上傳順序解析。
        - 後面的有效值可覆蓋前面的值。
        - 若同一欄位出現不同值，會加入 warning。
        """
        self.validate_files(uploaded_files)

        parsed_data = self._empty_result()
        document_reader = DocumentReader()
        source_files: list[str] = []
        warnings: list[str] = []
        parsed_file_count = 0
        unsupported_extraction_files: list[str] = []
        field_sources: dict[str, str] = {}

        for uploaded_file in uploaded_files:
            filename = str(
                getattr(
                    uploaded_file,
                    "name",
                    "unknown_file",
                )
            )
            source_files.append(filename)

            extension = Path(filename).suffix.lower()

            if extension not in self.automatically_parsed_extensions:
                unsupported_extraction_files.append(filename)
                continue

            try:
                if extension in {".pdf", ".docx", ".jpg", ".jpeg", ".png"}:
                    document_result = document_reader.read(uploaded_file)
                    warnings.extend(
                        str(item)
                        for item in document_result.get("warnings", [])
                        if item
                    )
                    extracted_text = str(document_result.get("text") or "")
                    file_result = (
                        self._parse_text(extracted_text.encode("utf-8"))
                        if extracted_text
                        else {}
                    )
                else:
                    file_result = self._parse_single_file(
                        uploaded_file=uploaded_file,
                        extension=extension,
                    )
            except Exception as exc:
                warnings.append(
                    f"無法解析 `{filename}`：{exc}"
                )
                continue

            parsed_file_count += 1

            if not file_result:
                warnings.append(
                    f"`{filename}` 未辨識到任何 PAP 標準欄位。"
                )
                continue

            for field_name, new_value in file_result.items():
                if (
                    field_name not in parsed_data
                    or new_value is None
                ):
                    continue

                old_value = parsed_data[field_name]

                if (
                    old_value is not None
                    and not self._values_equal(
                        old_value,
                        new_value,
                    )
                ):
                    old_source = field_sources.get(
                        field_name,
                        "較早上傳的檔案",
                    )
                    warnings.append(
                        f"欄位 `{field_name}` 出現不同值："
                        f"`{old_value}`（{old_source}）→ "
                        f"`{new_value}`（{filename}）。"
                        "已採用後者。"
                    )

                parsed_data[field_name] = new_value
                field_sources[field_name] = filename

        if unsupported_extraction_files:
            warnings.append(
                "下列檔案格式目前只接受上傳，"
                "尚未自動擷取內容："
                + "、".join(
                    unsupported_extraction_files
                )
                + "。目前請使用 Excel、CSV 或 TXT，"
                "或由醫師手動確認欄位。"
            )

        extracted_fields = [
            field_name
            for field_name, value
            in parsed_data.items()
            if value is not None
        ]

        missing_fields = [
            field_name
            for field_name, value
            in parsed_data.items()
            if value is None
        ]

        if not extracted_fields:
            warnings.append(
                "本次未自動擷取到 PAP 數值，"
                "畫面將使用預設值供人工確認。"
            )

        metadata = {
            "parser": self.__class__.__name__,
            "parser_version": "2.0.0",
            "automatic_extraction": bool(
                extracted_fields
            ),
            "file_count": len(uploaded_files),
            "parsed_file_count": parsed_file_count,
            "extracted_field_count": len(
                extracted_fields
            ),
            "extracted_fields": extracted_fields,
            "missing_fields": missing_fields,
            "field_sources": field_sources,
            "automatically_parsed_extensions": sorted(
                self.automatically_parsed_extensions
            ),
        }

        return ParsedClinicalData(
            module_id=self.module_id,
            data=parsed_data,
            source_files=source_files,
            warnings=warnings,
            metadata=metadata,
        )

    def _empty_result(self) -> dict[str, Any]:
        return {
            field_name: None
            for field_name in self.STANDARD_FIELDS
        }

    def _parse_single_file(
        self,
        uploaded_file: BinaryIO,
        extension: str,
    ) -> dict[str, Any]:
        """
        依副檔名解析單一檔案。
        """
        file_bytes = self._read_uploaded_bytes(
            uploaded_file
        )

        if extension in {".xlsx", ".xls"}:
            return self._parse_excel(file_bytes)

        if extension == ".csv":
            return self._parse_csv(file_bytes)

        if extension == ".txt":
            return self._parse_text(file_bytes)

        return {}

    @staticmethod
    def _read_uploaded_bytes(
        uploaded_file: BinaryIO,
    ) -> bytes:
        """
        安全取得 Streamlit UploadedFile 或一般 BinaryIO 的 bytes。
        """
        if hasattr(uploaded_file, "getvalue"):
            data = uploaded_file.getvalue()
            if isinstance(data, bytes):
                return data

        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        data = uploaded_file.read()

        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        if isinstance(data, str):
            return data.encode("utf-8")

        if not isinstance(data, bytes):
            raise TypeError(
                "無法取得上傳檔案的二進位內容。"
            )

        return data

    def _parse_excel(
        self,
        file_bytes: bytes,
    ) -> dict[str, Any]:
        """
        解析 Excel。

        支援常見形式：
        1. Field / Value 兩欄
        2. 第一列為欄名、第二列為數值
        3. 無標題的 key-value 表格
        """
        excel_file = pd.ExcelFile(
            io.BytesIO(file_bytes)
        )

        combined_result: dict[str, Any] = {}

        for sheet_name in excel_file.sheet_names:
            raw_frame = pd.read_excel(
                excel_file,
                sheet_name=sheet_name,
                header=None,
                dtype=object,
            )

            sheet_result = self._extract_from_dataframe(
                raw_frame
            )

            for field_name, value in (
                sheet_result.items()
            ):
                if value is not None:
                    combined_result[
                        field_name
                    ] = value

        return combined_result

    def _parse_csv(
        self,
        file_bytes: bytes,
    ) -> dict[str, Any]:
        """
        解析 CSV，自動嘗試常見編碼與分隔符號。
        """
        text = self._decode_text(file_bytes)

        try:
            dialect = csv.Sniffer().sniff(
                text[:4096],
                delimiters=",;\t|",
            )
            separator = dialect.delimiter
        except csv.Error:
            separator = ","

        frame = pd.read_csv(
            io.StringIO(text),
            sep=separator,
            header=None,
            dtype=object,
            engine="python",
        )

        return self._extract_from_dataframe(frame)

    def _parse_text(
        self,
        file_bytes: bytes,
    ) -> dict[str, Any]:
        """
        解析 TXT。

        支援：
        key: value
        key = value
        key<TAB>value
        key,value
        """
        text = self._decode_text(file_bytes)

        result: dict[str, Any] = {}

        lines = [
            raw_line.strip()
            for raw_line in text.splitlines()
            if raw_line.strip()
        ]

        for line_index, line in enumerate(lines):

            parts = re.split(
                r"\s*(?:[:：=,\t|])\s*",
                line,
                maxsplit=1,
            )

            if len(parts) != 2:
                # OCR 常把「欄位 值」辨識成以空白分隔的一行。
                parts = re.split(r"\s+", line, maxsplit=1)

            if len(parts) == 2:
                raw_key, raw_value = parts
            else:
                # PDF 表格文字層常把欄位和值拆成相鄰兩行。
                raw_key = line
                raw_value = (
                    lines[line_index + 1]
                    if line_index + 1 < len(lines)
                    else ""
                )

            standard_field = self._match_field(
                raw_key
            )

            if standard_field is None:
                continue

            normalized_value = (
                self._normalize_field_value(
                    standard_field,
                    raw_value,
                )
            )

            if normalized_value is not None:
                result[
                    standard_field
                ] = normalized_value

        return result

    def _extract_from_dataframe(
        self,
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        從任意表格中尋找欄位和值。
        """
        if frame.empty:
            return {}

        frame = frame.dropna(
            how="all",
        ).dropna(
            axis=1,
            how="all",
        )

        if frame.empty:
            return {}

        result: dict[str, Any] = {}

        # 形式一：逐列 key-value。
        for _, row in frame.iterrows():
            row_values = [
                value
                for value in row.tolist()
                if not self._is_missing(value)
            ]

            if len(row_values) < 2:
                continue

            standard_field = self._match_field(
                row_values[0]
            )

            if standard_field is None:
                continue

            normalized_value = (
                self._normalize_field_value(
                    standard_field,
                    row_values[1],
                )
            )

            if normalized_value is not None:
                result[
                    standard_field
                ] = normalized_value

        # 形式二：某一列是欄名，下一列是數值。
        row_count = len(frame.index)

        for row_position in range(
            max(row_count - 1, 0)
        ):
            header_row = frame.iloc[
                row_position
            ].tolist()

            value_row = frame.iloc[
                row_position + 1
            ].tolist()

            matched_header_count = 0
            header_row_result: dict[str, Any] = {}

            for column_position, raw_header in (
                enumerate(header_row)
            ):
                standard_field = self._match_field(
                    raw_header
                )

                if standard_field is None:
                    continue

                matched_header_count += 1

                if (
                    column_position
                    >= len(value_row)
                ):
                    continue

                raw_value = value_row[
                    column_position
                ]

                normalized_value = (
                    self._normalize_field_value(
                        standard_field,
                        raw_value,
                    )
                )

                if normalized_value is not None:
                    header_row_result[
                        standard_field
                    ] = normalized_value

            # 至少兩個欄名匹配時，才把它視為橫向表格。
            if matched_header_count >= 2:
                result.update(header_row_result)
                break

        # 形式三：掃描所有儲存格。
        # 若欄位名稱與數值被排成相鄰儲存格，也可辨識。
        rows, columns = frame.shape

        for row_index in range(rows):
            for column_index in range(columns):
                raw_cell = frame.iat[
                    row_index,
                    column_index,
                ]

                standard_field = self._match_field(
                    raw_cell
                )

                if standard_field is None:
                    continue

                candidate_positions = [
                    (
                        row_index,
                        column_index + 1,
                    ),
                    (
                        row_index + 1,
                        column_index,
                    ),
                ]

                for (
                    candidate_row,
                    candidate_column,
                ) in candidate_positions:
                    if (
                        candidate_row >= rows
                        or candidate_column >= columns
                    ):
                        continue

                    raw_value = frame.iat[
                        candidate_row,
                        candidate_column,
                    ]

                    normalized_value = (
                        self._normalize_field_value(
                            standard_field,
                            raw_value,
                        )
                    )

                    if normalized_value is not None:
                        result.setdefault(
                            standard_field,
                            normalized_value,
                        )
                        break

        return result

    def _match_field(
        self,
        raw_name: Any,
    ) -> str | None:
        """
        將來源欄位名稱對應到標準欄位。
        """
        if self._is_missing(raw_name):
            return None

        normalized_name = self._normalize_label(
            raw_name
        )

        if not normalized_name:
            return None

        for (
            standard_field,
            aliases,
        ) in self.FIELD_ALIASES.items():
            standard_normalized = (
                self._normalize_label(
                    standard_field
                )
            )

            if (
                normalized_name
                == standard_normalized
            ):
                return standard_field

            for alias in aliases:
                normalized_alias = (
                    self._normalize_label(alias)
                )

                if (
                    normalized_name
                    == normalized_alias
                ):
                    return standard_field

        return None

    @staticmethod
    def _normalize_label(value: Any) -> str:
        """
        標準化欄位名稱，移除空白、符號與單位干擾。
        """
        text = str(value).strip().lower()

        text = text.replace("₂", "2")
        text = text.replace("％", "%")

        return re.sub(
            r"[^a-z0-9\u4e00-\u9fff]+",
            "",
            text,
        )

    def _normalize_field_value(
        self,
        field_name: str,
        raw_value: Any,
    ) -> Any:
        """
        依欄位型別正規化資料。
        """
        if self._is_missing(raw_value):
            return None

        if field_name == "therapy_type":
            return self._normalize_therapy_type(
                raw_value
            )

        if field_name == "tolerance":
            return self._normalize_tolerance(
                raw_value
            )

        if field_name == "pressure_setting":
            return self._normalize_pressure(
                raw_value
            )

        if field_name in {
            "residual_ahi",
            "mask_leak",
            "treatment_spo2",
            "adherence_hours_per_night",
            "adherence_percentage",
        }:
            return self._extract_number(
                raw_value
            )

        return raw_value

    @staticmethod
    def _normalize_therapy_type(
        raw_value: Any,
    ) -> str | None:
        text = str(raw_value).strip().upper()

        if not text:
            return None

        known_types = [
            "APAP",
            "CPAP",
            "BIPAP",
            "BPAP",
            "ASV",
        ]

        for therapy_type in known_types:
            if therapy_type in text:
                return therapy_type

        return text

    @staticmethod
    def _normalize_tolerance(
        raw_value: Any,
    ) -> str | None:
        text = str(raw_value).strip().lower()

        if not text:
            return None

        positive_terms = {
            "good",
            "well",
            "excellent",
            "tolerated",
            "良好",
            "佳",
            "可耐受",
        }

        fair_terms = {
            "fair",
            "moderate",
            "average",
            "普通",
            "尚可",
            "中等",
        }

        negative_terms = {
            "poor",
            "bad",
            "intolerant",
            "not tolerated",
            "差",
            "不耐受",
            "無法耐受",
        }

        if any(
            term in text
            for term in positive_terms
        ):
            return "good"

        if any(
            term in text
            for term in fair_terms
        ):
            return "fair"

        if any(
            term in text
            for term in negative_terms
        ):
            return "poor"

        return text

    def _normalize_pressure(
        self,
        raw_value: Any,
    ) -> float | str | None:
        """
        固定壓力回傳 float；壓力範圍保留為字串，例如 6-14。
        """
        if self._is_missing(raw_value):
            return None

        text = str(raw_value).strip()

        range_match = re.search(
            r"(-?\d+(?:\.\d+)?)"
            r"\s*(?:-|–|—|~|to|至)\s*"
            r"(-?\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )

        if range_match:
            lower_value = float(
                range_match.group(1)
            )
            upper_value = float(
                range_match.group(2)
            )

            return (
                f"{lower_value:g}-"
                f"{upper_value:g}"
            )

        return self._extract_number(text)

    @staticmethod
    def _extract_number(
        raw_value: Any,
    ) -> float | None:
        if isinstance(
            raw_value,
            (int, float),
        ):
            if pd.isna(raw_value):
                return None

            return float(raw_value)

        text = str(raw_value).strip()

        if not text:
            return None

        text = text.replace(",", "")
        text = text.replace("％", "%")

        match = re.search(
            r"-?\d+(?:\.\d+)?",
            text,
        )

        if match is None:
            return None

        return float(match.group(0))

    @staticmethod
    def _decode_text(
        file_bytes: bytes,
    ) -> str:
        """
        嘗試常見文字編碼。
        """
        encodings = (
            "utf-8-sig",
            "utf-8",
            "big5",
            "cp950",
            "latin-1",
        )

        for encoding in encodings:
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue

        raise UnicodeDecodeError(
            "unknown",
            file_bytes,
            0,
            len(file_bytes),
            "無法辨識文字編碼",
        )

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True

        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            return False

        return bool(missing)

    @staticmethod
    def _values_equal(
        first_value: Any,
        second_value: Any,
    ) -> bool:
        if (
            isinstance(first_value, (int, float))
            and isinstance(
                second_value,
                (int, float),
            )
        ):
            return abs(
                float(first_value)
                - float(second_value)
            ) < 1e-9

        return str(first_value) == str(
            second_value
        )
