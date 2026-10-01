from pathlib import Path
from typing import Any, Dict

import pandas as pd


class DocumentReader:

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".csv",
        ".xlsx",
        ".xls",
        ".docx",
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
    }
    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

    SIGNATURE_EXTENSION_MAP = {
        "pdf": {".pdf"},
        "png": {".png"},
        "jpeg": {".jpg", ".jpeg"},
        "tiff": {".tif", ".tiff"},
        "bmp": {".bmp"},
        "zip_based": {".docx", ".xlsx"},
        "ole_compound": {".xls"},
    }

    """
    Generic document reader for clinical documents.
    """

    def read(self, uploaded_file) -> Dict[str, Any]:
        """
        Read a single uploaded file.
        """
        filename = getattr(
            uploaded_file,
            "name",
            "uploaded_file",
        )
        suffix = Path(filename).suffix.lower()

        result = {
            "filename": filename,
            "extension": suffix,
            "text": "",
            "tables": [],
            "metadata": {},
            "warnings": [],
        }

        # 先確認檔案內容能否正常讀取。
        try:
            file_bytes = self._get_file_bytes(uploaded_file)
        except Exception as exc:
            result["warnings"].append(
                f"無法讀取檔案內容："
                f"{type(exc).__name__}: {exc}"
            )
            return result

        result["metadata"]["file_size_bytes"] = len(file_bytes)
        result["metadata"]["file_size_mb"] = round(
            len(file_bytes) / (1024 * 1024),
            3,
        )

        # 阻擋完全沒有內容的空檔案。
        if not file_bytes:
            result["metadata"]["detected_signature"] = "empty"
            result["warnings"].append("檔案內容為空。")
            return result

        # 阻擋過大的檔案。
        if len(file_bytes) > self.MAX_FILE_SIZE_BYTES:
            max_size_mb = self.MAX_FILE_SIZE_BYTES // (1024 * 1024)
            result["warnings"].append(
                f"檔案大小超過限制，最大允許 {max_size_mb} MB。"
            )
            return result

        # 阻擋目前不支援的副檔名。
        if suffix not in self.SUPPORTED_EXTENSIONS:
            result["warnings"].append(
                f"目前不支援此檔案格式："
                f"{suffix or '無副檔名'}。"
            )
            return result

        # 偵測實際檔案簽章。
        detected_signature = self._detect_file_signature(file_bytes)
        result["metadata"]["detected_signature"] = detected_signature

        expected_extensions = self.SIGNATURE_EXTENSION_MAP.get(
            detected_signature
        )

        if (
            expected_extensions is not None
            and suffix not in expected_extensions
        ):
            expected_text = "、".join(sorted(expected_extensions))
            result["warnings"].append(
                f"檔案副檔名與內容格式可能不一致："
                f"目前副檔名為 {suffix or '無副檔名'}，"
                f"實際內容較像 {expected_text}。"
            )

        signature_required_extensions = {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
        }

        if (
            suffix in signature_required_extensions
            and detected_signature == "unknown"
        ):
            result["warnings"].append(
                f"檔案內容不像有效的 {suffix} 檔案，"
                "可能已損壞或副檔名不正確。"
            )

        if suffix == ".txt":
            result["text"] = self._read_txt(uploaded_file)

        elif suffix == ".csv":
            csv_result = self._read_csv(uploaded_file)
            result["text"] = csv_result["text"]
            result["tables"] = csv_result["tables"]
            result["warnings"].extend(csv_result["warnings"])

        elif suffix in {".xlsx", ".xls"}:
            excel_result = self._read_excel(uploaded_file)
            result["text"] = excel_result["text"]
            result["tables"] = excel_result["tables"]
            result["warnings"].extend(excel_result["warnings"])

        elif suffix == ".docx":
            docx_result = self._read_docx(uploaded_file)
            result["text"] = docx_result["text"]
            result["tables"] = docx_result["tables"]
            result["warnings"].extend(docx_result["warnings"])

        elif suffix == ".pdf":
            pdf_result = self._read_pdf(uploaded_file)
            result["text"] = pdf_result["text"]
            result["tables"] = pdf_result["tables"]
            result["metadata"].update(pdf_result["metadata"])
            result["warnings"].extend(pdf_result["warnings"])

        elif suffix in {
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
        }:
            image_result = self._read_image_ocr(uploaded_file)
            result["text"] = image_result["text"]
            result["tables"] = image_result["tables"]
            result["metadata"].update(image_result["metadata"])
            result["warnings"].extend(image_result["warnings"])

        return result

    def _detect_file_signature(self, file_bytes: bytes) -> str:
        """
        Detect common file formats from file signatures.

        Returns a descriptive format name rather than relying
        only on the filename extension.
        """
        if file_bytes.startswith(b"%PDF-"):
            return "pdf"

        if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"

        if file_bytes.startswith(b"\xff\xd8\xff"):
            return "jpeg"

        if file_bytes.startswith((b"II*\x00", b"MM\x00*")):
            return "tiff"

        if file_bytes.startswith(b"BM"):
            return "bmp"

        if file_bytes.startswith(b"PK\x03\x04"):
            return "zip_based"

        if file_bytes.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            return "ole_compound"

        return "unknown"
    

    def _get_file_bytes(self, uploaded_file) -> bytes:
        """
        Return file content as bytes.

        Supports:
        - Streamlit UploadedFile
        - Regular Python binary file objects
        """
        if hasattr(uploaded_file, "getvalue"):
            return uploaded_file.getvalue()

        uploaded_file.seek(0)
        return uploaded_file.read()

    def _read_txt(self, uploaded_file) -> str:
        """
        Read TXT content with common encodings.
        """
        file_bytes = self._get_file_bytes(uploaded_file)

        encodings = [
            "utf-8-sig",
            "utf-8",
            "big5",
            "cp950",
        ]

        for encoding in encodings:
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue

        return file_bytes.decode("utf-8", errors="replace")

    def _read_csv(self, uploaded_file) -> Dict[str, Any]:
        """
        Read CSV content and convert it into JSON-compatible records.
        """
        warnings = []
        file_bytes = self._get_file_bytes(uploaded_file)

        encodings = [
            "utf-8-sig",
            "utf-8",
            "big5",
            "cp950",
        ]

        dataframe = None
        successful_encoding = None

        for encoding in encodings:
            try:
                from io import BytesIO

                dataframe = pd.read_csv(
                    BytesIO(file_bytes),
                    encoding=encoding,
                )
                successful_encoding = encoding
                break

            except UnicodeDecodeError:
                continue

            except pd.errors.EmptyDataError:
                warnings.append("CSV 檔案沒有可讀取的資料。")
                return {
                    "text": "",
                    "tables": [],
                    "warnings": warnings,
                }

            except pd.errors.ParserError as exc:
                warnings.append(f"CSV 格式解析失敗：{exc}")
                return {
                    "text": "",
                    "tables": [],
                    "warnings": warnings,
                }

        if dataframe is None:
            warnings.append("無法判斷 CSV 檔案的文字編碼。")
            return {
                "text": "",
                "tables": [],
                "warnings": warnings,
            }

        dataframe = dataframe.where(
            pd.notna(dataframe),
            None,
        )

        records = dataframe.to_dict(orient="records")

        text = dataframe.to_csv(
            index=False,
            lineterminator="\n",
        )

        return {
            "text": text,
            "tables": [
                {
                    "sheet_name": "CSV",
                    "rows": records,
                    "row_count": len(dataframe),
                    "column_count": len(dataframe.columns),
                    "columns": [str(column) for column in dataframe.columns],
                    "encoding": successful_encoding,
                }
            ],
            "warnings": warnings,
        }

    def _read_excel(self, uploaded_file) -> Dict[str, Any]:
        """
        Read XLSX or XLS files and convert every worksheet
        into JSON-compatible records.
        """
        from io import BytesIO

        warnings = []
        tables = []
        text_parts = []

        file_bytes = self._get_file_bytes(uploaded_file)

        try:
            workbook = pd.ExcelFile(BytesIO(file_bytes))

            for sheet_name in workbook.sheet_names:
                dataframe = pd.read_excel(
                    workbook,
                    sheet_name=sheet_name,
                )

                dataframe.columns = [
                    str(column)
                    for column in dataframe.columns
                ]

                dataframe = dataframe.astype(object).where(
                    pd.notna(dataframe),
                    None,
                )

                records = dataframe.to_dict(orient="records")

                tables.append(
                    {
                        "sheet_name": str(sheet_name),
                        "rows": records,
                        "row_count": len(dataframe),
                        "column_count": len(dataframe.columns),
                        "columns": list(dataframe.columns),
                    }
                )

                text_parts.append(
                    f"工作表：{sheet_name}\n"
                    + dataframe.to_csv(
                        index=False,
                        lineterminator="\n",
                    )
                )

        except ValueError as exc:
            warnings.append(f"Excel 檔案解析失敗：{exc}")

        except Exception as exc:
            warnings.append(
                f"讀取 Excel 檔案時發生錯誤："
                f"{type(exc).__name__}: {exc}"
            )

        return {
            "text": "\n\n".join(text_parts),
            "tables": tables,
            "warnings": warnings,
        }
    
    def _read_docx(self, uploaded_file) -> Dict[str, Any]:
        """
        Read text and tables from a DOCX document.
        """
        from io import BytesIO

        warnings = []
        tables = []
        text_parts = []

        file_bytes = self._get_file_bytes(uploaded_file)

        try:
            from docx import Document
            document = Document(BytesIO(file_bytes))

            for paragraph in document.paragraphs:
                text = paragraph.text.strip()
                if text:
                    text_parts.append(text)

            for table_index, table in enumerate(document.tables, start=1):
                rows = []

                for row in table.rows:
                    values = [
                        cell.text.strip()
                        for cell in row.cells
                    ]
                    rows.append(values)

                tables.append(
                    {
                        "table_index": table_index,
                        "rows": rows,
                        "row_count": len(rows),
                        "column_count": (
                            max((len(row) for row in rows), default=0)
                        ),
                    }
                )

                text_parts.append(
                    f"表格 {table_index}：\n"
                    + "\n".join(
                        "\t".join(row)
                        for row in rows
                    )
                )

        except ModuleNotFoundError:
            # DOCX 本質上是 ZIP + XML。即使環境沒有 python-docx，
            # 仍可用標準函式庫擷取段落與表格文字，不讓上傳流程中斷。
            try:
                from zipfile import ZipFile
                from xml.etree import ElementTree

                with ZipFile(BytesIO(file_bytes)) as archive:
                    document_xml = archive.read("word/document.xml")

                root = ElementTree.fromstring(document_xml)
                namespace = {
                    "w": (
                        "http://schemas.openxmlformats.org/"
                        "wordprocessingml/2006/main"
                    )
                }
                for paragraph in root.findall(".//w:p", namespace):
                    text = "".join(
                        node.text or ""
                        for node in paragraph.findall(".//w:t", namespace)
                    ).strip()
                    if text:
                        text_parts.append(text)

                warnings.append(
                    "已使用內建 DOCX 文字讀取模式完成解析。"
                )
            except Exception as exc:
                warnings.append(
                    "無法讀取 DOCX 內容："
                    f"{type(exc).__name__}: {exc}"
                )

        except Exception as exc:
            warnings.append(
                "讀取 DOCX 檔案時發生錯誤："
                f"{type(exc).__name__}: {exc}"
            )

        return {
            "text": "\n\n".join(text_parts),
            "tables": tables,
            "warnings": warnings,
        }


    def _read_pdf(self, uploaded_file) -> Dict[str, Any]:
        """
        Read text from a PDF.

        If a page has no selectable text, automatically
        render the page and perform Traditional Chinese
        and English OCR.
        """
        from io import BytesIO

        import fitz
        import pytesseract
        from PIL import Image, ImageOps

        tesseract_path = Path(
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
        if tesseract_path.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

        warnings = []
        text_parts = []
        page_records = []
        metadata = {}

        file_bytes = self._get_file_bytes(uploaded_file)

        try:
            document = fitz.open(
                stream=file_bytes,
                filetype="pdf",
            )

            for page_index, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()
                extraction_method = "text"

                if not page_text:
                    extraction_method = "ocr"

                    try:
                        zoom = 2.5
                        matrix = fitz.Matrix(zoom, zoom)

                        pixmap = page.get_pixmap(
                            matrix=matrix,
                            alpha=False,
                        )

                        image = Image.open(
                            BytesIO(pixmap.tobytes("png"))
                        )

                        processed_image = ImageOps.exif_transpose(image)
                        processed_image = processed_image.convert("L")
                        processed_image = ImageOps.autocontrast(
                            processed_image
                        )

                        raw_page_text = pytesseract.image_to_string(
                            processed_image,
                            lang="chi_tra+eng",
                            config="--oem 3 --psm 6",
                        ).strip()

                        page_text = self._clean_ocr_text(raw_page_text)
                        if not page_text:
                            warnings.append(
                                f"第 {page_index} 頁沒有文字層，"
                                "OCR 也未辨識出文字。"
                            )

                    except pytesseract.TesseractNotFoundError:
                        warnings.append(
                            f"第 {page_index} 頁需要 OCR，"
                            "但找不到 Tesseract 執行程式。"
                        )

                    except Exception as exc:
                        warnings.append(
                            f"第 {page_index} 頁 OCR 發生錯誤："
                            f"{type(exc).__name__}: {exc}"
                        )

                page_records.append(
                    {
                        "page_number": page_index,
                        "text": page_text,
                        "character_count": len(page_text),
                        "extraction_method": extraction_method,
                    }
                )

                if page_text:
                    text_parts.append(
                        f"第 {page_index} 頁：\n{page_text}"
                    )

            metadata = {
                "page_count": document.page_count,
                "pdf_metadata": {
                    str(key): value
                    for key, value in document.metadata.items()
                    if value not in (None, "")
                },
                "pages": page_records,
                "ocr_language": "chi_tra+eng",
            }

            document.close()

        except Exception as exc:
            warnings.append(
                f"讀取 PDF 檔案時發生錯誤："
                f"{type(exc).__name__}: {exc}"
            )

        return {
            "text": "\n\n".join(text_parts),
            "tables": [],
            "metadata": metadata,
            "warnings": warnings,
        }


    def _read_image_ocr(self, uploaded_file) -> Dict[str, Any]:
        """
        Extract Traditional Chinese and English text from an image
        using Tesseract OCR.
        """
        from io import BytesIO

        import pytesseract
        from PIL import Image, ImageOps

        tesseract_path = Path(
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
        if tesseract_path.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

        warnings = []
        text = ""
        metadata = {}

        file_bytes = self._get_file_bytes(uploaded_file)

        try:
            with Image.open(BytesIO(file_bytes)) as image:
                image.load()

                metadata = {
                    "image_format": image.format,
                    "image_width": image.width,
                    "image_height": image.height,
                    "image_mode": image.mode,
                    "ocr_language": "chi_tra+eng",
                }

                processed_image = ImageOps.exif_transpose(image)
                processed_image = processed_image.convert("L")
                processed_image = ImageOps.autocontrast(processed_image)

                raw_text = pytesseract.image_to_string(
                    processed_image,
                    lang="chi_tra+eng",
                    config="--oem 3 --psm 6",
                ).strip()

                text = self._clean_ocr_text(raw_text)

                if not text:
                    warnings.append(
                        "OCR 未辨識出文字，可能是圖片解析度不足、"
                        "文字過小或版面較複雜。"
                    )

        except pytesseract.TesseractNotFoundError:
            warnings.append(
                "找不到 Tesseract 執行程式，請確認已安裝並加入 PATH。"
            )

        except Exception as exc:
            warnings.append(
                f"圖片 OCR 時發生錯誤："
                f"{type(exc).__name__}: {exc}"
            )

        return {
            "text": text,
            "tables": [],
            "metadata": metadata,
            "warnings": warnings,
        }

    def _clean_ocr_text(self, text: str) -> str:
        """
        Perform conservative cleanup on OCR output.

        This only normalizes whitespace and punctuation spacing.
        It does not guess or replace ambiguous letters and digits.
        """
        import re

        if not text:
            return ""

        cleaned_lines = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            # Remove spaces between adjacent Chinese characters.
            line = re.sub(
                r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])",
                "",
                line,
            )

            # Remove spaces before common punctuation.
            line = re.sub(
                r"\s+([，。；：！？、,%％])",
                r"\1",
                line,
            )

            # Remove spaces after opening brackets.
            line = re.sub(
                r"([（(\[])\s+",
                r"\1",
                line,
            )

            # Remove spaces before closing brackets.
            line = re.sub(
                r"\s+([）)\]])",
                r"\1",
                line,
            )

            # Normalize colon spacing.
            line = re.sub(
                r"\s*[:：]\s*",
                "：",
                line,
            )

            # Normalize slash spacing.
            line = re.sub(
                r"\s*/\s*",
                "/",
                line,
            )

            # Collapse remaining repeated spaces.
            line = re.sub(
                r"[ \t]+",
                " ",
                line,
            )

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)
