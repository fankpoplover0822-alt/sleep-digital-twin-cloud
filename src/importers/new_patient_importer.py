from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import pandas as pd


@dataclass
class NewPatientFiles:
    """
    新患者資料夾中找到的檔案與基本資料。
    """

    patient_id: str
    patient_folder: Path
    edf_file: Path
    stage_file: Path
    event_file: Path | None
    demographics_file: Path | None

    sex: Any = None
    age: float | None = None
    bmi: float | None = None
    demographics_found: bool = False
    demographics_sheet: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "patient_folder": str(
                self.patient_folder
            ),
            "edf_file": str(
                self.edf_file
            ),
            "stage_file": str(
                self.stage_file
            ),
            "event_file": (
                str(self.event_file)
                if self.event_file is not None
                else None
            ),
            "demographics_file": (
                str(self.demographics_file)
                if self.demographics_file
                is not None
                else None
            ),
            "sex": self.sex,
            "age": self.age,
            "BMI": self.bmi,
            "demographics_found": (
                self.demographics_found
            ),
            "demographics_sheet": (
                self.demographics_sheet
            ),
        }


class NewPatientImporter:
    """
    新患者匯入器第一階段。

    功能：
    1. 檢查 incoming 中的新患者資料夾。
    2. 尋找 EDF。
    3. 尋找 Stage。
    4. 尋找 Event Grid（可選）。
    5. 從基本資料 Excel 依 patient_id 找到
       sex、age、BMI。

    這一版先不執行 EDF、Stage、Event 時間對齊。
    """

    def __init__(
        self,
        patient_folder: str | Path,
        demographics_file: str | Path | None = None,
    ) -> None:
        self.patient_folder = Path(
            patient_folder
        ).resolve()

        self.demographics_file = (
            Path(demographics_file).resolve()
            if demographics_file is not None
            else None
        )

        self.patient_id = (
            self.patient_folder.name.strip()
        )

    def inspect(
        self,
    ) -> NewPatientFiles:
        """
        掃描新患者資料夾並讀取基本資料。
        """
        self._validate_patient_folder()

        edf_file = self._find_required_file(
            patterns=[
                "*_EDF.edf",
                "*EDF*.edf",
                "*.edf",
            ],
            file_description="EDF",
        )

        stage_file = self._find_required_file(
            patterns=[
                "*_stage.xls",
                "*_stage.xlsx",
                "*stage*.xls",
                "*stage*.xlsx",
            ],
            file_description="Stage",
        )

        event_file = self._find_optional_file(
            patterns=[
                "*_Event Grid.xls",
                "*_Event Grid.xlsx",
                "*Event Grid*.xls",
                "*Event Grid*.xlsx",
                "*event*.xls",
                "*event*.xlsx",
            ]
        )

        result = NewPatientFiles(
            patient_id=self.patient_id,
            patient_folder=(
                self.patient_folder
            ),
            edf_file=edf_file,
            stage_file=stage_file,
            event_file=event_file,
            demographics_file=(
                self.demographics_file
            ),
        )

        if (
            self.demographics_file
            is not None
        ):
            demographics = (
                self._find_demographics()
            )

            if demographics is not None:
                result.sex = demographics[
                    "sex"
                ]

                result.age = demographics[
                    "age"
                ]

                result.bmi = demographics[
                    "BMI"
                ]

                result.demographics_found = (
                    True
                )

                result.demographics_sheet = (
                    demographics[
                        "source_sheet"
                    ]
                )

        return result

    def _validate_patient_folder(
        self,
    ) -> None:
        if not self.patient_folder.exists():
            raise FileNotFoundError(
                "找不到新患者資料夾："
                f"{self.patient_folder}"
            )

        if not self.patient_folder.is_dir():
            raise NotADirectoryError(
                "新患者路徑不是資料夾："
                f"{self.patient_folder}"
            )

        if not self.patient_id:
            raise RuntimeError(
                "無法從資料夾名稱取得 "
                "patient_id。"
            )

    def _find_required_file(
        self,
        patterns: list[str],
        file_description: str,
    ) -> Path:
        matches = self._collect_matches(
            patterns
        )

        if not matches:
            raise FileNotFoundError(
                f"患者 {self.patient_id} "
                f"找不到 {file_description} 檔案。\n"
                f"資料夾：{self.patient_folder}"
            )

        if len(matches) > 1:
            latest_version = self._latest_versioned_file(matches)
            if latest_version is not None:
                return latest_version

            match_text = "\n".join(
                f"- {path.name}"
                for path in matches
            )

            raise RuntimeError(
                f"患者 {self.patient_id} 找到多個 "
                f"{file_description} 候選檔案，"
                "無法自動判斷：\n"
                f"{match_text}"
            )

        return matches[0]

    def _find_optional_file(
        self,
        patterns: list[str],
    ) -> Path | None:
        matches = self._collect_matches(
            patterns
        )

        if not matches:
            return None

        if len(matches) > 1:
            latest_version = self._latest_versioned_file(matches)
            if latest_version is not None:
                return latest_version

            match_text = "\n".join(
                f"- {path.name}"
                for path in matches
            )

            raise RuntimeError(
                f"患者 {self.patient_id} 找到多個 "
                "Event Grid 候選檔案，"
                "無法自動判斷：\n"
                f"{match_text}"
            )

        return matches[0]

    def _collect_matches(
        self,
        patterns: list[str],
    ) -> list[Path]:
        """
        使用多個 pattern 搜尋，但避免同一檔案重複。
        """
        found: dict[
            str,
            Path,
        ] = {}

        for pattern in patterns:
            for path in (
                self.patient_folder.glob(
                    pattern
                )
            ):
                if not path.is_file():
                    continue

                key = str(
                    path.resolve()
                ).lower()

                found[key] = path.resolve()

        return sorted(
            found.values(),
            key=lambda path: (
                path.name.lower()
            ),
        )

    @staticmethod
    def _latest_versioned_file(matches: list[Path]) -> Path | None:
        """Choose the newest complete-upload filename when legacy files coexist."""
        versioned: list[tuple[str, Path]] = []
        for path in matches:
            match = re.search(
                r"__upload_(\d{8}T\d{12})",
                path.stem,
                flags=re.IGNORECASE,
            )
            if match:
                versioned.append((match.group(1), path))
        if not versioned:
            return None
        return max(versioned, key=lambda item: item[0])[1]

    def _find_demographics(
        self,
    ) -> dict[str, Any] | None:
        if self.demographics_file is None:
            return None

        if not self.demographics_file.exists():
            raise FileNotFoundError(
                "找不到基本資料 Excel："
                f"{self.demographics_file}"
            )

        workbook = pd.ExcelFile(
            self.demographics_file
        )

        candidate_rows: list[
            dict[str, Any]
        ] = []

        for sheet_name in (
            workbook.sheet_names
        ):
            frame = pd.read_excel(
                workbook,
                sheet_name=sheet_name,
            )

            if frame.empty:
                continue

            frame.columns = [
                str(column).strip()
                for column in frame.columns
            ]

            column_lookup = {
                str(column)
                .strip()
                .lower(): column
                for column in frame.columns
            }

            id_column = (
                column_lookup.get("id")
                or column_lookup.get(
                    "patient_id"
                )
                or column_lookup.get(
                    "patient id"
                )
            )

            if id_column is None:
                continue

            normalized_ids = (
                frame[id_column]
                .astype(str)
                .str.strip()
            )

            matched = frame[
                normalized_ids
                == self.patient_id
            ].copy()

            if matched.empty:
                continue

            if len(matched) > 1:
                raise RuntimeError(
                    "基本資料 Excel 中同一個 "
                    f"ID 出現超過一次："
                    f"{self.patient_id}\n"
                    f"工作表：{sheet_name}"
                )

            row = matched.iloc[0]

            sex_column = (
                column_lookup.get("sex")
                or column_lookup.get(
                    "gender"
                )
                or column_lookup.get(
                    "性別"
                )
            )

            age_column = (
                column_lookup.get("age")
                or column_lookup.get(
                    "年齡"
                )
            )

            bmi_column = (
                column_lookup.get("bmi")
                or column_lookup.get(
                    "body mass index"
                )
            )

            candidate_rows.append(
                {
                    "sex": (
                        row[sex_column]
                        if sex_column
                        is not None
                        else None
                    ),
                    "age": self._to_optional_float(
                        row[age_column]
                        if age_column
                        is not None
                        else None
                    ),
                    "BMI": self._to_optional_float(
                        row[bmi_column]
                        if bmi_column
                        is not None
                        else None
                    ),
                    "source_sheet": (
                        sheet_name
                    ),
                }
            )

        if not candidate_rows:
            return None

        if len(candidate_rows) > 1:
            sheets = [
                row["source_sheet"]
                for row in candidate_rows
            ]

            raise RuntimeError(
                "基本資料 Excel 的多個工作表"
                "都找到相同患者 ID："
                f"{self.patient_id}\n"
                f"工作表：{sheets}"
            )

        return candidate_rows[0]

    @staticmethod
    def _to_optional_float(
        value: Any,
    ) -> float | None:
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except (
            TypeError,
            ValueError,
        ):
            pass

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def print_summary(
        result: NewPatientFiles,
    ) -> None:
        print("=" * 80)
        print("新患者檔案檢查結果")
        print("=" * 80)

        print(
            f"Patient ID："
            f"{result.patient_id}"
        )

        print(
            f"患者資料夾："
            f"{result.patient_folder}"
        )

        print(
            f"EDF："
            f"{result.edf_file.name}"
        )

        print(
            f"Stage："
            f"{result.stage_file.name}"
        )

        print(
            "Event Grid："
            + (
                result.event_file.name
                if result.event_file
                is not None
                else "未提供"
            )
        )

        print()
        print("基本資料：")

        if result.demographics_found:
            print(
                f"來源工作表："
                f"{result.demographics_sheet}"
            )

            print(
                f"sex：{result.sex}"
            )

            print(
                f"age：{result.age}"
            )

            print(
                f"BMI：{result.bmi}"
            )
        else:
            print(
                "找不到對應 patient_id。"
            )

        print("=" * 80)
