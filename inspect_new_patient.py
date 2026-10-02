from pathlib import Path

from src.importers.new_patient_importer import (
    NewPatientImporter,
)


PROJECT_ROOT = Path(__file__).resolve().parent

PATIENT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "incoming"
    / "20201014T221256 - d25c6"
)

DEMOGRAPHICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "demographics"
    / "patients.xlsx"
)


def main() -> None:
    importer = NewPatientImporter(
        patient_folder=PATIENT_FOLDER,
        demographics_file=DEMOGRAPHICS_FILE,
    )

    result = importer.inspect()

    importer.print_summary(result)


if __name__ == "__main__":
    main()
