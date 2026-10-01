from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT = PROJECT_ROOT / "build_treatment_refinement.py"
TEST_PATIENT = "__RULE_ENGINE_CROSS_TREATMENT_TEST__"
PATIENT_ROOT = PROJECT_ROOT / "data" / "inference" / TEST_PATIENT
STAGE_DIR = PATIENT_ROOT / "treatment_suitability"
REFINE_DIR = PATIENT_ROOT / "treatment_refinement"


def treatment_record(name: str, score: float, rank: int) -> dict:
    missing = [
        "耳鼻喉內視鏡",
        "睡眠內視鏡或動態阻塞評估",
        "鼻腔與上呼吸道影像",
    ]
    return {
        "treatment": name,
        "treatment_label": name,
        "rank": rank,
        "score": score,
        "supporting_factors": [],
        "limiting_factors": [],
        "required_confirmation": [],
        "candidate_procedures": [],
        "projected_score_changes": [],
        "confidence": {
            "available_evidence": [],
            "missing_evidence": missing,
            "available_count": 0,
            "total_count": len(missing),
            "completeness": 0.0,
            "confidence_level": "LOW",
        },
    }


def write_case(*, complete_concentric: bool) -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    REFINE_DIR.mkdir(parents=True, exist_ok=True)

    ranking = [
        treatment_record("CPAP", 80.0, 1),
        treatment_record("APAP", 70.0, 2),
        treatment_record("SURGERY", 25.0, 3),
        treatment_record("MAD", 40.0, 4),
        treatment_record("HGNS", 35.0, 5),
        treatment_record(
            "SLEEP_ARCHITECTURE_MODULATING_MEDICATION_REVIEW",
            0.0,
            6,
        ),
    ]

    stage1 = {
        "patient_id": TEST_PATIENT,
        "personalized_treatment_ranking": ranking,
        "clinical_prerequisites": [],
        "resolved_clinical_prerequisites": [],
    }

    pattern = (
        "complete_concentric_collapse"
        if complete_concentric
        else "anteroposterior"
    )

    clinical = {
        "patient_id": TEST_PATIENT,
        "data_version": 1,
        "ent": {
            "available": True,
            "findings": {
                "tonsil_hypertrophy": True,
                "nasal_septal_deviation": True,
                "inferior_turbinate_hypertrophy": True,
                "soft_palate_abnormality": True,
                "mallampati": 4,
                "nasal_obstruction": True,
            },
        },
        "dise": {
            "available": True,
            "findings": {
                "velum_collapse": "complete",
                "oropharyngeal_lateral_wall_collapse": "partial",
                "tongue_base_collapse": "complete",
                "epiglottis_collapse": "none",
                "collapse_pattern": pattern,
                "collapse_degree": "complete",
                "vote_classification": (
                    "V2C O1LAT T2AP E0"
                    if complete_concentric
                    else "V2AP O1LAT T2AP E0"
                ),
            },
        },
        "imaging": {
            "available": True,
            "findings": {
                "nasal_airway_narrowing": True,
                "retropalatal_airway_narrowing": True,
                "retroglossal_airway_narrowing": True,
                "craniofacial_abnormality": True,
                "other_anatomical_obstruction": (
                    "enlarged tongue base"
                ),
                "minimum_airway_area": 62.5,
            },
        },
        "pap": {"available": False, "findings": {}},
        "patient_preference": {
            "available": False,
            "findings": {},
        },
        "comorbidities": {
            "available": False,
            "findings": {},
        },
        "follow_up": {
            "available": False,
            "findings": {},
        },
    }

    (STAGE_DIR / "personalized_treatment_recommendation.json").write_text(
        json.dumps(stage1, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (REFINE_DIR / "clinical_decision_data.json").write_text(
        json.dumps(clinical, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def execute() -> dict:
    env = os.environ.copy()
    env["TREATMENT_REFINEMENT_DEBUG"] = "0"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--patient-id",
            TEST_PATIENT,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Treatment refinement test failed.\n\n"
            + result.stdout
            + "\n"
            + result.stderr
        )

    output = (
        REFINE_DIR
        / "refined_treatment_recommendation.json"
    )
    return json.loads(output.read_text(encoding="utf-8"))


def score_map(payload: dict) -> dict[str, float]:
    return {
        item["treatment"]: float(item["score"])
        for item in payload["personalized_treatment_ranking"]
    }


def factor_codes(payload: dict, treatment_name: str) -> set[str]:
    for item in payload["personalized_treatment_ranking"]:
        if item.get("treatment") != treatment_name:
            continue

        factors = (
            item.get("supporting_factors", [])
            + item.get("limiting_factors", [])
        )
        return {
            factor.get("code")
            for factor in factors
            if isinstance(factor, dict)
        }

    return set()


def main() -> int:
    if not SCRIPT.is_file():
        raise FileNotFoundError(SCRIPT)

    try:
        write_case(complete_concentric=False)
        normal = execute()
        normal_scores = score_map(normal)

        assert normal_scores["SURGERY"] > 25.0
        assert normal_scores["CPAP"] < 80.0
        assert normal_scores["APAP"] < 70.0
        assert normal_scores["HGNS"] > 35.0
        assert (
            "DISE_TONGUE_BASE_COLLAPSE_SUPPORTS_HGNS_EVALUATION"
            in factor_codes(normal, "HGNS")
        )

        write_case(complete_concentric=True)
        ccc = execute()
        ccc_scores = score_map(ccc)

        assert ccc_scores["HGNS"] < normal_scores["HGNS"]
        assert (
            "DISE_CCC_LIMITS_HGNS_ELIGIBILITY"
            in factor_codes(ccc, "HGNS")
        )

        print("PASS: cross-treatment anatomical rules are working.")
        print("Non-CCC scores:", normal_scores)
        print("CCC scores:", ccc_scores)
        return 0

    finally:
        if PATIENT_ROOT.exists():
            shutil.rmtree(PATIENT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
