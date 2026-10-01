from __future__ import annotations

import json
from pathlib import Path

from src.continual_learning.drug_trial import publish_combined_treatment_challenger


if __name__ == "__main__":
    result = publish_combined_treatment_challenger(Path(__file__).resolve().parent)
    print(json.dumps({
        key: result.get(key) for key in (
            "model_version", "base_four_treatment_component_version",
            "drug_trial_component_version", "combined_training_row_count",
            "labelled_drug_trial_rows", "unlabeled_patient_pool_count",
            "promotion_allowed", "deployment_status",
        )
    }, ensure_ascii=False, indent=2))
