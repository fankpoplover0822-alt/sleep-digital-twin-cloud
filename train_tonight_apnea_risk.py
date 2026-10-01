from pathlib import Path
from src.continual_learning.tonight_apnea_risk import train

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    bundle = train(ROOT, ROOT / "models" / "tonight_apnea_risk" / "model.joblib")
    print(f"trained {bundle['version']} with {bundle['trained_patients']} patients")
