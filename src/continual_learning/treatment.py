from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from src.continual_learning.governance import write_audit_event
from src.continual_learning.drug_trial import (
    MODEL_KEY as DRUG_TRIAL_TREATMENT,
    load_latest_drug_trial_adjustment,
)
from sklearn.preprocessing import OneHotEncoder


IDENTITY_COLUMNS = {
    "patient_id",
    "study_id",
    "snapshot_id",
    "snapshot_at",
    "outcome_status",
    "reviewer_id",
    "approved_at",
    "label_quality",
    "sample_weight",
}


class ShrinkageRegressor(RegressorMixin, BaseEstimator):
    """Blend a fitted model with the training-fold mean to reduce small-sample variance."""

    def __init__(self, estimator: Any, model_weight: float = 0.5):
        self.estimator = estimator
        self.model_weight = model_weight

    def fit(self, features: Any, target: Any, sample_weight: Any = None) -> "ShrinkageRegressor":
        values = np.asarray(target, dtype=float)
        self.target_mean_ = float(
            np.average(values, weights=sample_weight)
            if sample_weight is not None else np.mean(values)
        )
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(features, target, sample_weight=sample_weight)
        return self

    def predict(self, features: Any) -> np.ndarray:
        model_prediction = np.asarray(self.estimator_.predict(features), dtype=float)
        weight = float(np.clip(self.model_weight, 0.0, 1.0))
        return weight * model_prediction + (1.0 - weight) * self.target_mean_
OUTCOME_ONLY_COLUMNS = {
    "follow_up_ahi",
    "follow_up_min_spo2",
    "follow_up_days",
    "symptom_improved",
    "adherence_rate",
    "adverse_event",
    "outcome_status",
    "reviewer_id",
    "approved_at",
}
LEAKAGE_COLUMNS = {
    "scenario_note", "synthetic_scenario",
    "clinical_device_data_synthetic_scenario",
    "synthetic_wearable_scenario_applied",
    "synthetic_wearable_scenario_present",
    "synthetic_data_excluded_from_clinical_inference",
    "data_source_type", "clinical_device_data_data_source_type",
    # File/batch identifiers are not clinical predictors.  Keeping them lets
    # trees memorise an upload or snapshot instead of learning physiology.
    "snapshot_content_hash",
    "clinical_device_data_sync_received_at",
    "clinical_data_version",
    "clinical_last_updated_module",
}
REQUIRED_COLUMNS = {
    "patient_id",
    "study_id",
    "treatment",
    "baseline_ahi",
    "follow_up_ahi",
    "follow_up_days",
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _flatten_context(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            result.update(_flatten_context(child, name))
            if not isinstance(child, (dict, list, tuple)):
                result.setdefault(str(key), child)
        return result
    if not isinstance(value, (list, tuple)):
        result[prefix] = value
    return result


def _snapshot_hashes(frame: pd.DataFrame) -> pd.Series:
    """Backfill stable content hashes for snapshots created before dedup v2."""
    excluded = {
        "snapshot_id", "snapshot_at", "snapshot_content_hash",
        "recommendation_rank", "recommendation_score", "outcome_status",
    }
    columns = sorted(column for column in frame.columns if column not in excluded)
    return frame[columns].fillna("").astype(str).apply(
        lambda row: hashlib.sha256(
            "\x1f".join(f"{column}={row[column]}" for column in columns).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )


def record_treatment_snapshot(
    project_root: str | Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Persist the patient state used for each recommendation until outcomes mature."""
    root = Path(project_root).resolve()
    folder = (
        root
        / "data"
        / "continual_learning"
        / "treatment_snapshots"
    )
    folder.mkdir(parents=True, exist_ok=True)
    patient_id = str(result.get("patient_id", "unknown"))
    generated_at = str(result.get("generated_at") or _now())
    snapshot_id = (
        generated_at.replace(":", "").replace("-", "").replace("+", "_")
    )
    context = _flatten_context(result.get("patient_context", {}))
    rows: list[dict[str, Any]] = []
    for item in result.get("personalized_treatment_ranking", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "patient_id": patient_id,
                "snapshot_id": snapshot_id,
                "snapshot_at": generated_at,
                "treatment": item.get("treatment"),
                "recommendation_rank": item.get("rank"),
                "recommendation_score": item.get("score"),
                "outcome_status": "pending",
                **context,
            }
        )
    if not rows:
        return {"status": "no_treatment_rows"}
    safe_patient = "".join(
        character if character.isalnum() or character in "-_ " else "_"
        for character in patient_id
    )
    frame = pd.DataFrame(rows)
    fingerprint_columns = sorted(
        column for column in frame.columns
        if column not in {"snapshot_id", "snapshot_at"}
    )
    canonical = frame[fingerprint_columns].fillna("").astype(str).to_csv(index=False)
    content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    index_path = folder / "snapshot_content_index.json"
    try:
        content_index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        content_index = {}
    dedup_key = f"{patient_id}:{content_hash}"
    if dedup_key in content_index:
        return {
            "status": "duplicate_snapshot_skipped", "row_count": 0,
            "snapshot_id": content_index[dedup_key].get("snapshot_id"),
            "content_hash": content_hash,
            "reason": "Identical patient treatment state already exists; no training rows added.",
        }
    destination = folder / f"{safe_patient}_{snapshot_id}.csv"
    frame["snapshot_content_hash"] = content_hash
    frame.to_csv(
        destination,
        index=False,
        encoding="utf-8-sig",
    )
    receipt = {
        "status": "recorded",
        "snapshot_id": snapshot_id,
        "row_count": len(rows),
        "file": str(destination),
        "content_hash": content_hash,
    }
    content_index[dedup_key] = {
        "snapshot_id": snapshot_id, "file": str(destination), "recorded_at": _now(),
    }
    _atomic_json(index_path, content_index)
    receipt["model_update"] = {
        "status": "snapshot_recorded_for_weak_supervision",
        "reason": "推薦分數不是治療療效真值；只有經確認的追蹤結果可訓練治療效果 Challenger。",
    }
    return receipt


def _build_target(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    baseline = pd.to_numeric(frame["baseline_ahi"], errors="coerce").clip(lower=1)
    follow_up = pd.to_numeric(frame["follow_up_ahi"], errors="coerce")
    ahi_ratio = ((baseline - follow_up) / baseline).clip(-1, 1)
    target = weights["ahi_reduction_ratio"] * ((ahi_ratio + 1) / 2)
    if {"baseline_min_spo2", "follow_up_min_spo2"} <= set(frame.columns):
        spo2_gain = (
            pd.to_numeric(frame["follow_up_min_spo2"], errors="coerce")
            - pd.to_numeric(frame["baseline_min_spo2"], errors="coerce")
        ).clip(-20, 20)
        target += weights["spo2_improvement_normalized"] * ((spo2_gain + 20) / 40)
    else:
        target += weights["spo2_improvement_normalized"] * 0.5
    for column, weight, default in (
        ("symptom_improved", weights["symptom_improved"], 0.5),
        ("adherence_rate", weights["adherence_rate"], 0.5),
    ):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(default)
            target += weight * values.clip(0, 1)
        else:
            target += weight * default
    if "adverse_event" in frame:
        adverse = pd.to_numeric(frame["adverse_event"], errors="coerce").fillna(0)
        target += weights["adverse_event_free"] * (1 - adverse.clip(0, 1))
    else:
        target += weights["adverse_event_free"] * 0.5
    return target.clip(0, 1)


def train_treatment_models(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = json.loads(
        (root / "config" / "continual_learning.json").read_text(encoding="utf-8")
    )["treatment"]
    outcome_dir = root / "data" / "continual_learning" / "treatment_outcomes"
    files = sorted(outcome_dir.glob("outcomes_*.csv"))
    if not files:
        raise RuntimeError("尚無已確認的治療結果資料")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"治療結果資料缺少欄位：{sorted(missing)}")
    data["learning_target"] = _build_target(data, config["outcome_weights"])
    version = "treatment_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    output = root / "models" / "registry" / "treatment" / version
    output.mkdir(parents=True, exist_ok=False)
    summaries: dict[str, Any] = {}
    minimum_rows = int(config["minimum_outcomes_per_treatment"])
    minimum_patients = int(config["minimum_unique_patients_per_treatment"])
    for treatment, group in data.groupby(data["treatment"].astype(str)):
        patient_count = group["patient_id"].astype(str).nunique()
        if len(group) < minimum_rows or patient_count < minimum_patients:
            summaries[treatment] = {
                "status": "insufficient_data",
                "row_count": len(group),
                "patient_count": int(patient_count),
            }
            continue
        candidates = [
            column
            for column in group.columns
            if column not in IDENTITY_COLUMNS
            and column not in {"learning_target", "follow_up_ahi"}
        ]
        numeric = [
            column
            for column in candidates
            if pd.api.types.is_numeric_dtype(group[column])
        ]
        categorical = [column for column in candidates if column not in numeric]
        preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    numeric,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    categorical,
                ),
            ]
        )
        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=float(config["validation_fraction"]),
            random_state=42,
        )
        train_index, test_index = next(
            splitter.split(group[candidates], group["learning_target"], group["patient_id"])
        )
        model.fit(group.iloc[train_index][candidates], group.iloc[train_index]["learning_target"])
        prediction = model.predict(group.iloc[test_index][candidates])
        metrics = {
            "mae": float(mean_absolute_error(group.iloc[test_index]["learning_target"], prediction)),
            "r2": float(r2_score(group.iloc[test_index]["learning_target"], prediction))
            if len(test_index) >= 2
            else None,
        }
        model.fit(group[candidates], group["learning_target"])
        safe_name = "".join(c if c.isalnum() else "_" for c in treatment)
        joblib.dump(model, output / f"{safe_name}.joblib")
        _atomic_json(
            output / f"{safe_name}.json",
            {
                "treatment": treatment,
                "feature_columns": candidates,
                "numeric_columns": numeric,
                "categorical_columns": categorical,
                "row_count": len(group),
                "patient_count": int(patient_count),
                "metrics": metrics,
            },
        )
        summaries[treatment] = {
            "status": "trained",
            "row_count": len(group),
            "patient_count": int(patient_count),
            "metrics": metrics,
        }
    metadata = {
        "model_version": version,
        "model_family": "treatment_effect",
        "created_at": _now(),
        "sklearn_version": sklearn.__version__,
        "outcome_row_count": len(data),
        "treatments": summaries,
        "deployment_status": "RESEARCH_CHALLENGER_REQUIRES_INDEPENDENT_VALIDATION",
        "promotion_allowed": False,
    }
    _atomic_json(output / "metadata.json", metadata)
    registry_dir = root / "models" / "registry" / "treatment"
    _atomic_json(
        registry_dir / "registry.json",
        {"champion": None, "latest_challenger": version, "updated_at": _now(), "models": [metadata]},
    )
    return metadata


def train_adaptive_treatment_models(
    project_root: str | Path,
    *,
    allow_synthetic_demo: bool = False,
) -> dict[str, Any]:
    """Train a research Challenger; weak rows remain explicitly non-clinical."""
    root = Path(project_root).resolve()
    config = json.loads(
        (root / "config" / "continual_learning.json").read_text(encoding="utf-8")
    )["treatment"]
    snapshot_files = sorted(
        (
            root
            / "data"
            / "continual_learning"
            / "treatment_snapshots"
        ).glob("*.csv")
    )
    outcome_files = sorted(
        (
            root
            / "data"
            / "continual_learning"
            / "treatment_outcomes"
        ).glob("outcomes_*.csv")
    )
    frames: list[pd.DataFrame] = []

    if outcome_files:
        confirmed = pd.concat(
            [pd.read_csv(path) for path in outcome_files],
            ignore_index=True,
            sort=False,
        )
        missing = REQUIRED_COLUMNS - set(confirmed.columns)
        if missing:
            raise ValueError(f"治療結果資料缺少欄位：{sorted(missing)}")
        # 將每位患者／每種治療最新的 Digital Twin 與補充臨床特徵
        # 接到已確認療效列。療效欄位仍只來自 outcome，避免推薦快照
        # 被誤當答案；快照僅提供模型輸入特徵。
        # Only verified post-treatment outcomes may supervise a clinical model.
        # Recommendation snapshots and synthetic/demo rows are never clinical truth.
        confirmed = confirmed.copy()
        confirmed["outcome_status"] = (
            confirmed["outcome_status"].fillna("").astype(str).str.lower().str.strip()
        )
        for column in ("baseline_ahi", "follow_up_ahi", "follow_up_days"):
            confirmed[column] = pd.to_numeric(confirmed[column], errors="coerce")
        eligible = (
            confirmed["outcome_status"].eq("confirmed")
            & confirmed["baseline_ahi"].gt(0)
            & confirmed["follow_up_ahi"].ge(0)
            & confirmed["follow_up_days"].between(
                float(config.get("minimum_follow_up_days", 14)),
                float(config.get("maximum_follow_up_days", 730)),
            )
        )
        reviewer = confirmed.get("reviewer_id", pd.Series("", index=confirmed.index))
        if config.get("require_reviewer_id", True):
            eligible &= reviewer.fillna("").astype(str).str.strip().ne("")
        confirmed = confirmed.loc[eligible].copy()
        confirmed = confirmed.drop_duplicates(
            ["patient_id", "study_id", "treatment"], keep="last"
        )
        synthetic = (
            confirmed["patient_id"].astype(str).str.upper().str.startswith(("TEST_", "SYNTH_", "DEMO_"))
            | confirmed.get("reviewer_id", pd.Series("", index=confirmed.index))
            .fillna("").astype(str).str.upper().str.startswith(("TEST_", "SYNTH_", "DEMO_"))
        )
        if allow_synthetic_demo:
            confirmed["label_quality"] = np.where(
                synthetic, "synthetic_test_supervision", "confirmed_outcome"
            )
        else:
            confirmed = confirmed.loc[~synthetic].copy()
            confirmed["label_quality"] = "confirmed_outcome"
        if snapshot_files:
            snapshots = pd.concat(
                [pd.read_csv(path, low_memory=False) for path in snapshot_files],
                ignore_index=True,
                sort=False,
            )
            if {"patient_id", "treatment"}.issubset(snapshots.columns):
                if "snapshot_at" in snapshots.columns:
                    snapshots["snapshot_at"] = pd.to_datetime(
                        snapshots["snapshot_at"], errors="coerce", utc=True
                    )
                    snapshots = snapshots.sort_values("snapshot_at")
                snapshots = snapshots.drop_duplicates(
                    ["patient_id", "treatment"], keep="last"
                )
                snapshot_drop = {
                    "snapshot_id",
                    "snapshot_at",
                    "recommendation_rank",
                    "recommendation_score",
                    "outcome_status",
                }
                snapshot_features = snapshots.drop(
                    columns=[c for c in snapshot_drop if c in snapshots.columns],
                    errors="ignore",
                )
                merge_keys = ["patient_id", "treatment"]
                new_feature_columns = [
                    column
                    for column in snapshot_features.columns
                    if column in merge_keys
                    or (
                        column not in confirmed.columns
                        and not str(column).endswith("_snapshot")
                    )
                ]
                snapshot_features = snapshot_features[new_feature_columns]
                confirmed = confirmed.merge(
                    snapshot_features,
                    on=["patient_id", "treatment"],
                    how="left",
                    suffixes=("", "_snapshot"),
                    validate="many_to_one",
                )
        confirmed = confirmed.copy()
        confirmed["learning_target"] = _build_target(
            confirmed,
            config["outcome_weights"],
        )
        # Long-interval paired PSG remains genuine supervision, but other
        # clinical changes become more likely as the interval grows. Keep
        # these patients in training while reducing their influence instead
        # of silently discarding them.
        long_follow_up_threshold = float(
            config.get("long_follow_up_threshold_days", 730)
        )
        long_follow_up_weight = float(
            config.get("long_follow_up_sample_weight", 0.75)
        )
        confirmed["sample_weight"] = np.where(
            confirmed["follow_up_days"].gt(long_follow_up_threshold),
            long_follow_up_weight,
            1.0,
        )
        confirmed["follow_up_quality"] = np.where(
            confirmed["follow_up_days"].gt(long_follow_up_threshold),
            "confirmed_long_interval",
            "confirmed_standard_interval",
        )
        frames.append(confirmed)

    research = json.loads(
        (root / "config" / "continual_learning.json").read_text(encoding="utf-8")
    ).get("research_preview", {})
    if research.get("include_recommendation_snapshots_as_weak_training") and snapshot_files:
        snapshots = pd.concat(
            [pd.read_csv(path, low_memory=False) for path in snapshot_files],
            ignore_index=True,
            sort=False,
        )
        required_snapshot = {"patient_id", "treatment", "recommendation_score"}
        if required_snapshot.issubset(snapshots.columns):
            snapshots = snapshots.copy()
            calculated_hashes = _snapshot_hashes(snapshots)
            if "snapshot_content_hash" not in snapshots.columns:
                snapshots["snapshot_content_hash"] = calculated_hashes
            else:
                missing_hash = snapshots["snapshot_content_hash"].fillna("").astype(str).str.strip().eq("")
                snapshots.loc[missing_hash, "snapshot_content_hash"] = calculated_hashes.loc[missing_hash]
            snapshots["learning_target"] = (
                pd.to_numeric(snapshots["recommendation_score"], errors="coerce")
                .clip(0, 100)
                .div(100.0)
            )
            snapshots["study_id"] = snapshots.get(
                "snapshot_id", pd.Series(index=snapshots.index, dtype=object)
            ).fillna("snapshot")
            snapshots["label_quality"] = "weak_rule_supervision"
            snapshots["sample_weight"] = float(
                research.get("weak_snapshot_sample_weight", 0.25)
            )
            snapshots = snapshots.dropna(subset=["learning_target"])
            snapshots = snapshots.drop_duplicates(
                ["patient_id", "study_id", "treatment"], keep="last"
            )
            if "snapshot_content_hash" in snapshots.columns:
                snapshots = snapshots.drop_duplicates(
                    ["patient_id", "treatment", "snapshot_content_hash"], keep="last"
                )
            if not snapshots.empty:
                frames.append(snapshots)

    if not frames:
        raise RuntimeError("沒有可用的確認療效列或明確標示的弱監督測試列")

    data = pd.concat(frames, ignore_index=True, sort=False)
    data = data.dropna(subset=["patient_id", "treatment", "learning_target"])
    if data.empty:
        raise RuntimeError(
            "沒有符合條件的真實治療後療效資料：測試／合成資料與推薦快照不會被當成臨床真值。"
        )
    version = "treatment_adaptive_" + datetime.now().strftime(
        "%Y%m%dT%H%M%S%f"
    )
    registry_dir = root / "models" / "registry" / "treatment"
    final_output = registry_dir / version
    output = registry_dir / f".staging-{version}"
    output.mkdir(parents=True, exist_ok=False)
    summaries: dict[str, Any] = {}

    for treatment, group in data.groupby(data["treatment"].astype(str)):
        group = group.copy()
        candidates = [
            column
            for column in group.columns
            if column not in IDENTITY_COLUMNS
            and column not in OUTCOME_ONLY_COLUMNS
            and column not in LEAKAGE_COLUMNS
            and not str(column).endswith("synthetic_test_training")
            and group[column].notna().any()
            and column
            not in {
                "learning_target",
                "follow_up_ahi",
                "recommendation_score",
                "recommendation_rank",
            }
        ]
        candidates = [
            column for column in candidates
            if group[column].notna().mean() >= 0.02
            and group[column].nunique(dropna=True) > 1
        ]
        numeric = [
            column
            for column in candidates
            if pd.api.types.is_numeric_dtype(group[column])
        ]
        categorical = [column for column in candidates if column not in numeric]
        # Snapshot CSVs can represent the same categorical field as bool in one
        # upload and text in another. OneHotEncoder requires one consistent type.
        # Preserve missing values for imputation and normalize every observed
        # category to text before patient-level splitting and model fitting.
        for column in categorical:
            group[column] = group[column].map(
                lambda value: np.nan if pd.isna(value) else str(value)
            )
        transformers = []
        if numeric:
            transformers.append(
                (
                    "numeric",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    numeric,
                )
            )
        if categorical:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical,
                )
            )
        if not transformers:
            summaries[str(treatment)] = {
                "status": "insufficient_features",
                "row_count": len(group),
            }
            continue

        preprocessor = ColumnTransformer(transformers, sparse_threshold=0.0)
        candidate_regressors = {
            "ridge_alpha_0_1": Ridge(alpha=0.1),
            "ridge_alpha_1": Ridge(alpha=1.0),
            "ridge_alpha_10": Ridge(alpha=10.0),
            "random_forest_leaf_1": RandomForestRegressor(
                n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                random_state=42, n_jobs=-1,
            ),
            "shrinkage_rf_10pct": ShrinkageRegressor(
                RandomForestRegressor(
                    n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                    random_state=42, n_jobs=-1,
                ),
                model_weight=0.10,
            ),
            "shrinkage_rf_25pct": ShrinkageRegressor(
                RandomForestRegressor(
                    n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                    random_state=42, n_jobs=-1,
                ),
                model_weight=0.25,
            ),
            "shrinkage_rf_50pct": ShrinkageRegressor(
                RandomForestRegressor(
                    n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                    random_state=42, n_jobs=-1,
                ),
                model_weight=0.50,
            ),
            "random_forest_leaf_2": RandomForestRegressor(
                n_estimators=500, min_samples_leaf=2, max_features=0.8,
                random_state=42, n_jobs=-1,
            ),
            "random_forest_leaf_4": RandomForestRegressor(
                n_estimators=500, min_samples_leaf=4, max_features=1.0,
                random_state=42, n_jobs=-1,
            ),
            "extra_trees_leaf_1": ExtraTreesRegressor(
                n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                random_state=42, n_jobs=-1,
            ),
            "extra_trees_leaf_2": ExtraTreesRegressor(
                n_estimators=500, min_samples_leaf=2, max_features=0.8,
                random_state=42, n_jobs=-1,
            ),
            "gradient_boosting_huber": GradientBoostingRegressor(
                loss="huber", n_estimators=300, learning_rate=0.03,
                max_depth=2, min_samples_leaf=2, random_state=42,
            ),
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                loss="absolute_error", max_iter=300, learning_rate=0.04,
                max_leaf_nodes=7, min_samples_leaf=4, l2_regularization=1.0,
                random_state=42,
            ),
        }
        patient_count = group["patient_id"].astype(str).nunique()
        minimum_rows = int(config["minimum_outcomes_per_treatment"])
        minimum_patients = int(config["minimum_unique_patients_per_treatment"])
        if len(group) < minimum_rows or patient_count < minimum_patients:
            summaries[str(treatment)] = {
                "status": "insufficient_confirmed_outcomes",
                "row_count": int(len(group)),
                "patient_count": int(patient_count),
                "required_row_count": minimum_rows,
                "required_patient_count": minimum_patients,
            }
            continue
        metrics: dict[str, Any] = {
            "validation_status": "insufficient_unique_patients",
            "mae": None,
            "r2": None,
        }
        selected_model_name = "random_forest_leaf_2"
        model_selection_results: list[dict[str, Any]] = []
        if patient_count >= 2 and len(group) >= 4:
            fold_count = min(5, int(patient_count))
            splitter = GroupKFold(n_splits=fold_count)
            splits = list(splitter.split(
                group[candidates], group["learning_target"], group["patient_id"]
            ))
            normalized_weight = (
                group["sample_weight"]
                / group.groupby(group["patient_id"].astype(str))["sample_weight"]
                .transform("sum").clip(lower=1e-9)
            )
            best_predictions = None
            best_mae = float("inf")
            baseline_predictions = np.empty(len(group), dtype=float)
            for train_index, test_index in splits:
                baseline_predictions[test_index] = float(
                    np.average(
                        group.iloc[train_index]["learning_target"],
                        weights=normalized_weight.iloc[train_index],
                    )
                )
            baseline_mae = float(mean_absolute_error(
                group["learning_target"], baseline_predictions
            ))
            for candidate_name, regressor in candidate_regressors.items():
                predictions = np.empty(len(group), dtype=float)
                fold_maes: list[float] = []
                for train_index, test_index in splits:
                    fold_model = Pipeline([
                        ("preprocessor", clone(preprocessor)),
                        ("regressor", clone(regressor)),
                    ])
                    fold_model.fit(
                        group.iloc[train_index][candidates],
                        group.iloc[train_index]["learning_target"],
                        regressor__sample_weight=normalized_weight.iloc[train_index],
                    )
                    fold_prediction = fold_model.predict(
                        group.iloc[test_index][candidates]
                    )
                    predictions[test_index] = fold_prediction
                    fold_maes.append(float(mean_absolute_error(
                        group.iloc[test_index]["learning_target"], fold_prediction
                    )))
                candidate_mae = float(mean_absolute_error(
                    group["learning_target"], predictions
                ))
                model_selection_results.append({
                    "model": candidate_name,
                    "mean_mae": candidate_mae,
                    "fold_mae": fold_maes,
                })
                if candidate_mae < best_mae:
                    best_mae = candidate_mae
                    selected_model_name = candidate_name
                    best_predictions = predictions.copy()
            validation_target = group["learning_target"]
            target_std = float(validation_target.std(ddof=0))
            stable_r2 = len(group) >= 3 and target_std >= 0.05
            metrics = {
                "validation_status": "patient_group_cross_validation",
                "fold_count": fold_count,
                "selection_metric": "lowest_out_of_fold_mean_absolute_error",
                "selected_model": selected_model_name,
                "mae": best_mae,
                "mean_baseline_mae": baseline_mae,
                "outperforms_mean_baseline": bool(best_mae < baseline_mae),
                "target_standard_deviation": target_std,
                "r2": (
                    float(r2_score(validation_target, best_predictions))
                    if stable_r2
                    else None
                ),
                "r2_interpretation": (
                    "reported"
                    if stable_r2
                    else "not_reported_because_validation_target_variation_is_too_low"
                ),
                "candidate_results": sorted(
                    model_selection_results, key=lambda row: row["mean_mae"]
                ),
            }

        model = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("regressor", clone(candidate_regressors[selected_model_name])),
        ])
        model.fit(
            group[candidates],
            group["learning_target"],
            regressor__sample_weight=(
                group["sample_weight"]
                / group.groupby(group["patient_id"].astype(str))["sample_weight"]
                .transform("sum").clip(lower=1e-9)
            ),
        )
        safe_name = "".join(
            character if character.isalnum() else "_"
            for character in str(treatment)
        )
        joblib.dump(model, output / f"{safe_name}.joblib")
        model_metadata = {
            "treatment": str(treatment),
            "feature_columns": candidates,
            "numeric_columns": numeric,
            "categorical_columns": categorical,
            "row_count": len(group),
            "patient_count": int(patient_count),
            "weak_label_rows": int(
                group["label_quality"].eq("weak_rule_supervision").sum()
            ),
            "confirmed_outcome_rows": int(
                group["label_quality"].eq("confirmed_outcome").sum()
            ),
            "synthetic_test_rows": int(
                group["label_quality"].eq("synthetic_test_supervision").sum()
            ),
            "metrics": metrics,
            "selected_algorithm": selected_model_name,
            "score_adjustment_allowed": bool(
                metrics.get("validation_status") == "patient_group_cross_validation"
                and metrics.get("outperforms_mean_baseline") is True
            ),
        }
        _atomic_json(output / f"{safe_name}.json", model_metadata)
        summaries[str(treatment)] = {"status": "trained", **model_metadata}

    metadata = {
        "model_version": version,
        "model_family": "adaptive_treatment_effect",
        "created_at": _now(),
        "sklearn_version": sklearn.__version__,
        "training_row_count": len(data),
        "weak_label_rows": int(
            data["label_quality"].eq("weak_rule_supervision").sum()
        ),
        "confirmed_outcome_rows": int(
            data["label_quality"].eq("confirmed_outcome").sum()
        ),
        "synthetic_test_rows": int(
            data["label_quality"].eq("synthetic_test_supervision").sum()
        ),
        "deployment_status": "RESEARCH_CHALLENGER_REQUIRES_INDEPENDENT_VALIDATION",
        "promotion_allowed": False,
        "clinical_effectiveness_claim": False,
        "transaction_status": "COMMITTED",
        "feature_policy": "ALLOWLISTED_BASELINE_FEATURES_WITH_OUTCOME_AND_TEXT_LEAKAGE_EXCLUDED",
        "deduplication_policy": "PATIENT_STUDY_TREATMENT_AND_CONTENT_HASH",
        "validation_policy": "PATIENT_GROUP_HOLDOUT",
        "training_policy": (
            "mixed_confirmed_synthetic_and_weak_snapshot_research"
            if data["label_quality"].nunique() > 1
            else (
                "synthetic_demo_only"
                if data["label_quality"].eq("synthetic_test_supervision").all()
                else (
                    "weak_snapshot_research_only"
                    if data["label_quality"].eq("weak_rule_supervision").all()
                    else "confirmed_post_treatment_outcomes_only"
                )
            )
        ),
        "weak_recommendation_snapshots_used": bool(
            data["label_quality"].eq("weak_rule_supervision").any()
        ),
        "weak_snapshot_sample_weight": float(
            research.get("weak_snapshot_sample_weight", 0.25)
        ),
        "clinical_effectiveness_reason": (
            "尚須足量真實療效資料、時間外驗證、外部驗證及前瞻性臨床評估。"
        ),
        "treatments": summaries,
    }
    _atomic_json(output / "metadata.json", metadata)
    # A complete four-treatment version becomes visible atomically. An
    # interrupted run remains under .staging-* and can never be selected.
    output.replace(final_output)
    output = final_output
    registry_file = registry_dir / "registry.json"
    if registry_file.exists():
        try:
            registry = json.loads(registry_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            registry = {"champion": None, "previous_champion": None, "models": []}
    else:
        registry = {"champion": None, "previous_champion": None, "models": []}
    registry["latest_challenger"] = version
    registry["updated_at"] = _now()
    registry.setdefault("models", []).append(metadata)
    _atomic_json(registry_file, registry)
    write_audit_event(root, "_MODEL_REGISTRY_", {
        "event_type": "TREATMENT_CHALLENGER_COMMITTED",
        "model_version": version,
        "training_row_count": metadata["training_row_count"],
        "confirmed_outcome_rows": metadata["confirmed_outcome_rows"],
        "synthetic_test_rows": metadata["synthetic_test_rows"],
        "training_policy": metadata["training_policy"],
        "treatments": {key: value.get("status") for key, value in summaries.items()},
    })
    return metadata


def apply_learned_treatment_adjustment(
    project_root: str | Path,
    result: dict[str, Any],
    *,
    research_preview: bool = False,
) -> dict[str, Any]:
    """Apply bounded ML adjustments while keeping clinical rules authoritative."""
    root = Path(project_root).resolve()
    registry_file = root / "models" / "registry" / "treatment" / "registry.json"
    result = dict(result)
    if not registry_file.exists():
        result["treatment_learning"] = {
            "status": "rules_only",
            "reason": "尚未累積足夠且經確認的治療追蹤結果",
        }
        return result
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    version = (
        registry.get("latest_challenger")
        if research_preview
        else registry.get("champion")
    )
    model_dir = registry_file.parent / str(version)
    champion_metadata_file = model_dir / "metadata.json"
    if not version or not champion_metadata_file.exists():
        result["treatment_learning"] = {
            "status": "rules_only",
            "reason": "沒有通過獨立驗證與人工核准的治療效果 Champion",
        }
        return result
    champion_metadata = json.loads(champion_metadata_file.read_text(encoding="utf-8"))
    training_policy = champion_metadata.get("training_policy")
    research_policies = {
        "synthetic_demo_only",
        "weak_snapshot_research_only",
        "mixed_confirmed_synthetic_and_weak_snapshot_research",
    }
    policy_allowed = training_policy == "confirmed_post_treatment_outcomes_only" or (
        research_preview and training_policy in research_policies
    )
    if not policy_allowed:
        result["treatment_learning"] = {
            "status": "rules_only",
            "reason": (
                "此模型版本曾使用推薦快照或未標示真實療效政策，已禁止影響治療分數。"
            ),
            "blocked_model_version": version,
        }
        return result
    if champion_metadata.get("promotion_allowed") is not True and not research_preview:
        result["treatment_learning"] = {
            "status": "rules_only",
            "reason": "現有治療模型未通過獨立臨床驗證，已禁止影響病人排名",
            "blocked_model_version": version,
        }
        return result
    context = _flatten_context(result.get("patient_context", {}))
    ranking = result.get("personalized_treatment_ranking", [])
    config = json.loads(
        (root / "config" / "continual_learning.json").read_text(encoding="utf-8")
    )["treatment"]
    maximum = float(
        config.get("synthetic_demo_maximum_adjustment_points", 50.0)
        if training_policy == "synthetic_demo_only"
        else config["maximum_ml_adjustment_points"]
    )
    adjusted = []
    for item in ranking:
        current = dict(item)
        treatment = str(current.get("treatment", ""))
        safe_name = "".join(c if c.isalnum() else "_" for c in treatment)
        model_file = model_dir / f"{safe_name}.joblib"
        metadata_file = model_dir / f"{safe_name}.json"
        if not model_file.exists() or not metadata_file.exists():
            current["learned_adjustment"] = 0.0
            current["learned_status"] = "insufficient_data"
            adjusted.append(current)
            continue
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        feature_columns = metadata["feature_columns"]
        row = {"treatment": treatment, **context}
        frame = pd.DataFrame([{column: row.get(column) for column in feature_columns}])
        expected_utility = float(np.clip(joblib.load(model_file).predict(frame)[0], 0, 1))
        metrics = metadata.get("metrics", {})
        confirmed_outcomes = int(metadata.get("confirmed_outcome_rows", 0) or 0)
        minimum_confirmed = int(
            config.get("minimum_confirmed_outcomes_for_blending", 20)
        )
        if (
            metadata.get("score_adjustment_allowed") is not True
            or confirmed_outcomes < minimum_confirmed
        ):
            current["rule_based_score"] = float(current.get("score", 0.0))
            current["predicted_treatment_utility"] = expected_utility
            current["learned_adjustment"] = 0.0
            current["model_blending_weight"] = 0.0
            current["learned_status"] = "validation_blocked"
            current["learned_validation"] = {
                "status": "blocked_from_score_adjustment",
                "reason": (
                    "insufficient_confirmed_post_treatment_outcomes"
                    if confirmed_outcomes < minimum_confirmed
                    else "patient_level_validation_did_not_outperform_mean_baseline"
                ),
                "confirmed_outcome_rows": confirmed_outcomes,
                "minimum_confirmed_outcomes": minimum_confirmed,
                "metrics": metrics,
            }
            adjusted.append(current)
            continue

        # Blend the learned treatment benefit with the clinical prior instead
        # of adding another score on top of the same AHI/BMI/etc. evidence.
        # The model receives weight only from verified post-treatment labels,
        # patient-group validation and (when available) external validation.
        mae = float(metrics.get("mae", 1.0) or 1.0)
        baseline_mae = float(metrics.get("mean_baseline_mae", 0.0) or 0.0)
        relative_gain = (
            float(np.clip(1.0 - mae / baseline_mae, 0.0, 1.0))
            if baseline_mae > 0
            else 0.0
        )
        r2_reliability = float(np.clip(metrics.get("r2", 0.0) or 0.0, 0.0, 1.0))
        validation_reliability = (relative_gain + r2_reliability) / 2.0
        full_weight_n = max(
            minimum_confirmed,
            int(config.get("confirmed_outcomes_for_full_weight", 50)),
        )
        quantity_reliability = float(
            np.clip(confirmed_outcomes / full_weight_n, 0.0, 1.0)
        )
        externally_validated = bool(champion_metadata.get("promotion_allowed"))
        weight_cap = float(
            config.get(
                "maximum_validated_model_weight"
                if externally_validated
                else "maximum_unvalidated_model_weight",
                0.80 if externally_validated else 0.35,
            )
        )
        model_weight = float(
            np.clip(quantity_reliability * validation_reliability, 0.0, weight_cap)
        )
        base_score = float(current.get("score", 0.0))
        learned_score = expected_utility * 100.0
        blended_score = (1.0 - model_weight) * base_score + model_weight * learned_score
        current["rule_based_score"] = base_score
        current["predicted_treatment_utility"] = expected_utility
        current["model_benefit_score"] = learned_score
        current["model_blending_weight"] = model_weight
        current["clinical_prior_weight"] = 1.0 - model_weight
        current["learned_adjustment"] = blended_score - base_score
        current["score"] = float(np.clip(blended_score, 0, 100))
        current["learned_status"] = "confidence_weighted_blend"
        current["learned_validation"] = {
            "confirmed_outcome_rows": confirmed_outcomes,
            "relative_mae_gain": relative_gain,
            "r2_reliability": r2_reliability,
            "quantity_reliability": quantity_reliability,
            "externally_validated": externally_validated,
            "model_weight_cap": weight_cap,
        }
        adjusted.append(current)
    # The randomized drug-vs-placebo PSG study is a separate research
    # Challenger.  It may adjust only the medication research preview; it is
    # deliberately excluded from tonight-risk inference and the clinical
    # treatment Champion.
    drug_trial = load_latest_drug_trial_adjustment(root) if research_preview else None
    if drug_trial:
        for current in adjusted:
            if str(current.get("treatment")) != DRUG_TRIAL_TREATMENT:
                continue
            # The trial drug is unidentified and its challenger has not passed
            # promotion. Preserve the evidence for review, but do not add a
            # cohort-level effect directly to every individual patient's score.
            base_score = float(current.get("score", 0.0))
            current["drug_trial_rule_score"] = base_score
            current["drug_trial_adjustment"] = 0.0
            current["drug_trial_model_weight"] = 0.0
            current["drug_trial_learning"] = {
                "status": "research_evidence_retained_not_individualized",
                "model_version": drug_trial.get("model_version"),
                "drug_name": drug_trial.get("drug_name"),
                "complete_pair_count": drug_trial.get("complete_pair_count"),
                "drug_count": drug_trial.get("drug_count"),
                "placebo_count": drug_trial.get("placebo_count"),
                "estimated_relative_effect": drug_trial.get("estimated_relative_effect"),
                "warning": drug_trial.get("warning"),
                "reason": "藥名未提供且尚未通過獨立驗證，不直接改寫個別患者排序。",
            }
            break
    adjusted.sort(key=lambda item: float(item.get("score", 0)), reverse=True)
    for rank, item in enumerate(adjusted, start=1):
        item["rank"] = rank
    result["personalized_treatment_ranking"] = adjusted
    result["treatment_learning"] = {
        "status": (
            "research_preview_latest_challenger"
            if research_preview
            else "hybrid_rules_and_learning"
        ),
        "model_version": version,
        "research_preview": research_preview,
        "training_policy": training_policy,
        "deployment_status": next(
            (
                item.get("deployment_status")
                for item in reversed(registry.get("models", []))
                if item.get("model_version") == version
            ),
            "RESEARCH_CHALLENGER",
        ),
        "maximum_adjustment_points": maximum,
        "scoring_policy": "confidence_weighted_clinical_prior_and_model_benefit",
        "scoring_formula": (
            "final_score = clinical_prior_score * (1 - model_weight) "
            "+ predicted_treatment_benefit_score * model_weight; "
            "medical contraindications and safety guardrails are applied afterward"
        ),
        "model_weight_policy": {
            "minimum_confirmed_post_treatment_outcomes": int(
                config.get("minimum_confirmed_outcomes_for_blending", 20)
            ),
            "confirmed_outcomes_for_full_weight": int(
                config.get("confirmed_outcomes_for_full_weight", 50)
            ),
            "maximum_unvalidated_model_weight": float(
                config.get("maximum_unvalidated_model_weight", 0.35)
            ),
            "maximum_validated_model_weight": float(
                config.get("maximum_validated_model_weight", 0.80)
            ),
            "missing_data_policy": "reduces_confidence_only_not_score",
            "duplicate_feature_scoring_policy": "no_additive_double_counting",
        },
        "safety_layer": "原臨床規則與禁忌條件保留；學習模型只能有限度調整分數",
        "model_selection": {
            treatment_name: {
                "selected_algorithm": details.get("selected_algorithm"),
                "validation_status": details.get("metrics", {}).get("validation_status"),
                "fold_count": details.get("metrics", {}).get("fold_count"),
                "mae": details.get("metrics", {}).get("mae"),
                "mean_baseline_mae": details.get("metrics", {}).get("mean_baseline_mae"),
                "outperforms_mean_baseline": details.get("metrics", {}).get(
                    "outperforms_mean_baseline"
                ),
                "r2": details.get("metrics", {}).get("r2"),
                "selected_fold_mae": next(
                    (
                        candidate.get("fold_mae", [])
                        for candidate in details.get("metrics", {}).get(
                            "candidate_results", []
                        )
                        if candidate.get("model") == details.get("selected_algorithm")
                    ),
                    [],
                ),
                "score_adjustment_allowed": details.get("score_adjustment_allowed"),
            }
            for treatment_name, details in champion_metadata.get("treatments", {}).items()
            if isinstance(details, dict)
        },
    }
    if drug_trial:
        result["treatment_learning"]["drug_trial_challenger"] = {
            "model_version": drug_trial.get("model_version"),
            "complete_pair_count": drug_trial.get("complete_pair_count"),
            "drug_count": drug_trial.get("drug_count"),
            "placebo_count": drug_trial.get("placebo_count"),
            "research_score_adjustment_points": drug_trial.get(
                "research_score_adjustment_points"
            ),
            "affects_tonight_risk_model": False,
            "promotion_allowed": False,
        }
    if research_preview:
        result["treatment_learning"]["warning"] = (
            "此排名含弱監督測試列，僅用於驗證持續學習流程，禁止作為臨床決策。"
        )
    return result
