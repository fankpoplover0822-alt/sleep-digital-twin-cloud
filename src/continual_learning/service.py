from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.models.arousal_model import (
    ArousalModelConfig,
    build_arousal_pipeline,
    sanitize_feature_frame,
)


TARGET_COLUMN = "predict_arousal_next_30s"
GROUP_COLUMN = "patient_id"
KEY_COLUMNS = ["patient_id", "study_id", "epoch_index"]
MATCH_COLUMNS = ["patient_id", "epoch_index"]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_version(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"不支援的資料格式：{suffix}")


def _to_binary(series: pd.Series, name: str) -> pd.Series:
    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
    }
    converted = series.map(
        lambda value: mapping.get(str(value).strip().lower(), value)
    )
    numeric = pd.to_numeric(converted, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError(f"{name} 只能包含 0/1、true/false 或 yes/no")
    return numeric.astype(int)


@dataclass(frozen=True)
class LearningPaths:
    project_root: Path

    @property
    def root(self) -> Path:
        return self.project_root / "data" / "continual_learning"

    @property
    def pending(self) -> Path:
        return self.root / "pending"

    @property
    def approved(self) -> Path:
        return self.root / "approved"

    @property
    def rejected(self) -> Path:
        return self.root / "rejected"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def outcomes(self) -> Path:
        return self.root / "treatment_outcomes"

    @property
    def registry_root(self) -> Path:
        return self.project_root / "models" / "registry"

    @property
    def champion(self) -> Path:
        return self.registry_root / "champion"

    @property
    def challengers(self) -> Path:
        return self.registry_root / "challengers"

    @property
    def archived(self) -> Path:
        return self.registry_root / "archived"

    @property
    def registry_file(self) -> Path:
        return self.registry_root / "registry.json"

    @property
    def audit_log(self) -> Path:
        return self.root / "audit.jsonl"


def resolve_arousal_artifacts(
    project_root: str | Path,
    *,
    use_latest_challenger: bool = False,
) -> dict[str, Path]:
    """Use the validated Champion when present, otherwise the legacy model."""
    root = Path(project_root).resolve()
    champion = root / "models" / "registry" / "champion"
    legacy = root / "models" / "arousal_next_30s"
    registry_file = root / "models" / "registry" / "registry.json"
    registry = (
        json.loads(registry_file.read_text(encoding="utf-8"))
        if registry_file.exists() else {}
    )
    registered_champion = str(registry.get("champion") or "")
    # The registry is authoritative. The champion directory may contain a
    # stale serialized model copied by an older runtime, so its mere existence
    # must never override a registry entry that explicitly selects legacy.
    use_versioned_champion = bool(
        registered_champion
        and registered_champion != "legacy_arousal_next_30s"
        and (champion / "model.joblib").exists()
    )
    selected = champion if use_versioned_champion else legacy
    active_mode = "validated_champion" if use_versioned_champion else "legacy_model"
    if use_latest_challenger:
        if registry_file.exists():
            latest = registry.get("latest_challenger")
            if not latest:
                candidates = sorted(
                    (root / "models" / "registry" / "challengers").glob("arousal_*"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                latest = candidates[0].name if candidates else None
            challenger = root / "models" / "registry" / "challengers" / str(latest)
            if latest and (challenger / "model.joblib").exists():
                selected = challenger
                active_mode = "research_latest_challenger"
    return {
        "model": (
            selected / "model.joblib"
            if selected != legacy
            else selected / "arousal_next_30s.joblib"
        ),
        "features": selected / "feature_columns.json",
        "thresholds": selected / "selected_thresholds.json",
        "model_dir": selected,
        "active_mode": active_mode,
    }


class ContinualLearningService:
    """Versioned, human-gated continual learning for both model families."""

    def __init__(self, project_root: str | Path):
        self.paths = LearningPaths(Path(project_root).resolve())
        self.config = self._load_config()
        self.initialize()

    def _load_config(self) -> dict[str, Any]:
        path = self.paths.project_root / "config" / "continual_learning.json"
        if not path.exists():
            raise FileNotFoundError(f"找不到持續學習設定：{path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def initialize(self) -> None:
        for folder in (
            self.paths.pending,
            self.paths.approved,
            self.paths.rejected,
            self.paths.datasets,
            self.paths.outcomes,
            self.paths.champion,
            self.paths.challengers,
            self.paths.archived,
        ):
            folder.mkdir(parents=True, exist_ok=True)

        if not self.paths.registry_file.exists():
            _atomic_json(
                self.paths.registry_file,
                {
                    "schema_version": 1,
                    "champion": None,
                    "previous_champion": None,
                    "models": [],
                    "updated_at": _now(),
                },
            )

        legacy = (
            self.paths.project_root
            / "models"
            / "arousal_next_30s"
            / "arousal_next_30s.joblib"
        )
        champion_model = self.paths.champion / "model.joblib"
        if legacy.exists() and not champion_model.exists():
            shutil.copy2(legacy, champion_model)
            for name in (
                "feature_columns.json",
                "selected_thresholds.json",
                "training_summary.json",
            ):
                source = legacy.parent / name
                if source.exists():
                    shutil.copy2(source, self.paths.champion / name)
            registry = self.registry()
            registry["champion"] = "legacy_arousal_next_30s"
            registry["models"].append(
                {
                    "version": "legacy_arousal_next_30s",
                    "model_family": "arousal",
                    "status": "champion",
                    "created_at": _now(),
                }
            )
            registry["updated_at"] = _now()
            _atomic_json(self.paths.registry_file, registry)

    def registry(self) -> dict[str, Any]:
        return json.loads(self.paths.registry_file.read_text(encoding="utf-8"))

    def _audit(self, event: str, details: dict[str, Any]) -> None:
        self.paths.audit_log.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": _now(), "event": event, **_json_safe(details)}
        with self.paths.audit_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def status(self) -> dict[str, Any]:
        approved_files = list(self.paths.approved.glob("arousal_*.csv"))
        outcome_files = list(self.paths.outcomes.glob("outcomes_*.csv"))
        pending_files = list(self.paths.pending.glob("arousal_*.csv"))
        approved_rows = sum(len(pd.read_csv(path)) for path in approved_files)
        pending_rows = sum(len(pd.read_csv(path)) for path in pending_files)
        outcome_rows = sum(len(pd.read_csv(path)) for path in outcome_files)
        approved_patients: set[str] = set()
        for path in approved_files:
            frame = pd.read_csv(path, usecols=lambda c: c == GROUP_COLUMN)
            if GROUP_COLUMN in frame:
                approved_patients.update(frame[GROUP_COLUMN].astype(str))
        registry = self.registry()
        return {
            "champion_version": registry.get("champion"),
            "previous_champion": registry.get("previous_champion"),
            "approved_arousal_rows": approved_rows,
            "approved_new_patients": len(approved_patients),
            "pending_arousal_rows": pending_rows,
            "treatment_outcome_rows": outcome_rows,
            "arousal_retraining_ready": (
                approved_rows
                >= int(self.config["arousal"]["minimum_new_rows"])
                and len(approved_patients)
                >= int(self.config["arousal"]["minimum_new_patients"])
            ),
        }

    def ingest_arousal_feedback(
        self,
        features_file: str | Path,
        labels_file: str | Path,
        *,
        reviewer_id: str,
        approve: bool,
        label_source: str = "human_psg_review",
    ) -> dict[str, Any]:
        features_path = Path(features_file)
        labels_path = Path(labels_file)
        features = _read_table(features_path)
        labels = _read_table(labels_path)
        required = set(MATCH_COLUMNS + ["true_label"])
        missing = required - set(labels.columns)
        if missing:
            raise ValueError(f"標籤檔缺少欄位：{sorted(missing)}")
        feature_missing = set(MATCH_COLUMNS) - set(features.columns)
        if feature_missing:
            raise ValueError(f"特徵檔缺少欄位：{sorted(feature_missing)}")
        if not reviewer_id.strip():
            raise ValueError("核准資料必須填寫 reviewer_id")
        labels = labels.copy()
        if "study_id" not in labels.columns:
            labels["study_id"] = labels_path.stem
        if "study_id" not in features.columns:
            study_lookup = labels[
                MATCH_COLUMNS + ["study_id"]
            ].drop_duplicates(MATCH_COLUMNS)
            features = features.merge(
                study_lookup,
                on=MATCH_COLUMNS,
                how="left",
                validate="one_to_one",
            )
        labels["true_label"] = _to_binary(labels["true_label"], "true_label")
        if labels.duplicated(KEY_COLUMNS).any():
            raise ValueError("標籤檔含有重複 patient_id/study_id/epoch_index")
        if features.duplicated(KEY_COLUMNS).any():
            raise ValueError("特徵檔含有重複 patient_id/study_id/epoch_index")
        merged = features.merge(
            labels[KEY_COLUMNS + ["true_label"]],
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(labels):
            raise ValueError("部分標籤找不到對應的特徵資料")
        merged[TARGET_COLUMN] = merged.pop("true_label")
        merged["label_status"] = "approved" if approve else "pending"
        merged["label_source"] = label_source
        merged["reviewer_id"] = reviewer_id.strip()
        merged["reviewed_at"] = _now()
        batch_id = _safe_version("arousal")
        destination = (
            self.paths.approved if approve else self.paths.pending
        ) / f"{batch_id}.csv"
        merged.to_csv(destination, index=False, encoding="utf-8-sig")
        metadata = {
            "batch_id": batch_id,
            "status": merged["label_status"].iloc[0],
            "row_count": len(merged),
            "patient_count": int(merged[GROUP_COLUMN].astype(str).nunique()),
            "positive_count": int(merged[TARGET_COLUMN].sum()),
            "feature_source_sha256": _sha256(features_path),
            "label_source_sha256": _sha256(labels_path),
            "reviewer_id": reviewer_id.strip(),
            "created_at": _now(),
            "file": str(destination),
        }
        _atomic_json(destination.with_suffix(".json"), metadata)
        self._audit("arousal_feedback_ingested", metadata)
        return metadata

    def ingest_treatment_outcomes(
        self,
        outcomes_file: str | Path,
        *,
        reviewer_id: str,
    ) -> dict[str, Any]:
        source = Path(outcomes_file)
        frame = _read_table(source)
        required = {
            "patient_id",
            "study_id",
            "treatment",
            "baseline_ahi",
            "follow_up_ahi",
            "follow_up_days",
            "outcome_status",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"治療結果檔缺少欄位：{sorted(missing)}")
        approved = frame["outcome_status"].astype(str).str.lower().eq("confirmed")
        if not approved.all():
            raise ValueError("只有 outcome_status=confirmed 的資料可用於學習")
        numeric_required = [
            "baseline_ahi",
            "follow_up_ahi",
            "follow_up_days",
        ]
        for column in numeric_required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[numeric_required].isna().any().any():
            raise ValueError("AHI 與 follow_up_days 必須是有效數值")
        if (frame["baseline_ahi"] < 0).any() or (frame["follow_up_ahi"] < 0).any():
            raise ValueError("AHI 不可小於 0")
        if (frame["follow_up_days"] <= 0).any():
            raise ValueError("follow_up_days 必須大於 0")
        frame = frame.copy()
        snapshot_folder = (
            self.paths.project_root
            / "data"
            / "continual_learning"
            / "treatment_snapshots"
        )
        snapshot_files = sorted(snapshot_folder.glob("*.csv"))
        if snapshot_files:
            snapshots = pd.concat(
                [pd.read_csv(path) for path in snapshot_files],
                ignore_index=True,
                sort=False,
            )
            snapshots["snapshot_at"] = pd.to_datetime(
                snapshots["snapshot_at"],
                errors="coerce",
            )
            latest = (
                snapshots.sort_values("snapshot_at")
                .drop_duplicates(["patient_id", "treatment"], keep="last")
                .drop(columns=["outcome_status"], errors="ignore")
            )
            frame = frame.merge(
                latest,
                on=["patient_id", "treatment"],
                how="left",
                suffixes=("", "_snapshot"),
            )
        frame["reviewer_id"] = reviewer_id.strip()
        frame["approved_at"] = _now()
        batch_id = _safe_version("outcomes")
        destination = self.paths.outcomes / f"{batch_id}.csv"
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
        metadata = {
            "batch_id": batch_id,
            "row_count": len(frame),
            "patient_count": int(frame["patient_id"].astype(str).nunique()),
            "treatments": sorted(frame["treatment"].astype(str).unique().tolist()),
            "source_sha256": _sha256(source),
            "reviewer_id": reviewer_id.strip(),
            "created_at": _now(),
            "file": str(destination),
        }
        _atomic_json(destination.with_suffix(".json"), metadata)
        self._audit("treatment_outcomes_ingested", metadata)
        from src.continual_learning.treatment import (
            train_adaptive_treatment_models,
        )

        demo_prefixes = ("TEST_", "SYNTH_", "DEMO_")
        demo_mode = frame["patient_id"].astype(str).str.upper().str.startswith(
            demo_prefixes
        ).all()
        metadata["synthetic_demo_mode"] = bool(demo_mode)
        metadata["model_update"] = train_adaptive_treatment_models(
            self.paths.project_root,
            allow_synthetic_demo=bool(demo_mode),
        )
        return metadata

    def build_arousal_dataset(self) -> tuple[Path, dict[str, Any]]:
        base = self.paths.project_root / "data" / "processed" / "model_dataset.csv"
        if not base.exists():
            raise FileNotFoundError(f"找不到基礎訓練資料：{base}")
        frames = [pd.read_csv(base)]
        sources = [{"file": str(base), "sha256": _sha256(base)}]
        for path in sorted(self.paths.approved.glob("arousal_*.csv")):
            frame = pd.read_csv(path)
            removable = [
                "label_status",
                "label_source",
                "reviewer_id",
                "reviewed_at",
            ]
            frames.append(frame.drop(columns=removable, errors="ignore"))
            sources.append({"file": str(path), "sha256": _sha256(path)})
        combined = pd.concat(frames, ignore_index=True, sort=False)
        dedup = [
            column
            for column in ["patient_id", "study_id", "epoch_index"]
            if column in combined.columns
        ]
        combined = combined.drop_duplicates(subset=dedup, keep="last")
        version = _safe_version("dataset")
        destination = self.paths.datasets / f"{version}.csv"
        combined.to_csv(destination, index=False, encoding="utf-8-sig")
        metadata = {
            "dataset_version": version,
            "row_count": len(combined),
            "patient_count": int(combined[GROUP_COLUMN].astype(str).nunique()),
            "positive_count": int(_to_binary(combined[TARGET_COLUMN], TARGET_COLUMN).sum()),
            "sources": sources,
            "created_at": _now(),
            "file": str(destination),
        }
        _atomic_json(destination.with_suffix(".json"), metadata)
        self._audit("arousal_dataset_built", metadata)
        return destination, metadata

    def _feature_definition(self, data: pd.DataFrame) -> tuple[list[str], list[str]]:
        feature_file = (
            self.paths.project_root
            / "models"
            / "arousal_next_30s"
            / "feature_columns.json"
        )
        definition = json.loads(feature_file.read_text(encoding="utf-8"))
        numeric = [
            column for column in definition["numeric_columns"] if column in data.columns
        ]
        categorical = [
            column
            for column in definition["categorical_columns"]
            if column in data.columns
        ]
        missing = (
            set(definition["numeric_columns"] + definition["categorical_columns"])
            - set(data.columns)
        )
        if missing:
            raise ValueError(f"持續學習資料缺少正式模型特徵：{sorted(missing)}")
        return numeric, categorical

    @staticmethod
    def _metrics(
        truth: np.ndarray,
        probability: np.ndarray,
        threshold: float,
    ) -> dict[str, float]:
        prediction = (probability >= threshold).astype(int)
        return {
            "roc_auc": float(roc_auc_score(truth, probability)),
            "average_precision": float(average_precision_score(truth, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "specificity": float(
                recall_score(truth, prediction, pos_label=0, zero_division=0)
            ),
            "f1": float(f1_score(truth, prediction, zero_division=0)),
        }

    def train_arousal_challenger(
        self,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        status = self.status()
        if not force and not status["arousal_retraining_ready"]:
            raise RuntimeError(
                "尚未達到再訓練條件；可繼續累積已確認的新病患與 epoch。"
            )
        dataset_path, dataset_metadata = self.build_arousal_dataset()
        data = pd.read_csv(dataset_path)
        target = _to_binary(data[TARGET_COLUMN], TARGET_COLUMN)
        groups = data[GROUP_COLUMN].astype(str)
        numeric, categorical = self._feature_definition(data)
        features = sanitize_feature_frame(data[numeric + categorical])
        patient_count = int(groups.nunique())
        requested_splits = int(self.config["arousal"]["cross_validation_splits"])
        splits = min(requested_splits, patient_count)
        if splits < 2:
            raise RuntimeError("至少需要兩位不同病患才能進行病患分組驗證")
        threshold = float(self.config["arousal"]["decision_threshold"])
        splitter = StratifiedGroupKFold(
            n_splits=splits,
            shuffle=True,
            random_state=42,
        )
        oof = np.full(len(data), np.nan)
        fold_rows: list[dict[str, Any]] = []
        for fold, (train_index, test_index) in enumerate(
            splitter.split(features, target, groups), start=1
        ):
            train_patients = set(groups.iloc[train_index])
            test_patients = set(groups.iloc[test_index])
            if train_patients & test_patients:
                raise RuntimeError("偵測到病患層級資料洩漏")
            model = build_arousal_pipeline(
                numeric, categorical, ArousalModelConfig()
            )
            model.fit(features.iloc[train_index], target.iloc[train_index])
            probability = model.predict_proba(features.iloc[test_index])[:, 1]
            oof[test_index] = probability
            fold_rows.append(
                {
                    "fold": fold,
                    "train_rows": len(train_index),
                    "test_rows": len(test_index),
                    "train_patients": len(train_patients),
                    "test_patients": len(test_patients),
                    **self._metrics(
                        target.iloc[test_index].to_numpy(),
                        probability,
                        threshold,
                    ),
                }
            )
        if np.isnan(oof).any():
            raise RuntimeError("部分資料沒有病患分組 OOF 預測")
        metrics = self._metrics(target.to_numpy(), oof, threshold)
        model = build_arousal_pipeline(numeric, categorical, ArousalModelConfig())
        model.fit(features, target)
        version = _safe_version("arousal")
        output = self.paths.challengers / version
        output.mkdir(parents=True, exist_ok=False)
        joblib.dump(model, output / "model.joblib")
        pd.DataFrame(fold_rows).to_csv(
            output / "cross_validation_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            {
                GROUP_COLUMN: groups,
                "epoch_index": data.get("epoch_index"),
                "true_label": target,
                "predicted_probability": oof,
                "predicted_label": (oof >= threshold).astype(int),
            }
        ).to_csv(
            output / "out_of_fold_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _atomic_json(
            output / "feature_columns.json",
            {
                "numeric_columns": numeric,
                "categorical_columns": categorical,
                "target_column": TARGET_COLUMN,
            },
        )
        _atomic_json(
            output / "selected_thresholds.json",
            {"continual_learning": {"threshold": threshold}},
        )
        summary = {
            "model_version": version,
            "model_family": "arousal",
            "status": "challenger",
            "dataset_version": dataset_metadata["dataset_version"],
            "row_count": len(data),
            "patient_count": patient_count,
            "positive_count": int(target.sum()),
            "negative_count": int((target == 0).sum()),
            "decision_threshold": threshold,
            "overall_metrics": metrics,
            "created_at": _now(),
            "sklearn_version": sklearn.__version__,
            "model_sha256": _sha256(output / "model.joblib"),
        }
        _atomic_json(output / "training_summary.json", summary)
        registry = self.registry()
        registry["models"].append(summary)
        registry["latest_challenger"] = version
        registry["updated_at"] = _now()
        _atomic_json(self.paths.registry_file, registry)
        self._audit("arousal_challenger_trained", summary)
        return summary

    def evaluate_arousal_challenger(self, version: str) -> dict[str, Any]:
        summary_file = self.paths.challengers / version / "training_summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(f"找不到 Challenger：{version}")
        challenger = json.loads(summary_file.read_text(encoding="utf-8"))
        champion_summary_file = self.paths.champion / "training_summary.json"
        champion = (
            json.loads(champion_summary_file.read_text(encoding="utf-8"))
            if champion_summary_file.exists()
            else {}
        )
        candidate_metrics = challenger["overall_metrics"]
        champion_metrics = champion.get("overall_metrics", {})
        gates = self.config["arousal"]["promotion_gates"]
        checks = {
            "minimum_roc_auc": candidate_metrics["roc_auc"]
            >= float(gates["minimum_roc_auc"]),
            "minimum_recall": candidate_metrics["recall"]
            >= float(gates["minimum_recall"]),
            "minimum_specificity": candidate_metrics["specificity"]
            >= float(gates["minimum_specificity"]),
            "minimum_precision": candidate_metrics["precision"]
            >= float(gates["minimum_precision"]),
            "external_validation_approved": bool(
                self.config["arousal"].get("external_validation_approved", False)
            ),
        }
        if "roc_auc" in champion_metrics:
            checks["maximum_roc_auc_drop"] = (
                candidate_metrics["roc_auc"]
                >= champion_metrics["roc_auc"] - float(gates["maximum_roc_auc_drop"])
            )
        if "recall" in champion_metrics:
            checks["maximum_recall_drop"] = (
                candidate_metrics["recall"]
                >= champion_metrics["recall"] - float(gates["maximum_recall_drop"])
            )
        evaluation = {
            "version": version,
            "passed": all(checks.values()),
            "checks": checks,
            "challenger_metrics": candidate_metrics,
            "champion_metrics": champion_metrics,
            "evaluated_at": _now(),
        }
        _atomic_json(self.paths.challengers / version / "evaluation.json", evaluation)
        self._audit("arousal_challenger_evaluated", evaluation)
        return evaluation

    def promote_arousal_challenger(self, version: str) -> dict[str, Any]:
        evaluation_file = self.paths.challengers / version / "evaluation.json"
        if not evaluation_file.exists():
            raise RuntimeError("必須先評估 Challenger")
        evaluation = json.loads(evaluation_file.read_text(encoding="utf-8"))
        if not evaluation.get("passed"):
            raise RuntimeError("Challenger 未通過升級門檻")
        challenger = self.paths.challengers / version
        summary_file = challenger / "training_summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
        trained_version = str(summary.get("sklearn_version") or "")
        if trained_version and trained_version != sklearn.__version__:
            raise RuntimeError(
                f"候選模型使用 scikit-learn {trained_version} 建立，"
                f"目前環境為 {sklearn.__version__}；請先在目前環境重新訓練。"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            try:
                model = joblib.load(challenger / "model.joblib")
            except InconsistentVersionWarning as exc:
                raise RuntimeError(
                    "候選模型序列化版本與目前scikit-learn不相容；禁止發布，請重新訓練。"
                ) from exc
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("候選模型不支援 predict_proba")
        registry = self.registry()
        old_version = registry.get("champion")
        if any(self.paths.champion.iterdir()):
            archive_name = old_version or _safe_version("unknown_champion")
            archive_dir = self.paths.archived / archive_name
            if archive_dir.exists():
                archive_dir = self.paths.archived / _safe_version(archive_name)
            shutil.copytree(self.paths.champion, archive_dir)
            for path in self.paths.champion.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        for source in challenger.iterdir():
            if source.is_file():
                shutil.copy2(source, self.paths.champion / source.name)
        registry["previous_champion"] = old_version
        registry["champion"] = version
        for item in registry["models"]:
            if item.get("model_family") != "arousal":
                continue
            if item.get("version") == old_version:
                item["status"] = "archived"
            if item.get("version") == version:
                item["status"] = "champion"
                item["promoted_at"] = _now()
        registry["updated_at"] = _now()
        _atomic_json(self.paths.registry_file, registry)
        self._audit(
            "arousal_challenger_promoted",
            {"version": version, "previous_champion": old_version},
        )
        return {
            "champion": version,
            "previous_champion": old_version,
            "promoted_at": _now(),
        }
