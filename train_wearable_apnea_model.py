from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from src.continual_learning.apnea60 import build_baseline_target
from src.continual_learning.wearable_realtime import WATCH_FEATURES, feature_window

ROOT = Path(__file__).resolve().parent


def main():
    data = pd.read_csv(ROOT / "data" / "processed" / "model_dataset.csv")
    data["target"] = build_baseline_target(data).astype(int)
    valid = data["patient_id"].notna()
    for column in ("quality_core_features_valid", "usable_for_core_respiratory_model"):
        if column in data:
            valid &= data[column].fillna(False).astype(bool)
    data = data.loc[valid].reset_index(drop=True)
    x, y, groups = data[WATCH_FEATURES].copy(), data["target"].to_numpy(), data["patient_id"].astype(str)
    realtime_rows = []
    for sample_path in (ROOT / "data" / "realtime_wearable").glob("*/samples.csv"):
        patient_id = sample_path.parent.name
        samples = pd.read_csv(sample_path)
        labels = samples.get("apnea_observed_next_60s")
        if labels is None:
            continue
        metadata_path = ROOT / "data" / "processed" / patient_id / "patient_metadata.csv"
        age = bmi = np.nan
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path).iloc[0]
            age, bmi = metadata.get("age"), metadata.get("BMI")
        for index, label in labels.items():
            if pd.isna(label) or index < 9:
                continue
            features = feature_window(samples.iloc[max(0, index - 29):index + 1], age, bmi).iloc[0].to_dict()
            text_label = str(label).strip().lower()
            features["target"] = int(text_label in {"1", "true", "yes"})
            features["patient_id"] = patient_id
            source = str(samples.iloc[index].get("source", ""))
            if source.startswith("model_pseudo_label"):
                features["label_quality"] = "model_pseudo_label"
            elif source.startswith("uploaded:"):
                features["label_quality"] = "uploaded_unverified"
            else:
                features["label_quality"] = "observed_or_external_label"
            realtime_rows.append(features)
    if realtime_rows:
        realtime = pd.DataFrame(realtime_rows)
        verified = realtime.loc[
            realtime["label_quality"].eq("observed_or_external_label")
        ].copy()
        if not verified.empty:
            x = pd.concat([x, verified[WATCH_FEATURES]], ignore_index=True)
            y = np.concatenate([y, verified["target"].to_numpy(dtype=int)])
            groups = pd.concat([groups.reset_index(drop=True), verified["patient_id"].astype(str)], ignore_index=True)
    # x may contain additional clinician-labelled wearable windows.  The old
    # allocation used only the baseline PSG row count, so fold indices for the
    # appended rows could exceed the array boundary.
    oof = np.full(len(x), np.nan)
    for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, groups):
        model = make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.04, subsample=.85, colsample_bytree=.85, n_jobs=-1, random_state=42, eval_metric="logloss"))
        model.fit(x.iloc[train], y[train]); oof[test] = model.predict_proba(x.iloc[test])[:, 1]
    calibrator = LogisticRegression(max_iter=1000).fit(oof.reshape(-1, 1), y)
    probability = calibrator.predict_proba(oof.reshape(-1, 1))[:, 1]
    candidates = []
    for threshold in np.arange(.01, .991, .001):
        pred = probability >= threshold
        recall = recall_score(y, pred, zero_division=0)
        if recall >= .90:
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            candidates.append((tn / max(tn + fp, 1), precision_score(y, pred, zero_division=0), threshold, recall, fn, fp))
    specificity, precision, threshold, recall, fn, fp = max(candidates)
    notification_threshold = max(0.50, max(
        np.arange(.01, .991, .001),
        key=lambda value: balanced_accuracy_score(y, probability >= value),
    ))
    final = make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.04, subsample=.85, colsample_bytree=.85, n_jobs=-1, random_state=42, eval_metric="logloss"))
    final.fit(x, y)
    final_train_raw = final.predict_proba(x)[:, 1]
    version = "wearable60_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    # Keep the research challenger separate from the serving model.  A later
    # governed promotion step may copy it to model.joblib after independent
    # validation; training itself never overwrites the serving artifact.
    output = ROOT / "models" / "wearable_apnea_next_60s" / "challenger.joblib"; output.parent.mkdir(parents=True, exist_ok=True)
    pseudo_count = sum(row.get("label_quality") == "model_pseudo_label" for row in realtime_rows)
    uploaded_unverified_count = sum(row.get("label_quality") == "uploaded_unverified" for row in realtime_rows)
    observed_count = sum(row.get("label_quality") == "observed_or_external_label" for row in realtime_rows)
    joblib.dump({"model": final, "calibrator": calibrator, "final_train_raw_sorted": np.sort(final_train_raw), "oof_raw_sorted": np.sort(oof), "threshold": float(threshold), "notification_threshold": float(notification_threshold), "window_samples": 30, "version": version, "metrics": {"recall": float(recall), "specificity": float(specificity), "precision": float(precision), "false_negative": int(fn), "false_positive": int(fp)}, "patient_count": int(groups.nunique()), "realtime_labelled_windows": len(realtime_rows), "model_pseudo_label_windows": int(pseudo_count), "uploaded_unverified_windows": int(uploaded_unverified_count), "observed_or_external_label_windows": int(observed_count), "excluded_unverified_windows": int(pseudo_count + uploaded_unverified_count), "deployment_status": "RESEARCH_CHALLENGER_REQUIRES_INDEPENDENT_VALIDATION", "clinical_validation": False, "promotion_allowed": False}, output)
    print(joblib.load(output)["metrics"])


if __name__ == "__main__": main()
