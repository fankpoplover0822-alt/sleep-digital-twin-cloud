"""Two-stage, patient-grouped sequence alert models.

Stage 1 uses XGBoost for high-sensitivity candidate detection. Stage 2 uses a
BiLSTM over causal windows (past epochs through the current epoch) to reject
isolated false alarms. Probabilities are calibrated only from out-of-fold
patient predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from xgboost import XGBClassifier


@dataclass
class AlertPolicy:
    recall_floor: float = 0.90
    window_epochs: int = 7
    persistence_epochs: int = 2
    refractory_epochs: int = 2


class BiLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 24):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.20), nn.Linear(hidden * 2, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.rnn(values)
        return self.head(encoded[:, -1, :]).squeeze(1)


def quality_mask(frame: pd.DataFrame, horizon_epochs: int) -> pd.Series:
    """Reject ambiguous labels and rows without a complete future horizon."""
    mask = pd.Series(True, index=frame.index)
    for column in ("quality_core_features_valid", "usable_for_core_respiratory_model"):
        if column in frame:
            mask &= frame[column].fillna(False).astype(bool)
    position = frame.groupby(frame["patient_id"].astype(str), sort=False).cumcount()
    sizes = frame.groupby(frame["patient_id"].astype(str), sort=False)["patient_id"].transform("size")
    mask &= position < (sizes - horizon_epochs)
    if "label_confidence" in frame:
        mask &= pd.to_numeric(frame["label_confidence"], errors="coerce").fillna(0) >= 0.8
    return mask


def select_core_features(frame: pd.DataFrame, candidates: list[str], limit: int = 48) -> list[str]:
    preferred = ("flow", "airflow", "spo2", "oxygen", "thorax", "abdomen", "resp", "heart", "pulse", "snore", "position", "age", "bmi")
    usable = [c for c in candidates if c in frame and pd.api.types.is_numeric_dtype(frame[c])]
    ranked = sorted(usable, key=lambda c: (not any(k in c.lower() for k in preferred), c))
    return ranked[:limit]


def causal_sequences(values: np.ndarray, patients: np.ndarray, window: int) -> np.ndarray:
    output = np.zeros((len(values), window, values.shape[1]), dtype=np.float32)
    for index in range(len(values)):
        start = index
        while start > 0 and index - start + 1 < window and patients[start - 1] == patients[index]:
            start -= 1
        segment = values[start:index + 1]
        output[index, -len(segment):] = segment
        if len(segment) < window:
            output[index, : window - len(segment)] = segment[0]
    return output


def threshold_at_recall(y: np.ndarray, probability: np.ndarray, recall_floor: float) -> float:
    candidates: list[tuple[float, float, float]] = []
    for threshold in np.arange(0.01, 0.991, 0.001):
        pred = probability >= threshold
        recall = recall_score(y, pred, zero_division=0)
        if recall >= recall_floor:
            tn, fp, _, _ = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            candidates.append((float(threshold), float(tn / max(tn + fp, 1)), float(recall)))
    return max(candidates, key=lambda item: (item[1], item[0]))[0] if candidates else 0.10


def apply_temporal_policy(probability: np.ndarray, threshold: float, policy: AlertPolicy) -> np.ndarray:
    raw = probability >= threshold
    alert = np.zeros(len(raw), dtype=bool)
    cooldown = 0
    for index in range(len(raw)):
        if cooldown:
            cooldown -= 1
        start = max(0, index - policy.persistence_epochs + 1)
        persistent = int(raw[start:index + 1].sum()) >= policy.persistence_epochs
        very_high = probability[index] >= min(0.95, threshold + 0.25)
        if cooldown == 0 and (persistent or very_high):
            alert[index] = True
            cooldown = policy.refractory_epochs
    return alert


def train_bundle(frame: pd.DataFrame, target_column: str, candidate_features: list[str], output: Path, policy: AlertPolicy, horizon_epochs: int) -> dict[str, Any]:
    np.random.seed(42)
    torch.manual_seed(42)
    valid = quality_mask(frame, horizon_epochs)
    data = frame.loc[valid].reset_index(drop=True)
    y = data[target_column].fillna(False).astype(bool).astype(int).to_numpy()
    groups = data["patient_id"].astype(str).to_numpy()
    features = select_core_features(data, candidate_features)
    if len(np.unique(groups)) < 5 or len(np.unique(y)) < 2:
        raise ValueError("序列模型至少需要 5 位病人且正負標籤皆存在")
    raw = data[features].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    scaler = StandardScaler()
    values = scaler.fit_transform(imputer.fit_transform(raw)).astype(np.float32)
    sequences = causal_sequences(values, groups, policy.window_epochs)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    stage1_oof = np.full(len(data), np.nan)
    stage2_oof = np.full(len(data), np.nan)
    device = torch.device("cpu")
    for train_idx, valid_idx in splitter.split(values, y, groups):
        xgb = XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.75, objective="binary:logistic", eval_metric="logloss", n_jobs=-1, random_state=42)
        xgb.fit(values[train_idx], y[train_idx])
        stage1_oof[valid_idx] = xgb.predict_proba(values[valid_idx])[:, 1]
        net = BiLSTM(values.shape[1]).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=1e-4)
        positives = max(int(y[train_idx].sum()), 1)
        pos_weight = torch.tensor([(len(train_idx) - positives) / positives], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        x_train = torch.from_numpy(sequences[train_idx])
        y_train = torch.from_numpy(y[train_idx].astype(np.float32))
        net.train()
        for _ in range(18):
            order = torch.randperm(len(train_idx))
            for offset in range(0, len(order), 256):
                batch = order[offset:offset + 256]
                optimizer.zero_grad()
                loss = loss_fn(net(x_train[batch]), y_train[batch])
                loss.backward()
                optimizer.step()
        net.eval()
        with torch.no_grad():
            stage2_oof[valid_idx] = torch.sigmoid(net(torch.from_numpy(sequences[valid_idx]))).numpy()
    # Select the fusion weight only from patient-grouped OOF predictions. This
    # lets Arousal and Apnea use different XGBoost/BiLSTM mixtures.
    best = None
    for xgb_weight in np.arange(0.0, 1.001, 0.05):
        combined = xgb_weight * stage1_oof + (1.0 - xgb_weight) * stage2_oof
        candidate_calibrator = LogisticRegression(C=1.0, max_iter=1000).fit(combined.reshape(-1, 1), y)
        candidate_probability = candidate_calibrator.predict_proba(combined.reshape(-1, 1))[:, 1]
        candidate_threshold = threshold_at_recall(y, candidate_probability, policy.recall_floor)
        candidate_prediction = candidate_probability >= candidate_threshold
        ctn, cfp, cfn, ctp = confusion_matrix(y, candidate_prediction, labels=[0, 1]).ravel()
        candidate = (float(ctn / max(ctn + cfp, 1)), float(precision_score(y, candidate_prediction, zero_division=0)), float(xgb_weight), candidate_calibrator, candidate_probability, float(candidate_threshold))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    _, _, xgb_weight, calibrator, calibrated, threshold = best
    # The calibrated threshold controls medical-safety sensitivity. Temporal
    # persistence is an episode-management signal and must not veto an isolated
    # high-risk epoch, because doing so materially increased false negatives.
    prediction = calibrated >= threshold
    episode_onset = apply_temporal_policy(calibrated, threshold, policy)
    xgb_final = XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.75, objective="binary:logistic", eval_metric="logloss", n_jobs=-1, random_state=42)
    xgb_final.fit(values, y)
    net_final = BiLSTM(values.shape[1]).to(device)
    optimizer = torch.optim.AdamW(net_final.parameters(), lr=0.002, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max((len(y) - y.sum()) / max(y.sum(), 1), 1.0)], dtype=torch.float32))
    tensors = torch.from_numpy(sequences)
    targets = torch.from_numpy(y.astype(np.float32))
    net_final.train()
    for _ in range(24):
        for offset in range(0, len(y), 256):
            batch = slice(offset, min(offset + 256, len(y)))
            optimizer.zero_grad(); loss = loss_fn(net_final(tensors[batch]), targets[batch]); loss.backward(); optimizer.step()
    net_final.eval()
    with torch.no_grad():
        final_p2 = torch.sigmoid(net_final(tensors)).numpy()
    final_p1 = xgb_final.predict_proba(values)[:, 1]
    selected_oof_raw = xgb_weight * stage1_oof + (1.0 - xgb_weight) * stage2_oof
    final_train_raw = xgb_weight * final_p1 + (1.0 - xgb_weight) * final_p2
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    summary = {"architecture": "XGBoost candidate + BiLSTM sequence refinement + Platt calibration", "target": target_column, "patients": int(len(np.unique(groups))), "rows_after_label_qc": int(len(y)), "rejected_label_rows": int((~valid).sum()), "features": features, "window_epochs": policy.window_epochs, "xgboost_weight": float(xgb_weight), "bilstm_weight": float(1.0 - xgb_weight), "threshold": float(threshold), "recall": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)), "precision": float(precision_score(y, prediction, zero_division=0)), "false_negative": int(fn), "false_positive": int(fp), "episode_onset_count": int(episode_onset.sum()), "calibration": "Platt sigmoid on patient-grouped OOF predictions", "alert_policy": "calibrated risk triggers alert; temporal persistence/refractory merges episodes but never vetoes risk", "clinical_validation": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": features, "imputer": imputer, "scaler": scaler, "xgb": xgb_final, "bilstm_state": net_final.state_dict(), "hidden": 24, "calibrator": calibrator, "xgb_weight": xgb_weight, "final_train_raw_sorted": np.sort(final_train_raw), "oof_raw_sorted": np.sort(selected_oof_raw), "policy": policy, "threshold": threshold, "summary": summary}, output)
    return summary


def predict_bundle(bundle_path: Path, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    bundle = joblib.load(bundle_path)
    values = bundle["scaler"].transform(bundle["imputer"].transform(frame.reindex(columns=bundle["features"]).replace([np.inf, -np.inf], np.nan))).astype(np.float32)
    patients = frame["patient_id"].astype(str).to_numpy()
    sequences = causal_sequences(values, patients, bundle["policy"].window_epochs)
    p1 = bundle["xgb"].predict_proba(values)[:, 1]
    net = BiLSTM(values.shape[1], bundle["hidden"]); net.load_state_dict(bundle["bilstm_state"]); net.eval()
    with torch.no_grad(): p2 = torch.sigmoid(net(torch.from_numpy(sequences))).numpy()
    weight = float(bundle.get("xgb_weight", 0.60))
    raw = weight * p1 + (1.0 - weight) * p2
    if "final_train_raw_sorted" in bundle and "oof_raw_sorted" in bundle:
        raw = np.interp(raw, bundle["final_train_raw_sorted"], bundle["oof_raw_sorted"])
    calibrated = bundle["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]
    alerts = calibrated >= float(bundle["threshold"])
    return calibrated, alerts, bundle["summary"]
