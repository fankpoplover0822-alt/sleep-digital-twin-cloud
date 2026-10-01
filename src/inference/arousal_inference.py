from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.two_stage_sequence import predict_bundle


class ArousalInferenceEngine:
    def __init__(
        self,
        model_file: str | Path,
        feature_config_file: str | Path,
        threshold_file: str | Path,
    ) -> None:
        self.model_file = Path(model_file)
        self.feature_config_file = Path(
            feature_config_file
        )
        self.threshold_file = Path(
            threshold_file
        )

        if not self.model_file.exists():
            raise FileNotFoundError(
                f"找不到模型：{self.model_file}"
            )

        if not self.feature_config_file.exists():
            raise FileNotFoundError(
                "找不到特徵設定檔："
                f"{self.feature_config_file}"
            )

        if not self.threshold_file.exists():
            raise FileNotFoundError(
                "找不到門檻設定檔："
                f"{self.threshold_file}"
            )

        self.model = joblib.load(
            self.model_file
        )

        self.feature_config = (
            self._load_json(
                self.feature_config_file
            )
        )

        self.threshold_config = (
            self._load_json(
                self.threshold_file
            )
        )

        self.numeric_columns = list(
            self.feature_config.get(
                "numeric_columns",
                [],
            )
        )

        self.categorical_columns = list(
            self.feature_config.get(
                "categorical_columns",
                [],
            )
        )

        self.target_column = (
            self.feature_config.get(
                "target_column",
                "predict_arousal_next_30s",
            )
        )

        self.feature_columns = (
            self.numeric_columns
            + self.categorical_columns
        )

        if not self.feature_columns:
            raise RuntimeError(
                "feature_columns.json 中沒有"
                "任何模型特徵。"
            )

        self.alert_threshold = (
            self._extract_threshold()
        )
        self.sequence_bundle_file = self.model_file.parent / "two_stage_sequence.joblib"

    @staticmethod
    def _load_json(
        file_path: Path,
    ) -> dict[str, Any]:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                f"JSON 格式錯誤："
                f"{file_path}"
            )

        return data

    def _extract_threshold(
        self,
    ) -> float:
        """
        優先使用最佳 Balanced Accuracy 門檻，
        再依序嘗試最佳 F1 與高敏感度門檻。
        """
        candidates = [
            "continual_learning",
            "best_balanced_accuracy",
            "best_f1",
            (
                "high_sensitivity_recall_"
                "at_least_0_70"
            ),
        ]

        for key in candidates:
            section = (
                self.threshold_config.get(
                    key
                )
            )

            if not isinstance(
                section,
                dict,
            ):
                continue

            threshold = section.get(
                "threshold"
            )

            if threshold is None:
                continue

            try:
                threshold_value = float(
                    threshold
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                0.0
                <= threshold_value
                <= 1.0
            ):
                return threshold_value

        return 0.35

    def prepare_features(
        self,
        model_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        將新患者資料整理成模型訓練時相同的欄位順序。

        缺少欄位自動補 NaN。
        多餘欄位不送入模型。
        使用 reindex 一次建立 DataFrame，
        避免 DataFrame fragmentation 警告。
        """
        if model_data.empty:
            raise RuntimeError(
                "推論資料為空。"
            )

        missing_columns = [
            column
            for column in self.feature_columns
            if column
            not in model_data.columns
        ]

        prepared = model_data.reindex(
            columns=self.feature_columns
        ).copy()

        for column in self.numeric_columns:
            prepared[column] = (
                pd.to_numeric(
                    prepared[column],
                    errors="coerce",
                )
            )

        for column in (
            self.categorical_columns
        ):
            prepared[column] = (
                prepared[column]
                .astype("object")
            )

        prepared.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        if missing_columns:
            print(
                "警告：新患者資料缺少 "
                f"{len(missing_columns)} "
                "個模型欄位，已補為 NaN。"
            )

            print("缺少欄位：")

            for column in missing_columns:
                print(f"- {column}")

        return prepared

    def predict(
        self,
        model_data: pd.DataFrame,
    ) -> pd.DataFrame:
        prepared = self.prepare_features(
            model_data
        )

        if not hasattr(
            self.model,
            "predict_proba",
        ):
            raise RuntimeError(
                "載入的模型不支援 "
                "predict_proba()。"
            )

        sequence_summary = None
        temporal_alerts = None
        use_sequence_bundle = False
        if self.sequence_bundle_file.exists():
            challenger = joblib.load(self.sequence_bundle_file).get("summary", {})
            champion = self.threshold_config.get("continual_learning", {})
            use_sequence_bundle = (
                float(challenger.get("recall", 0.0)) >= float(self.threshold_config.get("high_sensitivity_recall_target", 0.90))
                and float(challenger.get("specificity", 0.0)) >= float(champion.get("specificity", 1.0))
                and float(challenger.get("precision", 0.0)) >= float(champion.get("precision", 1.0))
            )
        if use_sequence_bundle:
            probabilities, temporal_alerts, sequence_summary = predict_bundle(
                self.sequence_bundle_file, model_data
            )
            self.alert_threshold = float(sequence_summary["threshold"])
        else:
            probabilities = self.model.predict_proba(prepared)[:, 1]

        quality_valid = pd.Series(True, index=model_data.index, dtype=bool)
        for quality_column in (
            "quality_core_features_valid",
            "usable_for_core_respiratory_model",
        ):
            if quality_column in model_data.columns:
                values = model_data[quality_column]
                if values.dtype == object:
                    values = values.astype(str).str.strip().str.lower().isin(
                        {"true", "1", "yes"}
                    )
                else:
                    values = values.fillna(False).astype(bool)
                quality_valid &= values

        risk_positive = probabilities >= self.alert_threshold
        candidate_alerts = risk_positive if temporal_alerts is None else temporal_alerts
        alerts = candidate_alerts & quality_valid.to_numpy()

        result = self._build_output_base(
            model_data
        )

        result[
            "arousal_next_30s_probability"
        ] = probabilities

        result[
            "arousal_alert_threshold"
        ] = self.alert_threshold

        result[
            "arousal_next_30s_alert"
        ] = alerts

        result["arousal_signal_quality_valid"] = quality_valid.to_numpy()
        result["arousal_sensor_check_alert"] = (~quality_valid).to_numpy()
        result["arousal_high_risk_but_signal_invalid"] = (
            risk_positive & (~quality_valid.to_numpy())
        )

        result[
            "arousal_risk_level"
        ] = pd.cut(
            probabilities,
            bins=[
                -np.inf,
                0.20,
                self.alert_threshold,
                0.70,
                np.inf,
            ],
            labels=[
                "LOW",
                "MODERATE",
                "HIGH",
                "VERY_HIGH",
            ],
            include_lowest=True,
        ).astype(str)

        result[
            "model_target"
        ] = self.target_column

        result["arousal_alert_logic"] = (
            "two_stage_xgboost_bilstm_calibrated"
            if sequence_summary is not None
            else "random_forest_champion_sequence_challenger_not_promoted"
        )

        return result

    @staticmethod
    def _build_output_base(
        model_data: pd.DataFrame,
    ) -> pd.DataFrame:
        preferred_columns = [
            "patient_id",
            "epoch_index",
            "original_epoch",
            "start_time",
            "end_time",
            "stage",
            "edf_start_seconds",
            "edf_end_seconds",
            "quality_low_spo2_validity",
            "quality_missing_flow_rate",
            (
                "quality_missing_"
                "thorax_abdomen_correlation"
            ),
            "quality_missing_heart_rate",
            "quality_issue_count",
            "quality_any_issue",
            "quality_core_features_valid",
            (
                "usable_for_core_"
                "respiratory_model"
            ),
        ]

        available_columns = [
            column
            for column in preferred_columns
            if column in model_data.columns
        ]

        return model_data[
            available_columns
        ].copy()

    def print_summary(
        self,
        predictions: pd.DataFrame,
    ) -> None:
        if predictions.empty:
            print(
                "沒有推論結果。"
            )
            return

        alert_count = int(
            predictions[
                "arousal_next_30s_alert"
            ].sum()
        )

        print("=" * 80)
        print("Arousal 推論摘要")
        print("=" * 80)

        print(
            f"Epoch 數量："
            f"{len(predictions)}"
        )

        print(
            f"警報門檻："
            f"{self.alert_threshold:.3f}"
        )

        print(
            f"警報 Epoch："
            f"{alert_count}"
        )

        print(
            f"警報比例："
            f"{alert_count / len(predictions):.2%}"
        )

        print(
            "平均風險："
            f"{predictions['arousal_next_30s_probability'].mean():.4f}"
        )

        print(
            "最大風險："
            f"{predictions['arousal_next_30s_probability'].max():.4f}"
        )

        if (
            "quality_core_features_valid"
            in predictions.columns
        ):
            valid_count = int(
                predictions[
                    "quality_core_features_valid"
                ]
                .astype(bool)
                .sum()
            )

            print(
                "核心特徵品質合格："
                f"{valid_count}"
                f"/{len(predictions)}"
            )

        print("\n風險等級數量：")

        print(
            predictions[
                "arousal_risk_level"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

        print("=" * 80)
