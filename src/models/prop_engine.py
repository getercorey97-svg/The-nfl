import logging
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import norm, poisson
from src.config import MODEL_DIR, PROP_CONFIG, PROP_TARGETS, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PropPredictiveEngine:
    """Predictive engine modeling 20 player prop markets across specific distributions."""

    BASE_PROP_FEATURES = [
        "offense_pct", "defense_pct",
        "passing_attempts_roll_3", "passing_attempts_roll_5",
        "passing_yards_roll_3", "passing_yards_roll_5",
        "rushing_attempts_roll_3", "rushing_attempts_roll_5",
        "rushing_yards_roll_3", "rushing_yards_roll_5",
        "targets_roll_3", "targets_roll_5",
        "receptions_roll_3", "receptions_roll_5",
        "receiving_yards_roll_3", "receiving_yards_roll_5"
    ]

    def __init__(self):
        self.models: Dict[str, object] = {}
        self.rmse_residuals: Dict[str, float] = {}

    def _build_model_for_target(self, target: str):
        spec = PROP_CONFIG.get(target, {"type": "continuous", "distribution": "tweedie"})
        
        if spec["type"] == "binary":
            return LGBMClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=16,
                random_state=RANDOM_STATE,
                verbose=-1
            )
        elif spec["distribution"] == "poisson":
            return LGBMRegressor(
                objective="poisson",
                n_estimators=200,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=16,
                random_state=RANDOM_STATE,
                verbose=-1
            )
        elif spec["distribution"] == "tweedie":
            return LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=1.5,
                n_estimators=250,
                learning_rate=0.03,
                max_depth=5,
                num_leaves=24,
                random_state=RANDOM_STATE,
                verbose=-1
            )
        else: # Count (negative binomial proxy) / Extreme value
            return LGBMRegressor(
                objective="regression",
                n_estimators=200,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=18,
                random_state=RANDOM_STATE,
                verbose=-1
            )

    def train_all_props(self, prop_df: pd.DataFrame):
        """Fits dedicated estimators for all 20 prop target variables."""
        feature_cols = [col for col in self.BASE_PROP_FEATURES if col in prop_df.columns]
        X = prop_df[feature_cols].fillna(0.0)

        for target in PROP_TARGETS:
            if target not in prop_df.columns:
                logger.warning("Target %s not found in training frame; skipping.", target)
                continue

            y = prop_df[target].fillna(0.0).copy()
            spec = PROP_CONFIG.get(target, {})

            # Enforce non-negative bounds for distributions requiring y >= 0
            if spec.get("distribution") in ["tweedie", "poisson"]:
                y = np.maximum(y, 0.0)

            model = self._build_model_for_target(target)
            model.fit(X, y)
            self.models[target] = model

            # Calculate and store empirical residual standard error for continuous props
            if spec.get("type") == "continuous":
                preds = model.predict(X)
                residual_std = float(np.std(y - preds))
                self.rmse_residuals[target] = max(residual_std, 1.0)
            
            logger.info("Successfully trained model for prop: %s", target)

    def predict_prop(self, player_features: pd.DataFrame, prop_name: str, line: float) -> pd.DataFrame:
        """Calculates Expected Value (EV), Over Probability, and Under Probability given a betting line."""
        if prop_name not in self.models:
            raise ValueError(f"Model for prop '{prop_name}' has not been trained.")

        model = self.models[prop_name]
        spec = PROP_CONFIG[prop_name]
        feature_cols = [col for col in self.BASE_PROP_FEATURES if col in player_features.columns]
        X = player_features[feature_cols].fillna(0.0)

        results = player_features[["player_id", "player_name", "position", "team"]].copy() if "player_name" in player_features.columns else player_features[["player_id"]].copy()
        results["prop_name"] = prop_name
        results["market_line"] = line

        if spec["type"] == "binary":
            probs = model.predict_proba(X)[:, 1]
            results["expected_value"] = probs
            results["over_prob"] = probs
            results["under_prob"] = 1.0 - probs
        
        elif spec["distribution"] == "poisson":
            lambdas = np.maximum(model.predict(X), 0.001)
            results["expected_value"] = lambdas
            k = int(np.floor(line))
            under_probs = poisson.cdf(k, lambdas)
            results["under_prob"] = under_probs
            results["over_prob"] = 1.0 - under_probs

        else:
            means = np.maximum(model.predict(X), 0.0)
            results["expected_value"] = means
            sigma = self.rmse_residuals.get(prop_name, 10.0)
            z = (line - means) / sigma
            under_probs = norm.cdf(z)
            results["under_prob"] = under_probs
            results["over_prob"] = 1.0 - under_probs

        return results

    def save(self, filepath: str = None):
        target_path = filepath or (MODEL_DIR / "prop_engine.joblib")
        joblib.dump(self, target_path)
        logger.info("Saved PropPredictiveEngine to: %s", target_path)

    @staticmethod
    def load(filepath: str = None) -> "PropPredictiveEngine":
        target_path = filepath or (MODEL_DIR / "prop_engine.joblib")
        return joblib.load(target_path)
