import logging
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error
from src.config import MODEL_DIR, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class GameOutcomeEngine:
    """Predictive engine for Spreads (Margin), Totals, and Moneylines (Win Probabilities)."""

    FEATURE_COLS = [
        "spread_line", "total_line",
        "home_off_epa_per_play_roll_3", "home_off_epa_per_play_roll_6",
        "home_def_epa_per_play_roll_3", "home_def_epa_per_play_roll_6",
        "home_off_pass_epa_mean_roll_3", "home_off_rush_epa_mean_roll_3",
        "away_off_epa_per_play_roll_3", "away_off_epa_per_play_roll_6",
        "away_def_epa_per_play_roll_3", "away_def_epa_per_play_roll_6",
        "away_off_pass_epa_mean_roll_3", "away_off_rush_epa_mean_roll_3",
        "diff_epa_roll_3", "diff_def_epa_roll_3",
        "diff_epa_roll_6", "diff_def_epa_roll_6"
    ]

    def __init__(self):
        # Margin regressor (Home Score - Away Score)
        self.spread_model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=24,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        # Total regressor (Home Score + Away Score)
        self.total_model = LGBMRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        # Moneyline classifier (Home Win = 1)
        self.moneyline_model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=4,
            num_leaves=16,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        self.is_trained = False

    def train(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """Trains spread, total, and moneyline models on historical match data."""
        # Isolate completed games with scores
        df = train_df.dropna(subset=["home_score", "away_score", "spread_line", "total_line"]).copy()
        
        available_features = [col for col in self.FEATURE_COLS if col in df.columns]
        X = df[available_features].fillna(0.0)
        y_margin = df["margin"]
        y_total = df["game_total"]
        y_win = df["home_win"]

        logger.info("Training Game Engines on %d games with %d features.", len(X), len(available_features))
        self.spread_model.fit(X, y_margin)
        self.total_model.fit(X, y_total)
        self.moneyline_model.fit(X, y_win)
        self.is_trained = True

        # Training diagnostic baselines
        pred_margins = self.spread_model.predict(X)
        pred_totals = self.total_model.predict(X)
        pred_win_probs = self.moneyline_model.predict_proba(X)[:, 1]

        metrics = {
            "spread_mae": float(mean_absolute_error(y_margin, pred_margins)),
            "total_mae": float(mean_absolute_error(y_total, pred_totals)),
            "moneyline_brier": float(brier_score_loss(y_win, pred_win_probs))
        }
        logger.info("Game Engine In-Sample Metrics: %s", metrics)
        return metrics

    def predict_game(self, game_features: pd.DataFrame) -> pd.DataFrame:
        """Outputs expected margin, projected total, home win probability, and spread edges."""
        if not self.is_trained:
            raise RuntimeError("GameOutcomeEngine must be trained before predicting.")

        available_features = [col for col in self.FEATURE_COLS if col in game_features.columns]
        X = game_features[available_features].fillna(0.0)

        preds = game_features[["game_id", "season", "week", "home_team", "away_team", "spread_line", "total_line"]].copy()
        preds["pred_margin"] = self.spread_model.predict(X)
        preds["pred_total"] = self.total_model.predict(X)
        preds["home_win_prob"] = self.moneyline_model.predict_proba(X)[:, 1]
        preds["away_win_prob"] = 1.0 - preds["home_win_prob"]

        # Calculate betting edges against market closing lines
        # Market spread convention: negative spread implies home favorite
        preds["spread_edge"] = preds["pred_margin"] + preds["spread_line"]
        preds["total_edge"] = preds["pred_total"] - preds["total_line"]

        return preds

    def save(self, filepath: str = None):
        target_path = filepath or (MODEL_DIR / "game_outcome_engine.joblib")
        joblib.dump(self, target_path)
        logger.info("Saved GameOutcomeEngine to: %s", target_path)

    @staticmethod
    def load(filepath: str = None) -> "GameOutcomeEngine":
        target_path = filepath or (MODEL_DIR / "game_outcome_engine.joblib")
        return joblib.load(target_path)
