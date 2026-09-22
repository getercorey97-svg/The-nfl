import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from src.config import MODEL_DIR, PREDICTION_DIR, PROP_TARGETS
from src.models.game_engine import GameOutcomeEngine
from src.models.prop_engine import PropPredictiveEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PreMatchPredictor")

class PreMatchPredictor:
    """Generates immutable pre-match forecast records for upcoming fixtures."""

    def __init__(self):
        self.game_engine = GameOutcomeEngine.load()
        self.prop_engine = PropPredictiveEngine.load()

    def generate_game_forecasts(self, upcoming_games_features: pd.DataFrame) -> pd.DataFrame:
        """Calculates expected margins, win probabilities, and market edges for upcoming games."""
        logger.info("Generating pre-match forecasts for %d upcoming games.", len(upcoming_games_features))
        preds = self.game_engine.predict_game(upcoming_games_features)
        preds["forecast_timestamp"] = datetime.now(timezone.utc).isoformat()
        return preds

    def generate_prop_forecasts(self, active_player_features: pd.DataFrame, prop_lines: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Projects player prop expectations against active market lines.
        prop_lines format: {player_id: {prop_name: market_line}}
        """
        all_prop_preds = []
        for player_id, props in prop_lines.items():
            player_row = active_player_features[active_player_features["player_id"] == player_id]
            if player_row.empty:
                continue

            for prop_name, line in props.items():
                if prop_name not in PROP_TARGETS:
                    continue
                pred = self.prop_engine.predict_prop(player_row, prop_name, line)
                all_prop_preds.append(pred)

        if not all_prop_preds:
            return pd.DataFrame()

        prop_forecasts = pd.concat(all_prop_preds, ignore_index=True)
        prop_forecasts["forecast_timestamp"] = datetime.now(timezone.utc).isoformat()
        return prop_forecasts

    def lock_pre_match_ledger(self, season: int, week: int, game_preds: pd.DataFrame, prop_preds: pd.DataFrame):
        """Serializes pre-match forecasts into an immutable audit file prior to kickoff."""
        ledger_path = PREDICTION_DIR / f"forecast_season_{season}_week_{week}.json"
        
        ledger_data = {
            "season": season,
            "week": week,
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "games": game_preds.to_dict(orient="records"),
            "props": prop_preds.to_dict(orient="records") if not prop_preds.empty else []
        }

        with open(ledger_path, "w") as f:
            json.dump(ledger_data, f, indent=2)

        logger.info("Pre-match forecast ledger successfully locked: %s", ledger_path)
