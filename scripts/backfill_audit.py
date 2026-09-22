import json
import logging
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR, PREDICTION_DIR
from src.pipeline.predict import PreMatchPredictor
from src.pipeline.feedback import PostMortemAudit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AuditBackfiller")

def run_backfill(season: int = 2024, week: int = 1):
    logger.info("Loading processed features for Season %d Week %d...", season, week)
    games_df = pd.read_parquet(PROCESSED_DATA_DIR / "games_features.parquet")
    props_df = pd.read_parquet(PROCESSED_DATA_DIR / "props_features.parquet")

    week_games = games_df[(games_df["season"] == season) & (games_df["week"] == week)].copy()
    week_props = props_df[(props_df["season"] == season) & (props_df["week"] == week)].copy()

    if week_games.empty:
        logger.error("No games found for Season %d Week %d", season, week)
        return

    # 1. Generate pre-match forecast ledger
    predictor = PreMatchPredictor()
    game_preds = predictor.generate_game_forecasts(week_games)

    # Generate synthetic prop lines based on rolling averages for active players
    prop_lines = {}
    sample_players = week_props.dropna(subset=["player_id"]).head(20)
    for _, p in sample_players.iterrows():
        p_id = p["player_id"]
        prop_lines[p_id] = {
            "passing_yards": float(round(p.get("passing_yards_roll_3", 225.5), 1)),
            "rushing_yards": float(round(p.get("rushing_yards_roll_3", 48.5), 1)),
            "receiving_yards": float(round(p.get("receiving_yards_roll_3", 52.5), 1))
        }

    prop_preds = predictor.generate_prop_forecasts(week_props, prop_lines)
    predictor.lock_pre_match_ledger(season, week, game_preds, prop_preds)

    # 2. Execute post-mortem audit to grade every pick
    auditor = PostMortemAudit(season=season, week=week)
    audit_results = auditor.audit_completed_week(week_games, week_props)
    
    logger.info("Successfully generated audit details: %s", audit_results)

if __name__ == "__main__":
    run_backfill(season=2024, week=1)
