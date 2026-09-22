import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.ingestion.nfl_loader import NFLDataLoader
from src.features.game_features import GameFeatureEngine
from src.features.prop_features import PropFeatureEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FeatureBuilder")

def run():
    logger.info("Loading cached raw datasets...")
    loader = NFLDataLoader(start_year=2020, end_year=2026)

    pbp = loader.load_play_by_play()
    schedules = loader.load_schedules()
    player_stats = loader.load_player_stats()
    snap_counts = loader.load_snap_counts()

    # 1. Generate Game Matrix
    logger.info("Generating Game Feature Matrix...")
    game_engine = GameFeatureEngine(pbp_df=pbp, schedules_df=schedules)
    game_matrix = game_engine.assemble_game_matrix()
    game_out_path = PROCESSED_DATA_DIR / "games_features.parquet"
    game_matrix.to_parquet(game_out_path, index=False, compression="snappy")
    logger.info("Saved game features to: %s", game_out_path)

    # 2. Generate Prop Matrix
    logger.info("Generating Player Prop Feature Matrix across 20 markets...")
    prop_engine = PropFeatureEngine(pbp_df=pbp, player_stats_df=player_stats, snap_df=snap_counts)
    prop_matrix = prop_engine.build_prop_matrix()
    prop_out_path = PROCESSED_DATA_DIR / "props_features.parquet"
    prop_matrix.to_parquet(prop_out_path, index=False, compression="snappy")
    logger.info("Saved prop features to: %s", prop_out_path)

    logger.info("Feature engineering pipeline completed successfully.")

if __name__ == "__main__":
    run()
