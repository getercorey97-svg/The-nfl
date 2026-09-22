import sys
import logging
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR, MODEL_DIR
from src.models.game_engine import GameOutcomeEngine
from src.models.prop_engine import PropPredictiveEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelTrainer")

def run_training():
    logger.info("Loading processed training datasets...")
    game_file = PROCESSED_DATA_DIR / "games_features.parquet"
    prop_file = PROCESSED_DATA_DIR / "props_features.parquet"

    if not game_file.exists() or not prop_file.exists():
        raise FileNotFoundError("Processed feature matrices not found. Execute 'python scripts/build_features.py' first.")

    games_df = pd.read_parquet(game_file)
    props_df = pd.read_parquet(prop_file)

    # 1. Train Game Outcome Engine
    logger.info("--- Initializing Game Outcome Engine Training ---")
    game_engine = GameOutcomeEngine()
    game_metrics = game_engine.train(games_df)
    game_engine.save()
    logger.info("Game Outcome Models trained and serialized successfully.")

    # 2. Train 20-Prop Distributional Engine
    logger.info("--- Initializing 20-Prop Distributional Models Training ---")
    prop_engine = PropPredictiveEngine()
    prop_engine.train_all_props(props_df)
    prop_engine.save()
    logger.info("All 20 Player Prop Models trained and serialized successfully.")

if __name__ == "__main__":
    run_training()
