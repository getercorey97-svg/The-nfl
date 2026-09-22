import sys
import logging
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR, PREDICTION_DIR
from src.backtest.walk_forward import WalkForwardBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BacktestRunner")

def main():
    logger.info("Loading processed datasets for Walk-Forward Backtesting...")
    games_path = PROCESSED_DATA_DIR / "games_features.parquet"
    props_path = PROCESSED_DATA_DIR / "props_features.parquet"

    if not games_path.exists() or not props_path.exists():
        raise FileNotFoundError("Feature matrices missing. Run scripts/build_features.py first.")

    games_df = pd.read_parquet(games_path)
    props_df = pd.read_parquet(props_path)

    # Walk-forward evaluation across historical seasons (2022 through 2025)
    test_seasons = [2022, 2023, 2024, 2025]
    backtester = WalkForwardBacktester(games_df, props_df, test_seasons)

    # 1. Backtest Game Spreads
    logger.info("--- Executing Walk-Forward Game Backtest ---")
    game_bets, game_metrics = backtester.run_game_backtest(edge_threshold=2.0)
    game_bets_out = PREDICTION_DIR / "backtest_game_spreads.csv"
    game_bets.to_csv(game_bets_out, index=False)
    logger.info("Game Spread Results saved to %s", game_bets_out)
    print("\n================== GAME SPREAD AUDIT ==================")
    for k, v in game_metrics.items():
        print(f"{k.upper()}: {v}")
    print("=======================================================\n")

    # 2. Backtest Sample Player Props
    logger.info("--- Executing Walk-Forward Prop Backtest ---")
    prop_bets, prop_metrics = backtester.run_props_backtest(prop_sample_size=200)
    prop_bets_out = PREDICTION_DIR / "backtest_player_props.csv"
    prop_bets.to_csv(prop_bets_out, index=False)
    logger.info("Prop Results saved to %s", prop_bets_out)
    print("\n================== PLAYER PROP AUDIT ==================")
    for k, v in prop_metrics.items():
        print(f"{k.upper()}: {v}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
