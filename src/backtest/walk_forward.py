import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.config import PROP_TARGETS, RANDOM_STATE
from src.models.game_engine import GameOutcomeEngine
from src.models.prop_engine import PropPredictiveEngine
from src.backtest.calibrator import ProbabilityCalibrator
from src.backtest.evaluator import BacktestEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class WalkForwardBacktester:
    """Executes expanding-window out-of-sample backtesting without lookahead bias."""

    def __init__(self, games_df: pd.DataFrame, props_df: pd.DataFrame, test_seasons: List[int]):
        self.games = games_df.sort_values(["season", "week"]).reset_index(drop=True)
        self.props = props_df.sort_values(["season", "week"]).reset_index(drop=True)
        self.test_seasons = sorted(test_seasons)

    def run_game_backtest(self, edge_threshold: float = 1.5) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Walks forward season-by-season evaluating spread edge realization."""
        all_settled_bets = []

        for test_season in self.test_seasons:
            train_games = self.games[self.games["season"] < test_season].copy()
            eval_games = self.games[self.games["season"] == test_season].copy()

            if len(train_games) < 100 or len(eval_games) == 0:
                logger.warning("Insufficient data to backtest season %d. Skipping.", test_season)
                continue

            logger.info("Executing Walk-Forward on Season %d (Trained on < %d, N=%d)", 
                        test_season, test_season, len(train_games))

            engine = GameOutcomeEngine()
            engine.train(train_games)

            preds = engine.predict_game(eval_games)

            eval_sub = eval_games[["game_id", "margin", "game_total", "spread_line", "total_line"]].copy()
            preds = preds.merge(eval_sub, on=["game_id", "spread_line", "total_line"], how="inner")

            for _, row in preds.iterrows():
                # Spread Bet: Model projects home performance over market spread by threshold
                if row["spread_edge"] >= edge_threshold:
                    won = 1 if row["margin"] > row["spread_line"] else 0
                    push = 1 if row["margin"] == row["spread_line"] else 0
                    all_settled_bets.append({
                        "game_id": row["game_id"], "season": test_season, "week": row["week"],
                        "bet_type": "spread_home", "edge": row["spread_edge"],
                        "won": won, "push": push
                    })
                # Spread Bet: Model projects away performance under market spread by -threshold
                elif row["spread_edge"] <= -edge_threshold:
                    won = 1 if row["margin"] < row["spread_line"] else 0
                    push = 1 if row["margin"] == row["spread_line"] else 0
                    all_settled_bets.append({
                        "game_id": row["game_id"], "season": test_season, "week": row["week"],
                        "bet_type": "spread_away", "edge": abs(row["spread_edge"]),
                        "won": won, "push": push
                    })

        bets_df = pd.DataFrame(all_settled_bets)
        metrics = BacktestEvaluator.audit_betting_performance(bets_df)
        logger.info("Game Spread Backtest Summary: %s", metrics)
        return bets_df, metrics

    def run_props_backtest(self, prop_sample_size: int = 300) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Walks forward on player props to evaluate calibration and edge realization."""
        all_settled_props = []

        for test_season in self.test_seasons:
            train_props = self.props[self.props["season"] < test_season].copy()
            eval_props = self.props[self.props["season"] == test_season].copy()

            if len(train_props) < 500 or len(eval_props) == 0:
                continue

            logger.info("Executing Prop Backtest on Season %d", test_season)
            engine = PropPredictiveEngine()
            engine.train_all_props(train_props)

            test_targets = ["passing_yards", "rushing_yards", "receiving_yards"]
            for target in test_targets:
                sub = eval_props[eval_props[target] > 0].copy()
                if len(sub) > prop_sample_size:
                    sub = sub.sample(n=prop_sample_size, random_state=RANDOM_STATE)

                for _, row in sub.iterrows():
                    actual = row[target]
                    line = float(np.round(sub[target].median(), 1))
                    pred_res = engine.predict_prop(pd.DataFrame([row]), target, line)
                    over_prob = pred_res["over_prob"].iloc[0]

                    # Over edge: threshold > 57% probability (implied -110 line = 52.38%)
                    if over_prob >= 0.57:
                        won = 1 if actual > line else 0
                        push = 1 if actual == line else 0
                        all_settled_props.append({
                            "player_id": row["player_id"], "season": test_season,
                            "prop": target, "line": line, "actual": actual,
                            "over_prob": over_prob, "won": won, "push": push
                        })

        props_bet_df = pd.DataFrame(all_settled_props)
        metrics = BacktestEvaluator.audit_betting_performance(props_bet_df)
        logger.info("Prop Backtest Summary: %s", metrics)
        return props_bet_df, metrics
