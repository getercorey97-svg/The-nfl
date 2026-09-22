import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error
from src.config import PREDICTION_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PostMortemAudit")

class PostMortemAudit:
    """Conducts direct, simulation-free comparisons between pre-match forecasts and official empirical outcomes."""

    def __init__(self, season: int, week: int):
        self.season = season
        self.week = week
        self.ledger_path = PREDICTION_DIR / f"forecast_season_{season}_week_{week}.json"
        self.audit_log_path = PREDICTION_DIR / "post_mortem_audit_ledger.csv"

    def load_forecast_ledger(self) -> Dict:
        if not self.ledger_path.exists():
            raise FileNotFoundError(f"No locked pre-match forecast found for Season {self.season}, Week {self.week}.")
        with open(self.ledger_path, "r") as f:
            return json.load(f)

    def audit_completed_week(self, completed_games_df: pd.DataFrame, completed_player_stats_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluates model performance against empirical outcomes without simulations."""
        ledger = self.load_forecast_ledger()
        forecasted_games = pd.DataFrame(ledger.get("games", []))
        forecasted_props = pd.DataFrame(ledger.get("props", []))

        audit_results = {
            "season": self.season,
            "week": self.week,
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_games_audited": 0,
            "spread_mae": 0.0,
            "total_mae": 0.0,
            "moneyline_brier": 0.0,
            "moneyline_accuracy": 0.0,
            "spread_ats_win_rate": 0.0,
            "total_props_audited": 0,
            "props_over_under_accuracy": 0.0
        }

        # 1. Audit Game Projections
        if not forecasted_games.empty and not completed_games_df.empty:
            merged_games = forecasted_games.merge(
                completed_games_df[["game_id", "home_score", "away_score"]],
                on="game_id",
                how="inner"
            )

            if not merged_games.empty:
                merged_games["actual_margin"] = merged_games["home_score"] - merged_games["away_score"]
                merged_games["actual_total"] = merged_games["home_score"] + merged_games["away_score"]
                merged_games["actual_home_win"] = (merged_games["actual_margin"] > 0).astype(int)

                # Spread edge realization
                merged_games["home_covered"] = (merged_games["actual_margin"] > merged_games["spread_line"]).astype(int)
                merged_games["ats_pick"] = np.where(merged_games["spread_edge"] > 0, 1, 0)
                merged_games["ats_won"] = (merged_games["home_covered"] == merged_games["ats_pick"]).astype(int)

                audit_results["total_games_audited"] = len(merged_games)
                audit_results["spread_mae"] = float(mean_absolute_error(merged_games["actual_margin"], merged_games["pred_margin"]))
                audit_results["total_mae"] = float(mean_absolute_error(merged_games["actual_total"], merged_games["pred_total"]))
                audit_results["moneyline_brier"] = float(brier_score_loss(merged_games["actual_home_win"], merged_games["home_win_prob"]))
                audit_results["moneyline_accuracy"] = float(((merged_games["home_win_prob"] >= 0.50).astype(int) == merged_games["actual_home_win"]).mean() * 100.0)
                audit_results["spread_ats_win_rate"] = float(merged_games["ats_won"].mean() * 100.0)

        # 2. Audit Player Prop Projections
        if not forecasted_props.empty and not completed_player_stats_df.empty:
            prop_evals = []
            for _, prop_row in forecasted_props.iterrows():
                p_id = prop_row["player_id"]
                p_target = prop_row["prop_name"]
                line = prop_row["market_line"]

                match = completed_player_stats_df[
                    (completed_player_stats_df["player_id"] == p_id) &
                    (completed_player_stats_df["week"] == self.week)
                ]

                if not match.empty and p_target in match.columns:
                    actual_stat = match[p_target].iloc[0]
                    actual_over = 1 if actual_stat > line else 0
                    predicted_over = 1 if prop_row["over_prob"] >= 0.50 else 0
                    prop_evals.append(1 if actual_over == predicted_over else 0)

            if prop_evals:
                audit_results["total_props_audited"] = len(prop_evals)
                audit_results["props_over_under_accuracy"] = float(np.mean(prop_evals) * 100.0)

        # 3. Append to Historical Audit Ledger
        audit_row = pd.DataFrame([audit_results])
        if self.audit_log_path.exists():
            audit_row.to_csv(self.audit_log_path, mode="a", header=False, index=False)
        else:
            audit_row.to_csv(self.audit_log_path, mode="w", header=True, index=False)

        logger.info("Completed simulation-free post-mortem audit for Week %d. ATS Win Rate: %.2f%%, ML Brier: %.4f",
                    self.week, audit_results["spread_ats_win_rate"], audit_results["moneyline_brier"])
        return audit_results
