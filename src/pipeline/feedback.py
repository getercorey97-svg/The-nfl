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
    """Simulation-free evaluation comparing individual forecasts against empirical outcomes."""

    def __init__(self, season: int, week: int):
        self.season = season
        self.week = week
        self.ledger_path = PREDICTION_DIR / f"forecast_season_{season}_week_{week}.json"
        self.summary_log_path = PREDICTION_DIR / "post_mortem_audit_ledger.csv"
        self.details_log_path = PREDICTION_DIR / "prediction_audit_details.csv"

    def load_forecast_ledger(self) -> Dict:
        if not self.ledger_path.exists():
            raise FileNotFoundError(f"No locked pre-match forecast found for Season {self.season}, Week {self.week}.")
        with open(self.ledger_path, "r") as f:
            return json.load(f)

    def audit_completed_week(self, completed_games_df: pd.DataFrame, completed_player_stats_df: pd.DataFrame) -> Dict[str, float]:
        ledger = self.load_forecast_ledger()
        forecasted_games = pd.DataFrame(ledger.get("games", []))
        forecasted_props = pd.DataFrame(ledger.get("props", []))

        detailed_records = []
        audit_summary = {
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

        # 1. Audit Individual Game Projections
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

                for _, g in merged_games.iterrows():
                    actual_margin = g["actual_margin"]
                    spread_line = g["spread_line"]
                    spread_edge = g["spread_edge"]

                    # ATS Evaluation
                    if spread_edge != 0:
                        pick_team = g["home_team"] if spread_edge > 0 else g["away_team"]
                        home_covered = actual_margin > spread_line
                        won = 1 if ((spread_edge > 0 and home_covered) or (spread_edge < 0 and not home_covered)) else 0
                        push = 1 if actual_margin == spread_line else 0
                        status = "push" if push == 1 else ("won" if won == 1 else "lost")

                        detailed_records.append({
                            "season": self.season,
                            "week": self.week,
                            "category": "spread",
                            "item": f"{g['away_team']} @ {g['home_team']}",
                            "pick": f"{pick_team} (Edge: {spread_edge:+.1f})",
                            "projected": round(float(g["pred_margin"]), 1),
                            "line": float(spread_line),
                            "actual": float(actual_margin),
                            "error": round(abs(float(g["pred_margin"]) - actual_margin), 1),
                            "status": status
                        })

                    # Total Over/Under Evaluation
                    total_edge = g["total_edge"]
                    if abs(total_edge) >= 1.0:
                        pick_type = "OVER" if total_edge > 0 else "UNDER"
                        tot_won = 1 if ((total_edge > 0 and g["actual_total"] > g["total_line"]) or (total_edge < 0 and g["actual_total"] < g["total_line"])) else 0
                        tot_push = 1 if g["actual_total"] == g["total_line"] else 0
                        tot_status = "push" if tot_push == 1 else ("won" if tot_won == 1 else "lost")

                        detailed_records.append({
                            "season": self.season,
                            "week": self.week,
                            "category": "total",
                            "item": f"{g['away_team']} @ {g['home_team']}",
                            "pick": f"{pick_type} {g['total_line']}",
                            "projected": round(float(g["pred_total"]), 1),
                            "line": float(g["total_line"]),
                            "actual": float(g["actual_total"]),
                            "error": round(abs(float(g["pred_total"]) - float(g["actual_total"])), 1),
                            "status": tot_status
                        })

                # Compute Aggregates
                audit_summary["total_games_audited"] = len(merged_games)
                audit_summary["spread_mae"] = float(mean_absolute_error(merged_games["actual_margin"], merged_games["pred_margin"]))
                audit_summary["total_mae"] = float(mean_absolute_error(merged_games["actual_total"], merged_games["pred_total"]))
                audit_summary["moneyline_brier"] = float(brier_score_loss(merged_games["actual_home_win"], merged_games["home_win_prob"]))
                audit_summary["moneyline_accuracy"] = float(((merged_games["home_win_prob"] >= 0.50).astype(int) == merged_games["actual_home_win"]).mean() * 100.0)

                ats_records = [r for r in detailed_records if r["category"] == "spread"]
                if ats_records:
                    audit_summary["spread_ats_win_rate"] = float(np.mean([1 if r["status"] == "won" else 0 for r in ats_records]) * 100.0)

        # 2. Audit Individual Player Prop Projections
        if not forecasted_props.empty and not completed_player_stats_df.empty:
            for _, prop in forecasted_props.iterrows():
                p_id = prop["player_id"]
                p_target = prop["prop_name"]
                line = prop["market_line"]

                match = completed_player_stats_df[
                    (completed_player_stats_df["player_id"] == p_id) &
                    (completed_player_stats_df["week"] == self.week)
                ]

                if not match.empty and p_target in match.columns:
                    actual_stat = float(match[p_target].iloc[0])
                    over_prob = float(prop["over_prob"])
                    call = "OVER" if over_prob >= 0.50 else "UNDER"
                    
                    won = 1 if ((call == "OVER" and actual_stat > line) or (call == "UNDER" and actual_stat < line)) else 0
                    push = 1 if actual_stat == line else 0
                    status = "push" if push == 1 else ("won" if won == 1 else "lost")

                    player_display = prop.get("player_name", p_id)
                    detailed_records.append({
                        "season": self.season,
                        "week": self.week,
                        "category": "prop",
                        "item": f"{player_display} ({p_target})",
                        "pick": f"{call} {line} (P={over_prob:.2f})",
                        "projected": round(float(prop.get("expected_value", 0.0)), 1),
                        "line": float(line),
                        "actual": actual_stat,
                        "error": round(abs(float(prop.get("expected_value", 0.0)) - actual_stat), 1),
                        "status": status
                    })

            prop_records = [r for r in detailed_records if r["category"] == "prop"]
            if prop_records:
                audit_summary["total_props_audited"] = len(prop_records)
                audit_summary["props_over_under_accuracy"] = float(np.mean([1 if r["status"] == "won" else 0 for r in prop_records]) * 100.0)

        # 3. Persist Logs
        summary_df = pd.DataFrame([audit_summary])
        if self.summary_log_path.exists():
            summary_df.to_csv(self.summary_log_path, mode="a", header=False, index=False)
        else:
            summary_df.to_csv(self.summary_log_path, mode="w", header=True, index=False)

        if detailed_records:
            details_df = pd.DataFrame(detailed_records)
            if self.details_log_path.exists():
                details_df.to_csv(self.details_log_path, mode="a", header=False, index=False)
            else:
                details_df.to_csv(self.details_log_path, mode="w", header=True, index=False)

        return audit_summary
