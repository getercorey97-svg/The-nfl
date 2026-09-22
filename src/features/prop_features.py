import logging
from typing import List
import numpy as np
import pandas as pd
from src.config import PROP_TARGETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PropFeatureEngine:
    """Builds leak-free rolling features and defensive matchup metrics for 20 prop markets."""

    def __init__(self, pbp_df: pd.DataFrame, player_stats_df: pd.DataFrame, snap_df: pd.DataFrame):
        self.pbp = pbp_df.copy()
        self.stats = player_stats_df.copy()
        self.snaps = snap_df.copy()

    def extract_play_level_extremes(self) -> pd.DataFrame:
        """Calculates longest run, longest completion, and defensive events per player-week."""
        # Longest rush per player per game
        longest_rush = self.pbp[self.pbp["play_type"] == "run"].groupby(
            ["game_id", "season", "week", "rusher_player_id"]
        )["yards_gained"].max().reset_index().rename(
            columns={"rusher_player_id": "player_id", "yards_gained": "rushing_longest"}
        )

        # Longest pass per passer per game
        longest_pass = self.pbp[self.pbp["play_type"] == "pass"].groupby(
            ["game_id", "season", "week", "passer_player_id"]
        )["yards_gained"].max().reset_index().rename(
            columns={"passer_player_id": "player_id", "yards_gained": "passing_longest"}
        )

        # Longest reception per receiver per game
        longest_rec = self.pbp[(self.pbp["play_type"] == "pass") & (self.pbp["complete_pass"] == 1)].groupby(
            ["game_id", "season", "week", "receiver_player_id"]
        )["yards_gained"].max().reset_index().rename(
            columns={"receiver_player_id": "player_id", "yards_gained": "receiving_longest"}
        )

        # Defensive stats: Tackles and Sacks from PBP
        sacks = self.pbp.dropna(subset=["sack_player_id"]).groupby(
            ["game_id", "season", "week", "sack_player_id"]
        )["sack"].sum().reset_index().rename(
            columns={"sack_player_id": "player_id", "sack": "defensive_sacks"}
        )

        solo = self.pbp.dropna(subset=["solo_tackle_1_player_id"]).groupby(
            ["game_id", "season", "week", "solo_tackle_1_player_id"]
        )["solo_tackle"].sum().reset_index().rename(
            columns={"solo_tackle_1_player_id": "player_id", "solo_tackle": "solo"}
        )

        ast = self.pbp.dropna(subset=["assist_tackle_1_player_id"]).groupby(
            ["game_id", "season", "week", "assist_tackle_1_player_id"]
        )["assist_tackle"].sum().reset_index().rename(
            columns={"assist_tackle_1_player_id": "player_id", "assist_tackle": "ast"}
        )

        tackles = pd.merge(solo, ast, on=["game_id", "season", "week", "player_id"], how="outer").fillna(0)
        tackles["defensive_tackles_combined"] = tackles["solo"] + tackles["ast"]
        tackles.drop(columns=["solo", "ast"], inplace=True)

        return longest_rush, longest_pass, longest_rec, sacks, tackles

    def build_prop_matrix(self) -> pd.DataFrame:
        """Consolidates stats, snap shares, and rolling opportunities."""
        df = self.stats.copy()

        # Standardize target names matching config
        column_mapping = {
            "attempts": "passing_attempts",
            "completions": "passing_completions",
            "passing_tds": "passing_touchdowns",
            "interceptions": "passing_interceptions",
            "carries": "rushing_attempts",
            "rushing_tds": "rushing_touchdowns",
            "receiving_tds": "receiving_touchdowns",
            "fg_made": "field_goals_made",
            "pat_made": "extra_points_made"
        }
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

        # Merge extracted play-level targets
        l_rush, l_pass, l_rec, sacks, tackles = self.extract_play_level_extremes()
        for ext_df in [l_rush, l_pass, l_rec, sacks, tackles]:
            df = df.merge(ext_df, on=["game_id", "season", "week", "player_id"], how="left")

        # Anytime Touchdown Target
        rush_td = df["rushing_touchdowns"] if "rushing_touchdowns" in df.columns else 0
        rec_td = df["receiving_touchdowns"] if "receiving_touchdowns" in df.columns else 0
        df["anytime_td"] = ((rush_td + rec_td) > 0).astype(int)

        # Fill missing targets with zero
        for target in PROP_TARGETS:
            if target not in df.columns:
                df[target] = 0.0
            else:
                df[target] = df[target].fillna(0.0)

        # Attach snap percentages
        if not self.snaps.empty and "pfr_player_id" in self.snaps.columns:
            snap_sub = self.snaps[["game_id", "pfr_player_id", "offense_pct", "defense_pct"]].rename(
                columns={"pfr_player_id": "player_id"}
            )
            df = df.merge(snap_sub, on=["game_id", "player_id"], how="left")
            df["offense_pct"] = df["offense_pct"].fillna(0.0)
            df["defense_pct"] = df["defense_pct"].fillna(0.0)
        else:
            df["offense_pct"] = 0.0
            df["defense_pct"] = 0.0

        # Construct Leak-Free Rolling Features per Player
        df.sort_values(["player_id", "season", "week"], inplace=True)
        rolling_cols = [
            "passing_attempts", "passing_yards", "rushing_attempts", "rushing_yards",
            "targets", "receptions", "receiving_yards", "offense_pct"
        ]

        feature_dfs = []
        for player_id, group in df.groupby("player_id"):
            group = group.copy()
            for w in [3, 5]:
                shifted = group[rolling_cols].shift(1)
                roll = shifted.rolling(window=w, min_periods=1).mean()
                roll.columns = [f"{col}_roll_{w}" for col in rolling_cols]
                group = pd.concat([group, roll], axis=1)
            feature_dfs.append(group)

        full_df = pd.concat(feature_dfs, ignore_index=True)
        logger.info("Constructed Prop Feature Matrix: %s rows, %s columns", full_df.shape[0], full_df.shape[1])
        return full_df
