import logging
from typing import List
import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class GameFeatureEngine:
    """Extracts advanced team-level rolling metrics and opponent-adjusted baselines."""

    def __init__(self, pbp_df: pd.DataFrame, schedules_df: pd.DataFrame):
        self.pbp = pbp_df.copy()
        self.schedules = schedules_df.copy()

    def compute_team_game_stats(self) -> pd.DataFrame:
        """Aggregates play-by-play into single-team game summaries."""
        # Filter to scrimmage plays
        scrimmage = self.pbp[
            (self.pbp["play_type"].isin(["pass", "run"])) &
            (self.pbp["epa"].notnull()) &
            (self.pbp["posteam"].notnull()) &
            (self.pbp["defteam"].notnull())
        ].copy()

        # Offensive aggregations per game
        off_stats = scrimmage.groupby(["game_id", "season", "week", "posteam"]).agg(
            off_plays=("play_id", "count"),
            off_epa_total=("epa", "sum"),
            off_epa_per_play=("epa", "mean"),
            off_pass_epa_mean=("epa", lambda x: x[scrimmage.loc[x.index, "play_type"] == "pass"].mean()),
            off_rush_epa_mean=("epa", lambda x: x[scrimmage.loc[x.index, "play_type"] == "run"].mean()),
            off_success_rate=("success", "mean"),
            off_pass_rate=("pass", "mean"),
            off_neutral_pass_rate=(
                "pass",
                lambda x: x[
                    (scrimmage.loc[x.index, "score_differential"].between(-7, 7)) &
                    (scrimmage.loc[x.index, "qtr"].isin([1, 2, 3]))
                ].mean()
            ),
            off_explosive_rate=(
                "yards_gained",
                lambda x: (
                    (scrimmage.loc[x.index, "play_type"] == "pass") & (x >= 20) |
                    (scrimmage.loc[x.index, "play_type"] == "run") & (x >= 10)
                ).mean()
            )
        ).reset_index().rename(columns={"posteam": "team"})

        # Defensive aggregations per game
        def_stats = scrimmage.groupby(["game_id", "season", "week", "defteam"]).agg(
            def_plays=("play_id", "count"),
            def_epa_total=("epa", "sum"),
            def_epa_per_play=("epa", "mean"),
            def_pass_epa_mean=("epa", lambda x: x[scrimmage.loc[x.index, "play_type"] == "pass"].mean()),
            def_rush_epa_mean=("epa", lambda x: x[scrimmage.loc[x.index, "play_type"] == "run"].mean()),
            def_success_rate=("success", "mean"),
            def_explosive_rate=(
                "yards_gained",
                lambda x: (
                    (scrimmage.loc[x.index, "play_type"] == "pass") & (x >= 20) |
                    (scrimmage.loc[x.index, "play_type"] == "run") & (x >= 10)
                ).mean()
            )
        ).reset_index().rename(columns={"defteam": "team"})

        team_game = pd.merge(off_stats, def_stats, on=["game_id", "season", "week", "team"], how="outer")
        team_game.fillna(0, inplace=True)
        return team_game

    def build_rolling_team_features(self, team_game_df: pd.DataFrame) -> pd.DataFrame:
        """Applies backward rolling windows (lagged by 1 game) to eliminate data leakage."""
        team_game = team_game_df.sort_values(["team", "season", "week"]).reset_index(drop=True)
        
        metrics = [
            "off_epa_per_play", "off_pass_epa_mean", "off_rush_epa_mean",
            "off_success_rate", "off_explosive_rate", "off_neutral_pass_rate",
            "def_epa_per_play", "def_pass_epa_mean", "def_rush_epa_mean",
            "def_success_rate", "def_explosive_rate"
        ]

        rolling_dfs = []
        for team, group in team_game.groupby("team"):
            group = group.copy()
            for window in [3, 6]:
                # Strict .shift(1) ensures ONLY past performance is known before kickoff
                shifted = group[metrics].shift(1)
                rolling = shifted.rolling(window=window, min_periods=1).mean()
                rolling.columns = [f"{col}_roll_{window}" for col in metrics]
                group = pd.concat([group, rolling], axis=1)
            rolling_dfs.append(group)

        return pd.concat(rolling_dfs, ignore_index=True)

    def assemble_game_matrix(self) -> pd.DataFrame:
        """Merges home/away rolling statistics with betting lines and game metadata."""
        team_stats = self.compute_team_game_stats()
        rolling_stats = self.build_rolling_team_features(team_stats)

        # Merge with official schedule
        sched = self.schedules[[
            "game_id", "season", "week", "game_type", "home_team", "away_team",
            "home_score", "away_score", "spread_line", "total_line", "roof"
        ]].dropna(subset=["home_team", "away_team"]).copy()

        # Isolate home and away features
        home_feats = rolling_stats.copy()
        home_feats = home_feats.add_prefix("home_")
        sched = sched.merge(
            home_feats,
            left_on=["game_id", "season", "week", "home_team"],
            right_on=["home_game_id", "home_season", "home_week", "home_team"],
            how="inner"
        )

        away_feats = rolling_stats.copy()
        away_feats = away_feats.add_prefix("away_")
        sched = sched.merge(
            away_feats,
            left_on=["game_id", "season", "week", "away_team"],
            right_on=["away_game_id", "away_season", "away_week", "away_team"],
            how="inner"
        )

        # Calculate differential features
        sched["diff_epa_roll_3"] = sched["home_off_epa_per_play_roll_3"] - sched["away_off_epa_per_play_roll_3"]
        sched["diff_def_epa_roll_3"] = sched["home_def_epa_per_play_roll_3"] - sched["away_def_epa_per_play_roll_3"]
        sched["diff_epa_roll_6"] = sched["home_off_epa_per_play_roll_6"] - sched["away_off_epa_per_play_roll_6"]
        sched["diff_def_epa_roll_6"] = sched["home_def_epa_per_play_roll_6"] - sched["away_def_epa_per_play_roll_6"]

        # Empirical targets (for games already finalized)
        sched["margin"] = sched["home_score"] - sched["away_score"]
        sched["home_win"] = (sched["margin"] > 0).astype(int)
        sched["game_total"] = sched["home_score"] + sched["away_score"]
        sched["cover_spread"] = (sched["margin"] > sched["spread_line"]).astype(int)

        logger.info("Constructed Game Feature Matrix: %s rows, %s columns", sched.shape[0], sched.shape[1])
        return sched
