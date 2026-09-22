import logging
from typing import List, Optional
import pandas as pd
import polars as pl
import nflreadpy as nfl
from src.config import RAW_DATA_DIR, START_YEAR, CURRENT_YEAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PBP_REDUCED_COLUMNS = [
    "game_id", "season", "week", "play_id", "play_type", "epa", "posteam", "defteam",
    "success", "pass", "score_differential", "qtr", "yards_gained", "complete_pass",
    "rusher_player_id", "passer_player_id", "receiver_player_id", "sack_player_id",
    "solo_tackle_1_player_id", "assist_tackle_1_player_id", "sack", "solo_tackle", "assist_tackle"
]

class NFLDataLoader:
    """Manages low-memory ingestion of raw NFL datasets."""

    def __init__(self, start_year: int = START_YEAR, end_year: int = CURRENT_YEAR):
        self.start_year = start_year
        self.end_year = end_year
        self.seasons = list(range(start_year, end_year + 1))

    @staticmethod
    def _to_pandas(data) -> pd.DataFrame:
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
        elif hasattr(data, "to_pandas"):
            return data.to_pandas()
        return pd.DataFrame(data)

    def load_play_by_play(self, seasons: Optional[List[int]] = None, force_refresh: bool = False) -> pd.DataFrame:
        target_seasons = seasons or self.seasons
        cache_file = RAW_DATA_DIR / f"pbp_{min(target_seasons)}_{max(target_seasons)}.parquet"
        
        if cache_file.exists() and not force_refresh:
            logger.info("Loading cached Play-by-Play dataset from: %s", cache_file)
            return pd.read_parquet(cache_file)

        logger.info("Ingesting column-pruned Play-by-Play data for seasons: %s", target_seasons)
        raw_pbp = nfl.load_pbp(target_seasons)
        df = self._to_pandas(raw_pbp)
        
        # Retain only required modeling columns to keep memory usage under 250MB
        present_cols = [c for c in PBP_REDUCED_COLUMNS if c in df.columns]
        df = df[present_cols]

        df.to_parquet(cache_file, index=False, compression="snappy")
        logger.info("Saved %d rows to %s", len(df), cache_file)
        return df

    def load_player_stats(self, seasons: Optional[List[int]] = None, force_refresh: bool = False) -> pd.DataFrame:
        target_seasons = seasons or self.seasons
        cache_file = RAW_DATA_DIR / f"player_stats_{min(target_seasons)}_{max(target_seasons)}.parquet"

        if cache_file.exists() and not force_refresh:
            return pd.read_parquet(cache_file)

        raw_stats = nfl.load_player_stats(target_seasons)
        df = self._to_pandas(raw_stats)
        df.to_parquet(cache_file, index=False, compression="snappy")
        return df

    def load_schedules(self, seasons: Optional[List[int]] = None, force_refresh: bool = False) -> pd.DataFrame:
        target_seasons = seasons or self.seasons
        cache_file = RAW_DATA_DIR / f"schedules_{min(target_seasons)}_{max(target_seasons)}.parquet"

        if cache_file.exists() and not force_refresh:
            return pd.read_parquet(cache_file)

        raw_sched = nfl.load_schedules(target_seasons)
        df = self._to_pandas(raw_sched)
        df.to_parquet(cache_file, index=False, compression="snappy")
        return df

    def load_snap_counts(self, seasons: Optional[List[int]] = None, force_refresh: bool = False) -> pd.DataFrame:
        target_seasons = seasons or self.seasons
        cache_file = RAW_DATA_DIR / f"snap_counts_{min(target_seasons)}_{max(target_seasons)}.parquet"

        if cache_file.exists() and not force_refresh:
            return pd.read_parquet(cache_file)

        raw_snaps = nfl.load_snap_counts(target_seasons)
        df = self._to_pandas(raw_snaps)
        df.to_parquet(cache_file, index=False, compression="snappy")
        return df
