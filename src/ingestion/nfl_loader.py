import logging
from typing import List, Optional
import pandas as pd
import polars as pl
import nflreadpy as nfl
from src.config import RAW_DATA_DIR, START_YEAR, CURRENT_YEAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class NFLDataLoader:
    """Manages ingestion of raw NFL play-by-play, box scores, schedules, and snap counts."""

    def __init__(self, start_year: int = START_YEAR, end_year: int = CURRENT_YEAR):
        self.start_year = start_year
        self.end_year = end_year
        self.seasons = list(range(start_year, end_year + 1))

    @staticmethod
    def _to_pandas(data) -> pd.DataFrame:
        """Converts nflreadpy Polars DataFrame to Pandas DataFrame."""
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

        logger.info("Ingesting Play-by-Play data for seasons: %s", target_seasons)
        raw_pbp = nfl.load_pbp(target_seasons)
        df = self._to_pandas(raw_pbp)
        df.to_parquet(cache_file, index=False, compression="snappy")
        logger.info("Saved %d rows to %s", len(df), cache_file)
        return df

    def load_player_stats(self, seasons: Optional[List[int]] = None, force_refresh: bool = False) -> pd.DataFrame:
        target_seasons = seasons or self.seasons
        cache_file = RAW_DATA_DIR / f"player_stats_{min(target_seasons)}_{max(target_seasons)}.parquet"

        if cache_file.exists() and not force_refresh:
            logger.info("Loading cached Weekly Player Stats from: %s", cache_file)
            return pd.read_parquet(cache_file)

        logger.info("Ingesting Weekly Player Stats for seasons: %s", target_seasons)
        raw_stats = nfl.load_player_stats(target_seasons)
        df = self._to_pandas(raw_stats)
        df.to_parquet(cache_file, index=False, compression="snappy")
        logger.info("Saved %d rows to %s", len(df), cache_file)
        return df

    def load_schedules(self, seasons: Optional[List[int]] = None, force_refresh: bool = False) -> pd.DataFrame:
        target_seasons = seasons or self.seasons
        cache_file = RAW_DATA_DIR / f"schedules_{min(target_seasons)}_{max(target_seasons)}.parquet"

        if cache_file.exists() and not force_refresh:
            logger.info("Loading cached Schedules from: %s", cache_file)
            return pd.read_parquet(cache_file)

        logger.info("Ingesting Schedules for seasons: %s", target_seasons)
        raw_sched = nfl.load_schedules(target_seasons)
        df = self._to_pandas(raw_sched)
        df.to_parquet(cache_file, index=False, compression="snappy")
        logger.info("Saved %d schedule records to %s", len(df), cache_file)
        return df

    def load_snap_counts(self, seasons: Optional[List[int]] = None, force_refresh: bool = False) -> pd.DataFrame:
        target_seasons = seasons or self.seasons
        cache_file = RAW_DATA_DIR / f"snap_counts_{min(target_seasons)}_{max(target_seasons)}.parquet"

        if cache_file.exists() and not force_refresh:
            logger.info("Loading cached Snap Counts from: %s", cache_file)
            return pd.read_parquet(cache_file)

        logger.info("Ingesting Snap Counts for seasons: %s", target_seasons)
        raw_snaps = nfl.load_snap_counts(target_seasons)
        df = self._to_pandas(raw_snaps)
        df.to_parquet(cache_file, index=False, compression="snappy")
        logger.info("Saved %d snap count records to %s", len(df), cache_file)
        return df
