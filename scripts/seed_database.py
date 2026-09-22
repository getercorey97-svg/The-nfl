import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.nfl_loader import NFLDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatabaseSeeder")

def run_seed():
    logger.info("Executing cold-start database seed for 2020 through 2026...")
    loader = NFLDataLoader(start_year=2020, end_year=2026)

    # 1. Schedules & Betting Lines
    schedules = loader.load_schedules(force_refresh=True)
    logger.info("Schedules loaded: %s games", schedules.shape[0])

    # 2. Player Weekly Statistics
    player_stats = loader.load_player_stats(force_refresh=True)
    logger.info("Weekly Player Stats loaded: %s records", player_stats.shape[0])

    # 3. Snap Counts
    snap_counts = loader.load_snap_counts(force_refresh=True)
    logger.info("Snap counts loaded: %s records", snap_counts.shape[0])

    # 4. Play-by-Play & EPA metrics
    pbp = loader.load_play_by_play(force_refresh=True)
    logger.info("Play-by-play rows loaded: %s", pbp.shape[0])

    logger.info("Database cold-start seed successfully completed.")

if __name__ == "__main__":
    run_seed()
