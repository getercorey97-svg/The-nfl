import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import CURRENT_YEAR, PROCESSED_DATA_DIR
from src.ingestion.nfl_loader import NFLDataLoader
from src.features.game_features import GameFeatureEngine
from src.features.prop_features import PropFeatureEngine
from src.models.game_engine import GameOutcomeEngine
from src.models.prop_engine import PropPredictiveEngine
from src.pipeline.feedback import PostMortemAudit
from src.pipeline.predict import PreMatchPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WeeklyPipeline")

def run_pipeline(season: int, previous_week: int, upcoming_week: int):
    logger.info("================ STARTING AUTOMATED WEEKLY PIPELINE ================")
    logger.info("Season: %d | Audit Target: Week %d | Upcoming Target: Week %d", season, previous_week, upcoming_week)

    loader = NFLDataLoader(start_year=2020, end_year=season)

    # Step 1: Ingest fresh completed data
    logger.info("Phase 1: Ingesting newly verified games and box scores...")
    pbp = loader.load_play_by_play(force_refresh=True)
    schedules = loader.load_schedules(force_refresh=True)
    player_stats = loader.load_player_stats(force_refresh=True)
    snap_counts = loader.load_snap_counts(force_refresh=True)

    # Step 2: Post-Mortem Audit on Completed Prior Week
    if previous_week > 0:
        logger.info("Phase 2: Executing simulation-free post-mortem audit on Week %d...", previous_week)
        try:
            auditor = PostMortemAudit(season=season, week=previous_week)
            completed_games = schedules[(schedules["season"] == season) & (schedules["week"] == previous_week)]
            audit_metrics = auditor.audit_completed_week(completed_games, player_stats)
            logger.info("Post-Mortem Results: %s", audit_metrics)
        except Exception as e:
            logger.warning("Post-Mortem Audit skipped for Week %d: %s", previous_week, str(e))

    # Step 3: Recompute features and evolve models with newly confirmed data
    logger.info("Phase 3: Updating feature matrices and recalibrating models...")
    game_engine_feat = GameFeatureEngine(pbp_df=pbp, schedules_df=schedules)
    game_matrix = game_engine_feat.assemble_game_matrix()
    game_matrix.to_parquet(PROCESSED_DATA_DIR / "games_features.parquet", index=False)

    prop_engine_feat = PropFeatureEngine(pbp_df=pbp, player_stats_df=player_stats, snap_df=snap_counts)
    prop_matrix = prop_engine_feat.build_prop_matrix()
    prop_matrix.to_parquet(PROCESSED_DATA_DIR / "props_features.parquet", index=False)

    # Retrain estimators on expanded dataset
    logger.info("Refitting Game and 20-Prop Estimators on empirical history...")
    g_model = GameOutcomeEngine()
    g_model.train(game_matrix)
    g_model.save()

    p_model = PropPredictiveEngine()
    p_model.train_all_props(prop_matrix)
    p_model.save()

    # Step 4: Generate and lock pre-match predictions for upcoming week
    logger.info("Phase 4: Locking pre-match forecast ledger for Week %d...", upcoming_week)
    upcoming_games = game_matrix[
        (game_matrix["season"] == season) &
        (game_matrix["week"] == upcoming_week)
    ].copy()

    if not upcoming_games.empty:
        predictor = PreMatchPredictor()
        game_preds = predictor.generate_game_forecasts(upcoming_games)
        
        # Lock in immutable pre-match forecast snapshot
        predictor.lock_pre_match_ledger(season=season, week=upcoming_week, game_preds=game_preds, prop_preds=pd.DataFrame())
        logger.info("Upcoming Week %d pre-match predictions successfully locked.", upcoming_week)
    else:
        logger.info("No unplayed games detected in feature matrix for Season %d, Week %d.", season, upcoming_week)

    logger.info("================ WEEKLY PIPELINE COMPLETED SUCCESSFULLY ================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFL Weekly Automated Prediction and Audit Pipeline")
    parser.add_argument("--season", type=int, default=CURRENT_YEAR, help="NFL Season Year")
    parser.add_argument("--prev-week", type=int, default=1, help="Completed week for post-mortem audit")
    parser.add_argument("--next-week", type=int, default=2, help="Upcoming week for pre-match predictions")
    args = parser.parse_args()

    run_pipeline(season=args.season, previous_week=args.prev_week, upcoming_week=args.next_week)
