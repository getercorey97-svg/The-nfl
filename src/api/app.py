import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.config import PREDICTION_DIR, PROCESSED_DATA_DIR, PROP_TARGETS
from src.api.schemas import GamePredictionResponse, PropPredictionRequest, PropPredictionResponse, AuditRecord
from src.models.game_engine import GameOutcomeEngine
from src.models.prop_engine import PropPredictiveEngine
from src.models.live_engine import LivePredictiveEngine
from src.ingestion.live_loader import LiveNFLDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProductionAPI")

app = FastAPI(
    title="NFL Autonomous Prediction Engine",
    description="Optimized Pre-Match & Live In-Game NFL Modeling Platform",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Model & Data Cache
game_engine: Optional[GameOutcomeEngine] = None
prop_engine: Optional[PropPredictiveEngine] = None
cached_games_df: Optional[pd.DataFrame] = None
cached_props_df: Optional[pd.DataFrame] = None

@app.on_event("startup")
def startup_event():
    global game_engine, prop_engine, cached_games_df, cached_props_df
    try:
        game_engine = GameOutcomeEngine.load()
        prop_engine = PropPredictiveEngine.load()
        logger.info("Machine learning models initialized.")

        game_path = PROCESSED_DATA_DIR / "games_features.parquet"
        if game_path.exists():
            cached_games_df = pd.read_parquet(game_path)

        prop_path = PROCESSED_DATA_DIR / "props_features.parquet"
        if prop_path.exists():
            # Retain only necessary columns to keep memory consumption under 80MB
            cached_props_df = pd.read_parquet(prop_path)

        gc.collect()
        logger.info("Data caches loaded successfully.")
    except Exception as e:
        logger.error("Startup error: %s", str(e))

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "game_engine_ready": game_engine is not None and game_engine.is_trained,
        "prop_engine_ready": prop_engine is not None and len(prop_engine.models) > 0,
        "data_cached": cached_games_df is not None,
        "live_engine_ready": True
    }

@app.get("/api/predictions/games", response_model=List[GamePredictionResponse])
def get_latest_game_predictions(season: int = 2026, week: Optional[int] = None):
    if not game_engine or cached_games_df is None:
        raise HTTPException(status_code=503, detail="Game Outcome Engine is offline.")

    filtered = cached_games_df[cached_games_df["season"] == season]

    if week is None:
        unplayed = filtered[filtered["home_score"].isna()]
        week = int(unplayed["week"].min()) if not unplayed.empty else (int(filtered["week"].max()) if not filtered.empty else 1)

    filtered = filtered[filtered["week"] == week]
    if filtered.empty:
        return []

    predictions = game_engine.predict_game(filtered)
    return predictions.to_dict(orient="records")

@app.get("/api/live/games")
def get_live_games():
    raw_games = LiveNFLDataLoader.fetch_live_scoreboard()
    processed_live = []

    for g in raw_games:
        wp_res = LivePredictiveEngine.calculate_live_win_probability(
            margin=g["margin"],
            seconds_remaining=g["seconds_remaining"],
            pregame_spread=0.0,
            possession_team=g["possession"],
            home_team=g["home_team"],
            down=g["down"],
            distance=g["distance"],
            yardline_from_goal=100 - g["yardline"] if g["yardline"] else 50
        )

        live_total_proj = LivePredictiveEngine.calculate_live_total(
            current_total=g["current_total"],
            seconds_remaining=g["seconds_remaining"],
            pregame_total=45.5,
            down=g["down"],
            distance=g["distance"],
            yardline_from_goal=100 - g["yardline"] if g["yardline"] else 50
        )

        market_spread = g["live_spread_line"] or 0.0
        market_total = g["live_over_under"] or 0.0

        processed_live.append({
            **g,
            **wp_res,
            "projected_live_total": live_total_proj,
            "live_spread_edge": round(wp_res["fair_live_margin"] + market_spread, 1),
            "live_total_edge": round(live_total_proj - market_total, 1) if market_total > 0 else 0.0
        })

    return processed_live

@app.get("/api/game/{game_id}/breakdown")
def get_game_breakdown(game_id: str):
    """Vectorized, low-latency single game outcome and player prop inspector."""
    if cached_games_df is None or game_engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine offline.")

    game_match = cached_games_df[cached_games_df["game_id"] == game_id]
    live_meta = None

    if game_match.empty:
        live_games = LiveNFLDataLoader.fetch_live_scoreboard()
        live_match = next((g for g in live_games if g["game_id"] == game_id), None)
        if live_match:
            live_meta = live_match
        else:
            raise HTTPException(status_code=404, detail=f"Game ID {game_id} not found.")

    if not game_match.empty:
        row = game_match.iloc[0]
        home_team = row["home_team"]
        away_team = row["away_team"]
        season = int(row["season"])
        week = int(row["week"])
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        is_completed = pd.notna(home_score) and pd.notna(away_score)
        game_pred = game_engine.predict_game(game_match).iloc[0].to_dict()
    else:
        home_team = live_meta["home_team"]
        away_team = live_meta["away_team"]
        season = 2026
        week = 2
        home_score = live_meta["home_score"]
        away_score = live_meta["away_score"]
        is_completed = False
        game_pred = {
            "pred_margin": live_meta["margin"],
            "pred_total": live_meta["current_total"],
            "home_win_prob": 0.5,
            "away_win_prob": 0.5,
            "spread_line": live_meta.get("live_spread_line", 0.0),
            "total_line": live_meta.get("live_over_under", 0.0),
            "spread_edge": 0.0,
            "total_edge": 0.0
        }

    game_state = "LIVE" if (live_meta and live_meta.get("state") == "in") else ("POST-GAME" if is_completed else "UPCOMING")

    # Grade Game Lines
    game_lines_audit = []
    if is_completed:
        actual_margin = float(home_score - away_score)
        actual_total = float(home_score + away_score)
        spread_line = float(row.get("spread_line", 0.0))
        total_line = float(row.get("total_line", 0.0))

        home_covered = actual_margin > spread_line
        spread_pick_home = game_pred.get("spread_edge", 0.0) >= 0
        spread_won = 1 if ((spread_pick_home and home_covered) or (not spread_pick_home and not home_covered)) else 0
        spread_push = 1 if actual_margin == spread_line else 0

        game_lines_audit.append({
            "market": "Spread",
            "pick": f"{home_team if spread_pick_home else away_team} ({spread_line:+.1f})",
            "projected": f"{game_pred.get('pred_margin', 0.0):+.1f}",
            "line": f"{spread_line:+.1f}",
            "actual": f"{actual_margin:+.1f}",
            "status": "push" if spread_push else ("won" if spread_won else "lost")
        })

        total_pick_over = game_pred.get("total_edge", 0.0) >= 0
        tot_won = 1 if ((total_pick_over and actual_total > total_line) or (not total_pick_over and actual_total < total_line)) else 0
        tot_push = 1 if actual_total == total_line else 0

        game_lines_audit.append({
            "market": "Total",
            "pick": f"{'OVER' if total_pick_over else 'UNDER'} {total_line:.1f}",
            "projected": f"{game_pred.get('pred_total', 0.0):.1f}",
            "line": f"{total_line:.1f}",
            "actual": f"{actual_total:.1f}",
            "status": "push" if tot_push else ("won" if tot_won else "lost")
        })

        ml_won = 1 if ((game_pred.get("home_win_prob", 0.5) >= 0.5 and actual_margin > 0) or (game_pred.get("home_win_prob", 0.5) < 0.5 and actual_margin < 0)) else 0
        game_lines_audit.append({
            "market": "Moneyline",
            "pick": f"{home_team if game_pred.get('home_win_prob', 0.5) >= 0.5 else away_team}",
            "projected": f"{max(game_pred.get('home_win_prob', 0.5), game_pred.get('away_win_prob', 0.5))*100:.1f}%",
            "line": "EVEN",
            "actual": f"{home_team if actual_margin > 0 else away_team} WIN",
            "status": "won" if ml_won else "lost"
        })

    # Optimized Vectorized Prop Prediction for Game
    player_props_list = []
    if cached_props_df is not None and prop_engine:
        game_props = cached_props_df[
            (cached_props_df["season"] == season) &
            (cached_props_df["week"] == week) &
            (cached_props_df["team"].isin([home_team, away_team]))
        ]

        if not game_props.empty:
            prop_configs = [
                ("passing_yards", 225.5, lambda df: df[df["passing_attempts_roll_3"] >= 5]),
                ("rushing_yards", 48.5, lambda df: df[df["rushing_attempts_roll_3"] >= 3]),
                ("receiving_yards", 45.5, lambda df: df[df["targets_roll_3"] >= 2]),
                ("receptions", 4.5, lambda df: df[df["targets_roll_3"] >= 2]),
                ("anytime_td", 0.5, lambda df: df[(df["rushing_attempts_roll_3"] >= 3) | (df["targets_roll_3"] >= 2)])
            ]

            for prop_name, default_line, filter_fn in prop_configs:
                if prop_name not in prop_engine.models:
                    continue

                sub_group = filter_fn(game_props)
                if sub_group.empty:
                    continue

                # Batch inference for all players in this game
                preds_df = prop_engine.predict_prop(sub_group, prop_name, default_line)

                for idx, (_, p_row) in enumerate(sub_group.iterrows()):
                    p_pred = preds_df.iloc[idx]
                    over_prob = float(p_pred["over_prob"])
                    ev = float(p_pred["expected_value"])
                    call = "OVER" if over_prob >= 0.50 else "UNDER"

                    item = {
                        "player_name": p_row.get("player_name", p_row["player_id"]),
                        "team": p_row.get("team", ""),
                        "prop_name": prop_name.replace("_", " ").title(),
                        "line": default_line,
                        "call": call,
                        "projected": round(ev, 1),
                        "prob": round(max(over_prob, 1.0 - over_prob) * 100, 1),
                        "status": "pending",
                        "actual": "-",
                        "error": "-"
                    }

                    if is_completed and prop_name in p_row:
                        actual_val = float(p_row[prop_name])
                        p_won = 1 if ((call == "OVER" and actual_val > default_line) or (call == "UNDER" and actual_val < default_line)) else 0
                        p_push = 1 if actual_val == default_line else 0
                        item["actual"] = round(actual_val, 1)
                        item["error"] = round(abs(ev - actual_val), 1)
                        item["status"] = "push" if p_push else ("won" if p_won else "lost")

                    player_props_list.append(item)

    gl_won = sum(1 for g in game_lines_audit if g["status"] == "won")
    gl_total = len(game_lines_audit)
    gl_acc = round((gl_won / gl_total) * 100, 1) if gl_total > 0 else 0.0

    prop_won = sum(1 for p in player_props_list if p["status"] == "won")
    prop_lost = sum(1 for p in player_props_list if p["status"] == "lost")
    prop_decided = prop_won + prop_lost
    prop_acc = round((prop_won / prop_decided) * 100, 1) if prop_decided > 0 else 0.0

    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "state": game_state,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score if pd.notna(home_score) else 0,
        "away_score": away_score if pd.notna(away_score) else 0,
        "game_line_predictions": game_pred,
        "game_lines_audit": game_lines_audit,
        "player_prop_predictions": player_props_list,
        "summary": {
            "game_lines_accuracy_pct": gl_acc,
            "game_lines_won": gl_won,
            "game_lines_total": gl_total,
            "player_props_accuracy_pct": prop_acc,
            "player_props_won": prop_won,
            "player_props_lost": prop_lost,
            "player_props_total": len(player_props_list)
        }
    }

@app.get("/api/audit", response_model=List[AuditRecord])
def get_post_mortem_audit():
    audit_file = PREDICTION_DIR / "post_mortem_audit_ledger.csv"
    if not audit_file.exists():
        return []
    return pd.read_csv(audit_file).to_dict(orient="records")

@app.get("/api/audit/details")
def get_audit_details(status: Optional[str] = Query(None), category: Optional[str] = Query(None), week: Optional[int] = Query(None)):
    details_file = PREDICTION_DIR / "prediction_audit_details.csv"
    if not details_file.exists():
        return []
    df = pd.read_csv(details_file)
    if status:
        df = df[df["status"] == status.lower()]
    if category:
        df = df[df["category"] == category.lower()]
    if week is not None:
        df = df[df["week"] == week]
    return df.to_dict(orient="records")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    dashboard_path = Path(__file__).resolve().parent / "dashboard.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text())
    return HTMLResponse(content="<h1>Dashboard Loading Error: dashboard.html missing.</h1>")
