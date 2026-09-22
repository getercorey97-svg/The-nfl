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
    description="State-of-the-Art Pre-Match & Live In-Game NFL Modeling Platform",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

game_engine: Optional[GameOutcomeEngine] = None
prop_engine: Optional[PropPredictiveEngine] = None

@app.on_event("startup")
def startup_event():
    global game_engine, prop_engine
    try:
        game_engine = GameOutcomeEngine.load()
        prop_engine = PropPredictiveEngine.load()
        logger.info("Production models loaded into memory.")
    except Exception as e:
        logger.error("Error loading model artifacts: %s", str(e))

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "game_engine_ready": game_engine is not None and game_engine.is_trained,
        "prop_engine_ready": prop_engine is not None and len(prop_engine.models) > 0,
        "live_engine_ready": True
    }

@app.get("/api/predictions/games", response_model=List[GamePredictionResponse])
def get_latest_game_predictions(season: int = 2026, week: Optional[int] = None):
    if not game_engine:
        raise HTTPException(status_code=503, detail="Game Outcome Engine is offline.")

    game_matrix_path = PROCESSED_DATA_DIR / "games_features.parquet"
    if not game_matrix_path.exists():
        raise HTTPException(status_code=404, detail="Game feature matrix not found.")

    df = pd.read_parquet(game_matrix_path)
    filtered = df[df["season"] == season]

    if week is None:
        unplayed = filtered[filtered["home_score"].isna()]
        if not unplayed.empty:
            week = int(unplayed["week"].min())
        else:
            week = int(filtered["week"].max()) if not filtered.empty else 1

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
    """Retrieves full predictions, player prop projections, and empirical hit/miss evaluations for a single game."""
    game_file = PROCESSED_DATA_DIR / "games_features.parquet"
    prop_file = PROCESSED_DATA_DIR / "props_features.parquet"
    details_file = PREDICTION_DIR / "prediction_audit_details.csv"

    if not game_file.exists():
        raise HTTPException(status_code=404, detail="Game features repository offline.")

    games_df = pd.read_parquet(game_file)
    game_match = games_df[games_df["game_id"] == game_id]

    # Check live scoreboard if not found in processed parquet
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
        
        # Game lines prediction
        game_pred = game_engine.predict_game(game_match).iloc[0].to_dict() if game_engine else {}
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

    # Identify state
    if live_meta and live_meta.get("state") == "in":
        game_state = "LIVE"
    elif is_completed:
        game_state = "POST-GAME"
    else:
        game_state = "UPCOMING"

    # Evaluate Game Lines Performance
    game_lines_audit = []
    if is_completed:
        actual_margin = float(home_score - away_score)
        actual_total = float(home_score + away_score)
        spread_line = float(row.get("spread_line", 0.0))
        total_line = float(row.get("total_line", 0.0))

        # Spread
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

        # Total
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

        # Moneyline
        ml_won = 1 if ((game_pred.get("home_win_prob", 0.5) >= 0.5 and actual_margin > 0) or (game_pred.get("home_win_prob", 0.5) < 0.5 and actual_margin < 0)) else 0
        game_lines_audit.append({
            "market": "Moneyline",
            "pick": f"{home_team if game_pred.get('home_win_prob', 0.5) >= 0.5 else away_team}",
            "projected": f"{max(game_pred.get('home_win_prob', 0.5), game_pred.get('away_win_prob', 0.5))*100:.1f}%",
            "line": "EVEN",
            "actual": f"{home_team if actual_margin > 0 else away_team} WIN",
            "status": "won" if ml_won else "lost"
        })

    # Retrieve & Grade Player Props for this Game
    player_props_list = []
    if prop_file.exists():
        props_df = pd.read_parquet(prop_file)
        game_props = props_df[
            (props_df["season"] == season) &
            (props_df["week"] == week) &
            (props_df["team"].isin([home_team, away_team]))
        ]

        if not game_props.empty and prop_engine:
            common_props = [
                ("passing_yards", 225.5),
                ("rushing_yards", 52.5),
                ("receiving_yards", 48.5),
                ("receptions", 4.5),
                ("anytime_td", 0.5)
            ]

            for _, p_row in game_props.iterrows():
                p_id = p_row["player_id"]
                p_name = p_row.get("player_name", p_id)
                p_team = p_row.get("team", "")

                for prop_name, default_line in common_props:
                    if prop_name not in PROP_TARGETS:
                        continue

                    # Filter based on player snap role
                    if prop_name.startswith("pass") and p_row.get("passing_attempts_roll_3", 0) < 5:
                        continue
                    if prop_name.startswith("rush") and p_row.get("rushing_attempts_roll_3", 0) < 3:
                        continue
                    if prop_name.startswith("rec") and p_row.get("targets_roll_3", 0) < 2:
                        continue

                    pred_res = prop_engine.predict_prop(pd.DataFrame([p_row]), prop_name, default_line)
                    over_prob = float(pred_res["over_prob"].iloc[0])
                    ev = float(pred_res["expected_value"].iloc[0])
                    call = "OVER" if over_prob >= 0.50 else "UNDER"

                    prop_item = {
                        "player_name": p_name,
                        "team": p_team,
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
                        prop_item["actual"] = round(actual_val, 1)
                        prop_item["error"] = round(abs(ev - actual_val), 1)
                        prop_item["status"] = "push" if p_push else ("won" if p_won else "lost")

                    player_props_list.append(prop_item)

    # Compute overall game summary accuracy
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
    df = pd.read_csv(audit_file)
    return df.to_dict(orient="records")

@app.get("/api/audit/details")
def get_audit_details(
    status: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    week: Optional[int] = Query(None)
):
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
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NFL State-of-the-Art Prediction Engine</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen">
        <nav class="border-b border-slate-800 bg-slate-900/50 backdrop-blur px-6 py-4 flex flex-wrap justify-between items-center gap-4">
            <div class="flex items-center space-x-3">
                <span class="inline-block w-2.5 h-2.5 bg-emerald-500 rounded-full animate-pulse"></span>
                <h1 class="text-xl font-bold tracking-tight text-white">NFL AI Prediction Engine</h1>
            </div>
            <div class="flex space-x-2 text-sm font-medium">
                <button onclick="showTab('live')" id="tab-btn-live" class="px-3 py-1.5 rounded-lg bg-emerald-500/20 text-emerald-400 font-semibold">Live Betting</button>
                <button onclick="showTab('games')" id="tab-btn-games" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-white">Pre-Match</button>
                <button onclick="showTab('audit')" id="tab-btn-audit" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-white">Weekly Summary</button>
                <button onclick="showTab('details')" id="tab-btn-details" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-white">Pick Grader</button>
            </div>
        </nav>

        <main class="max-w-7xl mx-auto p-6">
            <!-- LIVE BETTING CENTER -->
            <section id="section-live">
                <div class="flex justify-between items-center mb-6">
                    <div>
                        <h2 class="text-2xl font-bold flex items-center gap-2">
                            <span>Live In-Game Edge Screener</span>
                            <span class="text-xs font-semibold px-2 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800">REAL-TIME</span>
                        </h2>
                        <p class="text-xs text-slate-400 mt-1">Click on any game card to view full game lines and player prop breakdown</p>
                    </div>
                    <button onclick="loadLiveGames()" class="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-lg text-sm font-semibold">Refresh Telemetry</button>
                </div>
                <div id="live-container" class="grid grid-cols-1 md:grid-cols-2 gap-6"></div>
            </section>

            <!-- PRE-MATCH FORECASTER -->
            <section id="section-games" class="hidden">
                <div class="flex flex-wrap justify-between items-center mb-6 gap-4">
                    <div>
                        <h2 class="text-2xl font-bold">Pre-Match Projections & Edges</h2>
                        <p class="text-xs text-slate-400 mt-1">Click on any game to inspect all game line and player prop predictions</p>
                    </div>
                    <div class="flex items-center space-x-3 text-sm">
                        <select id="select-season" onchange="loadGames()" class="bg-slate-900 border border-slate-700 px-3 py-1.5 rounded-lg text-white">
                            <option value="2026" selected>2026 Season</option>
                            <option value="2025">2025 Season</option>
                            <option value="2024">2024 Season</option>
                        </select>
                        <select id="select-week" onchange="loadGames()" class="bg-slate-900 border border-slate-700 px-3 py-1.5 rounded-lg text-white">
                            <option value="2" selected>Week 2</option>
                            <option value="1">Week 1</option>
                            <option value="3">Week 3</option>
                            <option value="4">Week 4</option>
                        </select>
                    </div>
                </div>
                <div id="games-container" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"></div>
            </section>

            <!-- AUDIT & GRADER SECTIONS -->
            <section id="section-audit" class="hidden">
                <h2 class="text-2xl font-bold mb-6">Weekly Empirical Performance Summary</h2>
                <div class="overflow-x-auto bg-slate-900 rounded-xl border border-slate-800">
                    <table class="w-full text-left border-collapse text-sm">
                        <thead>
                            <tr class="border-b border-slate-800 text-slate-400">
                                <th class="p-4">Season</th><th class="p-4">Week</th><th class="p-4">Games</th><th class="p-4">Spread MAE</th><th class="p-4">ATS Win %</th><th class="p-4">ML Brier</th><th class="p-4">ML Acc %</th><th class="p-4">Prop Acc %</th>
                            </tr>
                        </thead>
                        <tbody id="audit-table-body" class="divide-y divide-slate-800"></tbody>
                    </table>
                </div>
            </section>

            <section id="section-details" class="hidden">
                <div class="flex justify-between items-center mb-6">
                    <h2 class="text-2xl font-bold">Pick Grader (Hits & Misses)</h2>
                </div>
                <div class="overflow-x-auto bg-slate-900 rounded-xl border border-slate-800">
                    <table class="w-full text-left border-collapse text-sm">
                        <thead>
                            <tr class="border-b border-slate-800 text-slate-400">
                                <th class="p-4">Outcome</th><th class="p-4">Week</th><th class="p-4">Market</th><th class="p-4">Matchup</th><th class="p-4">Pick</th><th class="p-4">Projected</th><th class="p-4">Line</th><th class="p-4">Actual</th><th class="p-4">Error</th>
                            </tr>
                        </thead>
                        <tbody id="details-table-body" class="divide-y divide-slate-800"></tbody>
                    </table>
                </div>
            </section>
        </main>

        <!-- GAME DEEP-DIVE MODAL -->
        <div id="game-modal" class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 hidden flex justify-center items-center p-4">
            <div class="bg-slate-900 border border-slate-700 w-full max-w-5xl rounded-2xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
                <!-- Modal Header -->
                <div class="p-6 border-b border-slate-800 bg-slate-950 flex justify-between items-start">
                    <div>
                        <div class="flex items-center space-x-3 mb-2">
                            <span id="modal-status-badge" class="px-2.5 py-0.5 rounded text-xs font-bold uppercase tracking-wider"></span>
                            <span id="modal-week-text" class="text-xs text-slate-400 font-medium"></span>
                        </div>
                        <h2 id="modal-title" class="text-2xl font-black tracking-tight text-white"></h2>
                        <div id="modal-score" class="text-lg font-mono text-emerald-400 mt-1"></div>
                    </div>
                    <button onclick="closeModal()" class="text-slate-400 hover:text-white text-2xl font-bold p-1">&times;</button>
                </div>

                <!-- Accuracy Metrics Summary Banner -->
                <div class="grid grid-cols-2 md:grid-cols-4 gap-3 p-4 bg-slate-900/60 border-b border-slate-800 text-center text-xs">
                    <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                        <div class="text-slate-400 mb-1">Game Lines Accuracy</div>
                        <div id="modal-gl-acc" class="text-base font-bold font-mono text-white">-</div>
                    </div>
                    <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                        <div class="text-slate-400 mb-1">Game Lines Record</div>
                        <div id="modal-gl-rec" class="text-base font-bold font-mono text-white">-</div>
                    </div>
                    <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                        <div class="text-slate-400 mb-1">Player Props Accuracy</div>
                        <div id="modal-prop-acc" class="text-base font-bold font-mono text-white">-</div>
                    </div>
                    <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                        <div class="text-slate-400 mb-1">Player Props Record</div>
                        <div id="modal-prop-rec" class="text-base font-bold font-mono text-white">-</div>
                    </div>
                </div>

                <!-- Modal Sub-Tabs -->
                <div class="flex border-b border-slate-800 bg-slate-950/40 px-6 pt-3 space-x-6 text-sm">
                    <button onclick="showModalTab('props')" id="modal-tab-props" class="pb-3 border-b-2 border-emerald-400 text-emerald-400 font-semibold">Player Props Predictions</button>
                    <button onclick="showModalTab('lines')" id="modal-tab-lines" class="pb-3 border-b-2 border-transparent text-slate-400 hover:text-white">Game Line Predictions</button>
                </div>

                <!-- Modal Body Content -->
                <div class="p-6 overflow-y-auto space-y-6 flex-1">
                    <!-- Props View -->
                    <div id="modal-view-props">
                        <div class="flex justify-between items-center mb-4">
                            <span class="text-xs uppercase font-bold tracking-wider text-slate-400">All Individual Player Markets</span>
                            <div class="flex space-x-2 text-xs">
                                <button onclick="filterModalProps('all')" class="px-2.5 py-1 bg-slate-800 rounded hover:bg-slate-700">All</button>
                                <button onclick="filterModalProps('won')" class="px-2.5 py-1 bg-emerald-950 text-emerald-400 rounded hover:bg-emerald-900 border border-emerald-800">Hits Only</button>
                                <button onclick="filterModalProps('lost')" class="px-2.5 py-1 bg-rose-950 text-rose-400 rounded hover:bg-rose-900 border border-rose-800">Misses Only</button>
                            </div>
                        </div>
                        <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950">
                            <table class="w-full text-left text-xs border-collapse">
                                <thead>
                                    <tr class="border-b border-slate-800 text-slate-400 bg-slate-900/50">
                                        <th class="p-3">Status</th>
                                        <th class="p-3">Player</th>
                                        <th class="p-3">Team</th>
                                        <th class="p-3">Market</th>
                                        <th class="p-3">Model Pick</th>
                                        <th class="p-3">Projected</th>
                                        <th class="p-3">Line</th>
                                        <th class="p-3">Actual</th>
                                        <th class="p-3">Margin Err</th>
                                    </tr>
                                </thead>
                                <tbody id="modal-props-tbody" class="divide-y divide-slate-800 font-mono"></tbody>
                            </table>
                        </div>
                    </div>

                    <!-- Game Lines View -->
                    <div id="modal-view-lines" class="hidden">
                        <span class="text-xs uppercase font-bold tracking-wider text-slate-400 block mb-4">Game Outcomes & Market Lines</span>
                        <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950">
                            <table class="w-full text-left text-xs border-collapse">
                                <thead>
                                    <tr class="border-b border-slate-800 text-slate-400 bg-slate-900/50">
                                        <th class="p-3">Status</th>
                                        <th class="p-3">Market</th>
                                        <th class="p-3">Recommendation</th>
                                        <th class="p-3">Projected</th>
                                        <th class="p-3">Line</th>
                                        <th class="p-3">Actual</th>
                                    </tr>
                                </thead>
                                <tbody id="modal-lines-tbody" class="divide-y divide-slate-800 font-mono"></tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let activeModalProps = [];

            function showTab(tab) {
                ['live', 'games', 'audit', 'details'].forEach(t => {
                    document.getElementById('section-' + t).classList.toggle('hidden', t !== tab);
                    const btn = document.getElementById('tab-btn-' + t);
                    btn.className = (t === tab) 
                        ? 'px-3 py-1.5 rounded-lg bg-emerald-500/20 text-emerald-400 font-semibold'
                        : 'px-3 py-1.5 rounded-lg text-slate-400 hover:text-white';
                });

                if (tab === 'live') loadLiveGames();
                if (tab === 'games') loadGames();
                if (tab === 'audit') loadAudit();
                if (tab === 'details') loadDetails();
            }

            async function openGameModal(gameId) {
                const modal = document.getElementById('game-modal');
                modal.classList.remove('hidden');

                document.getElementById('modal-title').innerText = 'Loading game breakdown...';
                document.getElementById('modal-score').innerText = '';
                document.getElementById('modal-props-tbody').innerHTML = '<tr><td colspan="9" class="p-4 text-center text-slate-500">Querying model predictions...</td></tr>';
                document.getElementById('modal-lines-tbody').innerHTML = '';

                try {
                    const res = await fetch(`/api/game/${gameId}/breakdown`);
                    if (!res.ok) throw new Error('API status ' + res.status);
                    const data = await res.json();

                    document.getElementById('modal-title').innerText = `${data.away_team} @ ${data.home_team}`;
                    document.getElementById('modal-week-text').innerText = `SEASON ${data.season} • WEEK ${data.week}`;
                    
                    const badge = document.getElementById('modal-status-badge');
                    badge.innerText = data.state;
                    if (data.state === 'LIVE') {
                        badge.className = 'px-2.5 py-0.5 rounded text-xs font-bold uppercase bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse';
                        document.getElementById('modal-score').innerText = `${data.away_team} ${data.away_score} - ${data.home_score} ${data.home_team}`;
                    } else if (data.state === 'POST-GAME') {
                        badge.className = 'px-2.5 py-0.5 rounded text-xs font-bold uppercase bg-emerald-500/20 text-emerald-400 border border-emerald-500/40';
                        document.getElementById('modal-score').innerText = `Final: ${data.away_team} ${data.away_score} - ${data.home_score} ${data.home_team}`;
                    } else {
                        badge.className = 'px-2.5 py-0.5 rounded text-xs font-bold uppercase bg-slate-800 text-slate-400 border border-slate-700';
                        document.getElementById('modal-score').innerText = 'Matchup Upcoming';
                    }

                    // Accuracy Summary
                    document.getElementById('modal-gl-acc').innerText = data.state === 'POST-GAME' ? `${data.summary.game_lines_accuracy_pct}%` : 'PENDING';
                    document.getElementById('modal-gl-rec').innerText = `${data.summary.game_lines_won} / ${data.summary.game_lines_total}`;
                    document.getElementById('modal-prop-acc').innerText = data.state === 'POST-GAME' ? `${data.summary.player_props_accuracy_pct}%` : 'PENDING';
                    document.getElementById('modal-prop-rec').innerText = `${data.summary.player_props_won} Won / ${data.summary.player_props_lost} Lost`;

                    // Render Game Lines
                    const linesTbody = document.getElementById('modal-lines-tbody');
                    linesTbody.innerHTML = '';
                    if (data.game_lines_audit.length === 0) {
                        linesTbody.innerHTML = '<tr><td colspan="6" class="p-4 text-center text-slate-500">No finalized game lines to grade.</td></tr>';
                    } else {
                        data.game_lines_audit.forEach(l => {
                            let b = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-slate-800 text-slate-400">PENDING</span>';
                            if (l.status === 'won') b = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-emerald-950 text-emerald-400 border border-emerald-800">HIT</span>';
                            else if (l.status === 'lost') b = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-rose-950 text-rose-400 border border-rose-800">MISS</span>';
                            else if (l.status === 'push') b = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-slate-800 text-slate-300">PUSH</span>';

                            linesTbody.innerHTML += `
                                <tr class="hover:bg-slate-800/40">
                                    <td class="p-3">${b}</td>
                                    <td class="p-3 font-semibold text-white">${l.market}</td>
                                    <td class="p-3 text-sky-400">${l.pick}</td>
                                    <td class="p-3">${l.projected}</td>
                                    <td class="p-3 text-slate-400">${l.line}</td>
                                    <td class="p-3 font-bold text-white">${l.actual}</td>
                                </tr>
                            `;
                        });
                    }

                    // Store Props for Filtering
                    activeModalProps = data.player_prop_predictions;
                    renderModalProps(activeModalProps);

                } catch (err) {
                    document.getElementById('modal-title').innerText = 'Error loading breakdown';
                    document.getElementById('modal-props-tbody').innerHTML = `<tr><td colspan="9" class="p-4 text-center text-rose-500">${err.message}</td></tr>`;
                }
            }

            function renderModalProps(propsList) {
                const tbody = document.getElementById('modal-props-tbody');
                tbody.innerHTML = '';

                if (propsList.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="9" class="p-4 text-center text-slate-500">No player prop projections found for this game.</td></tr>';
                    return;
                }

                propsList.forEach(p => {
                    let b = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-slate-800 text-slate-400">PENDING</span>';
                    if (p.status === 'won') b = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-emerald-950 text-emerald-400 border border-emerald-800">HIT</span>';
                    else if (p.status === 'lost') b = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-rose-950 text-rose-400 border border-rose-800">MISS</span>';
                    else if (p.status === 'push') b = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-slate-800 text-slate-300">PUSH</span>';

                    tbody.innerHTML += `
                        <tr class="hover:bg-slate-800/40">
                            <td class="p-3">${b}</td>
                            <td class="p-3 font-semibold text-white">${p.player_name}</td>
                            <td class="p-3 text-slate-400">${p.team}</td>
                            <td class="p-3 text-slate-300">${p.prop_name}</td>
                            <td class="p-3 text-sky-400 font-bold">${p.call} ${p.line} (${p.prob}%)</td>
                            <td class="p-3">${p.projected}</td>
                            <td class="p-3 text-slate-400">${p.line}</td>
                            <td class="p-3 font-bold text-white">${p.actual}</td>
                            <td class="p-3 text-slate-400">${p.error}</td>
                        </tr>
                    `;
                });
            }

            function filterModalProps(filter) {
                if (filter === 'all') {
                    renderModalProps(activeModalProps);
                } else {
                    renderModalProps(activeModalProps.filter(p => p.status === filter));
                }
            }

            function showModalTab(tab) {
                document.getElementById('modal-view-props').classList.toggle('hidden', tab !== 'props');
                document.getElementById('modal-view-lines').classList.toggle('hidden', tab !== 'lines');
                
                document.getElementById('modal-tab-props').className = tab === 'props' 
                    ? 'pb-3 border-b-2 border-emerald-400 text-emerald-400 font-semibold'
                    : 'pb-3 border-b-2 border-transparent text-slate-400 hover:text-white';
                document.getElementById('modal-tab-lines').className = tab === 'lines' 
                    ? 'pb-3 border-b-2 border-emerald-400 text-emerald-400 font-semibold'
                    : 'pb-3 border-b-2 border-transparent text-slate-400 hover:text-white';
            }

            function closeModal() {
                document.getElementById('game-modal').classList.add('hidden');
            }

            async function loadLiveGames() {
                const container = document.getElementById('live-container');
                container.innerHTML = '<p class="text-slate-500 col-span-full">Polling live scoreboard & situations...</p>';

                try {
                    const res = await fetch('/api/live/games');
                    const games = await res.json();
                    container.innerHTML = '';

                    if (!Array.isArray(games) || games.length === 0) {
                        container.innerHTML = '<p class="text-slate-500 col-span-full">No active games currently live.</p>';
                        return;
                    }

                    games.forEach(g => {
                        const isLive = g.state === 'in';
                        const card = document.createElement('div');
                        card.className = 'bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden cursor-pointer hover:border-emerald-500/50 transition';
                        card.onclick = () => openGameModal(g.game_id);
                        card.innerHTML = `
                            <div class="flex justify-between items-center text-xs mb-3">
                                <span class="font-bold px-2 py-0.5 rounded ${isLive ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse' : 'bg-slate-800 text-slate-400'}">
                                    ${isLive ? 'LIVE • Q' + g.period + ' ' + g.display_clock : g.state.toUpperCase()}
                                </span>
                                <span class="text-slate-400">${g.down_distance_text}</span>
                            </div>

                            <div class="flex justify-between items-center mb-6">
                                <div class="text-xl font-black">${g.away_team} <span class="text-2xl font-mono text-white ml-2">${g.away_score}</span></div>
                                <div class="text-slate-500 text-sm font-semibold">VS</div>
                                <div class="text-xl font-black"><span class="text-2xl font-mono text-white mr-2">${g.home_score}</span> ${g.home_team}</div>
                            </div>

                            <div class="space-y-3 bg-slate-950 p-4 rounded-lg mb-4 text-xs font-mono">
                                <div class="flex justify-between">
                                    <span class="text-slate-400">Live Win Prob:</span>
                                    <span class="font-bold text-emerald-400">${g.home_team} ${(g.home_win_prob * 100).toFixed(1)}% | ${g.away_team} ${(g.away_win_prob * 100).toFixed(1)}%</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-slate-400">Fair In-Game Margin:</span>
                                    <span class="font-bold ${g.fair_live_margin > 0 ? 'text-emerald-400' : 'text-rose-400'}">${g.home_team} ${g.fair_live_margin > 0 ? '+' : ''}${g.fair_live_margin.toFixed(1)}</span>
                                </div>
                            </div>

                            <div class="text-center text-xs font-semibold text-emerald-400 bg-slate-800/40 py-2 rounded-lg">
                                Click to Inspect Game & Player Props &rarr;
                            </div>
                        `;
                        container.appendChild(card);
                    });
                } catch (err) {
                    container.innerHTML = `<p class="text-rose-500 col-span-full">Live telemetry error: ${err.message}</p>`;
                }
            }

            async function loadGames() {
                const season = document.getElementById('select-season').value;
                const week = document.getElementById('select-week').value;
                const container = document.getElementById('games-container');
                container.innerHTML = '<p class="text-slate-500 col-span-full">Loading projections...</p>';

                try {
                    const res = await fetch(`/api/predictions/games?season=${season}&week=${week}`);
                    const games = await res.json();
                    container.innerHTML = '';
                    
                    if (!Array.isArray(games) || games.length === 0) {
                        container.innerHTML = `<p class="text-slate-500 col-span-full">No active fixtures found for Season ${season} Week ${week}.</p>`;
                        return;
                    }

                    games.forEach(g => {
                        const card = document.createElement('div');
                        card.className = 'bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg cursor-pointer hover:border-emerald-500/50 transition';
                        card.onclick = () => openGameModal(g.game_id);
                        card.innerHTML = `
                            <div class="flex justify-between items-center text-xs text-slate-400 mb-2">
                                <span>WEEK ${g.week}</span>
                                <span>SPREAD: ${g.spread_line}</span>
                            </div>
                            <div class="text-lg font-bold mb-4">${g.away_team} @ ${g.home_team}</div>
                            <div class="grid grid-cols-2 gap-2 text-xs bg-slate-950 p-3 rounded-lg mb-3">
                                <div>Proj Margin: <span class="font-bold ${g.pred_margin > 0 ? 'text-emerald-400' : 'text-rose-400'}">${g.pred_margin.toFixed(1)}</span></div>
                                <div>Proj Total: <span class="font-bold text-sky-400">${g.pred_total.toFixed(1)}</span></div>
                                <div>Home Win: <span class="font-bold">${(g.home_win_prob * 100).toFixed(1)}%</span></div>
                                <div>Away Win: <span class="font-bold">${(g.away_win_prob * 100).toFixed(1)}%</span></div>
                            </div>
                            <div class="text-center text-xs font-semibold text-emerald-400 bg-slate-800/40 py-2 rounded-lg">
                                Click to Inspect Game & Player Props &rarr;
                            </div>
                        `;
                        container.appendChild(card);
                    });
                } catch (err) {
                    container.innerHTML = `<p class="text-rose-500 col-span-full">Failed to load games: ${err.message}</p>`;
                }
            }

            async function loadAudit() {
                const tbody = document.getElementById('audit-table-body');
                tbody.innerHTML = '<tr><td colspan="8" class="p-4 text-center text-slate-500">Loading audit summaries...</td></tr>';
                try {
                    const res = await fetch('/api/audit');
                    const logs = await res.json();
                    tbody.innerHTML = '';
                    logs.forEach(l => {
                        const row = document.createElement('tr');
                        row.className = 'hover:bg-slate-800/50';
                        row.innerHTML = `
                            <td class="p-4">${l.season}</td><td class="p-4">${l.week}</td><td class="p-4">${l.total_games_audited}</td>
                            <td class="p-4 font-mono">${l.spread_mae.toFixed(2)}</td>
                            <td class="p-4 font-semibold ${l.spread_ats_win_rate >= 52.4 ? 'text-emerald-400' : 'text-rose-400'}">${l.spread_ats_win_rate.toFixed(1)}%</td>
                            <td class="p-4 font-mono">${l.moneyline_brier.toFixed(4)}</td><td class="p-4">${l.moneyline_accuracy.toFixed(1)}%</td><td class="p-4">${l.props_over_under_accuracy.toFixed(1)}%</td>
                        `;
                        tbody.appendChild(row);
                    });
                } catch (err) {
                    tbody.innerHTML = `<tr><td colspan="8" class="p-4 text-center text-rose-500">Failed to load: ${err.message}</td></tr>`;
                }
            }

            async function loadDetails() {
                const tbody = document.getElementById('details-table-body');
                tbody.innerHTML = '<tr><td colspan="9" class="p-4 text-center text-slate-500">Loading detailed ledger...</td></tr>';
                try {
                    const res = await fetch('/api/audit/details');
                    const picks = await res.json();
                    tbody.innerHTML = '';
                    picks.forEach(p => {
                        const row = document.createElement('tr');
                        row.className = 'hover:bg-slate-800/50';
                        let badge = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-slate-800 text-slate-400">PUSH</span>';
                        if (p.status === 'won') badge = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-emerald-950 text-emerald-400 border border-emerald-800">HIT</span>';
                        else if (p.status === 'lost') badge = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-rose-950 text-rose-400 border border-rose-800">MISS</span>';

                        row.innerHTML = `
                            <td class="p-4">${badge}</td><td class="p-4">W${p.week}</td><td class="p-4 uppercase text-xs text-slate-400">${p.category}</td>
                            <td class="p-4 font-semibold">${p.item}</td><td class="p-4 font-mono text-sky-400">${p.pick}</td>
                            <td class="p-4 font-mono">${p.projected}</td><td class="p-4 font-mono text-slate-400">${p.line}</td>
                            <td class="p-4 font-mono font-bold">${p.actual}</td><td class="p-4 font-mono text-slate-400">${p.error}</td>
                        `;
                        tbody.appendChild(row);
                    });
                } catch (err) {
                    tbody.innerHTML = `<tr><td colspan="9" class="p-4 text-center text-rose-500">Failed to load details: ${err.message}</td></tr>`;
                }
            }

            setInterval(() => {
                if (!document.getElementById('section-live').classList.contains('hidden') && document.getElementById('game-modal').classList.contains('hidden')) {
                    loadLiveGames();
                }
            }, 10000);

            loadLiveGames();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
