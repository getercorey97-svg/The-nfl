import json
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.config import PREDICTION_DIR, PROCESSED_DATA_DIR
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
    version="2.0.0"
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

@app.get("/api/live/games")
def get_live_games():
    """Fetches real-time scores, situation, live win probabilities, and live line edges."""
    raw_games = LiveNFLDataLoader.fetch_live_scoreboard()
    processed_live = []

    for g in raw_games:
        # Compute live math
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

        spread_edge = round(wp_res["fair_live_margin"] + market_spread, 1)
        total_edge = round(live_total_proj - market_total, 1) if market_total > 0 else 0.0

        processed_live.append({
            **g,
            **wp_res,
            "projected_live_total": live_total_proj,
            "live_spread_edge": spread_edge,
            "live_total_edge": total_edge
        })

    return processed_live

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
                        <p class="text-xs text-slate-400 mt-1">Autonomous telemetry tracking score differential, field position, and in-game market edges</p>
                    </div>
                    <button onclick="loadLiveGames()" class="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-lg text-sm font-semibold flex items-center gap-2">
                        <span>Refresh Telemetry</span>
                    </button>
                </div>
                <div id="live-container" class="grid grid-cols-1 md:grid-cols-2 gap-6"></div>
            </section>

            <!-- PRE-MATCH FORECASTER -->
            <section id="section-games" class="hidden">
                <div class="flex flex-wrap justify-between items-center mb-6 gap-4">
                    <h2 class="text-2xl font-bold">Pre-Match Projections & Edges</h2>
                    <div class="flex items-center space-x-3 text-sm">
                        <select id="select-season" onchange="loadGames()" class="bg-slate-900 border border-slate-700 px-3 py-1.5 rounded-lg text-white">
                            <option value="2026" selected>2026 Season</option>
                            <option value="2025">2025 Season</option>
                            <option value="2024">2024 Season</option>
                        </select>
                        <select id="select-week" onchange="loadGames()" class="bg-slate-900 border border-slate-700 px-3 py-1.5 rounded-lg text-white">
                            <option value="3" selected>Week 3</option>
                            <option value="1">Week 1</option>
                            <option value="2">Week 2</option>
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

        <script>
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
                        card.className = 'bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden';
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
                                <div class="flex justify-between">
                                    <span class="text-slate-400">Live Total Projection:</span>
                                    <span class="font-bold text-sky-400">${g.projected_live_total.toFixed(1)} pts</span>
                                </div>
                            </div>

                            <div class="grid grid-cols-2 gap-3 text-center text-xs">
                                <div class="bg-slate-800/60 p-2.5 rounded-lg">
                                    <div class="text-slate-400 mb-1">Live Spread Edge</div>
                                    <div class="font-bold text-sm ${Math.abs(g.live_spread_edge) > 1.5 ? 'text-emerald-400' : 'text-slate-300'}">
                                        ${g.live_spread_edge > 0 ? '+' : ''}${g.live_spread_edge.toFixed(1)} pts
                                    </div>
                                </div>
                                <div class="bg-slate-800/60 p-2.5 rounded-lg">
                                    <div class="text-slate-400 mb-1">Live Total Edge</div>
                                    <div class="font-bold text-sm ${Math.abs(g.live_total_edge) > 1.5 ? 'text-emerald-400' : 'text-slate-300'}">
                                        ${g.live_total_edge > 0 ? '+' : ''}${g.live_total_edge.toFixed(1)} pts
                                    </div>
                                </div>
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
                        card.className = 'bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg';
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
                            <div class="text-xs font-semibold ${Math.abs(g.spread_edge) > 1.5 ? 'text-emerald-400' : 'text-slate-500'}">
                                Edge: ${g.spread_edge > 0 ? '+' : ''}${g.spread_edge.toFixed(1)} pts
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

            // Auto-refresh live telemetry every 10 seconds if on Live tab
            setInterval(() => {
                if (!document.getElementById('section-live').classList.contains('hidden')) {
                    loadLiveGames();
                }
            }, 10000);

            loadLiveGames();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
