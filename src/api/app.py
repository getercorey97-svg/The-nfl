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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProductionAPI")

app = FastAPI(
    title="NFL Autonomous Prediction Engine",
    description="State-of-the-Art NFL Game Outcome and 20-Player Prop Modeling Platform",
    version="1.1.0"
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
        "prop_engine_ready": prop_engine is not None and len(prop_engine.models) > 0
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
    if week is not None:
        filtered = filtered[filtered["week"] == week]

    if filtered.empty:
        return []

    predictions = game_engine.predict_game(filtered)
    return predictions.to_dict(orient="records")

@app.post("/api/predictions/prop", response_model=PropPredictionResponse)
def predict_single_prop(req: PropPredictionRequest):
    if not prop_engine:
        raise HTTPException(status_code=503, detail="Prop Predictive Engine is offline.")

    prop_matrix_path = PROCESSED_DATA_DIR / "props_features.parquet"
    if not prop_matrix_path.exists():
        raise HTTPException(status_code=404, detail="Prop feature matrix not found.")

    props_df = pd.read_parquet(prop_matrix_path)
    player_data = props_df[props_df["player_id"] == req.player_id]

    if player_data.empty:
        raise HTTPException(status_code=404, detail=f"Player ID {req.player_id} not found in feature baseline.")

    latest_profile = player_data.iloc[[-1]]
    pred_df = prop_engine.predict_prop(latest_profile, req.prop_name, req.market_line)
    return pred_df.iloc[0].to_dict()

@app.get("/api/audit", response_model=List[AuditRecord])
def get_post_mortem_audit():
    audit_file = PREDICTION_DIR / "post_mortem_audit_ledger.csv"
    if not audit_file.exists():
        return []
    df = pd.read_csv(audit_file)
    return df.to_dict(orient="records")

@app.get("/api/audit/details")
def get_audit_details(
    status: Optional[str] = Query(None, description="Filter by status: won, lost, push"),
    category: Optional[str] = Query(None, description="Filter by category: spread, total, prop"),
    week: Optional[int] = Query(None, description="Filter by week")
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
        <nav class="border-b border-slate-800 bg-slate-900/50 backdrop-blur px-6 py-4 flex justify-between items-center">
            <h1 class="text-xl font-bold tracking-tight text-emerald-400">NFL Prediction Engine</h1>
            <div class="flex space-x-4 text-sm font-medium">
                <button onclick="showTab('games')" class="hover:text-emerald-400 px-3 py-1">Game Forecaster</button>
                <button onclick="showTab('audit')" class="hover:text-emerald-400 px-3 py-1">Weekly Summary</button>
                <button onclick="showTab('details')" class="hover:text-emerald-400 px-3 py-1">Pick Grader (Hits & Misses)</button>
            </div>
        </nav>

        <main class="max-w-7xl mx-auto p-6">
            <!-- Game Predictions Section -->
            <section id="section-games">
                <div class="flex justify-between items-center mb-6">
                    <h2 class="text-2xl font-bold">Upcoming Game Projections & Edges</h2>
                    <button onclick="loadGames()" class="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-lg text-sm font-semibold">Refresh Board</button>
                </div>
                <div id="games-container" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"></div>
            </section>

            <!-- Weekly Summary Section -->
            <section id="section-audit" class="hidden">
                <h2 class="text-2xl font-bold mb-6">Weekly Empirical Performance Summary</h2>
                <div class="overflow-x-auto bg-slate-900 rounded-xl border border-slate-800">
                    <table class="w-full text-left border-collapse text-sm">
                        <thead>
                            <tr class="border-b border-slate-800 text-slate-400 bg-slate-900/80">
                                <th class="p-4">Season</th>
                                <th class="p-4">Week</th>
                                <th class="p-4">Games</th>
                                <th class="p-4">Spread MAE</th>
                                <th class="p-4">ATS Win %</th>
                                <th class="p-4">ML Brier</th>
                                <th class="p-4">ML Acc %</th>
                                <th class="p-4">Prop Accuracy %</th>
                            </tr>
                        </thead>
                        <tbody id="audit-table-body" class="divide-y divide-slate-800"></tbody>
                    </table>
                </div>
            </section>

            <!-- Detailed Pick Breakdown (Hits & Misses) -->
            <section id="section-details" class="hidden">
                <div class="flex flex-wrap justify-between items-center mb-6 gap-4">
                    <h2 class="text-2xl font-bold">Pick-by-Pick Performance Ledger</h2>
                    <div class="flex space-x-3 text-sm">
                        <select id="filter-status" onchange="loadDetails()" class="bg-slate-900 border border-slate-700 px-3 py-1.5 rounded-lg">
                            <option value="">All Outcomes</option>
                            <option value="won">Hits (Won)</option>
                            <option value="lost">Misses (Lost)</option>
                            <option value="push">Pushes</option>
                        </select>
                        <select id="filter-category" onchange="loadDetails()" class="bg-slate-900 border border-slate-700 px-3 py-1.5 rounded-lg">
                            <option value="">All Markets</option>
                            <option value="spread">Spreads</option>
                            <option value="total">Totals</option>
                            <option value="prop">Player Props</option>
                        </select>
                    </div>
                </div>

                <div class="overflow-x-auto bg-slate-900 rounded-xl border border-slate-800">
                    <table class="w-full text-left border-collapse text-sm">
                        <thead>
                            <tr class="border-b border-slate-800 text-slate-400 bg-slate-900/80">
                                <th class="p-4">Outcome</th>
                                <th class="p-4">Week</th>
                                <th class="p-4">Market</th>
                                <th class="p-4">Matchup / Player</th>
                                <th class="p-4">Recommendation</th>
                                <th class="p-4">Projected</th>
                                <th class="p-4">Line</th>
                                <th class="p-4">Actual</th>
                                <th class="p-4">Margin Error</th>
                            </tr>
                        </thead>
                        <tbody id="details-table-body" class="divide-y divide-slate-800"></tbody>
                    </table>
                </div>
            </section>
        </main>

        <script>
            function showTab(tab) {
                document.getElementById('section-games').classList.toggle('hidden', tab !== 'games');
                document.getElementById('section-audit').classList.toggle('hidden', tab !== 'audit');
                document.getElementById('section-details').classList.toggle('hidden', tab !== 'details');
                if (tab === 'games') loadGames();
                if (tab === 'audit') loadAudit();
                if (tab === 'details') loadDetails();
            }

            async function loadGames() {
                const res = await fetch('/api/predictions/games?season=2024&week=2');
                const games = await res.json();
                const container = document.getElementById('games-container');
                container.innerHTML = '';
                
                if (games.length === 0) {
                    container.innerHTML = '<p class="text-slate-500 col-span-full">No active game fixtures found.</p>';
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
            }

            async function loadAudit() {
                const res = await fetch('/api/audit');
                const logs = await res.json();
                const tbody = document.getElementById('audit-table-body');
                tbody.innerHTML = '';

                if (logs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="8" class="p-4 text-center text-slate-500">No empirical weekly summaries available yet.</td></tr>';
                    return;
                }

                logs.forEach(l => {
                    const row = document.createElement('tr');
                    row.className = 'hover:bg-slate-800/50';
                    row.innerHTML = `
                        <td class="p-4">${l.season}</td>
                        <td class="p-4">${l.week}</td>
                        <td class="p-4">${l.total_games_audited}</td>
                        <td class="p-4 font-mono">${l.spread_mae.toFixed(2)}</td>
                        <td class="p-4 font-semibold ${l.spread_ats_win_rate >= 52.4 ? 'text-emerald-400' : 'text-rose-400'}">${l.spread_ats_win_rate.toFixed(1)}%</td>
                        <td class="p-4 font-mono">${l.moneyline_brier.toFixed(4)}</td>
                        <td class="p-4">${l.moneyline_accuracy.toFixed(1)}%</td>
                        <td class="p-4">${l.props_over_under_accuracy.toFixed(1)}%</td>
                    `;
                    tbody.appendChild(row);
                });
            }

            async function loadDetails() {
                const status = document.getElementById('filter-status').value;
                const category = document.getElementById('filter-category').value;
                const params = new URLSearchParams();
                if (status) params.append('status', status);
                if (category) params.append('category', category);

                const res = await fetch('/api/audit/details?' + params.toString());
                const picks = await res.json();
                const tbody = document.getElementById('details-table-body');
                tbody.innerHTML = '';

                if (picks.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="9" class="p-4 text-center text-slate-500">No individual picks found matching criteria.</td></tr>';
                    return;
                }

                picks.forEach(p => {
                    const row = document.createElement('tr');
                    row.className = 'hover:bg-slate-800/50';
                    
                    let statusBadge = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-slate-800 text-slate-400">PUSH</span>';
                    if (p.status === 'won') {
                        statusBadge = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-emerald-950 text-emerald-400 border border-emerald-800">HIT</span>';
                    } else if (p.status === 'lost') {
                        statusBadge = '<span class="px-2 py-0.5 rounded text-xs font-bold bg-rose-950 text-rose-400 border border-rose-800">MISS</span>';
                    }

                    row.innerHTML = `
                        <td class="p-4">${statusBadge}</td>
                        <td class="p-4">W${p.week}</td>
                        <td class="p-4 uppercase text-xs font-semibold text-slate-400">${p.category}</td>
                        <td class="p-4 font-semibold">${p.item}</td>
                        <td class="p-4 font-mono text-sky-400">${p.pick}</td>
                        <td class="p-4 font-mono">${p.projected}</td>
                        <td class="p-4 font-mono text-slate-400">${p.line}</td>
                        <td class="p-4 font-mono font-bold">${p.actual}</td>
                        <td class="p-4 font-mono text-slate-400">${p.error}</td>
                    `;
                    tbody.appendChild(row);
                });
            }

            loadGames();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
