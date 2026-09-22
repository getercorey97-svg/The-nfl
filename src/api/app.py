import json
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
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
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model references
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
    """Retrieves upcoming game predictions and market edges."""
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
    """Calculates EV and Over/Under probabilities for a specific prop market and line."""
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
    """Returns simulation-free historical audit metrics comparing forecasts against empirical outcomes."""
    audit_file = PREDICTION_DIR / "post_mortem_audit_ledger.csv"
    if not audit_file.exists():
        return []

    df = pd.read_csv(audit_file)
    return df.to_dict(orient="records")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the complete front-end dashboard interface with prediction boards and audit ledgers."""
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
            <h1 class="text-xl font-bold tracking-tight text-emerald-400">NFL AI Prediction Engine</h1>
            <div class="flex space-x-4 text-sm font-medium">
                <button onclick="showTab('games')" class="hover:text-emerald-400 px-3 py-1">Game Forecaster</button>
                <button onclick="showTab('audit')" class="hover:text-emerald-400 px-3 py-1">Performance Audit</button>
            </div>
        </nav>

        <main class="max-w-7xl mx-auto p-6">
            <!-- Game Predictions Section -->
            <section id="section-games">
                <div class="flex justify-between items-center mb-6">
                    <h2 class="text-2xl font-bold">Upcoming Game Projections & Edges</h2>
                    <button onclick="loadGames()" class="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-lg text-sm font-semibold">Refresh Board</button>
                </div>
                <div id="games-container" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <!-- Cards dynamically injected -->
                </div>
            </section>

            <!-- Performance Audit Section -->
            <section id="section-audit" class="hidden">
                <h2 class="text-2xl font-bold mb-6">Empirical Post-Mortem Performance Audit</h2>
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
                        <tbody id="audit-table-body" class="divide-y divide-slate-800">
                            <!-- Audit rows dynamically injected -->
                        </tbody>
                    </table>
                </div>
            </section>
        </main>

        <script>
            function showTab(tab) {
                document.getElementById('section-games').classList.toggle('hidden', tab !== 'games');
                document.getElementById('section-audit').classList.toggle('hidden', tab !== 'audit');
                if (tab === 'games') loadGames();
                if (tab === 'audit') loadAudit();
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
                            <div>Home Win Prob: <span class="font-bold">${(g.home_win_prob * 100).toFixed(1)}%</span></div>
                            <div>Away Win Prob: <span class="font-bold">${(g.away_win_prob * 100).toFixed(1)}%</span></div>
                        </div>
                        <div class="text-xs font-semibold ${Math.abs(g.spread_edge) > 1.5 ? 'text-emerald-400' : 'text-slate-500'}">
                            Market Edge: ${g.spread_edge > 0 ? '+' : ''}${g.spread_edge.toFixed(1)} pts
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
                    tbody.innerHTML = '<tr><td colspan="8" class="p-4 text-center text-slate-500">No empirical audit logs available yet.</td></tr>';
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

            loadGames();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
