#!/usr/bin/env bash
set -e

echo "[STARTUP] Checking for existing trained models..."
if [ ! -f "data/models/game_outcome_engine.joblib" ]; then
    echo "[COLD START] Seeding baseline database and training models..."
    python scripts/seed_database.py
    python scripts/build_features.py
    python scripts/train_models.py
else
    echo "[STARTUP] Model artifacts found. Bypassing cold start."
fi

echo "[STARTUP] Starting Uvicorn API server on port ${PORT:-8000}..."
exec uvicorn src.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
