#!/usr/bin/env bash
set -e

echo "[STARTUP] Initializing directory hierarchy on persistent disk..."
mkdir -p data/raw data/processed data/models data/predictions

if [ -d "data_bundled" ]; then
    echo "[STARTUP] Syncing bundled models and datasets to persistent disk..."
    cp -rn data_bundled/* data/ 2>/dev/null || cp -r data_bundled/* data/ || true
fi

echo "[STARTUP] Verifying model artifacts..."
if [ -f "data/models/game_outcome_engine.joblib" ]; then
    echo "[STARTUP] Confirmed: game_outcome_engine.joblib is ready."
else
    echo "[WARNING] Model missing on disk. Generating fallback models..."
    python scripts/train_models.py || true
fi

echo "[STARTUP] Starting Uvicorn server on port ${PORT:-8000}..."
exec uvicorn src.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
