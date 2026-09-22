#!/usr/bin/env bash
set -e

echo "[STARTUP] Initializing directories on persistent disk..."
mkdir -p data/raw data/processed data/models data/predictions

echo "[STARTUP] Starting Uvicorn API server on port ${PORT:-8000}..."
exec uvicorn src.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
