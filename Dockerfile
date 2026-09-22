FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# Install runtime libraries for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, scripts, and bundled datasets/models
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data_bundled/

# Ensure start script is executable
RUN chmod +x ./scripts/render_start.sh

EXPOSE 8000

CMD ["./scripts/render_start.sh"]
