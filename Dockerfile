# ─────────────────────────────────────────────────────────────────────────────
# AlphaLab FastAPI — Dockerfile
# Build context: Alphalab/  (project root)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# Prevent .pyc files and enable unbuffered logs (visible in docker logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed by pandas/pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Copy data directory (parquet files needed at startup)
COPY data/ ./data/

# Expose FastAPI port
EXPOSE 8000

# Run with uvicorn — single worker is fine for this project
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]