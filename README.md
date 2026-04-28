# AlphaLab

Automated Factor Research & Portfolio Intelligence Platform

## What it does

Downloads stock data for 200+ NSE stocks, computes quantitative factors
(momentum, value, volatility, quality), backtests each factor using a
long-short portfolio, ranks them by Sharpe ratio, and combines the best
ones into a suggested portfolio — all automated daily.

## Quick start

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd alphalab

# 2. Start all services
docker-compose up

# 3. Open the UIs
# Airflow   → http://localhost:8080   (admin / admin)
# MLflow    → http://localhost:5000
# Grafana   → http://localhost:3001   (admin / admin)

# Airflow
# admin
# Cu2ce5x5TCnqzEch
```

## Project structure

```
alphalab/
├── src/
│   ├── ingestion/      Airflow DAGs — daily data fetch
│   ├── factors/        Factor engineering (momentum, value, etc.)
│   ├── backtest/       Long-short backtesting engine
│   ├── ranking/        Factor ranking + decay detection
│   ├── portfolio/      Composite scoring model
│   └── api/            FastAPI endpoints
├── frontend/           React dashboard (4 tabs)
├── data/
│   ├── raw/            OHLCV parquet files (DVC tracked)
│   └── features/       Computed factor scores (DVC tracked)
├── models/             Trained models (DVC tracked)
├── tests/              Unit + integration tests
└── docs/               HLD, LLD, test report, user manual
```

## Tech stack

| Purpose | Tool |
|---|---|
| Orchestration | Apache Airflow |
| Data versioning | DVC |
| Experiment tracking | MLflow |
| API | FastAPI |
| Frontend | React |
| Monitoring | Prometheus + Grafana |
| Containerisation | Docker Compose |
| Factor math | pandas + numpy |
| ML model | XGBoost + SHAP |

## Week-by-week build plan

- **Week 2** — Folder structure, Docker Compose, Airflow DAG, factor engineering, DVC
- **Week 3** — Backtesting engine, MLflow tracking, factor ranking, ML model
- **Week 4** — FastAPI, Prometheus, Grafana, React frontend, Docker
- **Week 5** — Docs, tests, demo polish



docker-compose build airflow
docker-compose up -d
docker-compose logs airflow | grep -i "password"
