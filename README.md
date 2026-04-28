# AlphaLab 🧪📈

> **A full-stack MLOps platform for quantitative factor research on NSE stocks.**
> Built with FastAPI · XGBoost · MLflow · DVC · Airflow · Prometheus · Grafana · React

---

## What is AlphaLab?

AlphaLab is a production-grade quantitative research system that automatically ingests daily stock data for 100 NSE-listed stocks, computes 21 factor signals, trains a composite ranking model, and serves predictions through a REST API with a React dashboard.

Think of it as a mini hedge fund research desk — fully automated, fully monitored, running on your laptop.

```
Daily data (yfinance) → Factor engine (DVC) → XGBoost model (MLflow)
       → FastAPI serving → React dashboard + Grafana monitoring
```

---

## Features

| Area | What it does |
|---|---|
| 📥 **Data ingestion** | Airflow DAG pulls OHLCV data for 100 NSE stocks every weekday at 9AM |
| 🧮 **Factor engine** | 21 quantitative factors across momentum, value, volatility, mean reversion |
| 🤖 **ML model** | XGBoost ranker trained on factor ranks, registered in MLflow Model Registry |
| 🔍 **Explainability** | SHAP values show which factors drove each stock's recommendation |
| 🚀 **REST API** | FastAPI with `/factors`, `/portfolio`, `/backtest`, `/health`, `/metrics` |
| 🖥️ **Dashboard** | React frontend with factor leaderboard, P&L charts, and portfolio view |
| 📊 **Monitoring** | Prometheus + Grafana with 13 panels tracking latency, errors, and drift |
| 🐳 **Containers** | Full stack runs in Docker Compose — one command to start everything |

---

## Prerequisites

Make sure you have these installed:

- **Docker** and **Docker Compose** (v2+)
- **Python 3.10+** (for running the frontend and local development)
- **Node.js 18+** and **npm** (for the React frontend)
- **Git** and **DVC**

```bash
# Verify
docker --version        # Docker version 24+
docker compose version  # Docker Compose version v2+
python3 --version       # Python 3.10+
node --version          # v18+
dvc --version           # 3+
```

---

## Project Structure

```
Alphalab/
├── dags/                        # Airflow DAG definitions
│   ├── alphalab_ingestion.py    # Daily data ingestion DAG
│   └── baseline_refresh_dag.py # Monthly drift baseline refresh
│
├── data/                        # All data (DVC tracked)
│   ├── raw/                     # Raw OHLCV parquets
│   ├── features/                # Factor parquets + baseline stats
│   ├── backtest/                # Factor P&L and rankings
│   └── models/                  # Composite scores + SHAP outputs
│
├── grafana/
│   └── provisioning/
│       ├── datasources/         # Auto-connect to Prometheus
│       └── dashboards/          # AlphaLab dashboard JSON
│
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   ├── factors/                 # Factor computation modules
│   │   ├── momentum.py
│   │   ├── value.py
│   │   ├── volatility.py
│   │   ├── mean_reversion.py
│   │   └── combine_factors.py
│   ├── models/                  # XGBoost training + SHAP
│   ├── monitoring/
│   │   └── calculate_baseline.py
│   └── tests/                   # pytest test suite (40 tests)
│
├── alerts/
│   └── alphalab_alerts.yml      # Prometheus alert rules
│
├── Dockerfile                   # FastAPI container
├── docker-compose.yml           # Full stack orchestration
├── prometheus.yml               # Prometheus config
├── alertmanager.yml             # Alertmanager config
├── dvc.yaml                     # DVC pipeline definition
└── requirements.txt
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/your-username/Alphalab.git
cd Alphalab
git checkout mlops
```

### 2. Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Pull data with DVC

```bash
dvc pull
```

> If this is a fresh clone with no remote, run the DVC pipeline instead:
> ```bash
> dvc repro
> ```
> This runs the full factor computation pipeline and generates all parquet files.

### 4. Start the backend stack

```bash
# Build the API image (first time only — takes ~2 min)
docker compose build api

# Start all services
docker compose up -d
```

This starts:
| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| FastAPI docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Alertmanager | http://localhost:9093 |
| Airflow | http://localhost:8080 |

### 5. Start the frontend

```bash
cd frontend          # or wherever your React app lives
npm install          # first time only
npm start
```

Frontend opens at **http://localhost:3001**

> ⚠️ Grafana runs on port 3000. React uses port 3001 automatically if 3000 is taken.

---

## Daily Usage

Once everything is set up, your daily workflow is simply:

```bash
# Start everything
docker compose up -d
cd frontend && npm start

# Stop everything
docker compose down
```

No rebuild needed unless you change `requirements.txt` or `Dockerfile`.

---

## The Three Views

### 🏆 Factor Leaderboard
See all 21 factors ranked by Sharpe ratio. Filter by status (active / decaying / weak). Each factor shows annual return, max drawdown, and win rate from backtesting.

### 📉 Factor Deep Dive
Click any factor to see its full cumulative P&L chart over the backtest period. The global SHAP importance chart shows which factors the composite model relies on most.

### 💼 Portfolio View
The main output — top 10 stocks to **long** and top 10 to **short**, each with:
- Composite alpha score and predicted return rank
- Top 3 SHAP factor drivers ("this stock is ranked #1 mainly because of strong 6-month momentum")
- Click any stock to see its full 21-factor profile

---

## API Reference

The API is self-documenting. Open **http://localhost:8000/docs** for the interactive Swagger UI.

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check — always 200 if process is running |
| `GET /ready` | Readiness check — 200 when all data files are loaded |
| `GET /metrics` | Prometheus metrics |
| `GET /factors?status=active&limit=25` | Factor leaderboard |
| `GET /portfolio?top_n=10` | Long/short portfolio with SHAP |
| `GET /backtest/{factor_name}` | Full P&L series for one factor |
| `GET /factors/shap` | Global SHAP feature importance |
| `GET /stocks/{ticker}` | Full factor profile for one stock |

**Example:**
```bash
# Get top 5 active factors
curl "http://localhost:8000/factors?status=active&limit=5"

# Get portfolio (top 10 long + short)
curl "http://localhost:8000/portfolio?top_n=10"

# Get backtest for 6-month momentum
curl "http://localhost:8000/backtest/mom_6m_rank"
```

---

## Monitoring

### Grafana Dashboard
Open **http://localhost:3000** → login with `admin / alphalab123`

The dashboard auto-loads with 13 panels across 5 sections:
- **API Health** — status, request rate, error rate, p95 latency
- **Request Metrics** — per-endpoint breakdown, latency percentiles
- **Factor Performance** — factor scores and drift PSI
- **Infrastructure** — CPU, memory, disk gauges
- **Active Alerts** — live Prometheus alert table

### Prometheus Alerts
Configured alert rules (fire automatically):

| Alert | Triggers when |
|---|---|
| `HighErrorRate` | HTTP 5xx errors > 5% for 2 min |
| `HighLatencyP95` | p95 latency > 500ms for 3 min |
| `APIDown` | API unreachable for 2 min |
| `FactorDriftDetected` | PSI score > 0.2 on any factor |
| `HighCPUUsage` | CPU > 85% for 5 min |
| `LowDiskSpace` | Disk < 15% free |

---

## Running the ML Pipeline

To regenerate all factor data and retrain the model from scratch:

```bash
# Run the full DVC pipeline
dvc repro

# Or run individual stages
dvc repro compute_momentum
dvc repro combine_factors
dvc repro train_model
```

Track experiments in MLflow:
```bash
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000
```

---

## Running Tests

```bash
# Run all 40 tests
pytest src/tests/ -v

# With coverage report
pytest src/tests/ --cov=src --cov-report=term-missing

# Run one file
pytest src/tests/test_momentum.py -v
```

---

## Compute Drift Baseline

Run this once after first setup, then it refreshes automatically every month via Airflow:

```bash
python src/monitoring/calculate_baseline.py
```

Outputs `data/features/baseline_stats.parquet` with 16 statistics per factor.

---

## Troubleshooting

**Grafana shows "No data"**
```bash
# Check Prometheus can reach the API
curl http://localhost:9090/api/v1/query?query=up
# Should show alphalab_api with value 1

# If datasource error — change URL in Grafana UI:
# Connections → Data sources → prometheus-1
# Change URL from localhost:9090 to http://prometheus:9090
```

**API not starting**
```bash
# Check logs
docker compose logs api --tail=50

# Most common cause: data files missing
# Fix: run dvc repro first, then restart
dvc repro
docker compose restart api
```

**Frontend can't reach API (CORS error)**
```bash
# Confirm API is running
curl http://localhost:8000/health
# Should return {"status": "ok"}

# Check your frontend API base URL points to localhost:8000
```

**DVC pull fails**
```bash
# If no remote is configured, reproduce from scratch
dvc repro --no-run-cache
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | Apache Airflow, yfinance |
| Feature engineering | Python, pandas, DVC |
| Model training | XGBoost, SHAP, MLflow |
| Model serving | FastAPI, uvicorn |
| Containerisation | Docker, Docker Compose |
| Monitoring | Prometheus, Grafana, Alertmanager |
| Frontend | React, JavaScript |
| Version control | Git, DVC |

---

## Author

**Swara Patil**
Indian Institute of Technology Madras (IITM)
B.Tech Project — MLOps

---

> *"The goal of AlphaLab is not to predict the market perfectly — it is to predict it systematically."*