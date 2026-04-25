"""
main.py — AlphaLab Week 4
===========================
FastAPI REST API serving factor research results to the React frontend.

ENDPOINTS
---------
  GET /health                  — liveness probe (always returns 200)
  GET /ready                   — readiness probe (200 when data is loaded)
  GET /metrics                 — Prometheus metrics (via instrumentator)
  GET /factors                 — factor leaderboard from factor_rankings.parquet
  GET /portfolio                — top 10 long + top 10 short with SHAP explanations
  GET /backtest/{factor_name}  — full P&L series for one factor

DATA LOADING STRATEGY
---------------------
All parquet files are loaded once at startup into module-level DataFrames.
The /ready endpoint returns 503 until all required files are loaded.
This avoids repeated disk I/O on every request.

CORS is enabled for all origins so the React dev server (localhost:3000)
can call the API (localhost:8000) without browser errors.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — resolve relative to this file so the API works from any cwd
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

PATHS = {
    "rankings":     os.path.join(BASE_DIR, "data/backtest/factor_rankings.parquet"),
    "backtest_summary": os.path.join(BASE_DIR, "data/backtest/factor_backtest_summary.parquet"),
    "scores":       os.path.join(BASE_DIR, "data/models/composite_scores.parquet"),
    "shap":         os.path.join(BASE_DIR, "data/models/shap_explanations.parquet"),
    "shap_global":  os.path.join(BASE_DIR, "data/models/shap_global_summary.parquet"),
    "all_factors":  os.path.join(BASE_DIR, "data/features/all_factors.parquet"),
}

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY DATA STORE
# ─────────────────────────────────────────────────────────────────────────────

class DataStore:
    """Holds all DataFrames loaded at startup."""
    rankings:          Optional[pd.DataFrame] = None
    backtest_summary:  Optional[pd.DataFrame] = None
    scores:            Optional[pd.DataFrame] = None
    shap:              Optional[pd.DataFrame] = None
    shap_global:       Optional[pd.DataFrame] = None
    all_factors:       Optional[pd.DataFrame] = None
    ready:             bool = False

store = DataStore()


def _load_parquet(path: str, name: str) -> Optional[pd.DataFrame]:
    """Loads a parquet file, logs a warning if missing."""
    if not os.path.exists(path):
        logger.warning(f"  {name}: file not found at {path}")
        return None
    df = pd.read_parquet(path)
    logger.info(f"  {name}: loaded {len(df)} rows from {path}")
    return df


def load_all_data():
    """Called at startup — loads all parquet files into the store."""
    logger.info("Loading data files...")
    store.rankings        = _load_parquet(PATHS["rankings"],         "rankings")
    store.backtest_summary= _load_parquet(PATHS["backtest_summary"], "backtest_summary")
    store.scores          = _load_parquet(PATHS["scores"],           "scores")
    store.shap            = _load_parquet(PATHS["shap"],             "shap")
    store.shap_global     = _load_parquet(PATHS["shap_global"],      "shap_global")
    store.all_factors     = _load_parquet(PATHS["all_factors"],      "all_factors")

    # Ready = minimum required files are present
    required = [store.rankings, store.scores]
    store.ready = all(df is not None for df in required)

    if store.ready:
        logger.info("✓ API is ready — all required data loaded")
    else:
        logger.warning("⚠ API not ready — some required files are missing")


# ─────────────────────────────────────────────────────────────────────────────
# APP LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_data()
    yield
    logger.info("API shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AlphaLab API",
    description="Factor research & portfolio construction API for NSE stocks",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics — exposes /metrics endpoint
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
    logger.info("Prometheus instrumentator enabled")
except ImportError:
    logger.warning("prometheus_fastapi_instrumentator not installed — /metrics unavailable")


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str

class ReadyResponse(BaseModel):
    status: str
    files_loaded: dict

class FactorRow(BaseModel):
    rank:           int
    factor:         str
    sharpe:         Optional[float]
    annual_return:  Optional[float]
    max_drawdown:   Optional[float]
    win_rate:       Optional[float]
    factor_decay:   Optional[float]
    status:         str

class FactorsResponse(BaseModel):
    factors:        list[FactorRow]
    total:          int
    active_count:   int
    decaying_count: int

class StockScore(BaseModel):
    ticker:                  str
    predicted_return:        Optional[float]
    composite_alpha_score:   Optional[float]
    alpha_rank:              Optional[int]
    recommendation:          str
    top_factor_1:            Optional[str]
    shap_1:                  Optional[float]
    top_factor_2:            Optional[str]
    shap_2:                  Optional[float]
    top_factor_3:            Optional[str]
    shap_3:                  Optional[float]
    explanation:             Optional[str]

class PortfolioResponse(BaseModel):
    long_portfolio:   list[StockScore]
    short_portfolio:  list[StockScore]
    total_stocks:     int
    generated_at:     str

class PnLPoint(BaseModel):
    date:            str
    cumulative_pnl:  float
    daily_return:    float

class BacktestResponse(BaseModel):
    factor:         str
    sharpe:         Optional[float]
    annual_return:  Optional[float]
    max_drawdown:   Optional[float]
    win_rate:       Optional[float]
    status:         Optional[str]
    pnl_series:     list[PnLPoint]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """Converts a value to float, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None

def _safe_int(val) -> Optional[int]:
    """Converts a value to int, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else int(f)
    except (TypeError, ValueError):
        return None


def _merge_shap(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges composite scores with SHAP explanations on ticker.
    If SHAP data is unavailable, returns scores_df unchanged.
    """
    if store.shap is None:
        return scores_df

    shap_cols = ["ticker", "top_factor_1", "shap_1", "top_factor_2",
                 "shap_2", "top_factor_3", "shap_3", "explanation"]
    available = [c for c in shap_cols if c in store.shap.columns]

    merged = scores_df.merge(store.shap[available], on="ticker", how="left")
    return merged


def _row_to_stock_score(row: pd.Series) -> StockScore:
    """Converts a DataFrame row to a StockScore response object."""
    return StockScore(
        ticker=                row["ticker"],
        predicted_return=      _safe_float(row.get("predicted_return")),
        composite_alpha_score= _safe_float(row.get("composite_alpha_score")),
        alpha_rank=            _safe_int(row.get("alpha_rank")),
        recommendation=        str(row.get("recommendation", "NEUTRAL")),
        top_factor_1=          row.get("top_factor_1"),
        shap_1=                _safe_float(row.get("shap_1")),
        top_factor_2=          row.get("top_factor_2"),
        shap_2=                _safe_float(row.get("shap_2")),
        top_factor_3=          row.get("top_factor_3"),
        shap_3=                _safe_float(row.get("shap_3")),
        explanation=           row.get("explanation"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """
    Liveness probe — always returns 200 if the process is running.
    Used by Docker / Kubernetes to know if the container is alive.
    """
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready():
    """
    Readiness probe — returns 200 only when all required data is loaded.
    Returns 503 if the API is still initialising or data files are missing.

    Used by the React frontend to show a loading screen on startup.
    """
    files_loaded = {
        "rankings":        store.rankings is not None,
        "scores":          store.scores is not None,
        "shap":            store.shap is not None,
        "backtest_summary":store.backtest_summary is not None,
    }

    if not store.ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status":       "not ready",
                "files_loaded": files_loaded,
            }
        )

    return {"status": "ready", "files_loaded": files_loaded}


@app.get("/factors", response_model=FactorsResponse, tags=["research"])
def get_factors(
    status: Optional[str] = None,
    limit:  int = 25,
):
    """
    Returns the factor leaderboard ranked by Sharpe ratio.

    Query params:
      status : filter by "active" | "decaying" | "weak" (optional)
      limit  : max rows to return (default 25)

    Used by Tab 1: Factor Leaderboard in the React frontend.
    """
    if store.rankings is None:
        raise HTTPException(
            status_code=503,
            detail="Factor rankings not loaded. Run factor_ranker.py first."
        )

    df = store.rankings.copy()

    if status:
        df = df[df["status"] == status]

    df = df.head(limit)

    factors = []
    for _, row in df.iterrows():
        factors.append(FactorRow(
            rank=          int(row.get("rank", 0)),
            factor=        str(row["factor"]),
            sharpe=        _safe_float(row.get("sharpe")),
            annual_return= _safe_float(row.get("annual_return")),
            max_drawdown=  _safe_float(row.get("max_drawdown")),
            win_rate=      _safe_float(row.get("win_rate")),
            factor_decay=  _safe_float(row.get("factor_decay")),
            status=        str(row.get("status", "unknown")),
        ))

    return FactorsResponse(
        factors=       factors,
        total=         len(store.rankings),
        active_count=  int((store.rankings["status"] == "active").sum()),
        decaying_count=int((store.rankings["status"] == "decaying").sum()),
    )


@app.get("/portfolio", response_model=PortfolioResponse, tags=["portfolio"])
def get_portfolio(top_n: int = 10):
    """
    Returns the top N long and top N short stocks with composite alpha
    scores and SHAP factor explanations.

    Query params:
      top_n : number of stocks per leg (default 10)

    Used by Tab 3: Portfolio View in the React frontend.
    """
    if store.scores is None:
        raise HTTPException(
            status_code=503,
            detail="Composite scores not loaded. Run composite_model.py first."
        )

    merged = _merge_shap(store.scores)

    long_df  = (
        merged[merged["recommendation"] == "LONG"]
        .sort_values("composite_alpha_score", ascending=False)
        .head(top_n)
    )
    short_df = (
        merged[merged["recommendation"] == "SHORT"]
        .sort_values("composite_alpha_score", ascending=True)
        .head(top_n)
    )

    from datetime import datetime, timezone
    return PortfolioResponse(
        long_portfolio=  [_row_to_stock_score(r) for _, r in long_df.iterrows()],
        short_portfolio= [_row_to_stock_score(r) for _, r in short_df.iterrows()],
        total_stocks=    len(merged),
        generated_at=    datetime.now(timezone.utc).isoformat(),
    )


@app.get("/backtest/{factor_name}", response_model=BacktestResponse, tags=["research"])
def get_backtest(factor_name: str):
    """
    Returns the full P&L time-series for a single factor backtest.

    Path param:
      factor_name : e.g. "mom_6m_rank", "value_composite_rank"

    The P&L series is reconstructed from the backtester's output parquet.
    If a per-factor P&L parquet exists in data/backtest/, it is loaded directly.
    Otherwise the summary metrics are returned with an empty pnl_series.

    Used by Tab 2: Factor Deep Dive in the React frontend.
    """
    if store.backtest_summary is None:
        raise HTTPException(
            status_code=503,
            detail="Backtest summary not loaded. Run backtester.py first."
        )

    # Validate factor name exists
    summary = store.backtest_summary
    if factor_name not in summary["factor"].values:
        valid = summary["factor"].tolist()
        raise HTTPException(
            status_code=404,
            detail=f"Factor '{factor_name}' not found. Valid factors: {valid}"
        )

    row = summary[summary["factor"] == factor_name].iloc[0]

    # ── Try to load per-factor P&L parquet ────────────────────────────────
    pnl_path = os.path.join(BASE_DIR, "data/backtest", f"{factor_name}_pnl.parquet")
    pnl_series = []

    if os.path.exists(pnl_path):
        pnl_df = pd.read_parquet(pnl_path)
        pnl_df["Date"] = pd.to_datetime(pnl_df["Date"])
        pnl_df = pnl_df.sort_values("Date")

        pnl_series = [
            PnLPoint(
                date=           r["Date"].strftime("%Y-%m-%d"),
                cumulative_pnl= round(float(r.get("cumulative_pnl", 0)), 6),
                daily_return=   round(float(r.get("daily_return", 0)), 6),
            )
            for _, r in pnl_df.iterrows()
        ]
        logger.info(f"Loaded {len(pnl_series)} P&L points for {factor_name}")
    else:
        logger.warning(f"No P&L parquet for {factor_name} — returning metrics only")

    # Get status from rankings if available
    status_val = None
    if store.rankings is not None and factor_name in store.rankings["factor"].values:
        status_val = store.rankings[
            store.rankings["factor"] == factor_name
        ]["status"].iloc[0]

    return BacktestResponse(
        factor=        factor_name,
        sharpe=        _safe_float(row.get("sharpe")),
        annual_return= _safe_float(row.get("annual_return")),
        max_drawdown=  _safe_float(row.get("max_drawdown")),
        win_rate=      _safe_float(row.get("win_rate")),
        status=        status_val,
        pnl_series=    pnl_series,
    )


@app.get("/factors/shap", tags=["research"])
def get_global_shap():
    """
    Returns global SHAP feature importance — mean |SHAP| per factor across
    all stocks. Shows which factors drive the composite model most.

    Used by Tab 2: Factor Deep Dive for the importance bar chart.
    """
    if store.shap_global is None:
        raise HTTPException(
            status_code=503,
            detail="SHAP global summary not loaded. Run shap_explainer.py first."
        )

    return store.shap_global.to_dict(orient="records")


@app.get("/stocks/{ticker}", tags=["portfolio"])
def get_stock(ticker: str):
    """
    Returns full factor profile + alpha score + SHAP explanation for one stock.

    Path param:
      ticker : e.g. "RELIANCE.NS", "ONGC.NS"

    Used by the Portfolio tab when a user clicks on a stock row.
    """
    ticker = ticker.upper()

    if store.scores is None:
        raise HTTPException(status_code=503, detail="Scores not loaded")

    score_row = store.scores[store.scores["ticker"] == ticker]
    if score_row.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    merged = _merge_shap(store.scores)
    row    = merged[merged["ticker"] == ticker].iloc[0]

    result = _row_to_stock_score(row)

    # Add raw factor values if all_factors is loaded
    if store.all_factors is not None:
        af_row = store.all_factors[store.all_factors["ticker"] == ticker]
        if not af_row.empty:
            factor_vals = af_row.iloc[0].drop("ticker").to_dict()
            # Clean NaN for JSON serialisation
            factor_vals = {
                k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                for k, v in factor_vals.items()
            }
            return {"score": result.model_dump(), "factors": factor_vals}

    return {"score": result.model_dump()}


# ─────────────────────────────────────────────────────────────────────────────
# DEV SERVER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,           # auto-reload on code changes during development
        log_level="info",
    )