"""
dag_stock_ingestion.py
─────────────────────
Airflow DAG: downloads daily OHLCV data for NSE stocks and saves as Parquet.

WHAT IS A DAG?
  DAG = Directed Acyclic Graph. In plain English:
  - A list of tasks
  - With a defined order (task B runs after task A)
  - No loops (acyclic — it never goes backwards)

  Airflow reads this file, registers the DAG, and runs it on the schedule
  you define. You never call this file with `python` yourself.

HOW AIRFLOW FINDS THIS FILE:
  Our docker-compose.yml mounts src/ingestion/ to /opt/airflow/dags/ inside
  the container. Airflow scans that folder every 30 seconds for .py files
  containing a DAG object.

SCHEDULE:
  Runs every day at 9:00 AM UTC.
  `catchup=False` means if the server was off for 3 days, it does NOT try
  to backfill those 3 missed runs — it just runs once now.
"""



import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from airflow import DAG
# from airflow.operators.python import PythonOperator
from airflow.providers.standard.operators.python import PythonOperator
# import requests
import time

# ── Logger ────────────────────────────────────────────────────────────────────
# Always use Python's logging module, never print().
# Airflow captures log output and shows it in the UI per task run.
logger = logging.getLogger(__name__)

# ── Stock universe ─────────────────────────────────────────────────────────────
# 100 NSE stocks across sectors. The ".NS" suffix is how yfinance identifies
# NSE-listed stocks (as opposed to BSE which uses ".BO").
# We'll expand this to 200 in Week 3 once the pipeline is stable.
NSE_STOCKS = [
    # Large cap — Nifty 50 core
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "NTPC.NS", "TECHM.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "HDFCLIFE.NS", "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ONGC.NS",

    # Mid cap — diversified sectors
    "PIDILITIND.NS", "HAVELLS.NS", "VOLTAS.NS", "BERGEPAINT.NS", "MARICO.NS",
    "DABUR.NS", "GODREJCP.NS", "COLPAL.NS", "BRITANNIA.NS", "TATACONSUM.NS",
    "ABCAPITAL.NS", "UNITDSPR.NS", "RADICO.NS", "VBL.NS", "JUBLFOOD.NS",
    "INDIGO.NS", "INDIAMART.NS", "IRCTC.NS", "CONCOR.NS", "APOLLOHOSP.NS",

    # IT & Tech
    "MPHASIS.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "OFSS.NS",
    "KPITTECH.NS", "TATAELXSI.NS", "CYIENT.NS", "LTTS.NS", "MANKIND.NS",

    # Banking & Finance
    "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "RBLBANK.NS",
    "INDUSINDBK.NS", "PNB.NS", "CANBK.NS", "BANKBARODA.NS", "UNIONBANK.NS",
    "CHOLAFIN.NS", "MUTHOOTFIN.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "TVSMOTOR.NS",

    # Pharma & Healthcare
    "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS", "LUPIN.NS",
    "BIOCON.NS", "TORNTPHARM.NS", "ALKEM.NS", "IPCALAB.NS", "ABBOTINDIA.NS",

    # Energy & Infra
    "BPCL.NS", "IOC.NS", "GAIL.NS", "COALINDIA.NS", "HINDALCO.NS",
    "JSWSTEEL.NS", "SAIL.NS", "NMDC.NS", "VEDL.NS", "NATIONALUM.NS",

    # Cement & Materials
    "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS", "RAMCOCEM.NS", "JKCEMENT.NS",
]

# ── Output path ────────────────────────────────────────────────────────────────
# Inside the Airflow container, /opt/airflow/data/ maps to our local data/ folder
# (set up in docker-compose.yml volumes).
DATA_DIR = "/opt/airflow/data/raw"


# ── Task functions ─────────────────────────────────────────────────────────────
# Each function below becomes one task in the DAG.
# Keep tasks small and focused — one job per task.

# def fetch_stock_data(**context) -> None:




def fetch_stock_data(**context) -> None:
    """
    Task 1: Download OHLCV data for all stocks and save as Parquet.

    WHAT IS OHLCV?
      Open   — price at market open
      High   — highest price during the day
      Low    — lowest price during the day
      Close  — price at market close (this is what most factors use)
      Volume — number of shares traded

    WHY 2 YEARS OF HISTORY?
      Factor backtests need enough history to be statistically meaningful.
      2 years = ~500 trading days per stock — enough for momentum, volatility,
      and mean reversion factors. We'll extend to 5 years in Week 3.

    ABOUT **context:
      Airflow passes a context dict to every PythonOperator function.
      It contains things like the run date, task instance, etc.
      We use context["ds"] to get the execution date as a string (YYYY-MM-DD).
    """
    # import time

# def fetch_stock_data(**context) -> None:
    execution_date = context["ds"]
    logger.info(f"Starting ingestion for execution date: {execution_date}")

    os.makedirs(DATA_DIR, exist_ok=True)

    all_data = []
    failed_stocks = []

    BATCH_SIZE = 10
    for i in range(0, len(NSE_STOCKS), BATCH_SIZE):
        batch = NSE_STOCKS[i:i + BATCH_SIZE]
        logger.info(f"Downloading batch {i//BATCH_SIZE + 1}: {batch}")

        try:
            df = yf.download(
                batch,
                period="2y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if df.empty:
                logger.warning(f"No data for batch {batch}")
                failed_stocks.extend(batch)
                continue

            df = df.stack(level=1, future_stack=True).rename_axis(["Date", "ticker"]).reset_index()
            all_data.append(df)
            logger.info(f"  Batch done: {len(df)} rows")

        except Exception as e:
            logger.error(f"Batch failed {batch}: {e}")
            failed_stocks.extend(batch)

        time.sleep(2)

    if not all_data:
        raise ValueError("No data downloaded for any stock — aborting")

    combined = pd.concat(all_data, ignore_index=True)
    output_path = os.path.join(DATA_DIR, f"{execution_date}.parquet")
    combined.to_parquet(output_path, index=False)

    logger.info(f"Saved {len(combined)} rows to {output_path}")
    logger.info(f"Failed stocks ({len(failed_stocks)}): {failed_stocks}")


def validate_data(**context) -> None:
    """
    Task 2: Basic sanity checks on the data we just downloaded.

    WHY VALIDATE?
      yfinance sometimes returns corrupt data, missing columns, or all-zero
      prices (especially for illiquid stocks). We catch this early rather
      than letting bad data silently corrupt our factor calculations.

    This is a simple validation — Week 2 Day 3 adds more thorough checks.
    """
    execution_date = context["ds"]
    file_path = os.path.join(DATA_DIR, f"{execution_date}.parquet")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expected file not found: {file_path}")

    df = pd.read_parquet(file_path)

    # ── Check 1: Required columns exist ───────────────────────────────────
    required_columns = {"Date", "Open", "High", "Low", "Close", "Volume", "ticker"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    # ── Check 2: Not empty ─────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Data file is empty")

    # ── Check 3: No all-null Close prices ─────────────────────────────────
    null_close = df["Close"].isnull().sum()
    null_pct = null_close / len(df) * 100
    if null_pct > 10:
        raise ValueError(f"Too many null Close prices: {null_pct:.1f}%")

    # ── Check 4: Prices are positive ──────────────────────────────────────
    negative_prices = (df["Close"] < 0).sum()
    if negative_prices > 0:
        raise ValueError(f"Found {negative_prices} negative Close prices")

    # ── Check 5: Enough stocks made it through ─────────────────────────────
    unique_tickers = df["ticker"].nunique()
    if unique_tickers < 80:   # allow up to 20 failures out of 100
        raise ValueError(f"Only {unique_tickers} stocks in data — expected 80+")

    logger.info(f"Validation passed:")
    logger.info(f"  Rows      : {len(df):,}")
    logger.info(f"  Stocks    : {unique_tickers}")
    logger.info(f"  Date range: {df['Date'].min()} → {df['Date'].max()}")
    logger.info(f"  Null Close: {null_pct:.2f}%")


# ── DAG definition ─────────────────────────────────────────────────────────────
# This is where we wire everything together.
# default_args apply to every task unless a task overrides them.

default_args = {
    "owner": "alphalab",
    "depends_on_past": False,       # don't wait for yesterday's run to succeed
    "email_on_failure": False,      # no email alerts (we use Prometheus instead)
    "email_on_retry": False,
    "retries": 2,                   # retry failed tasks twice before giving up
    "retry_delay": timedelta(minutes=5),  # wait 5 min between retries
}

with DAG(
    dag_id="stock_ingestion",           # unique name — shows in Airflow UI
    description="Daily OHLCV ingestion for NSE stocks via yfinance",
    schedule="0 9 * * 1-5",            # 9:00 AM UTC, Monday–Friday only
    #          │ │ │ │ └── day of week (1=Mon, 5=Fri)
    #          │ │ │ └──── month (every month)
    #          │ │ └────── day of month (every day)
    #          │ └──────── hour (9 AM UTC)
    #          └────────── minute (0)
    start_date=datetime(2024, 1, 1),
    catchup=False,                      # don't backfill missed runs
    default_args=default_args,
    tags=["ingestion", "week2"],        # labels in the Airflow UI
) as dag:

    # Task 1: fetch data
    fetch_task = PythonOperator(
        task_id="fetch_stock_data",
        python_callable=fetch_stock_data,
    )

    # Task 2: validate data
    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    # ── Task ordering ──────────────────────────────────────────────────────
    # The >> operator means "then". This reads: fetch first, then validate.
    # Airflow will not run validate_task if fetch_task fails.
    fetch_task >> validate_task