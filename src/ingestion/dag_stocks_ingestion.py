import logging
import os
from datetime import datetime, timedelta
import sys
sys.path.append("/opt/airflow/dags")

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
import time

logger = logging.getLogger(__name__)

NSE_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "NTPC.NS", "TECHM.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "HDFCLIFE.NS", "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ONGC.NS",
    "PIDILITIND.NS", "HAVELLS.NS", "VOLTAS.NS", "BERGEPAINT.NS", "MARICO.NS",
    "DABUR.NS", "GODREJCP.NS", "COLPAL.NS", "BRITANNIA.NS", "TATACONSUM.NS",
    "ABCAPITAL.NS", "UNITDSPR.NS", "RADICO.NS", "VBL.NS", "JUBLFOOD.NS",
    "INDIGO.NS", "INDIAMART.NS", "IRCTC.NS", "CONCOR.NS", "APOLLOHOSP.NS",
    "MPHASIS.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "OFSS.NS",
    "KPITTECH.NS", "TATAELXSI.NS", "CYIENT.NS", "LTTS.NS", "MANKIND.NS",
    "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "RBLBANK.NS",
    "INDUSINDBK.NS", "PNB.NS", "CANBK.NS", "BANKBARODA.NS", "UNIONBANK.NS",
    "CHOLAFIN.NS", "MUTHOOTFIN.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "TVSMOTOR.NS",
    "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS", "LUPIN.NS",
    "BIOCON.NS", "TORNTPHARM.NS", "ALKEM.NS", "IPCALAB.NS", "ABBOTINDIA.NS",
    "BPCL.NS", "IOC.NS", "GAIL.NS", "COALINDIA.NS", "HINDALCO.NS",
    "JSWSTEEL.NS", "SAIL.NS", "NMDC.NS", "VEDL.NS", "NATIONALUM.NS",
    "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS", "RAMCOCEM.NS", "JKCEMENT.NS",
]

DATA_DIR = "/opt/airflow/data/raw"

# ─────────────────────────────────────────
# FETCH (UNCHANGED ✅)
# ─────────────────────────────────────────
def fetch_stock_data(**context) -> None:
    import pandas as pd
    import yfinance as yf

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




def fetch_fundamentals(**context):
    import pandas as pd
    import yfinance as yf
    import time

    execution_date = context["ds"]

    # load tickers from raw data
    raw_path = os.path.join(DATA_DIR, f"{execution_date}.parquet")
    df = pd.read_parquet(raw_path)
    tickers = df["ticker"].unique().tolist()

    logger.info(f"Fetching fundamentals for {len(tickers)} stocks")

    records = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info or {}

            record = {
                "ticker": ticker,
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "ps_ratio": info.get("priceToSalesTrailing12Months"),
                "market_cap": info.get("marketCap"),
                "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
            }

            records.append(record)
            logger.info(f"{ticker}: PE={record['pe_ratio']}")

            time.sleep(1)  # prevent rate limit

        except Exception as e:
            logger.warning(f"{ticker} failed: {e}")

    fundamentals_df = pd.DataFrame(records)

    # save
    out_path = os.path.join("/opt/airflow/data/features", f"fundamentals_{execution_date}.parquet")
    fundamentals_df.to_parquet(out_path, index=False)

    logger.info(f"Fundamentals saved to {out_path}")
# ─────────────────────────────────────────
# VALIDATION LOGIC ✅
# ─────────────────────────────────────────
def validate_ohlcv(file_path: str) -> dict:
    import pandas as pd

    report = {
        "file": file_path,
        "passed": True,
        "errors": [],
        "warnings": [],
    }

    if not os.path.exists(file_path):
        report["passed"] = False
        report["errors"].append(f"File not found: {file_path}")
        return report

    df = pd.read_parquet(file_path)

    if df.empty:
        report["passed"] = False
        report["errors"].append("DataFrame is empty")
        return report

    if df["ticker"].nunique() < 50:
        report["warnings"].append("Too few tickers")

    if (df["Close"] < 0).any():
        report["passed"] = False
        report["errors"].append("Negative prices found")

    return report


# ─────────────────────────────────────────
# VALIDATION TASK (AIRFLOW)
# ─────────────────────────────────────────
def validate_data(**context):
    execution_date = context["ds"]
    file_path = os.path.join(DATA_DIR, f"{execution_date}.parquet")

    logger.info(f"Validating file: {file_path}")

    report = validate_ohlcv(file_path)

    if not report["passed"]:
        raise ValueError(f"Validation failed: {report['errors']}")

    logger.info("Validation passed")


# ─────────────────────────────────────────
# DAG
# ─────────────────────────────────────────
default_args = {
    "owner": "alphalab",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="stock_ingestion",
    description="Daily OHLCV ingestion for NSE stocks via yfinance",
    schedule="0 9 * * 1-5",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "week2"],
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_stock_data",
        python_callable=fetch_stock_data,
    )

    fundamentals_task = PythonOperator(
        task_id="fetch_fundamentals",
        python_callable=fetch_fundamentals,
    )

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    # pipeline
    # fetch_task >> validate_task
    fetch_task >> fundamentals_task >> validate_task