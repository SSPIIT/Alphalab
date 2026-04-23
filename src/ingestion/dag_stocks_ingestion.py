

import logging
import os
from datetime import datetime, timedelta
import sys
sys.path.append("/opt/airflow/dags")
# import pandas as pd
# import yfinance as yf
# from airflow import DAG
from airflow.sdk import DAG
# from airflow.operators.python import PythonOperator
from airflow.providers.standard.operators.python import PythonOperator
# import requests
import time

logger = logging.getLogger(__name__)
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
DATA_DIR = "/opt/airflow/data/raw"




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
                # no session parameter
            )
            # ... rest unchanged

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


# from validation import validate_ohlcv

# def validate_data(**context) -> None:
#     execution_date = context["ds"]
#     file_path = os.path.join(DATA_DIR, f"{execution_date}.parquet")
    
#     report = validate_ohlcv(file_path)
    
#     if not report["passed"]:
#         raise ValueError(f"Validation failed: {report['errors']}")

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
    schedule="0 9 * * 1-5",            # 9:00 AM UTC, Monday–Friday only
  
    start_date=datetime(2024, 1, 1),
    catchup=False,                   
    default_args=default_args,
    tags=["ingestion", "week2"],       
) as dag:
# task 1 
    fetch_task = PythonOperator(
        task_id="fetch_stock_data",
        python_callable=fetch_stock_data,
    )

# task 2 
    # validate_task = PythonOperator(
    #     task_id="validate_data",
    #     python_callable=validate_data,
    # )
    # fetch_task >> validate_task