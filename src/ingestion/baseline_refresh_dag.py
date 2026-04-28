"""
baseline_refresh_dag.py
-----------------------
Adds a monthly baseline refresh task to your existing Airflow setup.

Schedule: 1st of every month at 06:00 IST (00:30 UTC).
Runs calculate_baseline.py inside the same container.
"""

from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import sys
import os

# ✅ Ensure src/ is importable inside Docker
sys.path.insert(0, "/opt/airflow/project/src")

default_args = {
    "owner": "alphalab",
    "retries": 1,
    "email_on_failure": False,
}

def run_baseline(**context):
    from monitoring.calculate_baseline import calculate_baseline
    df = calculate_baseline()
    print(f"Baseline refresh complete. Features: {len(df)}")
    return len(df)

with DAG(
    dag_id="baseline_refresh",
    description="Monthly refresh of factor drift baseline statistics",
    schedule="30 0 1 * *",   # ✅ FIXED (Airflow 3)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["monitoring", "drift", "baseline"],
) as dag:

    refresh_baseline = PythonOperator(
        task_id="calculate_drift_baseline",
        python_callable=run_baseline,
        doc_md="""
        Computes mean/std/percentiles/skewness for all factor columns
        over the last 365 days and saves baseline stats.
        """,
    )