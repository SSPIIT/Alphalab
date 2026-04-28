import os
import shutil
import glob

# Latest raw OHLCV
raw_files = sorted(glob.glob("data/raw/*.parquet"))
if raw_files:
    shutil.copy(raw_files[-1], "data/raw/latest.parquet")
    print(f"Linked raw: {raw_files[-1]} → data/raw/latest.parquet")

# Latest fundamentals
fund_files = sorted(glob.glob("data/features/fundamentals_*.parquet"))
if fund_files:
    shutil.copy(fund_files[-1], "data/features/fundamentals_latest.parquet")
    print(f"Linked fundamentals: {fund_files[-1]} → data/features/fundamentals_latest.parquet")