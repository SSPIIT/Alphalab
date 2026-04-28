
import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

TRADING_DAYS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "12M": 252,
}


def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    
    logger.info(f"Computing momentum factors for {df['ticker'].nunique()} stocks")

    results = []

    for ticker, group in df.groupby("ticker"):
        try:
            group = group.sort_values("Date").reset_index(drop=True)

            if len(group) < TRADING_DAYS["12M"]:
                logger.warning(f"{ticker}: not enough history ({len(group)} days), skipping")
                continue

            latest_close = group["Close"].iloc[-1]

            row = {"ticker": ticker}

            for window_name, n_days in TRADING_DAYS.items():
                if window_name == "1M":
                    past_close = group["Close"].iloc[-n_days]
                else:
                    skip = TRADING_DAYS["1M"]
                    past_close = group["Close"].iloc[-(n_days + skip)]

                if past_close == 0 or pd.isna(past_close):
                    row[f"mom_{window_name.lower()}"] = np.nan
                else:
                    momentum = (latest_close - past_close) / past_close
                    row[f"mom_{window_name.lower()}"] = round(momentum, 6)

            results.append(row)

        except Exception as e:
            logger.error(f"Failed momentum for {ticker}: {e}")
            continue

    if not results:
        raise ValueError("No momentum scores computed — check input data")

    result_df = pd.DataFrame(results)

    logger.info(f"Momentum computed for {len(result_df)} stocks")
    logger.info(f"  mom_6m stats: mean={result_df['mom_6m'].mean():.3f}, "
                f"std={result_df['mom_6m'].std():.3f}")

    return result_df


def rank_momentum(momentum_df: pd.DataFrame) -> pd.DataFrame:
    
    ranked = momentum_df.copy()

    for col in ["mom_1m", "mom_3m", "mom_6m", "mom_12m"]:
        # pct=True gives percentile rank between 0 and 1
        # na_option="keep" leaves NaN as NaN instead of ranking it
        ranked[f"{col}_rank"] = ranked[col].rank(pct=True, na_option="keep")

    logger.info("Momentum ranks computed")
    return ranked


def run(raw_data_path: str, output_path: str) -> pd.DataFrame:
    
    logger.info(f"Loading raw data from {raw_data_path}")
    df = pd.read_parquet(raw_data_path)

    df["Date"] = pd.to_datetime(df["Date"])

    momentum_df = compute_momentum(df)

    momentum_df = rank_momentum(momentum_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    momentum_df.to_parquet(output_path, index=False)
    logger.info(f"Momentum factors saved to {output_path}")

    return momentum_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    raw_dir = "data/raw"
    files = sorted(os.listdir(raw_dir))
    latest = os.path.join(raw_dir, files[-1])

    result = run(
        raw_data_path=latest,
        output_path="../../data/features/momentum.parquet"
    )

    print("\n── Momentum Factor Sample ──")
    print(result.sort_values("mom_6m", ascending=False).head(10).to_string(index=False))
    print(f"\nShape: {result.shape}")