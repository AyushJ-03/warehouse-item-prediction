from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "grocery_chain_data.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    elif "transaction_date" in data.columns:
        data["transaction_date"] = pd.to_datetime(
            data["transaction_date"], errors="raise"
        )
    return data
