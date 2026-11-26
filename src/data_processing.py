from pathlib import Path

import numpy as np
import pandas as pd


RAW_REQUIRED_COLUMNS = {
    "transaction_date",
    "product_name",
    "quantity",
    "unit_price",
    "final_amount",
}


def clean_transactions(raw_data: pd.DataFrame) -> pd.DataFrame:
    missing = RAW_REQUIRED_COLUMNS.difference(raw_data.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {', '.join(sorted(missing))}")

    data = raw_data.copy()
    data["transaction_date"] = pd.to_datetime(
        data["transaction_date"], errors="coerce"
    )
    for column in ("quantity", "unit_price", "final_amount"):
        data[column] = pd.to_numeric(data[column], errors="raise")

    before = len(data)
    data = data.drop_duplicates()
    data = data.dropna(subset=["transaction_date", "product_name", "quantity", "final_amount"])
    data = data.loc[
        (data["quantity"] > 0)
        & (data["unit_price"] >= 0)
        & (data["final_amount"] >= 0)
    ].copy()
    data = data.rename(
        columns={
            "transaction_date": "Date",
            "product_name": "Product",
            "quantity": "Units_Sold",
        }
    )
    data = data.sort_values(["Date", "Product"]).reset_index(drop=True)
    data.attrs["input_rows"] = before
    data.attrs["removed_rows"] = before - len(data)
    return data


def create_business_data(
    clean_data: pd.DataFrame,
    unit_cost_ratio: float = 0.45,
    holding_cost_ratio: float = 0.02,
    lead_time_days: float = 3.5,
    capacity_multiplier: float = 1.2,
) -> pd.DataFrame:
    required = {"Date", "Product", "Units_Sold", "final_amount"}
    missing = required.difference(clean_data.columns)
    if missing:
        raise ValueError(
            f"Clean data is missing columns: {', '.join(sorted(missing))}"
        )
    if not 0 <= unit_cost_ratio < 1:
        raise ValueError("unit_cost_ratio must be at least 0 and less than 1.")
    if holding_cost_ratio < 0:
        raise ValueError("holding_cost_ratio cannot be negative.")
    if lead_time_days <= 0 or capacity_multiplier <= 0:
        raise ValueError("Lead time and capacity multiplier must be greater than zero.")
    if (clean_data["Units_Sold"] <= 0).any():
        raise ValueError("Clean transaction quantities must be positive.")

    data = clean_data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    data["Price"] = data["final_amount"] / data["Units_Sold"]
    data["Unit_Cost"] = data["Price"] * unit_cost_ratio
    data["Holding_Cost"] = data["Unit_Cost"] * holding_cost_ratio
    data["Shortage_Cost"] = (data["Price"] - data["Unit_Cost"]).clip(lower=0)
    data["Lead_Time_Days"] = float(lead_time_days)
    data["Lead_Time"] = data["Lead_Time_Days"]

    daily = (
        data.groupby(["Product", "Date"], as_index=False)["Units_Sold"]
        .sum()
        .groupby("Product")["Units_Sold"]
        .max()
    )
    daily_capacity = daily
    data["Daily_Capacity"] = data["Product"].map(daily_capacity).astype(float)
    data["Capacity"] = data["Daily_Capacity"] * capacity_multiplier
    data["Revenue"] = data["Units_Sold"] * data["Price"]

    numeric = [
        "Price",
        "Unit_Cost",
        "Holding_Cost",
        "Shortage_Cost",
        "Daily_Capacity",
        "Revenue",
    ]
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Business assumptions produced a non-finite value.")
    return data


def save_clean_data(raw_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    clean_data = clean_transactions(pd.read_csv(raw_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clean_data.to_csv(output_path, index=False)
    return clean_data
