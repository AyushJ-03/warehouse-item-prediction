import math
from statistics import NormalDist

import numpy as np
import pandas as pd


INVENTORY_COLUMNS = [
    "Product",
    "Avg_Daily_Demand",
    "Daily_Demand_Std",
    "Lead_Time_Days",
    "Service_Level",
    "Safety_Stock",
    "Reorder_Point",
    "Avg_Demand",
    "Demand_Std",
    "Avg_Lead_Time",
]


def calculate_inventory(
    data: pd.DataFrame,
    lead_time_days: float = 3.5,
    service_level: float = 0.95,
) -> pd.DataFrame:
    required = {"Date", "Product", "Units_Sold"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Business data is missing columns: {', '.join(sorted(missing))}")
    if lead_time_days <= 0:
        raise ValueError("lead_time_days must be greater than zero.")
    if not 0.5 < service_level < 1:
        raise ValueError("service_level must be greater than 0.5 and less than 1.")
    if data.empty:
        raise ValueError("Cannot calculate inventory targets for empty data.")
    if data[["Date", "Product", "Units_Sold"]].isna().any().any():
        raise ValueError("Inventory data cannot contain missing dates, products, or demand.")
    if not np.isfinite(data["Units_Sold"].to_numpy(dtype=float)).all():
        raise ValueError("Daily demand must be finite.")
    if (data["Units_Sold"] < 0).any():
        raise ValueError("Daily demand cannot be negative.")

    transactions = data.assign(Date=pd.to_datetime(data["Date"], errors="raise"))
    start, end = transactions["Date"].min(), transactions["Date"].max()
    dates = pd.date_range(start, end, freq="D")
    z_score = NormalDist().inv_cdf(service_level)
    records = []

    for product, group in transactions.groupby("Product", sort=True):
        daily = (
            group.groupby("Date")["Units_Sold"]
            .sum()
            .reindex(dates, fill_value=0)
            .astype(float)
        )
        average = float(daily.mean())
        standard_deviation = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
        safety_stock = z_score * standard_deviation * math.sqrt(lead_time_days)
        records.append(
            {
                "Product": product,
                "Avg_Daily_Demand": average,
                "Daily_Demand_Std": standard_deviation,
                "Lead_Time_Days": float(lead_time_days),
                "Service_Level": float(service_level),
                "Safety_Stock": safety_stock,
                "Reorder_Point": average * lead_time_days + safety_stock,
                "Avg_Demand": average,
                "Demand_Std": standard_deviation,
                "Avg_Lead_Time": float(lead_time_days),
            }
        )

    result = pd.DataFrame(records, columns=INVENTORY_COLUMNS)
    if (result["Safety_Stock"] < 0).any() or (result["Reorder_Point"] < 0).any():
        raise ValueError("Inventory targets cannot be negative.")
    return result
def save_inventory(
    data_path: str,
    output_path: str,
    lead_time_days: float = 3.5,
    service_level: float = 0.95,
) -> pd.DataFrame:
    data = pd.read_csv(data_path, parse_dates=["Date"])
    inventory = calculate_inventory(data, lead_time_days, service_level)
    inventory.to_csv(output_path, index=False)
    return inventory
