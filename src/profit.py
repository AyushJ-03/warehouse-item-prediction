import numpy as np
import pandas as pd


PROFIT_COLUMNS = [
    "Product",
    "Optimal_Production",
    "Price",
    "Unit_Cost",
    "Forecast_Demand",
    "Avg_Price",
    "Avg_Unit_Cost",
    "Revenue",
    "Production_Cost",
    "Holding_Cost",
    "Shortage_Cost",
    "Units_Sold",
    "Unmet_Demand",
    "Holding_Total",
    "Shortage_Total",
    "Profit",
]


def calculate_profit(
    business_data: pd.DataFrame,
    production_plan: pd.DataFrame,
) -> pd.DataFrame:
    business_required = {
        "Product",
        "Price",
        "Unit_Cost",
        "Holding_Cost",
        "Shortage_Cost",
    }
    plan_required = {"Product", "Optimal_Production", "Forecast_Demand"}
    missing_business = business_required.difference(business_data.columns)
    missing_plan = plan_required.difference(production_plan.columns)
    if missing_business:
        raise ValueError(
            f"Business data is missing columns: {', '.join(sorted(missing_business))}"
        )
    if missing_plan:
        raise ValueError(
            f"Production plan is missing columns: {', '.join(sorted(missing_plan))}"
        )
    if production_plan["Product"].duplicated().any():
        raise ValueError("Production plan must contain one row per product.")

    economics = business_data.groupby("Product", as_index=False).agg(
        Avg_Price=("Price", "mean"),
        Avg_Unit_Cost=("Unit_Cost", "mean"),
        Holding_Cost=("Holding_Cost", "mean"),
        Shortage_Cost=("Shortage_Cost", "mean"),
    )
    result = production_plan.merge(
        economics, on="Product", how="left", validate="one_to_one"
    )
    if result[["Avg_Price", "Avg_Unit_Cost", "Holding_Cost", "Shortage_Cost"]].isna().any().any():
        raise ValueError("Every planned product must have business cost and price data.")
    if (result[["Optimal_Production", "Forecast_Demand"]] < 0).any().any():
        raise ValueError("Production and demand cannot be negative.")
    if not np.isfinite(
        result[
            [
                "Optimal_Production",
                "Forecast_Demand",
                "Avg_Price",
                "Avg_Unit_Cost",
                "Holding_Cost",
                "Shortage_Cost",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Production, demand, prices, and costs must be finite.")

    result["Units_Sold"] = result[["Optimal_Production", "Forecast_Demand"]].min(axis=1)
    result["Unmet_Demand"] = (
        result["Forecast_Demand"] - result["Units_Sold"]
    ).clip(lower=0)
    result["Revenue"] = result["Units_Sold"] * result["Avg_Price"]
    result["Price"] = result["Avg_Price"]
    result["Unit_Cost"] = result["Avg_Unit_Cost"]
    result["Production_Cost"] = result["Optimal_Production"] * result["Avg_Unit_Cost"]
    result["Holding_Total"] = (
        result["Optimal_Production"] - result["Units_Sold"]
    ).clip(lower=0) * result["Holding_Cost"]
    result["Shortage_Total"] = result["Unmet_Demand"] * result["Shortage_Cost"]
    result["Profit"] = (
        result["Revenue"]
        - result["Production_Cost"]
        - result["Holding_Total"]
        - result["Shortage_Total"]
    )
    return result[PROFIT_COLUMNS]


def save_profit_results(
    business_path: str,
    production_path: str,
    output_path: str,
) -> pd.DataFrame:
    business_data = pd.read_csv(business_path)
    production_plan = pd.read_csv(production_path)
    results = calculate_profit(business_data, production_plan)
    results.to_csv(output_path, index=False)
    return results
