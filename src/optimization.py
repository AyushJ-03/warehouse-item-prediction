import math

import numpy as np
import pandas as pd
from pulp import (
    LpInteger,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpStatusOptimal,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
)


PRODUCTION_COLUMNS = [
    "Product",
    "Forecast_Demand",
    "Daily_Capacity",
    "Capacity",
    "Optimal_Production",
    "Unmet_Demand",
    "Solver_Status",
]


def optimize_production(
    business_data: pd.DataFrame,
    forecast: pd.DataFrame,
    capacity_multiplier: float = 1.2,
) -> pd.DataFrame:
    business_required = {
        "Product",
        "Unit_Cost",
        "Shortage_Cost",
        "Daily_Capacity",
    }
    forecast_required = {"Product", "Date", "Forecasted_Demand"}
    missing_business = business_required.difference(business_data.columns)
    missing_forecast = forecast_required.difference(forecast.columns)
    if missing_business:
        raise ValueError(
            f"Business data is missing columns: {', '.join(sorted(missing_business))}"
        )
    if missing_forecast:
        raise ValueError(
            f"Forecast data is missing columns: {', '.join(sorted(missing_forecast))}"
        )
    if capacity_multiplier <= 0:
        raise ValueError("capacity_multiplier must be greater than zero.")
    if forecast.empty:
        raise ValueError("Cannot optimize production for an empty forecast.")
    if not np.isfinite(forecast["Forecasted_Demand"].to_numpy(dtype=float)).all() or (
        forecast["Forecasted_Demand"] < 0
    ).any():
        raise ValueError("Forecasted demand must be finite and nonnegative.")

    forecast_dates = pd.to_datetime(forecast["Date"], errors="raise")
    if forecast_dates.nunique() == 0:
        raise ValueError("The forecast has no planning dates.")
    if forecast.duplicated(["Product", "Date"]).any():
        raise ValueError("The forecast must contain at most one row per product and date.")
    if "Date" in business_data.columns:
        history_ends = business_data.assign(
            Date=pd.to_datetime(business_data["Date"], errors="raise")
        ).groupby("Product")["Date"].max()
        forecast_starts = forecast.assign(_Date=forecast_dates).groupby("Product")[
            "_Date"
        ].min()
        matching_products = forecast_starts.index.intersection(history_ends.index)
        if matching_products.empty or (
            forecast_starts.loc[matching_products]
            <= history_ends.loc[matching_products]
        ).any():
            raise ValueError(
                "Each product's forecast must start strictly after its sales history."
            )
    forecast_with_dates = forecast.assign(Date=forecast_dates)
    product_date_counts = forecast_with_dates.groupby("Product")["Date"].nunique()
    if product_date_counts.nunique() != 1:
        raise ValueError("Every product must have the same complete forecast horizon.")
    for product, group in forecast_with_dates.groupby("Product"):
        ordered_dates = group["Date"].sort_values()
        if not ordered_dates.diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError(f"{product} forecast dates must be consecutive calendar days.")
    demand = (
        forecast.groupby("Product", as_index=False)["Forecasted_Demand"]
        .sum()
        .rename(columns={"Forecasted_Demand": "Forecast_Demand"})
    )
    demand["Horizon_Days"] = demand["Product"].map(product_date_counts)
    costs = business_data.groupby("Product", as_index=False).agg(
        Unit_Cost=("Unit_Cost", "mean"),
        Shortage_Cost=("Shortage_Cost", "mean"),
        Daily_Capacity=("Daily_Capacity", "mean"),
    )
    products = demand.merge(costs, on="Product", how="left", validate="one_to_one")
    if products[["Unit_Cost", "Shortage_Cost", "Daily_Capacity"]].isna().any().any():
        raise ValueError("Every forecast product must have costs and a daily capacity.")
    if (products[["Unit_Cost", "Shortage_Cost", "Daily_Capacity"]] < 0).any().any():
        raise ValueError("Costs and daily capacities cannot be negative.")
    if not np.isfinite(
        products[
            ["Forecast_Demand", "Unit_Cost", "Shortage_Cost", "Daily_Capacity"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Demand, costs, and capacities must be finite.")

    products["Capacity"] = (
        products["Daily_Capacity"] * capacity_multiplier * products["Horizon_Days"]
    ).apply(math.floor)

    model = LpProblem("Production_Optimization", LpMinimize)
    production = {
        product: LpVariable(f"production_{index}", lowBound=0, cat=LpInteger)
        for index, product in enumerate(products["Product"])
    }
    unmet = {
        product: LpVariable(f"unmet_{index}", lowBound=0)
        for index, product in enumerate(products["Product"])
    }
    product_rows = products.set_index("Product")

    for product in products["Product"]:
        row = product_rows.loc[product]
        model += (
            production[product] + unmet[product] == row["Forecast_Demand"],
            f"demand_balance_{len(model.constraints)}",
        )
        model += (
            production[product] <= row["Capacity"],
            f"capacity_limit_{len(model.constraints)}",
        )

    model += lpSum(
        product_rows.loc[product, "Unit_Cost"] * production[product]
        + product_rows.loc[product, "Shortage_Cost"] * unmet[product]
        for product in products["Product"]
    )

    status = model.solve(PULP_CBC_CMD(msg=False))
    if status != LpStatusOptimal:
        raise RuntimeError(f"Production optimization failed: {LpStatus[status]}.")

    results = []
    for product in products["Product"]:
        planned = production[product].value()
        shortage = unmet[product].value()
        if planned is None or shortage is None:
            raise RuntimeError(f"Solver did not return a solution for {product}.")
        row = product_rows.loc[product]
        results.append(
            {
                "Product": product,
                "Forecast_Demand": float(row["Forecast_Demand"]),
                "Daily_Capacity": float(row["Daily_Capacity"]),
                "Capacity": int(row["Capacity"]),
                "Optimal_Production": float(planned),
                "Unmet_Demand": float(shortage),
                "Solver_Status": LpStatus[status],
            }
        )

    result = pd.DataFrame(results, columns=PRODUCTION_COLUMNS)
    if (result["Optimal_Production"] > result["Capacity"] + 1e-6).any():
        raise RuntimeError("The solver returned a production plan above capacity.")
    if not np.allclose(
        result["Optimal_Production"] + result["Unmet_Demand"],
        result["Forecast_Demand"],
        rtol=0,
        atol=1e-5,
    ):
        raise RuntimeError("The solver returned a plan that does not balance demand.")
    return result


def save_production_plan(
    business_path: str,
    forecast_path: str,
    output_path: str,
    capacity_multiplier: float = 1.2,
) -> pd.DataFrame:
    business_data = pd.read_csv(business_path)
    forecast = pd.read_csv(forecast_path, parse_dates=["Date"])
    result = optimize_production(business_data, forecast, capacity_multiplier)
    result.to_csv(output_path, index=False)
    return result
