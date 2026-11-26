import numpy as np
import pandas as pd
import pytest

from src.data_processing import clean_transactions, create_business_data
from src.forecasting import (
    generate_forecast,
    summarize_model_comparison,
)
from src.inventory import calculate_inventory
from src.optimization import optimize_production
from src.profit import calculate_profit


def test_clean_transactions_removes_duplicates_returns_and_invalid_quantities():
    raw = pd.DataFrame(
        [
            ["2025-01-01", "Apples", 2, 10.0, 20.0],
            ["2025-01-01", "Apples", 2, 10.0, 20.0],
            ["2025-01-02", "Apples", 1, 10.0, -5.0],
            ["2025-01-03", "Apples", 0, 10.0, 0.0],
            ["not-a-date", "Apples", 1, 10.0, 10.0],
        ],
        columns=[
            "transaction_date",
            "product_name",
            "quantity",
            "unit_price",
            "final_amount",
        ],
    )

    clean = clean_transactions(raw)

    assert len(clean) == 1
    assert clean.loc[0, "Product"] == "Apples"
    assert clean.loc[0, "Units_Sold"] == 2
    assert clean.attrs["removed_rows"] == 4


def test_business_data_uses_daily_aggregate_as_capacity_proxy():
    clean = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]),
            "Product": ["Apples", "Apples", "Apples"],
            "Units_Sold": [2, 3, 4],
            "final_amount": [20.0, 30.0, 40.0],
        }
    )

    business = create_business_data(clean, capacity_multiplier=1.2)

    assert (business["Unit_Cost"] == 4.5).all()
    assert (business["Daily_Capacity"] == 5).all()
    assert (business["Capacity"] == 6).all()
    assert (business["Lead_Time_Days"] == 3.5).all()


def test_inventory_uses_calendar_days_including_zero_sales_days():
    business = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-01", "2025-01-03"]),
            "Product": ["Apples", "Apples"],
            "Units_Sold": [2, 4],
        }
    )

    inventory = calculate_inventory(business, lead_time_days=4, service_level=0.95)
    row = inventory.iloc[0]

    assert row["Avg_Daily_Demand"] == pytest.approx(2)
    assert row["Daily_Demand_Std"] == pytest.approx(2)
    assert row["Safety_Stock"] == pytest.approx(1.6448536269514722 * 4)
    assert row["Reorder_Point"] == pytest.approx(8 + 1.6448536269514722 * 4)


def test_forecast_contains_only_future_horizon_and_evaluation_metrics(monkeypatch):
    class ConstantModel:
        def predict(self, future):
            return pd.DataFrame({"yhat": np.full(len(future), 9.0)})

    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    business = pd.DataFrame(
        {
            "Date": list(dates) * 2,
            "Product": ["Apples"] * len(dates) + ["Bread"] * len(dates),
            "Units_Sold": [2.0] * len(dates) + [1.0] * len(dates),
        }
    )
    monkeypatch.setattr("src.forecasting._fit_prophet", lambda history: ConstantModel())
    monkeypatch.setattr(
        "src.forecasting._forecast_xgboost",
        lambda history, target_dates: np.full(len(target_dates), 2.0),
    )
    monkeypatch.setattr(
        "src.forecasting._forecast_arima",
        lambda history, horizon: np.full(horizon, 2.0),
    )
    monkeypatch.setattr(
        "src.forecasting._forecast_lstm",
        lambda history, horizon: np.full(horizon, 2.0),
    )

    forecast, metrics = generate_forecast(
        business, forecast_days=7, evaluation_days=30
    )

    assert len(forecast) == 14
    assert forecast.groupby("Product").size().to_dict() == {"Apples": 7, "Bread": 7}
    assert forecast["Date"].min() > dates.max()
    assert (forecast["Forecasted_Demand"] >= 0).all()
    assert set(forecast["Forecast_Model"]) == {"Prophet"}
    assert (forecast["Forecasted_Demand"] == 9.0).all()
    assert set(metrics["Product"]) == {"Apples", "Bread"}
    assert len(metrics) == 8
    assert metrics.groupby("Product")["Model"].nunique().to_dict() == {
        "Apples": 4,
        "Bread": 4,
    }
    assert metrics.groupby("Product")["Used_For_Forecast"].sum().to_dict() == {
        "Apples": 1,
        "Bread": 1,
    }
    assert set(metrics.loc[metrics["Used_For_Forecast"], "Model"]) == {"Prophet"}
    assert {"Validation_MAE", "Test_MAE", "Validation_RMSE", "Test_RMSE"} <= set(
        metrics.columns
    )


def test_model_comparison_summarizes_all_models():
    metrics = pd.DataFrame(
        {
            "Product": ["Apples", "Apples", "Bread", "Bread"],
            "Model": ["Naive", "Moving Average", "Naive", "Moving Average"],
            "Validation_MAE": [1.0, 2.0, 3.0, 1.0],
            "Validation_RMSE": [1.5, 2.5, 3.5, 1.5],
            "Test_MAE": [2.0, 1.0, 1.0, 3.0],
            "Test_RMSE": [2.5, 1.5, 1.5, 3.5],
            "Used_For_Forecast": [True, False, False, True],
        }
    )

    comparison = summarize_model_comparison(metrics).set_index("Model")

    assert comparison.loc["Naive", "Mean_Validation_MAE"] == 2.0
    assert comparison.loc["Moving Average", "Mean_Test_MAE"] == 2.0
    assert comparison.loc["Naive", "Validation_Wins"] == 1
    assert comparison.loc["Moving Average", "Test_Wins"] == 1
    assert comparison.loc["Naive", "Forecast_Products"] == 1
    assert comparison["Validation_Wins"].sum() == 2
    assert comparison["Test_Wins"].sum() == 2


def test_xgboost_features_use_prior_lags_and_calendar_features():
    from src.forecasting import _xgboost_feature

    past = np.arange(1, 29, dtype=float)
    date = pd.Timestamp("2025-01-29")

    features = _xgboost_feature(past, date)

    assert features[:5] == [28, 27, 22, 15, 1]
    assert features[5] == pytest.approx(25)
    assert features[6] == pytest.approx(14.5)
    assert len(features) == 11


def test_comparison_visualizations_include_four_models_and_all_products(tmp_path):
    from src.visualization import create_visualizations

    products = ["Apples", "Bread"]
    models = ["Prophet", "XGBoost", "LSTM", "ARIMA (1,1,1)"]
    metrics = pd.DataFrame(
        [
            {
                "Product": product,
                "Model": model,
                "Test_MAE": model_index + product_index + 1.0,
                "Test_RMSE": model_index + product_index + 1.5,
            }
            for product_index, product in enumerate(products)
            for model_index, model in enumerate(models)
        ]
    )
    forecast = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-03", "2025-01-03"]),
            "Product": products,
            "Forecasted_Demand": [2.0, 1.0],
        }
    )
    inventory = pd.DataFrame({"Product": products, "Safety_Stock": [1.0, 0.5]})
    production = pd.DataFrame({"Product": products, "Optimal_Production": [2, 1]})
    profit = pd.DataFrame(
        {
            "Product": products,
            "Profit": [10.0, 5.0],
            "Revenue": [20.0, 10.0],
            "Production_Cost": [8.0, 4.0],
            "Shortage_Total": [2.0, 1.0],
        }
    )

    figures = create_visualizations(
        forecast, metrics, inventory, production, profit, tmp_path
    )

    chart_names = {path.name for path in figures}
    assert "model_comparison_test_mae.png" in chart_names
    assert "model_comparison_test_rmse.png" in chart_names
    assert "model_comparison_test_mae_by_product.png" in chart_names
    assert all((tmp_path / name).stat().st_size > 0 for name in chart_names)


def test_optimizer_respects_capacity_and_reports_unmet_demand():
    business = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-01"]),
            "Product": ["Apples"],
            "Unit_Cost": [2.0],
            "Shortage_Cost": [4.0],
            "Daily_Capacity": [3.0],
        }
    )
    forecast = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "Product": ["Apples", "Apples"],
            "Forecasted_Demand": [5.1, 5.1],
        }
    )

    plan = optimize_production(business, forecast, capacity_multiplier=1.2)
    row = plan.iloc[0]

    assert row["Capacity"] == 7
    assert row["Optimal_Production"] == 7
    assert row["Unmet_Demand"] == pytest.approx(3.2)
    assert row["Solver_Status"] == "Optimal"
    assert row["Optimal_Production"] <= row["Capacity"]


def test_optimizer_rejects_forecast_dates_inside_sales_history():
    business = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-02"]),
            "Product": ["Apples"],
            "Unit_Cost": [2.0],
            "Shortage_Cost": [4.0],
            "Daily_Capacity": [3.0],
        }
    )
    forecast = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-02"]),
            "Product": ["Apples"],
            "Forecasted_Demand": [5.0],
        }
    )

    with pytest.raises(ValueError, match="strictly after"):
        optimize_production(business, forecast)


def test_profit_counts_sales_only_when_demand_is_served():
    business = pd.DataFrame(
        {
            "Product": ["Apples"],
            "Price": [10.0],
            "Unit_Cost": [2.0],
            "Holding_Cost": [0.04],
            "Shortage_Cost": [8.0],
        }
    )
    plan = pd.DataFrame(
        {
            "Product": ["Apples"],
            "Optimal_Production": [7.0],
            "Forecast_Demand": [10.2],
        }
    )

    result = calculate_profit(business, plan).iloc[0]

    assert result["Units_Sold"] == 7
    assert result["Revenue"] == 70
    assert result["Production_Cost"] == 14
    assert result["Unmet_Demand"] == pytest.approx(3.2)
    assert result["Shortage_Total"] == pytest.approx(25.6)
    assert result["Profit"] == pytest.approx(30.4)
