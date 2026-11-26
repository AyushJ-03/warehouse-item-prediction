from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet


MODEL_NAMES = [
    "Prophet",
    "XGBoost",
    "LSTM",
    "ARIMA (1,1,1)",
]
FORECAST_COLUMNS = ["Date", "Forecasted_Demand", "Product", "Forecast_Model"]
METRIC_COLUMNS = [
    "Product",
    "Model",
    "Validation_MAE",
    "Validation_RMSE",
    "Test_MAE",
    "Test_RMSE",
    "Used_For_Forecast",
]
COMPARISON_COLUMNS = [
    "Model",
    "Mean_Validation_MAE",
    "Mean_Validation_RMSE",
    "Mean_Test_MAE",
    "Mean_Test_RMSE",
    "Validation_Wins",
    "Test_Wins",
    "Forecast_Products",
]


def _daily_demand(data: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "Product", "Units_Sold"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Business data is missing columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Forecast input cannot contain missing dates, products, or demand.")
    if not np.isfinite(data["Units_Sold"].to_numpy(dtype=float)).all():
        raise ValueError("Forecast demand must be finite.")

    daily = (
        data.assign(Date=pd.to_datetime(data["Date"], errors="raise"))
        .groupby(["Product", "Date"], as_index=False)["Units_Sold"]
        .sum()
    )
    if daily.empty:
        raise ValueError("Cannot forecast an empty transaction dataset.")
    if (daily["Units_Sold"] < 0).any():
        raise ValueError("Units_Sold cannot be negative when forecasting demand.")
    return daily


def _complete_product_history(daily: pd.DataFrame, product: str) -> pd.DataFrame:
    product_data = daily.loc[daily["Product"] == product, ["Date", "Units_Sold"]]
    dates = pd.date_range(product_data["Date"].min(), product_data["Date"].max(), freq="D")
    return (
        product_data.set_index("Date")
        .reindex(dates, fill_value=0)
        .rename_axis("ds")
        .reset_index()
        .rename(columns={"Units_Sold": "y"})
    )


def _fit_prophet(history: pd.DataFrame) -> Prophet:
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=len(history) >= 730,
        daily_seasonality=False,
    )
    model.fit(history[["ds", "y"]])
    return model


def _forecast_arima(history: pd.DataFrame, horizon: int) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(
        history["y"].to_numpy(dtype=float),
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    return np.maximum(np.asarray(model.forecast(steps=horizon), dtype=float), 0)


def _xgboost_feature(past: np.ndarray, date: pd.Timestamp) -> list[float]:
    return [
        float(past[-1]),
        float(past[-2]),
        float(past[-7]),
        float(past[-14]),
        float(past[-28]),
        float(np.mean(past[-7:])),
        float(np.mean(past[-28:])),
        float(np.sin(2 * np.pi * date.dayofweek / 7)),
        float(np.cos(2 * np.pi * date.dayofweek / 7)),
        float(np.sin(2 * np.pi * date.dayofyear / 365.25)),
        float(np.cos(2 * np.pi * date.dayofyear / 365.25)),
    ]


def _forecast_xgboost(
    history: pd.DataFrame,
    target_dates: pd.Series,
) -> np.ndarray:
    from xgboost import XGBRegressor

    observed = history["y"].to_numpy(dtype=float)
    if len(observed) <= 28:
        raise ValueError("XGBoost forecasting requires at least 29 daily observations.")

    dates = pd.DatetimeIndex(pd.to_datetime(history["ds"]))
    features = np.asarray(
        [
            _xgboost_feature(observed[:index], dates[index])
            for index in range(28, len(observed))
        ],
        dtype=float,
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(features, observed[28:])

    values = list(observed)
    predictions = []
    for date in pd.DatetimeIndex(pd.to_datetime(target_dates)):
        future_features = np.asarray(
            [_xgboost_feature(np.asarray(values, dtype=float), date)],
            dtype=float,
        )
        prediction = max(float(model.predict(future_features)[0]), 0.0)
        predictions.append(prediction)
        values.append(prediction)
    return np.asarray(predictions, dtype=float)


def _forecast_lstm(history: pd.DataFrame, horizon: int) -> np.ndarray:
    import torch

    torch.manual_seed(42)
    values = history["y"].to_numpy(dtype=np.float32)
    lookback = 14
    scale = max(float(np.max(values)), 1.0)
    scaled = values / scale
    inputs = np.stack(
        [scaled[index - lookback : index] for index in range(lookback, len(scaled))]
    )
    targets = scaled[lookback:]
    x_train = torch.from_numpy(inputs).unsqueeze(-1)
    y_train = torch.from_numpy(targets).unsqueeze(-1)

    class DemandLSTM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent = torch.nn.LSTM(
                input_size=1,
                hidden_size=16,
                num_layers=1,
                batch_first=True,
            )
            self.output = torch.nn.Linear(16, 1)

        def forward(self, sequence):
            output, _ = self.recurrent(sequence)
            return self.output(output[:, -1, :])

    model = DemandLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss()
    model.train()
    for _ in range(40):
        optimizer.zero_grad()
        loss = loss_function(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    sequence = torch.from_numpy(scaled[-lookback:]).reshape(1, lookback, 1)
    predictions = []
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            prediction = model(sequence).reshape(1, 1, 1)
            predictions.append(float(prediction.item()) * scale)
            sequence = torch.cat((sequence[:, 1:, :], prediction), dim=1)
    return np.maximum(np.asarray(predictions, dtype=float), 0)


def _predict_model(
    model_name: str,
    history: pd.DataFrame,
    target_dates: pd.Series,
) -> np.ndarray:
    horizon = len(target_dates)
    if model_name == "ARIMA (1,1,1)":
        return _forecast_arima(history, horizon)
    if model_name == "Prophet":
        predicted = _fit_prophet(history).predict(
            pd.DataFrame({"ds": pd.to_datetime(target_dates)})
        )
        return np.maximum(predicted["yhat"].to_numpy(dtype=float), 0)
    if model_name == "XGBoost":
        return _forecast_xgboost(history, target_dates)
    if model_name == "LSTM":
        return _forecast_lstm(history, horizon)
    raise ValueError(f"Unsupported forecasting model: {model_name}")


def _error_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    errors = actual - predicted
    return (
        float(np.mean(np.abs(errors))),
        float(np.sqrt(np.mean(np.square(errors)))),
    )


def _evaluate_product(
    history: pd.DataFrame,
    product: str,
    evaluation_days: int,
) -> list[dict]:
    minimum_training_days = 30
    if len(history) <= 2 * evaluation_days + minimum_training_days:
        raise ValueError(
            f"{product} has {len(history)} daily observations; at least "
            f"{2 * evaluation_days + minimum_training_days + 1} are required "
            f"for two {evaluation_days}-day evaluation windows."
        )

    validation_start = len(history) - 2 * evaluation_days
    test_start = len(history) - evaluation_days
    training = history.iloc[:validation_start]
    validation = history.iloc[validation_start:test_start]
    test_training = history.iloc[:test_start]
    test = history.iloc[test_start:]
    validation_actual = validation["y"].to_numpy(dtype=float)
    test_actual = test["y"].to_numpy(dtype=float)

    scores = {}
    test_scores = {}
    for model_name in MODEL_NAMES:
        validation_prediction = _predict_model(
            model_name, training, validation["ds"]
        )
        test_prediction = _predict_model(
            model_name, test_training, test["ds"]
        )
        scores[model_name] = _error_metrics(validation_actual, validation_prediction)
        test_scores[model_name] = _error_metrics(test_actual, test_prediction)

    return [
        {
            "Product": product,
            "Model": model_name,
            "Validation_MAE": scores[model_name][0],
            "Validation_RMSE": scores[model_name][1],
            "Test_MAE": test_scores[model_name][0],
            "Test_RMSE": test_scores[model_name][1],
            "Used_For_Forecast": model_name == "Prophet",
        }
        for model_name in MODEL_NAMES
    ]


def summarize_model_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Product",
        "Model",
        "Validation_MAE",
        "Validation_RMSE",
        "Test_MAE",
        "Test_RMSE",
        "Used_For_Forecast",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Metrics are missing columns: {', '.join(sorted(missing))}")
    summary = (
        metrics.groupby("Model", as_index=False)
        .agg(
            Mean_Validation_MAE=("Validation_MAE", "mean"),
            Mean_Validation_RMSE=("Validation_RMSE", "mean"),
            Mean_Test_MAE=("Test_MAE", "mean"),
            Mean_Test_RMSE=("Test_RMSE", "mean"),
            Forecast_Products=("Used_For_Forecast", "sum"),
        )
    )
    validation_winners = (
        metrics.loc[metrics.groupby("Product")["Validation_MAE"].idxmin()]
        .groupby("Model")
        .size()
    )
    test_winners = (
        metrics.loc[metrics.groupby("Product")["Test_MAE"].idxmin()]
        .groupby("Model")
        .size()
    )
    summary["Validation_Wins"] = (
        summary["Model"].map(validation_winners).fillna(0).astype(int)
    )
    summary["Test_Wins"] = summary["Model"].map(test_winners).fillna(0).astype(int)
    return summary[COMPARISON_COLUMNS].sort_values(
        ["Mean_Test_MAE", "Model"], ignore_index=True
    )


def generate_forecast(
    data: pd.DataFrame,
    forecast_days: int = 30,
    evaluation_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if forecast_days < 1:
        raise ValueError("forecast_days must be at least 1.")
    if evaluation_days < 1:
        raise ValueError("evaluation_days must be at least 1.")

    daily = _daily_demand(data)
    forecasts = []
    metrics = []

    for product in sorted(daily["Product"].unique()):
        history = _complete_product_history(daily, product)
        product_metrics = _evaluate_product(history, product, evaluation_days)
        metrics.extend(product_metrics)
        future_dates = pd.date_range(
            history["ds"].iloc[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D",
        )
        predicted_values = _predict_model(
            "Prophet", history, pd.Series(future_dates)
        )
        forecasts.append(
            pd.DataFrame(
                {
                    "Date": future_dates,
                    "Forecasted_Demand": np.maximum(predicted_values, 0),
                    "Product": product,
                    "Forecast_Model": "Prophet",
                }
            )
        )

    forecast = pd.concat(forecasts, ignore_index=True)[FORECAST_COLUMNS]
    evaluation = pd.DataFrame(metrics, columns=METRIC_COLUMNS)
    return forecast, evaluation


def save_forecast(
    data_path: str | Path,
    output_path: str | Path,
    metrics_path: str | Path,
    forecast_days: int = 30,
    evaluation_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(data_path, parse_dates=["Date"])
    forecast, metrics = generate_forecast(data, forecast_days, evaluation_days)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    return forecast, metrics
