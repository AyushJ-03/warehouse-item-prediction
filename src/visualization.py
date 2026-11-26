from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def create_visualizations(
    forecast: pd.DataFrame,
    metrics: pd.DataFrame,
    inventory: pd.DataFrame,
    production: pd.DataFrame,
    profit: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []

    fig, axis = plt.subplots(figsize=(12, 6))
    for product, group in forecast.groupby("Product"):
        axis.plot(group["Date"], group["Forecasted_Demand"], label=product)
    axis.set(title="30-day demand forecast", xlabel="Date", ylabel="Forecasted units")
    axis.legend(ncol=3, fontsize="small")
    fig.tight_layout()
    created.append(_save(fig, output_dir / "demand_forecast.png"))

    created.append(
        _bar_chart(
            inventory,
            "Product",
            "Safety_Stock",
            "Safety stock by product",
            output_dir / "safety_stock.png",
        )
    )
    created.append(
        _bar_chart(
            production,
            "Product",
            "Optimal_Production",
            "Production plan by product",
            output_dir / "production_plan.png",
        )
    )
    created.append(
        _bar_chart(
            profit,
            "Product",
            "Profit",
            "Profit contribution by product",
            output_dir / "profit_contribution.png",
        )
    )

    fig, axis = plt.subplots(figsize=(12, 6))
    profit.set_index("Product")[
        ["Revenue", "Production_Cost", "Shortage_Total"]
    ].plot(kind="bar", ax=axis)
    axis.set(title="Revenue and costs by product", xlabel="Product", ylabel="Amount")
    axis.tick_params(axis="x", labelrotation=70)
    fig.tight_layout()
    created.append(_save(fig, output_dir / "revenue_vs_cost.png"))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    profit.set_index("Product")["Profit"].plot(kind="bar", ax=axes[0, 0], color="#4c956c")
    axes[0, 0].set_title("Profit contribution")
    production.set_index("Product")["Optimal_Production"].plot(
        kind="bar", ax=axes[0, 1], color="#3a86a8"
    )
    axes[0, 1].set_title("Production plan")
    inventory.set_index("Product")["Safety_Stock"].plot(
        kind="bar", ax=axes[1, 0], color="#f4a261"
    )
    axes[1, 0].set_title("Safety stock")
    profit.set_index("Product")[["Revenue", "Production_Cost"]].plot(
        kind="bar", ax=axes[1, 1]
    )
    axes[1, 1].set_title("Revenue vs production cost")
    for axis in axes.flat:
        axis.tick_params(axis="x", labelrotation=70)
    fig.suptitle("Business operations and profit overview")
    fig.tight_layout()
    created.append(_save(fig, output_dir / "operations_overview.png"))

    created.extend(_model_comparison_charts(metrics, output_dir))

    return created


def _model_comparison_charts(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    model_order = ["Prophet", "XGBoost", "LSTM", "ARIMA (1,1,1)"]
    colors_by_model = {
        "Prophet": "#176b87",
        "XGBoost": "#f4a261",
        "LSTM": "#4c956c",
        "ARIMA (1,1,1)": "#9b5de5",
    }
    required = {"Product", "Model", "Test_MAE", "Test_RMSE"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(
            f"Forecast metrics are missing columns: {', '.join(sorted(missing))}"
        )

    summary = (
        metrics.groupby("Model")[["Test_MAE", "Test_RMSE"]]
        .mean()
        .reindex(model_order)
    )
    if summary.isna().any().any():
        raise ValueError("Forecast metrics must include scores for all four models.")

    created = []
    for metric, label in [
        ("Test_MAE", "MAE"),
        ("Test_RMSE", "RMSE"),
    ]:
        fig, axis = plt.subplots(figsize=(9, 5))
        bars = axis.bar(
            model_order,
            summary[metric],
            color=[colors_by_model[model] for model in model_order],
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        axis.set(
            title=f"Mean test {label} by forecasting model",
            xlabel="Model",
            ylabel=f"{label} (units/day)",
        )
        axis.tick_params(axis="x", labelrotation=15)
        fig.tight_layout()
        created.append(
            _save(fig, output_dir / f"model_comparison_test_{label.lower()}.png")
        )

    per_product = metrics.pivot(
        index="Product",
        columns="Model",
        values="Test_MAE",
    ).reindex(columns=model_order)
    if per_product.isna().any().any():
        raise ValueError("Every product must have test MAE scores for all four models.")
    fig, axis = plt.subplots(figsize=(16, 7))
    per_product.plot(
        kind="bar",
        ax=axis,
        color=[colors_by_model[model] for model in model_order],
    )
    axis.set(
        title="Per-product test MAE by forecasting model",
        xlabel="Product",
        ylabel="MAE (units/day)",
    )
    axis.tick_params(axis="x", labelrotation=45)
    axis.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    created.append(_save(fig, output_dir / "model_comparison_test_mae_by_product.png"))
    return created


def _bar_chart(
    data: pd.DataFrame,
    category: str,
    value: str,
    title: str,
    output_path: Path,
) -> Path:
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.bar(data[category], data[value])
    axis.set(title=title, xlabel="Product", ylabel=value.replace("_", " "))
    axis.tick_params(axis="x", labelrotation=70)
    fig.tight_layout()
    return _save(fig, output_path)


def _save(figure: plt.Figure, output_path: Path) -> Path:
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path
