import argparse
from pathlib import Path

import pandas as pd

from src.data_processing import clean_transactions, create_business_data
from src.forecasting import generate_forecast, summarize_model_comparison
from src.inventory import calculate_inventory
from src.optimization import optimize_production
from src.profit import calculate_profit
from src.utils import PROJECT_ROOT, RAW_DATA_PATH
from src.visualization import create_visualizations


def run_pipeline(
    input_path: str | Path = RAW_DATA_PATH,
    output_dir: str | Path = PROJECT_ROOT / "data" / "processed",
    forecast_days: int = 30,
    evaluation_days: int = 30,
    lead_time_days: float = 3.5,
    service_level: float = 0.95,
    unit_cost_ratio: float = 0.45,
    holding_cost_ratio: float = 0.02,
    capacity_multiplier: float = 1.2,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = pd.read_csv(input_path)
    clean_data = clean_transactions(raw_data)
    business_data = create_business_data(
        clean_data,
        unit_cost_ratio=unit_cost_ratio,
        holding_cost_ratio=holding_cost_ratio,
        lead_time_days=lead_time_days,
        capacity_multiplier=capacity_multiplier,
    )
    forecast, forecast_metrics = generate_forecast(
        business_data,
        forecast_days=forecast_days,
        evaluation_days=evaluation_days,
    )
    model_comparison = summarize_model_comparison(forecast_metrics)
    inventory = calculate_inventory(
        business_data,
        lead_time_days=lead_time_days,
        service_level=service_level,
    )
    production = optimize_production(
        business_data,
        forecast,
        capacity_multiplier=capacity_multiplier,
    )
    profit = calculate_profit(business_data, production)

    outputs = {
        "clean_data": clean_data,
        "business_data": business_data,
        "forecast": forecast,
        "forecast_metrics": forecast_metrics,
        "model_comparison": model_comparison,
        "inventory_plan": inventory,
        "production_plan": production,
        "profit_results": profit,
    }
    for name, result in outputs.items():
        result.to_csv(output_dir / f"{name}.csv", index=False)
    figures = create_visualizations(
        forecast,
        forecast_metrics,
        inventory,
        production,
        profit,
        output_dir / "visualizations",
    )

    print(
        f"Pipeline complete: {len(clean_data):,} cleaned transactions, "
        f"{forecast['Product'].nunique()} products, "
        f"{forecast_days}-day forecast; "
        f"{len(figures)} charts and CSV outputs saved to {output_dir.resolve()}."
    )
    return outputs


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean grocery data, forecast demand, plan inventory/production, and estimate profit."
    )
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--evaluation-days", type=int, default=30)
    parser.add_argument("--lead-time-days", type=float, default=3.5)
    parser.add_argument("--service-level", type=float, default=0.95)
    parser.add_argument("--unit-cost-ratio", type=float, default=0.45)
    parser.add_argument("--holding-cost-ratio", type=float, default=0.02)
    parser.add_argument("--capacity-multiplier", type=float, default=1.2)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        forecast_days=args.forecast_days,
        evaluation_days=args.evaluation_days,
        lead_time_days=args.lead_time_days,
        service_level=args.service_level,
        unit_cost_ratio=args.unit_cost_ratio,
        holding_cost_ratio=args.holding_cost_ratio,
        capacity_multiplier=args.capacity_multiplier,
    )


if __name__ == "__main__":
    main()
