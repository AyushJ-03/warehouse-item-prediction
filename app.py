from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from src.app_support import INPUT_COLUMNS, build_output_archive, normalize_uploaded_columns
from src.run_pipeline import run_pipeline


FIELD_LABELS = {
    "transaction_date": "Transaction date",
    "product_name": "Product name",
    "quantity": "Quantity sold",
    "unit_price": "Unit price",
    "final_amount": "Final sales amount",
}
HEADER_ALIASES = {
    "transaction_date": {"date", "salesdate", "invoicedate", "orderdate"},
    "product_name": {"product", "item", "sku", "productid"},
    "quantity": {"qty", "units", "unitssold", "quantitysold"},
    "unit_price": {"price", "priceperunit", "sellingprice"},
    "final_amount": {"amount", "totalamount", "salesamount", "total"},
}
RESULT_LABELS = {
    "forecast": "Demand forecast",
    "forecast_metrics": "Forecast model scores",
    "model_comparison": "Model comparison summary",
    "inventory_plan": "Inventory plan",
    "production_plan": "Production plan",
    "profit_results": "Estimated profit results",
}
CHART_TITLES = {
    "demand_forecast": "Demand forecast",
    "safety_stock": "Safety stock by product",
    "production_plan": "Production plan",
    "profit_contribution": "Profit contribution",
    "revenue_vs_cost": "Revenue and costs",
    "operations_overview": "Operations overview",
    "model_comparison_test_mae": "Mean test MAE comparison",
    "model_comparison_test_rmse": "Mean test RMSE comparison",
    "model_comparison_test_mae_by_product": "Per-product test MAE comparison",
}


def _normalized_header(header: str) -> str:
    return "".join(character for character in header.lower() if character.isalnum())


def _suggested_column(columns: list[str], field: str) -> int:
    target = _normalized_header(field)
    aliases = {_normalized_header(alias) for alias in HEADER_ALIASES[field]}
    for index, column in enumerate(columns):
        normalized = _normalized_header(str(column))
        if normalized == target or normalized in aliases:
            return index
    return 0


def _read_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(BytesIO(uploaded_file.getvalue()))


def _render_results(analysis: dict) -> None:
    outputs = analysis["outputs"]
    forecast = outputs["forecast"]
    inventory = outputs["inventory_plan"]
    production = outputs["production_plan"]
    profit = outputs["profit_results"]

    st.caption(
        f"Results for {analysis['filename']} · completed {analysis['completed_at']}. "
        "Business costs and capacity are estimates unless replaced with verified data."
    )
    metric_columns = st.columns(5)
    metric_columns[0].metric("Products", f"{forecast['Product'].nunique():,}")
    metric_columns[1].metric(
        "Forecast demand",
        f"{forecast['Forecasted_Demand'].sum():,.1f} units",
    )
    metric_columns[2].metric(
        "Planned production",
        f"{production['Optimal_Production'].sum():,.0f} units",
    )
    metric_columns[3].metric(
        "Unmet demand",
        f"{production['Unmet_Demand'].sum():,.1f} units",
    )
    metric_columns[4].metric(
        "Estimated profit",
        f"{profit['Profit'].sum():,.2f}",
        help="Uses estimated selling prices, costs, and capacity assumptions.",
    )

    summary_tab, forecast_tab, plans_tab, comparison_tab, downloads_tab = st.tabs(
        ["Summary", "Demand forecast", "Inventory & production", "Model comparison", "Downloads"]
    )

    with summary_tab:
        st.subheader("Recommended operational outputs")
        st.write(
            "Use the product-level demand forecast to plan inventory and production. "
            "Review unmet demand and estimated profit before acting on the plan."
        )
        st.dataframe(
            production[
                [
                    "Product",
                    "Forecast_Demand",
                    "Optimal_Production",
                    "Unmet_Demand",
                    "Solver_Status",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        st.image(analysis["charts"]["operations_overview"], width="stretch")

    with forecast_tab:
        st.subheader("Daily forecast by product")
        st.dataframe(forecast, width="stretch", hide_index=True)
        st.image(analysis["charts"]["demand_forecast"], width="stretch")

    with plans_tab:
        inventory_tab, production_tab, profit_tab = st.tabs(
            ["Inventory", "Production", "Estimated profit"]
        )
        with inventory_tab:
            st.dataframe(inventory, width="stretch", hide_index=True)
            st.image(analysis["charts"]["safety_stock"], width="stretch")
        with production_tab:
            st.dataframe(production, width="stretch", hide_index=True)
            st.image(analysis["charts"]["production_plan"], width="stretch")
        with profit_tab:
            st.dataframe(profit, width="stretch", hide_index=True)
            st.image(analysis["charts"]["profit_contribution"], width="stretch")
            st.image(analysis["charts"]["revenue_vs_cost"], width="stretch")

    with comparison_tab:
        st.info(
            "Prophet supplies the final forecasts. XGBoost, LSTM, and ARIMA are "
            "benchmarks; their scores do not change the selected forecasting model."
        )
        st.dataframe(
            outputs["model_comparison"], width="stretch", hide_index=True
        )
        st.dataframe(
            outputs["forecast_metrics"], width="stretch", hide_index=True
        )
        for chart_name in (
            "model_comparison_test_mae",
            "model_comparison_test_rmse",
            "model_comparison_test_mae_by_product",
        ):
            st.image(
                analysis["charts"][chart_name],
                caption=CHART_TITLES[chart_name],
                width="stretch",
            )

    with downloads_tab:
        st.download_button(
            "Download all results (CSV files and charts)",
            data=analysis["archive"],
            file_name="business_operations_results.zip",
            mime="application/zip",
            type="primary",
        )
        st.subheader("Download individual tables")
        download_columns = st.columns(2)
        for index, (key, label) in enumerate(RESULT_LABELS.items()):
            download_columns[index % 2].download_button(
                label,
                data=outputs[key].to_csv(index=False).encode("utf-8"),
                file_name=f"{key}.csv",
                mime="text/csv",
                key=f"download-{key}",
            )


def main() -> None:
    st.set_page_config(
        page_title="AI Business Operations Planner",
        layout="wide",
    )
    st.title("AI Business Operations & Profit Optimization")
    st.write(
        "Upload sales history to generate product demand forecasts, inventory targets, "
        "a production plan, and estimated financial results."
    )
    st.caption(
        "This is a planning aid. Current inventory, supplier lead times, verified "
        "production capacities, and accounting costs are not included unless supplied "
        "through the project's assumptions."
    )

    with st.sidebar:
        st.header("Planning settings")
        forecast_days = st.number_input(
            "Forecast horizon (days)", min_value=1, max_value=365, value=30
        )
        evaluation_days = st.number_input(
            "Model evaluation window (days)", min_value=7, max_value=365, value=30
        )
        lead_time_days = st.number_input(
            "Supplier lead time (days)", min_value=0.1, value=3.5, step=0.5
        )
        service_level = st.slider(
            "Target service level", min_value=0.51, max_value=0.99, value=0.95
        )
        unit_cost_ratio = st.slider(
            "Estimated unit cost / selling price",
            min_value=0.0,
            max_value=0.95,
            value=0.45,
        )
        holding_cost_ratio = st.number_input(
            "Estimated holding cost / unit cost",
            min_value=0.0,
            value=0.02,
            step=0.01,
        )
        capacity_multiplier = st.number_input(
            "Capacity multiplier", min_value=0.1, value=1.2, step=0.1
        )

    template = pd.DataFrame(
        columns=[
            "transaction_date",
            "product_name",
            "quantity",
            "unit_price",
            "final_amount",
        ]
    )
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="sales_data_template.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader("Upload company sales data (CSV)", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to begin. The required fields are date, product, quantity, unit price, and final sales amount.")
    else:
        try:
            raw_data = _read_upload(uploaded_file)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
            st.error(f"Could not read the uploaded CSV: {error}")
            return

        if raw_data.empty or len(raw_data.columns) < len(INPUT_COLUMNS):
            st.error(
                "The uploaded CSV must contain data and at least five columns for "
                "date, product, quantity, unit price, and final sales amount."
            )
            return

        st.write(f"Loaded {len(raw_data):,} rows and {len(raw_data.columns)} columns.")
        mapping = {}
        csv_columns = [str(column) for column in raw_data.columns]
        with st.expander("Match your CSV columns", expanded=True):
            for field in sorted(INPUT_COLUMNS):
                mapping[field] = st.selectbox(
                    FIELD_LABELS[field],
                    options=csv_columns,
                    index=_suggested_column(csv_columns, field),
                    key=f"mapping-{field}",
                )

        run_clicked = st.button("Run full business analysis", type="primary")
        if run_clicked:
            st.session_state.pop("analysis", None)
            try:
                normalized_data = normalize_uploaded_columns(raw_data, mapping)
                with st.spinner(
                    "Running data cleaning, four-model evaluation, Prophet forecasting, "
                    "inventory planning, production optimization, and profit estimation..."
                ):
                    with tempfile.TemporaryDirectory(
                        prefix="business-operations-"
                    ) as temporary_directory:
                        temporary_path = Path(temporary_directory)
                        input_path = temporary_path / "company_sales.csv"
                        output_path = temporary_path / "results"
                        normalized_data.to_csv(input_path, index=False)
                        outputs = run_pipeline(
                            input_path=input_path,
                            output_dir=output_path,
                            forecast_days=int(forecast_days),
                            evaluation_days=int(evaluation_days),
                            lead_time_days=float(lead_time_days),
                            service_level=float(service_level),
                            unit_cost_ratio=float(unit_cost_ratio),
                            holding_cost_ratio=float(holding_cost_ratio),
                            capacity_multiplier=float(capacity_multiplier),
                        )
                        chart_directory = output_path / "visualizations"
                        charts = {
                            path.stem: path.read_bytes()
                            for path in chart_directory.glob("*.png")
                        }
                        missing_charts = set(CHART_TITLES).difference(charts)
                        if missing_charts:
                            raise RuntimeError(
                                "The pipeline did not generate expected charts: "
                                + ", ".join(sorted(missing_charts))
                            )
                        archive = build_output_archive(output_path)

                st.session_state["analysis"] = {
                    "outputs": outputs,
                    "charts": charts,
                    "archive": archive,
                    "filename": uploaded_file.name,
                    "completed_at": datetime.now().astimezone().strftime(
                        "%Y-%m-%d %H:%M %Z"
                    ),
                }
                st.success("Analysis complete. Review the tabs below or download the outputs.")
            except (
                ValueError,
                RuntimeError,
                OSError,
                pd.errors.ParserError,
            ) as error:
                st.error(f"Analysis could not be completed: {error}")

    analysis = st.session_state.get("analysis")
    if analysis is not None:
        _render_results(analysis)


if __name__ == "__main__":
    main()
