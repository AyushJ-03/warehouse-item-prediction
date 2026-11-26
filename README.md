# AI-Powered Business Operations & Profit Optimization System

An end-to-end analytics project for grocery operations. It cleans transaction data, forecasts product demand, estimates inventory reorder targets, creates a capacity-constrained production plan, evaluates profit, and produces data for the Tableau dashboard.

## Requirements

- Python 3.11 or 3.12
- The source CSV at `data/raw/grocery_chain_data.csv`
- The packages listed in `requirements.txt`
- Tableau Desktop (optional; required only to refresh the packaged workbook)

### Windows PowerShell setup

Run these commands from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the complete pipeline

From the project root:

```powershell
python -m src.run_pipeline
```

## Use the business-facing web app

Install the requirements, then start the local app from the project root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

In the browser, upload the company's sales CSV, map its columns to transaction
date, product, quantity, unit price, and final sales amount, and select planning
assumptions. The app runs the complete pipeline and displays product forecasts,
inventory targets, production recommendations, unmet demand, estimated profit,
and model-comparison results. Download all generated CSVs and charts as a ZIP or
download the result tables individually. Analysis results are held in the local
app session; uploaded company data is not added to the repository's sample data.

The app's inventory, cost, and capacity values remain estimates unless actual
company inputs are available. It does not account for current on-hand stock or
open purchase orders.

The command reads `data/raw/grocery_chain_data.csv` and recreates these files in `data/processed/`:

| Output | Contents |
|---|---|
| `clean_data.csv` | Validated, de-duplicated transaction records |
| `business_data.csv` | Clean transactions plus documented model assumptions |
| `forecast.csv` | Exactly one future forecast row per product and calendar day, with the selected model |
| `forecast_metrics.csv` | Per-product MAE/RMSE for Prophet, XGBoost, LSTM, and ARIMA on separate validation and test windows |
| `model_comparison.csv` | Cross-product mean errors and per-product win counts for the four models |
| `inventory_plan.csv` | Daily-demand-based safety stock and reorder points |
| `production_plan.csv` | Integer production, capacity, and unmet demand by product |
| `profit_results.csv` | Revenue, production/holding/shortage costs, and estimated profit |
| `visualizations/` | Nine refreshed PNG charts, including aggregate and per-product four-model comparisons |

Forecast the next 60 days, evaluate on a 30-day holdout, and use a 4-day lead time:

```powershell
python -m src.run_pipeline --forecast-days 60 --evaluation-days 30 --lead-time-days 4
```

Run the automated tests:

```powershell
python -m pytest
```

Regenerate the submission report PDFs after running the pipeline:

```powershell
python report\generate_reports.py
```

## Workflow and model assumptions

1. **Cleaning:** Duplicate records are removed. Rows with invalid dates, nonpositive quantities, negative prices, or negative final amounts are excluded. Refund/return rows with negative final amounts are not modeled as demand.
2. **Business assumptions:** Effective selling price is `final_amount / quantity`. Unit cost is 45% of that price; holding cost is 2% of unit cost; shortage cost is the estimated gross margin (`max(price - unit cost, 0)`). These are illustrative assumptions, not supplier-accounting data. The default lead time is a deterministic 3.5 days (not randomly assigned).
3. **Demand forecasting:** Sales are aggregated to daily product demand and missing calendar dates are filled with zero. Prophet, XGBoost with lag/calendar features, an LSTM network, and ARIMA(1,1,1) are evaluated for every product on the same chronological validation and test windows. Prophet is always used to generate the final future forecast for every product; the other three models are comparison benchmarks. The held-out test scores are reported but do not change the forecasting model used downstream.
4. **Inventory:** Average and sample standard deviation are calculated using calendar-day demand, including zero-sale dates. Safety stock uses the selected service-level z-score and `std(daily demand) * sqrt(lead time)`; reorder point is average daily demand times lead time plus safety stock.
5. **Capacity and production:** The raw data has no factory-capacity field. A transparent proxy estimates a product's daily capacity as its highest observed daily sales volume; the planning capacity is this value times the configurable capacity multiplier and forecast horizon. This is a heuristic, not verified factory capacity. The linear program uses integer production, never produces above forecast demand or capacity, and records unmet demand rather than silently violating constraints.
6. **Profit:** Revenue is earned only on forecast demand served. Production costs apply to every produced unit; holding cost applies to production beyond demand; shortage cost applies to unmet demand. The current plan normally avoids overproduction, so holding cost is typically zero.
7. **Visualization:** The pipeline writes nine updated charts to `data/processed/visualizations/`, including mean test MAE, mean test RMSE, and per-product test MAE comparisons for the four models. The CSV outputs are the source data for the Tableau workbook. Existing `.twbx` extracts/screenshots are historical presentation artifacts; after regenerating outputs, open the workbook in Tableau, reconnect/replace its data sources with the new files, refresh the extract, and save a newly packaged `.twbx`. Tableau Desktop is not required to run the Python pipeline.

All model assumptions can be changed through command-line options shown by:

```powershell
python -m src.run_pipeline --help
```

## Repository layout

```text
dashboard/      Tableau workbook and screenshots
data/raw/       Input grocery transactions
data/processed/ Generated pipeline outputs
notebooks/      Guided, stage-by-stage Jupyter notebooks
report/         Project summary and review report
src/            Reusable pipeline modules and command-line entry point
tests/          Automated pipeline logic tests
```

## Limitations

- The dataset is transaction-level and does not contain actual inventory on hand, purchase orders, supplier lead times, verified manufacturing capacities, or accounting costs. Inventory and production results are therefore planning estimates.
- The single held-out test period is useful for a project demonstration but is not a substitute for rolling-origin backtesting or operational monitoring. Prophet is fixed as the final forecasting model; the comparison scores do not select the deployed model.
- The capacity proxy is based on historical daily sales, not a measured production constraint. Replace it with actual capacities before using the plan operationally.
- The supplied Tableau workbook must be refreshed manually in Tableau after the regenerated CSVs change; its embedded extract is not updated by the Python command.
- The LSTM benchmark is trained on CPU with compact demonstration settings so the comparison can run without a GPU.
