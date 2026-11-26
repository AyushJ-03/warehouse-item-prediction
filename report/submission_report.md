# AI-Powered Business Operations and Profit Optimization System

## Abstract

This project develops a reproducible analytics pipeline for grocery operations planning. It links transaction cleaning, product-level demand forecasting, inventory targets, capacity-constrained production, and estimated profitability. Prophet, XGBoost, LSTM, and ARIMA are compared for all 18 products on common chronological validation and test windows. Prophet is used for every final future forecast; XGBoost, LSTM, and ARIMA are comparison benchmarks. On the included dataset's test window, ARIMA had the lowest mean MAE (0.7572 units/day) and RMSE (1.1442). Capacity and cost inputs are illustrative proxies because the transaction data do not contain verified factory or accounting data.

## 1. Introduction

Retail operators need to connect demand estimates to inventory, production, and financial planning. Forecasts alone do not indicate how much to produce, whether constraints can be met, or what financial outcomes may follow. This project demonstrates an integrated workflow that transforms grocery transactions into operational planning estimates.

The project is intended as an academic decision-support prototype. Its outputs should not be treated as operational recommendations until its assumptions are replaced with measured business data.

## 2. Problem statement and objectives

The problem addressed is the separation between demand prediction and downstream operational decisions. The project objectives are:

1. Clean and standardize historical grocery transactions.
2. Forecast daily demand for multiple products.
3. Compare forecasting methods on time-ordered data.
4. Estimate safety stock and reorder points from daily demand variability.
5. Plan production under a stated capacity proxy and expose unmet demand.
6. Estimate revenue, costs, and profit for the resulting plan.
7. Export reproducible results for analysis and dashboard presentation.

## 3. Dataset

The included raw CSV contains 1,980 transaction records spanning 18 products. Cleaning removes duplicate records and rows with invalid dates, nonpositive quantities, negative prices, or negative final amounts. The resulting dataset contains 1,967 transactions. The pipeline aggregates product quantities to calendar-day demand and fills dates without a transaction for a product with zero demand.

The source data do not provide actual inventory on hand, purchase orders, supplier lead times, verified production capacity, or accounting unit costs. Those quantities are modeled with explicit assumptions described in Section 5.

## 4. Methodology

### 4.1 Data preparation and business assumptions

Effective selling price is calculated as `final_amount / quantity`. Unit cost is assumed to be 45% of selling price; holding cost is 2% of unit cost; shortage cost is estimated gross margin. The default lead time is a deterministic 3.5 days.

Daily capacity is estimated from the maximum observed aggregate daily sales for each product. The planning-horizon capacity is this daily proxy multiplied by a 1.2 capacity factor and the number of forecast days. This proxy makes the optimization mathematically testable but is not a measurement of factory capability.

### 4.2 Forecasting models

Each product's daily demand series is evaluated using four methods:

- **Prophet:** an additive model for trend and weekly seasonality, with yearly seasonality when the training history is long enough [1].
- **XGBoost:** gradient-boosted trees using lags 1, 2, 7, 14, and 28, rolling means, and cyclic weekday/year features; 100 trees and recursive multi-step prediction [3].
- **LSTM:** a recurrent neural network with a 14-day input window and 16 hidden units; trained for 40 epochs and forecast recursively [4].
- **ARIMA(1,1,1):** a univariate autoregressive integrated moving-average statistical benchmark [2].

### 4.3 Chronological comparison design

For each product, the series is divided into three ordered periods: an initial training segment, a 30-day validation segment, and the final 30-day test segment. All four models are scored on the same validation window and on the same later test window, using only history available before each window. Prophet is fixed as the production forecasting method for every product, as required for this comparison. The test results are reported for comparison and do not determine which model is used for future forecasts.

MAE measures average absolute error and is less sensitive to large errors. RMSE gives larger errors more influence. Both metrics are reported in units per product per day. Cross-product summary metrics are unweighted means of product-level errors, so each product contributes equally regardless of sales volume.

### 4.4 Inventory, production, and profit

Daily product demand mean and sample standard deviation are used to estimate safety stock at a 95% service level and the default lead time. Reorder point is average demand during lead time plus safety stock.

An integer linear program minimizes estimated production and shortage costs. Production cannot exceed forecast demand or the planning-horizon capacity proxy; unmet demand is explicit in the solution. Profit is estimated using revenue on served forecast demand, production cost on all produced units, holding cost on excess production, and shortage cost on unmet demand.

### 4.5 Reproducibility

The complete pipeline can be run from the project root using:

```powershell
python -m src.run_pipeline
```

Generated CSV outputs, including product-level model metrics and the aggregate comparison table, are saved under `data/processed/`. Automated checks are run with:

```powershell
python -m pytest
```

The current verification run completed all 10 tests.

## 5. Forecast model comparison

The following table reports the actual regenerated results from the included data. Validation and test scores compare the four methods on identical chronological windows. Prophet is used to generate the final forecasts for all products, independent of these rankings.

| Model | Mean validation MAE | Mean validation RMSE | Mean test MAE | Mean test RMSE | Validation wins | Test wins | Forecast products |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARIMA (1,1,1) | **0.7866** | **1.2181** | **0.7572** | **1.1442** | 6 | 4 | 0 |
| LSTM | 0.7880 | 1.2196 | 0.7622 | 1.1474 | 3 | **5** | 0 |
| Prophet | 0.7921 | 1.2247 | 0.7624 | 1.1489 | 2 | 3 | **18** |
| XGBoost | 0.8100 | 1.2499 | 0.7954 | 1.1962 | **7** | 6 | 0 |

*MAE and RMSE are measured in units per product per day. “Wins” count the lowest-error method per product. “Forecast products” counts products for which that method generated the final future forecast; Prophet is fixed for all 18.*

ARIMA has the lowest average test MAE (0.7572) and RMSE (1.1442) in this split. LSTM is close to Prophet on mean test MAE (0.7622 versus 0.7624), while XGBoost has the highest mean test MAE (0.7954). Per-product wins vary by metric: LSTM has five test-MAE wins and XGBoost has six. Prophet was deliberately used for all 18 final forecasts; this is the chosen deployment rule, not a claim that Prophet won the accuracy comparison.

The test-set comparison is retrospective evidence for this dataset and period. It does not establish that the same ranking will hold for future periods or other retailers. A single validation/test split is limited evidence; rolling-origin evaluation would provide a stronger comparison.

The machine-readable scores are in `data/processed/forecast_metrics.csv` and `data/processed/model_comparison.csv`. The comparison graphs are `model_comparison_test_mae.png`, `model_comparison_test_rmse.png`, and `model_comparison_test_mae_by_product.png` under `data/processed/visualizations/`.

## 6. Operational results

The full pipeline completed with 1,967 cleaned transactions, 18 products, and 540 Prophet forecast rows (30 per product), covering 20 July through 4 September 2025. The production model returned an optimal solver status for all products, stayed within the modeled capacities, and reported total unmet demand of 9.33 units across the horizon. Estimated total portfolio profit is 1,825.31 in an unspecified dataset currency.

These financial and capacity figures are illustrative outputs from assumed costs and proxy capacity. They are not verified savings, audited profit, or a tested factory plan.

## 7. Discussion

Prophet, XGBoost, and LSTM are the forecasting approaches; ARIMA is a statistical benchmark. The experiment evaluates all four, but Prophet is fixed as the forecasting method used by the inventory and production stages.

### Why retain Prophet as a core model?

Prophet is used for all products because it provides an interpretable trend and weekly-seasonality model with less hand-built lag and calendar feature engineering than XGBoost or LSTM. In this implementation, yearly seasonality is enabled only when at least 730 daily observations are available. Holiday effects and forecast intervals are not used by the planning stages. The measured errors do **not** show Prophet as the most accurate model in this split: ARIMA has lower average test errors, and LSTM is very close to Prophet. Prophet is the fixed model required by the project, not the measured accuracy winner. The model ranking may change under repeated rolling-origin tests.

The downstream planning stages are tightly dependent on forecast quality and on the capacity and cost assumptions. A small forecast error does not by itself establish that the production plan is feasible in a real facility. Real capacity, inventory, supplier lead-time, and financial inputs are required to make the decisions operationally meaningful.

## 8. Limitations and future work

- Replace the capacity proxy, cost ratios, and lead-time assumption with verified operational values.
- Perform rolling-origin backtesting across multiple forecast origins and report product-level uncertainty.
- Consider prediction intervals and asymmetric costs for over-forecasting versus under-forecasting.
- Incorporate stock on hand, replenishment orders, shelf life, waste, and supplier constraints into the inventory and production models.
- Refresh the packaged Tableau workbook and screenshots from the regenerated outputs before final submission.
- Confirm the institution's required citation style, formatting, and report structure before submitting.

## 9. Conclusion

This project demonstrates a reproducible path from grocery transaction data to Prophet forecasts, inventory targets, production decisions, profit estimates, and visual outputs. A four-model chronological comparison evaluates Prophet against XGBoost, LSTM, and ARIMA using shared validation and test windows. Prophet supplies every final forecast; the alternatives provide performance benchmarks. The system is suitable as an academic prototype; its recommendations require validation with real operational and accounting data before practical use.

## References

1. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician, 72*(1), 37–45. https://doi.org/10.1080/00031305.2017.1380080
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
3. Hyndman, R. J., & Athanasopoulos, G. (2021). Simple methods. In *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/simple-methods.html
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785
4. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
