# Project Review and Repair Report

**Project:** AI-Powered Business Operations & Profit Optimization System  
**Updated:** 2026-11-16  
**Status:** Reproducible Prophet forecasting pipeline with a four-model comparison; Tableau workbook refresh remains manual.

## Summary

The project provides a single root-relative pipeline for cleaning transactions, creating business assumptions, forecasting with Prophet, comparing Prophet against XGBoost, LSTM, and ARIMA, planning inventory and production, estimating profit, and producing current visualizations. Prophet is used for every final product forecast; the other three models are comparison benchmarks. The seven notebooks use the same reusable modules.

The full pipeline completed on the included dataset: 1,980 raw transaction rows became 1,967 cleaned rows across 18 products. Four methods were evaluated per product, producing 72 product/model score rows across shared 30-day validation and test windows. Prophet generated 540 future forecasts (30 per product).

## Forecast model comparison

| Model | Mean validation MAE | Mean validation RMSE | Mean test MAE | Mean test RMSE | Validation wins | Test wins | Forecast products |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARIMA (1,1,1) | **0.7866** | **1.2181** | **0.7572** | **1.1442** | 6 | 4 | 0 |
| LSTM | 0.7880 | 1.2196 | 0.7622 | 1.1474 | 3 | 5 | 0 |
| Prophet | 0.7921 | 1.2247 | 0.7624 | 1.1489 | 2 | 3 | **18** |
| XGBoost | 0.8100 | 1.2499 | 0.7954 | 1.1962 | **7** | **6** | 0 |

Errors are in units per product per day. Cross-product figures are unweighted means of product-level scores. ARIMA has the lowest average test MAE and RMSE in this split; XGBoost has the most validation wins, and LSTM has the most test MAE wins. Prophet is intentionally used to forecast all 18 products regardless of the comparison rankings.

The comparison files are `data/processed/forecast_metrics.csv` (72 product/model rows) and `data/processed/model_comparison.csv` (aggregate report table). Three model-comparison charts show mean test MAE, mean test RMSE, and per-product test MAE. A submission-ready narrative and references are in [submission_report.md](submission_report.md) and [submission_report.pdf](submission_report.pdf).

## Corrections and validation

- Removed machine-specific notebook paths and added `python -m src.run_pipeline`.
- Corrected future-only Prophet forecast generation and compared it with XGBoost, LSTM, and ARIMA on common chronological windows.
- Kept a separate later test window for reporting; test scores are not used to choose the fixed Prophet forecasting method.
- Corrected inventory statistics to use calendar-day demand and made lead time deterministic.
- Added capacity-constrained integer production with explicit unmet demand and checked solver status.
- Corrected revenue to count only served forecast demand.
- Added project setup/run documentation, automated tests, and nine refreshed charts, including the three requested model-comparison graphs.
- **Automated tests: 10 passed.** The regenerated metrics contain four models per product with finite validation and test scores. All 540 future forecasts use Prophet, with 30 rows per product; production stays within capacity and production plus unmet demand balances forecast demand.
- The model comparison is a single historical split on small product-level data. It is not evidence that the rankings will generalize to future dates or other businesses; rolling-origin backtesting is recommended.
- Estimated profit is 1,825.31 in an unspecified dataset currency, and modeled unmet demand is 9.33 units. Cost ratios and capacity are assumptions/proxies, not verified accounting or factory measurements.

## Remaining work

- Refresh and repackage the Tableau workbook and screenshots using current CSVs in Tableau Desktop.
- Replace illustrative cost, lead-time, and capacity assumptions with actual operational data before treating results as business recommendations.
- Check submission formatting requirements with the institution and personalize the report cover page if needed.
