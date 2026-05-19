# Empirical Equity Return Prediction

This project investigates whether firm-level price features and market-state variables contain predictive information for next-month equity returns.

It is designed as a portfolio-style quantitative research project using machine learning. The focus is on predicting the next month's return by using historical price features, comparing different Machine learning tools 
such as linear, tree-based, and boosting models. Then we backtest how well a portfolio using this prediction would perform in an idealised setting (no transaction cost).

The prediction is inspired by Gu, Kelly, and Xiu's "Empirical Asset Pricing via Machine Learning", which studies how machine learning methods can be used to predict the cross-section of equity returns.

## Project Overview

The prediction pipeline uses historical equity prices from Yahoo Finance to build monthly cross-sectional prediction datasets. For each stock-month observation, the model uses lagged return, volatility, and market beta features to predict the following month's return.

Current modeling approaches include:
- Ridge regressor
- Random forest regressor
- Extra Trees regressor
- XGBoost regressor
- LightGBM regressor

The evaluation is based on out-of-sample metrics that compare model forecasts against a risk-free-rate baseline using an R-squared-style metric. Each model is trained on a rolling 20-year window, predicts the following year, and is refit annually. The evaluation period is split into three regimes:

- 1990-1999: validation period used for model and parameter selection
- 2000-2009: first test period, covering the dot-com crash and global financial crisis
- 2010-2019: second test period, covering the post-crisis decade
### Prediction
For each period we predict the returns and compare the R^2 to the benchmark of the risk free rate which gives the prediction performance:

| Period   | Ridge model | RF model | Extra tree model | XGB model | LGBM model |
|----------| ---: | ---: | ---: | ---: | ---: |
| 1990-1999 | 0.034 | 0.029 | 0.037 | 0.038 | 0.037 |
| 2000-2009 | -0.010 | -0.023 | -0.016 | -0.009 | -0.012 |
| 2010-2019 | 0.033 | 0.042 | 0.044 | 0.041 | 0.039 |

The performance of the prediction is shown in the following figure:
<img src="data/results/performance_slice.jpg" alt="Prediction Performance by Year" width="750">

The strongest results occur in the 1990s validation period and the 2010s test period. All models struggled during 2000-2009, particularly around the dot-com crash and global financial crisis, which suggests that the signal has a strong market dependence.
### Backtest

To test if we can build a strategy that uses the prediction to generate a portfolio of long and short stocks depending on their prediction signal. The portfolio rebalances each month. The compounded average return performance is in the following graphic: 

| Period   | Ridge model | RF model | Extra tree model | XGB model | LGBM model |
|----------| ---: | ---: | ---: | ---: | ---: |
| 1990-1999 | 0.128 | 0.138 | 0.160 | 0.150 | 0.129 |
| 2000-2009 | 0.041 | 0.018 | 0.038 | 0.046 | 0.055 |
| 2010-2019 | 0.081 | 0.052 | 0.095 | 0.081 | 0.038 |

The performance of the backtest is shown in the following figure:

<img src="data/results/graphic_backtest.png" alt="Prediction Performance by Year" width="750">

## How to Run

The project can be run with the following terminal commands:

```bash
pip install -r requirements.txt
python scripts/S1_download_data.py
python scripts/S2_process_data.py
python scripts/run_model.py
```

## Limitations

- The current universe is manually selected and therefore subject to survivorship bias. Yahoo Finance data can also include revisions and ticker-history complications.
- The current evaluation focuses on predictive error rather than full tradability. The results should not be interpreted as evidence of a deployable trading strategy until portfolio construction, transaction costs, turnover, and risk controls are added.

## Repository Structure

```text
.
+-- README.md                         # Project overview, methodology, limitations, and headline results
+-- .gitignore                        # Excludes environments, caches, notebooks, and generated data
+-- data/
|   +-- raw/                          # Local downloaded Yahoo Finance data, ignored by Git to reduce file size
|   +-- processed/                    # Local generated feature panel, ignored by Git to reduce file size
|   +-- results/                      # Results
+-- scripts/
|   +-- S1_download_data.py            # Download raw equity, benchmark, volume, and risk-free-rate data
|   +-- S2_process_data.py             # Build the processed monthly feature panel
|   +-- S3_model_validation.ipynb      # Local notebook for model/parameter validation
|   +-- run_model.py                   # Run rolling-window model evaluation and save CSV results
+-- src/
    +-- data_loader.py                 # Data download and persistence helpers
    +-- features.py                    # Feature engineering and rolling train/test splits
    +-- Models.py                      # Candidate model definitions
    +-- evaluate_prediction.py         # Error and R-squared-style evaluation metrics
```

## Methodology

The project uses a walk-forward cross-sectional prediction setup rather than a random train/test split.

1. Download adjusted close prices for the stock universe, S&P 500 as the benchmark, and the Treasury bill yield as the risk-free-rate proxy from Yahoo Finance.
2. Convert daily prices to month-end observations and build one stock-month row per asset using only information available at that month-end.
3. Train and compare models with different parameters on a rolling 20-year history during the 1990-1999 validation period.
4. Predict the following out-of-sample year using the models with the best-performing parameters.
5. Refit annually and aggregate results by regime.

The current stock-level features are:

- One-month, three-month, six-month, and twelve-month trailing stock returns
- One-month, three-month, six-month, and twelve-month trailing daily-return volatility
- Rolling one-year beta estimated against S&P 500 daily returns
- Beta-scaled benchmark return
- Beta-scaled benchmark volatility

The prediction target is the next-month stock return. The evaluation compares model forecast errors against a risk-free-rate baseline using an out-of-sample R-squared-style metric:

```text
R2_oos = 1 - sum((prediction - realized_return)^2) / sum((realized_return - risk_free_rate)^2)
```

The experiment is split into three regimes:

- Validation: 1990-1999, used for model and parameter selection
- Test 1: 2000-2009, covering the dot-com crash and global financial crisis
- Test 2: 2010-2019, covering the post-crisis decade

This design is meant to test whether the feature set contains persistent predictive information across distinct market environments.






## Summary

The results suggest that simple return, volatility, and beta-based features contain some predictive information for next-month equity returns, but the signal is unstable across market regimes. The models perform best during calmer periods and struggle during major market stress events. These findings are consistent with the view that return prediction is difficult and that predictive performance should be evaluated alongside portfolio construction, transaction costs, turnover, and risk controls.

The next step is to see if this prediction translates in a better performing portfolio. 

## Reference

Gu, Shihao, Bryan Kelly, and Dacheng Xiu. "Empirical asset pricing via machine learning." The Review of Financial Studies 33.5 (2020): 2223-2273.
