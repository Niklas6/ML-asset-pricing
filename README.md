# Empirical Equity Return Prediction

This project investigates whether simple firm-level price features and market-state variables can predict next-month equity returns in a walk-forward setting. It is intended as a portfolio-style quant research project: the focus is on reproducible feature engineering, out-of-sample validation, and comparison between linear, tree-based, boosting, and neural-network models.

## Project Overview

The pipeline uses historical equity prices from Yahoo Finance to build monthly cross-sectional prediction datasets. For each stock-month observation, the model uses lagged return, volatility, and market beta features to predict the following month's return.

Current modelling approaches include:

- Ridge regression
- Random forest regressor
- Extra tree regressor
- XGBoost regressor
- LightGBM regressor

The evaluation is currently based on out-of-sample metrics comparing the result to the risk-free rate using an R-squared-style metric. The model is trained on a rolling 20-year window, predicts the following year, and is refit annually. I split the evaluation period into three regimes:

- 1990-1999: validation period used for model and parameter selection
- 2000-2009: first test period, covering the dot-com crash and global financial crisis
- 2010-2019: second test period, covering the post-crisis decade

| Period | Extra tree model | LGBM model | RF model | Ridge model | XGB model |
|--------| ---: | ---: | ---: | ---: | ---: |
| 1990-1999 | 0.031 | 0.031 | 0.023 | 0.029 | 0.033 |
| 2000-2009 | -0.012 | -0.009 | -0.021 | -0.003 | -0.004 |
| 2010-2019 | 0.043 | 0.039 | 0.038 | 0.031 | 0.041 |

## Limitations

The current universe is manually selected and therefore subject to survivorship bias. Yahoo Finance data can also include revisions and ticker-history complications.

The current evaluation focuses on predictive error rather than full tradability. The results should not be interpreted as evidence of a deployable trading strategy until portfolio construction, transaction costs, turnover, and risk controls are added.

## Repository Structure

```text
.
+-- README.md                         # Project overview, methodology, limitations, and headline results
+-- .gitignore                        # Excludes environments, caches, notebooks, and generated data
+-- data/
|   +-- raw/                          # Local downloaded Yahoo Finance data, ignored by Git
|   +-- processed/                    # Local generated feature panel, ignored by Git
|   +-- results/                      # Results
+-- scripts/
|   +-- S1_download_data.py            # Download raw equity, benchmark, volume, and risk-free-rate data
|   +-- S2_process_data.py             # Build the processed monthly feature panel
|   +-- run_model.py                   # Run rolling-window model evaluation and save CSV results
|   +-- S3_model_validation.ipynb      # Local notebook for model/parameter validation
|   +-- X1_data_exploration.ipynb      # Local exploratory data-analysis notebook
|   +-- X2_postprocess.ipynb           # Local notebook for result post-processing/plots
+-- src/
    +-- data_loader.py                 # Data download and persistence helpers
    +-- features.py                    # Feature engineering and rolling train/test splits
    +-- Model_circus.py                # Candidate model definitions
    +-- evaluate_prediction.py         # Error and R-squared-style evaluation metrics
```

## Methodology

The project uses a walk-forward cross-sectional prediction setup rather than a random train/test split.

1. Download adjusted close prices and volumes for the stock universe from Yahoo Finance.
2. Download the S&P 500 as the benchmark and the 13-week Treasury bill yield as the risk-free-rate proxy.
3. Convert daily prices to month-end observations.
4. Build one stock-month row per asset with only information available at that month-end.
5. Train models on a rolling 20-year history.
6. Predict the following out-of-sample year.
7. Refit annually and aggregate results by regime.

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
