# Empirical Equity Return Prediction

This project investigates whether simple firm-level price features and market-state variables can predict next-month equity returns in a walk-forward setting. It is intended as a portfolio-style quant research project: the focus is on reproducible feature engineering, out-of-sample validation, and comparison between linear, tree-based, boosting, and neural-network models.

## Project Overview

The pipeline uses historical equity prices from Yahoo Finance to build monthly cross-sectional prediction datasets. For each stock-month observation, the model uses lagged return, volatility, and market beta features to predict the following month's return.

Current modelling approaches include:

- Ridge regression
- Decision trees
- Random forests
- Extra trees
- XGBoost regressors
- A small neural-network regressor

The evaluation is currently based on walk-forward out-of-sample R-squared and RMSE-style error metrics. A portfolio backtest layer is planned so the predictions can also be evaluated using rank correlation, long-short returns, Sharpe ratio, drawdown, turnover, and transaction-cost sensitivity.

## Repository Structure

```text
.
+-- data/
|   +-- raw/                    # Downloaded market data, ignored by Git
|   +-- processed/              # Generated feature matrices, ignored by Git
|   +-- results/evaluation/     # Saved validation/test summaries
+-- notebooks/                  # Exploratory notebooks
+-- scripts/
|   +-- S1_download_data.py     # Download raw price, benchmark, and risk-free data
|   +-- S2_process_data.py      # Build processed feature dataset
|   +-- run_model.py            # Run walk-forward model validation
+-- src/
    +-- data_loader.py          # Data download and persistence helpers
    +-- features.py             # Feature engineering and train/validation splits
    +-- Model_circus.py         # Model definitions
    +-- evaluate_prediction.py  # Prediction evaluation helpers
```

## Methodology

The feature construction currently includes:

- One-month, three-month, six-month, and twelve-month stock returns
- Monthly and rolling return volatility
- Rolling one-year market beta against the S&P 500
- Beta-scaled market return and market volatility features
- Next-month stock return as the prediction target

The train/validation design is walk-forward:

1. Train on a fixed historical window.
2. Validate on the following out-of-sample year.
3. Roll the window forward and repeat.
4. Aggregate performance across validation years.

This avoids random train/test splits, which are usually inappropriate for time-series financial prediction.

## Setup

Create and activate a virtual environment, then install the main dependencies:

```bash
pip install pandas numpy scikit-learn xgboost yfinance statsmodels jupyter
```

The project was developed on Windows with Python 3.14, but the code should be portable to a recent Python 3 version once dependencies are installed.

## Reproducing the Pipeline

From the `scripts` directory:

```bash
python S1_download_data.py
python S2_process_data.py
python run_model.py
```

The current scripts use paths relative to the `scripts` directory. A future cleanup step should replace these with project-root-relative paths so commands can be run from the repository root.

## Current Status

The project is functional as a research prototype, but it is not yet a fully polished public research repo. The core idea and modelling loop are in place, but the following items should be addressed before treating the results as final:

- Add a clean dependency file, such as `requirements.txt` or `pyproject.toml`.
- Refactor relative paths so scripts run from the project root.
- Clean unused imports, commented code, empty files, and archived experiments.
- Add tests for feature alignment, beta calculation, train/validation splits, and evaluation metrics.
- Re-run and commit a complete final validation/test result table.
- Add portfolio-style metrics beyond predictive R-squared.
- Document survivorship bias from the manually selected stock universe.

## Limitations

The current universe is manually selected and therefore subject to survivorship bias. Yahoo Finance data can also include revisions and ticker-history complications. The current evaluation focuses on predictive error rather than full tradability, so the results should not be interpreted as evidence of a deployable trading strategy until portfolio construction, transaction costs, turnover, and risk controls are added.

## Intended CV Framing

A concise CV description after cleanup could be:

> Built a walk-forward equity return prediction pipeline using lagged returns, volatility, market beta, and benchmark features; compared Ridge, tree-based models, XGBoost, and neural networks using out-of-sample validation and planned portfolio-level performance metrics.
