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
- A small neural-network regressor

The evaluation is currently based on ou-of-sample metrics comparing the result to the risk-free rate in the R^2 metrics. The model is trained on a rolling 20-year window, predicts the following year, and is refit annually. I split the evaluation period into three regimes:


- 1990-1999: validation period used for model and parameter selection
- 2000-2009: first test period, covering the dot-com crash and global financial crisis
- 2010-2019: second test period, covering the post-crisis decade

| Year | Extra tree model | Ridge model | XGBmodel1 | XGBmodel2 |
|------| ---: | ---: | ---: | ---: |
| 1990-1999 | 0.031 | 0.028 | 0.033 | 0.033 |
| 2000-2009 | -0.009 | 0.002 | -0.004 | -0.004 |
| 2010-2019 | 0.043 | 0.036 | 0.041 | 0.041 |




## Limitations

The current universe is manually selected and therefore subject to survivorship bias. Especially, since Yahoo Finance data can also include revisions and ticker-history complications. 

The current evaluation focuses on predictive error rather than full tradability. The results should not be interpreted as evidence of a deployable trading strategy until portfolio construction, transaction costs, turnover, and risk controls are added.


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


