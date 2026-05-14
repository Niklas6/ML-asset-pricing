import yfinance as yf
import pandas as pd
from pathlib import Path


def download_price_data(tickers, start_date, end_date):
    """
    Download adjusted close prices and volume data from Yahoo Finance.
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        group_by="ticker",
        progress=False
    )

    prices = pd.DataFrame()
    volumes = pd.DataFrame()



    for ticker in tickers:
        prices[ticker] = data[ticker]["Close"]
        volumes[ticker] = data[ticker]["Volume"]

    return prices, volumes

def download_riskfree_rate(start_date, end_date):
    rf = yf.Ticker("^IRX").history(
        start=start_date,
        end=end_date
    )

    rf = rf[["Close"]].rename(columns={"Close": "rf_monthly"})
    rf_monthly = (1 + rf.resample("ME").last() / 100) ** (1 / 12)-1
    rf_monthly.index = rf_monthly.index.date
    rf_monthly.index.name = "Date"
    return rf_monthly


def save_raw_data(prices, volumes,prices_bench, volumes_bench,risk_free_rate, output_dir="data/raw"):
    """
    Save raw prices and volumes.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prices.to_csv(output_path / "prices.csv")
    volumes.to_csv(output_path / "volumes.csv")

    prices_bench.to_csv(output_path / "prices_bench.csv")
    volumes_bench.to_csv(output_path / "volumes_bench.csv")
    risk_free_rate.to_csv(output_path / "risk_free_rate.csv")






def load_raw_data(data_dir="data/raw"):
    """
    Load previously saved raw prices and volumes.
    """
    data_path = Path(data_dir)

    prices = pd.read_csv(
        data_path / "prices.csv",
        index_col=0,
        parse_dates=True
    )

    volumes = pd.read_csv(
        data_path / "volumes.csv",
        index_col=0,
        parse_dates=True
    )

    return prices, volumes



def save_processed_data(X,output_dir="data/processed"):
    """
    Save processed prices and volumes.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X.to_csv(output_path / "X.csv")


def read_processed_data(data_dir="data/processed"):
    """
    Load processed prices and volumes.
    """
    data_path = Path(data_dir)

    X = pd.read_csv(
        data_path / "X.csv",
        index_col=0,
        parse_dates=True
    )
    return X
