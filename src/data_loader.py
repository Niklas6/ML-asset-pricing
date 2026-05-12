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
        try:
            prices[ticker] = data[ticker]["Close"]
            volumes[ticker] = data[ticker]["Volume"]
        except Exception:
            print(f"Skipping {ticker}: data not available")

    return prices, volumes


def save_raw_data(prices, volumes, output_dir="data/raw"):
    """
    Save raw prices and volumes.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prices.to_csv(output_path / "prices.csv")
    volumes.to_csv(output_path / "volumes.csv")






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



def save_processed_data(Z,output_dir="data/processed"):
    """
    Save processed prices and volumes.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    Z.to_csv(output_path / "Z.csv")


def read_processed_data(data_dir="data/processed"):
    """
    Load processed prices and volumes.
    """
    data_path = Path(data_dir)

    Z = pd.read_csv(
        data_path / "Z.csv",
        index_col=0,
        parse_dates=True
    )
    return Z
