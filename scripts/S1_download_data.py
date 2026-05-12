import sys
sys.path.append("../")

from src.data_loader import download_price_data, save_raw_data, load_raw_data, save_processed_data
from src.features import generate_features






def main() -> None:
    tickers = [
        # Long-history core
        "KO", "PG", "XOM", "CVX", "IBM",
        "MMM", "JNJ", "MRK", "PFE", "BMY",
        "LLY", "ABT", "CL", "GIS", "KMB",
        "MO", "MCD", "DIS", "BA", "CAT",
        "DE", "EMR", "ITW", "SHW", "IP",
        "SO", "DUK", "ED", "AEP", "T",

        # Later but still useful
        "PEP", "WMT", "LOW", "HD", "COST",
        "JPM", "BAC", "WFC", "UNH", "MSFT",
        "AAPL", "INTC", "ORCL", "CSCO", "ADBE",

        # Modern growth/tech
        "AMZN", "GOOGL", "NVDA", "META", "AVGO"
    ]

    prices, volumes = download_price_data(
        tickers=tickers,
        start_date="1960-01-31",
        end_date="2025-12-31"
    )
    save_raw_data(prices, volumes, output_dir="../data/raw")

    Z = generate_features(train_start_date="1965-01-31", test_end_date="2025-12-31")

    save_processed_data(Z, output_dir="../data/processed")

    #Z.to_csv("../data/processed/Z.csv", parse_dates=True)




if __name__ == "__main__":
    main()
