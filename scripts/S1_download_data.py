import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import download_price_data, save_raw_data, download_riskfree_rate



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

        # Modern growth / tech
        "AMZN", "GOOGL", "NVDA", "META", "AVGO",

        # Added financials
        "GS", "MS", "C", "USB", "BK", "AXP",

        # Added consumer / retail
        "NKE", "SBUX", "TGT", "CVS", "KR",

        # Added industrials / transport / defense
        "HON", "UPS", "FDX", "LMT", "RTX", "GE",

        # Added semiconductors / tech hardware
        "TXN", "QCOM", "AMD", "MU", "AMAT", "LRCX",

        # Added communication / media
        "CMCSA", "NFLX",

        # Added utilities / real estate
        "NEE", "EXC", "D", "O", "SPG",

        # Added energy
        "COP", "SLB", "EOG", "OXY",

        # Added healthcare / biotech / medtech
        "AMGN", "GILD", "MDT", "ISRG", "SYK", "TMO",

        # Added asset managers / insurance
        "BLK", "SCHW", "CB",

        # Added materials / commodities / staples
        "APD", "NEM", "FCX", "ADM", "EL", "TROW"
    ]

    tickers_benchmark= ['^GSPC']
    start_date="1960-01-31"
    end_date= "2025-12-31"


    prices, volumes = download_price_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    prices_bench, volumes_bench = download_price_data(
        tickers=tickers_benchmark,
        start_date=start_date,
        end_date=end_date
    )
    risk_free_rate = download_riskfree_rate(start_date, end_date)
    save_raw_data(prices, volumes,prices_bench, volumes_bench,risk_free_rate, output_dir=PROJECT_ROOT/'data'/'raw')




    #Z.to_csv("../data/processed/Z.csv", parse_dates=True)




if __name__ == "__main__":
    main()
