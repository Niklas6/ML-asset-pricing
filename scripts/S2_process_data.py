import sys
sys.path.append("../")

from src.data_loader import download_price_data, save_raw_data, load_raw_data, save_processed_data
from src.features import generate_features






def main() -> None:

    X = generate_features(train_start_date="1965-01-31", test_end_date="2025-12-31")

    save_processed_data(X, output_dir="../data/processed")

    print(X.columns)
    #Z.to_csv("../data/processed/Z.csv", parse_dates=True)




if __name__ == "__main__":
    main()
