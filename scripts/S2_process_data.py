import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import  save_processed_data
from src.features import generate_features






def main() -> None:

    X = generate_features(train_start_date="1965-01-31", test_end_date="2025-12-31")

    save_processed_data(X, output_dir=PROJECT_ROOT/'data'/'processed')




if __name__ == "__main__":
    main()
