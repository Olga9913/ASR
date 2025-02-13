import argparse

import gdown

from src.utils.load_utils import download_best_model


def main(path):
    downloaded_path = download_best_model(path)
    print(f"Model downloaded to: {downloaded_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the best model.")
    parser.add_argument("--path", required=False, type=str, help="Path to save the model")
    args = parser.parse_args()
    main(args.path)