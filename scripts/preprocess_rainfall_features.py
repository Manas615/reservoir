#!/usr/bin/env python3
import argparse

from sediment_hotspot.preprocessing.rainfall_adapter import rainfall_prediction_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rainfall-predictions", default="rainfall_predictions.csv")
    parser.add_argument("--output-dir", default="data/processed/rainfall")
    args = parser.parse_args()
    print(rainfall_prediction_features(args.rainfall_predictions, args.output_dir))


if __name__ == "__main__":
    main()
