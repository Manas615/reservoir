#!/usr/bin/env python3
import argparse

from sediment_hotspot.preprocessing.flow import build_flow_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-csv", required=True)
    parser.add_argument("--output", default="data/processed/flow_features.csv")
    args = parser.parse_args()
    print(build_flow_features(args.flow_csv, args.output))


if __name__ == "__main__":
    main()
