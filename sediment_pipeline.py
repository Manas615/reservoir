#!/usr/bin/env python3
"""Pipeline entry point for sediment hotspot modules.

Rainfall forecasting is intentionally not retrained here. Use the completed
rainfall prediction output as an input via --rainfall.
"""

import argparse

from sediment_hotspot.preprocessing.flow import build_flow_features
from sediment_hotspot.preprocessing.landuse import landuse_erosion_probability
from sediment_hotspot.preprocessing.rainfall_adapter import rainfall_prediction_features
from sediment_hotspot.preprocessing.satellite import preprocess_multispectral_image
from sediment_hotspot.preprocessing.terrain import compute_terrain_features
from sediment_hotspot.training.train_fusion import train
from sediment_hotspot.inference.predict_hotspots import predict


def main():
    parser = argparse.ArgumentParser(description="Sediment hotspot prediction pipeline")
    parser.add_argument("--satellite-image")
    parser.add_argument("--dem")
    parser.add_argument("--landcover")
    parser.add_argument("--flow-csv")
    parser.add_argument("--rainfall", default="rainfall_predictions.csv")
    parser.add_argument("--train-manifest")
    parser.add_argument("--model")
    parser.add_argument("--sample")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    if args.preprocess:
        if args.satellite_image:
            print(preprocess_multispectral_image(args.satellite_image))
        if args.dem:
            print(compute_terrain_features(args.dem))
        if args.landcover:
            print(landuse_erosion_probability(args.landcover))
        if args.flow_csv:
            print(build_flow_features(args.flow_csv))
        print(rainfall_prediction_features(args.rainfall))

    if args.train:
        if not args.train_manifest:
            raise ValueError("--train-manifest is required with --train")
        print(train(args.train_manifest, args.epochs, batch_size=4, output_dir="data/models"))

    if args.predict:
        if not args.model or not args.sample:
            raise ValueError("--model and --sample are required with --predict")
        print(predict(args.model, args.sample))


if __name__ == "__main__":
    main()
