#!/usr/bin/env python3
import argparse

from sediment_hotspot.preprocessing.satellite import preprocess_multispectral_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Sentinel-2 or Landsat multiband GeoTIFF")
    parser.add_argument("--output-dir", default="data/processed/satellite")
    parser.add_argument("--qa-band", type=int, default=None)
    args = parser.parse_args()
    print(preprocess_multispectral_image(args.image, args.output_dir, qa_band=args.qa_band))


if __name__ == "__main__":
    main()
