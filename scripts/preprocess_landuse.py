#!/usr/bin/env python3
import argparse

from sediment_hotspot.preprocessing.landuse import landuse_erosion_probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landcover", required=True, help="ESA WorldCover or remapped MODIS GeoTIFF")
    parser.add_argument("--output-dir", default="data/processed/landuse")
    args = parser.parse_args()
    print(landuse_erosion_probability(args.landcover, args.output_dir))


if __name__ == "__main__":
    main()
