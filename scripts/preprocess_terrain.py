#!/usr/bin/env python3
import argparse

from sediment_hotspot.preprocessing.terrain import compute_terrain_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", required=True, help="SRTM or ASTER DEM GeoTIFF")
    parser.add_argument("--output-dir", default="data/processed/terrain")
    args = parser.parse_args()
    print(compute_terrain_features(args.dem, args.output_dir))


if __name__ == "__main__":
    main()
