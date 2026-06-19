#!/usr/bin/env python3
"""Create Earth Engine acquisition metadata for public reservoir datasets.

Run Earth Engine exports from the Code Editor or extend this script with your
authenticated ee.Initialize() flow. It records the exact public collections to
use for reproducibility.
"""

import argparse

from sediment_hotspot.data.public_sources import write_earth_engine_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservoir", required=True)
    parser.add_argument("--aoi-geojson", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", default="data/raw/earth_engine")
    args = parser.parse_args()
    print(write_earth_engine_manifest(args.reservoir, args.aoi_geojson, args.start_date, args.end_date, args.output_dir))


if __name__ == "__main__":
    main()
