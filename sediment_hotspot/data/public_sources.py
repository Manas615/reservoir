"""Public dataset acquisition helpers.

These functions intentionally use real public sources only. They prepare query
metadata or Earth Engine exports; they do not fabricate fallback rasters.
"""

from pathlib import Path
from typing import Iterable

from sediment_hotspot.utils.io import ensure_dir


SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
LANDSAT_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
SRTM_COLLECTION = "USGS/SRTMGL1_003"
ASTER_COLLECTION = "NASA/ASTER_GED/AG100_003"
ESA_WORLDCOVER_COLLECTION = "ESA/WorldCover/v200"
MODIS_LANDCOVER_COLLECTION = "MODIS/061/MCD12Q1"


def write_earth_engine_manifest(
    reservoir_name: str,
    aoi_geojson: str | Path,
    start_date: str,
    end_date: str,
    output_dir: str | Path = "data/raw/earth_engine",
) -> Path:
    """Write a reproducible acquisition manifest for Earth Engine exports."""
    output_dir = ensure_dir(output_dir)
    manifest = output_dir / f"{reservoir_name}_ee_manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"reservoir={reservoir_name}",
                f"aoi_geojson={aoi_geojson}",
                f"start_date={start_date}",
                f"end_date={end_date}",
                f"sentinel2={SENTINEL2_COLLECTION}",
                f"landsat={LANDSAT_COLLECTION}",
                f"srtm={SRTM_COLLECTION}",
                f"aster={ASTER_COLLECTION}",
                f"esa_worldcover={ESA_WORLDCOVER_COLLECTION}",
                f"modis_landcover={MODIS_LANDCOVER_COLLECTION}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_required_rasters(paths: Iterable[str | Path]) -> list[Path]:
    missing = [Path(path) for path in paths if not Path(path).exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required public-data raster exports are missing:\n"
            f"{formatted}\nRun scripts/download_public_data_gee.py with a valid AOI."
        )
    return [Path(path) for path in paths]
