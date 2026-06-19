from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from sediment_hotspot.config import RASTER
from sediment_hotspot.utils.io import ensure_dir


def _read_band(src: rasterio.DatasetReader, band_index: int) -> np.ndarray:
    return src.read(band_index, out_dtype="float32", resampling=Resampling.bilinear)


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + 1e-6)


def cloud_mask_from_qa(qa_band: np.ndarray) -> np.ndarray:
    """Return True for clear pixels using Sentinel/Landsat QA-style bit masks."""
    qa = qa_band.astype("uint32")
    cloud = (qa & (1 << 10)) > 0
    cirrus = (qa & (1 << 11)) > 0
    return ~(cloud | cirrus)


def preprocess_multispectral_image(
    image_path: str | Path,
    output_dir: str | Path = "data/processed/satellite",
    blue_band: int = 1,
    green_band: int = 2,
    red_band: int = 3,
    nir_band: int = 4,
    swir_band: int = 5,
    qa_band: int | None = None,
) -> dict[str, Path]:
    """Extract water, turbidity, plume, and shoreline products from a raster.

    Expected band order can be overridden for Sentinel-2 or Landsat exports.
    """
    image_path = Path(image_path)
    output_dir = ensure_dir(output_dir)

    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        blue = _read_band(src, blue_band)
        green = _read_band(src, green_band)
        red = _read_band(src, red_band)
        nir = _read_band(src, nir_band)
        swir = _read_band(src, swir_band)
        clear = np.ones_like(green, dtype=bool)
        if qa_band is not None:
            clear = cloud_mask_from_qa(_read_band(src, qa_band))

    ndwi = normalized_difference(green, nir)
    mndwi = normalized_difference(green, swir)
    ndti = normalized_difference(red, green)
    water_mask = (ndwi > RASTER.ndwi_threshold) & (mndwi > 0) & clear

    turbidity = np.where(water_mask, ndti, np.nan)
    threshold = np.nanpercentile(turbidity, RASTER.turbidity_percentile)
    plume_mask = np.where(water_mask, turbidity >= threshold, False)

    profile.update(count=1, dtype="float32", nodata=None, compress="lzw")
    outputs = {
        "water_mask": output_dir / f"{image_path.stem}_water_mask.tif",
        "turbidity": output_dir / f"{image_path.stem}_turbidity.tif",
        "plume_mask": output_dir / f"{image_path.stem}_plume_mask.tif",
        "clean_stack": output_dir / f"{image_path.stem}_clean_stack.npy",
    }

    with rasterio.open(outputs["water_mask"], "w", **profile) as dst:
        dst.write(water_mask.astype("float32"), 1)
    with rasterio.open(outputs["turbidity"], "w", **profile) as dst:
        dst.write(np.nan_to_num(turbidity, nan=0).astype("float32"), 1)
    with rasterio.open(outputs["plume_mask"], "w", **profile) as dst:
        dst.write(plume_mask.astype("float32"), 1)

    clean_stack = np.stack(
        [
            np.where(clear, band, 0)
            for band in (blue, green, red, nir, swir, np.nan_to_num(ndti))
        ],
        axis=0,
    ).astype("float32")
    np.save(outputs["clean_stack"], clean_stack)
    return outputs


def shoreline_change_area(previous_water_mask: str | Path, current_water_mask: str | Path) -> dict[str, float]:
    with rasterio.open(previous_water_mask) as prev, rasterio.open(current_water_mask) as curr:
        prev_mask = prev.read(1) > 0.5
        curr_mask = curr.read(1) > 0.5
        pixel_area = abs(curr.transform.a * curr.transform.e)

    gained = np.logical_and(~prev_mask, curr_mask).sum() * pixel_area
    lost = np.logical_and(prev_mask, ~curr_mask).sum() * pixel_area
    return {"water_gain_area": float(gained), "water_loss_area": float(lost)}
