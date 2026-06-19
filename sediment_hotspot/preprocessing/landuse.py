from pathlib import Path

import numpy as np
import rasterio

from sediment_hotspot.utils.io import ensure_dir


ESA_WORLDCOVER_EROSION_WEIGHTS = {
    10: 0.20,  # tree cover
    20: 0.45,  # shrubland
    30: 0.55,  # grassland
    40: 0.85,  # cropland
    50: 0.65,  # built-up
    60: 0.95,  # bare/sparse vegetation
    80: 0.05,  # permanent water
    90: 0.30,  # herbaceous wetland
}


def landuse_erosion_probability(
    landcover_path: str | Path,
    output_dir: str | Path = "data/processed/landuse",
) -> dict[str, Path]:
    landcover_path = Path(landcover_path)
    output_dir = ensure_dir(output_dir)
    with rasterio.open(landcover_path) as src:
        classes = src.read(1)
        profile = src.profile.copy()

    probability = np.zeros_like(classes, dtype="float32")
    for class_id, weight in ESA_WORLDCOVER_EROSION_WEIGHTS.items():
        probability[classes == class_id] = weight

    agriculture = classes == 40
    bare = classes == 60
    urban = classes == 50
    erosion_hotspot = probability >= 0.65

    profile.update(count=1, dtype="float32", compress="lzw")
    outputs = {
        "source_probability": output_dir / f"{landcover_path.stem}_sediment_source_probability.tif",
        "erosion_hotspot": output_dir / f"{landcover_path.stem}_erosion_hotspot.tif",
        "class_stack": output_dir / f"{landcover_path.stem}_class_stack.npy",
    }
    with rasterio.open(outputs["source_probability"], "w", **profile) as dst:
        dst.write(probability, 1)
    with rasterio.open(outputs["erosion_hotspot"], "w", **profile) as dst:
        dst.write(erosion_hotspot.astype("float32"), 1)
    np.save(outputs["class_stack"], np.stack([agriculture, bare, urban, erosion_hotspot, probability], axis=0).astype("float32"))
    return outputs


def temporal_landuse_change(previous_landcover: str | Path, current_landcover: str | Path) -> dict[str, float]:
    with rasterio.open(previous_landcover) as prev, rasterio.open(current_landcover) as curr:
        prev_arr = prev.read(1)
        curr_arr = curr.read(1)
        pixel_area = abs(curr.transform.a * curr.transform.e)
    changed = prev_arr != curr_arr
    exposed_gain = np.isin(curr_arr, [40, 50, 60]) & ~np.isin(prev_arr, [40, 50, 60])
    return {
        "changed_area": float(changed.sum() * pixel_area),
        "erosion_prone_gain_area": float(exposed_gain.sum() * pixel_area),
    }
