from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

from sediment_hotspot.utils.io import ensure_dir


def compute_terrain_features(dem_path: str | Path, output_dir: str | Path = "data/processed/terrain") -> dict[str, Path]:
    dem_path = Path(dem_path)
    output_dir = ensure_dir(output_dir)
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        profile = src.profile.copy()
        xres = abs(src.transform.a)
        yres = abs(src.transform.e)

    dzdy, dzdx = np.gradient(dem, yres, xres)
    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
    curvature = ndimage.laplace(dem)
    gradient = np.sqrt(dzdx**2 + dzdy**2)
    susceptibility = _normalize(slope) * 0.45 + _normalize(curvature.clip(min=0)) * 0.25 + _normalize(gradient) * 0.30

    profile.update(count=1, dtype="float32", compress="lzw")
    outputs = {
        "slope": output_dir / f"{dem_path.stem}_slope.tif",
        "curvature": output_dir / f"{dem_path.stem}_curvature.tif",
        "gradient": output_dir / f"{dem_path.stem}_gradient.tif",
        "erosion_susceptibility": output_dir / f"{dem_path.stem}_erosion_susceptibility.tif",
    }
    for name, path in outputs.items():
        data = {
            "slope": slope,
            "curvature": curvature,
            "gradient": gradient,
            "erosion_susceptibility": susceptibility,
        }[name]
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data.astype("float32"), 1)
    return outputs


def _normalize(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype="float32")
    lo, hi = np.nanpercentile(finite, [2, 98])
    return np.clip((values - lo) / (hi - lo + 1e-6), 0, 1)
