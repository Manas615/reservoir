from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sediment_hotspot.utils.io import require_file


class SedimentHotspotDataset(Dataset):
    """Dataset driven by real preprocessed artifact paths listed in a manifest CSV."""

    REQUIRED_COLUMNS = [
        "satellite_stack",
        "terrain_stack",
        "landuse_stack",
        "flow_sequence",
        "rain_features",
        "target_heatmap",
        "zone_risk",
    ]

    def __init__(self, manifest_csv: str | Path):
        self.manifest = pd.read_csv(require_file(manifest_csv, "training manifest"))
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.manifest.columns]
        if missing:
            raise ValueError(f"Training manifest is missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.manifest.iloc[idx]
        return {
            "satellite": _load_tensor(row["satellite_stack"]),
            "terrain": _load_tensor(row["terrain_stack"]),
            "landuse": _load_tensor(row["landuse_stack"]),
            "flow_sequence": _load_tensor(row["flow_sequence"]),
            "rain_features": _load_tensor(row["rain_features"]),
            "target_heatmap": _load_heatmap(row["target_heatmap"]),
            "zone_risk": _load_tensor(row["zone_risk"]),
        }


def _load_tensor(path: str | Path) -> torch.Tensor:
    return torch.from_numpy(np.load(require_file(path, "dataset tensor")).astype("float32"))


def _load_heatmap(path: str | Path) -> torch.Tensor:
    tensor = _load_tensor(path)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    raise ValueError(f"Target heatmap must have 2 or 3 dimensions: {path}")
