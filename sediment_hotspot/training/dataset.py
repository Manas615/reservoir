from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sediment_hotspot.config import MODEL
from sediment_hotspot.utils.io import require_file


class SedimentHotspotDataset(Dataset):
    REQUIRED_COLUMNS = [
        "satellite_stack",
        "terrain_stack",
        "landuse_stack",
        "flow_sequence",
        "rain_features",
        "zone_risk",
    ]
    OPTIONAL_SHAPES = {
        "dsdi_features": (MODEL.dsdi_features,),
        "dsdi_feature_map": (MODEL.dsdi_features, *MODEL.image_size),
        "graph_nodes": (5, MODEL.graph_features),
        "graph_adjacency": (5, 5),
        "previous_heatmap": (1, *MODEL.image_size),
        "forcing_change": (1,),
    }
    LABEL_COLUMNS = {
        "adaptive": "target_heatmap_adaptive",
        "fixed": "target_heatmap_fixed",
        "default": "target_heatmap",
    }

    def __init__(self, manifest_input: str | pd.DataFrame, label_preference: str = "adaptive"):
        if isinstance(manifest_input, pd.DataFrame):
            self.manifest = manifest_input.reset_index(drop=True)
        else:
            self.manifest = pd.read_csv(require_file(manifest_input, "training manifest"))
        missing = [column for column in self.REQUIRED_COLUMNS if column not in self.manifest.columns]
        if missing:
            raise ValueError(f"Training manifest is missing columns: {missing}")
        self.label_preference = label_preference
        self.target_column = self._resolve_target_column()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.manifest.iloc[index]
        sample = {
            "satellite": _load_array(row.satellite_stack),
            "terrain": _load_array(row.terrain_stack),
            "landuse": _load_array(row.landuse_stack),
            "flow_sequence": _load_array(row.flow_sequence),
            "rain_features": _load_array(row.rain_features),
            "target_heatmap": _load_heatmap(row[self.target_column]),
            "zone_risk": _load_array(row.zone_risk),
            "label_source": str(row.get("label_source", row.get("real_observation_type", "unspecified"))),
            "label_type": str(row.get("label_type", "unspecified")),
        }
        for key, shape in self.OPTIONAL_SHAPES.items():
            sample[key] = _load_optional(row.get(key), shape)
        for key, column in self.LABEL_COLUMNS.items():
            if column in row and isinstance(row[column], str) and Path(row[column]).exists():
                sample[column] = _load_heatmap(row[column])
        return sample

    def _resolve_target_column(self) -> str:
        preferred = self.LABEL_COLUMNS.get(self.label_preference, self.LABEL_COLUMNS["default"])
        if preferred in self.manifest.columns:
            return preferred
        if self.LABEL_COLUMNS["default"] in self.manifest.columns:
            return self.LABEL_COLUMNS["default"]
        available = [column for column in self.LABEL_COLUMNS.values() if column in self.manifest.columns]
        if available:
            return available[0]
        raise ValueError(
            "Training manifest requires one of target_heatmap, target_heatmap_fixed, or target_heatmap_adaptive."
        )


def split_dataset_manifest(
    manifest_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal and reservoir-aware splitting without future leakage."""
    df = manifest_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_val].reset_index(drop=True)
    test_df = df.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df, test_df


def _load_array(path: str) -> torch.Tensor:
    array = np.load(require_file(path, "dataset tensor")).astype("float32")
    return torch.from_numpy(array)


def _load_heatmap(path: str) -> torch.Tensor:
    heatmap = _load_array(path)
    return heatmap.unsqueeze(0) if heatmap.ndim == 2 else heatmap


def _load_optional(value, shape: tuple[int, ...]) -> torch.Tensor:
    if isinstance(value, str) and Path(value).exists():
        tensor = _load_array(value)
        if len(shape) == 1 and tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        return tensor
    return torch.zeros(shape, dtype=torch.float32)
