from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sediment_hotspot.utils.io import ensure_dir, require_file


def save_heatmap_png(heatmap_npy: str | Path, output_path: str | Path = "data/outputs/hotspot_heatmap.png") -> Path:
    heatmap = np.load(require_file(heatmap_npy, "hotspot heatmap"))
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
    plt.colorbar(label="Sediment accumulation risk")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def zone_risk_labels(zone_csv: str | Path) -> list[str]:
    df = pd.read_csv(require_file(zone_csv, "zone risk table"))
    return [f"{row.zone} -> {row.risk * 100:.0f}% sediment accumulation risk" for row in df.itertuples()]
