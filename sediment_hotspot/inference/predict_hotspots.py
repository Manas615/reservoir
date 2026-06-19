import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
from sediment_hotspot.utils.io import ensure_dir, require_file


def predict(model_path: str, sample_npz: str, output_dir: str = "data/outputs") -> dict[str, Path]:
    checkpoint = torch.load(require_file(model_path, "fusion model"), map_location="cpu")
    sample = np.load(require_file(sample_npz, "inference sample"))

    model = SedimentHotspotFusionNet()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    batch = {
        "satellite": _batched(sample["satellite"]),
        "terrain": _batched(sample["terrain"]),
        "landuse": _batched(sample["landuse"]),
        "flow_sequence": _batched(sample["flow_sequence"]),
        "rain_features": _batched(sample["rain_features"]),
    }
    with torch.no_grad():
        output = model(batch)

    out_dir = ensure_dir(output_dir)
    heatmap_path = out_dir / "future_sediment_hotspot_heatmap.npy"
    zone_path = out_dir / "zone_sediment_risk.csv"
    attention_path = out_dir / "modality_attention.csv"

    np.save(heatmap_path, output["heatmap"].squeeze().numpy())
    zone_risk = output["zone_risk"].squeeze().numpy()
    pd.DataFrame({"zone": [f"Zone {chr(65 + i)}" for i in range(len(zone_risk))], "risk": zone_risk}).to_csv(zone_path, index=False)
    pd.DataFrame(
        {
            "modality": ["satellite", "flow", "terrain", "landuse", "rain"],
            "attention_weight": output["attention"].squeeze().numpy(),
        }
    ).to_csv(attention_path, index=False)
    return {"heatmap": heatmap_path, "zone_risk": zone_path, "attention": attention_path}


def _batched(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype("float32")).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-dir", default="data/outputs")
    args = parser.parse_args()
    print(predict(args.model, args.sample, args.output_dir))


if __name__ == "__main__":
    main()
