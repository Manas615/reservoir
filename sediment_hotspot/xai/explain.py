from pathlib import Path

import numpy as np
import pandas as pd

from sediment_hotspot.utils.io import require_file, write_table


def attention_reason_codes(attention_csv: str | Path, output_path: str | Path = "data/outputs/explanation.csv") -> Path:
    attention = pd.read_csv(require_file(attention_csv, "attention weights"))
    top = attention.sort_values("attention_weight", ascending=False).head(3)
    phrases = {
        "rain": "heavy rainfall forecast",
        "terrain": "steep and erosive upstream terrain",
        "landuse": "agricultural, bare, or urban sediment source areas",
        "flow": "high transport capacity from discharge and inflow",
        "satellite": "recent muddy-water plume observations",
    }
    explanation = "High sediment risk due to " + " + ".join(phrases.get(m, m) for m in top["modality"]) + "."
    out = pd.DataFrame({"explanation": [explanation], "top_modalities": [", ".join(top["modality"])]})
    return write_table(out, output_path)


def gradcam_placeholder(feature_map: np.ndarray, weights: np.ndarray) -> np.ndarray:
    cam = np.maximum((feature_map * weights.reshape(-1, 1, 1)).sum(axis=0), 0)
    cam = cam / (cam.max() + 1e-6)
    return cam.astype("float32")


def shap_export_note(output_path: str | Path = "data/outputs/shap_instructions.txt") -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Use shap.DeepExplainer or shap.GradientExplainer on SedimentHotspotFusionNet "
        "with a background batch from the real training manifest. Export per-modality "
        "SHAP values beside attention weights for audit review.\n",
        encoding="utf-8",
    )
    return path
