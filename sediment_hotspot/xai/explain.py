from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sediment_hotspot.config import MODEL
from sediment_hotspot.utils.io import ensure_dir, require_file, write_table


def compute_satellite_gradcam(model, batch: dict[str, torch.Tensor]) -> np.ndarray:
    """Compute PyTorch Grad-CAM on the satellite spatial CNN branch using retain_grad."""
    model.eval()
    x = batch["satellite"].clone().detach().requires_grad_(True)
    feat = model.satellite_cnn.encoder(x)
    feat.retain_grad()
    pooled = model.satellite_cnn.pool(feat).flatten(1)

    batch_mod = dict(batch)
    terrain_rep = model.terrain_cnn(batch["terrain"])
    landuse_rep = model.landuse_cnn(batch["landuse"])
    flow_rep = model.flow_lstm(batch["flow_sequence"])
    rain_rep = model.rain_encoder(batch["rain_features"])

    device = x.device
    batch_size = x.shape[0]
    z = batch.get("dsdi_features", torch.zeros((batch_size, MODEL.dsdi_features), device=device))
    dsdi_out = model.dsdi(z)
    dsdi_rep = model.dsdi_encoder(z, dsdi_out)
    graph_nodes = batch.get("graph_nodes", torch.zeros((batch_size, 5, MODEL.graph_features), device=device))
    graph_rep, _ = model.gnn(graph_nodes, batch.get("graph_adjacency", None))

    reps = {
        "satellite": pooled,
        "flow": flow_rep,
        "terrain": terrain_rep,
        "landuse": landuse_rep,
        "rain": rain_rep,
        "graph": graph_rep,
        "dsdi": dsdi_rep,
    }
    fused, _ = model.fusion(reps, dsdi_prior=dsdi_out)
    heatmap = model.heatmap_head(fused).view(-1, 1, *model.image_size)

    score = heatmap.mean()
    model.zero_grad()
    score.backward()

    grad = feat.grad
    if grad is None:
        weights = torch.ones((feat.shape[0], feat.shape[1], 1, 1), device=feat.device)
    else:
        weights = grad.mean(dim=(-1, -2), keepdim=True)
    cam = F.relu((weights * feat).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=model.image_size, mode="bilinear", align_corners=False)
    cam_np = cam.squeeze().detach().cpu().numpy()
    cam_norm = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-6)
    return cam_norm.astype("float32")


def compute_feature_attributions(model, batch: dict[str, torch.Tensor]) -> pd.DataFrame:
    """Compute gradient-based feature importance for input environmental modalities."""
    model.eval()
    batch_g = {k: v.clone().detach().requires_grad_(True) if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in batch.items()}
    out = model(batch_g)
    score = out["heatmap"].mean()
    score.backward()

    attributions = {}
    for name in ["satellite", "terrain", "landuse", "flow_sequence", "rain_features", "dsdi_features"]:
        if name in batch_g and isinstance(batch_g[name], torch.Tensor) and batch_g[name].grad is not None:
            attributions[name] = float(batch_g[name].grad.abs().mean().item())

    total = sum(attributions.values()) + 1e-8
    rows = [{"modality": k, "attribution_score": v / total} for k, v in attributions.items()]
    return pd.DataFrame(rows).sort_values("attribution_score", ascending=False)


def attention_reason_codes(attention_csv: str | Path, output_path: str | Path = "data/outputs/explanation.csv") -> Path:
    attention = pd.read_csv(require_file(attention_csv, "attention weights"))
    sort_col = "fusion_weight" if "fusion_weight" in attention.columns else ("attention_weight" if "attention_weight" in attention.columns else attention.columns[-1])
    top = attention.sort_values(sort_col, ascending=False).head(3)
    phrases = {
        "rain": "heavy rainfall forecast",
        "terrain": "steep and erosive upstream terrain",
        "landuse": "agricultural, bare, or urban sediment source areas",
        "flow": "high transport capacity from discharge and inflow",
        "satellite": "recent muddy-water plume observations",
        "graph": "upstream watershed sediment connectivity",
        "dsdi": "elevated dynamic deposition index",
    }
    explanation = "High sediment risk due to " + " + ".join(phrases.get(m, m) for m in top["modality"]) + "."
    out = pd.DataFrame({"explanation": [explanation], "top_modalities": [", ".join(top["modality"])]})
    return write_table(out, output_path)
