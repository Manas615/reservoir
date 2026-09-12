import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

from sediment_hotspot.config import MODEL, SEDIMENT
from sediment_hotspot.losses.physics_loss import physics_informed_sediment_loss
from sediment_hotspot.losses.temporal_loss import temporal_consistency_loss
from sediment_hotspot.models.causal_attention import CausalAttentionFusion
from sediment_hotspot.models.contrastive import ModalContrastiveAlignment
from sediment_hotspot.models.dsdi import DynamicSedimentDepositionIndex
from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
from sediment_hotspot.models.gnn import WatershedReservoirGNN
from sediment_hotspot.models.uncertainty import mc_dropout_predict, classify_uncertainty_regions
from sediment_hotspot.preprocessing.adaptive_threshold import adaptive_hotspot_threshold, generate_derived_labels
from sediment_hotspot.xai.explain import compute_satellite_gradcam, compute_feature_attributions


def test_dsdi_computation():
    dsdi_module = DynamicSedimentDepositionIndex(MODEL.dsdi_features)
    x = torch.rand(2, MODEL.dsdi_features)
    out = dsdi_module(x)
    assert "index" in out
    assert out["index"].shape == (2, 1)
    assert (out["index"] >= 0.0).all() and (out["index"] <= 1.0).all()
    assert out["feature_weights"].shape == (2, MODEL.dsdi_features)


def test_adaptive_threshold():
    turbidity = np.random.rand(128, 128)
    fixed_thresh = np.quantile(turbidity, 0.85)
    adapt_thresh = adaptive_hotspot_threshold(turbidity, rainfall=0.9, flow=0.8)
    assert adapt_thresh >= 0.0 and adapt_thresh <= 1.0
    labels = generate_derived_labels(turbidity, rainfall=0.9, flow=0.8)
    assert "derived_adaptive_label" in labels
    assert labels["derived_adaptive_label"].shape == (128, 128)


def test_causal_attention_fusion():
    dim = MODEL.hidden_dim
    caf = CausalAttentionFusion(dim, modality_names=MODEL.modality_names)
    reps = {name: torch.randn(2, dim) for name in MODEL.modality_names}
    fused, weights = caf(reps)
    assert fused.shape == (2, dim)
    assert weights["final"].shape == (2, len(MODEL.modality_names))
    assert torch.allclose(weights["final"].sum(dim=-1), torch.tensor([1.0, 1.0]), atol=1e-5)


def test_gnn_graph_network():
    gnn = WatershedReservoirGNN(MODEL.graph_features, MODEL.hidden_dim, layers=2)
    nodes = torch.randn(2, 5, MODEL.graph_features)
    pooled, info = gnn(nodes)
    assert pooled.shape == (2, MODEL.hidden_dim)
    assert "node_embeddings" in info


def test_contrastive_alignment():
    dim = MODEL.hidden_dim
    mca = ModalContrastiveAlignment(dim, projection=32)
    reps = {"satellite": torch.randn(4, dim), "flow": torch.randn(4, dim), "rain": torch.randn(4, dim)}
    out = mca(reps)
    assert "loss" in out
    assert out["loss"] >= 0.0


def test_physics_loss_gradient():
    batch = {
        "dsdi_features": torch.rand(2, MODEL.dsdi_features, requires_grad=False),
    }
    pred_heatmap = torch.rand(2, 1, 128, 128, requires_grad=True)
    pred_dsdi = torch.rand(2, 1, requires_grad=True)
    prediction = {"heatmap": pred_heatmap, "dsdi": pred_dsdi}

    loss, comps = physics_informed_sediment_loss(prediction, batch)
    loss.backward()
    assert pred_heatmap.grad is not None
    assert loss.item() >= 0.0


def test_temporal_consistency_loss():
    curr = torch.rand(2, 1, 128, 128)
    prev = curr + 0.3
    loss, comps = temporal_consistency_loss(curr, prev, forcing_change=torch.zeros(2))
    assert loss.item() > 0.0


def test_end_to_end_fusion_net():
    model = SedimentHotspotFusionNet()
    batch = {
        "satellite": torch.rand(2, 6, 128, 128),
        "terrain": torch.rand(2, 4, 128, 128),
        "landuse": torch.rand(2, 5, 128, 128),
        "flow_sequence": torch.rand(2, 10, 8),
        "rain_features": torch.rand(2, 4),
        "dsdi_features": torch.rand(2, 11),
        "graph_nodes": torch.rand(2, 5, 8),
    }
    out = model(batch)
    assert out["heatmap"].shape == (2, 1, 128, 128)
    assert out["zone_risk"].shape == (2, MODEL.zones)
    assert out["dsdi"].shape == (2, 1)
    assert out["attention"].shape == (2, 7)


def test_uncertainty_mc_dropout():
    model = SedimentHotspotFusionNet()
    batch = {
        "satellite": torch.rand(1, 6, 128, 128),
        "terrain": torch.rand(1, 4, 128, 128),
        "landuse": torch.rand(1, 5, 128, 128),
        "flow_sequence": torch.rand(1, 10, 8),
        "rain_features": torch.rand(1, 4),
        "dsdi_features": torch.rand(1, 11),
        "graph_nodes": torch.rand(1, 5, 8),
    }
    res = mc_dropout_predict(model, batch, samples=5)
    assert res["mean"].shape == (1, 1, 128, 128)
    assert res["variance"].shape == (1, 1, 128, 128)
    cats = classify_uncertainty_regions(res["mean"].squeeze().numpy(), res["variance"].squeeze().numpy())
    assert "high_risk_high_confidence" in cats


def test_xai_gradcam():
    model = SedimentHotspotFusionNet()
    batch = {
        "satellite": torch.rand(1, 6, 128, 128),
        "terrain": torch.rand(1, 4, 128, 128),
        "landuse": torch.rand(1, 5, 128, 128),
        "flow_sequence": torch.rand(1, 10, 8),
        "rain_features": torch.rand(1, 4),
        "dsdi_features": torch.rand(1, 11),
        "graph_nodes": torch.rand(1, 5, 8),
    }
    cam = compute_satellite_gradcam(model, batch)
    assert cam.shape == (128, 128)
    assert (cam >= 0.0).all() and (cam <= 1.0).all()


if __name__ == "__main__":
    test_dsdi_computation()
    print("✓ test_dsdi_computation passed")
    test_adaptive_threshold()
    print("✓ test_adaptive_threshold passed")
    test_causal_attention_fusion()
    print("✓ test_causal_attention_fusion passed")
    test_gnn_graph_network()
    print("✓ test_gnn_graph_network passed")
    test_contrastive_alignment()
    print("✓ test_contrastive_alignment passed")
    test_physics_loss_gradient()
    print("✓ test_physics_loss_gradient passed")
    test_temporal_consistency_loss()
    print("✓ test_temporal_consistency_loss passed")
    test_end_to_end_fusion_net()
    print("✓ test_end_to_end_fusion_net passed")
    test_uncertainty_mc_dropout()
    print("✓ test_uncertainty_mc_dropout passed")
    test_xai_gradcam()
    print("✓ test_xai_gradcam passed")
    print("\n🎉 ALL 10 TESTS COMPLETED SUCCESSFULLY!")
