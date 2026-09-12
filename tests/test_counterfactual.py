import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from sediment_hotspot.config import MODEL, SEDIMENT
from sediment_hotspot.counterfactual.interventions import apply_intervention
from sediment_hotspot.counterfactual.sensitivity import compute_spatial_sensitivity, zone_sensitivity_analysis
from sediment_hotspot.counterfactual.counterfactual import predict_counterfactual
from sediment_hotspot.counterfactual.scenario_engine import run_all_scenarios
from sediment_hotspot.losses.counterfactual_loss import CounterfactualConsistencyLoss
from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
from sediment_hotspot.evaluate import compute_expected_calibration_error, compute_centroid_displacement


def test_intervention_preserves_static_modalities():
    batch = {
        "satellite": torch.rand(2, 6, 128, 128),
        "terrain": torch.rand(2, 4, 128, 128),
        "landuse": torch.rand(2, 5, 128, 128),
        "flow_sequence": torch.rand(2, 10, 8),
        "rain_features": torch.rand(2, 4),
        "dsdi_features": torch.rand(2, 11),
        "graph_nodes": torch.rand(2, 5, 8),
    }
    cf_batch, meta = apply_intervention(batch, "rainfall", 0.20)
    assert torch.allclose(cf_batch["satellite"], batch["satellite"])
    assert torch.allclose(cf_batch["terrain"], batch["terrain"])
    assert torch.allclose(cf_batch["landuse"], batch["landuse"])
    assert torch.allclose(cf_batch["graph_nodes"], batch["graph_nodes"])
    assert not torch.allclose(cf_batch["rain_features"], batch["rain_features"])


def test_sensitivity_computation():
    base = np.zeros((128, 128), dtype="float32")
    cf = np.ones((128, 128), dtype="float32") * 0.20
    sens = compute_spatial_sensitivity(base, cf, magnitude=0.20)
    assert sens.shape == (128, 128)
    assert np.allclose(sens, 1.0, atol=1e-2)

    bz = np.array([0.2] * 8)
    cfz = np.array([0.4] * 8)
    df = zone_sensitivity_analysis(bz, cfz, sens, magnitude=0.20)
    assert len(df) == 8
    assert "vulnerability_category" in df.columns


def test_counterfactual_loss_gradient():
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
    base_out = model(batch)
    loss_fn = CounterfactualConsistencyLoss(tolerance=0.08)
    loss, comps = loss_fn(model, batch, base_out)
    assert loss.item() >= 0.0
    loss.backward()
    assert model.satellite_cnn.encoder[0].weight.grad is not None


def test_predict_counterfactual_end_to_end():
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
    res = predict_counterfactual(model, batch, intervention_type="rainfall", magnitude=0.20, mc_samples=3)
    assert res.baseline_heatmap.shape == (128, 128)
    assert res.counterfactual_heatmap.shape == (128, 128)
    assert res.difference_heatmap.shape == (128, 128)
    assert res.sensitivity_heatmap.shape == (128, 128)
    assert len(res.zone_analysis) == 8


def test_ece_and_spatial_metrics():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.65, 0.2, 0.3, 0.85, 0.15])
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0

    m1 = np.zeros((128, 128))
    m1[10:20, 10:20] = 1.0
    m2 = np.zeros((128, 128))
    m2[10:20, 10:20] = 1.0
    disp = compute_centroid_displacement(m1, m2)
    assert disp == 0.0


if __name__ == "__main__":
    test_intervention_preserves_static_modalities()
    print("✓ test_intervention_preserves_static_modalities passed")
    test_sensitivity_computation()
    print("✓ test_sensitivity_computation passed")
    test_counterfactual_loss_gradient()
    print("✓ test_counterfactual_loss_gradient passed")
    test_predict_counterfactual_end_to_end()
    print("✓ test_predict_counterfactual_end_to_end passed")
    test_ece_and_spatial_metrics()
    print("✓ test_ece_and_spatial_metrics passed")
    print("\n🎉 ALL CSDN UNIT TESTS COMPLETED SUCCESSFULLY!")
