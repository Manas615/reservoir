from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
import torch

from sediment_hotspot.counterfactual.interventions import apply_intervention, InterventionType
from sediment_hotspot.counterfactual.sensitivity import compute_spatial_sensitivity, zone_sensitivity_analysis
from sediment_hotspot.models.uncertainty import mc_dropout_predict, classify_uncertainty_regions


@dataclass
class CounterfactualPrediction:
    baseline_heatmap: np.ndarray
    counterfactual_heatmap: np.ndarray
    difference_heatmap: np.ndarray
    sensitivity_heatmap: np.ndarray
    baseline_uncertainty: np.ndarray
    counterfactual_uncertainty: np.ndarray
    uncertainty_difference: np.ndarray
    zone_analysis: pd.DataFrame
    metadata: dict[str, object]
    baseline_categories: dict[str, np.ndarray]
    counterfactual_categories: dict[str, np.ndarray]


def predict_counterfactual(
    model,
    batch: dict[str, torch.Tensor],
    intervention_type: InterventionType = "rainfall",
    magnitude: float = 0.20,
    mc_samples: int = 20,
) -> CounterfactualPrediction:
    """Executes a full CSDN counterfactual pass with uncertainty and sensitivity."""
    model.eval()

    # 1. Baseline prediction + MC Dropout uncertainty
    base_unc = mc_dropout_predict(model, batch, samples=mc_samples)
    base_heat = base_unc["mean"].squeeze().cpu().numpy()
    base_var = base_unc["variance"].squeeze().cpu().numpy()

    with torch.no_grad():
        base_out = model(batch)
        base_zones = base_out["zone_risk"].squeeze().cpu().numpy()

    # 2. Counterfactual transformation & prediction
    cf_batch, meta = apply_intervention(batch, intervention_type, magnitude)
    cf_unc = mc_dropout_predict(model, cf_batch, samples=mc_samples)
    cf_heat = cf_unc["mean"].squeeze().cpu().numpy()
    cf_var = cf_unc["variance"].squeeze().cpu().numpy()

    with torch.no_grad():
        cf_out = model(cf_batch)
        cf_zones = cf_out["zone_risk"].squeeze().cpu().numpy()

    # 3. Differences & Sensitivity
    diff_heat = (cf_heat - base_heat).astype("float32")
    diff_unc = (cf_var - base_var).astype("float32")
    sens_heat = compute_spatial_sensitivity(base_heat, cf_heat, magnitude)

    # 4. Zone sensitivity ranking
    zone_df = zone_sensitivity_analysis(base_zones, cf_zones, sens_heat, magnitude)

    # 5. 4-tier risk-confidence categorization
    base_cats = classify_uncertainty_regions(base_heat, base_var)
    cf_cats = classify_uncertainty_regions(cf_heat, cf_var)

    meta.update({
        "mean_baseline_risk": float(base_heat.mean()),
        "mean_counterfactual_risk": float(cf_heat.mean()),
        "mean_risk_change": float(diff_heat.mean()),
        "max_risk_increase": float(diff_heat.max()),
        "max_risk_decrease": float(diff_heat.min()),
        "global_mean_sensitivity": float(sens_heat.mean()),
        "top_vulnerable_zone": str(zone_df.iloc[0]["zone"]),
    })

    return CounterfactualPrediction(
        baseline_heatmap=base_heat,
        counterfactual_heatmap=cf_heat,
        difference_heatmap=diff_heat,
        sensitivity_heatmap=sens_heat,
        baseline_uncertainty=base_var,
        counterfactual_uncertainty=cf_var,
        uncertainty_difference=diff_unc,
        zone_analysis=zone_df,
        metadata=meta,
        baseline_categories=base_cats,
        counterfactual_categories=cf_cats,
    )
