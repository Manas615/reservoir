import torch
import torch.nn.functional as F

DSDI_INDEX = {
    "rainfall": 0,
    "rainfall_intensity": 1,
    "discharge": 2,
    "inflow": 3,
    "velocity": 4,
    "slope": 5,
    "curvature": 6,
    "erosion": 7,
    "landuse": 8,
    "turbidity": 9,
    "plume": 10,
}

def physics_informed_sediment_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_score = prediction["heatmap"].mean(dim=(-1, -2, -3))
    dsdi_prior = prediction["dsdi"].squeeze(-1) if prediction["dsdi"].ndim > 1 else prediction["dsdi"]
    features = batch["dsdi_features"].clamp(0, 1)

    rainfall = features[:, DSDI_INDEX["rainfall"]]
    intensity = features[:, DSDI_INDEX["rainfall_intensity"]]
    discharge = features[:, DSDI_INDEX["discharge"]]
    inflow = features[:, DSDI_INDEX["inflow"]]
    velocity = features[:, DSDI_INDEX["velocity"]]
    slope = features[:, DSDI_INDEX["slope"]]
    curvature = features[:, DSDI_INDEX["curvature"]]
    erosion = features[:, DSDI_INDEX["erosion"]]
    landuse = features[:, DSDI_INDEX["landuse"]]
    turbidity = features[:, DSDI_INDEX["turbidity"]]
    plume = features[:, DSDI_INDEX["plume"]]

    erosion_supply = 0.6 * erosion + 0.4 * landuse
    transport_potential = 0.5 * (rainfall * inflow + intensity * discharge)
    retention = ((1 - velocity) + (1 - slope) + curvature) / 3.0
    optical_signal = 0.5 * (turbidity + plume)
    capacity_drop = torch.relu(transport_potential - velocity)
    expected = torch.clamp(
        0.28 * erosion_supply
        + 0.22 * transport_potential
        + 0.20 * retention
        + 0.15 * optical_signal
        + 0.15 * capacity_drop,
        0,
        1,
    )

    components = {
        "velocity": (pred_score * velocity).mean(),
        "erosion": torch.relu(erosion_supply - pred_score).mean(),
        "transport": torch.relu(transport_potential - pred_score).mean(),
        "deposition_drop": torch.relu(capacity_drop - pred_score).mean(),
        "prior_alignment": F.mse_loss(pred_score, torch.maximum(expected, dsdi_prior.detach())),
    }
    total = torch.stack(list(components.values())).mean()
    return total, components
