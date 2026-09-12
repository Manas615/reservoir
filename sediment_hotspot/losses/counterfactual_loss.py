import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from sediment_hotspot.counterfactual.interventions import apply_intervention


class CounterfactualConsistencyLoss(nn.Module):
    """Physics-informed counterfactual consistency loss.
    Enforces directional plausibility under environmental perturbations during training.
    """

    def __init__(self, tolerance: float = 0.08):
        super().__init__()
        self.tolerance = tolerance

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        base_prediction: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Randomly sample intervention type and magnitude during training pass
        intervention_type = random.choice(["rainfall", "flow", "inflow"])
        magnitude = random.choice([-0.25, -0.15, +0.15, +0.25])

        cf_batch, _ = apply_intervention(batch, intervention_type, magnitude)
        cf_pred = model(cf_batch)

        base_heat = base_prediction["heatmap"]
        cf_heat = cf_pred["heatmap"]
        delta_heat = cf_heat - base_heat

        features = batch.get("dsdi_features", torch.zeros((base_heat.shape[0], 11), device=base_heat.device))
        erosion_vulnerability = (features[:, 7] * 0.6 + features[:, 8] * 0.4).view(-1, 1, 1, 1)
        velocity = features[:, 4].view(-1, 1, 1, 1)
        retention_propensity = ((1.0 - velocity) + (1.0 - features[:, 5].view(-1, 1, 1, 1))) / 2.0

        penalties = []

        if intervention_type in ("rainfall", "inflow") and magnitude > 0:
            # Positive rainfall/inflow should not cause an unphysical drop in hotspot risk in erosion-prone areas
            unphysical_drop = F.relu(-delta_heat - self.tolerance) * erosion_vulnerability
            penalties.append(unphysical_drop.mean())
        elif intervention_type in ("rainfall", "inflow") and magnitude < 0:
            # Negative rainfall should generally reduce sediment supply/hotspot risk
            unphysical_spike = F.relu(delta_heat - self.tolerance) * erosion_vulnerability
            penalties.append(unphysical_spike.mean())
        elif intervention_type == "flow" and magnitude < 0:
            # Reduced flow/velocity should increase deposition in high-retention settling zones
            unphysical_clearing = F.relu(-delta_heat - self.tolerance) * retention_propensity
            penalties.append(unphysical_clearing.mean())

        # Spatial smoothness regularization on counterfactual differences
        diff_grad_y = torch.abs(delta_heat[:, :, 1:, :] - delta_heat[:, :, :-1, :])
        diff_grad_x = torch.abs(delta_heat[:, :, :, 1:] - delta_heat[:, :, :, :-1])
        smoothness = 0.5 * (diff_grad_y.mean() + diff_grad_x.mean())
        penalties.append(0.05 * smoothness)

        total_loss = torch.stack(penalties).sum() if penalties else delta_heat.sum() * 0.0
        return total_loss, {
            "cf_total": total_loss.detach(),
            "cf_delta_mean": delta_heat.mean().detach(),
            "cf_smoothness": smoothness.detach(),
        }
