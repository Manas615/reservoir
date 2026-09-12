"""Dynamic Sediment Deposition Index (DSDI).

DSDI(x, y, t) combines normalized hydro-meteorological forcing, terrain
retention, erosion supply, and optical plume evidence into a dynamic prior.
The weighting is conditioned on the current environment instead of being fixed.
"""

import torch
from torch import nn

from sediment_hotspot.config import MODEL


DSDI_FEATURES = MODEL.dsdi_feature_names


class DynamicSedimentDepositionIndex(nn.Module):
    def __init__(self, features: int = MODEL.dsdi_features):
        super().__init__()
        self.features = features
        self.component_gate = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 5),
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(features + 5, 32),
            nn.ReLU(),
            nn.Linear(32, features),
        )

    def forward(self, dsdi_features: torch.Tensor, feature_map: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        feature_vector = self._normalize(dsdi_features)
        component_scores = self._component_scores(feature_vector)
        component_weights = torch.softmax(self.component_gate(component_scores), dim=-1)
        transformed = self._transform_raw_features(feature_vector)
        feature_context = torch.cat([transformed, component_scores], dim=-1)
        feature_weights = torch.softmax(self.feature_gate(feature_context), dim=-1)
        index = 0.65 * (component_scores * component_weights).sum(dim=-1, keepdim=True)
        index = index + 0.35 * (transformed * feature_weights).sum(dim=-1, keepdim=True)
        index = index.clamp(0, 1)

        dsdi_map = None
        if feature_map is not None:
            dsdi_map = self._map_forward(feature_map)
        return {
            "index": index,
            "map": dsdi_map,
            "component_scores": component_scores,
            "component_weights": component_weights,
            "feature_weights": feature_weights,
        }

    def _map_forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature_map.shape
        flat = feature_map.permute(0, 2, 3, 1).reshape(batch * height * width, channels)
        normalized = self._normalize(flat)
        component_scores = self._component_scores(normalized)
        component_weights = torch.softmax(self.component_gate(component_scores), dim=-1)
        transformed = self._transform_raw_features(normalized)
        feature_context = torch.cat([transformed, component_scores], dim=-1)
        feature_weights = torch.softmax(self.feature_gate(feature_context), dim=-1)
        dsdi = 0.65 * (component_scores * component_weights).sum(dim=-1, keepdim=True)
        dsdi = dsdi + 0.35 * (transformed * feature_weights).sum(dim=-1, keepdim=True)
        return dsdi.view(batch, height, width).unsqueeze(1).clamp(0, 1)

    def _component_scores(self, values: torch.Tensor) -> torch.Tensor:
        rainfall = 0.6 * values[:, 0] + 0.4 * values[:, 1]
        flow_arrival = 0.5 * values[:, 2] + 0.5 * values[:, 3]
        retention = ((1 - values[:, 4]) + (1 - values[:, 5]) + values[:, 6]) / 3.0
        sediment_supply = 0.55 * values[:, 7] + 0.45 * values[:, 8]
        optical_evidence = 0.5 * values[:, 9] + 0.5 * values[:, 10]
        return torch.stack(
            [rainfall, flow_arrival, retention, sediment_supply, optical_evidence],
            dim=-1,
        ).clamp(0, 1)

    def _transform_raw_features(self, values: torch.Tensor) -> torch.Tensor:
        transformed = values.clone()
        transformed[:, 4] = 1 - transformed[:, 4]
        transformed[:, 5] = 1 - transformed[:, 5]
        return transformed.clamp(0, 1)

    @staticmethod
    def _normalize(values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values
        return values.clamp(0, 1)


class DSDIFeatureEncoder(nn.Module):
    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features + 1 + 5, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

    def forward(self, dsdi_features: torch.Tensor, dsdi_prior: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(
            torch.cat(
                [
                    dsdi_features.clamp(0, 1),
                    dsdi_prior["index"],
                    dsdi_prior["component_scores"],
                ],
                dim=-1,
            )
        )
