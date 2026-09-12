"""Causal Attention Fusion (CAF).

Modalities are fused using three signals:
1. learned attention from latent representations,
2. causal influence scores from Rainfall -> Flow -> Erosion -> Transport -> Deposition,
3. DSDI-derived prior compatibility.
"""

import math

import torch
from torch import nn

from sediment_hotspot.config import MODEL


class CausalAttentionFusion(nn.Module):
    def __init__(self, dim: int, modality_names: tuple[str, ...] = MODEL.modality_names, temperature: float = 1.0):
        super().__init__()
        self.modality_names = modality_names
        self.temperature = temperature
        self.learned_score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )
        self.causal_energy = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )
        self.prior_projection = nn.Sequential(
            nn.Linear(6, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, len(modality_names)),
        )
        self.register_buffer("causal_matrix", self._build_causal_matrix(modality_names))

    def forward(self, representations: dict[str, torch.Tensor], dsdi_prior: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        names = list(representations)
        states = torch.stack([representations[name] for name in names], dim=1)
        learned_logits = self.learned_score(states).squeeze(-1)
        learned = torch.softmax(learned_logits / self.temperature, dim=1)

        causal_energy = torch.sigmoid(self.causal_energy(states)).squeeze(-1)
        causal_matrix = self._select_submatrix(names)
        causal_logits = causal_energy @ causal_matrix
        causal = torch.softmax(causal_logits / self.temperature, dim=1)

        prior_logits = torch.zeros_like(learned_logits)
        if dsdi_prior is not None:
            prior_signal = torch.cat([dsdi_prior["index"], dsdi_prior["component_scores"]], dim=-1)
            projected = self.prior_projection(prior_signal)
            keep = [self.modality_names.index(name) for name in names]
            prior_logits = projected[:, keep]

        final_logits = learned_logits + causal_logits + prior_logits
        final = torch.softmax(final_logits / self.temperature, dim=1)
        fused = (states * final.unsqueeze(-1)).sum(dim=1)
        return fused, {
            "learned": learned,
            "causal": causal,
            "prior": torch.softmax(prior_logits / self.temperature, dim=1),
            "final": final,
            "modality_order": names,
        }

    def _select_submatrix(self, names: list[str]) -> torch.Tensor:
        indices = [self.modality_names.index(name) for name in names]
        return self.causal_matrix[indices][:, indices]

    @staticmethod
    def _build_causal_matrix(names: tuple[str, ...]) -> torch.Tensor:
        base = torch.eye(len(names), dtype=torch.float32)
        edges = {
            ("rain", "flow"): 1.0,
            ("rain", "landuse"): 0.4,
            ("rain", "dsdi"): 0.8,
            ("terrain", "landuse"): 0.6,
            ("terrain", "dsdi"): 0.7,
            ("landuse", "dsdi"): 0.9,
            ("flow", "graph"): 0.8,
            ("flow", "satellite"): 0.7,
            ("flow", "dsdi"): 0.9,
            ("graph", "satellite"): 0.6,
            ("graph", "dsdi"): 0.8,
            ("satellite", "dsdi"): 0.7,
        }
        for (src, dst), weight in edges.items():
            if src in names and dst in names:
                i = names.index(src)
                j = names.index(dst)
                base[i, j] = weight
                base[j, j] = max(base[j, j], weight / math.sqrt(2.0))
        return base
