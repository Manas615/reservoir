"""Multi-Modal Contrastive Alignment (MMCA)."""

import torch
from torch import nn
import torch.nn.functional as F


class ModalContrastiveAlignment(nn.Module):
    def __init__(self, hidden: int, projection: int = 64, temperature: float = 0.15):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, projection),
        )
        self.temperature = temperature

    def forward(self, representations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if len(representations) < 2:
            zero = next(iter(representations.values())).sum() * 0 if representations else torch.tensor(0.0)
            return {"loss": zero, "embeddings": {}}

        projected = {
            name: F.normalize(self.head(value), dim=-1)
            for name, value in representations.items()
        }
        batch_size = next(iter(projected.values())).shape[0]
        if batch_size < 2:
            zero = next(iter(projected.values())).sum() * 0
            return {"loss": zero, "embeddings": projected}

        losses = []
        names = list(projected)
        targets = torch.arange(batch_size, device=next(iter(projected.values())).device)
        for idx, anchor_name in enumerate(names):
            for positive_name in names[idx + 1 :]:
                anchor = projected[anchor_name]
                positive = projected[positive_name]
                logits_ab = anchor @ positive.T / self.temperature
                logits_ba = positive @ anchor.T / self.temperature
                losses.append(F.cross_entropy(logits_ab, targets))
                losses.append(F.cross_entropy(logits_ba, targets))
        return {"loss": torch.stack(losses).mean(), "embeddings": projected}
