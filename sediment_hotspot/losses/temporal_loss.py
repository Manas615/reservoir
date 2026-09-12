import torch

def temporal_consistency_loss(
    current: torch.Tensor,
    previous: torch.Tensor | None,
    forcing_change: torch.Tensor | None = None,
    tolerance: float = 0.15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if previous is None or (isinstance(previous, torch.Tensor) and previous.sum() == 0):
        zero = current.sum() * 0.0
        return zero, {"temporal_error": zero, "forcing_support": zero}

    delta = (current - previous).abs()
    if forcing_change is None:
        forcing_change = torch.zeros(current.shape[0], device=current.device)
    support = forcing_change.reshape(-1, 1, 1, 1).clamp(0, 1)
    allowance = tolerance + support
    penalty = torch.relu(delta - allowance).mean()
    return penalty, {
        "temporal_error": delta.mean(),
        "forcing_support": support.mean(),
    }
