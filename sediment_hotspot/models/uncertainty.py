import numpy as np
import torch


def enable_mc_dropout(model) -> None:
    model.eval()
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def mc_dropout_predict(model, batch: dict[str, torch.Tensor], samples: int = 20) -> dict[str, torch.Tensor]:
    enable_mc_dropout(model)
    predictions = []
    with torch.no_grad():
        for _ in range(samples):
            predictions.append(model(batch)["heatmap"])
    stacked = torch.stack(predictions, dim=0)
    mean = stacked.mean(dim=0)
    variance = stacked.var(dim=0, unbiased=False)
    entropy = -(mean.clamp(1e-6, 1 - 1e-6) * torch.log(mean.clamp(1e-6, 1 - 1e-6)) + (1 - mean).clamp(1e-6, 1) * torch.log((1 - mean).clamp(1e-6, 1)))
    return {"mean": mean, "variance": variance, "entropy": entropy, "samples": stacked}


def classify_uncertainty_regions(risk_map, uncertainty_map, risk_threshold: float = 0.5, uncertainty_quantile: float = 0.75) -> dict[str, np.ndarray]:
    risk = np.asarray(risk_map)
    uncertainty = np.asarray(uncertainty_map)
    cutoff = float(np.quantile(uncertainty, uncertainty_quantile)) if uncertainty.size else 0.0
    high_risk = risk >= risk_threshold
    high_uncertainty = uncertainty >= cutoff
    categories = np.zeros_like(risk, dtype=np.uint8)
    categories[np.logical_and(high_risk, ~high_uncertainty)] = 1
    categories[np.logical_and(high_risk, high_uncertainty)] = 2
    return {
        "categories": categories,
        "high_risk_high_confidence": np.logical_and(high_risk, ~high_uncertainty),
        "high_risk_low_confidence": np.logical_and(high_risk, high_uncertainty),
        "low_risk": ~high_risk,
    }
