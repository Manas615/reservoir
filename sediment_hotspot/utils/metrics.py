import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def iou_score(mask_true, mask_pred, threshold: float = 0.5) -> float:
    true = np.asarray(mask_true) > threshold
    pred = np.asarray(mask_pred) > threshold
    union = np.logical_or(true, pred).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(true, pred).sum() / union)


def dice_coefficient(mask_true, mask_pred, threshold: float = 0.5) -> float:
    true = np.asarray(mask_true) > threshold
    pred = np.asarray(mask_pred) > threshold
    denom = true.sum() + pred.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(true, pred).sum() / denom)
