import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, mean_absolute_error,
    mean_squared_error, r2_score
)
from scipy.ndimage import center_of_mass


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_centroid_displacement(y_true_map: np.ndarray, y_pred_map: np.ndarray) -> float:
    """Calculates spatial Euclidean displacement between ground-truth and predicted hotspot centroids."""
    t_mask = y_true_map >= 0.5
    p_mask = y_pred_map >= 0.5
    if not np.any(t_mask) or not np.any(p_mask):
        return 0.0
    cy_t, cx_t = center_of_mass(t_mask.astype(float))
    cy_p, cx_p = center_of_mass(p_mask.astype(float))
    return float(np.sqrt((cy_t - cy_p)**2 + (cx_t - cx_p)**2))


def evaluate_outputs(
    y_true,
    y_prob,
    zone_true=None,
    zone_pred=None,
    uncertainty=None,
    rainfall_true=None,
    rainfall_pred=None,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    b = (p >= 0.5).astype(int)
    y_bin = (y >= 0.5).astype(int)

    intersection = np.logical_and(y_bin, b).sum()
    union = np.logical_or(y_bin, b).sum()
    iou = float(intersection / max(union, 1))
    dice = float(2 * intersection / max(y_bin.sum() + b.sum(), 1))

    out = {
        "iou": iou,
        "dice": dice,
        "precision": float(precision_score(y_bin, b, zero_division=0)),
        "recall": float(recall_score(y_bin, b, zero_division=0)),
        "f1": float(f1_score(y_bin, b, zero_division=0)),
        "high_risk_detection_rate": float(recall_score(y_bin, b, zero_division=0)),
        "false_hotspot_rate": float(np.logical_and(~y_bin.astype(bool), b.astype(bool)).sum() / max((~y_bin.astype(bool)).sum(), 1)),
        "brier_score": float(brier_score_loss(y_bin, np.clip(p, 0.0, 1.0))),
        "ece": compute_expected_calibration_error(y_bin, p),
    }

    if len(np.unique(y_bin)) > 1:
        try:
            out["auc"] = float(roc_auc_score(y_bin, p))
            out["pr_auc"] = float(average_precision_score(y_bin, p))
        except Exception:
            out["auc"] = 0.5
            out["pr_auc"] = float(y_bin.mean())
    else:
        out["auc"] = 0.5
        out["pr_auc"] = float(y_bin.mean())

    if zone_true is not None and zone_pred is not None:
        zt = np.asarray(zone_true, dtype=float).ravel()
        zp = np.asarray(zone_pred, dtype=float).ravel()
        out["zone_mae"] = float(mean_absolute_error(zt, zp))
        out["zone_rmse"] = float(mean_squared_error(zt, zp) ** 0.5)
        out["zone_r2"] = float(r2_score(zt, zp)) if len(np.unique(zt)) > 1 else 1.0

    if rainfall_true is not None and rainfall_pred is not None:
        rt = np.asarray(rainfall_true, dtype=float).ravel()
        rp = np.asarray(rainfall_pred, dtype=float).ravel()
        out["rainfall_mae"] = float(mean_absolute_error(rt, rp))
        out["rainfall_rmse"] = float(mean_squared_error(rt, rp) ** 0.5)
        out["rainfall_r2"] = float(r2_score(rt, rp)) if len(np.unique(rt)) > 1 else 1.0

    if uncertainty is not None:
        u = np.asarray(uncertainty, dtype=float).ravel()
        err = np.abs(y - p)
        if len(u) > 1 and np.std(u) > 1e-8 and np.std(err) > 1e-8:
            out["uncertainty_error_correlation"] = float(np.corrcoef(err, u)[0, 1])
        else:
            out["uncertainty_error_correlation"] = 0.0

    # Centroid displacement if 2D or 3D
    yt_arr = np.asarray(y_true)
    yp_arr = np.asarray(y_prob)
    if yt_arr.ndim >= 2 and yp_arr.ndim >= 2:
        if yt_arr.ndim == 4:
            disps = [compute_centroid_displacement(yt_arr[i, 0], yp_arr[i, 0]) for i in range(yt_arr.shape[0])]
            out["centroid_displacement"] = float(np.mean(disps))
        elif yt_arr.ndim == 2:
            out["centroid_displacement"] = compute_centroid_displacement(yt_arr, yp_arr)

    return out
