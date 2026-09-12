import numpy as np
import pandas as pd


def compute_spatial_sensitivity(
    baseline_map: np.ndarray,
    counterfactual_map: np.ndarray,
    magnitude: float,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Computes pixel-level spatial sensitivity: S_i = |Delta H_i| / (|magnitude| + epsilon)."""
    base = np.asarray(baseline_map, dtype=float)
    cf = np.asarray(counterfactual_map, dtype=float)
    delta = np.abs(cf - base)
    if abs(magnitude) < 1e-6:
        return np.zeros_like(delta, dtype="float32")
    sensitivity = delta / (abs(magnitude) + epsilon)
    return sensitivity.astype("float32")


def zone_sensitivity_analysis(
    baseline_zones: np.ndarray,
    counterfactual_zones: np.ndarray,
    sensitivity_map: np.ndarray,
    magnitude: float,
) -> pd.DataFrame:
    """Calculates zone-wise vulnerability and sensitivity rankings."""
    bz = np.asarray(baseline_zones, dtype=float).ravel()
    cfz = np.asarray(counterfactual_zones, dtype=float).ravel()
    num_zones = len(bz)
    delta_z = cfz - bz

    # Partition sensitivity map across zones (assuming horizontal strip zones)
    h = sensitivity_map.shape[0]
    strip_h = h // num_zones
    zone_mean_sens = []
    for i in range(num_zones):
        z_sens = sensitivity_map[i * strip_h : (i + 1) * strip_h, :].mean()
        zone_mean_sens.append(float(z_sens))

    df = pd.DataFrame({
        "zone": [f"Zone {chr(65 + i)}" for i in range(num_zones)],
        "baseline_risk": bz,
        "counterfactual_risk": cfz,
        "risk_change": delta_z,
        "sensitivity": zone_mean_sens,
    })

    q75 = df["sensitivity"].quantile(0.75)
    q25 = df["sensitivity"].quantile(0.25)

    def categorize(s):
        if s >= q75:
            return "Highly Sensitive"
        elif s >= q25:
            return "Moderately Sensitive"
        else:
            return "Stable"

    df["vulnerability_category"] = df["sensitivity"].apply(categorize)
    df = df.sort_values("sensitivity", ascending=False).reset_index(drop=True)
    df["sensitivity_rank"] = df.index + 1
    return df
