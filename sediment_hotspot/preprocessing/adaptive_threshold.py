import numpy as np

from sediment_hotspot.config import RASTER


def adaptive_hotspot_threshold(
    turbidity,
    rainfall: float = 0.0,
    flow: float = 0.0,
    season: float = 0.0,
    reservoir_condition: float = 0.5,
    historical_turbidity=None,
) -> float:
    current = np.asarray(turbidity, dtype=float)
    history = np.asarray(historical_turbidity if historical_turbidity is not None else current, dtype=float)
    finite_history = history[np.isfinite(history)]
    finite_current = current[np.isfinite(current)]
    if finite_history.size == 0 and finite_current.size == 0:
        return float(RASTER.adaptive_threshold_floor)

    reference = finite_history if finite_history.size else finite_current
    base = float(np.nanquantile(reference, RASTER.fixed_turbidity_quantile))
    history_shift = float(np.nanmean(reference) - base)

    # Dynamic hydrological factors:
    # High wetness (rainfall + flow) increases sediment influx and lowers the threshold
    # for designating an area as an active hotspot risk.
    wetness = 0.25 * np.clip(rainfall, 0, 1) + 0.20 * np.clip(flow, 0, 1)
    seasonal_term = 0.05 * np.sin(2 * np.pi * season)
    reservoir_term = 0.05 * (np.clip(reservoir_condition, 0, 1) - 0.5)
    anomaly_term = 0.05 * np.tanh(history_shift)

    factor = 1.0 - (wetness + seasonal_term + reservoir_term + anomaly_term) * 0.3
    raw_threshold = base * factor

    # Bounded between the 60th and 95th percentiles of empirical turbidity
    q_min = float(np.nanquantile(reference, 0.60))
    q_max = float(np.nanquantile(reference, 0.95))
    threshold = float(np.clip(raw_threshold, max(RASTER.adaptive_threshold_floor, q_min), min(RASTER.adaptive_threshold_ceiling, q_max)))
    return threshold


def generate_derived_labels(
    turbidity,
    rainfall: float = 0.0,
    flow: float = 0.0,
    season: float = 0.0,
    reservoir_condition: float = 0.5,
    historical_turbidity=None,
    observation_source: str = "satellite_turbidity",
) -> dict[str, object]:
    turbidity = np.asarray(turbidity, dtype=float)
    fixed_threshold = float(np.nanquantile(turbidity[np.isfinite(turbidity)], RASTER.fixed_turbidity_quantile))
    adaptive_threshold = adaptive_hotspot_threshold(
        turbidity,
        rainfall=rainfall,
        flow=flow,
        season=season,
        reservoir_condition=reservoir_condition,
        historical_turbidity=historical_turbidity,
    )
    fixed_label = (turbidity >= fixed_threshold).astype("uint8")
    adaptive_label = (turbidity >= adaptive_threshold).astype("uint8")
    return {
        "real_observation": turbidity,
        "real_observation_type": observation_source,
        "derived_fixed_label": fixed_label,
        "derived_adaptive_label": adaptive_label,
        "fixed_threshold": fixed_threshold,
        "adaptive_threshold": adaptive_threshold,
        "label_type": "derived_from_real_satellite_observations",
        "target_disclaimer": "Derived labels are not measured ground truth.",
    }
