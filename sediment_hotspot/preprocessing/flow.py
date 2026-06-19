from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sediment_hotspot.utils.io import read_table, write_table


FLOW_COLUMNS = ["river_discharge", "reservoir_inflow", "water_velocity"]


def build_flow_features(flow_csv: str | Path, output_path: str | Path = "data/processed/flow_features.csv") -> Path:
    df = read_table(flow_csv)
    missing = [col for col in ["timestamp", *FLOW_COLUMNS] if col not in df.columns]
    if missing:
        raise ValueError(f"Flow data is missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["month"] = df["timestamp"].dt.month
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    df["monsoon_phase"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    for col in FLOW_COLUMNS:
        df[f"{col}_roll7"] = df[col].rolling(7, min_periods=1).mean()
        df[f"{col}_lag1"] = df[col].shift(1).bfill()

    transport_capacity = (
        _norm(df["river_discharge"]) * 0.40
        + _norm(df["reservoir_inflow"]) * 0.35
        + _norm(df["water_velocity"]) * 0.25
    )
    df["sediment_transport_risk_score"] = transport_capacity
    df["flow_based_deposition_tendency"] = 1 - _norm(df["water_velocity"]) + _norm(df["reservoir_inflow"]) * 0.25
    df["flow_based_deposition_tendency"] = df["flow_based_deposition_tendency"].clip(0, 1)

    numeric = df.select_dtypes(include=[np.number]).columns
    scaled = StandardScaler().fit_transform(df[numeric])
    scaled_df = pd.DataFrame(scaled, columns=[f"scaled_{col}" for col in numeric])
    out = pd.concat([df.reset_index(drop=True), scaled_df], axis=1)
    return write_table(out, output_path)


def _norm(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min() + 1e-6)
