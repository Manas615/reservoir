from pathlib import Path

import numpy as np
import pandas as pd

from sediment_hotspot.utils.io import ensure_dir, require_file


def rainfall_prediction_features(
    rainfall_predictions_csv: str | Path,
    output_dir: str | Path = "data/processed/rainfall",
) -> Path:
    """Convert existing rainfall forecast output into fusion-model features."""
    df = pd.read_csv(require_file(rainfall_predictions_csv, "rainfall prediction output"))
    candidates = [
        "predicted_rainfall_next_hour",
        "current_rainfall",
        "current_humidity",
        "current_temp",
    ]
    missing = [col for col in candidates if col not in df.columns]
    if missing:
        raise ValueError(f"Rainfall prediction output is missing columns: {missing}")

    latest = df.tail(min(len(df), 24))
    features = np.array(
        [
            latest["predicted_rainfall_next_hour"].mean(),
            latest["predicted_rainfall_next_hour"].max(),
            latest["current_rainfall"].fillna(0).mean(),
            latest["current_humidity"].fillna(0).mean(),
        ],
        dtype="float32",
    )
    output_path = ensure_dir(output_dir) / "rain_features.npy"
    np.save(output_path, features)
    return output_path
