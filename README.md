# Reservoir Rainfall and Sediment Hotspot Prediction

Machine learning pipeline for river-zone rainfall forecasting, with an optional sediment hotspot prediction module for reservoir monitoring.

The rainfall model is trained from historical Open-Meteo weather data instead of repeated live samples. The sediment module can then consume rainfall predictions alongside satellite imagery, flow, terrain, and land-use features.

## Features

- Downloads hourly historical weather data for 14 Indian river zones.
- Builds lag, rolling average, trend, and time-based rainfall features.
- Trains an XGBoost regressor for next-hour rainfall prediction.
- Produces live rainfall predictions from current Open-Meteo forecast data.
- Includes a separate PyTorch-based sediment hotspot package. See `SEDIMENT_HOTSPOT.md`.

## Project Layout

```text
reservoir/
├── fetch_historical_weather.py      # Download historical Open-Meteo data
├── feature_engineering.py           # Build ML training features
├── train_model.py                   # Train rainfall model
├── river_weather_predictions.py     # Run live rainfall prediction
├── pipeline.py                      # Orchestrate fetch, feature, train steps
├── sediment_pipeline.py             # Sediment workflow entry point
├── sediment_hotspot/                # Sediment hotspot package
├── scripts/                         # Preprocessing and training helpers
├── dashboards/                      # Streamlit dashboard
├── configs/                         # Example sediment config
└── requirements.txt
```

Generated data and model outputs are intentionally ignored by Git:

```text
historical_weather.csv
training_dataset.csv
rainfall_model.pkl
rainfall_predictions.csv
data/raw/
data/processed/
data/models/
data/outputs/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Open-Meteo does not require an API key.

## Rainfall Pipeline

Run the full training workflow:

```bash
python pipeline.py --all
```

Or run each stage separately:

```bash
python pipeline.py --fetch
python pipeline.py --engineer
python pipeline.py --train
```

Expected generated files:

```text
historical_weather.csv      # Raw hourly weather data
training_dataset.csv        # Engineered model features
rainfall_model.pkl          # Trained XGBoost model
```

After training, run live predictions:

```bash
python river_weather_predictions.py
```

This writes `rainfall_predictions.csv`, which can also be used as an input feature source for sediment hotspot prediction.

## River Zones

Krishna:

- `K1_Mahabaleshwar` - 17.9237, 73.6586
- `K2_Sangli_Almatti` - 16.8544, 74.5642
- `K3_Raichur_Kurnool` - 16.2076, 77.3463
- `K4_Nagarjuna_Sagar` - 16.5750, 79.3167
- `K5_Vijayawada_Delta` - 16.5062, 80.6480

Narmada:

- `N1_Amarkantak` - 22.6747, 81.7590
- `N2_Jabalpur` - 23.1815, 79.9864
- `N3_Omkareshwar` - 22.2452, 76.1510
- `N4_Bharuch` - 21.7051, 72.9959

Kaveri:

- `C1_Talakaveri` - 12.3855, 75.4894
- `C2_Kodagu` - 12.3375, 75.8069
- `C3_Mysuru_KRS` - 12.2958, 76.6394
- `C4_Mettur_Dam` - 11.7870, 77.8008
- `C5_Thanjavur_Delta` - 10.7867, 79.1378

## Model Inputs

Raw weather variables:

- `temperature`
- `humidity`
- `rainfall`
- `pressure`
- `wind_speed`
- `cloud_cover`

Engineered variables:

- 1, 2, and 3 hour lag features for each raw variable.
- 3 hour rolling averages for rainfall, humidity, and temperature.
- Pressure, humidity, temperature, and wind change features.
- Hour, day, and month time features.
- `rainfall_next_hour` regression target.
- `rainfall_class` category for no, light, moderate, and heavy rainfall.

## Sediment Hotspot Module

The sediment system combines rainfall predictions with:

- Sentinel-2 or Landsat imagery
- River flow and inflow sequences
- DEM terrain features
- Land-use features

See `SEDIMENT_HOTSPOT.md` for the public data sources, preprocessing commands, training manifest, inference workflow, and dashboard command.

## References

- Open-Meteo: https://open-meteo.com/
- XGBoost: https://xgboost.readthedocs.io/
- PyTorch: https://pytorch.org/
