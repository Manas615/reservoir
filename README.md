# River Rainfall Forecasting - Historical Data Pipeline

## Overview

This project implements a **production-grade rainfall forecasting system** using historical weather data and machine learning.

It now also includes a separate **AI sediment hotspot prediction system** that consumes the completed rainfall forecast output and combines Sentinel/Landsat imagery, flow data, DEM terrain features, and land-use features. Rainfall forecasting is not rebuilt by the sediment modules. See `SEDIMENT_HOTSPOT.md`.

**Key Principle**: We use **2020-2025 historical weather data** to train the model, not repetitive real-time collection.

---

## Project Structure

```
Historical Weather Data (Open-Meteo API, 2020-2025)
        ↓
fetch_historical_weather.py (→ historical_weather.csv: 500k+ rows)
        ↓
feature_engineering.py (Lag features, rolling averages, trends)
        ↓ (→ training_dataset.csv: Engineered features)
train_model.py (XGBoost classifier)
        ↓ (→ rainfall_model.pkl)
river_weather.py (Real-time predictions only)
```

---

## River Zones Covered

### Krishna (5 zones)
- K1_Mahabaleshwar (17.9237, 73.6586)
- K2_Sangli_Almatti (16.8544, 74.5642)
- K3_Raichur_Kurnool (16.2076, 77.3463)
- K4_Nagarjuna_Sagar (16.5750, 79.3167)
- K5_Vijayawada_Delta (16.5062, 80.6480)

### Narmada (4 zones)
- N1_Amarkantak (22.6747, 81.7590)
- N2_Jabalpur (23.1815, 79.9864)
- N3_Omkareshwar (22.2452, 76.1510)
- N4_Bharuch (21.7051, 72.9959)

### Kaveri (5 zones)
- C1_Talakaveri (12.3855, 75.4894)
- C2_Kodagu (12.3375, 75.8069)
- C3_Mysuru_KRS (12.2958, 76.6394)
- C4_Mettur_Dam (11.7870, 77.8008)
- C5_Thanjavur_Delta (10.7867, 79.1378)

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
# Run all steps (fetch → engineer → train)
python pipeline.py --all

# Or run individual steps:
python pipeline.py --fetch      # Fetch historical data
python pipeline.py --engineer   # Create features
python pipeline.py --train      # Train model
```

### Expected Output

After running the complete pipeline:

```
historical_weather.csv      (~500 MB)  [14 zones × 6 years hourly data]
training_dataset.csv        (~800 MB)  [Engineered features]
rainfall_model.pkl          (~10 MB)   [Trained XGBoost model]
```

---

## Data Features

### Raw Weather Variables (hourly)
- `temperature` (°C)
- `humidity` (%)
- `rainfall` (mm)
- `pressure` (hPa)
- `wind_speed` (km/h)
- `cloud_cover` (%)

### Engineered Features
- **Lag features**: rainfall_lag_1/2/3, temperature_lag_1/2/3, etc.
- **Rolling averages**: rainfall_avg_3, humidity_avg_3, temperature_avg_3
- **Change features**: pressure_change, humidity_change, temperature_change, wind_change
- **Time features**: hour, day, month
- **Target variable**: rainfall_next_hour (0-999 mm)
- **Classification**: rainfall_class (0=no rain, 1=light, 2=moderate, 3=heavy)

---

## Model Information

### Algorithm
- **XGBoost Regressor** for continuous rainfall prediction
- Trained on ~500k historical records
- Cross-validated on temporal splits

### Performance Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Hyperparameters
```python
XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

---

## Real-Time Predictions

Once the model is trained, use `river_weather.py` for live predictions:

```python
import joblib
import requests
import pandas as pd

# Load trained model
model = joblib.load('rainfall_model.pkl')

# Fetch current weather (Open-Meteo current forecast)
# Process with same features as training
# Predict: model.predict(features)
```

The `river_weather.py` script handles:
1. Fetching current weather from Open-Meteo
2. Creating lag features from historical context
3. Making predictions
4. Alerting for high rainfall probability

---

## API Sources

### Historical Data (Training)
- **Open-Meteo Archive API**: https://archive-api.open-meteo.com/v1/archive
- Free, no API key required
- Hourly historical data (1940 onwards)

### Real-Time Data (Predictions)
- **Open-Meteo Forecast API**: https://api.open-meteo.com/v1/forecast
- Free, no API key required
- Current weather + 7-day forecast

---

## Why Historical Data First?

### Problem with Real-Time Collection
```
Collecting same weather repeatedly
→ Rainfall: 0.0, 0.0, 0.0, ...
→ Temperature: 28.5, 28.6, 28.4, ...
→ Model sees same patterns repeatedly
→ OVERFITS on current weather
→ Cannot predict different conditions
```

### Solution: Historical Data (2020-2025)
```
500k+ observations across 6 years
→ Diverse rainfall patterns (monsoon, dry season, etc.)
→ Wide temperature/humidity/pressure ranges
→ Different seasonal behaviors per zone
→ Model learns generalizable patterns
→ Works for future unseen weather conditions
```

---

## Production Workflow

```
Week 1: Build & Train Model
  ├─ Fetch historical data (2020-2025)
  ├─ Feature engineering
  ├─ Train XGBoost
  └─ Validate on test set

Week 2 onwards: Live Predictions
  ├─ Use trained model daily
  ├─ No more training data collection
  ├─ Quarterly retraining (add new year's data)
  └─ Monitor performance metrics
```

---

## Troubleshooting

### fetch_historical_weather.py timeout
```bash
# Increase timeout in fetch_historical_weather.py
response = requests.get(url, timeout=60)  # Default: 30s
```

### Memory issues with large CSV
```bash
# Process in chunks if needed:
for chunk in pd.read_csv('historical_weather.csv', chunksize=10000):
    # Process chunk
    pass
```

### Model not performing well
1. Check data quality: `historical_weather.csv` shape (should be ~500k rows)
2. Verify features: `training_dataset.csv` has no NaN values
3. Review feature importance from trained model
4. Consider seasonal adjustments per river zone

---

## References

- **Open-Meteo**: https://open-meteo.com/
- **XGBoost**: https://xgboost.readthedocs.io/
- **Time Series Forecasting**: Feature engineering from temporal data

---

## Author Notes

This is a **production-ready approach** for river monitoring systems:
- Uses free APIs (no cost)
- 500k+ data points for robust training
- Extensible to other rivers/zones
- Can integrate with early warning systems

**Next improvements**:
- Add LSTM for sequential patterns
- Include satellite rainfall estimates
- Integrate ensemble models
- Real-time alert system
