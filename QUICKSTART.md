# 🚀 Quick Start Guide - Historical Data Pipeline

## The Problem We Solved

❌ **Old Approach**: Collect weather 50 times (same conditions) → Model overfits → Can't predict new scenarios

✅ **New Approach**: Use 6 years of historical data (500k+ records) → Robust training → Works for all conditions

---

## Installation (5 minutes)

```bash
# Clone/navigate to project
cd ~/reservoir

# Install dependencies
pip install -r requirements.txt
```

That's it! No API keys needed (Open-Meteo is free).

---

## Step-by-Step Execution

### Option 1: Run Everything (Recommended)
```bash
python pipeline.py --all
```

**What happens:**
1. Downloads 6 years of hourly weather data (2020-2025) for 14 zones
2. Creates lag features, rolling averages, time features
3. Trains XGBoost model
4. Saves trained model

**Time**: ~5-10 minutes (download) + 2 minutes (training)

---

### Option 2: Run Step-by-Step

```bash
# Step 1: Fetch historical weather data
python pipeline.py --fetch
# Output: historical_weather.csv (~500 MB, 500k+ rows)

# Step 2: Engineer features
python pipeline.py --engineer  
# Output: training_dataset.csv (~800 MB, with lag/rolling features)

# Step 3: Train model
python pipeline.py --train
# Output: rainfall_model.pkl (~10 MB, trained XGBoost)
```

---

## What You Get

After running the pipeline, you'll have:

| File | Size | Purpose |
|------|------|---------|
| `historical_weather.csv` | ~500 MB | Raw weather data (6 years, 14 zones, hourly) |
| `training_dataset.csv` | ~800 MB | Engineered features for ML |
| `rainfall_model.pkl` | ~10 MB | Trained XGBoost model |

---

## Make Real-Time Predictions

```bash
python river_weather_predictions.py
```

**Output**: Rainfall predictions for next hour across all zones

```
[PRED] Krishna   | K1_Mahabaleshwar     | Predicted:   2.34mm (Light)
[PRED] Krishna   | K2_Sangli_Almatti    | Predicted:  15.67mm (Heavy) ⚠️ HIGH RAINFALL EXPECTED
[PRED] Narmada   | N1_Amarkantak        | Predicted:   0.00mm (No Rain)
...
```

---

## Data Details

### Historical Data (2020-2025)

**Source**: Open-Meteo Archive API (free, no API key)

**Hourly Variables**:
- Temperature (°C)
- Humidity (%)
- Rainfall (mm)
- Pressure (hPa)
- Wind Speed (km/h)
- Cloud Cover (%)

**14 River Zones**:
- **Krishna**: 5 zones (Mahabaleshwar → Vijayawada Delta)
- **Narmada**: 4 zones (Amarkantak → Bharuch)
- **Kaveri**: 5 zones (Talakaveri → Thanjavur Delta)

### Engineered Features

```
Raw Variables (6) +
Lag Features (3×6 = 18) +
Rolling Averages (3) +
Change Features (4) +
Time Features (3)
= 34 features total
```

Example:
- `rainfall_lag_1`: Rainfall 1 hour ago
- `rainfall_avg_3`: Average rainfall over 3 hours
- `pressure_change`: How much pressure changed
- `hour`, `day`, `month`: Time-based features

### Target Variables

- **`rainfall_next_hour`**: Continuous value (0-999 mm)
- **`rainfall_class`**: 
  - 0 = No rain
  - 1 = Light (< 2.5 mm)
  - 2 = Moderate (2.5-10 mm)
  - 3 = Heavy (> 10 mm)

---

## Model Performance

**XGBoost Configuration**:
- 100 trees
- Max depth: 6
- Learning rate: 0.1

**Metrics** (calculated during training):
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R²: Coefficient of determination

See `train_model.py` output for exact metrics.

---

## Project Files Reference

```
reservoir/
├── fetch_historical_weather.py    ← Download historical data
├── feature_engineering.py         ← Create ML features
├── train_model.py                 ← Train XGBoost
├── river_weather_predictions.py   ← Make live predictions
├── pipeline.py                    ← Orchestrate entire workflow
│
├── historical_weather.csv         ← Generated (raw data)
├── training_dataset.csv           ← Generated (engineered)
├── rainfall_model.pkl             ← Generated (trained model)
│
├── requirements.txt               ← Python dependencies
├── README.md                      ← Full documentation
└── QUICKSTART.md                  ← This file
```

---

## Troubleshooting

### Q: "historical_weather.csv not found"
**A**: Run `python pipeline.py --fetch` first to download data

### Q: "rainfall_model.pkl not found"
**A**: Run `python pipeline.py --train` first to train the model

### Q: Script times out fetching data
**A**: Open-Meteo API has rate limits. The script waits 1 second between zone requests (should be fine). If still timing out:
- Edit `fetch_historical_weather.py` line 66: change `timeout=30` to `timeout=60`

### Q: Out of memory
**A**: If processing large CSVs:
```python
# Process in chunks instead
for chunk in pd.read_csv('historical_weather.csv', chunksize=50000):
    # process chunk
```

### Q: Model predictions don't look right
**A**: 
1. Check data quality: `head -5 historical_weather.csv`
2. Check feature engineering: `head -5 training_dataset.csv`
3. Look at model output from `python pipeline.py --train`
4. Compare predictions with current weather (should correlate)

---

## For Your Research Paper

**Key Points**:
- Historical data: 6 years (2020-2025)
- Dataset size: 500k+ observations
- Features: 34 engineered features + 6 raw variables
- Model: XGBoost (100 trees, depth=6)
- Zones: 14 across 3 major Indian rivers
- Prediction: Next-hour rainfall (hourly forecast)

**Methodology**:
```
Historical API (Free)
    ↓
Temporal Feature Engineering
    ↓
Time Series Validation
    ↓
XGBoost Regressor
    ↓
Real-time Predictions
```

---

## Next Steps (Advanced)

1. **Add LSTM**: Sequential models for better temporal patterns
2. **Ensemble Models**: Combine XGBoost + LSTM + Linear models
3. **Alert System**: Trigger alarms for high-risk rainfall
4. **Database Integration**: Store predictions for historical analysis
5. **Web Dashboard**: Visualize predictions across zones
6. **Mobile App**: Push notifications for flood risk

---

## Need Help?

- **Full Documentation**: See `README.md`
- **API Docs**: https://open-meteo.com/
- **XGBoost Tutorial**: https://xgboost.readthedocs.io/
- **Time Series ML**: https://scikit-learn.org/stable/modules/time_series.html

---

## Summary

```
1 command = entire pipeline:
$ python pipeline.py --all

Result: Trained rainfall forecasting model for 14 zones
Time: ~15 minutes total (mostly downloading)
Size: ~1.3 GB (historical data + engineered features)
Cost: $0 (free APIs)
```

✅ You're ready to make real-time predictions!
