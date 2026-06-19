# AI Sediment Hotspot Prediction System

This adds the non-rainfall modules for predicting future sediment deposition hotspots inside reservoirs. The existing rainfall forecasting code remains separate; its `rainfall_predictions.csv` output is consumed as a feature input.

## Modules

- Satellite imagery: Sentinel-2 or Landsat GeoTIFF preprocessing, cloud masking, water extraction, turbidity index, plume masks, shoreline-change area.
- River flow: discharge, inflow, velocity, seasonal rolling/lag features, transport risk, deposition tendency.
- DEM and terrain: SRTM or ASTER DEM slope, elevation gradient, curvature, erosion susceptibility.
- Land use: ESA WorldCover or remapped MODIS classes, erosion-prone source probability, temporal change.
- Multi-modal model: PyTorch CNN encoders, LSTM flow encoder, rainfall feature encoder, attention fusion, hotspot heatmap and zone-risk heads.
- Explainability: attention summaries, SHAP integration note, Grad-CAM utility hook.
- Dashboard: Streamlit heatmap, risk timeline, zone-wise predictions, attention visualization.

## Public Data Inputs

Use real public datasets only:

- Sentinel-2 SR: `COPERNICUS/S2_SR_HARMONIZED`
- Landsat Collection 2 L2: `LANDSAT/LC08/C02/T1_L2`
- SRTM DEM: `USGS/SRTMGL1_003`
- ASTER DEM: `NASA/ASTER_GED/AG100_003`
- ESA WorldCover: `ESA/WorldCover/v200`
- MODIS Land Cover: `MODIS/061/MCD12Q1`

Create an acquisition manifest:

```bash
python scripts/download_public_data_gee.py \
  --reservoir krs \
  --aoi-geojson data/raw/aoi/krs.geojson \
  --start-date 2021-01-01 \
  --end-date 2025-12-31
```

## Preprocessing

```bash
python scripts/preprocess_satellite.py --image data/raw/satellite/sentinel2_krs_2025_09.tif --qa-band 6
python scripts/preprocess_terrain.py --dem data/raw/dem/srtm_krs.tif
python scripts/preprocess_landuse.py --landcover data/raw/landuse/esa_worldcover_krs.tif
python scripts/preprocess_flow.py --flow-csv data/raw/flow/krs_flow.csv
python scripts/preprocess_rainfall_features.py --rainfall-predictions rainfall_predictions.csv
```

Flow CSV columns:

```text
timestamp,river_discharge,reservoir_inflow,water_velocity
```

## Training Manifest

Training uses real labeled samples. Each row points to preprocessed tensors and labels:

```text
satellite_stack,terrain_stack,landuse_stack,flow_sequence,rain_features,target_heatmap,zone_risk
```

Train:

```bash
python scripts/train_sediment_fusion.py --manifest data/processed/training_manifest.csv --epochs 20
```

Metrics reported include MAE, RMSE, R2, IoU, and Dice coefficient.

## Inference

Create an `.npz` sample with keys:

```text
satellite,terrain,landuse,flow_sequence,rain_features
```

Predict:

```bash
python scripts/predict_sediment_hotspots.py \
  --model data/models/sediment_fusion_model.pt \
  --sample data/processed/inference_sample.npz
```

Outputs:

- `data/outputs/future_sediment_hotspot_heatmap.npy`
- `data/outputs/zone_sediment_risk.csv`
- `data/outputs/modality_attention.csv`

## Dashboard

```bash
streamlit run dashboards/sediment_dashboard.py
```
