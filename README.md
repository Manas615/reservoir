# Reservoir Rainfall and Sediment Hotspot Prediction

## A Multi-Modal AI System for Spatial Sediment Risk Analysis

An AI-based environmental monitoring and decision-support system that combines short-term rainfall forecasting with multimodal sediment hotspot prediction inside reservoirs.

The system integrates meteorological data, satellite imagery, terrain information, land-use information, river-flow observations, graph connectivity, and derived physical indicators. Its primary outputs are spatial sediment-risk heatmaps, zone-wise risk scores, uncertainty estimates, explainability outputs, and counterfactual scenario analysis.

The project is designed as a research-oriented prototype with an emphasis on reproducibility, multimodal learning, spatial reasoning, uncertainty awareness, and interpretable environmental decision support.

---

## 1. Project Overview

Reservoir sedimentation progressively reduces usable storage capacity and can affect irrigation, hydropower, water supply, flood management, and long-term reservoir operation.

Conventional sediment assessment is frequently based on periodic surveys or aggregate sediment estimates. This project instead focuses on predicting **where sedimentation risk is likely to be concentrated** and how that risk may change under different environmental conditions.

The system has two connected stages:

1. **Rainfall Forecasting**
   - Historical Open-Meteo weather data is used to train an XGBoost model for next-hour rainfall prediction.
   - Live weather information is converted into rainfall predictions for 14 river zones.

2. **Sediment Hotspot Prediction**
   - Predicted rainfall is combined with satellite, terrain, land-use, flow, graph, and physical features.
   - A multimodal PyTorch model produces pixel-level hotspot probabilities and zone-level sediment risk.
   - Additional modules provide physical constraints, causal structure, graph propagation, contrastive alignment, uncertainty estimation, explainability, and counterfactual analysis.

---

## 2. Key Capabilities

### Rainfall Forecasting

- Open-Meteo historical and forecast data integration
- Next-hour rainfall prediction using XGBoost
- Lag, rolling, trend, temporal, and rainfall-severity features
- Live prediction pipeline
- Rainfall alerts based on severity categories

### Multimodal Sediment Prediction

The sediment model integrates seven information streams:

1. Satellite imagery
2. Terrain
3. Land use
4. River flow
5. Rainfall
6. Watershed/reservoir graph information
7. Dynamic Sediment Deposition Index (DSDI)

### Research-Oriented Components

- Dynamic Sediment Deposition Index (DSDI)
- Physics-Informed Sediment Loss (PISL)
- Causal Attention Fusion
- Watershed/Reservoir Graph Neural Network
- Multi-Modal Contrastive Alignment
- Temporal Consistency Loss
- Adaptive Hotspot Thresholding
- Uncertainty-Aware Hotspot Prediction
- Counterfactual Sediment Dynamics / intervention analysis
- Explainability through attention and gradient-based methods
- Sediment source-to-hotspot attribution can be added as a graph-based extension

---

## 3. End-to-End Architecture

```text
                         RAINFALL SUBSYSTEM

┌───────────────────────┐
│ Open-Meteo Historical │
│ Weather Data          │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Feature Engineering   │
│ Lags / Rolling /      │
│ Trends / Time         │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ XGBoost Regressor     │
│ Next-Hour Rainfall    │
└──────────┬────────────┘
           │
           ├──────────────► rainfall_model.pkl
           │
           ▼
┌───────────────────────┐
│ Live Open-Meteo       │
│ Forecast Inference    │
└──────────┬────────────┘
           │
           ▼
    rainfall_predictions.csv
           │
           │
           ▼

                    SEDIMENT SUBSYSTEM

 ┌────────────┐  ┌──────────┐  ┌────────────┐
 │ Satellite  │  │   DEM    │  │ Land Cover │
 └─────┬──────┘  └────┬─────┘  └─────┬──────┘
       │              │              │
       ▼              ▼              ▼
 ┌────────────┐  ┌──────────┐  ┌────────────┐
 │ Satellite  │  │ Terrain  │  │ Land-Use   │
 │ Features   │  │ Features │  │ Features   │
 └─────┬──────┘  └────┬─────┘  └─────┬──────┘
       │              │              │
       └──────────────┼──────────────┘
                      │
 ┌────────────┐  ┌────────────┐  ┌────────────┐
 │ River Flow │  │ Rainfall   │  │ DSDI       │
 └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
       │               │               │
       └───────────────┼───────────────┘
                       ▼
              ┌─────────────────────┐
              │ Multimodal Fusion   │
              │ CNN + LSTM + GNN +  │
              │ DSDI + Attention    │
              └──────────┬──────────┘
                         │
          ┌──────────────┼───────────────┐
          │              │               │
          ▼              ▼               ▼
   Hotspot Heatmap   Zone Risk      DSDI / Attention
          │              │               │
          └──────────────┼───────────────┘
                         ▼
              ┌─────────────────────┐
              │ Uncertainty / XAI   │
              │ / Counterfactuals   │
              └──────────┬──────────┘
                         ▼
                Streamlit Dashboard
