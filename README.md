# Reservoir Rainfall and Sediment Hotspot Prediction System

An end-to-end multi-modal deep learning and physics-informed framework for predicting reservoir sediment deposition hotspots, zone-wise sedimentation risks, and upstream rainfall forcing.

---

## 1. Algorithmic Contributions
The sediment prediction subsystem introduces the following specific algorithmic components:

1. **Dynamic Sediment Deposition Index (DSDI)**: A dynamic prior index combining 11 environmental factors (rainfall forcing, river discharge, reservoir inflow, water velocity, terrain slope, curvature, erosion susceptibility, land-use probability, satellite NDTI turbidity, and plume intensity) with learned adaptive gating.
2. **Causal Attention Fusion (CAF)**: Multi-modal fusion combining learned feature relevance, a domain causal prior matrix (Rainfall $\to$ Flow $\to$ Erosion $\to$ Transport $\to$ Deposition), and DSDI compatibility.
3. **Physics-Informed Sediment Loss (PISL)**: Differentiable loss constraints enforcing physical relationships among sediment supply, transport capacity, flow velocity, and deposition drop.
4. **Adaptive Hotspot Thresholding (AHT)**: Dynamic thresholding conditioned on hydrological regime, rainfall wetness, seasonality, and local historical turbidity variance.
5. **Watershed-to-Reservoir Graph Network (WR-GNN)**: Graph attention network modeling topological and hydrological connectivity from upstream watershed tributaries to reservoir zones.
6. **Multi-Modal Contrastive Alignment (MMCA)**: InfoNCE contrastive representation learning aligning cross-modal embeddings for identical hydrological events while separating discordant contexts.
7. **Uncertainty-Aware Hotspot Prediction (UAHP)**: Monte Carlo Dropout inference providing pixel-wise mean risk, epistemic uncertainty maps, and automated risk-confidence triage categories (High Risk + High Confidence, High Risk + Low Confidence, Low Risk).
8. **Temporal Consistency Constraint**: Regularization penalizing abrupt, unphysical spatial changes between consecutive observation timestamps conditioned on environmental forcing change.

---

## 2. Preserved Rainfall Subsystem
The upstream rainfall forecasting subsystem is fully preserved:
- Open-Meteo historical & real-time weather integration
- Lagged and rolling feature engineering (`feature_engineering.py`)
- XGBoost forecasting model (`rainfall_model.pkl`)
- Real-time zone forecasts saved to `rainfall_predictions.csv`
- Rainfall feature adapter for sediment fusion compatibility

---

## 3. Quick Start & Execution

### A. Run Unit Tests
```bash
python tests/test_sediment_hotspot.py
```

### B. Generate Multi-Modal Training Data & Manifest
```bash
python scripts/generate_synthetic_data.py
```

### C. Train the Full Proposed Fusion Model
```bash
python scripts/train_sediment_fusion.py --manifest data/processed/manifest.csv --epochs 15
```

### D. Run Baseline Models Comparison
```bash
python sediment_hotspot/baselines.py --manifest data/processed/manifest.csv --output results/baseline_results.csv
```

### E. Run Ablation Experiments
```bash
python sediment_hotspot/ablation.py --manifest data/processed/manifest.csv --output results/ablation_results.csv
```

### F. Run Inference & Export Monitoring Artifacts
```bash
python scripts/predict_sediment_hotspots.py --model data/models/sediment_fusion_model.pt --sample data/processed/sample_0.npz --output-dir data/outputs
```

### G. Launch Interactive Dashboard
```bash
streamlit run dashboards/sediment_dashboard.py
```

---

## 4. Evaluation Outputs
- `results/baseline_results.csv`: Comparison across Random Forest, XGBoost, CNN-only, BiLSTM-only, CNN+BiLSTM, Standard Attention, and Proposed Full Model on an identical test split.
- `results/ablation_results.csv`: Step-by-step ablation metrics validating the marginal contribution of each proposed algorithm.
- `data/outputs/`: Inference maps (`future_sediment_hotspot_heatmap.npy`, `dsdi_heatmap.npy`, `uncertainty_heatmap.npy`, `zone_sediment_risk.csv`, `modality_attention.csv`, `causal_attention.csv`, `risk_timeline.csv`, `prediction_metadata.json`).
