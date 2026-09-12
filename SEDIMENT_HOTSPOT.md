# Sediment Hotspot Prediction Subsystem Architecture

## Overview
This subsystem integrates multi-modal remote sensing, terrain morphometry, hydrometric flow sequences, meteorological forecasts, and watershed topological graphs into a unified physics-informed neural framework.

### Core Mathematical Components
- **DSDI**: $\text{DSDI}(x,y,t) = G(\text{forcing}, \text{transport}, \text{erosion}, \text{retention}, \text{optical})$
- **Causal Attention**: $A = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + C_{\text{causal}} + P_{\text{DSDI}}\right)$
- **Loss Formulation**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{hotspot}} + \lambda_{\text{zone}}\mathcal{L}_{\text{zone}} + \lambda_{\text{phys}}\mathcal{L}_{\text{physics}} + \lambda_{\text{temp}}\mathcal{L}_{\text{temporal}} + \lambda_{\text{cons}}\mathcal{L}_{\text{consistency}} + \lambda_{\text{cont}}\mathcal{L}_{\text{contrastive}}$$

### Label Provenance & Limitations
- All training and evaluation targets document explicit provenance (`REAL_OBSERVATION`, `DERIVED_FROM_SATELLITE`, `PHYSICS_DERIVED`, or `SIMULATED/AUGMENTED`).
- Satellite-derived plume masks and adaptive thresholds are proxy indicators of sediment concentration and are clearly designated as derived remote-sensing labels.
