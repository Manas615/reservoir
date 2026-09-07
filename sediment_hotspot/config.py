from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path = Path("."); raw: Path = Path("data/raw"); processed: Path = Path("data/processed"); models: Path = Path("data/models"); outputs: Path = Path("data/outputs"); rainfall_predictions: Path = Path("rainfall_predictions.csv")
@dataclass(frozen=True)
class RasterConfig:
    target_crs: str = "EPSG:4326"; target_resolution_m: int = 30; cloud_probability_threshold: float = 40.0; ndwi_threshold: float = 0.15; turbidity_percentile: float = 85.0
@dataclass(frozen=True)
class ModelConfig:
    image_channels: int = 6; terrain_channels: int = 4; landuse_channels: int = 5; flow_features: int = 8; rain_features: int = 4; hidden_dim: int = 128; zones: int = 8; graph_features: int = 7; dsdi_features: int = 11; dropout: float = .20
@dataclass(frozen=True)
class SedimentConfig:
    use_dsdi: bool = True; use_causal_attention: bool = True; use_gnn: bool = True; use_contrastive: bool = True; use_uncertainty: bool = True; use_adaptive_threshold: bool = True
    lambda_zone: float = 1.; lambda_physics: float = .25; lambda_temporal: float = .10; lambda_consistency: float = .15; lambda_contrastive: float = .05; mc_dropout_samples: int = 20
PATHS = Paths(); RASTER = RasterConfig(); MODEL = ModelConfig(); SEDIMENT = SedimentConfig()
