import torch
from torch import nn

from sediment_hotspot.config import MODEL
from sediment_hotspot.models.cnn import SpatialCNN
from sediment_hotspot.models.lstm import FlowLSTM


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim: int, modalities: int):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(feature_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.modalities = modalities

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(features, dim=1)
        weights = torch.softmax(self.attention(stacked), dim=1)
        fused = (stacked * weights).sum(dim=1)
        return fused, weights.squeeze(-1)


class SedimentHotspotFusionNet(nn.Module):
    """CNN + LSTM + attention fusion model for sediment hotspot prediction."""

    def __init__(self, image_size: tuple[int, int] = (128, 128)):
        super().__init__()
        self.satellite_cnn = SpatialCNN(MODEL.image_channels, MODEL.hidden_dim)
        self.terrain_cnn = SpatialCNN(MODEL.terrain_channels, MODEL.hidden_dim)
        self.landuse_cnn = SpatialCNN(MODEL.landuse_channels, MODEL.hidden_dim)
        self.flow_lstm = FlowLSTM(MODEL.flow_features, MODEL.hidden_dim)
        self.rain_encoder = nn.Sequential(nn.Linear(MODEL.rain_features, MODEL.hidden_dim), nn.ReLU())
        self.fusion = AttentionFusion(MODEL.hidden_dim, modalities=5)

        h, w = image_size
        self.heatmap_head = nn.Sequential(
            nn.Linear(MODEL.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, h * w),
            nn.Sigmoid(),
        )
        self.zone_head = nn.Sequential(
            nn.Linear(MODEL.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, MODEL.zones),
            nn.Sigmoid(),
        )

        self.image_size = image_size

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        satellite = self.satellite_cnn(batch["satellite"])
        terrain = self.terrain_cnn(batch["terrain"])
        landuse = self.landuse_cnn(batch["landuse"])
        flow = self.flow_lstm(batch["flow_sequence"])
        rain = self.rain_encoder(batch["rain_features"])
        fused, attention = self.fusion([satellite, flow, terrain, landuse, rain])
        heatmap = self.heatmap_head(fused).view(-1, 1, *self.image_size)
        zones = self.zone_head(fused)
        return {"heatmap": heatmap, "zone_risk": zones, "attention": attention}
