import torch
from torch import nn

from sediment_hotspot.config import MODEL, SEDIMENT
from sediment_hotspot.models.causal_attention import CausalAttentionFusion
from sediment_hotspot.models.cnn import SpatialCNN
from sediment_hotspot.models.contrastive import ModalContrastiveAlignment
from sediment_hotspot.models.dsdi import DynamicSedimentDepositionIndex, DSDIFeatureEncoder
from sediment_hotspot.models.gnn import WatershedReservoirGNN
from sediment_hotspot.models.lstm import FlowLSTM


class SpatialDecoder(nn.Module):
    """Deep spatial decoder with progressive bilinear upsampling and convolutions.
    Preserves 2D spatial coherence and continuity for sediment hotspot risk prediction.
    """

    def __init__(self, in_features: int = 128, out_channels: int = 1, image_size: tuple[int, int] = (128, 128)):
        super().__init__()
        self.image_size = image_size
        self.proj = nn.Sequential(
            nn.Linear(in_features, 128 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Progressive upsampling: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.up1 = block(128, 64)  # 8x8
        self.up2 = block(64, 48)   # 16x16
        self.up3 = block(48, 32)   # 32x32
        self.up4 = block(32, 24)   # 64x64
        self.up5 = block(24, 16)   # 128x128
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = self.proj(x).view(b, 128, 4, 4)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)
        out = self.out_conv(h)
        if out.shape[-2:] != self.image_size:
            out = nn.functional.interpolate(out, size=self.image_size, mode="bilinear", align_corners=False)
        return out


class SedimentHotspotFusionNet(nn.Module):
    """End-to-End Sediment Hotspot Prediction Network.
    Integrates DSDI + WR-GNN + Causal Attention Fusion + Multi-Modal Contrastive Alignment + Spatial Decoder.
    """

    def __init__(self, image_size: tuple[int, int] = (128, 128), config=SEDIMENT):
        super().__init__()
        self.config = config
        self.image_size = image_size
        h = MODEL.hidden_dim

        self.satellite_cnn = SpatialCNN(MODEL.image_channels, h)
        self.terrain_cnn = SpatialCNN(MODEL.terrain_channels, h)
        self.landuse_cnn = SpatialCNN(MODEL.landuse_channels, h)
        self.flow_lstm = FlowLSTM(MODEL.flow_features, h)
        self.rain_encoder = nn.Sequential(
            nn.Linear(MODEL.rain_features, h),
            nn.ReLU(),
        )

        self.dsdi = DynamicSedimentDepositionIndex(MODEL.dsdi_features)
        self.dsdi_encoder = DSDIFeatureEncoder(MODEL.dsdi_features, h)
        self.gnn = WatershedReservoirGNN(MODEL.graph_features, h, layers=config.graph_layers)

        self.fusion = CausalAttentionFusion(h, modality_names=MODEL.modality_names, temperature=config.causal_temperature)
        self.contrastive = ModalContrastiveAlignment(h, projection=MODEL.projection_dim)
        self.dropout = nn.Dropout(MODEL.dropout)

        self.heatmap_head = SpatialDecoder(in_features=h, out_channels=1, image_size=image_size)
        self.zone_head = nn.Sequential(
            nn.Linear(h, 128),
            nn.ReLU(),
            nn.Linear(128, MODEL.zones),
            nn.Sigmoid(),
        )
        self.dsdi_head = nn.Sequential(
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = batch["satellite"].device
        batch_size = batch["satellite"].shape[0]

        satellite_rep = self.satellite_cnn(batch["satellite"])
        terrain_rep = self.terrain_cnn(batch["terrain"])
        landuse_rep = self.landuse_cnn(batch["landuse"])
        flow_rep = self.flow_lstm(batch["flow_sequence"])
        rain_rep = self.rain_encoder(batch["rain_features"])

        z = batch.get("dsdi_features", torch.zeros((batch_size, MODEL.dsdi_features), device=device))
        dsdi_map_input = batch.get("dsdi_feature_map", None)
        dsdi_out = self.dsdi(z, feature_map=dsdi_map_input)
        dsdi_rep = self.dsdi_encoder(z, dsdi_out)

        graph_nodes = batch.get("graph_nodes", torch.zeros((batch_size, 5, MODEL.graph_features), device=device))
        graph_adj = batch.get("graph_adjacency", None)
        graph_rep, _ = self.gnn(graph_nodes, graph_adj)

        representations = {
            "satellite": satellite_rep,
            "flow": flow_rep,
            "terrain": terrain_rep,
            "landuse": landuse_rep,
            "rain": rain_rep,
            "graph": graph_rep,
            "dsdi": dsdi_rep,
        }

        fused, weights = self.fusion(representations, dsdi_prior=dsdi_out)
        contrastive_res = self.contrastive(representations)

        fused_drop = self.dropout(fused)
        heatmap = self.heatmap_head(fused_drop).view(-1, 1, *self.image_size)
        zone_risk = self.zone_head(fused_drop)
        dsdi_pred = self.dsdi_head(fused_drop)

        return {
            "heatmap": heatmap,
            "zone_risk": zone_risk,
            "dsdi": dsdi_out["index"],
            "dsdi_map": dsdi_out["map"],
            "dsdi_prediction": dsdi_pred,
            "dsdi_feature_weights": dsdi_out["feature_weights"],
            "attention": weights["final"],
            "learned_attention": weights["learned"],
            "causal_attention": weights["causal"],
            "prior_attention": weights["prior"],
            "fusion_weight": weights["final"],
            "contrastive_loss": contrastive_res["loss"],
            "contrastive_embeddings": contrastive_res.get("embeddings", {}),
            "representations": representations,
        }
