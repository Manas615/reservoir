import torch
from torch import nn
from sediment_hotspot.config import MODEL,SEDIMENT
from sediment_hotspot.models.cnn import SpatialCNN
from sediment_hotspot.models.lstm import FlowLSTM
from sediment_hotspot.models.dsdi import DynamicSedimentDepositionIndex,DSDIFeatureEncoder
from sediment_hotspot.models.gnn import WatershedReservoirGNN
from sediment_hotspot.models.causal_attention import CausalAttentionFusion
from sediment_hotspot.models.contrastive import ModalContrastiveAlignment
class SedimentHotspotFusionNet(nn.Module):
 """DSDI + WR-GNN + CAF sediment model; rainfall subsystem remains an external input."""
 def __init__(self,image_size=(128,128),config=SEDIMENT):
  super().__init__(); self.config=config; h=MODEL.hidden_dim; self.satellite_cnn=SpatialCNN(MODEL.image_channels,h); self.terrain_cnn=SpatialCNN(MODEL.terrain_channels,h); self.landuse_cnn=SpatialCNN(MODEL.landuse_channels,h); self.flow_lstm=FlowLSTM(MODEL.flow_features,h); self.rain_encoder=nn.Sequential(nn.Linear(MODEL.rain_features,h),nn.ReLU()); self.dsdi=DynamicSedimentDepositionIndex(MODEL.dsdi_features); self.dsdi_encoder=DSDIFeatureEncoder(MODEL.dsdi_features,h); self.gnn=WatershedReservoirGNN(MODEL.graph_features,h); self.fusion=CausalAttentionFusion(h,7); self.contrastive=ModalContrastiveAlignment(h); self.dropout=nn.Dropout(MODEL.dropout); self.heatmap_head=nn.Sequential(nn.Linear(h,256),nn.ReLU(),nn.Dropout(MODEL.dropout),nn.Linear(256,image_size[0]*image_size[1]),nn.Sigmoid()); self.zone_head=nn.Sequential(nn.Linear(h,128),nn.ReLU(),nn.Linear(128,MODEL.zones),nn.Sigmoid()); self.dsdi_head=nn.Sequential(nn.Linear(h,1),nn.Sigmoid()); self.image_size=image_size
 def forward(self,batch):
    satellite=self.satellite_cnn(batch["satellite"]); terrain=self.terrain_cnn(batch["terrain"]); landuse=self.landuse_cnn(batch["landuse"]); flow=self.flow_lstm(batch["flow_sequence"]); rain=self.rain_encoder(batch["rain_features"]); z=batch.get("dsdi_features",torch.zeros((satellite.shape[0],MODEL.dsdi_features),device=satellite.device)); index,index_weights=self.dsdi(z); graph_nodes=batch.get("graph_nodes",torch.zeros((satellite.shape[0],1,MODEL.graph_features),device=satellite.device)); graph,_=self.gnn(graph_nodes,batch.get("graph_adjacency")); dsdi=self.dsdi_encoder(z,index); reps=[satellite,flow,terrain,landuse,rain,graph,dsdi]; fused,weights=self.fusion(reps,index); fused=self.dropout(fused); heatmap=self.heatmap_head(fused).view(-1,1,*self.image_size); return {"heatmap":heatmap,"zone_risk":self.zone_head(fused),"dsdi":index,"dsdi_prediction":self.dsdi_head(fused),"dsdi_feature_weights":index_weights,"attention":weights["final"],"learned_attention":weights["learned"],"causal_attention":weights["causal"],"fusion_weight":weights["final"],"contrastive_loss":self.contrastive(reps)}
