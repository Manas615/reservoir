import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from sediment_hotspot.config import SEDIMENT
from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
from sediment_hotspot.training.dataset import SedimentHotspotDataset
from sediment_hotspot.losses.physics_loss import physics_informed_sediment_loss
from sediment_hotspot.losses.temporal_loss import temporal_consistency_loss
from sediment_hotspot.utils.io import ensure_dir
def train(manifest,epochs,batch_size,output_dir):
 data=SedimentHotspotDataset(manifest); loader=DataLoader(data,batch_size=batch_size,shuffle=True); model=SedimentHotspotFusionNet(); opt=torch.optim.AdamW(model.parameters(),lr=1e-4); bce=nn.BCELoss(); mse=nn.MSELoss()
 for epoch in range(epochs):
  model.train()
  for batch in loader:
   o=model(batch); loss=bce(o["heatmap"],batch["target_heatmap"])+SEDIMENT.lambda_zone*mse(o["zone_risk"],batch["zone_risk"])+SEDIMENT.lambda_physics*physics_informed_sediment_loss(o["heatmap"],o["dsdi"],batch["dsdi_features"])+SEDIMENT.lambda_temporal*temporal_consistency_loss(o["heatmap"],batch["previous_heatmap"])+SEDIMENT.lambda_consistency*mse(o["dsdi_prediction"],o["dsdi"].detach())+SEDIMENT.lambda_contrastive*o["contrastive_loss"]; opt.zero_grad(); loss.backward(); opt.step()
  print(f"epoch={epoch+1} loss={loss.item():.4f}")
 path=ensure_dir(output_dir)/"sediment_fusion_model.pt"; torch.save({"model_state":model.state_dict(),"config":SEDIMENT.__dict__},path); return path
if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--output-dir",default="data/models"); a=p.parse_args(); print(train(a.manifest,a.epochs,a.batch_size,a.output_dir))
