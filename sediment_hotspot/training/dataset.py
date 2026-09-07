from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sediment_hotspot.utils.io import require_file
class SedimentHotspotDataset(Dataset):
 REQUIRED_COLUMNS=["satellite_stack","terrain_stack","landuse_stack","flow_sequence","rain_features","target_heatmap","zone_risk"]
 OPTIONAL={"dsdi_features":(11,),"graph_nodes":(1,7),"graph_adjacency":(1,1),"previous_heatmap":(1,128,128)}
 def __init__(self,manifest_csv):
  self.manifest=pd.read_csv(require_file(manifest_csv,"training manifest")); missing=[c for c in self.REQUIRED_COLUMNS if c not in self.manifest];
  if missing: raise ValueError(f"Training manifest is missing columns: {missing}")
 def __len__(self): return len(self.manifest)
 def __getitem__(self,i):
  r=self.manifest.iloc[i]; out={"satellite":_load(r.satellite_stack),"terrain":_load(r.terrain_stack),"landuse":_load(r.landuse_stack),"flow_sequence":_load(r.flow_sequence),"rain_features":_load(r.rain_features),"target_heatmap":_heat(r.target_heatmap),"zone_risk":_load(r.zone_risk)}
  for key,shape in self.OPTIONAL.items(): out[key]=_load(r[key]) if key in r and isinstance(r[key],str) and Path(r[key]).exists() else torch.zeros(shape)
  return out
def _load(path): return torch.from_numpy(np.load(require_file(path,"dataset tensor")).astype("float32"))
def _heat(path):
 x=_load(path); return x.unsqueeze(0) if x.ndim==2 else x
