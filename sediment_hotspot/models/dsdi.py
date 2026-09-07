"""Dynamic Sediment Deposition Index (DSDI).
DSDI(z)=sum_i softmax(g(z))_i * z_i, with condition-dependent weights.
All inputs are normalized to [0,1]; velocity is inverted since slow water deposits."""
import torch
from torch import nn
DSDI_FEATURES=("rainfall","rainfall_intensity","discharge","inflow","velocity","slope","curvature","erosion","landuse_erosion","turbidity","plume")
class DynamicSedimentDepositionIndex(nn.Module):
 def __init__(self,features=11):
  super().__init__(); self.gate=nn.Sequential(nn.Linear(features,features),nn.Tanh(),nn.Linear(features,features)); self.register_buffer("sign",torch.tensor([1,1,1,1,-1,1,1,1,1,1,1.]))
 def forward(self,z):
  z=z.clamp(0,1); x=torch.where(self.sign[None]<0,1-z,z); w=torch.softmax(self.gate(z),-1); return (w*x).sum(-1,keepdim=True),w
class DSDIFeatureEncoder(nn.Module):
 def __init__(self,features,hidden): super().__init__(); self.net=nn.Sequential(nn.Linear(features+1,hidden),nn.ReLU(),nn.LayerNorm(hidden))
 def forward(self,z,index): return self.net(torch.cat([z,index],-1))
