import torch
from torch import nn
import torch.nn.functional as F
class ModalContrastiveAlignment(nn.Module):
 def __init__(self,hidden,projection=64,temperature=.15): super().__init__(); self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,projection)); self.temperature=temperature
 def forward(self,representations):
  z=F.normalize(self.head(torch.stack(representations,1)),dim=-1)
  if z.shape[0]<2:return z.sum()*0
  logits=z[:,0]@z[:,1:].mean(1).T/self.temperature; return F.cross_entropy(logits,torch.arange(z.shape[0],device=z.device))
