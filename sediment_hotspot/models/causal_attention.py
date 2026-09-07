"""Causal Attention Fusion: rainfall -> flow -> erosion -> transport -> deposition."""
import torch
from torch import nn
class CausalAttentionFusion(nn.Module):
 def __init__(self,dim,modalities=7):
  super().__init__(); self.score=nn.Sequential(nn.Linear(dim,64),nn.Tanh(),nn.Linear(64,1)); self.prior_scale=nn.Parameter(torch.tensor(1.)); m=torch.tensor([[.3,.4,.1,.1,.6,.2,.4],[.6,.8,.2,.1,.8,.5,.7],[.2,.5,.8,.6,.2,.4,.6],[.2,.4,.6,.8,.2,.3,.5],[.6,.8,.2,.2,.2,.3,.5],[.3,.7,.4,.3,.5,.6,.7],[.5,.7,.6,.5,.7,.7,.5]]); self.register_buffer("causal_matrix",m[:modalities,:modalities])
 def forward(self,features,prior):
  x=torch.stack(features,1); logits=self.score(x).squeeze(-1); learned=torch.softmax(logits,1); causal_logits=learned@self.causal_matrix; causal=torch.softmax(causal_logits,1); final=torch.softmax(logits+causal_logits+self.prior_scale*prior.reshape(-1,1),1); return (x*final.unsqueeze(-1)).sum(1),{"learned":learned,"causal":causal,"final":final}
