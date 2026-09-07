import torch
from torch import nn
class WatershedReservoirGNN(nn.Module):
 """Directed message passing; nodes may be watersheds, tributaries, inlets, or zones."""
 def __init__(self,features,hidden): super().__init__(); self.self_layer=nn.Linear(features,hidden); self.neighbor=nn.Linear(features,hidden)
 def forward(self,nodes,adjacency=None):
  if adjacency is None: adjacency=torch.eye(nodes.shape[1],device=nodes.device).expand(nodes.shape[0],-1,-1)
  a=adjacency.float()/adjacency.sum(-1,keepdim=True).clamp_min(1); encoded=torch.relu(self.self_layer(nodes)+self.neighbor(torch.bmm(a,nodes))); return encoded.mean(1),encoded
