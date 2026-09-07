import torch
import torch.nn.functional as F
def physics_informed_sediment_loss(pred,dsdi,z):
 p=pred.mean((-1,-2)).squeeze(1); expected=(.30*z[:,0]+.25*z[:,3]+.30*z[:,7]+.15*(1-z[:,4])).clamp(0,1); return F.mse_loss(p,expected)+.5*F.mse_loss(p,dsdi.squeeze(-1).detach())
