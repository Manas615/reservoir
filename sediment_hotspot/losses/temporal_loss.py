def temporal_consistency_loss(current,previous,forcing_change=None):
 if previous is None:return current.sum()*0
 loss=(current-previous).abs().mean()
 return loss if forcing_change is None else loss*(1-forcing_change.detach().clamp(0,1).mean())
