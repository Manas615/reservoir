"""Run ablations only on a real manifest; this module never fabricates scores."""
from dataclasses import replace
from sediment_hotspot.config import SEDIMENT
EXPERIMENTS={"baseline_cnn_bilstm":dict(use_dsdi=False,use_causal_attention=False,use_gnn=False,use_contrastive=False,use_uncertainty=False),"attention":dict(use_causal_attention=False),"dsdi":dict(use_dsdi=False),"causal_attention":dict(use_causal_attention=False),"physics_loss":dict(lambda_physics=0.),"adaptive_thresholding":dict(use_adaptive_threshold=False),"gnn":dict(use_gnn=False),"contrastive":dict(use_contrastive=False),"uncertainty":dict(use_uncertainty=False),"full_proposed":dict()}
def configurations(): return {name:replace(SEDIMENT,**change) for name,change in EXPERIMENTS.items()}
