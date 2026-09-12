import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sediment_hotspot.config import SEDIMENT
from sediment_hotspot.evaluate import evaluate_outputs
from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
from sediment_hotspot.training.dataset import SedimentHotspotDataset, split_dataset_manifest
from sediment_hotspot.training.train_fusion import train
from sediment_hotspot.utils.io import ensure_dir

ABLATION_CONFIGS = {
    "Full Proposed Model + CSDN": dict(use_dsdi=True, use_causal_attention=True, use_gnn=True, use_contrastive=True, use_uncertainty=True, use_physics_loss=True, use_temporal_constraint=True, use_counterfactual_loss=True),
    "- CSDN Loss": dict(use_dsdi=True, use_causal_attention=True, use_gnn=True, use_contrastive=True, use_uncertainty=True, use_physics_loss=True, use_temporal_constraint=True, use_counterfactual_loss=False),
    "- Physics Loss (PISL)": dict(use_dsdi=True, use_causal_attention=True, use_gnn=True, use_contrastive=True, use_uncertainty=True, use_physics_loss=False, use_temporal_constraint=True, use_counterfactual_loss=True),
    "- Causal Attention (CAF)": dict(use_dsdi=True, use_causal_attention=False, use_gnn=True, use_contrastive=True, use_uncertainty=True, use_physics_loss=True, use_temporal_constraint=True, use_counterfactual_loss=True),
    "- WR-GNN": dict(use_dsdi=True, use_causal_attention=True, use_gnn=False, use_contrastive=True, use_uncertainty=True, use_physics_loss=True, use_temporal_constraint=True, use_counterfactual_loss=True),
    "- MMCA Contrastive": dict(use_dsdi=True, use_causal_attention=True, use_gnn=True, use_contrastive=False, use_uncertainty=True, use_physics_loss=True, use_temporal_constraint=True, use_counterfactual_loss=True),
    "- Temporal Consistency": dict(use_dsdi=True, use_causal_attention=True, use_gnn=True, use_contrastive=True, use_uncertainty=True, use_physics_loss=True, use_temporal_constraint=False, use_counterfactual_loss=True),
    "- DSDI": dict(use_dsdi=False, use_causal_attention=True, use_gnn=True, use_contrastive=True, use_uncertainty=True, use_physics_loss=True, use_temporal_constraint=True, use_counterfactual_loss=True),
    "Baseline (CNN+BiLSTM)": dict(use_dsdi=False, use_causal_attention=False, use_gnn=False, use_contrastive=False, use_uncertainty=False, use_physics_loss=False, use_temporal_constraint=False, use_counterfactual_loss=False),
}


def run_ablation_study(manifest_csv: str, output_csv: str = "results/ablation_results.csv", epochs: int = 6, batch_size: int = 4):
    ensure_dir(Path(output_csv).parent)
    df = pd.read_csv(manifest_csv)
    train_df, val_df, test_df = split_dataset_manifest(df)
    test_ds = SedimentHotspotDataset(test_df)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    records = []
    for exp_name, overrides in ABLATION_CONFIGS.items():
        print(f"Running Ablation: {exp_name}...")
        cfg = replace(SEDIMENT, **overrides)
        res = train(df, epochs=epochs, batch_size=batch_size, config=cfg)
        model = res["model"]
        model.eval()

        all_pred_heat, all_true_heat = [], []
        all_pred_zone, all_true_zone = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                all_pred_heat.append(out["heatmap"].numpy())
                all_true_heat.append(batch["target_heatmap"].numpy())
                all_pred_zone.append(out["zone_risk"].numpy())
                all_true_zone.append(batch["zone_risk"].numpy())

        pred_h = np.concatenate(all_pred_heat, axis=0)
        true_h = np.concatenate(all_true_heat, axis=0)
        pred_z = np.concatenate(all_pred_zone, axis=0)
        true_z = np.concatenate(all_true_zone, axis=0)

        metrics = evaluate_outputs(true_h, pred_h, zone_true=true_z, zone_pred=pred_z)
        metrics["ablation_stage"] = exp_name
        for k, v in overrides.items():
            metrics[k] = v
        records.append(metrics)

    res_df = pd.DataFrame(records)
    res_df.to_csv(output_csv, index=False)
    print(f"Saved ablation results to {output_csv}")
    return res_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="results/ablation_results.csv")
    p.add_argument("--epochs", type=int, default=6)
    a = p.parse_args()
    run_ablation_study(a.manifest, a.output, a.epochs)
