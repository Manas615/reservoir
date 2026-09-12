import argparse
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from sediment_hotspot.config import SEDIMENT
from sediment_hotspot.losses.physics_loss import physics_informed_sediment_loss
from sediment_hotspot.losses.temporal_loss import temporal_consistency_loss
from sediment_hotspot.losses.counterfactual_loss import CounterfactualConsistencyLoss
from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
from sediment_hotspot.training.dataset import SedimentHotspotDataset, split_dataset_manifest
from sediment_hotspot.utils.io import ensure_dir


def train(
    manifest: str | pd.DataFrame,
    epochs: int = 15,
    batch_size: int = 4,
    output_dir: str = "data/models",
    lr: float = 1e-4,
    device: str = "cpu",
    config=SEDIMENT,
) -> dict:
    if isinstance(manifest, str):
        full_df = pd.read_csv(manifest)
    else:
        full_df = manifest

    train_df, val_df, test_df = split_dataset_manifest(full_df)
    train_ds = SedimentHotspotDataset(train_df)
    val_ds = SedimentHotspotDataset(val_df) if len(val_df) > 0 else train_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SedimentHotspotFusionNet(config=config).to(device)
    cf_loss_fn = CounterfactualConsistencyLoss(tolerance=config.counterfactual_tolerance)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_losses = {
            "L_hotspot": 0.0, "L_zone": 0.0, "L_physics": 0.0,
            "L_temporal": 0.0, "L_consistency": 0.0, "L_contrastive": 0.0,
            "L_counterfactual": 0.0, "L_total": 0.0,
        }
        steps = 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)

            l_hotspot = bce(out["heatmap"], batch["target_heatmap"])
            l_zone = mse(out["zone_risk"], batch["zone_risk"])

            l_physics, _ = physics_informed_sediment_loss(out, batch) if config.use_physics_loss else (torch.tensor(0.0, device=device), {})
            l_temporal, _ = temporal_consistency_loss(out["heatmap"], batch.get("previous_heatmap"), batch.get("forcing_change"), tolerance=config.temporal_change_tolerance) if config.use_temporal_constraint else (torch.tensor(0.0, device=device), {})
            l_consistency = mse(out["dsdi_prediction"], out["dsdi"].detach()) if config.use_dsdi else torch.tensor(0.0, device=device)
            l_contrastive = out["contrastive_loss"] if config.use_contrastive else torch.tensor(0.0, device=device)

            if config.use_counterfactual_loss:
                l_cf, _ = cf_loss_fn(model, batch, out)
            else:
                l_cf = torch.tensor(0.0, device=device)

            total_loss = (
                l_hotspot
                + config.lambda_zone * l_zone
                + config.lambda_physics * l_physics
                + config.lambda_temporal * l_temporal
                + config.lambda_consistency * l_consistency
                + config.lambda_contrastive * l_contrastive
                + config.lambda_counterfactual * l_cf
            )

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            epoch_losses["L_hotspot"] += l_hotspot.item()
            epoch_losses["L_zone"] += l_zone.item()
            epoch_losses["L_physics"] += l_physics.item()
            epoch_losses["L_temporal"] += l_temporal.item()
            epoch_losses["L_consistency"] += l_consistency.item()
            epoch_losses["L_contrastive"] += l_contrastive.item()
            epoch_losses["L_counterfactual"] += l_cf.item()
            epoch_losses["L_total"] += total_loss.item()
            steps += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(steps, 1)

        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(batch)
                val_loss += bce(out["heatmap"], batch["target_heatmap"]).item()
                val_steps += 1
        val_loss /= max(val_steps, 1)
        epoch_losses["val_loss"] = val_loss
        epoch_losses["epoch"] = epoch + 1
        history.append(epoch_losses)

        print(f"Epoch {epoch+1}/{epochs} | Total: {epoch_losses['L_total']:.4f} | Hotspot: {epoch_losses['L_hotspot']:.4f} | Phys: {epoch_losses['L_physics']:.4f} | CF: {epoch_losses['L_counterfactual']:.4f} | Val: {val_loss:.4f}")

    out_p = ensure_dir(output_dir) / "sediment_fusion_model.pt"
    torch.save({"model_state": model.state_dict(), "config": config.__dict__}, out_p)
    return {"model_path": out_p, "model": model, "history": history, "train_df": train_df, "val_df": val_df, "test_df": test_df}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output-dir", default="data/models")
    a = p.parse_args()
    train(a.manifest, a.epochs, a.batch_size, a.output_dir)


if __name__ == "__main__":
    main()
