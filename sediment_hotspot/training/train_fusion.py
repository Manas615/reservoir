import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
from sediment_hotspot.training.dataset import SedimentHotspotDataset
from sediment_hotspot.utils.io import ensure_dir
from sediment_hotspot.utils.metrics import dice_coefficient, iou_score, regression_metrics


def train(manifest: str, epochs: int, batch_size: int, output_dir: str) -> Path:
    dataset = SedimentHotspotDataset(manifest)
    if len(dataset) < 2:
        raise ValueError("Training requires at least two real labeled samples in the manifest.")

    train_len = max(1, int(len(dataset) * 0.8))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_len else []

    model = SedimentHotspotFusionNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            output = model(batch)
            loss = bce(output["heatmap"], batch["target_heatmap"]) + mse(output["zone_risk"], batch["zone_risk"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, val_loader)
        print(f"epoch={epoch + 1} loss={loss.item():.4f} metrics={metrics}")

    output_path = ensure_dir(output_dir) / "sediment_fusion_model.pt"
    torch.save({"model_state": model.state_dict()}, output_path)
    return output_path


def evaluate(model, loader) -> dict[str, float]:
    if not loader:
        return {}
    model.eval()
    heat_true, heat_pred, zone_true, zone_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            heat_true.append(batch["target_heatmap"].numpy())
            heat_pred.append(output["heatmap"].numpy())
            zone_true.append(batch["zone_risk"].numpy())
            zone_pred.append(output["zone_risk"].numpy())
    ht = torch.from_numpy(np.concatenate(heat_true, axis=0)).flatten().numpy()
    hp = torch.from_numpy(np.concatenate(heat_pred, axis=0)).flatten().numpy()
    zt = torch.from_numpy(np.concatenate(zone_true, axis=0)).flatten().numpy()
    zp = torch.from_numpy(np.concatenate(zone_pred, axis=0)).flatten().numpy()
    return {
        **regression_metrics(zt, zp),
        "iou": iou_score(ht, hp),
        "dice": dice_coefficient(ht, hp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", default="data/models")
    args = parser.parse_args()
    print(train(args.manifest, args.epochs, args.batch_size, args.output_dir))


if __name__ == "__main__":
    main()
