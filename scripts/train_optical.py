import os
import argparse

import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from lumos.model import PCAutoencoder, PD2Latent, ShapeNet
from lumos.data import WaveguideDataset


def precompute_latents(ae, dataset, batch_size: int, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optical_signals = []
    latent_vectors = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_points, batch_optical, _ = batch
            batch_points = batch_points.to(device).float()
            z = ae.encoder(batch_points)
            latent_vectors.append(z.cpu())
            optical_signals.append(batch_optical)
            if (batch_idx + 1) % 200 == 0:
                print(f"Precomputing latents: {batch_idx + 1}/{len(loader)} batches")
    return torch.cat(latent_vectors, dim=0), torch.cat(optical_signals, dim=0)


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = 0.0
    for batch_x, batch_z in loader:
        batch_x, batch_z = batch_x.to(device), batch_z.to(device)
        optimizer.zero_grad()
        pred_z = model(batch_x)
        loss = torch.mean((pred_z - batch_z) ** 2)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * batch_x.size(0)
    return loss_sum / max(1, len(loader))


@torch.no_grad()
def evaluate_epoch(model, loader, device):
    model.eval()
    loss_sum = 0.0
    for batch_x, batch_z in loader:
        batch_x, batch_z = batch_x.to(device), batch_z.to(device)
        pred_z = model(batch_x)
        loss = torch.mean((pred_z - batch_z) ** 2)
        loss_sum += loss.item() * batch_x.size(0)
    return loss_sum / max(1, len(loader))


def train(model, train_loader, val_loader, optimizer, sched, cfg: dict, device, ckpt_dir: str):
    tr = cfg["train_optical"]
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float("inf")
    best_path = None
    epochs_without_improve = 0

    for epoch in range(tr["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_epoch(model, val_loader, device)
        sched.step()

        print(f"Epoch {epoch + 1}/{tr['epochs']}, Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improve = 0
            best_path = os.path.join(ckpt_dir, f"model_ep{epoch + 1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, best_path)
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= tr["early_stop_patience"]:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    return best_val, best_path


def main(cfg: dict):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    tr = cfg["train_optical"]
    out_cfg = cfg["output"]

    # Output always goes into the current run directory
    run_dir = os.path.join(out_cfg["base_dir"], out_cfg["run_name"])
    ckpt_dir = os.path.join(run_dir, "optical_checkpoints")
    combined_path = os.path.join(run_dir, "best_combined.pth")

    # AE checkpoint: use ae_run_name override if set, otherwise same run
    ae_run = tr.get("ae_run_name") or out_cfg["run_name"]
    ae_run_dir = os.path.join(out_cfg["base_dir"], ae_run)
    ae_ckpt_path = os.path.join(ae_run_dir, "checkpoints", "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WaveguideDataset(data_cfg["path"], stride=data_cfg["stride"])
    sample_point, sample_optical, _ = dataset[0]
    print(f"Sample point cloud shape: {sample_point.shape}")
    print(f"Sample optical signal shape: {sample_optical.shape}")

    ae = PCAutoencoder(
        latent_dim=model_cfg["latent_dim"],
        num_points=model_cfg["num_points"],
        tnet1=model_cfg["tnet1"],
        tnet2=model_cfg["tnet2"],
    ).to(device)
    best_ae = torch.load(ae_ckpt_path, map_location=device)
    ae.load_state_dict(best_ae["model_state_dict"])
    ae.eval()
    print(f"Autoencoder loaded from {ae_ckpt_path}")

    print("Precomputing latent vectors...")
    latent_vectors, optical_signals = precompute_latents(ae, dataset, tr["batch_size"], device)
    print(f"Latent vectors shape: {latent_vectors.shape}")

    X = optical_signals.float()
    Z = latent_vectors.float()
    tensor_dataset = torch.utils.data.TensorDataset(X, Z)

    train_size = int(data_cfg["train_split"] * len(tensor_dataset))
    val_size = int(data_cfg["val_split"] * len(tensor_dataset))
    test_size = len(tensor_dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(
        tensor_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )

    train_loader = DataLoader(train_dataset, batch_size=tr["batch_size"], shuffle=True,
                              num_workers=tr["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=tr["batch_size"], shuffle=False,
                            num_workers=tr["num_workers"])

    model = PD2Latent(in_features=model_cfg["optical_dim"], out_features=model_cfg["latent_dim"]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=tr["lr"], momentum=0.9)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr["epochs"])


    print("Starting training...")
    best_val, best_path = train(model, train_loader, val_loader, optimizer, sched, cfg, device, ckpt_dir)
    print(f"Best model saved at {best_path} with Val Loss: {best_val:.6f}")

    combined_model = ShapeNet(
        pd_in_features=model_cfg["optical_dim"],
        latent_dim=model_cfg["latent_dim"],
        num_points=model_cfg["num_points"],
        tnet1=model_cfg["tnet1"],
        tnet2=model_cfg["tnet2"],
    )
    pd2latent_ckpt = torch.load(best_path, map_location=device)
    combined_model.pd2latent.load_state_dict(pd2latent_ckpt["model_state_dict"])
    combined_model.decoder.load_state_dict(ae.decoder.state_dict())

    torch.save({
        "epoch": pd2latent_ckpt.get("epoch"),
        "val_loss": best_val,
        "optical_dim": model_cfg["optical_dim"],
        "latent_dim": model_cfg["latent_dim"],
        "num_points": model_cfg["num_points"],
        "tnet1": model_cfg["tnet1"],
        "tnet2": model_cfg["tnet2"],
        "pd2latent_state_dict": combined_model.pd2latent.state_dict(),
        "decoder_state_dict": combined_model.decoder.state_dict(),
    }, combined_path)
    print(f"Combined model saved to {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file")
    parser.add_argument("--run-name", default=None, help="Override output.run_name from config")
    parser.add_argument("--ae-run-name", default=None, help="Load AE weights from a different run (overrides train_optical.ae_run_name)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name
    if args.ae_run_name is not None:
        cfg["train_optical"]["ae_run_name"] = args.ae_run_name

    import random
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    main(cfg)
