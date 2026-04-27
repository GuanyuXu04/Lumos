import os
import gc
import random
import argparse

import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lumos.model import PCAutoencoder
from lumos.data import WaveguideDataset
from lumos.ddp import (
    is_dist, is_main, ddp_setup, ddp_cleanup, all_reduce_mean,
    select_free_gpus, apply_visible_devices,
)
from lumos.losses import chamfer_distance, repulsion_loss
from lumos.viz import plot_pc_pair


def get_bn_momentum(step: int, cfg: dict) -> float:
    tr = cfg["train_ae"]
    bn_momentum = tr["bn_init_decay"] * (tr["bn_decay_rate"] ** ((step * tr["batch_size"]) // tr["bn_decay_step"]))
    return min(tr["bn_decay_clip"], 1 - bn_momentum)


def _set_bn_momentum(module: nn.Module, momentum: float):
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        module.momentum = momentum


_fixed_inp = None
_fixed_tgt = None


@torch.no_grad()
def save_progress_plot(epoch: int, model, device, out_dir: str, prefix: str = "progress_fixed"):
    if not is_main():
        return
    global _fixed_inp, _fixed_tgt
    assert _fixed_inp is not None
    model.eval()
    pred = model(_fixed_inp).squeeze(0).detach().cpu().numpy()
    tgt  = _fixed_tgt.numpy()
    os.makedirs(out_dir, exist_ok=True)
    plot_pc_pair(
        tgt, pred,
        save_path=os.path.join(out_dir, f"{prefix}_ep{epoch + 1:03d}.png"),
        left_title=f"Original (Epoch {epoch + 1})",
        right_title="Reconstructed",
    )


# -------------------- Train / Eval --------------------
def train_loop(rank: int, cfg: dict):
    local_rank = rank

    if "LOCAL_RANK" in os.environ:
        local_rank = ddp_setup()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    tr = cfg["train_ae"]
    loss_cfg = tr["loss"]
    out_cfg = cfg["output"]

    run_dir = os.path.join(out_cfg["base_dir"], out_cfg["run_name"])
    fig_dir = os.path.join(run_dir, "figs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    log_path = os.path.join(run_dir, "record_log.txt")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # In DDP, let rank 0 build any missing cache files first, then barrier so
    # other ranks always load from cache (avoids redundant computation and
    # simultaneous writes to the same cache file).
    if is_dist():
        import torch.distributed as dist
        if local_rank != 0:
            dist.barrier()

    dataset = WaveguideDataset(data_cfg["path"], stride=data_cfg["stride"], verbose=is_main())

    if is_dist():
        if local_rank == 0:
            dist.barrier()
    n_total = len(dataset)
    n_train = int(data_cfg["train_split"] * n_total)
    n_val = int(data_cfg["val_split"] * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )

    train_sampler = DistributedSampler(train_set, shuffle=True) if is_dist() else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if is_dist() else None
    test_sampler = DistributedSampler(test_set, shuffle=False) if is_dist() else None

    train_loader = DataLoader(
        train_set, batch_size=tr["batch_size"], shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=tr["num_workers"], pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=tr["batch_size"], shuffle=False,
        sampler=val_sampler, num_workers=tr["num_workers"], pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=tr["batch_size"], shuffle=False,
        sampler=test_sampler, num_workers=tr["num_workers"], pin_memory=True, drop_last=False,
    )

    model = PCAutoencoder(
        latent_dim=model_cfg["latent_dim"],
        num_points=model_cfg["num_points"],
        tnet1=model_cfg["tnet1"],
        tnet2=model_cfg["tnet2"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=tr["base_lr"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=tr["scheduler"]["factor"], patience=tr["scheduler"]["patience"]
    )

    start_epoch = 0
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state, strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        if is_main():
            print(f"Resumed from epoch {ckpt['epoch'] + 1} (val {ckpt['val_loss']:.4f})")

    if is_dist():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, static_graph=False,
        )

    global _fixed_inp, _fixed_tgt
    if is_main():
        item = test_set[42]
        pts = item[0] if isinstance(item, (list, tuple)) else item
        _fixed_tgt = pts.detach().cpu() if isinstance(pts, torch.Tensor) else torch.tensor(pts, dtype=torch.float32)
        _fixed_inp = _fixed_tgt.unsqueeze(0).to(device).float()
    if is_dist():
        import torch.distributed as dist
        dist.barrier()

    if is_main():
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Config: {cfg}\n")
            f.write("epoch,train_loss,val_loss\n")

    def one_epoch(epoch: int) -> float:
        if is_dist(): train_sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        num_batches = 0

        for batch_idx, (pts, _, _) in enumerate(train_loader):
            points = pts.to(device).float()

            global_step = epoch * max(1, len(train_loader)) + batch_idx
            bn_mom = get_bn_momentum(global_step, cfg)
            base = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            base.apply(lambda m: _set_bn_momentum(m, bn_mom))

            optimizer.zero_grad(set_to_none=True)
            pred = model(points)
            cd, _ = chamfer_distance(pred, points)
            rep = repulsion_loss(pred, k=loss_cfg["repulsion_k"], h=loss_cfg["repulsion_h"])
            loss = cd + rep
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            num_batches += 1

            if is_main():
                print(f"Epoch {epoch + 1} Batch {batch_idx + 1}/{len(train_loader)}, CD: {cd.item():.4f}, Rep: {rep.item():.4f}")

        loss_tensor = torch.tensor([loss_sum, num_batches], dtype=torch.float32, device=device)
        all_reduce_mean(loss_tensor)
        return (loss_tensor[0] / torch.clamp_min(loss_tensor[1], 1.0)).item()

    @torch.no_grad()
    def evaluate(loader: DataLoader) -> float:
        model.eval()
        loss_sum = 0.0
        num_batches = 0
        for pts, _, _ in loader:
            points = pts.to(device).float()
            pred = model(points)
            cd, _ = chamfer_distance(pred, points)
            rep = repulsion_loss(pred, k=loss_cfg["repulsion_k"], h=loss_cfg["repulsion_h"])
            loss_sum += float((cd + rep).item())
            num_batches += 1

        loss_tensor = torch.tensor([loss_sum, num_batches], dtype=torch.float32, device=device)
        all_reduce_mean(loss_tensor)
        return (loss_tensor[0] / torch.clamp_min(loss_tensor[1], 1.0)).item()

    best_val = float("inf")
    epochs_without_improve = 0

    for epoch in range(start_epoch, tr["max_epochs"]):
        train_loss = one_epoch(epoch)
        val_loss = evaluate(val_loader)

        if is_main():
            print(f"Epoch {epoch + 1}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch + 1},{train_loss:.6f},{val_loss:.6f}\n")
            save_progress_plot(epoch, model, device, fig_dir)

        sched.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improve = 0
            if is_main():
                state_dict = (model.module.state_dict()
                              if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                              else model.state_dict())
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                }, best_path)
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= tr["early_stop_patience"]:
            if is_main():
                print(f"Early stopping at epoch {epoch + 1}")
            break

    test_loss = evaluate(test_loader)
    if is_main():
        print(f"Test Loss: {test_loss:.4f}")

    ddp_cleanup()


# -------------------- Spawn worker (used by multiprocessing.spawn) --------------------
def _spawn_worker(rank: int, world_size: int, cfg: dict):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    train_loop(rank, cfg)


# -------------------- Main --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file")
    parser.add_argument("--run-name", default=None, help="Override output.run_name from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    # Decide which GPUs to use BEFORE any torch.cuda.* call so we can still
    # narrow CUDA_VISIBLE_DEVICES (a CUDA context locks visibility).
    gpus_cfg = cfg["train_ae"].get("gpus", "auto")
    min_free_gib = float(cfg["train_ae"].get("min_free_gpu_mem_gib", 20))
    launched_by_torchrun = "LOCAL_RANK" in os.environ

    if gpus_cfg == "auto" and not launched_by_torchrun:
        selected, free = select_free_gpus(min_free_mib=int(min_free_gib * 1024))
        if free:
            free_str = ", ".join(f"cuda:{i}={free[i] / 1024:.1f}GiB" for i in range(len(free)))
            print(f"[GPU] Free memory per visible device: {free_str}")
        if selected and len(selected) < len(free):
            phys = apply_visible_devices(selected)
            print(f"[GPU] Selected physical GPU(s) {','.join(phys)} "
                  f"(>= {min_free_gib:.1f} GiB free); skipping the rest.")
        elif selected:
            print(f"[GPU] All {len(selected)} visible GPU(s) meet the {min_free_gib:.1f} GiB threshold.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_available = torch.cuda.device_count()
    n_gpus = n_available if gpus_cfg == "auto" else min(int(gpus_cfg), n_available)

    if "LOCAL_RANK" in os.environ:
        # Already launched by torchrun — let it manage ranks
        train_loop(rank=int(os.environ["LOCAL_RANK"]), cfg=cfg)
    elif n_gpus > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        torch.multiprocessing.spawn(_spawn_worker, args=(n_gpus, cfg), nprocs=n_gpus)
    else:
        train_loop(rank=0, cfg=cfg)
