import os
import gc
import math
import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import PCAutoencoder
from data import WaveguideDataset

# -------------------- DDP utils --------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0

def is_main() -> bool:
    return get_rank() == 0

def ddp_setup():
    # torchrun sets LOCAL_RANK, RANK, WORLD_SIZE
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank

def ddp_cleanup():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()

def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """In-place all-reduce mean over world."""
    if is_dist():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x

# -------------------- Repro / perf --------------------
torch.backends.cudnn.benchmark = True
random.seed(56)
np.random.seed(56)
torch.manual_seed(56)

# -------------------- Hyperparams --------------------
BATCH_SIZE = 64
MAX_EPOCH = 25
BASE_LEARNING_RATE = 0.001
DECAY_STEP = 100000
DECAY_RATE = 0.2
BN_INIT_DECAY = 0.9
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = 200000
BN_DECAY_CLIP = 0.5

LATENT_DIM = 512
NUM_POINTS = 4096
TNET1 = False
TNET2 = False

DATA_PATH = "Data/Combined_Data"
FIG_DIR = "checkpoints/4096_512_figs"
CKPT_DIR = "checkpoints/4096_512_checkpoints"
BEST_PATH = os.path.join(CKPT_DIR, "best_model.pth")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
LOG_PATH = os.path.join(FIG_DIR, "record_log.txt")

gc.collect()
torch.cuda.empty_cache()

# -------------------- Losses --------------------
def chamfer_distance(S1: torch.Tensor, S2: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if S1.dim() == 2: S1 = S1.unsqueeze(0)
    if S2.dim() == 2: S2 = S2.unsqueeze(0)
    d2 = torch.cdist(S1, S2, p=2).pow(2)         # (B,P,Q)
    d1_min, _ = d2.min(dim=2)                    # (B,P)
    d2_min, _ = d2.min(dim=1)                    # (B,Q)
    loss = (d1_min.sum(dim=1) + d2_min.sum(dim=1)).mean()
    return loss, (d1_min, d2_min)

def repulsion_loss(pred: torch.Tensor, k: int = 10, h: float = 0.5) -> torch.Tensor:
    B, P, _ = pred.shape
    if P <= 1: return pred.new_tensor(0.0)
    k_eff = min(k, P - 1)
    d = torch.cdist(pred, pred, p=2)
    d2 = d.pow(2)
    _, idx = d.topk(k=k_eff + 1, largest=False)
    nn_d2 = torch.gather(d2, 2, idx[:, :, 1:])
    ker = torch.exp(-nn_d2 / (h * h))
    return ker.sum(dim=(1, 2)).mean() * 100.0

def get_lr_for_step(step: int) -> float:
    return BASE_LEARNING_RATE * (DECAY_RATE ** (step // DECAY_STEP))

def get_bn_momentum(step: int, batch_size: int = BATCH_SIZE) -> float:
    bn_momentum = BN_INIT_DECAY * (BN_DECAY_DECAY_RATE ** ((step * batch_size) // BN_DECAY_DECAY_STEP))
    bn_decay = min(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def _set_bn_momentum(module, momentum: float):
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        module.momentum = momentum

# -------------------- Plot helpers (rank-0 only) --------------------
def _set_axes_equal(ax1, ax2):
    x_limits = ax1.get_xlim3d(); y_limits = ax1.get_ylim3d(); z_limits = ax1.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0]); y_range = abs(y_limits[1] - y_limits[0]); z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    xm = sum(x_limits)/2; ym = sum(y_limits)/2; zm = sum(z_limits)/2
    for ax in (ax1, ax2):
        ax.set_xlim3d([xm-max_range/2, xm+max_range/2])
        ax.set_ylim3d([ym-max_range/2, ym+max_range/2])
        ax.set_zlim3d([zm-max_range/2, zm+max_range/2])

_fixed_inp = None
_fixed_tgt = None

@torch.no_grad()
def save_progress_plot(epoch: int, model, device, out_dir=FIG_DIR, prefix="progress_fixed"):
    if not is_main(): return
    global _fixed_inp, _fixed_tgt
    assert _fixed_inp is not None
    model.eval()
    pred = model(_fixed_inp).squeeze(0).detach().cpu()
    tgt  = _fixed_tgt

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(tgt[:,0], tgt[:,1], tgt[:,2], s=1, c=tgt[:,2], cmap="viridis")
    ax1.set_title("Original"); ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(pred[:,0], pred[:,1], pred[:,2], s=1, c=pred[:,2], cmap="viridis")
    ax2.set_title("Reconstructed"); ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    _set_axes_equal(ax1, ax2)

    fig.suptitle(f"Epoch {epoch+1}")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{prefix}_ep{epoch+1:03d}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

# -------------------- Train / Eval --------------------
def train_loop(rank: int):
    local_rank = rank  # for clarity

    # DDP init
    if "LOCAL_RANK" in os.environ:
        local_rank = ddp_setup()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Dataset + samplers
    dataset = WaveguideDataset(DATA_PATH, stride=3)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val   = int(0.1 * n_total)
    n_test  = n_total - n_train - n_val

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(56)
    )

    train_sampler = DistributedSampler(train_set, shuffle=True) if is_dist() else None
    val_sampler   = DistributedSampler(val_set, shuffle=False) if is_dist() else None
    test_sampler  = DistributedSampler(test_set, shuffle=False) if is_dist() else None

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        sampler=val_sampler, num_workers=4, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        sampler=test_sampler, num_workers=4, pin_memory=True, drop_last=False,
    )

    # Model/opt/sched
    model = PCAutoencoder(latent_dim=LATENT_DIM, num_points=NUM_POINTS, tnet1=TNET1, tnet2=TNET2).to(device)

    # Resume logic (works with non-DDP and DDP checkpoints)
    optimizer = optim.Adam(model.parameters(), lr=BASE_LEARNING_RATE)
    sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=3)

    start_epoch = 0
    if os.path.exists(BEST_PATH):
        ckpt = torch.load(BEST_PATH, map_location=device)
        state = ckpt["model_state_dict"]
        # Handle plain vs DistributedDataParallel keys
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        if is_main():
            print(f"Loaded best model from epoch {ckpt['epoch'] + 1} (val {ckpt['val_loss']:.4f})")

    if is_dist():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank, 
            find_unused_parameters=True,
            static_graph=False,
        )

    # Fixed example for progress plots (rank-0 only input, then broadcast to all for shape)
    global _fixed_inp, _fixed_tgt
    if is_main():
        item = test_set[42]
        pts = item[0] if isinstance(item, (list, tuple)) else item
        _fixed_tgt = (pts.detach().cpu() if isinstance(pts, torch.Tensor)
                      else torch.tensor(pts, dtype=torch.float32))
        _fixed_inp = _fixed_tgt.unsqueeze(0).to(device).float()
    if is_dist():
        # Ensure all ranks wait until fixed data prepared on rank-0
        dist.barrier()

    # Log file header (rank-0)
    if is_main():
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("Configuration:\n")
            f.write(f"ReduceLROnPlateau: factor=0.2, patience=3\n")
            f.write(f"repulsion_loss: k=10, h=0.5\n")
            f.write("epoch,train_loss,val_loss\n")

    def one_epoch(epoch: int) -> float:
        if is_dist(): train_sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        num_batches = 0

        for batch_idx, (pts, _, _) in enumerate(train_loader):
            points = pts.to(device).float()
            target = points

            global_step = epoch * max(1, len(train_loader)) + batch_idx
            bn_mom = get_bn_momentum(global_step)
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model)\
                .apply(lambda m: _set_bn_momentum(m, bn_mom))

            optimizer.zero_grad(set_to_none=True)
            pred = model(points)
            cd, _ = chamfer_distance(pred, target)
            rep = repulsion_loss(pred)
            loss = cd + rep
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            num_batches += 1

            if is_main():
                print(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)}, CD: {cd.item():.4f}, Rep: {rep.item():.4f}")

        # Average over workers
        loss_tensor = torch.tensor([loss_sum, num_batches], dtype=torch.float32, device=device)
        all_reduce_mean(loss_tensor)
        avg = (loss_tensor[0] / torch.clamp_min(loss_tensor[1], 1.0)).item()
        return avg

    @torch.no_grad()
    def evaluate(loader: DataLoader) -> float:
        model.eval()
        loss_sum = 0.0
        num_batches = 0
        for (pts, _, _) in loader:
            points = pts.to(device).float()
            target = points
            pred = model(points)
            cd, _ = chamfer_distance(pred, target)
            rep = repulsion_loss(pred)
            loss = cd + rep
            loss_sum += float(loss.item())
            num_batches += 1

        # All-reduce to get global mean
        loss_tensor = torch.tensor([loss_sum, num_batches], dtype=torch.float32, device=device)
        all_reduce_mean(loss_tensor)
        return (loss_tensor[0] / torch.clamp_min(loss_tensor[1], 1.0)).item()

    # -------------------- Training --------------------
    best_val = float("inf")
    best_snapshot = None
    epochs_without_improve = 0

    for epoch in range(start_epoch, MAX_EPOCH):
        train_loss = one_epoch(epoch)
        val_loss = evaluate(val_loader)

        if is_main():
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f}\n")
            save_progress_plot(epoch, model, device)

        sched.step(val_loss)

        # Save best (rank-0)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_without_improve = 0
            if is_main():
                state = {
                    "epoch": epoch,
                    "model_state_dict": (model.module.state_dict()
                                         if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                                         else model.state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                }
                torch.save(state, BEST_PATH)
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= 20:
            if is_main(): print(f"Early stopping at epoch {epoch+1}")
            break

    # -------------------- Test --------------------
    test_loss = evaluate(test_loader)
    if is_main():
        print(f"Test Loss: {test_loss:.4f}")

    ddp_cleanup()

# -------------------- Main --------------------
if __name__ == "__main__":
    # Use torchrun to spawn 4 processes; each will call this once.
    train_loop(rank=int(os.environ.get("LOCAL_RANK", 0)))
