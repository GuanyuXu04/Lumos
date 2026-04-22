"""
Train sweep (multiprocess, multi-GPU) with TWO CLI options only:
  --sweep {led,pd}
  --order {sage,natural,sage_reverse}

What it does
------------
- Fixed dataset split (train/val/test) saved once and reused across all runs.
- Cache X_full (optical_norm) and Z_full (AE encoder latent) once and reuse.
- For each K:
    * LED sweep: K=1..30, each step adds 5 contiguous features (one LED group).
    *  PD sweep: K=1..5, each step adds 30 features (one PD across all LEDs: every 5th feature).
  Repeat training 8 times with different fixed seeds (8 processes).
  Training is distributed on 4 GPUs with 2 processes per GPU.

- Each repeat saves the epoch checkpoint with minimum validation loss.
- After all K finished, aggregate TEST latent MSE mean/std per K and plot error bars.

Hardcoded (per your request)
----------------------------
DATA_PATH, BEST_AE_EP, STRIDE, EPOCHS, LR, BATCH_SIZE, repeats=8, GPUs=4, procs_per_gpu=2.
SAGE CSV paths:
  LED: experiments/L512/NP4096/sage_results/sage_led_importance.csv
   PD: experiments/L512/NP4096/sage_results/sage_pd_importance.csv

Output
------
experiments/L512/NP4096/sweep_mp_v2/
  cache/precomputed_XZ_stride3_splitseed56.pt
  split_indices.json
  sweep_{led|pd}/order_{order}/Kxx/seedYYYY/{best_pd2latent.pth, train_log.csv, result.json}
  aggregate_{sweep}_{order}.csv
  mse_vs_k_{sweep}_{order}.png
"""
import re
import os
import csv
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

from lumos.model import PD2Latent, PCAutoencoder
from lumos.data import WaveguideDataset


# ---------------------------- Hardcoded config ----------------------------
SEED_SPLIT = 56

DATA_PATH = "Data/Combined_Data/Combined_Data"
BEST_AE_EP = 9

NUM_LEDS = 30
NUM_PDS = 5
OPTICAL_DIM_FULL = 150  # 30*5

LATENT_DIM = 512
NUM_POINTS = 4096
TNET1 = False
TNET2 = False
STRIDE = 3

BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-4

REPEATS = 12
BASE_SEED = 1000  # run seeds = BASE_SEED + i

NUM_GPUS = 4
WORKERS_PER_GPU = 3  # training: 3 processes per GPU

SAGE_LED_CSV = "experiments/L512/NP4096/sage_results/sage_led_importance.csv"
SAGE_PD_CSV  = "experiments/L512/NP4096/sage_results/sage_pd_importance.csv"

ROOT_DIR = Path("experiments/L512/NP4096/sweep_mp")
CACHE_DIR = ROOT_DIR / "cache"
CACHE_PATH = CACHE_DIR / f"precomputed_XZ_stride{STRIDE}_splitseed{SEED_SPLIT}.pt"
SPLIT_PATH = ROOT_DIR / "split_indices.json"


# ---------------------------- CLI ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", choices=["led", "pd"], required=True, help="led: add LED groups (5 feats), pd: add PD groups (30 feats)")
    p.add_argument("--order", choices=["sage", "natural", "sage_reverse"], required=True, help="ordering of groups to add")
    return p.parse_args()


# ---------------------------- Utilities ----------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_split_indices(n_total: int, seed: int) -> Tuple[List[int], List[int], List[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def _detect_group_key(fieldnames: List[str]) -> str:
    # Prefer explicit names, otherwise any key containing "Group"
    for k in ["LED_Group", "PD_Group", "Group", "group", "feature_group", "Feature_Group"]:
        if k in fieldnames:
            return k
    for k in fieldnames:
        if "group" in k.lower():
            return k
    # fallback first column
    return fieldnames[0]


def load_rank_from_sage_csv(csv_path: str, expected_groups: int) -> List[int]:
    """
    Generic rank loader. Returns list of group ids (0-based) sorted by descending SAGE_Value.
    Accepts group names like 'LED_1', 'PD_2', or plain integers.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SAGE CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise ValueError(f"Empty CSV: {csv_path}")
        group_key = _detect_group_key(reader.fieldnames)
        # find sage value key
        sage_key = "SAGE_Value" if "SAGE_Value" in reader.fieldnames else None
        if sage_key is None:
            # pick first numeric-like column besides group
            cand = [k for k in reader.fieldnames if k != group_key]
            sage_key = cand[0]

    def parse_id(s: str) -> int:
        s = str(s).strip()
        # Extract last integer from string
        m = None
        for mm in re.finditer(r"\d+", s):
            m = mm
        if m:
            return int(m.group(0)) - 1  # assume 1-based in csv
        # if already integer-like
        return int(s)

    rows.sort(key=lambda r: float(r[sage_key]), reverse=True)
    rank = [parse_id(r[group_key]) for r in rows]

    if expected_groups is not None and len(rank) != expected_groups:
        print(f"[WARN] Expected {expected_groups} groups in {csv_path}, got {len(rank)}")
    return rank


def group_order(sweep: str, order: str) -> List[int]:
    """
    Returns ordered list of group IDs.
    - led groups: 0..29
    -  pd groups: 0..4
    """
    if sweep == "led":
        if order == "natural":
            return list(range(NUM_LEDS))
        sage_rank = load_rank_from_sage_csv(SAGE_LED_CSV, expected_groups=NUM_LEDS)
    else:
        if order == "natural":
            return list(range(NUM_PDS))
        sage_rank = load_rank_from_sage_csv(SAGE_PD_CSV, expected_groups=NUM_PDS)

    if order == "sage":
        return sage_rank
    elif order == "sage_reverse":
        return list(reversed(sage_rank))
    else:
        raise ValueError(f"Unknown order: {order}")

def describe_enabled_groups(sweep: str, ordered_groups: List[int], k: int) -> str:
    """Return a human-readable string of which groups are enabled for this K."""
    enabled = ordered_groups[:k]
    if sweep == "led":
        return " ".join([f"LED{gid+1:02d}" for gid in enabled])
    else:
        return " ".join([f"PD{gid+1}" for gid in enabled])

def enabled_feature_map(sweep: str, ordered_groups: List[int], k: int) -> Dict[int, List[int]]:
    """Optional: show mapping group_id -> feature indices."""
    mp = {}
    for gid in ordered_groups[:k]:
        mp[gid] = features_for_group(sweep, gid)
    return mp

def features_for_group(sweep: str, group_id: int) -> List[int]:
    """
    LED sweep: group_id=led, pick contiguous 5 features: [5*led .. 5*led+4]
    PD sweep : group_id=pd,  pick every-5th feature: [pd, pd+5, pd+10, ..., pd+145] (30 feats)
    """
    if sweep == "led":
        base = group_id * NUM_PDS
        return list(range(base, base + NUM_PDS))
    else:
        pd = group_id
        return [pd + NUM_PDS * led for led in range(NUM_LEDS)]


def feature_indices_for_topk_groups(sweep: str, ordered_groups: List[int], k: int) -> List[int]:
    idx: List[int] = []
    for gid in ordered_groups[:k]:
        idx.extend(features_for_group(sweep, gid))
    return idx


@torch.no_grad()
def precompute_XZ_if_needed(dataset: WaveguideDataset, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cache X_full (optical_norm) and Z_full (encoder latent) for the full dataset.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        ckpt = torch.load(CACHE_PATH, map_location="cpu")
        return ckpt["X_full"].to(torch.float32), ckpt["Z_full"].to(torch.float32)

    ae = PCAutoencoder(latent_dim=LATENT_DIM, num_points=NUM_POINTS, tnet1=TNET1, tnet2=TNET2).to(device)
    best_ae_path = f"experiments/L{LATENT_DIM}/NP{NUM_POINTS}/checkpoints/model_ep{BEST_AE_EP}.pth"
    best_ae = torch.load(best_ae_path, map_location=device)
    ae.load_state_dict(best_ae["model_state_dict"])
    ae.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    optical_list, latent_list = [], []

    for pts, _opt_raw, opt_norm in loader:
        pts = pts.to(device).float()
        z = ae.encoder(pts).cpu()
        optical_list.append(opt_norm.cpu())
        latent_list.append(z)

    X_full = torch.cat(optical_list, dim=0).to(torch.float32)
    Z_full = torch.cat(latent_list, dim=0).to(torch.float32)

    torch.save({"X_full": X_full, "Z_full": Z_full}, CACHE_PATH)
    return X_full, Z_full


def latent_mse_on_test(model: PD2Latent, X_test: torch.Tensor, Z_test: torch.Tensor, device: torch.device) -> float:
    ds = TensorDataset(X_test, Z_test)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    mse_sum, n = 0.0, 0
    with torch.no_grad():
        for bx, bz in loader:
            bx = bx.to(device, non_blocking=True).float()
            bz = bz.to(device, non_blocking=True).float()
            pred = model(bx)
            per = torch.mean((pred - bz) ** 2, dim=1)
            mse_sum += per.sum().item()
            n += bx.size(0)
    return mse_sum / max(1, n)


def train_one_repeat(
    *,
    k: int,
    worker_id: int,
    feature_idx: List[int],
    X_full: torch.Tensor,
    Z_full: torch.Tensor,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    seed_run: int,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Train a PD2Latent for one (K, seed). Save best epoch checkpoint by min val loss.
    Compute TEST latent MSE using the best checkpoint.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_pd2latent.pth"
    result_path = out_dir / "result.json"
    log_path = out_dir / "train_log.csv"

    # Skip if already done
    if best_ckpt_path.exists() and result_path.exists():
        with open(result_path, "r") as f:
            return json.load(f)

    set_all_seeds(seed_run)

    in_dim = len(feature_idx)

    X_train = X_full[train_idx][:, feature_idx]
    Z_train = Z_full[train_idx]
    X_val = X_full[val_idx][:, feature_idx]
    Z_val = Z_full[val_idx]
    X_test = X_full[test_idx][:, feature_idx]
    Z_test = Z_full[test_idx]

    train_ds = TensorDataset(X_train, Z_train)
    val_ds = TensorDataset(X_val, Z_val)

    g = torch.Generator().manual_seed(seed_run)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0, generator=g, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = PD2Latent(in_features=in_dim, out_features=LATENT_DIM).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    def _mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - gt) ** 2)

    best_val = float("inf")
    best_epoch = -1

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "lr"])

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for bx, bz in train_loader:
            bx = bx.to(device, non_blocking=True).float()
            bz = bz.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = _mse(pred, bz)
            loss.backward()
            optimizer.step()
            tr_sum += loss.item() * bx.size(0)
            tr_n += bx.size(0)
        tr_loss = tr_sum / max(1, tr_n)

        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for bx, bz in val_loader:
                bx = bx.to(device, non_blocking=True).float()
                bz = bz.to(device, non_blocking=True).float()
                pred = model(bx)
                loss = _mse(pred, bz)
                va_sum += loss.item() * bx.size(0)
                va_n += bx.size(0)
        va_loss = va_sum / max(1, va_n)

        sched.step()
        lr_now = optimizer.param_groups[0]["lr"]
        if worker_id == 0:
            print(f"[K={k:02d}] Epoch {ep:03d}/{EPOCHS} | train {tr_loss:.6f} | val {va_loss:.6f} | lr {lr_now:.2e}", flush=True)

        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, f"{tr_loss:.10f}", f"{va_loss:.10f}", f"{lr_now:.6e}"])

        if va_loss < best_val:
            best_val = float(va_loss)
            best_epoch = ep
            torch.save(
                {
                    "k": k,
                    "seed": seed_run,
                    "in_dim": in_dim,
                    "feature_idx": feature_idx,
                    "epoch": ep,
                    "val_loss": best_val,
                    "model_state_dict": model.state_dict(),
                },
                best_ckpt_path,
            )

    # Test MSE using best checkpoint
    best = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(best["model_state_dict"])
    model.to(device)
    test_mse = latent_mse_on_test(model, X_test, Z_test, device)

    result = {
        "k": k,
        "seed": seed_run,
        "in_dim": in_dim,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "test_latent_mse": float(test_mse),
        "checkpoint": str(best_ckpt_path),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------- Worker ----------------------------
def worker_loop(worker_id: int, gpu_id: int, task_q: mp.Queue, result_q: mp.Queue, 
                X_full: torch.Tensor, Z_full: torch.Tensor, split_path: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    # LOAD SPLIT ONCE PER WORKER
    with open(split_path, "r") as f:
        sp = json.load(f)
    train_idx, val_idx, test_idx = sp["train_idx"], sp["val_idx"], sp["test_idx"]

    while True:
        task = task_q.get()
        if task is None:
            break
        try:
            t0 = time.time()
            res = train_one_repeat(
                k=task["k"],
                worker_id=worker_id,
                feature_idx=task["feature_idx"],
                X_full=X_full, # These are now shared memory tensors
                Z_full=Z_full,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                seed_run=task["seed"],
                device=device,
                out_dir=Path(task["out_dir"]),
            )
            res["worker_id"] = worker_id
            res["gpu_id"] = gpu_id
            res["time_sec"] = float(time.time() - t0)
            result_q.put(res)
        except Exception as e:
            result_q.put({"k": task["k"], "seed": task["seed"], "error": repr(e)})


def main() -> None:
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = WaveguideDataset(DATA_PATH, stride=STRIDE)
    n_total = len(dataset)

    # Fixed split
    if SPLIT_PATH.exists():
        with open(SPLIT_PATH, "r") as f:
            sp = json.load(f)
        if len(sp["train_idx"]) + len(sp["val_idx"]) + len(sp["test_idx"]) != n_total:
            raise RuntimeError("Existing split_indices.json mismatch with dataset length.")
        print(f"[Split] Loaded: {SPLIT_PATH}")
    else:
        train_idx, val_idx, test_idx = make_split_indices(n_total, SEED_SPLIT)
        with open(SPLIT_PATH, "w") as f:
            json.dump({"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}, f)
        print(f"[Split] Saved: {SPLIT_PATH}")

    # Cache X/Z once (on cuda:0 if available)
    if not CACHE_PATH.exists():
        device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _ = precompute_XZ_if_needed(dataset, device0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[Cache] Saved: {CACHE_PATH}")
    else:
        print(f"[Cache] Found: {CACHE_PATH}")
    
    # LOAD AND SHARE MEMORY (Add this part)
    print(f"[Cache] Loading into Shared Memory...")
    ckpt = torch.load(CACHE_PATH, map_location="cpu")
    X_full = ckpt["X_full"].to(torch.float32).share_memory_()
    Z_full = ckpt["Z_full"].to(torch.float32).share_memory_()

    # Prepare order + sweep
    groups = group_order(args.sweep, args.order)
    sweep_dir = ROOT_DIR / f"sweep_{args.sweep}" / f"order_{args.order}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / "group_order.json", "w") as f:
        json.dump({"sweep": args.sweep, "order": args.order, "groups": groups}, f, indent=2)

    # seeds
    seeds = [BASE_SEED + i for i in range(REPEATS)]
    with open(sweep_dir / "run_seeds.json", "w") as f:
        json.dump({"base_seed": BASE_SEED, "seeds": seeds}, f, indent=2)

    # Start workers (8 procs over 4 GPUs, 2 per GPU)
    total_workers = NUM_GPUS * WORKERS_PER_GPU
    task_q, result_q = mp.Queue(), mp.Queue()
    workers = []
    for wid in range(total_workers):
        gpu_id = wid % NUM_GPUS
        p = mp.Process(
            target=worker_loop, 
            args=(wid, gpu_id, task_q, result_q, X_full, Z_full, str(SPLIT_PATH)), 
            daemon=True
        )
        p.start()
        workers.append(p)

    # K range
    if args.sweep == "led":
        k_max = NUM_LEDS
    else:
        k_max = NUM_PDS

    # 1. Dispatch ALL tasks for ALL K immediately
    total_tasks = 0
    for k in range(1, k_max + 1):
        enabled_str = describe_enabled_groups(args.sweep, groups, k)
        feat_idx = feature_indices_for_topk_groups(args.sweep, groups, k)
        print(f"\n[K={k:02d}] Enabled groups ({args.order}): {enabled_str}")
        print(f"[K={k:02d}] in_dim = {len(feat_idx)} | feature_idx = {feat_idx}")
        for s in seeds:
            out_dir = sweep_dir / f"K{k:02d}" / f"seed{s}"
            task_q.put({"k": k, "feature_idx": feat_idx, "seed": s, "out_dir": str(out_dir)})
            total_tasks += 1

    # 2. Collect results as they finish (order doesn't matter)
    all_results = []
    print(f"Dispatched {total_tasks} tasks. Collecting results...")
    for i in range(total_tasks):
        res = result_q.get()
        if "error" in res:
            print(f"Error in task: {res}")
        else:
            all_results.append(res)
            #print(f"[{i+1}/{total_tasks}] Finished K={res['k']} Seed={res['seed']} on GPU {res['gpu_id']}")

    # 3. Aggregate results for the CSV/Plot
    agg_rows = []
    for k in range(1, k_max + 1):
        k_results = [r for r in all_results if r["k"] == k]
        mses = np.array([r["test_latent_mse"] for r in k_results])
        
        row = {
            "K": k, 
            "in_dim": len(feature_indices_for_topk_groups(args.sweep, groups, k)), 
            "test_mse_mean": float(mses.mean()), 
            "test_mse_std": float(mses.std(ddof=1)) if len(mses) > 1 else 0.0
        }
        agg_rows.append(row)

    # Stop workers
    for _ in workers:
        task_q.put(None)
    for p in workers:
        p.join()

    # Save aggregate + plot
    agg_csv = ROOT_DIR / f"aggregate_{args.sweep}_{args.order}.csv"
    with open(agg_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)

    import matplotlib.pyplot as plt
    ks = [r["K"] for r in agg_rows]
    means = [r["test_mse_mean"] for r in agg_rows]
    stds = [r["test_mse_std"] for r in agg_rows]

    plt.figure()
    plt.errorbar(ks, means, yerr=stds, fmt='-o', capsize=4)
    plt.xlabel("K")
    plt.ylabel("Test latent MSE")
    plt.title(f"Test latent MSE vs K ({args.sweep} sweep, order={args.order}, repeats={REPEATS})")
    plt.grid(True, alpha=0.3)
    fig_path = ROOT_DIR / f"mse_vs_k_{args.sweep}_{args.order}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nDone.")
    print("Aggregate:", agg_csv)
    print("Figure   :", fig_path)


if __name__ == "__main__":
    main()
