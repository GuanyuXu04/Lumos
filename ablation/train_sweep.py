"""
Optical feature importance sweep — training (multiprocess, multi-GPU).

For each K from 1 to num_groups, trains PD2Latent using the top-K most important
optical feature groups, repeated N times with different seeds.

Usage:
    python ablation/train_sweep.py --sweep {led,pd} --order {sage,natural,sage_reverse}

Outputs (to {base_dir}/{run_name}/sweep/):
    split_indices.json
    cache/precomputed_XZ.pt
    sweep_{led|pd}/order_{order}/K{k:02d}/seed{s}/{best_pd2latent.pth, train_log.csv, result.json}
    aggregate_{sweep}_{order}.csv
    mse_vs_k_{sweep}_{order}.png
"""
import re
import csv
import json
import random
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader, TensorDataset

from lumos.data import WaveguideDataset
from lumos.model import PD2Latent, PCAutoencoder


WORKERS_PER_GPU = 3  # parallel training workers per GPU


# ----------------------------- CLI + Config -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--sweep", choices=["led", "pd"], required=True)
    p.add_argument("--order", choices=["sage", "natural", "sage_reverse"], required=True)
    return p.parse_args()


def resolve_num_gpus(cfg: dict) -> int:
    n_available = torch.cuda.device_count()
    gpus_cfg = cfg["train_ae"].get("gpus", "auto")
    if gpus_cfg == "auto" or n_available == 0:
        return max(1, n_available)
    return max(1, min(int(gpus_cfg), n_available))


# ----------------------------- Group helpers -----------------------------

def load_sage_rank(csv_path: str, n_groups: int) -> List[int]:
    """Load 0-based group IDs sorted by descending SAGE importance."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        group_key = next(k for k in reader.fieldnames if "Group" in k)
    rows.sort(key=lambda r: float(r["SAGE_Value"]), reverse=True)
    rank = [int(re.findall(r"\d+", r[group_key])[-1]) - 1 for r in rows]  # CSV is 1-based
    if len(rank) != n_groups:
        print(f"[WARN] Expected {n_groups} groups in {csv_path}, got {len(rank)}")
    return rank


def group_order(sweep: str, order: str, num_leds: int, num_pds: int, sage_dir: Path) -> List[int]:
    if order == "natural":
        return list(range(num_leds if sweep == "led" else num_pds))
    csv_name = "sage_led_importance.csv" if sweep == "led" else "sage_pd_importance.csv"
    n = num_leds if sweep == "led" else num_pds
    rank = load_sage_rank(str(sage_dir / csv_name), n)
    return rank if order == "sage" else list(reversed(rank))


def features_for_group(sweep: str, group_id: int, num_leds: int, num_pds: int) -> List[int]:
    if sweep == "led":
        base = group_id * num_pds
        return list(range(base, base + num_pds))
    else:
        return [group_id + num_pds * led for led in range(num_leds)]


def feature_indices(sweep: str, groups: List[int], k: int, num_leds: int, num_pds: int) -> List[int]:
    idx: List[int] = []
    for gid in groups[:k]:
        idx.extend(features_for_group(sweep, gid, num_leds, num_pds))
    return idx


# ----------------------------- Data helpers -----------------------------

def make_split(n_total: int, train_split: float, val_split: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    n_train = int(train_split * n_total)
    n_val   = int(val_split * n_total)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


@torch.no_grad()
def precompute_XZ(dataset, ae, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    X_list, Z_list = [], []
    for pts, _, opt_norm in loader:
        Z_list.append(ae.encoder(pts.to(device).float()).cpu())
        X_list.append(opt_norm.cpu())
    return torch.cat(X_list).float(), torch.cat(Z_list).float()


def latent_mse(model, X: torch.Tensor, Z: torch.Tensor, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(TensorDataset(X, Z), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for bx, bz in loader:
            pred = model(bx.to(device).float())
            total += ((pred - bz.to(device).float()) ** 2).mean(dim=1).sum().item()
            n += bx.size(0)
    return total / max(1, n)


# ----------------------------- Training -----------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(*, k: int, worker_id: int, feature_idx: List[int],
              X_full: torch.Tensor, Z_full: torch.Tensor,
              train_idx: List[int], val_idx: List[int], test_idx: List[int],
              seed: int, device: torch.device, out_dir: Path,
              batch_size: int, epochs: int, lr: float) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path   = out_dir / "best_pd2latent.pth"
    result_path = out_dir / "result.json"
    log_path    = out_dir / "train_log.csv"

    if ckpt_path.exists() and result_path.exists():
        with open(result_path) as f:
            return json.load(f)

    set_all_seeds(seed)
    in_dim = len(feature_idx)
    latent_dim = Z_full.shape[1]

    X_tr  = X_full[train_idx][:, feature_idx];  Z_tr  = Z_full[train_idx]
    X_val = X_full[val_idx][:, feature_idx];    Z_val = Z_full[val_idx]
    X_te  = X_full[test_idx][:, feature_idx];   Z_te  = Z_full[test_idx]

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(X_tr, Z_tr), batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0, generator=g, pin_memory=True)
    val_loader   = DataLoader(TensorDataset(X_val, Z_val), batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    model     = PD2Latent(in_features=in_dim, out_features=latent_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    sched     = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_val, best_epoch = float("inf"), -1

    for ep in range(1, epochs + 1):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for bx, bz in train_loader:
            bx, bz = bx.to(device).float(), bz.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(bx) - bz) ** 2).mean()
            loss.backward()
            optimizer.step()
            tr_sum += loss.item() * bx.size(0)
            tr_n += bx.size(0)
        tr_loss = tr_sum / max(1, tr_n)

        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for bx, bz in val_loader:
                bx, bz = bx.to(device).float(), bz.to(device).float()
                va_sum += ((model(bx) - bz) ** 2).mean().item() * bx.size(0)
                va_n += bx.size(0)
        va_loss = va_sum / max(1, va_n)
        sched.step()
        lr_now = optimizer.param_groups[0]["lr"]

        if worker_id == 0:
            print(f"[K={k:02d}] Ep {ep:03d}/{epochs} | tr {tr_loss:.6f} | val {va_loss:.6f} | lr {lr_now:.2e}", flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tr_loss:.8f}", f"{va_loss:.8f}", f"{lr_now:.6e}"])

        if va_loss < best_val:
            best_val, best_epoch = va_loss, ep
            torch.save({
                "k": k, "seed": seed, "in_dim": in_dim, "feature_idx": feature_idx,
                "epoch": ep, "val_loss": best_val, "model_state_dict": model.state_dict(),
            }, ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state_dict"])
    test_mse = latent_mse(model.to(device), X_te, Z_te, batch_size, device)

    result = {
        "k": k, "seed": seed, "in_dim": in_dim,
        "best_epoch": best_epoch, "best_val_loss": float(best_val),
        "test_latent_mse": float(test_mse), "checkpoint": str(ckpt_path),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


# ----------------------------- Worker -----------------------------

def worker_loop(worker_id: int, gpu_id: int, task_q: mp.Queue, result_q: mp.Queue,
                X_full: torch.Tensor, Z_full: torch.Tensor, split_path: str) -> None:
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    with open(split_path) as f:
        sp = json.load(f)
    train_idx, val_idx, test_idx = sp["train_idx"], sp["val_idx"], sp["test_idx"]

    while True:
        task = task_q.get()
        if task is None:
            break
        try:
            t0 = time.time()
            res = train_one(
                k=task["k"], worker_id=worker_id, feature_idx=task["feature_idx"],
                X_full=X_full, Z_full=Z_full,
                train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                seed=task["seed"], device=device, out_dir=Path(task["out_dir"]),
                batch_size=task["batch_size"], epochs=task["epochs"], lr=task["lr"],
            )
            result_q.put({**res, "worker_id": worker_id, "gpu_id": gpu_id, "time_sec": time.time() - t0})
        except Exception as e:
            result_q.put({"k": task["k"], "seed": task["seed"], "error": repr(e)})


# ----------------------------- Main -----------------------------

def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.run_name:
        cfg["output"]["run_name"] = args.run_name

    seed      = cfg["seed"]
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    abl_cfg   = cfg["ablation"]
    out_cfg   = cfg["output"]

    run_dir    = Path(out_cfg["base_dir"]) / out_cfg["run_name"]
    sweep_root = run_dir / "sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)
    split_path = sweep_root / "split_indices.json"
    cache_path = sweep_root / "cache" / f"precomputed_XZ_stride{data_cfg['stride']}_seed{seed}.pt"
    sage_dir   = run_dir / "sage_results"
    ae_path    = run_dir / "checkpoints" / "best_model.pth"

    num_leds   = abl_cfg["num_leds"]
    num_pds    = model_cfg["optical_dim"] // num_leds
    repeats    = abl_cfg["repeats"]
    base_seed  = abl_cfg["base_seed"]
    batch_size = abl_cfg["batch_size"]
    epochs     = abl_cfg["epochs"]
    lr         = float(abl_cfg["lr"])
    seeds      = [base_seed + i for i in range(repeats)]

    num_gpus      = resolve_num_gpus(cfg)
    total_workers = num_gpus * WORKERS_PER_GPU
    device_str    = f"{num_gpus} GPU(s)" if torch.cuda.is_available() else "CPU"
    print(f"Using {device_str}, {total_workers} workers")

    # Dataset + split
    dataset = WaveguideDataset(data_cfg["path"], stride=data_cfg["stride"])
    if split_path.exists():
        with open(split_path) as f:
            sp = json.load(f)
        if sum(len(sp[k]) for k in ("train_idx", "val_idx", "test_idx")) != len(dataset):
            raise RuntimeError("Existing split_indices.json does not match dataset size.")
        print(f"[Split] Loaded: {split_path}")
    else:
        ti, vi, tei = make_split(len(dataset), data_cfg["train_split"], data_cfg["val_split"], seed)
        with open(split_path, "w") as f:
            json.dump({"train_idx": ti, "val_idx": vi, "test_idx": tei}, f)
        print(f"[Split] Saved: {split_path}")

    # Cache X/Z
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"[Cache] Found: {cache_path}")
    else:
        device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ae = PCAutoencoder(latent_dim=model_cfg["latent_dim"], num_points=model_cfg["num_points"],
                           tnet1=model_cfg["tnet1"], tnet2=model_cfg["tnet2"]).to(device0)
        ae.load_state_dict(torch.load(ae_path, map_location=device0)["model_state_dict"])
        ae.eval()
        X_full, Z_full = precompute_XZ(dataset, ae, batch_size, device0)
        torch.save({"X_full": X_full, "Z_full": Z_full}, cache_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[Cache] Saved: {cache_path}")

    print("[Cache] Loading into shared memory...")
    cached = torch.load(cache_path, map_location="cpu")
    X_full = cached["X_full"].to(torch.float32).share_memory_()
    Z_full = cached["Z_full"].to(torch.float32).share_memory_()

    # Group ordering
    k_max     = num_leds if args.sweep == "led" else num_pds
    groups    = group_order(args.sweep, args.order, num_leds, num_pds, sage_dir)
    sweep_dir = sweep_root / f"sweep_{args.sweep}" / f"order_{args.order}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / "group_order.json", "w") as f:
        json.dump({"sweep": args.sweep, "order": args.order, "groups": groups}, f, indent=2)

    # Start workers
    task_q, result_q = mp.Queue(), mp.Queue()
    workers = []
    for wid in range(total_workers):
        p = mp.Process(target=worker_loop,
                       args=(wid, wid % num_gpus, task_q, result_q, X_full, Z_full, str(split_path)),
                       daemon=True)
        p.start()
        workers.append(p)

    # Dispatch all tasks upfront
    total_tasks = 0
    for k in range(1, k_max + 1):
        feat_idx = feature_indices(args.sweep, groups, k, num_leds, num_pds)
        for s in seeds:
            task_q.put({
                "k": k, "feature_idx": feat_idx, "seed": s,
                "out_dir": str(sweep_dir / f"K{k:02d}" / f"seed{s}"),
                "batch_size": batch_size, "epochs": epochs, "lr": lr,
            })
            total_tasks += 1

    # Collect results
    all_results = []
    print(f"Dispatched {total_tasks} tasks. Collecting...")
    for _ in range(total_tasks):
        res = result_q.get()
        if "error" in res:
            print(f"  ERROR: {res}")
        else:
            all_results.append(res)

    # Stop workers
    for _ in workers:
        task_q.put(None)
    for p in workers:
        p.join()

    # Aggregate per K
    agg_rows = []
    for k in range(1, k_max + 1):
        mses = np.array([r["test_latent_mse"] for r in all_results if r["k"] == k])
        if len(mses) == 0:
            print(f"[WARN] No results for K={k}")
            continue
        agg_rows.append({
            "K": k,
            "in_dim": len(feature_indices(args.sweep, groups, k, num_leds, num_pds)),
            "test_mse_mean": float(mses.mean()),
            "test_mse_std":  float(mses.std(ddof=1)) if len(mses) > 1 else 0.0,
        })

    agg_csv = sweep_root / f"aggregate_{args.sweep}_{args.order}.csv"
    with open(agg_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)

    plt.figure()
    plt.errorbar([r["K"] for r in agg_rows], [r["test_mse_mean"] for r in agg_rows],
                 yerr=[r["test_mse_std"] for r in agg_rows], fmt="-o", capsize=4)
    plt.xlabel("K")
    plt.ylabel("Test latent MSE")
    plt.title(f"Test latent MSE vs K ({args.sweep} sweep, {args.order}, n={repeats})")
    plt.grid(True, alpha=0.3)
    fig_path = sweep_root / f"mse_vs_k_{args.sweep}_{args.order}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nDone.")
    print("Aggregate:", agg_csv)
    print("Figure   :", fig_path)


if __name__ == "__main__":
    main()
