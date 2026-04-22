
"""
Chamfer evaluation (multiprocess, multi-GPU) with TWO CLI options only:
  --sweep {led,pd}
  --order {sage,natural,sage_reverse}

This script evaluates TEST Chamfer Distance for the trained runs produced by
train_optical_sweep_mp_v2.py, for each (K, seed).

Key constraint (hardcoded)
--------------------------
Chamfer is heavy, so we run ONLY ONE process per GPU.
We have 4 GPUs => 4 worker processes. With 8 repeats, tasks naturally complete in 2 "rounds".

Outputs
-------
experiments/L512/NP4096/sweep_mp/
  chamfer_{sweep}_{order}.csv            (per-run results)
  aggregate_chamfer_{sweep}_{order}.csv  (mean/std per K)
  chamfer_vs_k_{sweep}_{order}.png
"""

import os
import csv
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

from lumos.model import PD2Latent, PCAutoencoder
from lumos.data import WaveguideDataset
from lumos.metrics import chamfer_distance_batched


# ---------------------------- Hardcoded config ----------------------------
SEED_SPLIT = 56

DATA_PATH = "Data/Combined_Data/Combined_Data"
BEST_AE_EP = 9

NUM_LEDS = 30
NUM_PDS = 5

LATENT_DIM = 512
NUM_POINTS = 4096
TNET1 = False
TNET2 = False
STRIDE = 3

# Chamfer eval batch size: keep smaller than training
BATCH_SIZE = 32

REPEATS = 12
BASE_SEED = 1000

NUM_GPUS = 4
WORKERS_PER_GPU = 1  # IMPORTANT: only one chamfer process per GPU

ROOT_DIR = Path("experiments/L512/NP4096/sweep_mp")
SPLIT_PATH = ROOT_DIR / "split_indices.json"


# ---------------------------- CLI (only two options) ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", choices=["led", "pd"], required=True)
    p.add_argument("--order", choices=["sage", "natural", "sage_reverse"], required=True)
    return p.parse_args()


# ---------------------------- Worker ----------------------------
@torch.no_grad()
def eval_one_run(
    *,
    ckpt_path: Path,
    feature_idx: List[int],
    test_subset: Subset,
    ae: PCAutoencoder,
    device: torch.device,
    worker_id: int = 0,
) -> float:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_dim = int(ckpt["in_dim"])

    model = PD2Latent(in_features=in_dim, out_features=LATENT_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ae.eval()

    loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    cd_sum = 0.0
    n = 0
    for points_gt, _opt_raw, opt_norm in loader:
        points_gt = points_gt.to(device, non_blocking=True).float()
        x = opt_norm[:, feature_idx].to(device, non_blocking=True).float()
        z = model(x)
        points_pred = ae.decoder(z)
        cd_b = chamfer_distance_batched(points_pred, points_gt)  # (B,)
        cd_sum += cd_b.sum().item()
        n += points_gt.size(0)

        if worker_id == 0:
            print(f"[eval] Processed {n} / {len(test_subset)} samples", end="\r", flush=True)

    return cd_sum / max(1, n)


def worker_loop(worker_id: int, gpu_id: int, task_q: mp.Queue, result_q: mp.Queue, split_path: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    # dataset + split
    dataset = WaveguideDataset(DATA_PATH, stride=STRIDE)
    with open(split_path, "r") as f:
        sp = json.load(f)
    test_idx = sp["test_idx"]
    test_set = Subset(dataset, test_idx)

    # AE (decoder) once per worker
    ae = PCAutoencoder(latent_dim=LATENT_DIM, num_points=NUM_POINTS, tnet1=TNET1, tnet2=TNET2).to(device)
    best_ae_path = f"experiments/L{LATENT_DIM}/NP{NUM_POINTS}/checkpoints/model_ep{BEST_AE_EP}.pth"
    best_ae = torch.load(best_ae_path, map_location=device)
    ae.load_state_dict(best_ae["model_state_dict"])
    ae.eval()

    while True:
        task = task_q.get()
        if task is None:
            break
        try:
            t0 = time.time()
            cd = eval_one_run(
                ckpt_path=Path(task["ckpt_path"]),
                feature_idx=task["feature_idx"],
                test_subset=test_set,
                ae=ae,
                device=device,
                worker_id=worker_id,
            )
            result_q.put(
                {
                    "k": task["k"],
                    "seed": task["seed"],
                    "in_dim": task["in_dim"],
                    "test_chamfer": float(cd),
                    "ckpt_path": task["ckpt_path"],
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "time_sec": float(time.time() - t0),
                }
            )
        except Exception as e:
            result_q.put({"k": task["k"], "seed": task["seed"], "ckpt_path": task["ckpt_path"], "error": repr(e), "worker_id": worker_id, "gpu_id": gpu_id})


def main() -> None:
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Missing split file: {SPLIT_PATH}. Run training script first.")

    sweep_dir = ROOT_DIR / f"sweep_{args.sweep}" / f"order_{args.order}"
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Missing trained sweep dir: {sweep_dir}. Run training script first.")

    seeds = [BASE_SEED + i for i in range(REPEATS)]

    # Determine K range
    k_max = NUM_LEDS if args.sweep == "led" else NUM_PDS

    # Start workers (4 total, 1 per GPU)
    total_workers = NUM_GPUS * WORKERS_PER_GPU
    task_q, result_q = mp.Queue(), mp.Queue()
    workers = []
    for wid in range(total_workers):
        gpu_id = wid % NUM_GPUS
        p = mp.Process(target=worker_loop, args=(wid, gpu_id, task_q, result_q, str(SPLIT_PATH)), daemon=True)
        p.start()
        workers.append(p)

    # Per-run output CSV
    per_run_csv = ROOT_DIR / f"chamfer_{args.sweep}_{args.order}.csv"
    with open(per_run_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "seed", "in_dim", "test_chamfer", "ckpt_path", "gpu_id", "time_sec"])

    agg_rows = []
    # Process K sequentially (keeps queue small, and makes debugging easier)
    for k in range(1, k_max + 1):
        # Build tasks for 8 repeats
        tasks = []
        for s in seeds:
            run_dir = sweep_dir / f"K{k:02d}" / f"seed{s}"
            ckpt_path = run_dir / "best_pd2latent.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu")
            feat_idx = ckpt["feature_idx"]
            in_dim = int(ckpt["in_dim"])

            tasks.append({"k": k, "seed": s, "feature_idx": feat_idx, "in_dim": in_dim, "ckpt_path": str(ckpt_path)})

        # Dispatch
        for t in tasks:
            task_q.put(t)

        # Collect
        results_k = []
        while len(results_k) < len(tasks):
            res = result_q.get()
            if res.get("k") == k and res.get("seed") in seeds:
                results_k.append(res)
            else:
                result_q.put(res)
                time.sleep(0.01)

        errs = [r for r in results_k if "error" in r]
        if errs:
            print(f"[K={k:02d}] Errors:")
            for e in errs:
                print(" ", e)
            raise RuntimeError(f"Chamfer eval failed for K={k:02d}")

        # write per-run + aggregate
        cds = np.array([r["test_chamfer"] for r in results_k], dtype=np.float64)
        mean_cd = float(cds.mean())
        std_cd = float(cds.std(ddof=1)) if len(cds) > 1 else 0.0

        agg_rows.append({"K": k, "test_chamfer_mean": mean_cd, "test_chamfer_std": std_cd})

        with open(per_run_csv, "a", newline="") as f:
            w = csv.writer(f)
            for r in results_k:
                w.writerow([k, r["seed"], r["in_dim"], f"{r['test_chamfer']:.10f}", r["ckpt_path"], r["gpu_id"], f"{r['time_sec']:.2f}"])

        print(f"[{args.sweep.upper()}][{args.order}] K={k:02d}  TEST Chamfer {mean_cd:.8f} ± {std_cd:.8f}")

    # Stop workers
    for _ in workers:
        task_q.put(None)
    for p in workers:
        p.join()

    # Save aggregate CSV
    agg_csv = ROOT_DIR / f"aggregate_chamfer_{args.sweep}_{args.order}.csv"
    with open(agg_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)

    # Plot
    import matplotlib.pyplot as plt
    ks = [r["K"] for r in agg_rows]
    means = [r["test_chamfer_mean"] for r in agg_rows]
    stds = [r["test_chamfer_std"] for r in agg_rows]

    plt.figure()
    plt.errorbar(ks, means, yerr=stds, fmt='-o', capsize=4)
    plt.xlabel("K")
    plt.ylabel("Test Chamfer Distance")
    plt.title(f"Test Chamfer vs K ({args.sweep} sweep, order={args.order}, repeats={REPEATS})")
    plt.grid(True, alpha=0.3)
    fig_path = ROOT_DIR / f"chamfer_vs_k_{args.sweep}_{args.order}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nDone.")
    print("Per-run CSV :", per_run_csv)
    print("Aggregate   :", agg_csv)
    print("Figure      :", fig_path)


if __name__ == "__main__":
    main()
