"""
Optical feature importance sweep — Chamfer evaluation (multiprocess, multi-GPU).
Loads trained checkpoints from train_sweep.py and computes test Chamfer Distance.

Chamfer is memory-heavy, so only 1 worker per GPU is used.

Usage:
    python ablation/evaluate_sweep.py --sweep {led,pd} --order {sage,natural,sage_reverse}

Outputs (to {base_dir}/{run_name}/sweep/):
    chamfer_{sweep}_{order}.csv
    aggregate_chamfer_{sweep}_{order}.csv
    chamfer_vs_k_{sweep}_{order}.png
"""
import csv
import json
import time
import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader, Subset

from lumos.data import WaveguideDataset
from lumos.metrics import chamfer_distance_batched
from lumos.model import PD2Latent, PCAutoencoder


BATCH_SIZE = 32  # smaller than training; Chamfer is memory-heavy


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


# ----------------------------- Worker -----------------------------

@torch.no_grad()
def eval_one(*, ckpt_path: Path, feature_idx: List[int], test_set: Subset,
             ae: PCAutoencoder, latent_dim: int, device: torch.device,
             worker_id: int = 0) -> float:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = PD2Latent(in_features=int(ckpt["in_dim"]), out_features=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    cd_sum, n = 0.0, 0
    for pts_gt, _, opt_norm in loader:
        pts_gt = pts_gt.to(device).float()
        z = model(opt_norm[:, feature_idx].to(device).float())
        pts_pred = ae.decoder(z)
        cd_sum += chamfer_distance_batched(pts_pred, pts_gt).sum().item()
        n += pts_gt.size(0)
        if worker_id == 0:
            print(f"[eval] {n} / {len(test_set)}", end="\r", flush=True)
    return cd_sum / max(1, n)


def worker_loop(worker_id: int, gpu_id: int, task_q: mp.Queue, result_q: mp.Queue,
                cfg: dict, split_path: str) -> None:
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    out_cfg   = cfg["output"]
    run_dir   = Path(out_cfg["base_dir"]) / out_cfg["run_name"]

    dataset = WaveguideDataset(data_cfg["path"], stride=data_cfg["stride"])
    with open(split_path) as f:
        test_idx = json.load(f)["test_idx"]
    test_set = Subset(dataset, test_idx)

    ae = PCAutoencoder(latent_dim=model_cfg["latent_dim"], num_points=model_cfg["num_points"],
                       tnet1=model_cfg["tnet1"], tnet2=model_cfg["tnet2"]).to(device)
    ae.load_state_dict(
        torch.load(run_dir / "checkpoints" / "best_model.pth", map_location=device)["model_state_dict"]
    )
    ae.eval()

    while True:
        task = task_q.get()
        if task is None:
            break
        try:
            t0 = time.time()
            cd = eval_one(
                ckpt_path=Path(task["ckpt_path"]), feature_idx=task["feature_idx"],
                test_set=test_set, ae=ae, latent_dim=model_cfg["latent_dim"],
                device=device, worker_id=worker_id,
            )
            result_q.put({
                "k": task["k"], "seed": task["seed"], "in_dim": task["in_dim"],
                "test_chamfer": float(cd), "ckpt_path": task["ckpt_path"],
                "worker_id": worker_id, "gpu_id": gpu_id, "time_sec": time.time() - t0,
            })
        except Exception as e:
            result_q.put({
                "k": task["k"], "seed": task["seed"], "ckpt_path": task["ckpt_path"],
                "error": repr(e), "worker_id": worker_id, "gpu_id": gpu_id,
            })


# ----------------------------- Main -----------------------------

def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.run_name:
        cfg["output"]["run_name"] = args.run_name

    abl_cfg = cfg["ablation"]
    out_cfg = cfg["output"]

    run_dir    = Path(out_cfg["base_dir"]) / out_cfg["run_name"]
    sweep_root = run_dir / "sweep"
    split_path = sweep_root / "split_indices.json"
    sweep_dir  = sweep_root / f"sweep_{args.sweep}" / f"order_{args.order}"

    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}. Run train_sweep.py first.")
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Missing sweep dir: {sweep_dir}. Run train_sweep.py first.")

    num_leds  = abl_cfg["num_leds"]
    num_pds   = cfg["model"]["optical_dim"] // num_leds
    repeats   = abl_cfg["repeats"]
    base_seed = abl_cfg["base_seed"]
    seeds     = [base_seed + i for i in range(repeats)]
    k_max     = num_leds if args.sweep == "led" else num_pds

    num_gpus   = resolve_num_gpus(cfg)
    device_str = f"{num_gpus} GPU(s)" if torch.cuda.is_available() else "CPU"
    print(f"Using {device_str}, {num_gpus} workers (1 per GPU)")

    # Start workers (1 per GPU for Chamfer)
    task_q, result_q = mp.Queue(), mp.Queue()
    workers = []
    for wid in range(num_gpus):
        p = mp.Process(target=worker_loop,
                       args=(wid, wid % num_gpus, task_q, result_q, cfg, str(split_path)),
                       daemon=True)
        p.start()
        workers.append(p)

    per_run_csv = sweep_root / f"chamfer_{args.sweep}_{args.order}.csv"
    with open(per_run_csv, "w", newline="") as f:
        csv.writer(f).writerow(["K", "seed", "in_dim", "test_chamfer", "ckpt_path", "gpu_id", "time_sec"])

    agg_rows = []
    for k in range(1, k_max + 1):
        tasks = []
        for s in seeds:
            ckpt_path = sweep_dir / f"K{k:02d}" / f"seed{s}" / "best_pd2latent.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            tasks.append({
                "k": k, "seed": s, "feature_idx": ckpt["feature_idx"],
                "in_dim": int(ckpt["in_dim"]), "ckpt_path": str(ckpt_path),
            })

        for t in tasks:
            task_q.put(t)

        # Collect results for this K (workers may process tasks out of order)
        results_k = []
        while len(results_k) < len(tasks):
            res = result_q.get()
            if res.get("k") == k and res.get("seed") in seeds:
                results_k.append(res)
            else:
                result_q.put(res)  # belongs to a future K; put back
                time.sleep(0.01)

        errors = [r for r in results_k if "error" in r]
        if errors:
            for e in errors:
                print(f"  Error: {e}")
            raise RuntimeError(f"Chamfer eval failed for K={k}")

        cds     = np.array([r["test_chamfer"] for r in results_k])
        mean_cd = float(cds.mean())
        std_cd  = float(cds.std(ddof=1)) if len(cds) > 1 else 0.0
        agg_rows.append({"K": k, "test_chamfer_mean": mean_cd, "test_chamfer_std": std_cd})

        with open(per_run_csv, "a", newline="") as f:
            w = csv.writer(f)
            for r in results_k:
                w.writerow([k, r["seed"], r["in_dim"], f"{r['test_chamfer']:.10f}",
                            r["ckpt_path"], r["gpu_id"], f"{r['time_sec']:.2f}"])

        print(f"[{args.sweep.upper()}][{args.order}] K={k:02d}  Chamfer {mean_cd:.8f} ± {std_cd:.8f}")

    # Stop workers
    for _ in workers:
        task_q.put(None)
    for p in workers:
        p.join()

    agg_csv = sweep_root / f"aggregate_chamfer_{args.sweep}_{args.order}.csv"
    with open(agg_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)

    plt.figure()
    plt.errorbar([r["K"] for r in agg_rows], [r["test_chamfer_mean"] for r in agg_rows],
                 yerr=[r["test_chamfer_std"] for r in agg_rows], fmt="-o", capsize=4)
    plt.xlabel("K")
    plt.ylabel("Test Chamfer Distance")
    plt.title(f"Chamfer vs K ({args.sweep} sweep, {args.order}, n={repeats})")
    plt.grid(True, alpha=0.3)
    fig_path = sweep_root / f"chamfer_vs_k_{args.sweep}_{args.order}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nDone.")
    print("Per-run CSV :", per_run_csv)
    print("Aggregate   :", agg_csv)
    print("Figure      :", fig_path)


if __name__ == "__main__":
    main()
