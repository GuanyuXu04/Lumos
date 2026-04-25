"""
Custom SAGE implementation for optical feature group importance.
Also runnable as a script to compute and save importance CSVs.

Usage:
    python ablation/sage_importance.py [--config PATH] [--run-name NAME]

Outputs (to {base_dir}/{run_name}/sage_results/):
    sage_led_importance.csv
    sage_pd_importance.csv
"""
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from lumos.config import HARDWARE
from lumos.data import WaveguideDataset
from lumos.model import PCAutoencoder, PD2Latent


# ----------------------------- SAGE Core -----------------------------

class _ImportanceTracker:
    """Welford online mean and SE tracker for SAGE convergence."""
    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.sum_sq = np.zeros(dim, dtype=np.float64)
        self.N = 0

    def update(self, scores: np.ndarray) -> None:
        B = scores.shape[0]
        self.N += B
        d_old = scores - self.mean
        self.mean += d_old.sum(axis=0) / self.N
        d_new = scores - self.mean
        self.sum_sq += (d_old * d_new).sum(axis=0)

    @property
    def values(self) -> np.ndarray:
        return self.mean

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.sum_sq / max(self.N, 1) ** 2)


def _stddev_ratio(values: np.ndarray, std: np.ndarray, eps: float = 1e-12) -> float:
    gap = max(float(values.max() - values.min()), eps)
    return float(std.max()) / gap


@torch.no_grad()
def sage_group_importance(
    X: torch.Tensor,
    Y: torch.Tensor,
    model: torch.nn.Module,
    groups: List[List[int]],
    background_size: int = 512,
    outer_batch_size: int = 32,
    bg_chunk_size: int = 64,
    thresh: float = 0.025,
    min_outer_samples: int = 2000,
    max_outer_samples: int = 50000,
    seed: int = 0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sampling-based SAGE importance for arbitrary feature groups.

    Args:
        X:      (N, D) normalized optical features (on CPU)
        Y:      (N, out_dim) latent ground truth (on CPU)
        model:  PD2Latent
        groups: list of G feature-index lists defining each group

    Returns:
        values: (G,) importance values
        std:    (G,) standard errors
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    N, D, G, out_dim = X.shape[0], X.shape[1], len(groups), Y.shape[1]
    model.eval()
    rng = np.random.default_rng(seed)

    # (G, D) binary mask: group_mask[g, d] = 1 if feature d belongs to group g
    group_mask = torch.zeros(G, D, dtype=torch.float32, device=device)
    for g, inds in enumerate(groups):
        for j in inds:
            group_mask[g, j] = 1.0

    # Background samples for marginal imputation
    m = min(background_size, N)
    bg_idx = torch.from_numpy(rng.choice(N, size=m, replace=False).astype(np.int64))
    X_bg = X.index_select(0, bg_idx).to(device, dtype=torch.float32)   # (m, D)
    marginal_pred = model(X_bg).mean(dim=0)                             # (out_dim,)

    def restricted_predict(Xb: torch.Tensor, Sg: torch.Tensor) -> torch.Tensor:
        """
        Average prediction over m imputations, replacing non-coalition features
        with background samples.
        Xb: (B, D), Sg: (B, G) bool coalition mask
        Returns: (B, out_dim)
        """
        B = Xb.shape[0]
        keep = (Sg.float() @ group_mask) > 0.0          # (B, D) bool
        acc = torch.zeros(B, out_dim, device=device)
        for s in range(0, m, bg_chunk_size):
            bg = X_bg[s:s + bg_chunk_size]               # (mc, D)
            mc = bg.shape[0]
            x_filled = torch.where(
                keep[:, None].expand(B, mc, D),
                Xb[:, None].expand(B, mc, D),
                bg[None].expand(B, mc, D),
            )                                            # (B, mc, D)
            acc += model(x_filled.reshape(B * mc, D)).reshape(B, mc, out_dim).sum(dim=1)
        return acc / m

    tracker = _ImportanceTracker(dim=G)
    while tracker.N < max_outer_samples:
        B = min(outer_batch_size, max_outer_samples - tracker.N)
        idx = torch.from_numpy(rng.integers(0, N, size=B, dtype=np.int64))
        Xb = X.index_select(0, idx).to(device, dtype=torch.float32)
        Yb = Y.index_select(0, idx).to(device, dtype=torch.float32)

        perms = np.stack([rng.permutation(G) for _ in range(B)])  # (B, G)
        Sg = torch.zeros(B, G, dtype=torch.bool, device=device)
        loss_prev = ((marginal_pred - Yb) ** 2).mean(dim=1)
        scores = np.zeros((B, G), dtype=np.float64)
        rows = torch.arange(B, device=device)

        for j in range(G):
            g_np = perms[:, j]
            Sg[rows, torch.from_numpy(g_np).to(device)] = True
            loss = ((restricted_predict(Xb, Sg) - Yb) ** 2).mean(dim=1)
            scores[np.arange(B), g_np] = (loss_prev - loss).cpu().numpy()
            loss_prev = loss

        tracker.update(scores)
        ratio = _stddev_ratio(tracker.values, tracker.std)

        if verbose and tracker.N % (outer_batch_size * 10) == 0:
            print(f"[sage] N={tracker.N:6d}  ratio={ratio:.4f} (thresh={thresh:.4f})")

        if tracker.N >= min_outer_samples and ratio < thresh:
            if verbose:
                print(f"[sage] Converged at N={tracker.N}  ratio={ratio:.4f}")
            break

    return tracker.values.astype(np.float64), tracker.std.astype(np.float64)


# ----------------------------- Script entrypoint -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.run_name:
        cfg["output"]["run_name"] = args.run_name

    seed = cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    out_cfg   = cfg["output"]

    run_dir  = Path(out_cfg["base_dir"]) / out_cfg["run_name"]
    num_leds = HARDWARE.num_leds
    num_pds  = HARDWARE.num_pds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load AE
    ae = PCAutoencoder(
        latent_dim=model_cfg["latent_dim"], num_points=model_cfg["num_points"],
        tnet1=model_cfg["tnet1"], tnet2=model_cfg["tnet2"],
    ).to(device)
    ae.load_state_dict(
        torch.load(run_dir / "checkpoints" / "best_model.pth", map_location=device)["model_state_dict"]
    )
    ae.eval()

    # Load PD2Latent
    model = PD2Latent(in_features=model_cfg["optical_dim"], out_features=model_cfg["latent_dim"]).to(device)
    model.load_state_dict(
        torch.load(run_dir / "best_combined.pth", map_location=device)["pd2latent_state_dict"]
    )
    model.eval()

    # Test split
    dataset = WaveguideDataset(data_cfg["path"], stride=data_cfg["stride"])
    n = len(dataset)
    n_train = int(data_cfg["train_split"] * n)
    n_val   = int(data_cfg["val_split"] * n)
    _, _, test_set = random_split(
        dataset, [n_train, n_val, n - n_train - n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # Precompute test latents
    print("Precomputing test latents...")
    loader = DataLoader(test_set, batch_size=64, shuffle=False)
    X_list, Z_list = [], []
    with torch.no_grad():
        for pts, _, opt_norm in loader:
            Z_list.append(ae.encoder(pts.to(device).float()).cpu())
            X_list.append(opt_norm)
    X_test = torch.cat(X_list).float()
    Z_test = torch.cat(Z_list).float()
    print(f"Test set: X={X_test.shape}, Z={Z_test.shape}")

    # Group definitions
    led_groups = [list(range(i * num_pds, (i + 1) * num_pds)) for i in range(num_leds)]
    pd_groups  = [list(range(i, num_leds * num_pds, num_pds))  for i in range(num_pds)]

    out_dir = run_dir / "sage_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for groups, names, label, csv_name in [
        (led_groups, [f"LED_{i+1}" for i in range(num_leds)], "LED", "sage_led_importance.csv"),
        (pd_groups,  [f"PD_{i+1}"  for i in range(num_pds)],  "PD",  "sage_pd_importance.csv"),
    ]:
        print(f"\nComputing SAGE for {len(groups)} {label} groups...")
        values, stds = sage_group_importance(
            X_test, Z_test, model, groups, device=device, seed=seed,
        )
        df = pd.DataFrame({f"{label}_Group": names, "SAGE_Value": values, "Std_Err": stds})
        df = df.sort_values("SAGE_Value", ascending=False)
        csv_path = out_dir / csv_name
        df.to_csv(csv_path, index=False)
        print(df.to_string(index=False))
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
