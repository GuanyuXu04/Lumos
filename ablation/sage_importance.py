import numpy as np
import torch
from typing import List, Optional, Tuple, Dict

class ImportanceTracker:
    """
    Track the running mean of SAGE contributions and an uncertainty estimate using Welford updates.

    The returned std is an estimate of the standard error (SE) of the mean contributions,
    consistent with the SAGE repo's tracker usage for the convergence heuristic.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.mean = np.zeros(dim, dtype=np.float64)
        self.sum_squares = np.zeros(dim, dtype=np.float64)
        self.N = 0

    def update(self, scores: np.ndarray) -> None:
        """
        scores: (B, dim), where each row is the per-sample contribution vector.
        """
        if scores.ndim != 2 or scores.shape[1] != self.dim:
            raise ValueError(f"scores must have shape (B, {self.dim}), got {scores.shape}")

        B = scores.shape[0]
        self.N += B

        diff = scores - self.mean
        self.mean += np.sum(diff, axis=0) / self.N
        diff2 = scores - self.mean
        self.sum_squares += np.sum(diff * diff2, axis=0)

    @property
    def values(self) -> np.ndarray:
        return self.mean

    @property
    def var(self) -> np.ndarray:
        # Variance of the mean estimate (SE^2) ~ sum_squares / N^2
        return self.sum_squares / (max(self.N, 1) ** 2)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


def stddev_ratio(values: np.ndarray, std: np.ndarray, eps: float = 1e-12) -> float:
    """
    Convergence heuristic used in the SAGE repo: max(std) / (max(values)-min(values)).
    """
    gap = float(max(values.max() - values.min(), eps))
    return float(np.max(std)) / gap


@torch.no_grad()
def sage_led_group_importance(
    X: torch.Tensor,               # (N,150) torch tensor (recommended on CPU)
    Y: torch.Tensor,               # (N,512) torch tensor (recommended on CPU)
    model: torch.nn.Module,        # PD2Latent
    background_size: int = 512,    # inner samples m
    outer_batch_size: int = 32,    # number of outer samples processed per iteration
    bg_chunk_size: int = 64,       # process background in chunks to save GPU memory
    thresh: float = 0.025,         # convergence threshold (SAGE repo default)
    min_outer_samples: int = 2000,
    max_outer_samples: int = 50000,
    seed: int = 0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Sampling-based SAGE for 30 LED groups (each group is 5 features in X).

    Input tensors:
      X: (N,150) normalized optical vector (torch.Tensor)
      Y: (N,512) latent ground truth (torch.Tensor)

    Returns:
      values: (30,) SAGE importance (numpy)
      std:    (30,) SE-like uncertainty estimate (numpy)
      info:   diagnostics (dict)
    """
    if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor):
        raise TypeError("X and Y must be torch.Tensor")
    if X.ndim != 2:
        raise ValueError(f"X must be (N,_), got {tuple(X.shape)}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be (N,512), got {tuple(Y.shape)}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same N, got {X.shape[0]} vs {Y.shape[0]}")

    N = int(X.shape[0])
    D = int(X.shape[1])
    G = 30
    out_dim = int(Y.shape[1])

    # Resolve device from model if not specified
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()

    # Use NumPy RNG for permutation + index sampling (indices are later moved to the correct device)
    rng = np.random.default_rng(seed)

    # Define 30 LED groups, each group is 5 consecutive features in X
    groups: List[List[int]] = [list(range(g * 5, (g + 1) * 5)) for g in range(G)]

    # Build a (G, D) mapping matrix for fast group->feature expansion
    group_to_feat = torch.zeros((G, D), dtype=torch.float32, device=device)
    for g, inds in enumerate(groups):
        for j in inds:
            group_to_feat[g, j] = 1.0

    # Sample background data for marginal imputation (inner samples m)
    m = int(min(background_size, N))
    bg_idx_np = rng.choice(N, size=m, replace=False).astype(np.int64)
    bg_idx = torch.from_numpy(bg_idx_np).to(X.device, non_blocking=True)
    X_bg = X.index_select(0, bg_idx).to(device=device, dtype=torch.float32, non_blocking=True)  # (m,150)

    # Compute marginal baseline prediction: mean_i f(x_i)
    # For speed, approximate it with the mean prediction over the background set.
    marginal_pred = model(X_bg).mean(dim=0)  # (512,)

    # Define restricted prediction via marginal imputation:
    # y_hat = (1/m) sum_k f(x_S, x_-S^k), where x_-S^k are drawn from the marginal background.
    def restricted_predict(X_batch: torch.Tensor, Sg_batch: torch.Tensor) -> torch.Tensor:
        """
        X_batch: (B,150) float32 on `device`
        Sg_batch: (B,30) bool on `device`, True => group is ON (in coalition S)
        Returns: (B,512) restricted prediction averaged over m imputations.
        """
        B = int(X_batch.shape[0])

        # Expand group mask to feature mask: keep features whose groups are ON
        feat_keep = (Sg_batch.float() @ group_to_feat) > 0.0   # (B,150) bool
        feat_keep_3d = feat_keep[:, None, :]                   # (B,1,150)

        acc = torch.zeros((B, out_dim), dtype=torch.float32, device=device)

        for s in range(0, m, bg_chunk_size):
            e = min(s + bg_chunk_size, m)
            bg = X_bg[s:e]             # (mc,150)
            mc = int(bg.shape[0])

            # Broadcast to (B,mc,150)
            bg_expand = bg[None, :, :].expand(B, mc, D)
            x_expand = X_batch[:, None, :].expand(B, mc, D)
            keep_expand = feat_keep_3d.expand(B, mc, D)

            # Replace missing (OFF) features by background samples
            x_filled = torch.where(keep_expand, x_expand, bg_expand)  # (B,mc,150)

            # Run model on flattened batch
            y_flat = model(x_filled.reshape(B * mc, D))               # (B*mc,512)
            acc += y_flat.reshape(B, mc, out_dim).sum(dim=1)          # sum over mc

        return acc / float(m)

    tracker = ImportanceTracker(dim=G)
    it = 0
    ratio = np.inf

    # Pre-create an index base on the model device (used for fast scatter updates)
    while tracker.N < max_outer_samples:
        B = int(min(outer_batch_size, max_outer_samples - tracker.N))

        # Sample outer data points (x,y) from the dataset
        idx_np = rng.integers(0, N, size=B, dtype=np.int64)
        idx = torch.from_numpy(idx_np).to(X.device, non_blocking=True)

        Xb = X.index_select(0, idx).to(device=device, dtype=torch.float32, non_blocking=True)  # (B,150)
        Yb = Y.index_select(0, idx).to(device=device, dtype=torch.float32, non_blocking=True)  # (B,512)

        # Sample a random permutation of groups for each example (B permutations)
        perms = np.tile(np.arange(G, dtype=np.int64), (B, 1))
        for b in range(B):
            rng.shuffle(perms[b])

        # Coalition set S starts empty for each example (torch bool on device)
        Sg = torch.zeros((B, G), dtype=torch.bool, device=device)

        # Per-example previous loss: MSE mean over 512 dims
        loss_prev = ((marginal_pred[None, :] - Yb) ** 2).mean(dim=1)  # (B,)

        # Store per-example contributions (each step assigns delta to one group)
        scores = np.zeros((B, G), dtype=np.float64)

        row_idx = torch.arange(B, device=device)

        # Iterate j=1..G, add one group at a time following permutation order
        for j in range(G):
            g_np = perms[:, j]  # (B,)
            g = torch.from_numpy(g_np).to(device=device, non_blocking=True)

            # Turn on the newly added group for each example
            Sg[row_idx, g] = True

            y_hat = restricted_predict(Xb, Sg)
            loss = ((y_hat - Yb) ** 2).mean(dim=1)  # (B,)

            delta = (loss_prev - loss).detach().cpu().numpy()         # (B,)
            scores[np.arange(B), g_np] = delta                        # assign to the newly added group
            loss_prev = loss

        tracker.update(scores)
        it += 1

        vals = tracker.values
        std = tracker.std
        ratio = stddev_ratio(vals, std)

        if verbose and (it % 10 == 0 or (tracker.N >= min_outer_samples and ratio < thresh)):
            print(f"[sage] iter={it:5d} outer_samples={tracker.N:7d} "
                  f"m={m:4d} StdDevRatio={ratio:.4f} (thresh={thresh:.4f})")

        # Stop if converged (as in the SAGE repo)
        if tracker.N >= min_outer_samples and ratio < thresh:
            break
    
    return tracker.values.astype(np.float64), tracker.std.astype(np.float64)