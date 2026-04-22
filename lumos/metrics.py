import numpy as np
from typing import Tuple
from scipy.spatial import cKDTree
import torch

@torch.no_grad()
def chamfer_distance_batched(P: torch.Tensor, G: torch.Tensor):
    """
    Compute the Standard Chamfer Distance between two point clouds
    P: (B, N, 3) predicted points
    G: (B, M, 3) ground-truth points
    Returns:
        Chamfer distance = mean_{p in P} d(p, G) + mean_{g in G} d(g, P)
        Shape: (B,)
    """
    D = torch.cdist(P, G, p=2)
    d_pg = D.min(dim=2).values
    d_gp = D.min(dim=1).values
    cd = d_pg.mean(dim=1) + d_gp.mean(dim=1)
    return cd

@torch.no_grad()
def F_score_batched(P: torch.Tensor, G: torch.Tensor, tau: float):
    """
    Compute the F-score between two point clouds
    P: (B, N, 3) predicted points
    G: (B, M, 3) ground-truth points
    tau: distance threshold
    Returns:
        F-score = 2 * (precision * recall) / (precision + recall)
        precision = mean_{p in P} 1(d(p, G) < tau)
        recall = mean_{g in G} 1(d(g, P) < tau)
        F-score: (B,)
    """
    D = torch.cdist(P, G, p=2)
    d_pg = D.min(dim=2).values
    d_gp = D.min(dim=1).values

    precision = (d_pg < tau).float().mean(dim=1)
    recall = (d_gp < tau).float().mean(dim=1)

    f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f_score


def F_score(origin: np.ndarray, predicted: np.ndarray, tau: float = 1.0):
    if origin.ndim != 2 or origin.shape[1] != 3:
        raise ValueError("Origin point cloud must be of shape (N, 3)")
    if predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("Predicted point cloud must be of shape (M, 3)")

    tree_gt = cKDTree(origin)
    tree_pred = cKDTree(predicted)
    d_gt_to_pred, _ = tree_gt.query(predicted, k=1)
    d_pred_to_gt, _ = tree_pred.query(origin, k=1)

    recall = np.mean(d_gt_to_pred < tau)
    precision = np.mean(d_pred_to_gt < tau)

    if precision + recall == 0:
        return 0.0
    f_score = 2 * (precision * recall) / (precision + recall)
    return f_score

def chamfer_distance(pred_pts: np.ndarray, gt_pts: np.ndarray, squared: bool = True) -> float:
    """Compute bi-directional Chamfer distance between two point clouds.

    Args:
        pred_pts: (N,3) predicted points
        gt_pts:   (M,3) ground-truth points
        squared:  if True, use squared Euclidean distances (common in literature);
                  if False, use Euclidean distances.
    Returns:
        Chamfer distance = mean_{p in pred} d(p, GT)^?  +  mean_{g in GT} d(g, P)^?
        where ? is squared or not depending on `squared`.

    Notes:
        This is an O(N*M) implementation using NumPy broadcasting, which is fine
        for ~1-4k points. For larger clouds, replace with a KD-tree approach.
    """
    P = np.asarray(pred_pts, dtype=np.float64)
    G = np.asarray(gt_pts, dtype=np.float64)
    assert P.ndim == 2 and P.shape[1] == 3, "pred_pts must be (N,3)"
    assert G.ndim == 2 and G.shape[1] == 3, "gt_pts must be (M,3)"

    # Compute pairwise squared distances: (N,M)
    # d2 = ||P||^2 + ||G||^2 - 2 P.G^T
    P2 = np.sum(P * P, axis=1, keepdims=True)  # (N,1)
    G2 = np.sum(G * G, axis=1, keepdims=True).T  # (1,M)
    dot = P @ G.T  # (N,M)
    d2 = P2 + G2 - 2.0 * dot
    # Clamp tiny negatives from numerical error
    np.maximum(d2, 0.0, out=d2)

    # NN distances
    d2_P_to_G = np.min(d2, axis=1)  # (N,)
    d2_G_to_P = np.min(d2, axis=0)  # (M,)

    if squared:
        return float(d2_P_to_G.mean() + d2_G_to_P.mean())
    else:
        return float(np.sqrt(d2_P_to_G).mean() + np.sqrt(d2_G_to_P).mean())