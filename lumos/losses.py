from typing import Tuple
import torch


def chamfer_distance(S1: torch.Tensor, S2: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if S1.dim() == 2: S1 = S1.unsqueeze(0)
    if S2.dim() == 2: S2 = S2.unsqueeze(0)
    d2 = torch.cdist(S1, S2, p=2).pow(2)
    d1_min, _ = d2.min(dim=2)
    d2_min, _ = d2.min(dim=1)
    loss = (d1_min.sum(dim=1) + d2_min.sum(dim=1)).mean()
    return loss, (d1_min, d2_min)


def repulsion_loss(pred: torch.Tensor, k: int = 10, h: float = 0.5) -> torch.Tensor:
    B, P, _ = pred.shape
    if P <= 1:
        return pred.new_tensor(0.0)
    k_eff = min(k, P - 1)
    d = torch.cdist(pred, pred, p=2)
    d2 = d.pow(2)
    _, idx = d.topk(k=k_eff + 1, largest=False)
    nn_d2 = torch.gather(d2, 2, idx[:, :, 1:])
    ker = torch.exp(-nn_d2 / (h * h))
    return ker.sum(dim=(1, 2)).mean() * 100.0
