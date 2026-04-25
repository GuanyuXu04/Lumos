import os
import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def is_main() -> bool:
    return get_rank() == 0


def ddp_setup() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    return local_rank



def ddp_cleanup():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x
