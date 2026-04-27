import os
import subprocess
import torch
import torch.distributed as dist


def query_free_gpu_memory_mib() -> list[int]:
    """Free memory (MiB) per visible GPU, in CUDA index order.

    Uses nvidia-smi so no CUDA context is created — safe to call before
    setting CUDA_VISIBLE_DEVICES. Honors an already-set CUDA_VISIBLE_DEVICES
    by filtering / reordering the result to match. Returns [] on failure.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    free = [int(x) for x in out.strip().splitlines() if x.strip()]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        try:
            idx = [int(x) for x in visible.split(",") if x.strip()]
            free = [free[i] for i in idx if 0 <= i < len(free)]
        except ValueError:
            pass
    return free


def select_free_gpus(min_free_mib: int) -> tuple[list[int], list[int]]:
    """Pick GPUs (logical indices) with at least ``min_free_mib`` MiB free.

    Returns (selected_logical_indices, free_mib_per_visible_gpu). If no GPU
    meets the threshold, falls back to the single GPU with the most free
    memory. Returns ([], []) if the query fails.
    """
    free = query_free_gpu_memory_mib()
    if not free:
        return [], []
    selected = [i for i, f in enumerate(free) if f >= min_free_mib]
    if not selected:
        selected = [max(range(len(free)), key=lambda j: free[j])]
    return selected, free


def apply_visible_devices(selected: list[int]) -> list[str]:
    """Set CUDA_VISIBLE_DEVICES to ``selected`` (indices into the current
    visibility), translating back to physical IDs if CUDA_VISIBLE_DEVICES
    was already set. Returns the resulting list of physical IDs.
    """
    prev = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if prev:
        phys_map = [x.strip() for x in prev.split(",") if x.strip()]
        phys = [phys_map[i] for i in selected]
    else:
        phys = [str(i) for i in selected]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(phys)
    return phys


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
