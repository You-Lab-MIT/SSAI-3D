"""Device selection with an optional GPU memory budget."""
import logging
import os

import torch

logger = logging.getLogger('ssai3d')

# Memory taken by the CUDA context itself, outside PyTorch's allocator.
CUDA_CONTEXT_GB = 0.6


def setup_device(gpu=0, max_mem_gb=None):
    """Select the device used by all later SSAI-3D calls in this process.

    Args:
        gpu (int | None): CUDA device index, or None / -1 for CPU.
        max_mem_gb (float | None): Total memory budget on that GPU, including
            the CUDA context. PyTorch raises an out-of-memory error instead of
            growing past it. None falls back to the SSAI3D_MAX_MEM_GB
            environment variable, then to no limit.

    Returns:
        torch.device
    """
    if gpu is None or gpu < 0 or not torch.cuda.is_available():
        if gpu is not None and gpu >= 0:
            logger.warning('CUDA is not available; running on CPU.')
        return torch.device('cpu')

    torch.cuda.set_device(gpu)
    if max_mem_gb is None and os.environ.get('SSAI3D_MAX_MEM_GB'):
        max_mem_gb = float(os.environ['SSAI3D_MAX_MEM_GB'])
    if max_mem_gb is not None:
        total = torch.cuda.get_device_properties(gpu).total_memory
        budget = (max_mem_gb - CUDA_CONTEXT_GB) * 1024**3
        if budget <= 0:
            raise ValueError(f'max_mem_gb={max_mem_gb} is below the CUDA context size.')
        torch.cuda.set_per_process_memory_fraction(min(budget / total, 1.0), gpu)
        logger.info(f'GPU {gpu}: memory capped at {max_mem_gb} GB.')
    return torch.device('cuda', gpu)


def peak_memory_gb():
    """Peak allocator memory on the current device since the last reset, plus context."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_reserved() / 1024**3 + CUDA_CONTEXT_GB
