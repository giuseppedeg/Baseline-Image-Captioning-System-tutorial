"""Reproducibility helpers.

Seeding every source of randomness is necessary for a comparison between two
training runs to mean anything. It is not sufficient: cuDNN selects convolution
algorithms by benchmarking, and the fastest algorithm is not always
deterministic. ``set_seed(..., deterministic=True)`` disables that search,
which typically costs throughput but makes a run bitwise repeatable.

The recommended practice in this tutorial is to seed always, and to enable
determinism only when investigating a discrepancy between two runs.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["set_seed", "seed_worker"]


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch, optionally enforcing determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # the preprocessing scripts do not require torch
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Surfaces any remaining non-deterministic kernel as an explicit error
        # rather than as silently irreproducible results.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Per-worker seeding for :class:`torch.utils.data.DataLoader`.

    Each worker process inherits a distinct PyTorch seed but *not* a distinct
    NumPy or ``random`` seed. Data augmentation implemented with either of
    those would therefore repeat identically across workers. Pass this function
    as ``worker_init_fn`` to avoid that.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
