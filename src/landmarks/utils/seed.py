# Siddharth Mehta, CS5330 PRCV, Final Project
# Sets every random seed. Both arms have to see the same data in the same order,
# or a difference between them could just be luck.

from __future__ import annotations

import os
import random

import numpy as np


# Seeds every random source so both arms see the same data in the
# same order.
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# Gives each DataLoader worker its own reproducible random stream.
def worker_init_fn(worker_id: int) -> None:
    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
