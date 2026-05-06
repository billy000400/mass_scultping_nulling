"""Core utilities: seeding, class inference."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_num_classes(y: np.ndarray) -> Optional[int]:
    if y is None:
        return None
    y = np.asarray(y)
    if y.ndim == 1:
        return int(y.max()) + 1 if y.size > 0 else None
    if y.ndim == 2:
        return y.shape[1]
    return None
