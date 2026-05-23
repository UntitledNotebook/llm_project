from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, rank_offset: int = 0) -> None:
    final_seed = int(seed) + int(rank_offset)
    os.environ["PYTHONHASHSEED"] = str(final_seed)
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)
