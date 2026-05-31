"""Global constants and reproducibility utilities."""

import random
import os
import numpy as np

SEED = 42
TEXT_COL = "text"
LABEL_COL = "label"
CLASSES = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
N_CLASSES = 3


def set_global_seed(seed: int = SEED) -> None:
    """Fix all random seeds for full reproducibility (Manning et al., 2008)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


set_global_seed()
