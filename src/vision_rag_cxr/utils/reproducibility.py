"""재현성 유틸리티."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Python, NumPy, Torch random seed를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch가 설치되지 않은 preprocessing 환경에서도 동작해야 한다.
        pass
