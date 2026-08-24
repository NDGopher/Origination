from __future__ import annotations

import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Seed Python and NumPy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
    except ImportError:
        pass


def season_label(start_year: int) -> str:
    """e.g. 2014 -> '1415' (football-data.co.uk path segment)."""
    end = (start_year + 1) % 100
    return f"{start_year % 100:02d}{end:02d}"


def season_from_date(date) -> int:
    """Football season start year: Aug–Dec -> year, Jan–Jul -> year-1."""
    import pandas as pd

    d = pd.Timestamp(date)
    return int(d.year if d.month >= 8 else d.year - 1)
