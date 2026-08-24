"""
Gradient boosting scaffold — enabled after Poisson baseline shows measurable CLV.

Do not use until walk-forward CLV for Dixon–Coles is logged and understood.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class LightGBMMarketModel:
    """
    Multiclass 1X2 LightGBM on feature matrix.
    Placeholder fit/predict — wire after baseline is solid.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMMarketModel":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("lightgbm required") from exc
        logger.warning("LightGBMMarketModel is experimental — prefer Dixon–Coles until CLV proven")
        train = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            {
                "objective": "multiclass",
                "num_class": 3,
                "learning_rate": self.params.get("learning_rate", 0.05),
                "num_leaves": self.params.get("num_leaves", 31),
                "min_child_samples": self.params.get("min_child_samples", 40),
                "verbosity": -1,
            },
            train,
            num_boost_round=int(self.params.get("n_estimators", 400)),
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)
