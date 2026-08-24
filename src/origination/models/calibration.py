"""
Probability calibration for multiclass 1X2 and separate binary O/U 2.5.

Fit ONLY on training / calibration-holdout predictions — never on the test fold.

1X2 methods: none | platt | isotonic | temperature
O/U methods (independent): none | temperature | platt
  - temperature: single T on logit(p_over25)
  - platt: logistic regression on p_over25 → calibrated over; under = 1 - over
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CalibMethod = Literal["none", "platt", "isotonic", "temperature"]
OuCalibMethod = Literal["none", "platt", "temperature"]

PROB_COLS_1X2 = ["p_home", "p_draw", "p_away"]


class ProbabilityCalibrator:
    """
    Calibrators for 1X2 (+ optional independent O/U).

    - 1X2 platt / isotonic: one-vs-rest then renormalize
    - 1X2 temperature: single T on log-probs
    - OU temperature / platt: applied only to p_over25 / p_under25
    """

    def __init__(
        self,
        method: CalibMethod = "none",
        *,
        ou_method: OuCalibMethod = "none",
    ) -> None:
        self.method = method
        self.ou_method: OuCalibMethod = ou_method
        self._ovr_1x2: list[Any] = []
        self._ovr_ou: list[Any] = []
        self._ou_platt: Any | None = None
        self.temperature: float = 1.0
        self.ou_temperature: float = 1.0
        self.fitted = False

    def _make_binary(self) -> Any:
        if self.method == "platt":
            return LogisticRegression(solver="lbfgs", max_iter=1000)
        if self.method == "isotonic":
            return IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1.0 - 1e-4)
        return None

    def _fit_temperature(self, probs: np.ndarray, y: np.ndarray) -> float:
        logits = np.log(np.clip(probs, 1e-8, 1.0))

        def nll(t: float) -> float:
            if t <= 1e-3:
                return 1e12
            scaled = logits / t
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            exp = np.exp(scaled)
            p = exp / exp.sum(axis=1, keepdims=True)
            return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))

        res = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded")
        return float(res.x)

    def _fit_binary_temperature(self, p: np.ndarray, y: np.ndarray) -> float:
        """Temperature on Bernoulli logits for Over."""
        logits = np.log(np.clip(p, 1e-8, 1.0 - 1e-8)) - np.log(
            1.0 - np.clip(p, 1e-8, 1.0 - 1e-8)
        )

        def nll(t: float) -> float:
            if t <= 1e-3:
                return 1e12
            z = logits / t
            # stable sigmoid
            pos = z >= 0
            exp_nz = np.empty_like(z)
            exp_nz[pos] = np.exp(-z[pos])
            exp_nz[~pos] = np.exp(z[~pos])
            p_cal = np.empty_like(z)
            p_cal[pos] = 1.0 / (1.0 + exp_nz[pos])
            p_cal[~pos] = exp_nz[~pos] / (1.0 + exp_nz[~pos])
            p_cal = np.clip(p_cal, 1e-8, 1.0 - 1e-8)
            return float(-np.mean(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal)))

        res = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded")
        return float(res.x)

    @staticmethod
    def _apply_binary_temperature(p: np.ndarray, t: float) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
        logits = np.log(p) - np.log(1.0 - p)
        z = logits / max(t, 1e-3)
        return 1.0 / (1.0 + np.exp(-z))

    def _ou_labels(self, rows: pd.DataFrame, m: pd.DataFrame) -> np.ndarray | None:
        if "p_over25" not in rows.columns or "total_goals" not in m.columns:
            return None
        totals = []
        for mid in rows["match_id"]:
            row = m.loc[mid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            totals.append(float(row["total_goals"]))
        return (np.asarray(totals) > 2.5).astype(int)

    def _fit_ou(self, rows: pd.DataFrame, m: pd.DataFrame) -> None:
        self._ou_platt = None
        self._ovr_ou = []
        self.ou_temperature = 1.0
        if self.ou_method == "none":
            return
        y_over = self._ou_labels(rows, m)
        if y_over is None or len(y_over) < 50:
            logger.warning("OU calibration skipped — insufficient labels")
            self.ou_method = "none"
            return
        p = rows["p_over25"].astype(float).values
        if self.ou_method == "temperature":
            self.ou_temperature = self._fit_binary_temperature(p, y_over)
            logger.info(
                "Fitted OU temperature={:.3f} on {} matches",
                self.ou_temperature,
                len(rows),
            )
        elif self.ou_method == "platt":
            clf = LogisticRegression(solver="lbfgs", max_iter=1000)
            clf.fit(np.clip(p, 1e-4, 1 - 1e-4).reshape(-1, 1), y_over)
            self._ou_platt = clf
            logger.info("Fitted OU Platt on {} matches", len(rows))

    def _transform_ou(self, out: pd.DataFrame) -> pd.DataFrame:
        if self.ou_method == "none" or "p_over25" not in out.columns:
            return out
        p = out["p_over25"].astype(float).values
        if self.ou_method == "temperature":
            po = self._apply_binary_temperature(p, self.ou_temperature)
        elif self.ou_method == "platt" and self._ou_platt is not None:
            po = self._ou_platt.predict_proba(np.clip(p, 1e-4, 1 - 1e-4).reshape(-1, 1))[:, 1]
        else:
            return out
        po = np.clip(po, 1e-6, 1.0 - 1e-6)
        out["p_over25"] = po
        out["p_under25"] = 1.0 - po
        return out

    def fit(
        self,
        preds: pd.DataFrame,
        matches: pd.DataFrame,
        *,
        calibrate_ou: bool = True,
    ) -> "ProbabilityCalibrator":
        if self.method == "none" and self.ou_method == "none":
            self.fitted = True
            return self

        m = matches.set_index("match_id")
        rows = preds[preds["match_id"].isin(m.index)].copy()
        if len(rows) < 50:
            logger.warning("Calibration set too small ({}): leaving uncalibrated", len(rows))
            self.method = "none"
            self.ou_method = "none"
            self.fitted = True
            return self

        y_1x2 = []
        for mid in rows["match_id"]:
            row = m.loc[mid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            y_1x2.append({"H": 0, "D": 1, "A": 2}[str(row["ftr"])])
        y_1x2_arr = np.asarray(y_1x2)
        probs = rows[PROB_COLS_1X2].astype(float).values
        probs = probs / probs.sum(axis=1, keepdims=True)

        if self.method == "temperature":
            self.temperature = self._fit_temperature(probs, y_1x2_arr)
            logger.info(
                "Fitted temperature={:.3f} on {} matches", self.temperature, len(rows)
            )
        elif self.method != "none":
            self._ovr_1x2 = []
            for k, col in enumerate(PROB_COLS_1X2):
                x = rows[col].astype(float).values.reshape(-1, 1)
                y = (y_1x2_arr == k).astype(int)
                clf = self._make_binary()
                if self.method == "platt":
                    clf.fit(np.clip(x, 1e-4, 1 - 1e-4), y)
                else:
                    clf.fit(x.ravel(), y)
                self._ovr_1x2.append(clf)

            # Legacy: OVR OU when 1X2 is platt/isotonic and ou_method unset path
            self._ovr_ou = []
            if (
                calibrate_ou
                and self.ou_method == "none"
                and "p_over25" in rows.columns
                and "total_goals" in m.columns
            ):
                # Keep old behaviour only if explicitly using ovr via calibrate_ou
                # with method platt/isotonic — skipped when ou_method is set separately.
                pass

            logger.info(
                "Fitted {} calibrator on {} matches (1x2 ovr={})",
                self.method,
                len(rows),
                len(self._ovr_1x2),
            )

        if calibrate_ou:
            self._fit_ou(rows, m)

        self.fitted = True
        return self

    def _transform_binary(self, clf: Any, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        if self.method == "platt":
            return clf.predict_proba(np.clip(x, 1e-4, 1 - 1e-4))[:, 1]
        return clf.predict(x.ravel())

    def transform(self, preds: pd.DataFrame) -> pd.DataFrame:
        out = preds.copy()
        if not self.fitted:
            return out

        if self.method == "temperature":
            probs = out[PROB_COLS_1X2].astype(float).values
            probs = probs / probs.sum(axis=1, keepdims=True)
            logits = np.log(np.clip(probs, 1e-8, 1.0)) / self.temperature
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            mat = exp / exp.sum(axis=1, keepdims=True)
            for k, col in enumerate(PROB_COLS_1X2):
                out[col] = mat[:, k]
            if "p_ah0_home" in out.columns:
                denom = out["p_home"] + out["p_away"]
                out["p_ah0_home"] = np.where(denom > 0, out["p_home"] / denom, np.nan)
                out["p_ah0_away"] = np.where(denom > 0, out["p_away"] / denom, np.nan)
        elif self.method != "none" and self._ovr_1x2:
            calibrated = []
            for k, col in enumerate(PROB_COLS_1X2):
                calibrated.append(self._transform_binary(self._ovr_1x2[k], out[col].values))
            mat = np.column_stack(calibrated)
            mat = np.clip(mat, 1e-6, None)
            mat = mat / mat.sum(axis=1, keepdims=True)
            for k, col in enumerate(PROB_COLS_1X2):
                out[col] = mat[:, k]
            if "p_ah0_home" in out.columns:
                denom = out["p_home"] + out["p_away"]
                out["p_ah0_home"] = np.where(denom > 0, out["p_home"] / denom, np.nan)
                out["p_ah0_away"] = np.where(denom > 0, out["p_away"] / denom, np.nan)

        out = self._transform_ou(out)
        return out


def build_calibrator(cfg: dict[str, Any]) -> ProbabilityCalibrator:
    cal_cfg = cfg.get("model", {}).get("calibration", {})
    method = cal_cfg.get("method", "none")
    ou_method = cal_cfg.get("ou_method", "none")
    if method not in ("none", "platt", "isotonic", "temperature"):
        logger.warning("Unknown calibration method {}; using none", method)
        method = "none"
    if ou_method not in ("none", "platt", "temperature"):
        logger.warning("Unknown OU calibration method {}; using none", ou_method)
        ou_method = "none"
    return ProbabilityCalibrator(method=method, ou_method=ou_method)  # type: ignore[arg-type]
