"""Tests for fold-safe probability calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from origination.models.calibration import ProbabilityCalibrator


def _toy_preds_matches(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Overconfident raw probs
    true = rng.integers(0, 3, size=n)
    raw = np.full((n, 3), 0.1)
    raw[np.arange(n), true] = 0.8
    # Add noise and renormalize
    raw = raw + rng.normal(0, 0.05, size=raw.shape)
    raw = np.clip(raw, 0.01, None)
    raw = raw / raw.sum(axis=1, keepdims=True)

    preds = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "p_home": raw[:, 0],
            "p_draw": raw[:, 1],
            "p_away": raw[:, 2],
            "p_over25": rng.uniform(0.3, 0.7, n),
            "p_under25": 0.0,
        }
    )
    preds["p_under25"] = 1.0 - preds["p_over25"]
    ftr = np.array(["H", "D", "A"])[true]
    totals = rng.integers(0, 6, size=n).astype(float)
    matches = pd.DataFrame(
        {
            "match_id": preds["match_id"],
            "ftr": ftr,
            "total_goals": totals,
        }
    )
    return preds, matches


def test_isotonic_renormalizes():
    preds, matches = _toy_preds_matches()
    cal = ProbabilityCalibrator("isotonic").fit(preds, matches)
    out = cal.transform(preds)
    sums = out[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_platt_renormalizes():
    preds, matches = _toy_preds_matches()
    cal = ProbabilityCalibrator("platt").fit(preds, matches)
    out = cal.transform(preds)
    sums = out[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_temperature_renormalizes():
    preds, matches = _toy_preds_matches()
    cal = ProbabilityCalibrator("temperature").fit(preds, matches)
    out = cal.transform(preds)
    sums = out[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)
    assert cal.temperature > 0


def test_none_passthrough():
    preds, matches = _toy_preds_matches(80)
    cal = ProbabilityCalibrator("none").fit(preds, matches)
    out = cal.transform(preds)
    assert np.allclose(out["p_home"], preds["p_home"])


def test_ou_temperature_independent_of_1x2():
    preds, matches = _toy_preds_matches(300, seed=1)
    # Bias over probs so T != 1
    preds = preds.copy()
    preds["p_over25"] = np.clip(preds["p_over25"] ** 0.5, 0.05, 0.95)
    preds["p_under25"] = 1.0 - preds["p_over25"]
    cal = ProbabilityCalibrator("temperature", ou_method="temperature").fit(preds, matches)
    out = cal.transform(preds)
    assert cal.ou_temperature > 0
    sums = out[["p_over25", "p_under25"]].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)
    # 1X2 still calibrated / sums to 1
    assert np.allclose(out[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0, atol=1e-5)


def test_ou_platt_changes_over_probs():
    preds, matches = _toy_preds_matches(300, seed=2)
    cal = ProbabilityCalibrator("temperature", ou_method="platt").fit(preds, matches)
    out = cal.transform(preds)
    assert cal._ou_platt is not None
    assert np.allclose(out["p_over25"] + out["p_under25"], 1.0, atol=1e-5)
