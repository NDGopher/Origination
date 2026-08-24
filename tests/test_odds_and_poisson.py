"""Unit tests for vig removal, CLV, and scoreline markets."""

from __future__ import annotations

import numpy as np
import pytest

from origination.models.poisson import markets_from_matrix, score_matrix
from origination.utils.odds import (
    clv_odds,
    clv_probability,
    fair_probs,
    kelly_fraction,
    remove_vig_multiplicative,
    remove_vig_power,
)


def test_vig_removal_sums_to_one():
    odds = np.array([1.80, 3.60, 4.50])
    for method in ("multiplicative", "power"):
        p = fair_probs(odds, method=method)
        assert abs(p.sum() - 1.0) < 1e-6
        assert np.all(p > 0)


def test_power_vs_multiplicative_favorite():
    # Heavy favorite — power typically assigns slightly more to favorite than raw normalize
    odds = np.array([1.20, 7.0, 15.0])
    pm = remove_vig_multiplicative(odds)
    pp = remove_vig_power(odds)
    assert abs(pp.sum() - 1.0) < 1e-6
    assert abs(pm.sum() - 1.0) < 1e-6


def test_clv_signs():
    assert clv_probability(0.55, 0.50) > 0
    assert clv_odds(0.55, 2.0) > 0  # EV = 0.1
    assert kelly_fraction(0.55, 2.0) > 0
    assert kelly_fraction(0.40, 2.0) == 0.0


def test_totals_intercept_params():
    from origination.models.poisson import totals_intercept_params

    en, sh, cl, mode, dsh, mar = totals_intercept_params(
        {
            "model": {
                "hierarchical": {
                    "totals_intercept": True,
                    "totals_shrink": 0.2,
                    "totals_clip": 0.1,
                },
                "dixon_coles": {
                    "totals_intercept": {
                        "enabled": True,
                        "mode": "lift_only",
                        "dampen_shrink": 0.9,
                        "min_abs_raw": 0.05,
                    }
                },
            }
        }
    )
    assert en is True
    assert sh == pytest.approx(0.2)
    assert cl == pytest.approx(0.1)
    assert mode == "lift_only"
    assert dsh == pytest.approx(0.9)
    assert mar == pytest.approx(0.05)
    en2, *_ = totals_intercept_params({"model": {}})
    assert en2 is False


def test_score_matrix_normalization():
    mat = score_matrix(1.4, 1.1, max_goals=8, rho=-0.08, dixon_coles=True)
    assert abs(mat.sum() - 1.0) < 1e-6
    mk = markets_from_matrix(mat)
    assert abs(mk["p_home"] + mk["p_draw"] + mk["p_away"] - 1.0) < 1e-5
    assert abs(mk["p_over25"] + mk["p_under25"] - 1.0) < 1e-5
