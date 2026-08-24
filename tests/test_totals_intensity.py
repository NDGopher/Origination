"""Totals congestion / rest intensity channel smoke tests."""

from __future__ import annotations

import pandas as pd

from origination.models.poisson import DixonColesModel


def test_congestion_jointly_scales_both_sides():
    model = DixonColesModel(
        intensity_adj_cfg={
            "enabled": True,
            "ppda_coef": 0.0,
            "deep_coef": 0.0,
            "congestion_coef": -0.10,
            "rest_coef": 0.0,
        }
    )
    base = pd.Series(
        {
            "home_games_last_7": 0.0,
            "away_games_last_7": 0.0,
            "home_rest_days": 7.0,
            "away_rest_days": 7.0,
            "ctx_lam_mult_home": 1.0,
            "ctx_lam_mult_away": 1.0,
        }
    )
    congested = base.copy()
    congested["home_games_last_7"] = 2.0
    congested["away_games_last_7"] = 2.0

    lam0, mu0 = model._intensity_multipliers_from_row(base)
    lam1, mu1 = model._intensity_multipliers_from_row(congested)
    assert lam1 < lam0 and mu1 < mu0
    assert abs((lam1 / lam0) - (mu1 / mu0)) < 1e-9


def test_rest_coef_moves_totals():
    model = DixonColesModel(
        intensity_adj_cfg={
            "enabled": True,
            "ppda_coef": 0.0,
            "deep_coef": 0.0,
            "congestion_coef": 0.0,
            "rest_coef": 0.10,
        }
    )
    short = pd.Series(
        {
            "home_games_last_7": 1.0,
            "away_games_last_7": 1.0,
            "home_rest_days": 3.0,
            "away_rest_days": 3.0,
            "ctx_lam_mult_home": 1.0,
            "ctx_lam_mult_away": 1.0,
        }
    )
    long = short.copy()
    long["home_rest_days"] = 10.0
    long["away_rest_days"] = 10.0
    ls, ms = model._intensity_multipliers_from_row(short)
    ll, ml = model._intensity_multipliers_from_row(long)
    assert ls > ll and ms > ml
