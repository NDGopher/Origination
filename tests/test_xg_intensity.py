"""Tests for goals / xG / blend intensity sources."""

from __future__ import annotations

import numpy as np
import pandas as pd

from origination.models.poisson import DixonColesModel


def _toy_train(n: int = 120, with_xg: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    teams = [f"T{i}" for i in range(6)]
    rows = []
    day = pd.Timestamp("2018-08-01")
    for i in range(n):
        h, a = teams[i % 6], teams[(i + 1) % 6]
        hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        row = {
            "match_id": f"m{i}",
            "date": day + pd.Timedelta(days=i),
            "season": 2018,
            "home_team": h,
            "away_team": a,
            "home_goals": hg,
            "away_goals": ag,
            "ftr": "H" if hg > ag else ("A" if ag > hg else "D"),
        }
        if with_xg:
            row["home_xg"] = max(0.1, hg + rng.normal(0, 0.3))
            row["away_xg"] = max(0.1, ag + rng.normal(0, 0.3))
        rows.append(row)
    return pd.DataFrame(rows)


def test_fit_goals_predicts():
    df = _toy_train(with_xg=False)
    m = DixonColesModel(intensity_source="goals", use_dc=True)
    m.fit(df)
    p = m.predict_match("T0", "T1")
    assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1) < 1e-5


def test_fit_xg_predicts():
    df = _toy_train(with_xg=True)
    m = DixonColesModel(intensity_source="xg", use_dc=True)
    m.fit(df)
    assert m.strengths is not None
    assert m.strengths.intensity_source == "xg"
    p = m.predict_dataframe(df.tail(5))
    assert len(p) == 5
    assert np.allclose(p[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0, atol=1e-4)


def test_blend_falls_back_without_xg():
    df = _toy_train(with_xg=False)
    m = DixonColesModel(intensity_source="blend", blend_xg_weight=0.7)
    m.fit(df)
    assert m.strengths is not None
    assert m.strengths.intensity_source == "goals"
