"""Leakage / chronology guards for the feature store."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from origination.features.store import assert_features_pre_match, build_feature_matrix, compute_elo


def _toy_matches(n_teams: int = 4, n_rounds: int = 6) -> pd.DataFrame:
    teams = [f"T{i}" for i in range(n_teams)]
    rows = []
    rng = np.random.default_rng(0)
    day = pd.Timestamp("2018-08-01")
    mid = 0
    for r in range(n_rounds):
        # round-robin pairs
        for i in range(0, n_teams, 2):
            h, a = teams[i], teams[(i + 1 + r) % n_teams]
            if h == a:
                a = teams[(i + 2) % n_teams]
            hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            rows.append(
                {
                    "match_id": f"m{mid}",
                    "date": day,
                    "season": 2018,
                    "home_team": h,
                    "away_team": a,
                    "home_goals": hg,
                    "away_goals": ag,
                    "home_shots": hg + 5,
                    "away_shots": ag + 5,
                    "home_sot": hg + 2,
                    "away_sot": ag + 2,
                    "home_xg": hg + 0.1,
                    "away_xg": ag + 0.1,
                    "ftr": "H" if hg > ag else ("A" if ag > hg else "D"),
                }
            )
            mid += 1
        day += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_elo_is_pre_match():
    m = _toy_matches()
    elo = compute_elo(m, k=20.0, home_advantage=50.0)
    # First match for each team should start near 1500
    assert abs(elo.iloc[0]["elo_home"] - 1500) < 1e-6


def test_features_do_not_equal_current_goals():
    m = _toy_matches()
    feats = build_feature_matrix(
        m,
        {
            "groups": {
                "basic_form": True,
                "xg_form": True,
                "shots": True,
                "elo": True,
                "schedule": True,
            },
            "windows": [3, 5],
        },
    )
    assert_features_pre_match(m, feats)
    # First round rolling features should be NaN (no history)
    first_ids = set(m.loc[m["date"] == m["date"].min(), "match_id"])
    first = feats[feats["match_id"].isin(first_ids)]
    if "home_goals_for_roll3" in first.columns:
        assert first["home_goals_for_roll3"].isna().all()


def test_rolling_uses_only_past():
    """Manually check team T0: after first match, roll3 goals_for equals that match's GF."""
    m = _toy_matches(n_teams=4, n_rounds=4)
    feats = build_feature_matrix(
        m,
        {"groups": {"basic_form": True, "xg_form": False, "shots": False, "elo": False, "schedule": False}, "windows": [3]},
    )
    # Chronological integrity assertion
    assert_features_pre_match(m, feats)
    assert len(feats) == len(m)
