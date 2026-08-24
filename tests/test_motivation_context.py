"""Tests for table-position motivation context adjustment."""

from __future__ import annotations

import pandas as pd

from origination.features.context_adjustments import MotivationAdjustment, apply_context_adjustments


def _mini_season() -> pd.DataFrame:
    """Small synthetic season to exercise lagged table logic."""
    fixtures = [
        ("m1", "2020-08-01", "Arsenal", "Chelsea", 2, 0),
        ("m2", "2020-08-02", "Liverpool", "Arsenal", 1, 1),
        ("m3", "2020-08-08", "Chelsea", "Liverpool", 0, 2),
        ("m4", "2020-08-09", "Arsenal", "Liverpool", 3, 0),
        ("m5", "2020-08-15", "Chelsea", "Arsenal", 1, 1),
        ("m6", "2020-08-16", "Liverpool", "Chelsea", 2, 1),
    ]
    rows = []
    for mid, date, h, a, hg, ag in fixtures:
        rows.append(
            {
                "match_id": mid,
                "date": pd.Timestamp(date),
                "season": 2020,
                "home_team": h,
                "away_team": a,
                "home_goals": hg,
                "away_goals": ag,
            }
        )
    return pd.DataFrame(rows)


def test_motivation_uses_prior_results_only():
    matches = _mini_season()
    adj = MotivationAdjustment().apply(
        matches,
        config={"min_games": 1, "season_length": 4, "late_games_left": 10},
    )
    f = adj.features.set_index("match_id")
    assert f.loc["m1", "games_played_home"] == 0.0
    assert f.loc["m1", "title_race_home"] == 0.0
    # Arsenal played m1 before m2 (away)
    assert f.loc["m2", "games_played_away"] == 1.0
    assert f.loc["m2", "points_away"] == 3.0


def test_motivation_intensity_yaml_gated():
    matches = _mini_season()
    off = MotivationAdjustment().apply(matches, config={"stakes_coef": 0.0, "min_games": 1})
    assert len(off.intensity_multipliers) == 0

    on = MotivationAdjustment().apply(
        matches,
        config={"stakes_coef": 0.05, "min_games": 1, "season_length": 4, "late_games_left": 10},
    )
    assert len(on.intensity_multipliers) > 0
    assert "lam_mult_home" in on.intensity_multipliers.columns


def test_motivation_yaml_enable_disable():
    matches = _mini_season()
    on = apply_context_adjustments(
        matches,
        {"enabled": True, "motivation": {"enabled": True, "min_games": 1}},
    )
    assert on.meta["motivation"]["wired"] is True
    assert "stakes_home" in on.features.columns

    off = apply_context_adjustments(
        matches,
        {"enabled": True, "motivation": {"enabled": False}},
    )
    assert "motivation" not in off.meta


def test_ah_settle_quarter_line():
    from origination.models.poisson import ah_settle_fraction

    assert ah_settle_fraction(1.0, -0.25, "ah_home") == 1.0
    assert abs(ah_settle_fraction(0.0, -0.25, "ah_home") - 0.25) < 1e-9
    assert ah_settle_fraction(0.0, 0.0, "ah_home") == 0.5
