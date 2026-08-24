"""Tests for coaching-change context adjustment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from origination.features.context_adjustments import CoachingChangeAdjustment, apply_context_adjustments


def test_coaching_change_flags_new_manager(tmp_path: Path):
    csv = tmp_path / "changes.csv"
    csv.write_text(
        "team,change_date,notes\nArsenal,2020-01-01,test\n",
        encoding="utf-8",
    )
    matches = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "date": pd.to_datetime(["2019-12-01", "2020-01-15"]),
            "home_team": ["Arsenal", "Arsenal"],
            "away_team": ["Chelsea", "Chelsea"],
        }
    )
    adj = CoachingChangeAdjustment().apply(
        matches,
        config={
            "changes_path": str(csv),
            "new_coach_days": 60,
            "new_coach_games": 10,
            "bounce_coef": 0.0,
        },
    )
    f = adj.features.set_index("match_id")
    assert f.loc["m1", "new_coach_home"] == 0.0
    assert f.loc["m2", "new_coach_home"] == 1.0
    assert f.loc["m2", "coach_days_in_charge_home"] >= 0


def test_coaching_yaml_enable():
    matches = pd.DataFrame(
        {
            "match_id": ["m1"],
            "date": pd.to_datetime(["2020-01-15"]),
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
        }
    )
    # Uses repo CSV if present
    on = apply_context_adjustments(
        matches,
        {
            "enabled": True,
            "coaching_change": {
                "enabled": True,
                "new_coach_days": 60,
                "new_coach_games": 8,
                "bounce_coef": 0.0,
            },
        },
    )
    assert "new_coach_home" in on.features.columns
    assert on.meta["coaching_change"]["wired"] is True
