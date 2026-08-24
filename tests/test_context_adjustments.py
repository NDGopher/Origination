"""Tests for context_adjustments scaffolding."""

from __future__ import annotations

import pandas as pd

from origination.features.context_adjustments import (
    ADJUSTMENT_REGISTRY,
    apply_context_adjustments,
    merge_context_into_features,
)


def test_registry_has_expected_keys():
    expected = {
        "injuries",
        "lineups",
        "formations",
        "motivation",
        "weather",
        "coaching_change",
        "referee",
        "travel",
    }
    assert expected.issubset(set(ADJUSTMENT_REGISTRY))


def test_disabled_returns_empty():
    matches = pd.DataFrame(
        {"match_id": ["m1", "m2"], "date": pd.to_datetime(["2020-01-01", "2020-01-08"])}
    )
    r = apply_context_adjustments(matches, {"enabled": False})
    assert len(r.features) == 0


def test_enabled_injuries_adds_zero_columns():
    matches = pd.DataFrame(
        {"match_id": ["m1", "m2"], "date": pd.to_datetime(["2020-01-01", "2020-01-08"])}
    )
    r = apply_context_adjustments(
        matches,
        {"enabled": True, "injuries": {"enabled": True}},
    )
    assert "injury_attack_delta_home" in r.features.columns
    feats = pd.DataFrame({"match_id": ["m1", "m2"], "elo_diff": [10.0, -5.0]})
    merged = merge_context_into_features(feats, r)
    assert "injury_attack_delta_home" in merged.columns
    assert merged["injury_attack_delta_home"].eq(0.0).all()
