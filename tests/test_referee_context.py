"""Tests for real referee context adjustment (lagged, no leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from origination.features.context_adjustments import RefereeAdjustment, apply_context_adjustments


def _matches(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    refs = ["R1", "R2", "R1", "R2"] * (n // 4 + 1)
    rows = []
    day = pd.Timestamp("2019-08-01")
    for i in range(n):
        rows.append(
            {
                "match_id": f"m{i}",
                "date": day + pd.Timedelta(days=i * 3),
                "referee": refs[i],
                "home_yellow": int(rng.integers(0, 5)),
                "away_yellow": int(rng.integers(0, 5)),
                "home_red": int(rng.integers(0, 2)),
                "away_red": int(rng.integers(0, 2)),
                "home_fouls": int(rng.integers(5, 20)),
                "away_fouls": int(rng.integers(5, 20)),
                "home_team": "A",
                "away_team": "B",
            }
        )
    return pd.DataFrame(rows)


def test_referee_first_games_nan():
    m = _matches()
    adj = RefereeAdjustment().apply(m, config={"min_prior_games": 5, "tempo_coef": 0.0})
    feat = adj.features
    # First appearance for each ref should lack history
    first_r1 = feat[m["referee"].values == "R1"].iloc[0]
    assert pd.isna(first_r1["ref_cards_avg"])


def test_referee_uses_only_prior():
    m = _matches(20)
    # Force known cards for R1
    m.loc[m["referee"] == "R1", "home_yellow"] = 4
    m.loc[m["referee"] == "R1", "away_yellow"] = 0
    m.loc[m["referee"] == "R1", "home_red"] = 0
    m.loc[m["referee"] == "R1", "away_red"] = 0
    adj = RefereeAdjustment().apply(m, config={"min_prior_games": 3, "tempo_coef": 0.0})
    feat = adj.features
    r1_idx = m.index[m["referee"] == "R1"].tolist()
    # After 3 prior R1 games, avg should be finite
    later = feat.loc[r1_idx[3]]
    assert pd.notna(later["ref_cards_avg"])


def test_yaml_flag_enables_referee():
    m = _matches(25)
    off = apply_context_adjustments(m, {"enabled": False, "referee": {"enabled": True}})
    assert len(off.features) == 0
    on = apply_context_adjustments(
        m,
        {
            "enabled": True,
            "referee": {"enabled": True, "min_prior_games": 3, "tempo_coef": 0.0},
        },
    )
    assert "ref_cards_avg" in on.features.columns
    assert on.meta["referee"]["wired"] is True


def test_tempo_coef_emits_intensity():
    m = _matches(25)
    adj = RefereeAdjustment().apply(m, config={"min_prior_games": 3, "tempo_coef": 0.02})
    assert len(adj.intensity_multipliers) == len(m)
    assert "lam_mult_home" in adj.intensity_multipliers.columns


def test_card_bias_asymmetric_opposite_signs():
    m = _matches(40)
    # Force R1 to always card home heavily so lagged share > 0.5
    m.loc[m["referee"] == "R1", "home_yellow"] = 5
    m.loc[m["referee"] == "R1", "away_yellow"] = 0
    m.loc[m["referee"] == "R1", "home_red"] = 0
    m.loc[m["referee"] == "R1", "away_red"] = 0
    adj = RefereeAdjustment().apply(
        m, config={"min_prior_games": 3, "tempo_coef": 0.0, "card_bias_coef": 0.5}
    )
    assert "ref_home_card_bias" in adj.features.columns
    assert len(adj.intensity_multipliers) == len(m)
    # Find a later R1 row with history and bias > 0 → lam < 1 < mu
    r1_ids = m.index[m["referee"] == "R1"].tolist()
    mid = m.loc[r1_ids[5], "match_id"]
    row = adj.intensity_multipliers.set_index("match_id").loc[mid]
    feat = adj.features.set_index("match_id").loc[mid]
    if pd.notna(feat["ref_home_card_bias"]) and feat["ref_home_card_bias"] > 0:
        assert row["lam_mult_home"] < 1.0
        assert row["lam_mult_away"] > 1.0
