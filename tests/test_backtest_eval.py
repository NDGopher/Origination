"""Backtest evaluation helpers — CLV and bet selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from origination.backtesting.walk_forward import evaluate_predictions


def test_evaluate_selects_positive_edge_only():
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": pd.Timestamp("2019-01-01"),
                "season": 2018,
                "home_team": "A",
                "away_team": "B",
                "ftr": "H",
                "home_goals": 2,
                "away_goals": 0,
                "total_goals": 2,
                "close_h": 2.0,
                "close_d": 3.5,
                "close_a": 3.8,
                "close_over25": 1.9,
                "close_under25": 1.9,
            }
        ]
    )
    # Fair ~ roughly equal-ish after vig; model hugely on home → edge
    preds = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "p_home": 0.70,
                "p_draw": 0.20,
                "p_away": 0.10,
                "p_over25": 0.55,
                "p_under25": 0.45,
            }
        ]
    )
    bets = evaluate_predictions(
        preds,
        matches,
        {
            "edge_threshold": 0.05,
            "markets": ["1x2"],
            "stake": {"method": "flat", "unit": 1.0, "max_stake": 5.0},
            "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
        },
    )
    assert len(bets) >= 1
    assert (bets["edge"] >= 0.05).all()
    assert bets.iloc[0]["side"] == "H"
    assert bets.iloc[0]["clv_prob"] > 0
