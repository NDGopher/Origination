"""Tests for elite-layer scaffolds and residual interactions."""

import pandas as pd

from origination.features.elite import (
    LeagueMeanShrinker,
    NullFormationEncoder,
    NullPlayerStrengthProvider,
    build_hierarchical_shrinker,
)
from origination.models.residual import _add_interaction_features


def test_null_player_provider_shape():
    m = pd.DataFrame({"match_id": ["a", "b"]})
    out = NullPlayerStrengthProvider().lineup_strength(m)
    assert list(out["match_id"]) == ["a", "b"]
    assert out["lineup_confirmed"].eq(False).all()


def test_null_formation_encoder_dims():
    m = pd.DataFrame({"match_id": ["a"]})
    out = NullFormationEncoder().encode(m, config={"embedding_dim": 3})
    assert "form_emb_home_0" in out.columns
    assert "form_emb_home_2" in out.columns


def test_league_mean_shrinker():
    s = LeagueMeanShrinker(share_attack=1.0, share_defence=0.0)
    atk, dfn = s.shrink({"A": 1.0, "B": -1.0}, {"A": 0.5, "B": -0.5})
    assert abs(atk["A"] - atk["B"]) < 1e-9
    assert dfn["A"] == 0.5


def test_hierarchical_disabled_by_default():
    assert build_hierarchical_shrinker({"model": {}}) is None
    assert build_hierarchical_shrinker({"model": {"hierarchical": {"enabled": True}}}) is not None


def test_residual_interactions_append():
    X = pd.DataFrame(
        {
            "match_id": [1, 2],
            "lambda_home": [1.2, 1.0],
            "lambda_away": [0.8, 1.1],
            "p_home": [0.4, 0.5],
        }
    )
    out, cols = _add_interaction_features(
        X, ["lambda_home", "lambda_away", "p_home"], enabled=True
    )
    assert "ix_lambda_home__lambda_away" in cols
    assert abs(out["ix_lambda_home__lambda_away"].iloc[0] - 0.96) < 1e-9
