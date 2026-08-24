"""Score Predictions explain helpers."""

from origination.gameday.score_explain import (
    apply_score_only_projection,
    explain_match,
    score_only_offset,
    score_profile,
    team_form_aligned,
    team_snapshot,
)
import pandas as pd


def test_score_profile_high_low():
    assert score_profile(3.4, 0.55) == "HIGH"
    assert score_profile(2.8, 0.66) == "HIGH"
    assert score_profile(2.1, 0.45) == "LOW"
    assert score_profile(2.6, 0.35) == "LOW"
    assert score_profile(2.7, 0.52) == "MID"


def test_explain_uses_xg_and_elo():
    feat = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": pd.Timestamp("2026-08-01"),
                "home_team": "Ajax",
                "away_team": "Heerenveen",
                "home_xg_for_ewm": 2.1,
                "away_xg_for_ewm": 1.4,
                "home_xg_against_ewm": 1.1,
                "away_xg_against_ewm": 1.6,
                "home_points_roll5": 2.0,
                "away_points_roll5": 0.8,
                "elo_home": 1680,
                "elo_away": 1520,
                "home_rest_days": 7,
                "away_rest_days": 6,
                "_home_c": "Ajax",
                "_away_c": "Heerenveen",
            }
        ]
    )
    why = explain_match(home="Ajax", away="Heerenveen", feat=feat, proj_total=3.4, lean="OVER")
    assert "high total" in why
    assert "xG" in why
    assert "Elo" in why
    assert "form" in why
    assert "ppg" in why


def test_snapshot_matches_sporting_alias():
    feat = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-08-08"),
                "home_team": "Estrela",
                "away_team": "Sp Lisbon",
                "home_xg_for_ewm": 1.0,
                "away_xg_for_ewm": 1.8,
                "home_points_roll5": 1.0,
                "away_points_roll5": 2.2,
                "elo_home": 1500,
                "elo_away": 1700,
                "_home_c": "Estrela da Amadora",
                "_away_c": "Sporting Lisbon",
            }
        ]
    )
    snap = team_snapshot(feat, "Sporting CP")
    assert snap.get("xg_for") == 1.8
    assert snap.get("elo") == 1700


def test_aligned_form_fallback():
    aligned = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-08-01"),
                "home_team": "Club Brugge",
                "away_team": "Gent",
                "home_goals": 3,
                "away_goals": 1,
                "_home_c": "Club Brugge",
                "_away_c": "Gent",
            },
            {
                "date": pd.Timestamp("2026-08-08"),
                "home_team": "Anderlecht",
                "away_team": "Club Brugge",
                "home_goals": 0,
                "away_goals": 2,
                "_home_c": "Anderlecht",
                "_away_c": "Club Brugge",
            },
        ]
    )
    form = team_form_aligned(aligned, "Club Brugge", n=5)
    assert form["n"] == 2
    assert form["pts5"] == 3.0
    why = explain_match(
        home="Club Brugge",
        away="Gent",
        feat=pd.DataFrame(),
        aligned=aligned,
        proj_total=3.2,
        lean="OVER",
        league_key="Belgium",
    )
    assert "form" in why
    assert "weak O/U hist" in why


def test_score_only_offset_cools_serie_a_keeps_epl():
    assert score_only_offset("EPL") == 0.0
    assert score_only_offset("SerieA") < 0
    ph, pa, tot, po, pu, off = apply_score_only_projection(1.8, 1.4, "SerieA", p_over=0.58)
    assert off < 0
    assert tot < 3.2
    assert po < 0.58
    ph2, pa2, tot2, po2, _, off2 = apply_score_only_projection(1.5, 1.2, "EPL", p_over=0.50)
    assert off2 == 0.0
    assert tot2 == 2.7
    assert po2 == 0.50
