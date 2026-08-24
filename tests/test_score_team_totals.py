"""Team-total Score helpers + Pinnacle extract."""

from origination.gameday.score_team_totals import team_total_edge
from origination.models.poisson import team_total_over_prob
from origination.data_ingestion.pinnacle_odds import _extract_team_totals_main
from origination.utils.team_names import TeamNameMapper
from origination.utils.league_registry import get_league


def test_team_total_over_prob_half_lines():
    # lam=1.5 → P(X>=2) for over 1.5
    p15 = team_total_over_prob(1.5, 1.5)
    assert 0.4 < p15 < 0.6
    assert team_total_over_prob(3.0, 0.5) > 0.9
    assert team_total_over_prob(0.4, 2.5) < 0.1


def test_team_total_edge_vs_pin():
    e = team_total_edge(proj_goals=2.2, pin_line=1.5, pin_over=1.70, pin_under=2.20)
    assert e["line"] == 1.5
    assert e["p_over"] is not None and e["p_over"] > 0.5
    assert e["edge_over_pp"] is not None
    assert e["lean"] in ("OVER", "UNDER")


def test_extract_team_totals_from_sample():
    markets = [
        {
            "type": "team_total",
            "period": 0,
            "status": "open",
            "matchupId": 1,
            "side": "home",
            "key": "s;0;tt;1.5;home",
            "prices": [
                {"designation": "over", "points": 1.5, "price": -110},
                {"designation": "under", "points": 1.5, "price": -110},
            ],
        },
        {
            "type": "team_total",
            "period": 0,
            "status": "open",
            "matchupId": 1,
            "side": "away",
            "key": "s;0;tt;1.5;away",
            "prices": [
                {"designation": "over", "points": 1.5, "price": 120},
                {"designation": "under", "points": 1.5, "price": -150},
            ],
        },
    ]
    df = _extract_team_totals_main(markets)
    assert len(df) == 1
    assert float(df.iloc[0]["pin_tt_home_line"]) == 1.5
    assert float(df.iloc[0]["pin_tt_away_line"]) == 1.5


def test_turkey_scotland_pinnacle_ids():
    assert get_league("Turkey")["pinnacle_league_id"] == 2592
    assert get_league("Scotland")["pinnacle_league_id"] == 2421


def test_ligue1_belgium_aliases():
    m = TeamNameMapper()
    assert m.canonicalize("Paris Saint-Germain") == m.canonicalize("PSG") == "Paris Saint-Germain"
    assert m.canonicalize("Standard Liege") == "Standard Liege"
    assert m.canonicalize("KVC Westerlo") == "Westerlo"
