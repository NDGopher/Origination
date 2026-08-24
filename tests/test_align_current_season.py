"""Current-season align: Understat-only rows + football-data xG fill."""

from __future__ import annotations

import pandas as pd
import pytest

from origination.data_ingestion.align import align_matches
from origination.data_ingestion.football_data import _normalize_frame
from origination.utils.team_names import DEFAULT_MAPPER


def _fd_row(
    date: str,
    home: str,
    away: str,
    hg: int,
    ag: int,
    *,
    odds: bool = True,
    xg_fd: tuple[float, float] | None = None,
) -> dict:
    row = {
        "Date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "ftr": "H" if hg > ag else ("A" if ag > hg else "D"),
        "close_h": 2.0 if odds else pd.NA,
        "close_d": 3.2 if odds else pd.NA,
        "close_a": 3.8 if odds else pd.NA,
        "season": 2025,
    }
    if xg_fd is not None:
        row["home_xg_fd"], row["away_xg_fd"] = xg_fd
    return row


def test_append_understat_only_results_without_odds():
    fd = pd.DataFrame(
        [
            _fd_row("2025-08-16", "Sevilla", "Valencia", 1, 0),
            _fd_row("2025-08-17", "Real Madrid", "Osasuna", 2, 1),
        ]
    )
    us = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-08-15"),
                "home_team": "Alaves",
                "away_team": "Getafe",
                "home_goals": 3,
                "away_goals": 0,
                "home_xg": 1.8,
                "away_xg": 0.4,
                "understat_id": 1,
                "season": 2026,
            }
        ]
    )
    out = align_matches(fd, understat=us, require_odds=True, min_date="2014-08-01")
    extra = out[out["home_team"] == "Alaves"]
    assert len(extra) == 1
    assert extra["result_source"].iloc[0] == "understat"
    assert pd.isna(extra["close_h"].iloc[0])
    assert extra["home_xg"].iloc[0] == pytest.approx(1.8)
    assert extra["home_goals"].iloc[0] == 3
    assert extra["ftr"].iloc[0] == "H"
    assert len(out) == 3


def test_fill_xg_from_football_data_hxg():
    fd = pd.DataFrame([_fd_row("2026-08-14", "Sporting", "Guimaraes", 3, 2, xg_fd=(2.1, 0.9))])
    out = align_matches(fd, understat=None, require_odds=True, min_date="2014-08-01")
    assert out["home_xg"].iloc[0] == pytest.approx(2.1)
    assert out["away_xg"].iloc[0] == pytest.approx(0.9)
    assert out["result_source"].iloc[0] == "football_data"


def test_normalize_rejects_wrong_div():
    df = pd.DataFrame(
        {
            "Div": ["EC", "EC"],
            "Date": ["15/08/26", "16/08/26"],
            "HomeTeam": ["Barnet", "York"],
            "AwayTeam": ["Oldham", "Rochdale"],
            "FTHG": [1, 2],
            "FTAG": [0, 1],
            "FTR": ["H", "H"],
        }
    )
    with pytest.raises(ValueError, match="wrong league"):
        _normalize_frame(df, "E0", 2026, DEFAULT_MAPPER)


def test_pulse_extra_results_append():
    fd = pd.DataFrame([_fd_row("2025-08-16", "Arsenal", "Wolves", 2, 0)])
    pulse = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-08-22"),
                "home_team": "Liverpool",
                "away_team": "Bournemouth",
                "home_goals": 2,
                "away_goals": 1,
                "home_xg": pd.NA,
                "away_xg": pd.NA,
                "season": 2026,
            }
        ]
    )
    out = align_matches(
        fd, understat=None, extra_results=pulse, require_odds=True, min_date="2014-08-01"
    )
    extra = out[out["home_team"] == "Liverpool"]
    assert len(extra) == 1
    assert extra["result_source"].iloc[0] == "pulse"
    assert extra["home_goals"].iloc[0] == 2
