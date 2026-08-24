"""Multi-league registry for fixtures, Pinnacle, and gameday."""

from __future__ import annotations

from typing import Any

# Keys used in paths: fixtures_upcoming_{KEY}.csv, pinnacle_ou25_{KEY}.csv
LEAGUES: dict[str, dict[str, Any]] = {
    "EPL": {
        "name": "Premier League",
        "fd_code": "E0",
        "understat": "EPL",
        "aligned": "matches_aligned.parquet",
        "config": "configs/default.yaml",
        "pinnacle_league_id": 1980,
        "pulse": True,  # Premier League Pulse API
        "packs": ["EPL_aggressive", "EPL_overs_short_exp"],
        "status": "production",
    },
    "Bundesliga": {
        "name": "Bundesliga",
        "fd_code": "D1",
        "understat": "Bundesliga",
        "aligned": "matches_aligned_D1.parquet",
        "config": "configs/league_D1_bundesliga.yaml",
        "pinnacle_league_id": 1842,
        "pulse": False,
        "packs": ["Bundesliga_unders_short_exp"],
        "status": "paper",
    },
    "Championship": {
        "name": "Championship",
        "fd_code": "E1",
        "understat": None,
        "aligned": "matches_aligned_E1.parquet",
        "config": "configs/league_E1_championship.yaml",
        "pinnacle_league_id": 1977,
        "pulse": False,
        "packs": [],  # no active system — research only
        "status": "research",
    },
    "SerieA": {
        "name": "Serie A",
        "fd_code": "I1",
        "understat": "Serie_A",
        "aligned": "matches_aligned_I1.parquet",
        "config": "configs/league_I1_serie_a.yaml",
        "pinnacle_league_id": 2436,
        "pulse": False,
        "packs": ["SerieA_away_ml_exp"],
        "status": "paper",
    },
    "LaLiga": {
        "name": "La Liga",
        "fd_code": "SP1",
        "understat": "La_Liga",
        "aligned": "matches_aligned_SP1.parquet",
        "config": "configs/league_SP1_la_liga.yaml",
        "pinnacle_league_id": 2196,
        "pulse": False,
        "packs": ["LaLiga_home_ml_short_exp"],
        "status": "paper",
    },
    "MLS": {
        "name": "Major League Soccer",
        "fd_code": "USA",
        "fd_new_url": "https://www.football-data.co.uk/new/USA.csv",
        "understat": None,
        "aligned": "matches_aligned_MLS.parquet",
        "config": "configs/league_MLS.yaml",
        "pinnacle_league_id": 2663,
        "pulse": False,
        "packs": [],
        "status": "score_preds",
    },
    # --- iter22 expansion (research until promoted) ---
    "Ligue1": {
        "name": "Ligue 1",
        "fd_code": "F1",
        "understat": "Ligue_1",
        "aligned": "matches_aligned_F1.parquet",
        "config": "configs/league_F1_ligue1.yaml",
        "pinnacle_league_id": 2036,
        "pulse": False,
        "packs": [],
        "status": "research",
    },
    "Eredivisie": {
        "name": "Eredivisie",
        "fd_code": "N1",
        "understat": None,
        "aligned": "matches_aligned_N1.parquet",
        "config": "configs/league_N1_eredivisie.yaml",
        "pinnacle_league_id": 1928,
        "pulse": False,
        "packs": [],
        "status": "research",
    },
    "PrimeiraLiga": {
        "name": "Primeira Liga",
        "fd_code": "P1",
        "understat": None,
        "aligned": "matches_aligned_P1.parquet",
        "config": "configs/league_P1_primeira.yaml",
        "pinnacle_league_id": 2386,
        "pulse": False,
        "packs": ["PrimeiraLiga_ah_e12_exp"],  # Primary live (e12); e10 optional sibling
        "status": "paper",
        "paper_packs_backtest": ["PrimeiraLiga_ah_e12_exp", "PrimeiraLiga_ah_short_exp"],
        "watch_packs": ["PrimeiraLiga_home_ml_watch"],
    },
    "Belgium": {
        "name": "Belgium Pro League",
        "fd_code": "B1",
        "understat": None,
        "aligned": "matches_aligned_B1.parquet",
        "config": "configs/league_B1_belgium.yaml",
        "pinnacle_league_id": 1817,
        "pulse": False,
        "packs": [],
        "status": "research",
        "watch_packs": ["Belgium_overs_short_watch"],
    },
    "Scotland": {
        "name": "Scottish Premiership",
        "fd_code": "SC0",
        "understat": None,
        "aligned": "matches_aligned_SC0.parquet",
        "config": "configs/league_SC0_scotland.yaml",
        "pinnacle_league_id": 2421,
        "pulse": False,
        "packs": [],
        "status": "research",
    },
    "Turkey": {
        "name": "Turkey Super Lig",
        "fd_code": "T1",
        "understat": None,
        "aligned": "matches_aligned_T1.parquet",
        "config": "configs/league_T1_turkey.yaml",
        "pinnacle_league_id": 2592,
        "pulse": False,
        "packs": [],
        "status": "research",
    },
    "Austria": {
        "name": "Austrian Bundesliga",
        "fd_code": "A1",
        "understat": None,
        "aligned": "matches_aligned_A1.parquet",
        "config": "configs/league_A1_austria.yaml",
        "pinnacle_league_id": 1838,
        "pulse": False,
        "packs": [],
        "status": "research",
    },
}


def get_league(key: str) -> dict[str, Any]:
    k = key.strip()
    # aliases
    aliases = {
        "E0": "EPL",
        "D1": "Bundesliga",
        "E1": "Championship",
        "I1": "SerieA",
        "SP1": "LaLiga",
        "USA": "MLS",
        "F1": "Ligue1",
        "N1": "Eredivisie",
        "P1": "PrimeiraLiga",
        "B1": "Belgium",
        "SC0": "Scotland",
        "T1": "Turkey",
        "A1": "Austria",
    }
    k = aliases.get(k, k)
    if k not in LEAGUES:
        raise KeyError(f"Unknown league key {key!r}. Known: {sorted(LEAGUES)}")
    return {"key": k, **LEAGUES[k]}


def list_league_keys() -> list[str]:
    return list(LEAGUES.keys())
