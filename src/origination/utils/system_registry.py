"""Frozen metadata for protected / paper systems — rules never edited here."""

from __future__ import annotations

from typing import Any

# Historical performance summaries (from walk-forward / significance reports).
# Update only when a new signed evaluation is accepted — never change rules.
PROTECTED_SYSTEMS: list[dict[str, Any]] = [
    {
        "id": "EPL_unders",
        "name": "EPL Unders",
        "pack": "EPL_aggressive",
        "league": "EPL",
        "market": "OU Under",
        "side": "under",
        "min_odds": 2.00,
        "max_odds": 4.00,
        "edge_thr": 0.08,
        "status": "production",
        "live": True,
        "rules_text": "Under 2.00–4.00 @ edge ≥8%",
        "history": {
            "n": 495,
            "roi": 0.094,
            "hit": None,
            "seasons_pos": 7,
            "seasons_n": 10,
            "max_dd_u": -6.4,
            "t_stat": 1.61,
            "source": "iter21 significance / EPL vol06",
        },
        "odds_col": "pin_under25",
        "edge_col": "edge_under_vs_pinnacle",
        "prob_col": "p_under25",
        "fair_odds_col": "fair_odds_under25",
        "book_odds_col": "book_under25",
        "book_edge_col": "edge_under_vs_book",
        "american_col": "pin_american_under25",
    },
    {
        "id": "EPL_overs_short",
        "name": "EPL short Overs",
        "pack": "EPL_overs_short_exp",
        "league": "EPL",
        "market": "OU Over",
        "side": "over",
        "min_odds": 1.60,
        "max_odds": 2.50,
        "edge_thr": 0.10,
        "status": "production",
        "live": True,
        "rules_text": "Over 1.60–2.50 @ edge ≥10%",
        "history": {
            "n": 202,
            "roi": 0.079,
            "hit": None,
            "seasons_pos": 7,
            "seasons_n": 10,
            "max_dd_u": -3.3,
            "t_stat": 1.09,
            "source": "iter21 significance / EPL vol06",
        },
        "odds_col": "pin_over25",
        "edge_col": "edge_over_vs_pinnacle",
        "prob_col": "p_over25",
        "fair_odds_col": "fair_odds_over25",
        "book_odds_col": "book_over25",
        "book_edge_col": "edge_over_vs_book",
        "american_col": "pin_american_over25",
    },
    {
        "id": "Bundesliga_unders",
        "name": "Bundesliga Unders",
        "pack": "Bundesliga_unders_short_exp",
        "league": "Bundesliga",
        "market": "OU Under",
        "side": "under",
        "min_odds": 1.70,
        "max_odds": 2.50,
        "edge_thr": 0.10,
        "status": "paper",
        "live": True,
        "rules_text": "Under 1.70–2.50 @ edge ≥10%",
        "history": {
            "n": 161,
            "roi": 0.086,
            "hit": None,
            "seasons_pos": 8,
            "seasons_n": 10,
            "max_dd_u": -1.7,
            "t_stat": 0.97,
            "source": "iter20/21 Bundesliga thresh05",
        },
        "odds_col": "pin_under25",
        "edge_col": "edge_under_vs_pinnacle",
        "prob_col": "p_under25",
        "fair_odds_col": "fair_odds_under25",
        "book_odds_col": "book_under25",
        "book_edge_col": "edge_under_vs_book",
        "american_col": "pin_american_under25",
    },
    {
        "id": "LaLiga_home_ml",
        "name": "La Liga Home ML",
        "pack": "LaLiga_home_ml_short_exp",
        "league": "LaLiga",
        "market": "1X2 Home",
        "side": "H",
        "min_odds": 1.01,
        "max_odds": 1.80,
        "edge_thr": 0.08,
        "status": "paper",
        "live": True,
        "rules_text": "Home @ edge ≥8%, max odds 1.80",
        "history": {
            "n": 124,
            "roi": 0.203,
            "hit": None,
            "seasons_pos": 8,
            "seasons_n": 10,
            "max_dd_u": -2.0,
            "t_stat": 3.54,
            "source": "iter21 LaLiga vol06",
        },
        "odds_col": "odds_1x2_h",
        "edge_col": "edge_1x2_h",
        "prob_col": "p_home",
        "fair_odds_col": None,
        "book_odds_col": "book_h",
        "book_edge_col": "edge_1x2_h_vs_book",
        "american_col": "pin_h_american",
    },
    {
        "id": "SerieA_away_ml",
        "name": "Serie A Away ML",
        "pack": "SerieA_away_ml_exp",
        "league": "SerieA",
        "market": "1X2 Away",
        "side": "A",
        "min_odds": 1.01,
        "max_odds": 2.00,
        "edge_thr": 0.03,
        "status": "paper",
        "live": True,
        "rules_text": "Away @ edge ≥3%, max odds 2.00",
        "history": {
            "n": 246,
            "roi": 0.124,
            "hit": None,
            "seasons_pos": 10,
            "seasons_n": 10,
            "max_dd_u": 0.0,
            "t_stat": 2.41,
            "source": "iter21 SerieA vol06",
        },
        "odds_col": "odds_1x2_a",
        "edge_col": "edge_1x2_a",
        "prob_col": "p_away",
        "fair_odds_col": None,
        "book_odds_col": "book_a",
        "book_edge_col": "edge_1x2_a_vs_book",
        "american_col": "pin_a_american",
    },
]

PAPER_SYSTEMS: list[dict[str, Any]] = [
    {
        "id": "Primeira_ah_e12",
        "name": "Primeira Liga AH e12%",
        "pack": "PrimeiraLiga_ah_e12_exp",
        "league": "PrimeiraLiga",
        "market": "AH",
        "side": "best",  # either side meeting filters
        "min_odds": 1.01,
        "max_odds": 1.90,
        "edge_thr": 0.12,
        "status": "paper_live",
        "live": True,  # primary live scan pack when Pin AH present
        "rules_text": "AH @ edge ≥12%, max odds 1.90",
        "history": {
            "n": 161,
            "roi": 0.192,
            "hit": None,
            "seasons_pos": 10,
            "seasons_n": 10,
            "max_dd_u": -3.7,
            "t_stat": 3.05,
            "source": "iter25 promote · e12 primary (e10-only slice flat) · Pin closes preferred",
        },
        "odds_col": "pin_ahh",  # evaluated per-side in scan
        "edge_col": "edge_ah_home",
        "prob_col": "p_ah_home",
        "fair_odds_col": "fair_odds_ah_home",
        "book_odds_col": "book_ahh",
        "book_edge_col": "edge_ah_home_vs_book",
        "american_col": "pin_ahh_american",
    },
    {
        "id": "Primeira_ah_short",
        "name": "Primeira Liga AH e10% (wider sibling)",
        "pack": "PrimeiraLiga_ah_short_exp",
        "league": "PrimeiraLiga",
        "market": "AH",
        "side": "best",
        "min_odds": 1.01,
        "max_odds": 1.90,
        "edge_thr": 0.10,
        "status": "paper_sibling",
        "live": False,  # optional wider sibling — not in default live scan
        "rules_text": "AH @ edge ≥10%, max odds 1.90 (wider optional sibling)",
        "history": {
            "n": 242,
            "roi": 0.126,
            "hit": None,
            "seasons_pos": 8,
            "seasons_n": 10,
            "max_dd_u": -6.8,
            "t_stat": 2.46,
            "source": "iter23/25 · superseded as primary by e12; keep for optional wider scans",
        },
        "odds_col": "pin_ahh",
        "edge_col": "edge_ah_home",
        "prob_col": "p_ah_home",
        "fair_odds_col": "fair_odds_ah_home",
        "book_odds_col": "book_ahh",
        "book_edge_col": "edge_ah_home_vs_book",
        "american_col": "pin_ahh_american",
    },
]


def live_systems() -> list[dict[str, Any]]:
    return [s for s in PROTECTED_SYSTEMS if s.get("live")] + [
        s for s in PAPER_SYSTEMS if s.get("live")
    ]


def protected_live_only() -> list[dict[str, Any]]:
    return list(PROTECTED_SYSTEMS)


def get_system(name_or_id: str) -> dict[str, Any] | None:
    key = name_or_id.strip().lower()
    for s in PROTECTED_SYSTEMS + PAPER_SYSTEMS:
        if s["id"].lower() == key or s["name"].lower() == key or s["pack"].lower() == key:
            return s
    return None


def history_summary(sys_: dict[str, Any]) -> str:
    h = sys_.get("history") or {}
    n = h.get("n")
    roi = h.get("roi")
    sp, sn = h.get("seasons_pos"), h.get("seasons_n")
    dd = h.get("max_dd_u")
    parts = []
    if n is not None and roi is not None:
        parts.append(f"n={n} ROI={100*roi:+.1f}%")
    if sp is not None and sn is not None:
        parts.append(f"seasons+={sp}/{sn}")
    if dd is not None:
        parts.append(f"maxDD={dd:+.1f}u")
    return " · ".join(parts) if parts else "—"
