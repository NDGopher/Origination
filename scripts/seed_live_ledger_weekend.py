#!/usr/bin/env python
"""Seed the first settled live play (Sporting AH e12 away, 14 Aug). Rules unchanged."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.gameday.live_ledger import record_from_scan, settle_open, write_report

plays = pd.DataFrame(
    [
        {
            "qualifies": True,
            "system_id": "Primeira_ah_e12",
            "system": "Primeira Liga AH e12%",
            "league": "PrimeiraLiga",
            "match_id": "20260814_SportingCP_VitoriaGuimaraes",
            "date": "2026-08-14",
            "home_team": "Sporting CP",
            "away_team": "Vitoria Guimaraes",
            "market": "AH",
            "side": "ah_away",
            "ah_line": -1.5,
            "pin_odds": 1.8772,
            "edge_vs_pin": 0.29415,
            "edge_thr": 0.12,
            "fair_odds": 1.2278,
            "book_odds": None,
            "recommendation": "PLAY",
        }
    ]
)

n_add = record_from_scan(plays)
n_set = settle_open()
path = write_report()
print(f"seeded={n_add} settled={n_set} report={path}", flush=True)
