#!/usr/bin/env python
"""Autopsy on filtered 1X2 portfolio (iter10)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting.performance_autopsy import (
    build_autopsy_tables,
    save_autopsy,
    write_autopsy_summary,
)
from origination.data_ingestion.align import load_aligned
from origination.utils import load_config, resolve_data_dir, setup_logging


SPECS = [
    ("baseline_no_filter", "bets_baseline_no_filter.parquet"),
    ("max_odds_1.80", "bets_max_odds_1.80.parquet"),
    ("max_odds_2.00", "bets_max_odds_2.00.parquet"),
    ("market_fav_only", "bets_market_fav_only.parquet"),
    ("fav_and_max_2.70", "bets_fav_and_max_2.70.parquet"),
]


def main() -> None:
    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    src = ROOT / "experiments" / "bet_filter_grid_iter10"
    out_root = ROOT / "experiments" / "autopsy_iter10_filters"
    out_root.mkdir(parents=True, exist_ok=True)

    for label, fname in SPECS:
        path = src / fname
        if not path.exists():
            print("SKIP", path)
            continue
        bets = pd.read_parquet(path)
        # Autopsy wants all markets; filter scripts saved full evaluate output
        tables = build_autopsy_tables(bets, matches, label=label)
        out = out_root / label
        save_autopsy(tables, out)
        write_autopsy_summary(tables, out / "SUMMARY.md", label=label)
        print("Wrote", out)


if __name__ == "__main__":
    main()
