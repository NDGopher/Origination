#!/usr/bin/env python
"""Generate probabilities / edges for upcoming fixtures (same path as backtest).

Prefer ``scripts/run_gameday_sheet.py`` for the full OU sheet + pack flags.
This script remains a thin wrapper around ``predict_upcoming``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from loguru import logger

from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.prediction.upcoming import load_odds_csv, predict_upcoming
from origination.utils import load_config, resolve_data_dir, set_global_seed, setup_logging
from origination.utils.seeding import season_from_date


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict upcoming matches")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--fixtures", required=True, help="CSV with match_id,date,home_team,away_team")
    p.add_argument("--odds-file", default=None, help="Optional odds CSV keyed by match_id")
    p.add_argument("--out", default=None, help="Output CSV path")
    p.add_argument("--fast", action="store_true", help="Skip residual OOS fit")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    setup_logging(args.log_level)
    set_global_seed(int(cfg.get("project", {}).get("seed", 42)))
    data_dir = resolve_data_dir(cfg)

    history = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    if cfg.get("features", {}).get("groups", {}).get("understat_advanced", False):
        hist = load_understat_team_history(data_dir / "raw" / "understat")
        history = enrich_matches_with_understat_advanced(history, hist)

    upcoming = pd.read_csv(args.fixtures, parse_dates=["date"])
    if "season" not in upcoming.columns:
        upcoming["season"] = upcoming["date"].map(season_from_date)
    odds = load_odds_csv(Path(args.odds_file)) if args.odds_file else None

    out = predict_upcoming(history, upcoming, cfg, odds=odds, apply_residual=not args.fast)
    out_path = Path(args.out) if args.out else data_dir / "processed" / "predictions_upcoming.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Wrote {}", out_path)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
