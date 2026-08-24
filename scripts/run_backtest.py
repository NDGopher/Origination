#!/usr/bin/env python
"""Run walk-forward CLV backtest and log experiment artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger

from origination.backtesting import run_walk_forward
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.utils import (
    load_config,
    resolve_data_dir,
    resolve_experiments_dir,
    set_global_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward CLV backtest")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--aligned",
        default=None,
        help="Path to matches_aligned.parquet (default: data/interim/matches_aligned.parquet)",
    )
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
    experiments_dir = resolve_experiments_dir(cfg)
    aligned_path = (
        Path(args.aligned) if args.aligned else data_dir / "interim" / "matches_aligned.parquet"
    )
    if not aligned_path.exists():
        raise SystemExit(
            f"Aligned data not found: {aligned_path}\n"
            "Run: python scripts/update_data.py --config configs/default.yaml"
        )

    matches = load_aligned(aligned_path)
    if cfg.get("features", {}).get("groups", {}).get("understat_advanced", False):
        hist = load_understat_team_history(data_dir / "raw" / "understat")
        matches = enrich_matches_with_understat_advanced(matches, hist)

    logger.info("Loaded {} aligned matches from {}", len(matches), aligned_path)

    result = run_walk_forward(matches, cfg, experiments_dir=experiments_dir)
    printable = {k: v for k, v in result.summary.items() if k != "by_threshold"}
    print(json.dumps(printable, indent=2, default=str))
    if len(result.by_threshold):
        print("\nBy edge threshold:")
        print(result.by_threshold.to_string(index=False))
    if len(result.by_season):
        print("\nBy season:")
        print(result.by_season.to_string(index=False))


if __name__ == "__main__":
    main()
