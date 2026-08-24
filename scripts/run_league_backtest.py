#!/usr/bin/env python
"""Walk-forward + multi-market report for a non-EPL league config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from loguru import logger

from origination.backtesting import run_walk_forward
from origination.backtesting.multi_market_report import (
    build_multi_market_report,
    save_multi_market_report,
)
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.utils import load_config, resolve_data_dir, resolve_experiments_dir, set_global_seed, setup_logging


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    setup_logging("INFO")
    set_global_seed(42)

    if args.label:
        cfg.setdefault("project", {})["experiment_label"] = args.label

    data_dir = resolve_data_dir(cfg)
    exp_dir = resolve_experiments_dir(cfg)
    align_name = cfg.get("data", {}).get("align", {}).get("output", "matches_aligned.parquet")
    aligned_path = data_dir / "interim" / align_name
    if not aligned_path.exists():
        raise SystemExit(
            f"Missing {aligned_path}. Run: python scripts/update_data.py --config {args.config}"
        )

    matches = load_aligned(aligned_path)
    if cfg.get("features", {}).get("groups", {}).get("understat_advanced", False):
        hist = load_understat_team_history(data_dir / "raw" / "understat")
        matches = enrich_matches_with_understat_advanced(matches, hist)

    logger.info("League run | matches={} | config={}", len(matches), cfg_path.name)
    result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
    print(json.dumps({k: v for k, v in result.summary.items() if k != "by_threshold"}, indent=2, default=str))

    bt = dict(cfg.get("backtest", {}))
    bt["markets"] = ["1x2", "ou25", "ah"]
    code = cfg.get("leagues", [{}])[0].get("code", "XX")
    label = cfg.get("project", {}).get("experiment_label") or f"league_{code}"
    tables = build_multi_market_report(result.predictions, matches, bt, label=label)
    out_dir_name = cfg.get("project", {}).get("multi_league_dir", "multi_league_iter8")
    out = exp_dir / out_dir_name / label
    save_multi_market_report(tables, out)
    save_multi_market_report(tables, exp_dir / result.experiment_id / "multi_market")
    print("Wrote", out)


if __name__ == "__main__":
    main()
