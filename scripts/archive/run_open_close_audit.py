#!/usr/bin/env python
"""Open→close CLV audit on current best walk-forward predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from loguru import logger

from origination.backtesting.open_close_audit import run_open_close_audit, save_open_close_audit
from origination.data_ingestion.align import load_aligned
from origination.utils import load_config, resolve_data_dir, resolve_experiments_dir, setup_logging


def main() -> None:
    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    exp_dir = resolve_experiments_dir(cfg)

    # Prefer promoted iter5 combo experiment
    candidates = sorted(exp_dir.glob("*iter5_combo_ou_temp_rest*"))
    if not candidates:
        candidates = sorted(exp_dir.glob("*motiv_stakes*"))
    if not candidates:
        raise SystemExit("No experiment predictions found")
    exp = candidates[-1]
    preds = pd.read_parquet(exp / "predictions.parquet")
    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    # Align to prediction universe
    matches = matches[matches["match_id"].isin(preds["match_id"])].copy()

    bt = dict(cfg.get("backtest", {}))
    bt["markets"] = ["1x2", "ou25", "ah"]
    tables = run_open_close_audit(preds, matches, bt, label=exp.name)
    out = save_open_close_audit(tables, exp_dir / "open_close_audit_iter6")
    # also copy into experiment
    save_open_close_audit(tables, exp / "open_close_audit")

    print("Experiment:", exp.name)
    print("\n=== INTERPRETATION ===")
    print(tables["interpretation"].to_string(index=False))
    print("\n=== LOG LOSS ===")
    print(tables["log_loss"].to_string(index=False))
    print("\n=== BETS @ 3% open vs close ===")
    b = tables["bets_by_source"]
    print(b[b.edge_threshold == 0.03].to_string(index=False))
    print("\n=== OPEN-SELECTED CLV ===")
    print(tables["open_selected_clv"].to_string(index=False))
    print("\n=== LINE MOVES ===")
    print(tables["line_moves"].to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
