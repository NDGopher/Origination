#!/usr/bin/env python
"""
Iter19 — Fresh walk-forward for Championship + Serie A (modern totals stack).

Does not touch EPL configs or packs. Writes experiment folders under experiments/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger

from origination.backtesting import run_walk_forward, save_experiment
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.utils import load_config, resolve_data_dir, set_global_seed, setup_logging

JOBS = [
    {
        "name": "Championship",
        "config": "configs/league_E1_championship.yaml",
        "aligned": "matches_aligned_E1.parquet",
        "label": "iter19_Championship_thresh_intercept",
        "understat": False,
    },
    {
        "name": "SerieA",
        "config": "configs/league_I1_serie_a.yaml",
        "aligned": "matches_aligned_I1.parquet",
        "label": "iter19_SerieA_signed_intercept",
        "understat": True,
    },
]


def main() -> None:
    setup_logging("INFO")
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    for job in JOBS:
        if only and job["name"] not in only and job["label"] not in only:
            continue
        cfg_path = ROOT / job["config"]
        cfg = load_config(cfg_path)
        set_global_seed(int(cfg.get("project", {}).get("seed", 42)))
        data_dir = resolve_data_dir(cfg)
        exp_dir = resolve_data_dir(cfg).parent / cfg.get("project", {}).get(
            "experiments_dir", "experiments"
        )
        # experiments_dir is usually relative to project root
        exp_dir = ROOT / "experiments"

        matches = load_aligned(data_dir / "interim" / job["aligned"])
        if job["understat"] and cfg.get("features", {}).get("groups", {}).get(
            "understat_advanced", False
        ):
            hist = load_understat_team_history(data_dir / "raw" / "understat")
            matches = enrich_matches_with_understat_advanced(matches, hist)

        cfg.setdefault("project", {})["experiment_label"] = job["label"]
        logger.info("=== WF {} (n={}) ===", job["name"], len(matches))
        result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
        path = save_experiment(result, exp_dir)
        logger.info("Saved {} -> {}", job["name"], path)
        # Quick OU pack probes
        from origination.backtesting.walk_forward import evaluate_predictions

        preds = result.predictions
        for side, lo, hi, edge, tag in [
            ("under", 2.00, 4.00, 0.08, "u_2-4_e8"),
            ("under", 1.80, 3.00, 0.08, "u_18-3_e8"),
            ("over", 1.60, 2.50, 0.10, "o_16-25_e10"),
            ("over", 1.50, 2.50, 0.08, "o_15-25_e8"),
        ]:
            bt = {
                "markets": ["ou25"],
                "edge_threshold": edge,
                "edge_threshold_by_market": {"ou25": edge},
                "bet_filters": {
                    "enabled": True,
                    "rules": [
                        {
                            "markets": ["ou25"],
                            "min_odds": lo,
                            "max_odds": hi,
                            "allow_sides": [side],
                        }
                    ],
                },
                "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
                "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
            }
            bets = evaluate_predictions(preds, matches, bt)
            if len(bets) == 0:
                logger.info("  {} {}: n=0", job["name"], tag)
                continue
            roi = float(bets["profit"].sum() / bets["stake"].sum())
            logger.info("  {} {}: n={} ROI={:+.1%}", job["name"], tag, len(bets), roi)


if __name__ == "__main__":
    main()
