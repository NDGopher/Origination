#!/usr/bin/env python
"""
Run walk-forward for best + motivation configs, then write multi-market reports.

Usage:
  python scripts/run_multi_market_report.py
  python scripts/run_multi_market_report.py --from-experiment <id> --label best
"""

from __future__ import annotations

import argparse
import copy
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--from-experiment", default=None, help="Reuse predictions from experiment id")
    p.add_argument("--label", default="model")
    p.add_argument("--motivation-coefs", default=None, help="JSON dict of motivation coefs; omit = off")
    p.add_argument("--out-subdir", default="multi_market_iter4")
    return p.parse_args()


def _load_matches(cfg):
    data_dir = resolve_data_dir(cfg)
    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    matches = enrich_matches_with_understat_advanced(
        matches, load_understat_team_history(data_dir / "raw" / "understat")
    )
    return matches


def main() -> None:
    args = parse_args()
    setup_logging("INFO")
    set_global_seed(42)
    cfg0 = load_config(ROOT / "configs" / "default.yaml")
    exp_dir = resolve_experiments_dir(cfg0)
    matches = _load_matches(cfg0)

    bt_cfg = copy.deepcopy(cfg0.get("backtest", {}))
    bt_cfg["markets"] = ["1x2", "ou25", "ah"]

    configs_to_run: list[tuple[str, dict]] = []

    if args.from_experiment:
        configs_to_run = []  # load only
        exp_path = exp_dir / args.from_experiment
        preds = pd.read_parquet(exp_path / "predictions.parquet")
        tables = build_multi_market_report(preds, matches, bt_cfg, label=args.label)
        out = save_multi_market_report(tables, exp_dir / args.out_subdir / args.label)
        print(json.dumps({"out": str(out), "ll": tables["ll_summary"].to_dict(orient="records")}, indent=2, default=str))
        return

    # Best stack (motivation off)
    configs_to_run.append(("best_iter3", {"enabled": False}))

    motiv_coefs = {"enabled": True}
    if args.motivation_coefs:
        motiv_coefs.update(json.loads(args.motivation_coefs))
    else:
        # Default measured candidate from grid winner file if present, else mild stakes
        cmp_path = exp_dir / "motivation_comparison_iter4.csv"
        if cmp_path.exists():
            cmp = pd.read_csv(cmp_path).sort_values("log_loss_1x2")
            # first row that has motivation enabled with any nonzero coef
            for _, r in cmp.iterrows():
                if str(r["label"]) in ("motiv_off", "motiv_feat_only"):
                    continue
                for k in ("title_coef", "releg_coef", "dead_rubber_coef", "stakes_coef", "motivation_diff_coef"):
                    if pd.notna(r.get(k)) and float(r[k]) != 0.0:
                        motiv_coefs[k] = float(r[k])
                motiv_coefs["label_src"] = r["label"]
                break
        if len(motiv_coefs) <= 1:
            motiv_coefs.update({"stakes_coef": 0.05})
    configs_to_run.append(("motivation_best", motiv_coefs))

    all_summary = []
    report_root = exp_dir / args.out_subdir
    for label, motiv in configs_to_run:
        cfg = copy.deepcopy(cfg0)
        cfg["project"]["experiment_label"] = f"mm_{label}"
        cfg.setdefault("backtest", {})["markets"] = ["1x2", "ou25", "ah"]
        ctx = cfg.setdefault("features", {}).setdefault("context_adjustments", {})
        ctx["enabled"] = True
        if motiv.get("enabled", True) is False:
            ctx["motivation"] = {"enabled": False}
        else:
            ctx["motivation"] = {
                "enabled": True,
                "season_length": 38,
                "safety_rank": 17,
                "title_pts_gap": 6.0,
                "releg_pts_gap": 6.0,
                "euro_pts_gap": 6.0,
                "min_games": 8,
                "late_games_left": 12,
                "title_coef": float(motiv.get("title_coef", 0.0) or 0.0),
                "releg_coef": float(motiv.get("releg_coef", 0.0) or 0.0),
                "dead_rubber_coef": float(motiv.get("dead_rubber_coef", 0.0) or 0.0),
                "stakes_coef": float(motiv.get("stakes_coef", 0.0) or 0.0),
                "motivation_diff_coef": float(motiv.get("motivation_diff_coef", 0.0) or 0.0),
            }
        logger.info("Running walk-forward for {} | motivation={}", label, ctx["motivation"])
        result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
        tables = build_multi_market_report(
            result.predictions, matches, bt_cfg, label=label
        )
        save_multi_market_report(tables, report_root / label)
        # Also copy into experiment folder
        save_multi_market_report(tables, exp_dir / result.experiment_id / "multi_market")
        all_summary.append(tables["summary"].assign(experiment_id=result.experiment_id))
        print(json.dumps({
            "label": label,
            "experiment_id": result.experiment_id,
            "ll": result.summary.get("log_loss_1x2"),
            "roi": result.summary.get("roi"),
            "motiv": ctx["motivation"],
        }, indent=2, default=str))

    if all_summary:
        combined = pd.concat(all_summary, ignore_index=True)
        combined.to_csv(report_root / "combined_summary.csv", index=False)
        print("Wrote", report_root / "combined_summary.csv")


if __name__ == "__main__":
    main()
