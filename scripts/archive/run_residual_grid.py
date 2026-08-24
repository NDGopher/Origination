#!/usr/bin/env python
"""Measure residual hybrid vs DC-only baseline (iter6)."""

from __future__ import annotations

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
from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.utils import load_config, resolve_data_dir, resolve_experiments_dir, set_global_seed, setup_logging


def _run(cfg0, matches, exp_dir, label: str, residual_cfg: dict | None):
    cfg = copy.deepcopy(cfg0)
    cfg["project"]["experiment_label"] = label
    if residual_cfg is None:
        cfg.setdefault("model", {}).setdefault("residual", {})["enabled"] = False
    else:
        cfg.setdefault("model", {})["residual"] = residual_cfg
    cfg.setdefault("backtest", {})["markets"] = ["1x2", "ou25", "ah"]
    logger.warning("=== {} residual={} ===", label, cfg["model"].get("residual"))
    result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
    s = result.summary
    test_m = matches[matches["match_id"].isin(result.predictions["match_id"])]
    ou = evaluate_predictions(
        result.predictions, test_m, {**cfg["backtest"], "markets": ["ou25"]}, edge_threshold=0.03
    )
    stake = float(ou["stake"].sum()) if len(ou) else 0.0
    profit = float(ou["profit"].sum()) if len(ou) else 0.0
    row = {
        "label": label,
        "residual_enabled": bool(cfg["model"].get("residual", {}).get("enabled")),
        "alpha_1x2": cfg["model"].get("residual", {}).get("alpha_1x2"),
        "alpha_ou": cfg["model"].get("residual", {}).get("alpha_ou"),
        "backend": cfg["model"].get("residual", {}).get("backend"),
        "log_loss_1x2": s.get("log_loss_1x2"),
        "log_loss_edge_vs_market": s.get("log_loss_edge_vs_market"),
        "log_loss_ou25": s.get("log_loss_ou25"),
        "log_loss_edge_vs_market_ou25": s.get("log_loss_edge_vs_market_ou25"),
        "roi_mixed": s.get("roi"),
        "ou_roi_3pct": (profit / stake) if stake else None,
        "ou_n_3pct": len(ou),
        "n_bets": s.get("n_bets"),
        "experiment_id": result.experiment_id,
    }
    print(json.dumps(row, indent=2, default=str))
    return row, result


def main() -> None:
    setup_logging("WARNING")
    set_global_seed(42)
    cfg0 = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg0)
    exp_dir = resolve_experiments_dir(cfg0)
    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    matches = enrich_matches_with_understat_advanced(
        matches, load_understat_team_history(data_dir / "raw" / "understat")
    )

    base_residual = {
        "enabled": True,
        "backend": "lightgbm",
        "min_oos_rows": 400,
        "max_oos_seasons": 4,
        "params": {
            "n_estimators": 250,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 40,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "seed": 42,
        },
    }

    specs = [
        ("iter6_dc_only", None),
        ("iter6_resid_a0p25", {**base_residual, "alpha_1x2": 0.25, "alpha_ou": 0.25}),
        ("iter6_resid_a0p35", {**base_residual, "alpha_1x2": 0.35, "alpha_ou": 0.35}),
        ("iter6_resid_a0p50", {**base_residual, "alpha_1x2": 0.50, "alpha_ou": 0.50}),
        ("iter6_resid_1x2_only", {**base_residual, "alpha_1x2": 0.35, "alpha_ou": 0.0}),
    ]

    rows = []
    results = {}
    for label, rcfg in specs:
        row, result = _run(cfg0, matches, exp_dir, label, rcfg)
        rows.append(row)
        results[label] = result

    cmp = pd.DataFrame(rows).sort_values("log_loss_1x2")
    out = exp_dir / "residual_comparison_iter6.csv"
    cmp.to_csv(out, index=False)
    print("\n=== RANKED BY 1X2 LL ===")
    print(cmp.to_string(index=False))
    print("Wrote", out)

    # Multi-market for baseline + best residual (if any improves)
    baseline_ll = float(cmp.loc[cmp["label"] == "iter6_dc_only", "log_loss_1x2"].iloc[0])
    best = cmp.iloc[0]
    report_labels = ["iter6_dc_only"]
    if best["label"] != "iter6_dc_only" and float(best["log_loss_1x2"]) < baseline_ll - 1e-6:
        report_labels.append(best["label"])
    elif best["label"] != "iter6_dc_only":
        # still report best residual attempt even if not improved
        report_labels.append(str(cmp[cmp["residual_enabled"] == True].sort_values("log_loss_1x2").iloc[0]["label"]))

    bt = dict(cfg0.get("backtest", {}))
    bt["markets"] = ["1x2", "ou25", "ah"]
    report_root = exp_dir / "multi_market_iter6"
    all_sum = []
    for lab in report_labels:
        if lab not in results:
            continue
        tables = build_multi_market_report(results[lab].predictions, matches, bt, label=lab)
        save_multi_market_report(tables, report_root / lab)
        save_multi_market_report(tables, exp_dir / results[lab].experiment_id / "multi_market")
        all_sum.append(tables["summary"])
    if all_sum:
        pd.concat(all_sum, ignore_index=True).to_csv(report_root / "combined_summary.csv", index=False)
        print("Wrote", report_root)


if __name__ == "__main__":
    main()
