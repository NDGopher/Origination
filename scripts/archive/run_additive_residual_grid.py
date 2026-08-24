#!/usr/bin/env python
"""Grid additive logit residual α on EPL best stack (iter7)."""

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


def _run(cfg0, matches, exp_dir, label, residual_cfg):
    cfg = copy.deepcopy(cfg0)
    cfg["project"]["experiment_label"] = label
    cfg["model"]["residual"] = residual_cfg
    cfg.setdefault("backtest", {})["markets"] = ["1x2", "ou25", "ah"]
    logger.warning("=== {} ===", label)
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
        "mode": residual_cfg.get("mode"),
        "enabled": residual_cfg.get("enabled"),
        "alpha_1x2": residual_cfg.get("alpha_1x2"),
        "alpha_ou": residual_cfg.get("alpha_ou"),
        "log_loss_1x2": s.get("log_loss_1x2"),
        "log_loss_edge_vs_market": s.get("log_loss_edge_vs_market"),
        "log_loss_ou25": s.get("log_loss_ou25"),
        "log_loss_edge_vs_market_ou25": s.get("log_loss_edge_vs_market_ou25"),
        "roi_mixed": s.get("roi"),
        "ou_roi_3pct": (profit / stake) if stake else None,
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

    base = {
        "enabled": True,
        "mode": "additive",
        "backend": "lightgbm",
        "label_smoothing": 0.02,
        "delta_clip": 2.0,
        "min_oos_rows": 300,
        "max_oos_seasons": 4,
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 50,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "reg_lambda": 1.0,
            "seed": 42,
        },
    }

    specs = [
        ("iter7_dc_only", {**base, "enabled": False, "alpha_1x2": 0.0, "alpha_ou": 0.0}),
        ("iter7_add_a0p05", {**base, "alpha_1x2": 0.05, "alpha_ou": 0.05}),
        ("iter7_add_a0p10", {**base, "alpha_1x2": 0.10, "alpha_ou": 0.10}),
        ("iter7_add_a0p15", {**base, "alpha_1x2": 0.15, "alpha_ou": 0.15}),
        ("iter7_add_a0p20", {**base, "alpha_1x2": 0.20, "alpha_ou": 0.20}),
        ("iter7_add_1x2_a0p10", {**base, "alpha_1x2": 0.10, "alpha_ou": 0.0}),
    ]

    rows, results = [], {}
    for label, rcfg in specs:
        row, result = _run(cfg0, matches, exp_dir, label, rcfg)
        rows.append(row)
        results[label] = result

    cmp = pd.DataFrame(rows).sort_values("log_loss_1x2")
    out = exp_dir / "additive_residual_comparison_iter7.csv"
    cmp.to_csv(out, index=False)
    print("\n=== RANKED BY 1X2 LL ===")
    print(cmp.to_string(index=False))
    print("Wrote", out)

    baseline_ll = float(cmp.loc[cmp["label"] == "iter7_dc_only", "log_loss_1x2"].iloc[0])
    report = ["iter7_dc_only"]
    best = cmp.iloc[0]
    if best["label"] != "iter7_dc_only":
        report.append(best["label"])
    # always include best additive among enabled
    en = cmp[cmp["enabled"] == True]
    if len(en):
        b2 = en.sort_values("log_loss_1x2").iloc[0]["label"]
        if b2 not in report:
            report.append(b2)

    bt = dict(cfg0.get("backtest", {}))
    bt["markets"] = ["1x2", "ou25", "ah"]
    root = exp_dir / "multi_market_iter7"
    parts = []
    for lab in report:
        tables = build_multi_market_report(results[lab].predictions, matches, bt, label=lab)
        save_multi_market_report(tables, root / lab)
        parts.append(tables["summary"])
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(root / "combined_summary.csv", index=False)

    # Promotion hint
    winners = cmp[(cmp["enabled"] == True) & (cmp["log_loss_1x2"] <= baseline_ll + 1e-9)]
    print("\nPromote candidates (LL <= baseline):")
    print(winners.to_string(index=False) if len(winners) else "none")


if __name__ == "__main__":
    main()
