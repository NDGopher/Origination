#!/usr/bin/env python
"""EPL player-strength (squad quality) + mild hierarchical grid — iter9."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting import run_walk_forward
from origination.backtesting.multi_market_report import (
    build_multi_market_report,
    save_multi_market_report,
)
from origination.backtesting.performance_autopsy import (
    build_autopsy_tables,
    save_autopsy,
    write_autopsy_summary,
)
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.features.squad_quality import load_understat_players
from origination.utils import load_config, resolve_data_dir, resolve_experiments_dir, set_global_seed, setup_logging


VARIANTS = [
    {
        "label": "iter9_baseline",
        "lineups": {"enabled": False},
        "hierarchical": {"enabled": False},
        "skip_if_done": True,
    },
    {
        "label": "iter9_squad_c0p05",
        "lineups": {
            "enabled": True,
            "provider": "understat_squad_quality",
            "strength_coef": 0.05,
            "understat_league": "EPL",
            "raw_dir": str(ROOT / "data" / "raw" / "understat"),
            "top_n": 15,
            "delta_scale": 0.15,
        },
        "hierarchical": {"enabled": False},
    },
    {
        "label": "iter9_squad_c0p10",
        "lineups": {
            "enabled": True,
            "provider": "understat_squad_quality",
            "strength_coef": 0.10,
            "understat_league": "EPL",
            "raw_dir": str(ROOT / "data" / "raw" / "understat"),
            "top_n": 15,
            "delta_scale": 0.15,
        },
        "hierarchical": {"enabled": False},
    },
    {
        "label": "iter9_squad_c0p15",
        "lineups": {
            "enabled": True,
            "provider": "understat_squad_quality",
            "strength_coef": 0.15,
            "understat_league": "EPL",
            "raw_dir": str(ROOT / "data" / "raw" / "understat"),
            "top_n": 15,
            "delta_scale": 0.15,
        },
        "hierarchical": {"enabled": False},
    },
    {
        "label": "iter9_hier_s0p05",
        "lineups": {"enabled": False},
        "hierarchical": {
            "enabled": True,
            "share_attack": 0.05,
            "share_defence": 0.05,
            "cross_league": False,
        },
    },
    {
        "label": "iter9_squad_c0p10_hier_s0p05",
        "lineups": {
            "enabled": True,
            "provider": "understat_squad_quality",
            "strength_coef": 0.10,
            "understat_league": "EPL",
            "raw_dir": str(ROOT / "data" / "raw" / "understat"),
            "top_n": 15,
            "delta_scale": 0.15,
        },
        "hierarchical": {
            "enabled": True,
            "share_attack": 0.05,
            "share_defence": 0.05,
            "cross_league": False,
        },
    },
]


def main() -> None:
    setup_logging("INFO")
    set_global_seed(42)
    # Pre-parse players once
    load_understat_players(ROOT / "data" / "raw" / "understat", leagues=["EPL"])

    base = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(base)
    exp_dir = resolve_experiments_dir(base)
    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    hist = load_understat_team_history(data_dir / "raw" / "understat")
    matches = enrich_matches_with_understat_advanced(matches, hist)

    rows = []
    out_mm = exp_dir / "multi_market_iter9"
    out_aut = exp_dir / "autopsy_iter9" / "player_grid"
    out_mm.mkdir(parents=True, exist_ok=True)
    out_aut.mkdir(parents=True, exist_ok=True)

    baseline_preds = None
    baseline_eid = exp_dir / "20260805T202730Z_iter9_baseline"
    if (baseline_eid / "predictions.parquet").exists():
        baseline_preds = pd.read_parquet(baseline_eid / "predictions.parquet")
        print("Loaded existing baseline predictions", baseline_eid.name)

    for v in VARIANTS:
        # Reuse completed baseline experiment
        if v["label"] == "iter9_baseline" and baseline_preds is not None:
            s = json.loads((baseline_eid / "summary.json").read_text())
            rows.append(
                {
                    "label": v["label"],
                    "strength_coef": 0.0,
                    "hierarchical": False,
                    "share": 0.0,
                    "log_loss_1x2": s.get("log_loss_1x2"),
                    "log_loss_edge_vs_market": s.get("log_loss_edge_vs_market"),
                    "log_loss_ou25": s.get("log_loss_ou25"),
                    "log_loss_edge_vs_market_ou25": s.get("log_loss_edge_vs_market_ou25"),
                    "roi": s.get("roi"),
                    "experiment_id": baseline_eid.name,
                }
            )
            print("SKIP rerun baseline")
            continue

        cfg = copy.deepcopy(base)
        cfg.setdefault("project", {})["experiment_label"] = v["label"]
        ctx = cfg.setdefault("features", {}).setdefault("context_adjustments", {})
        ctx["lineups"] = dict(v["lineups"])
        cfg.setdefault("model", {})["hierarchical"] = dict(v["hierarchical"])

        print(f"=== {v['label']} ===")
        result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
        s = result.summary
        row = {
            "label": v["label"],
            "strength_coef": v["lineups"].get("strength_coef", 0.0) if v["lineups"].get("enabled") else 0.0,
            "hierarchical": bool(v["hierarchical"].get("enabled")),
            "share": v["hierarchical"].get("share_attack", 0.0),
            "log_loss_1x2": s.get("log_loss_1x2"),
            "log_loss_edge_vs_market": s.get("log_loss_edge_vs_market"),
            "log_loss_ou25": s.get("log_loss_ou25"),
            "log_loss_edge_vs_market_ou25": s.get("log_loss_edge_vs_market_ou25"),
            "roi": s.get("roi"),
            "experiment_id": result.experiment_id,
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in row if k != "experiment_id"}, indent=2))

        bt = dict(cfg.get("backtest", {}))
        bt["markets"] = ["1x2", "ou25", "ah"]
        tables = build_multi_market_report(result.predictions, matches, bt, label=v["label"])
        save_multi_market_report(tables, out_mm / v["label"])

        aut = build_autopsy_tables(
            result.bets,
            matches,
            label=v["label"],
            base_preds=baseline_preds,
            resid_preds=result.predictions if baseline_preds is not None else None,
        )
        save_autopsy(aut, out_aut / v["label"])
        write_autopsy_summary(aut, out_aut / v["label"] / "SUMMARY.md", label=v["label"])

    cmp = pd.DataFrame(rows).sort_values("log_loss_1x2")
    cmp.to_csv(exp_dir / "player_strength_comparison_iter9.csv", index=False)
    print(cmp.to_string(index=False))
    print("Wrote", exp_dir / "player_strength_comparison_iter9.csv")


if __name__ == "__main__":
    main()
