#!/usr/bin/env python
"""EPL additive residual enrichment grid (interactions / deeper LGBM)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting import run_walk_forward
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.utils import load_config, resolve_data_dir, resolve_experiments_dir, set_global_seed, setup_logging


VARIANTS = [
    {
        "label": "iter8_resid_baseline",
        "residual": {},  # keep default α=0.10
    },
    {
        "label": "iter8_resid_interactions",
        "residual": {"interactions": True},
    },
    {
        "label": "iter8_resid_deeper",
        "residual": {
            "params": {
                "n_estimators": 400,
                "learning_rate": 0.04,
                "num_leaves": 31,
                "min_child_samples": 40,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.7,
                "reg_lambda": 1.5,
                "seed": 42,
            }
        },
    },
    {
        "label": "iter8_resid_ix_deeper",
        "residual": {
            "interactions": True,
            "params": {
                "n_estimators": 400,
                "learning_rate": 0.04,
                "num_leaves": 31,
                "min_child_samples": 40,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.7,
                "reg_lambda": 1.5,
                "seed": 42,
            },
        },
    },
    {
        "label": "iter8_resid_ix_a0p08",
        "residual": {
            "interactions": True,
            "alpha_1x2": 0.08,
            "alpha_ou": 0.08,
        },
    },
]


def main() -> None:
    setup_logging("INFO")
    set_global_seed(42)
    base = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(base)
    exp_dir = resolve_experiments_dir(base)
    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    hist = load_understat_team_history(data_dir / "raw" / "understat")
    matches = enrich_matches_with_understat_advanced(matches, hist)

    rows = []
    for v in VARIANTS:
        cfg = copy.deepcopy(base)
        cfg.setdefault("project", {})["experiment_label"] = v["label"]
        r = cfg.setdefault("model", {}).setdefault("residual", {})
        for k, val in v["residual"].items():
            if k == "params":
                r["params"] = {**r.get("params", {}), **val}
            else:
                r[k] = val
        print(f"=== {v['label']} ===")
        result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
        s = result.summary
        rows.append(
            {
                "label": v["label"],
                "interactions": bool(r.get("interactions", False)),
                "alpha_1x2": r.get("alpha_1x2", 0.1),
                "num_leaves": r.get("params", {}).get("num_leaves"),
                "n_estimators": r.get("params", {}).get("n_estimators"),
                "log_loss_1x2": s.get("log_loss_1x2"),
                "log_loss_edge_vs_market": s.get("log_loss_edge_vs_market"),
                "log_loss_ou25": s.get("log_loss_ou25"),
                "log_loss_edge_vs_market_ou25": s.get("log_loss_edge_vs_market_ou25"),
                "roi": s.get("roi"),
                "experiment_id": result.experiment_id,
            }
        )
        print(json.dumps({k: rows[-1][k] for k in rows[-1] if k != "experiment_id"}, indent=2))

    out = exp_dir / "residual_enrichment_iter8.csv"
    pd.DataFrame(rows).sort_values("log_loss_1x2").to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
