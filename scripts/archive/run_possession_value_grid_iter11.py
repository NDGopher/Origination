#!/usr/bin/env python
"""Possession-value (OBV-lite) intensity grid focused on OU — iter11."""

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
from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.features.possession_value import (
    enrich_matches_with_possession_value,
)
from origination.utils import (
    load_config,
    resolve_data_dir,
    resolve_experiments_dir,
    set_global_seed,
    setup_logging,
)


PV_PATH = ROOT / "data" / "interim" / "understat_possession_value.parquet"

VARIANTS = [
    {"label": "iter11_baseline", "possession_value": False, "pv_coef": 0.0},
    {"label": "iter11_pv_c0p05", "possession_value": True, "pv_coef": 0.05},
    {"label": "iter11_pv_c0p10", "possession_value": True, "pv_coef": 0.10},
    {"label": "iter11_pv_c0p15", "possession_value": True, "pv_coef": 0.15},
    {"label": "iter11_pv_c0p20", "possession_value": True, "pv_coef": 0.20},
]


def _market_roi(bets: pd.DataFrame, market: str) -> dict:
    sub = bets[bets["market"] == market] if bets is not None and len(bets) else bets
    if sub is None or len(sub) == 0:
        return {f"n_{market}": 0, f"roi_{market}": None, f"hit_{market}": None}
    st = float(sub["stake"].sum())
    return {
        f"n_{market}": int(len(sub)),
        f"roi_{market}": float(sub["profit"].sum()) / st if st else None,
        f"hit_{market}": float(sub["won"].mean()),
    }


def main() -> None:
    setup_logging("INFO")
    if not PV_PATH.exists():
        raise SystemExit(f"Missing {PV_PATH}; run scripts/build_possession_value.py")

    base = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(base)
    exp_dir = resolve_experiments_dir(base)
    set_global_seed(int(base.get("project", {}).get("seed", 42)))

    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    if base.get("features", {}).get("groups", {}).get("understat_advanced", False):
        hist = load_understat_team_history(data_dir / "raw" / "understat")
        matches = enrich_matches_with_understat_advanced(matches, hist)

    pv = pd.read_parquet(PV_PATH)
    matches = enrich_matches_with_possession_value(matches, pv)

    # Keep 1X2 filter on for ROI realism
    filter_rules = {
        "enabled": True,
        "rules": [{"markets": ["1x2"], "max_odds": 1.80}],
    }

    rows = []
    for v in VARIANTS:
        cfg = copy.deepcopy(base)
        cfg["project"]["experiment_label"] = v["label"]
        cfg["features"]["groups"]["possession_value"] = bool(v["possession_value"])
        cfg["model"]["dixon_coles"]["intensity_adjustments"]["pv_coef"] = float(v["pv_coef"])
        cfg["backtest"]["bet_filters"] = filter_rules

        print(f"\n=== {v['label']} pv_coef={v['pv_coef']} ===")
        result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
        s = result.summary
        row = {
            "label": v["label"],
            "experiment_id": result.experiment_id,
            "pv_coef": v["pv_coef"],
            "log_loss_1x2": s.get("log_loss_1x2"),
            "log_loss_ou25": s.get("log_loss_ou25"),
            "log_loss_market_ou25": s.get("log_loss_market_ou25"),
            "log_loss_edge_vs_market_ou25": s.get("log_loss_edge_vs_market_ou25"),
            "roi_all": s.get("roi"),
            "n_bets": s.get("n_bets"),
        }
        row.update(_market_roi(result.bets, "1x2"))
        row.update(_market_roi(result.bets, "ou25"))
        row.update(_market_roi(result.bets, "ah"))
        rows.append(row)
        print(json.dumps(row, indent=2, default=str))

        if v["label"] in ("iter11_baseline", "iter11_pv_c0p10"):
            mm = build_multi_market_report(
                result.predictions, matches, cfg.get("backtest", {}), label=v["label"]
            )
            save_multi_market_report(mm, exp_dir / f"multi_market_{v['label']}")

    out = exp_dir / "possession_value_comparison_iter11.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
