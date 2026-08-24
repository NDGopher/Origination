#!/usr/bin/env python
"""Possession Value v2 — orthogonal intensity + OU-specialist residual (iter12)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting import run_walk_forward
from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.features.possession_value import (
    build_possession_value_table,
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

# Promoted EPL book filters (iter11/12 candidate — under_max4 if refine promotes)
PROMOTED_FILTERS = {
    "enabled": True,
    "rules": [
        {"markets": ["1x2"], "max_odds": 1.80},
        {
            "markets": ["ou25"],
            "min_odds": 2.00,
            "max_odds": 4.00,
            "allow_sides": ["under"],
            "max_edge": 0.12,
        },
    ],
}
PROMOTED_EDGE = {"ou25": 0.08, "ah": 0.05}


VARIANTS = [
    {
        "label": "iter12_baseline",
        "possession_value": False,
        "pv_coef": 0.0,
        "alpha_1x2": 0.10,
        "alpha_ou": 0.10,
    },
    {
        "label": "iter12_pv_v2_i0p08",
        "possession_value": True,
        "pv_coef": 0.08,
        "pv_center": 0.0,
        "alpha_1x2": 0.10,
        "alpha_ou": 0.10,
    },
    {
        "label": "iter12_pv_v2_i0p12",
        "possession_value": True,
        "pv_coef": 0.12,
        "pv_center": 0.0,
        "alpha_1x2": 0.10,
        "alpha_ou": 0.10,
    },
    {
        "label": "iter12_ou_spec_a0p15",
        "possession_value": True,
        "pv_coef": 0.0,
        "alpha_1x2": 0.0,  # freeze 1X2 residual — OU specialist
        "alpha_ou": 0.15,
    },
    {
        "label": "iter12_ou_spec_a0p20",
        "possession_value": True,
        "pv_coef": 0.0,
        "alpha_1x2": 0.0,
        "alpha_ou": 0.20,
    },
    {
        "label": "iter12_ou_spec_i0p08_a0p15",
        "possession_value": True,
        "pv_coef": 0.08,
        "pv_center": 0.0,
        "alpha_1x2": 0.0,
        "alpha_ou": 0.15,
    },
]


def _roi(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None
    st = float(df["stake"].sum())
    return float(df["profit"].sum()) / st if st else None


def main() -> None:
    setup_logging("INFO")
    base = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(base)
    exp_dir = resolve_experiments_dir(base)
    set_global_seed(int(base.get("project", {}).get("seed", 42)))

    # Rebuild PV table with v2 composite
    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    cache = data_dir / "raw" / "understat" / "match_rosters"
    pv = build_possession_value_table(cache, matches, max_workers=12)
    PV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pv.to_parquet(PV_PATH, index=False)
    print(f"PV v2 table mean_obv={pv['pv_obv'].mean():.3f} mean_v1={pv['pv_obv_v1'].mean():.3f}")

    if base.get("features", {}).get("groups", {}).get("understat_advanced", False):
        hist = load_understat_team_history(data_dir / "raw" / "understat")
        matches = enrich_matches_with_understat_advanced(matches, hist)
    matches = enrich_matches_with_possession_value(matches, pv)

    rows = []
    for v in VARIANTS:
        cfg = copy.deepcopy(base)
        cfg["project"]["experiment_label"] = v["label"]
        cfg["features"]["groups"]["possession_value"] = bool(v["possession_value"])
        cfg["features"]["pv_deep_orth_coef"] = 0.35
        adj = cfg["model"]["dixon_coles"]["intensity_adjustments"]
        adj["pv_coef"] = float(v["pv_coef"])
        adj["pv_center"] = float(v.get("pv_center", 0.0))
        cfg["model"]["residual"]["alpha_1x2"] = float(v["alpha_1x2"])
        cfg["model"]["residual"]["alpha_ou"] = float(v["alpha_ou"])
        # Use refine-friendly filters during WF betting summary; also re-score after
        cfg["backtest"]["bet_filters"] = {
            "enabled": True,
            "rules": [
                {"markets": ["1x2"], "max_odds": 1.80},
                {"markets": ["ou25"], "min_odds": 2.00},
            ],
        }
        cfg["backtest"]["edge_threshold_by_market"] = {"ou25": 0.08, "ah": 0.05}

        print(f"\n=== {v['label']} ===")
        result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
        s = result.summary

        # Re-score with refined under_max4 pack
        bt = copy.deepcopy(cfg["backtest"])
        bt["bet_filters"] = PROMOTED_FILTERS
        bt["edge_threshold_by_market"] = PROMOTED_EDGE
        bets_ref = evaluate_predictions(
            result.predictions, matches, bt, edge_threshold=0.03
        )
        bo = bets_ref[bets_ref["market"] == "ou25"]
        ba = bets_ref[bets_ref["market"] == "ah"]
        b1 = bets_ref[bets_ref["market"] == "1x2"]

        row = {
            "label": v["label"],
            "experiment_id": result.experiment_id,
            "pv_coef": v["pv_coef"],
            "alpha_1x2": v["alpha_1x2"],
            "alpha_ou": v["alpha_ou"],
            "log_loss_1x2": s.get("log_loss_1x2"),
            "log_loss_ou25": s.get("log_loss_ou25"),
            "log_loss_edge_vs_market_ou25": s.get("log_loss_edge_vs_market_ou25"),
            "roi_all_wf": s.get("roi"),
            "n_ou_ref": int(len(bo)),
            "roi_ou_ref": _roi(bo),
            "hit_ou_ref": float(bo["won"].mean()) if len(bo) else None,
            "roi_ah_ref": _roi(ba),
            "roi_1x2_ref": _roi(b1),
            "roi_all_ref": _roi(bets_ref),
        }
        rows.append(row)
        print(json.dumps(row, indent=2, default=str))

    out = exp_dir / "possession_value_v2_comparison_iter12.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
