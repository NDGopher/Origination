#!/usr/bin/env python
"""
Iter13: totals modeling + elite layer alone/combo study (EPL WF).

Layers under test (alone + selected combos):
- hierarchical shrink (on/off)
- possession_value v2 intensity (off / 0.05)
- residual OU alpha (0.10 / 0.15) with 1X2 alpha held at 0.10
- OU-specialist residual (alpha_1x2=0, alpha_ou=0.15)

Evaluation uses *mild universal* filters (not EPL under-only pack) plus
odds-band ROI for 1.60–2.00 and 1.50–3.00.
"""

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

# Mild universal book for scoring (not EPL-specific under-only)
MILD_FILTERS = {
    "enabled": True,
    "rules": [
        {"markets": ["1x2"], "max_odds": 2.00},
        {"markets": ["ou25"], "min_odds": 1.50, "max_odds": 3.00},
        {"markets": ["ah"], "min_odds": 1.50, "max_odds": 3.00},
    ],
}
MILD_EDGE = {"ou25": 0.05, "ah": 0.05}

# Also score EPL pack for contrast
EPL_PACK = {
    "enabled": True,
    "rules": [
        {"markets": ["1x2"], "max_odds": 1.80},
        {"markets": ["ou25"], "min_odds": 2.00, "max_odds": 4.00, "allow_sides": ["under"]},
        {"markets": ["ah"], "max_odds": 1.90},
    ],
}
EPL_EDGE = {"ou25": 0.08, "ah": 0.05}


VARIANTS = [
    # Baselines / single layers
    {
        "label": "L_base",
        "hierarchical": True,
        "possession_value": False,
        "pv_coef": 0.0,
        "alpha_1x2": 0.10,
        "alpha_ou": 0.10,
    },
    {
        "label": "L_hier_off",
        "hierarchical": False,
        "possession_value": False,
        "pv_coef": 0.0,
        "alpha_1x2": 0.10,
        "alpha_ou": 0.10,
    },
    {
        "label": "L_pv_i0p05",
        "hierarchical": True,
        "possession_value": True,
        "pv_coef": 0.05,
        "alpha_1x2": 0.10,
        "alpha_ou": 0.10,
    },
    {
        "label": "L_ou_a0p15",
        "hierarchical": True,
        "possession_value": False,
        "pv_coef": 0.0,
        "alpha_1x2": 0.10,
        "alpha_ou": 0.15,
    },
    {
        "label": "L_ou_spec",
        "hierarchical": True,
        "possession_value": True,
        "pv_coef": 0.0,
        "alpha_1x2": 0.0,
        "alpha_ou": 0.15,
    },
    # Combinations
    {
        "label": "C_hier_ou_a0p15",
        "hierarchical": True,
        "possession_value": False,
        "pv_coef": 0.0,
        "alpha_1x2": 0.10,
        "alpha_ou": 0.15,
    },
    {
        "label": "C_pv_ou_a0p15",
        "hierarchical": True,
        "possession_value": True,
        "pv_coef": 0.05,
        "alpha_1x2": 0.10,
        "alpha_ou": 0.15,
    },
    {
        "label": "C_hier_off_ou_a0p15",
        "hierarchical": False,
        "possession_value": False,
        "pv_coef": 0.0,
        "alpha_1x2": 0.10,
        "alpha_ou": 0.15,
    },
]


def _roi(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None
    st = float(df["stake"].sum())
    return float(df["profit"].sum()) / st if st else None


def _band(df: pd.DataFrame, lo: float, hi: float) -> tuple[int, float | None]:
    if df is None or len(df) == 0:
        return 0, None
    sub = df[(df["close_odds"] >= lo) & (df["close_odds"] <= hi)]
    return int(len(sub)), _roi(sub)


def score_book(preds, matches, filt, edge) -> dict:
    bt = {
        "markets": ["1x2", "ou25", "ah"],
        "edge_threshold": 0.03,
        "edge_threshold_by_market": edge,
        "bet_filters": filt,
        "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
        "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
    }
    bets = evaluate_predictions(preds, matches, bt, edge_threshold=0.03)
    bo = bets[bets["market"] == "ou25"]
    n_short, r_short = _band(bo, 1.60, 2.00)
    n_mid, r_mid = _band(bo, 1.50, 3.00)
    return {
        "n_ou": int(len(bo)),
        "roi_ou": _roi(bo),
        "n_ou_1p6_2p0": n_short,
        "roi_ou_1p6_2p0": r_short,
        "n_ou_1p5_3p0": n_mid,
        "roi_ou_1p5_3p0": r_mid,
        "roi_all": _roi(bets),
        "n_all": int(len(bets)),
    }


def main() -> None:
    setup_logging("INFO")
    base = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(base)
    exp_dir = resolve_experiments_dir(base)
    set_global_seed(42)

    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    if base.get("features", {}).get("groups", {}).get("understat_advanced", False):
        hist = load_understat_team_history(data_dir / "raw" / "understat")
        matches = enrich_matches_with_understat_advanced(matches, hist)

    # Ensure PV table exists (v2 composite)
    if not PV_PATH.exists():
        cache = data_dir / "raw" / "understat" / "match_rosters"
        pv = build_possession_value_table(cache, matches, max_workers=12)
        pv.to_parquet(PV_PATH, index=False)
    else:
        pv = pd.read_parquet(PV_PATH)
    matches = enrich_matches_with_possession_value(matches, pv)

    rows = []
    for v in VARIANTS:
        cfg = copy.deepcopy(base)
        cfg["project"]["experiment_label"] = f"iter13_{v['label']}"
        cfg["model"]["hierarchical"]["enabled"] = bool(v["hierarchical"])
        cfg["features"]["groups"]["possession_value"] = bool(v["possession_value"])
        cfg["features"]["pv_deep_orth_coef"] = 0.35
        cfg["model"]["dixon_coles"]["intensity_adjustments"]["pv_coef"] = float(v["pv_coef"])
        cfg["model"]["dixon_coles"]["intensity_adjustments"]["pv_center"] = 0.0
        cfg["model"]["residual"]["alpha_1x2"] = float(v["alpha_1x2"])
        cfg["model"]["residual"]["alpha_ou"] = float(v["alpha_ou"])
        # WF internal betting uses mild filters
        cfg["backtest"]["bet_filters"] = MILD_FILTERS
        cfg["backtest"]["edge_threshold_by_market"] = MILD_EDGE

        print(f"\n=== {v['label']} ===")
        result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
        s = result.summary
        mild = score_book(result.predictions, matches, MILD_FILTERS, MILD_EDGE)
        epl = score_book(result.predictions, matches, EPL_PACK, EPL_EDGE)
        row = {
            "label": v["label"],
            "experiment_id": result.experiment_id,
            "hierarchical": v["hierarchical"],
            "pv_coef": v["pv_coef"],
            "alpha_1x2": v["alpha_1x2"],
            "alpha_ou": v["alpha_ou"],
            "log_loss_1x2": s.get("log_loss_1x2"),
            "log_loss_ou25": s.get("log_loss_ou25"),
            "ou_gap": s.get("log_loss_edge_vs_market_ou25"),
            **{f"mild_{k}": val for k, val in mild.items()},
            **{f"eplpack_{k}": val for k, val in epl.items()},
        }
        rows.append(row)
        print(json.dumps(row, indent=2, default=str))

    out = exp_dir / "iter13_layer_study.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
