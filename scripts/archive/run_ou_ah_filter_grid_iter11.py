#!/usr/bin/env python
"""Grid OU/AH bet filters on top of 1X2 max_odds=1.80 (iter11)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.utils import load_config, resolve_data_dir, setup_logging


RULE_SETS = [
    {
        "label": "1x2_only",
        "rules": [{"markets": ["1x2"], "max_odds": 1.80}],
    },
    {
        "label": "ou_max_2.20",
        "rules": [
            {"markets": ["1x2"], "max_odds": 1.80},
            {"markets": ["ou25"], "max_odds": 2.20},
        ],
    },
    {
        "label": "ou_max_2.00",
        "rules": [
            {"markets": ["1x2"], "max_odds": 1.80},
            {"markets": ["ou25"], "max_odds": 2.00},
        ],
    },
    {
        "label": "ou_under_only",
        "rules": [
            {"markets": ["1x2"], "max_odds": 1.80},
            {"markets": ["ou25"], "allow_sides": ["under"]},
        ],
    },
    {
        "label": "ou_over_only",
        "rules": [
            {"markets": ["1x2"], "max_odds": 1.80},
            {"markets": ["ou25"], "allow_sides": ["over"]},
        ],
    },
    {
        "label": "ou_fav_side",
        "rules": [
            {"markets": ["1x2"], "max_odds": 1.80},
            {"markets": ["ou25"], "require_ou_favorite": True},
        ],
    },
    {
        "label": "ou_fav_max_2.20",
        "rules": [
            {"markets": ["1x2"], "max_odds": 1.80},
            {"markets": ["ou25"], "require_ou_favorite": True, "max_odds": 2.20},
        ],
    },
    {
        "label": "ah_max_2.10",
        "rules": [
            {"markets": ["1x2"], "max_odds": 1.80},
            {"markets": ["ah"], "max_odds": 2.10},
        ],
    },
    {
        "label": "ah_max_1.95",
        "rules": [
            {"markets": ["1x2"], "max_odds": 1.80},
            {"markets": ["ah"], "max_odds": 1.95},
        ],
    },
    {
        "label": "ou_under_ah_max_2.10",
        "rules": [
            {"markets": ["1x2"], "max_odds": 1.80},
            {"markets": ["ou25"], "allow_sides": ["under"], "max_odds": 2.20},
            {"markets": ["ah"], "max_odds": 2.10},
        ],
    },
    {
        "label": "stack_tight",
        "rules": [
            {"markets": ["1x2"], "max_odds": 1.80},
            {"markets": ["ou25"], "require_ou_favorite": True, "max_odds": 2.00},
            {"markets": ["ah"], "max_odds": 1.95},
        ],
    },
]


SPECS = [
    ("EPL", "20260805T212804Z_iter10_hier_baseline", "matches_aligned.parquet"),
    ("Championship", "20260805T200446Z_league_E1_champ_iter8", "matches_aligned_E1.parquet"),
    ("Bundesliga", "20260805T193621Z_league_D1_xg_resid", "matches_aligned_D1.parquet"),
    ("SerieA", "20260805T194400Z_league_I1_serie_a", "matches_aligned_I1.parquet"),
    ("LaLiga", "20260805T195244Z_league_SP1_la_liga", "matches_aligned_SP1.parquet"),
]


def _roi(df: pd.DataFrame) -> float:
    if df is None or len(df) == 0:
        return float("nan")
    st = float(df["stake"].sum())
    return float(df["profit"].sum()) / st if st else float("nan")


def main() -> None:
    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    out_dir = ROOT / "experiments" / "ou_ah_filter_grid_iter11"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for league, eid, aligned in SPECS:
        exp = ROOT / "experiments" / eid
        if not (exp / "predictions.parquet").exists():
            print("SKIP", league)
            continue
        preds = pd.read_parquet(exp / "predictions.parquet")
        matches = load_aligned(data_dir / "interim" / aligned)
        for rs in RULE_SETS:
            bt = copy.deepcopy(cfg.get("backtest", {}))
            bt["bet_filters"] = {"enabled": True, "rules": rs["rules"]}
            bets = evaluate_predictions(preds, matches, bt, edge_threshold=0.03)
            b1 = bets[bets["market"] == "1x2"] if len(bets) else bets
            bo = bets[bets["market"] == "ou25"] if len(bets) else bets
            ba = bets[bets["market"] == "ah"] if len(bets) else bets
            row = {
                "league": league,
                "label": rs["label"],
                "n_all": int(len(bets)),
                "roi_all": _roi(bets),
                "n_1x2": int(len(b1)),
                "roi_1x2": _roi(b1),
                "n_ou": int(len(bo)),
                "roi_ou": _roi(bo),
                "hit_ou": float(bo["won"].mean()) if len(bo) else np.nan,
                "avg_odds_ou": float(bo["close_odds"].mean()) if len(bo) else np.nan,
                "n_ah": int(len(ba)),
                "roi_ah": _roi(ba),
                "hit_ah": float(ba["won"].mean()) if len(ba) else np.nan,
            }
            rows.append(row)
            if league == "EPL":
                print(
                    f"EPL {rs['label']:22s} OU n={row['n_ou']:4d} ROI={row['roi_ou']:+.2%}  "
                    f"AH n={row['n_ah']:4d} ROI={row['roi_ah']:+.2%}  ALL={row['roi_all']:+.2%}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "filter_grid.csv", index=False)
    epl = df[df["league"] == "EPL"].sort_values("roi_ou", ascending=False)
    lines = ["# OU/AH filter grid (iter11)\n\n", "## EPL sorted by OU ROI\n\n", epl.to_string(index=False), "\n"]
    (out_dir / "SUMMARY.md").write_text("".join(lines), encoding="utf-8")
    print("Wrote", out_dir / "filter_grid.csv")


if __name__ == "__main__":
    main()
