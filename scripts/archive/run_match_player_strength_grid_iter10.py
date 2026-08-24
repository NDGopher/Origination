#!/usr/bin/env python
"""Match-level player strength coef grid on hierarchical stack — iter10."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting import run_walk_forward, save_experiment
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
from origination.utils import (
    load_config,
    resolve_data_dir,
    resolve_experiments_dir,
    set_global_seed,
    setup_logging,
)


ROSTERS = ROOT / "data" / "interim" / "understat_match_rosters.parquet"

VARIANTS = [
    {
        "label": "iter10_hier_baseline",
        "lineups": {"enabled": False},
        "bet_filters": {"enabled": False},
    },
    {
        "label": "iter10_match_c0p03",
        "lineups": {
            "enabled": True,
            "provider": "understat_match_players",
            "strength_coef": 0.03,
            "rosters_parquet": str(ROSTERS),
            "delta_scale": 0.15,
            "xi_size": 11,
            "starter_minutes": 45,
        },
        "bet_filters": {"enabled": False},
    },
    {
        "label": "iter10_match_c0p05",
        "lineups": {
            "enabled": True,
            "provider": "understat_match_players",
            "strength_coef": 0.05,
            "rosters_parquet": str(ROSTERS),
            "delta_scale": 0.15,
            "xi_size": 11,
            "starter_minutes": 45,
        },
        "bet_filters": {"enabled": False},
    },
    {
        "label": "iter10_match_c0p08",
        "lineups": {
            "enabled": True,
            "provider": "understat_match_players",
            "strength_coef": 0.08,
            "rosters_parquet": str(ROSTERS),
            "delta_scale": 0.15,
            "xi_size": 11,
            "starter_minutes": 45,
        },
        "bet_filters": {"enabled": False},
    },
    {
        "label": "iter10_match_c0p05_fav_max2p7",
        "lineups": {
            "enabled": True,
            "provider": "understat_match_players",
            "strength_coef": 0.05,
            "rosters_parquet": str(ROSTERS),
            "delta_scale": 0.15,
            "xi_size": 11,
            "starter_minutes": 45,
        },
        "bet_filters": {
            "enabled": True,
            "apply_markets": ["1x2"],
            "max_odds": 2.70,
            "require_market_favorite": True,
        },
    },
]


def main() -> None:
    setup_logging("INFO")
    if not ROSTERS.exists():
        raise SystemExit(
            f"Missing rosters parquet: {ROSTERS}\n"
            "Run scripts/ingest_understat_match_rosters.py first"
        )

    base = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(base)
    exp_dir = resolve_experiments_dir(base)
    set_global_seed(int(base.get("project", {}).get("seed", 42)))

    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    if base.get("features", {}).get("groups", {}).get("understat_advanced", False):
        hist = load_understat_team_history(data_dir / "raw" / "understat")
        matches = enrich_matches_with_understat_advanced(matches, hist)

    rows = []
    for v in VARIANTS:
        cfg = copy.deepcopy(base)
        cfg["project"]["experiment_label"] = v["label"]
        lineups = copy.deepcopy(cfg["features"]["context_adjustments"]["lineups"])
        lineups.update(v["lineups"])
        cfg["features"]["context_adjustments"]["lineups"] = lineups
        if v.get("bet_filters") is not None:
            cfg.setdefault("backtest", {})["bet_filters"] = v["bet_filters"]

        print(f"\n=== {v['label']} ===")
        result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
        save_experiment(result, exp_dir)
        s = result.summary
        row = {
            "label": v["label"],
            "experiment_id": result.experiment_id,
            "log_loss_1x2": s.get("log_loss_1x2"),
            "log_loss_market_1x2": s.get("log_loss_market_1x2"),
            "log_loss_edge_vs_market": s.get("log_loss_edge_vs_market"),
            "log_loss_ou25": s.get("log_loss_ou25"),
            "roi": s.get("roi"),
            "n_bets": s.get("n_bets"),
            "hit_rate": s.get("hit_rate"),
        }
        bets = result.bets
        if bets is not None and len(bets):
            b1 = bets[bets["market"] == "1x2"]
            if len(b1):
                st = float(b1["stake"].sum())
                row["roi_1x2"] = float(b1["profit"].sum()) / st if st else None
                row["n_bets_1x2"] = int(len(b1))
                row["hit_rate_1x2"] = float(b1["won"].mean())
        rows.append(row)
        print(json.dumps(row, indent=2, default=str))

        # Autopsy + multi-market for key variants
        if "c0p05" in v["label"] or v["label"] == "iter10_hier_baseline":
            out = exp_dir / f"autopsy_iter10_{v['label']}"
            tables = build_autopsy_tables(
                result.bets, matches, label=v["label"], resid_preds=result.predictions
            )
            save_autopsy(tables, out)
            write_autopsy_summary(tables, out / "SUMMARY.md", label=v["label"])

            mm = build_multi_market_report(
                result.predictions,
                matches,
                cfg.get("backtest", {}),
                label=v["label"],
            )
            save_multi_market_report(mm, exp_dir / f"multi_market_iter10_{v['label']}")

    out = exp_dir / "match_player_strength_comparison_iter10.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
