#!/usr/bin/env python
"""Deep performance autopsy across EPL + other leagues (iter9)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting.performance_autopsy import (
    build_autopsy_tables,
    save_autopsy,
    write_autopsy_summary,
)
from origination.data_ingestion.align import load_aligned
from origination.utils import load_config, resolve_data_dir, resolve_experiments_dir, setup_logging


# Best known experiments per league (iter8)
SPECS = [
    {
        "label": "EPL",
        "experiment_id": "20260805T200011Z_iter8_resid_ix_deeper",
        "aligned": "matches_aligned.parquet",
        "base_experiment_id": "20260805T193957Z_iter8_resid_baseline",  # shallower residual as proxy base
    },
    {
        "label": "Championship",
        "experiment_id": "20260805T200446Z_league_E1_champ_iter8",
        "aligned": "matches_aligned_E1.parquet",
        "base_experiment_id": None,
    },
    {
        "label": "Bundesliga",
        "experiment_id": "20260805T193621Z_league_D1_xg_resid",
        "aligned": "matches_aligned_D1.parquet",
        "base_experiment_id": None,
    },
    {
        "label": "SerieA",
        "experiment_id": "20260805T194400Z_league_I1_serie_a",
        "aligned": "matches_aligned_I1.parquet",
        "base_experiment_id": None,
    },
    {
        "label": "LaLiga",
        "experiment_id": "20260805T195244Z_league_SP1_la_liga",
        "aligned": "matches_aligned_SP1.parquet",
        "base_experiment_id": None,
    },
]


def main() -> None:
    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    exp_dir = resolve_experiments_dir(cfg)
    out_root = exp_dir / "autopsy_iter9"
    out_root.mkdir(parents=True, exist_ok=True)

    drag_parts = []
    pos_parts = []
    overall_parts = []

    for spec in SPECS:
        eid = exp_dir / spec["experiment_id"]
        if not eid.exists():
            print("SKIP missing", eid)
            continue
        bets = pd.read_parquet(eid / "bets.parquet")
        preds = pd.read_parquet(eid / "predictions.parquet")
        matches = load_aligned(data_dir / "interim" / spec["aligned"])
        base_preds = None
        if spec.get("base_experiment_id"):
            bp = exp_dir / spec["base_experiment_id"] / "predictions.parquet"
            if bp.exists():
                base_preds = pd.read_parquet(bp)

        tables = build_autopsy_tables(
            bets,
            matches,
            label=spec["label"],
            base_preds=base_preds,
            resid_preds=preds if base_preds is not None else None,
            edge_threshold=float(bets["edge_threshold"].iloc[0]) if "edge_threshold" in bets.columns else 0.03,
        )
        league_out = out_root / spec["label"]
        save_autopsy(tables, league_out)
        write_autopsy_summary(tables, league_out / "SUMMARY.md", label=spec["label"])
        if "biggest_drags" in tables:
            d = tables["biggest_drags"].copy()
            d.insert(0, "league", spec["label"])
            drag_parts.append(d)
        if "closest_positive" in tables:
            p = tables["closest_positive"].copy()
            p.insert(0, "league", spec["label"])
            pos_parts.append(p)
        if "overall" in tables:
            o = tables["overall"].copy()
            o.insert(0, "league", spec["label"])
            overall_parts.append(o)
        print(spec["label"], "n_bets", len(bets), "->", league_out)

    if overall_parts:
        pd.concat(overall_parts, ignore_index=True).to_csv(out_root / "overall_by_league.csv", index=False)
    if drag_parts:
        pd.concat(drag_parts, ignore_index=True).to_csv(out_root / "biggest_drags_all.csv", index=False)
    if pos_parts:
        pd.concat(pos_parts, ignore_index=True).to_csv(out_root / "closest_positive_all.csv", index=False)

    # Cross-league narrative
    lines = ["# Iteration 9 — Cross-league autopsy rollup", ""]
    if overall_parts:
        ov = pd.concat(overall_parts, ignore_index=True)
        lines.append("## Overall by league / market")
        lines.append("")
        lines.append(
            ov.query("value in ['1x2','ou25','ah','all']")[
                ["league", "market", "n_bets", "hit_rate", "roi", "avg_odds", "avg_edge", "t_stat"]
            ].to_string(index=False)
        )
        lines.append("")
    if drag_parts:
        dr = pd.concat(drag_parts, ignore_index=True).sort_values("roi").head(20)
        lines.append("## Global biggest ROI drags (n≥50)")
        lines.append("")
        lines.append(
            dr[["league", "market", "segment", "value", "n_bets", "hit_rate", "roi", "avg_odds", "t_stat"]].to_string(
                index=False
            )
        )
        lines.append("")
    if pos_parts:
        po = pd.concat(pos_parts, ignore_index=True).sort_values("roi", ascending=False).head(20)
        lines.append("## Global closest to positive (n≥50)")
        lines.append("")
        lines.append(
            po[["league", "market", "segment", "value", "n_bets", "hit_rate", "roi", "avg_odds", "t_stat"]].to_string(
                index=False
            )
        )
        lines.append("")
    (out_root / "ROLLUP.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out_root / "ROLLUP.md")


if __name__ == "__main__":
    main()
