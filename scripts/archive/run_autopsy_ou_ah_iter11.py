#!/usr/bin/env python
"""OU + AH deep autopsy AFTER 1X2 max_odds filter (iter11)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting.performance_autopsy import (
    build_autopsy_tables,
    save_autopsy,
    summarize_bets,
    write_autopsy_summary,
)
from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.utils import load_config, resolve_data_dir, setup_logging


SPECS = [
    {
        "label": "EPL",
        "experiment_id": "20260805T212804Z_iter10_hier_baseline",
        "aligned": "matches_aligned.parquet",
    },
    {
        "label": "Championship",
        "experiment_id": "20260805T200446Z_league_E1_champ_iter8",
        "aligned": "matches_aligned_E1.parquet",
    },
    {
        "label": "Bundesliga",
        "experiment_id": "20260805T193621Z_league_D1_xg_resid",
        "aligned": "matches_aligned_D1.parquet",
    },
    {
        "label": "SerieA",
        "experiment_id": "20260805T194400Z_league_I1_serie_a",
        "aligned": "matches_aligned_I1.parquet",
    },
    {
        "label": "LaLiga",
        "experiment_id": "20260805T195244Z_league_SP1_la_liga",
        "aligned": "matches_aligned_SP1.parquet",
    },
]


def _filter_1x2_short(bt: dict) -> dict:
    out = copy.deepcopy(bt)
    out["bet_filters"] = {
        "enabled": True,
        "apply_markets": ["1x2"],
        "max_odds": 1.80,
        "require_market_favorite": False,
        "block_draws": False,
    }
    return out


def main() -> None:
    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    exp_root = ROOT / "experiments"
    out_root = exp_root / "autopsy_iter11_ou_ah"
    out_root.mkdir(parents=True, exist_ok=True)

    rollup_rows = []
    drag_parts = []
    pos_parts = []

    for spec in SPECS:
        eid = exp_root / spec["experiment_id"]
        if not (eid / "predictions.parquet").exists():
            print("SKIP missing", eid)
            continue
        preds = pd.read_parquet(eid / "predictions.parquet")
        matches = load_aligned(data_dir / "interim" / spec["aligned"])
        bt = _filter_1x2_short(cfg.get("backtest", {}))
        bets = evaluate_predictions(preds, matches, bt, edge_threshold=0.03)

        # Focus tables on OU + AH (keep 1X2 in overall for context)
        ou_ah = bets[bets["market"].isin(["ou25", "ah"])].copy()
        label = spec["label"]
        league_dir = out_root / label
        tables_all = build_autopsy_tables(bets, matches, label=f"{label}_filtered_book")
        save_autopsy(tables_all, league_dir / "full_book")
        write_autopsy_summary(
            tables_all, league_dir / "full_book" / "SUMMARY.md", label=f"{label} full (1X2≤1.80)"
        )

        tables_ou_ah = build_autopsy_tables(ou_ah, matches, label=f"{label}_ou_ah")
        save_autopsy(tables_ou_ah, league_dir / "ou_ah_only")
        write_autopsy_summary(
            tables_ou_ah, league_dir / "ou_ah_only" / "SUMMARY.md", label=f"{label} OU+AH only"
        )

        # Extra: over vs under and AH sides with odds buckets
        for mkt in ("ou25", "ah"):
            sub = bets[bets["market"] == mkt]
            s = summarize_bets(sub)
            rollup_rows.append({"league": label, "market": mkt, "segment": "overall", **s})
            if mkt == "ou25":
                for side in ("over", "under"):
                    g = sub[sub["side"] == side]
                    rollup_rows.append(
                        {"league": label, "market": mkt, "segment": f"side={side}", **summarize_bets(g)}
                    )
            if "odds_bucket" in tables_ou_ah:
                ob = tables_ou_ah["odds_bucket"]
                ob = ob[ob["market"] == mkt] if "market" in ob.columns else ob
                for _, r in ob.iterrows():
                    rollup_rows.append(
                        {
                            "league": label,
                            "market": mkt,
                            "segment": f"odds:{r.get('value')}",
                            "n_bets": r.get("n_bets"),
                            "hit_rate": r.get("hit_rate"),
                            "roi": r.get("roi"),
                            "avg_odds": r.get("avg_odds"),
                            "t_stat": r.get("t_stat"),
                        }
                    )

        if "biggest_drags" in tables_ou_ah and len(tables_ou_ah["biggest_drags"]):
            d = tables_ou_ah["biggest_drags"].copy()
            d.insert(0, "league", label)
            drag_parts.append(d.head(15))
        if "closest_positive" in tables_ou_ah and len(tables_ou_ah["closest_positive"]):
            p = tables_ou_ah["closest_positive"].copy()
            p.insert(0, "league", label)
            pos_parts.append(p.head(15))

        print(
            f"{label}: bets={len(bets)} ou={len(bets[bets.market=='ou25'])} "
            f"ah={len(bets[bets.market=='ah'])} "
            f"OU_ROI={summarize_bets(bets[bets.market=='ou25']).get('roi')} "
            f"AH_ROI={summarize_bets(bets[bets.market=='ah']).get('roi')}"
        )

    rollup = pd.DataFrame(rollup_rows)
    rollup.to_csv(out_root / "ROLLUP_segments.csv", index=False)
    if drag_parts:
        pd.concat(drag_parts, ignore_index=True).to_csv(out_root / "biggest_drags.csv", index=False)
    if pos_parts:
        pd.concat(pos_parts, ignore_index=True).to_csv(out_root / "closest_positive.csv", index=False)

    # Human rollup
    lines = [
        "# Iter11 — OU + AH autopsy (after 1X2 max_odds=1.80)\n\n",
        "1X2 short-price filter applied; OU/AH unfiltered in the same book.\n\n",
    ]
    if len(rollup):
        overall = rollup[rollup["segment"] == "overall"][
            ["league", "market", "n_bets", "hit_rate", "roi", "avg_odds", "t_stat"]
        ]
        lines.append("## Overall OU / AH\n\n")
        lines.append(overall.to_string(index=False))
        lines.append("\n\n## Over vs Under\n\n")
        sides = rollup[rollup["segment"].astype(str).str.startswith("side=")]
        if len(sides):
            lines.append(
                sides[["league", "market", "segment", "n_bets", "hit_rate", "roi", "t_stat"]].to_string(
                    index=False
                )
            )
            lines.append("\n")
    (out_root / "ROLLUP.md").write_text("".join(lines), encoding="utf-8")
    print("Wrote", out_root)


if __name__ == "__main__":
    main()
