#!/usr/bin/env python
"""Grid underdog / long-shot bet filters on a saved walk-forward experiment."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting.walk_forward import _summarize, evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.utils import load_config, resolve_data_dir, setup_logging


FILTERS = [
    {"label": "baseline_no_filter", "enabled": False},
    {
        "label": "max_odds_2.70",
        "enabled": True,
        "apply_markets": ["1x2"],
        "max_odds": 2.70,
        "require_market_favorite": False,
    },
    {
        "label": "max_odds_2.00",
        "enabled": True,
        "apply_markets": ["1x2"],
        "max_odds": 2.00,
        "require_market_favorite": False,
    },
    {
        "label": "max_odds_1.80",
        "enabled": True,
        "apply_markets": ["1x2"],
        "max_odds": 1.80,
        "require_market_favorite": False,
    },
    {
        "label": "market_fav_only",
        "enabled": True,
        "apply_markets": ["1x2"],
        "max_odds": None,
        "require_market_favorite": True,
    },
    {
        "label": "fav_and_max_2.70",
        "enabled": True,
        "apply_markets": ["1x2"],
        "max_odds": 2.70,
        "require_market_favorite": True,
    },
    {
        "label": "fav_and_max_2.00",
        "enabled": True,
        "apply_markets": ["1x2"],
        "max_odds": 2.00,
        "require_market_favorite": True,
    },
    {
        "label": "no_draws_max_2.70",
        "enabled": True,
        "apply_markets": ["1x2"],
        "max_odds": 2.70,
        "require_market_favorite": False,
        "block_draws": True,
    },
]


def portfolio_ll(preds: pd.DataFrame, matches: pd.DataFrame, bet_mids: set) -> dict:
    """Log-loss on the subset of matches that still have ≥1 bet."""
    if not bet_mids:
        return {"log_loss_1x2_portfolio": np.nan, "n_portfolio_matches": 0}
    m = matches.set_index("match_id")
    sub = preds[preds["match_id"].isin(bet_mids)]
    ll = []
    for _, row in sub.iterrows():
        if row["match_id"] not in m.index:
            continue
        match = m.loc[row["match_id"]]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        probs = np.array([row["p_home"], row["p_draw"], row["p_away"]], dtype=float)
        probs = probs / probs.sum()
        probs = np.clip(probs, 1e-6, 1.0)
        outcome = {"H": 0, "D": 1, "A": 2}[str(match["ftr"])]
        ll.append(float(-np.log(probs[outcome])))
    return {
        "log_loss_1x2_portfolio": float(np.mean(ll)) if ll else np.nan,
        "n_portfolio_matches": int(len(ll)),
    }


def main() -> None:
    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    exp_id = "20260805T205243Z_iter9_hier_s0p05"
    exp = ROOT / "experiments" / exp_id
    preds = pd.read_parquet(exp / "predictions.parquet")
    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    bt = copy.deepcopy(cfg.get("backtest", {}))

    rows = []
    out_dir = ROOT / "experiments" / "bet_filter_grid_iter10"
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in FILTERS:
        label = f["label"]
        bt_cfg = copy.deepcopy(bt)
        filt = {k: v for k, v in f.items() if k != "label"}
        # Drop None max_odds key so filter ignores it
        if filt.get("max_odds") is None:
            filt.pop("max_odds", None)
        bt_cfg["bet_filters"] = filt
        bets = evaluate_predictions(preds, matches, bt_cfg, edge_threshold=0.03)
        # 1X2-only portfolio view
        bets_1x2 = bets[bets["market"] == "1x2"] if len(bets) else bets
        summary = _summarize(bets_1x2 if len(bets_1x2) else bets, preds, matches)
        port = portfolio_ll(
            preds, matches, set(bets_1x2["match_id"].unique()) if len(bets_1x2) else set()
        )
        stake = float(bets_1x2["stake"].sum()) if len(bets_1x2) else 0.0
        profit = float(bets_1x2["profit"].sum()) if len(bets_1x2) else 0.0
        row = {
            "label": label,
            "filter": json.dumps(filt),
            "n_bets_1x2": int(len(bets_1x2)),
            "n_bets_all": int(len(bets)),
            "hit_rate_1x2": float(bets_1x2["won"].mean()) if len(bets_1x2) else np.nan,
            "roi_1x2": profit / stake if stake else np.nan,
            "units_profit_1x2": profit,
            "avg_odds_1x2": float(bets_1x2["close_odds"].mean()) if len(bets_1x2) else np.nan,
            "avg_edge_1x2": float(bets_1x2["edge"].mean()) if len(bets_1x2) else np.nan,
            "log_loss_1x2_full": summary.get("log_loss_1x2"),
            "log_loss_market_1x2": summary.get("log_loss_market_1x2"),
            **port,
        }
        # Also all-market ROI for context
        if len(bets):
            st = float(bets["stake"].sum())
            row["roi_all_markets"] = float(bets["profit"].sum()) / st if st else np.nan
        else:
            row["roi_all_markets"] = np.nan
        rows.append(row)
        bets.to_parquet(out_dir / f"bets_{label}.parquet", index=False)
        print(
            f"{label:22s} n={row['n_bets_1x2']:5d} hit={row['hit_rate_1x2']:.3f} "
            f"ROI={row['roi_1x2']:+.2%} avg_odds={row['avg_odds_1x2']:.2f} "
            f"portLL={row['log_loss_1x2_portfolio']:.4f}"
            if row["n_bets_1x2"]
            else f"{label}: no bets"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "filter_grid.csv", index=False)
    lines = [
        "# Bet filter grid (iter10)\n",
        f"Source experiment: `{exp_id}` (hier 0.05)\n\n",
        "| " + " | ".join(df.columns) + " |\n",
        "| " + " | ".join(["---"] * len(df.columns)) + " |\n",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in df.columns) + " |\n")
    (out_dir / "SUMMARY.md").write_text("".join(lines), encoding="utf-8")
    print("\nWrote", out_dir / "filter_grid.csv")


if __name__ == "__main__":
    main()
