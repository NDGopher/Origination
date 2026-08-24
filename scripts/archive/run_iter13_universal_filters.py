#!/usr/bin/env python
"""
Iter13: mild universal filters + odds-band diagnostics across 5 leagues.

Compares EPL-specific iter12 pack vs milder packs that keep bets in ~1.50–3.00
and do not hard-require under-only / EPL quirks.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting.performance_autopsy import summarize_bets
from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.utils import load_config, resolve_data_dir, setup_logging


SPECS = [
    ("EPL", "20260805T212804Z_iter10_hier_baseline", "matches_aligned.parquet"),
    ("Championship", "20260805T200446Z_league_E1_champ_iter8", "matches_aligned_E1.parquet"),
    ("Bundesliga", "20260805T193621Z_league_D1_xg_resid", "matches_aligned_D1.parquet"),
    ("SerieA", "20260805T194400Z_league_I1_serie_a", "matches_aligned_I1.parquet"),
    ("LaLiga", "20260805T195244Z_league_SP1_la_liga", "matches_aligned_SP1.parquet"),
]

# Filter packs: mild / universal first; EPL-specific last for contrast
PACKS = [
    {
        "label": "raw_e03",
        "rules": [],
        "edge": {},
        "enabled": False,
    },
    {
        "label": "band_ou_1p5_3p0_e03",
        "rules": [{"markets": ["ou25"], "min_odds": 1.50, "max_odds": 3.00}],
        "edge": {"ou25": 0.03},
    },
    {
        "label": "band_ou_1p5_3p0_e05",
        "rules": [{"markets": ["ou25"], "min_odds": 1.50, "max_odds": 3.00}],
        "edge": {"ou25": 0.05},
    },
    {
        "label": "band_ou_1p5_3p0_e08",
        "rules": [{"markets": ["ou25"], "min_odds": 1.50, "max_odds": 3.00}],
        "edge": {"ou25": 0.08},
    },
    {
        "label": "band_all_1p5_3p0_e05",
        "rules": [
            {"markets": ["1x2"], "min_odds": 1.50, "max_odds": 3.00},
            {"markets": ["ou25"], "min_odds": 1.50, "max_odds": 3.00},
            {"markets": ["ah"], "min_odds": 1.50, "max_odds": 3.00},
        ],
        "edge": {"1x2": 0.03, "ou25": 0.05, "ah": 0.05},
    },
    {
        "label": "mild_1x2_2p0_ou_band_e05",
        "rules": [
            {"markets": ["1x2"], "max_odds": 2.00},
            {"markets": ["ou25"], "min_odds": 1.50, "max_odds": 3.00},
        ],
        "edge": {"ou25": 0.05, "ah": 0.05},
    },
    {
        "label": "mild_1x2_2p0_ou_band_e08",
        "rules": [
            {"markets": ["1x2"], "max_odds": 2.00},
            {"markets": ["ou25"], "min_odds": 1.50, "max_odds": 3.00},
        ],
        "edge": {"ou25": 0.08, "ah": 0.05},
    },
    {
        "label": "short_ou_1p6_2p0_e05",
        "rules": [{"markets": ["ou25"], "min_odds": 1.60, "max_odds": 2.00}],
        "edge": {"ou25": 0.05},
    },
    {
        "label": "short_ou_1p6_2p0_e08",
        "rules": [{"markets": ["ou25"], "min_odds": 1.60, "max_odds": 2.00}],
        "edge": {"ou25": 0.08},
    },
    {
        "label": "iter12_epl_pack",
        "rules": [
            {"markets": ["1x2"], "max_odds": 1.80},
            {"markets": ["ou25"], "min_odds": 2.00, "max_odds": 4.00, "allow_sides": ["under"]},
            {"markets": ["ah"], "max_odds": 1.90},
        ],
        "edge": {"ou25": 0.08, "ah": 0.05},
    },
]


def _roi(df: pd.DataFrame) -> float:
    if df is None or len(df) == 0:
        return float("nan")
    st = float(df["stake"].sum())
    return float(df["profit"].sum()) / st if st else float("nan")


def _band_roi(df: pd.DataFrame, lo: float, hi: float) -> tuple[int, float]:
    if df is None or len(df) == 0:
        return 0, float("nan")
    sub = df[(df["close_odds"] >= lo) & (df["close_odds"] <= hi)]
    return int(len(sub)), _roi(sub)


def eval_pack(preds, matches, cfg, pack: dict) -> pd.DataFrame:
    bt = copy.deepcopy(cfg.get("backtest", {}))
    enabled = pack.get("enabled", True)
    bt["bet_filters"] = {
        "enabled": enabled and bool(pack.get("rules")),
        "rules": pack.get("rules") or [],
    }
    if not enabled:
        bt["bet_filters"] = {"enabled": False}
    bt["edge_threshold_by_market"] = dict(pack.get("edge") or {})
    # Clear aggressive defaults when testing raw/mild
    if pack["label"].startswith("raw") or pack["label"].startswith("band") or pack["label"].startswith("short"):
        if not pack.get("edge"):
            bt["edge_threshold_by_market"] = {}
    return evaluate_predictions(preds, matches, bt, edge_threshold=0.03)


def main() -> None:
    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    out = ROOT / "experiments" / "iter13_universal_filters"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    band_rows = []

    for league, eid, aligned in SPECS:
        exp = ROOT / "experiments" / eid
        if not (exp / "predictions.parquet").exists():
            print("SKIP", league)
            continue
        preds = pd.read_parquet(exp / "predictions.parquet")
        matches = load_aligned(data_dir / "interim" / aligned)

        for pack in PACKS:
            bets = eval_pack(preds, matches, cfg, pack)
            bo = bets[bets["market"] == "ou25"] if len(bets) else bets
            ba = bets[bets["market"] == "ah"] if len(bets) else bets
            b1 = bets[bets["market"] == "1x2"] if len(bets) else bets
            s_ou = summarize_bets(bo)
            row = {
                "league": league,
                "pack": pack["label"],
                "n_ou": int(len(bo)),
                "roi_ou": _roi(bo),
                "hit_ou": float(bo["won"].mean()) if len(bo) else np.nan,
                "avg_odds_ou": float(bo["close_odds"].mean()) if len(bo) else np.nan,
                "t_ou": s_ou.get("t_stat"),
                "n_ah": int(len(ba)),
                "roi_ah": _roi(ba),
                "n_1x2": int(len(b1)),
                "roi_1x2": _roi(b1),
                "n_all": int(len(bets)),
                "roi_all": _roi(bets),
            }
            rows.append(row)

            # Odds-band breakdown on OU bets from this pack
            for lo, hi, name in [
                (1.50, 1.60, "1.50-1.60"),
                (1.60, 2.00, "1.60-2.00"),
                (2.00, 2.50, "2.00-2.50"),
                (2.50, 3.00, "2.50-3.00"),
                (3.00, 4.00, "3.00-4.00"),
                (1.50, 3.00, "1.50-3.00"),
            ]:
                n, r = _band_roi(bo, lo, hi)
                band_rows.append(
                    {
                        "league": league,
                        "pack": pack["label"],
                        "band": name,
                        "n": n,
                        "roi": r,
                    }
                )

            if league == "EPL":
                print(
                    f"EPL {pack['label']:28s} OU n={row['n_ou']:4d} ROI={row['roi_ou']:+.2%} "
                    f"avg={row['avg_odds_ou']:.2f} ALL={row['roi_all']:+.2%}"
                    if row["n_ou"]
                    else f"EPL {pack['label']}: no OU"
                )

    df = pd.DataFrame(rows)
    bands = pd.DataFrame(band_rows)
    df.to_csv(out / "filter_pack_multileague.csv", index=False)
    bands.to_csv(out / "ou_odds_bands.csv", index=False)

    # Cross-league summary: mean OU ROI / worst league for each pack
    summary_lines = ["# Iter13 mild universal filters\n\n"]
    for pack_label, g in df.groupby("pack"):
        ou = g[["league", "n_ou", "roi_ou", "roi_all"]].copy()
        mean_ou = float(ou["roi_ou"].mean())
        min_ou = float(ou["roi_ou"].min())
        n_pos = int((ou["roi_ou"] > 0).sum())
        summary_lines.append(
            f"## `{pack_label}`\n"
            f"- mean OU ROI across leagues: **{mean_ou:+.2%}** | worst: **{min_ou:+.2%}** | "
            f"leagues +EV OU: {n_pos}/5\n\n"
        )
        summary_lines.append(ou.to_string(index=False) + "\n\n")

    # Short-band focus table (1.60-2.00) for key packs
    short = bands[bands["band"] == "1.60-2.00"]
    summary_lines.append("## Short OU band 1.60–2.00\n\n")
    pivot = short.pivot_table(index="pack", columns="league", values="roi", aggfunc="first")
    summary_lines.append(pivot.to_string() + "\n\n")
    n_pivot = short.pivot_table(index="pack", columns="league", values="n", aggfunc="first")
    summary_lines.append("### n\n\n" + n_pivot.to_string() + "\n")

    (out / "SUMMARY.md").write_text("".join(summary_lines), encoding="utf-8")

    # Pick best mild pack by: maximize mean OU ROI subject to worst > -8% and mean avg odds in band
    mild = df[~df["pack"].isin(["iter12_epl_pack", "raw_e03"])].copy()
    rank = (
        mild.groupby("pack")
        .agg(mean_ou=("roi_ou", "mean"), worst_ou=("roi_ou", "min"), mean_all=("roi_all", "mean"))
        .sort_values("mean_ou", ascending=False)
    )
    rank.to_csv(out / "pack_rank_cross_league.csv")
    print("\nCross-league pack rank (mean OU ROI):")
    print(rank.to_string())
    print("Wrote", out)


if __name__ == "__main__":
    main()
