#!/usr/bin/env python
"""
Iter19 — Full OU filter grid on fresh Championship + Serie A walk-forward preds.
Does not touch EPL packs. Appends diagnosis/grid into experiments/iter19_other_leagues/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.utils import load_config, resolve_data_dir, setup_logging
from origination.utils.odds import two_way_fair

OUT = ROOT / "experiments" / "iter19_other_leagues"

ARTIFACTS = [
    {
        "league": "Championship",
        "label": "fresh_thresh",
        "experiment_id": "20260811T164035Z_iter19_Championship_thresh_intercept",
        "aligned": "matches_aligned_E1.parquet",
    },
    {
        "league": "SerieA",
        "label": "fresh_signed",
        "experiment_id": "20260811T165028Z_iter19_SerieA_signed_intercept",
        "aligned": "matches_aligned_I1.parquet",
    },
]

UNDER_BANDS = [
    (1.70, 2.50),
    (1.80, 3.00),
    (2.00, 3.00),
    (2.00, 4.00),
    (2.20, 3.50),
    (2.50, 4.50),
]
OVER_BANDS = [
    (1.40, 2.00),
    (1.50, 2.20),
    (1.50, 2.50),
    (1.60, 2.50),
    (1.70, 2.80),
    (1.80, 3.00),
    (2.00, 3.50),
]
EDGES = [0.05, 0.08, 0.10, 0.12]


def _roi(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None
    st = float(df["stake"].sum())
    return float(df["profit"].sum()) / st if st else None


def _hit(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None
    return float((df["won"].astype(float) > 0.5).mean())


def _season_pos(df: pd.DataFrame) -> tuple[int, int]:
    if df is None or len(df) == 0 or "season" not in df.columns:
        return 0, 0
    pos = n = 0
    for _, g in df.groupby("season"):
        r = _roi(g)
        if r is None:
            continue
        n += 1
        if r > 0:
            pos += 1
    return pos, n


def _score_ou(preds, matches, *, side, min_odds, max_odds, edge):
    bt = {
        "markets": ["ou25"],
        "edge_threshold": edge,
        "edge_threshold_by_market": {"ou25": edge},
        "bet_filters": {
            "enabled": True,
            "rules": [
                {
                    "markets": ["ou25"],
                    "min_odds": min_odds,
                    "max_odds": max_odds,
                    "allow_sides": [side],
                }
            ],
        },
        "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
        "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
    }
    bets = evaluate_predictions(preds, matches, bt, edge_threshold=edge)
    return bets[(bets["market"] == "ou25") & (bets["side"] == side)].copy()


def diagnose(preds, matches, league, label):
    m = matches.set_index("match_id")
    rows = []
    for _, r in preds.iterrows():
        mid = r["match_id"]
        if mid not in m.index:
            continue
        match = m.loc[mid]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        tg = match.get("total_goals")
        if pd.isna(tg):
            continue
        lam = float(r.get("lambda_home", np.nan))
        mu = float(r.get("lambda_away", np.nan))
        if not (np.isfinite(lam) and np.isfinite(mu)):
            continue
        p_o = float(r["p_over25"])
        o_odds = match.get("close_over25")
        u_odds = match.get("close_under25")
        fair_o = np.nan
        if pd.notna(o_odds) and pd.notna(u_odds) and float(o_odds) > 1 and float(u_odds) > 1:
            fair_o, _ = two_way_fair(float(o_odds), float(u_odds), method="power")
        rows.append(
            {
                "sum_lambda": lam + mu,
                "total_goals": float(tg),
                "p_over25": p_o,
                "actual_over": float(tg) > 2.5,
                "edge_over": p_o - fair_o if np.isfinite(fair_o) else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    edge = df.dropna(subset=["edge_over"])
    corr_edge = (
        float(np.corrcoef(edge["edge_over"], edge["actual_over"].astype(float))[0, 1])
        if len(edge) > 50
        else None
    )
    return {
        "league": league,
        "label": label,
        "n": int(len(df)),
        "mean_goals": float(df["total_goals"].mean()),
        "mean_sum_lambda": float(df["sum_lambda"].mean()),
        "goals_bias": float(df["sum_lambda"].mean() - df["total_goals"].mean()),
        "corr_lambda_goals": float(np.corrcoef(df["sum_lambda"], df["total_goals"])[0, 1]),
        "over_rate_actual": float(df["actual_over"].mean()),
        "over_rate_model": float(df["p_over25"].mean()),
        "brier_ou25": float(((df["p_over25"] - df["actual_over"].astype(float)) ** 2).mean()),
        "corr_edge_vs_actual_over": corr_edge,
        "pct_high_totals_ge4": float((df["total_goals"] >= 4).mean()),
        "pct_low_totals_le1": float((df["total_goals"] <= 1).mean()),
    }


def main() -> None:
    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    OUT.mkdir(parents=True, exist_ok=True)

    diag_rows = []
    grid_rows = []
    season_dumps = []

    for art in ARTIFACTS:
        pred_path = ROOT / "experiments" / art["experiment_id"] / "predictions.parquet"
        aligned = data_dir / "interim" / art["aligned"]
        preds = pd.read_parquet(pred_path)
        matches = load_aligned(aligned)
        print(f"=== {art['league']} / {art['label']} n={len(preds)} ===")
        diag = diagnose(preds, matches, art["league"], art["label"])
        diag["experiment_id"] = art["experiment_id"]
        diag_rows.append(diag)
        print(
            f"  bias={diag['goals_bias']:+.3f} corr_lam={diag['corr_lambda_goals']:.3f} "
            f"corr_edge={diag['corr_edge_vs_actual_over']}"
        )

        for side, bands in [("under", UNDER_BANDS), ("over", OVER_BANDS)]:
            for lo, hi in bands:
                for edge in EDGES:
                    bets = _score_ou(
                        preds, matches, side=side, min_odds=lo, max_odds=hi, edge=edge
                    )
                    roi = _roi(bets)
                    hit = _hit(bets)
                    spos, sn = _season_pos(bets)
                    tag = f"{art['league']}_{art['label']}_{side}_{lo}-{hi}_e{int(edge*100)}"
                    row = {
                        "tag": tag,
                        "league": art["league"],
                        "label": art["label"],
                        "side": side,
                        "min_odds": lo,
                        "max_odds": hi,
                        "edge": edge,
                        "n": len(bets),
                        "roi": roi,
                        "hit": hit,
                        "seasons_pos": spos,
                        "seasons_n": sn,
                        "experiment_id": art["experiment_id"],
                    }
                    grid_rows.append(row)
                    if roi is not None and roi >= 0.02 and len(bets) >= 80 and spos >= max(1, sn // 2 + 1):
                        print(
                            f"  CAND {tag}: n={len(bets)} ROI={roi:+.1%} "
                            f"hit={hit:.1%} seas={spos}/{sn}"
                        )
                        if "season" in bets.columns:
                            seas = (
                                bets.groupby("season")
                                .apply(
                                    lambda g: pd.Series(
                                        {"n": len(g), "roi": _roi(g), "hit": _hit(g)}
                                    ),
                                    include_groups=False,
                                )
                                .reset_index()
                            )
                            seas["tag"] = tag
                            season_dumps.append(seas)

    diag_df = pd.DataFrame(diag_rows)
    grid_df = pd.DataFrame(grid_rows)
    diag_df.to_csv(OUT / "fresh_diagnosis.csv", index=False)
    grid_df.to_csv(OUT / "fresh_filter_grid.csv", index=False)

    # Merge into master diagnosis / filter_grid if present
    master_d = OUT / "diagnosis.csv"
    master_g = OUT / "filter_grid.csv"
    if master_d.exists():
        old = pd.read_csv(master_d)
        # drop prior fresh rows if re-run
        old = old[~old["label"].isin(["fresh_thresh", "fresh_signed"])]
        pd.concat([old, diag_df], ignore_index=True).to_csv(master_d, index=False)
    if master_g.exists():
        old = pd.read_csv(master_g)
        old = old[~old["label"].isin(["fresh_thresh", "fresh_signed"])]
        pd.concat([old, grid_df], ignore_index=True).to_csv(master_g, index=False)

    prom = grid_df[
        (grid_df["n"] >= 80)
        & (grid_df["roi"].notna())
        & (grid_df["roi"] >= 0.02)
        & (grid_df["seasons_pos"] >= (grid_df["seasons_n"] // 2 + 1))
    ].sort_values("roi", ascending=False)
    prom.to_csv(OUT / "fresh_promising.csv", index=False)
    print("\n=== PROMISING (n>=80, ROI>=2%, majority seasons) ===")
    if len(prom) == 0:
        print("  (none)")
    else:
        print(prom.to_string(index=False))

    # Top 15 by ROI regardless of season majority (n>=60)
    top = (
        grid_df[(grid_df["n"] >= 60) & grid_df["roi"].notna()]
        .sort_values("roi", ascending=False)
        .head(20)
    )
    top.to_csv(OUT / "fresh_top20.csv", index=False)
    print("\n=== TOP 20 by ROI (n>=60) ===")
    print(top[["tag", "n", "roi", "hit", "seasons_pos", "seasons_n"]].to_string(index=False))

    if season_dumps:
        pd.concat(season_dumps, ignore_index=True).to_csv(
            OUT / "fresh_candidate_seasons.csv", index=False
        )


if __name__ == "__main__":
    main()
