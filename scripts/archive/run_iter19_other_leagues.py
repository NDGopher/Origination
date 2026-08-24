#!/usr/bin/env python
"""
Iter19 — Other-league totals hunt (EPL packs protected / untouched).

1) Diagnosis per league (model bias, totals corr, market sharpness proxy)
2) Wide dedicated filter grids (Unders & Overs separately) on existing preds
3) Season stability for any candidate with ROI>0 and n>=80

Does NOT modify EPL_aggressive or EPL_overs_short_exp rules.
"""

from __future__ import annotations

import itertools
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
from origination.utils.odds import two_way_fair

OUT = ROOT / "experiments" / "iter19_other_leagues"

# Best available preds. EPL included only as protected control (not to re-tune).
ARTIFACTS = [
    {
        "league": "EPL",
        "label": "vol06",
        "experiment_id": "20260810T201358Z_iter17_EPL_vol06",
        "aligned": "matches_aligned.parquet",
        "role": "control_protected",
    },
    {
        "league": "Bundesliga",
        "label": "vol06",
        "experiment_id": "20260810T214911Z_iter17_Bundesliga_vol06",
        "aligned": "matches_aligned_D1.parquet",
        "role": "primary",
    },
    {
        "league": "Bundesliga",
        "label": "thresh05",
        "experiment_id": "20260810T174732Z_iter16_Bundesliga_int_thresh05",
        "aligned": "matches_aligned_D1.parquet",
        "role": "primary",
    },
    {
        "league": "LaLiga",
        "label": "base_signed",
        "experiment_id": "20260810T223204Z_iter17_LaLiga_base",
        "aligned": "matches_aligned_SP1.parquet",
        "role": "primary",
    },
    {
        "league": "LaLiga",
        "label": "vol06",
        "experiment_id": "20260810T224919Z_iter17_LaLiga_vol06",
        "aligned": "matches_aligned_SP1.parquet",
        "role": "primary",
    },
    {
        "league": "Championship",
        "label": "legacy_aug5",
        "experiment_id": "20260805T191312Z_league_E1_champ",
        "aligned": "matches_aligned_E1.parquet",
        "role": "stale_goals_only",
    },
    {
        "league": "SerieA",
        "label": "legacy_aug5",
        "experiment_id": "20260805T194400Z_league_I1_serie_a",
        "aligned": "matches_aligned_I1.parquet",
        "role": "stale_pre_intercept",
    },
    {
        "league": "Championship",
        "label": "fresh_thresh",
        "experiment_id": "20260811T164035Z_iter19_Championship_thresh_intercept",
        "aligned": "matches_aligned_E1.parquet",
        "role": "primary_fresh",
    },
    {
        "league": "SerieA",
        "label": "fresh_signed",
        "experiment_id": "20260811T165028Z_iter19_SerieA_signed_intercept",
        "aligned": "matches_aligned_I1.parquet",
        "role": "primary_fresh",
    },
]

# Protected EPL packs (score only, never mutate)
EPL_UNDER_PACK = {
    "enabled": True,
    "rules": [
        {"markets": ["1x2"], "max_odds": 1.80},
        {"markets": ["ou25"], "min_odds": 2.00, "max_odds": 4.00, "allow_sides": ["under"]},
        {"markets": ["ah"], "max_odds": 1.90},
    ],
}
EPL_OVER_PACK = {
    "enabled": True,
    "rules": [
        {"markets": ["ou25"], "min_odds": 1.60, "max_odds": 2.50, "allow_sides": ["over"]},
    ],
}

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


def _score_ou(
    preds: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    side: str,
    min_odds: float,
    max_odds: float,
    edge: float,
) -> pd.DataFrame:
    filt = {
        "enabled": True,
        "rules": [
            {
                "markets": ["ou25"],
                "min_odds": min_odds,
                "max_odds": max_odds,
                "allow_sides": [side],
            }
        ],
    }
    bt = {
        "markets": ["ou25"],
        "edge_threshold": edge,
        "edge_threshold_by_market": {"ou25": edge},
        "bet_filters": filt,
        "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
        "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
    }
    bets = evaluate_predictions(preds, matches, bt, edge_threshold=edge)
    return bets[(bets["market"] == "ou25") & (bets["side"] == side)].copy()


def diagnose(preds: pd.DataFrame, matches: pd.DataFrame, league: str, label: str) -> dict:
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
        fair_o = fair_u = np.nan
        if pd.notna(o_odds) and pd.notna(u_odds) and float(o_odds) > 1 and float(u_odds) > 1:
            fair_o, fair_u = two_way_fair(float(o_odds), float(u_odds), method="power")
        rows.append(
            {
                "sum_lambda": lam + mu,
                "total_goals": float(tg),
                "p_over25": p_o,
                "actual_over": float(tg) > 2.5,
                "edge_over": p_o - fair_o if np.isfinite(fair_o) else np.nan,
                "close_over": float(o_odds) if pd.notna(o_odds) else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return {"league": league, "label": label, "n": 0}

    # Brier / calibration for OU
    brier = float(((df["p_over25"] - df["actual_over"].astype(float)) ** 2).mean())
    # Market favorite hit (shorter OU price)
    mkt = df.dropna(subset=["close_over"])
    # Prefer under when over odds > under odds roughly — use fair edge sign
    # Sharpness proxy: |claimed edge| vs realized — correlation of edge_over with actual_over
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
        "brier_ou25": brier,
        "mean_abs_edge_over": float(edge["edge_over"].abs().mean()) if len(edge) else None,
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
    control_rows = []
    candidate_detail = []

    for art in ARTIFACTS:
        pred_path = ROOT / "experiments" / art["experiment_id"] / "predictions.parquet"
        aligned = data_dir / "interim" / art["aligned"]
        if not pred_path.exists() or not aligned.exists():
            print(f"SKIP missing {art['league']} {art['label']}")
            continue
        preds = pd.read_parquet(pred_path)
        matches = load_aligned(aligned)
        print(f"=== {art['league']} / {art['label']} ({art['role']}) n_pred={len(preds)} ===")

        diag = diagnose(preds, matches, art["league"], art["label"])
        diag["role"] = art["role"]
        diag["experiment_id"] = art["experiment_id"]
        diag_rows.append(diag)

        # Protected EPL control only
        if art["league"] == "EPL":
            for name, pack, edge in [
                ("EPL_aggressive_unders", EPL_UNDER_PACK, 0.08),
                ("EPL_overs_short_exp", EPL_OVER_PACK, 0.10),
            ]:
                bt = {
                    "markets": ["1x2", "ou25", "ah"] if "aggressive" in name else ["ou25"],
                    "edge_threshold": 0.03,
                    "edge_threshold_by_market": {"ou25": edge, "ah": 0.05},
                    "bet_filters": pack,
                    "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
                    "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
                }
                bets = evaluate_predictions(preds, matches, bt)
                ou = bets[(bets["market"] == "ou25")]
                if "unders" in name:
                    ou = ou[ou["side"] == "under"]
                else:
                    ou = ou[ou["side"] == "over"]
                sp, sn = _season_pos(ou)
                control_rows.append(
                    {
                        "pack": name,
                        "n": int(len(ou)),
                        "roi": _roi(ou),
                        "hit": _hit(ou),
                        "seasons_pos": sp,
                        "seasons_n": sn,
                        "note": "PROTECTED — do not modify",
                    }
                )
            continue  # do not grid-search new EPL packs

        # Dedicated grids (non-EPL only)
        for side, bands in [("under", UNDER_BANDS), ("over", OVER_BANDS)]:
            for (lo, hi), edge in itertools.product(bands, EDGES):
                bets = _score_ou(preds, matches, side=side, min_odds=lo, max_odds=hi, edge=edge)
                sp, sn = _season_pos(bets)
                row = {
                    "league": art["league"],
                    "label": art["label"],
                    "role": art["role"],
                    "side": side,
                    "min_odds": lo,
                    "max_odds": hi,
                    "edge": edge,
                    "n": int(len(bets)),
                    "roi": _roi(bets),
                    "hit": _hit(bets),
                    "avg_odds": float(bets["close_odds"].mean()) if len(bets) else None,
                    "avg_edge": float(bets["edge"].mean()) if len(bets) else None,
                    "seasons_pos": sp,
                    "seasons_n": sn,
                    "units": float(bets["profit"].sum()) if len(bets) else 0.0,
                }
                grid_rows.append(row)

                # Keep promising for seasonal table
                if (
                    row["n"] >= 80
                    and row["roi"] is not None
                    and row["roi"] >= 0.02
                    and sn >= 5
                    and sp >= max(3, int(0.55 * sn))
                ):
                    # season detail
                    seasons = []
                    for season, g in bets.groupby("season"):
                        seasons.append(
                            {
                                "league": art["league"],
                                "label": art["label"],
                                "side": side,
                                "band": f"{lo:.2f}-{hi:.2f}",
                                "edge": edge,
                                "season": int(season),
                                "n": int(len(g)),
                                "roi": _roi(g),
                                "hit": _hit(g),
                            }
                        )
                    candidate_detail.extend(seasons)

    diag_df = pd.DataFrame(diag_rows)
    grid_df = pd.DataFrame(grid_rows)
    ctrl_df = pd.DataFrame(control_rows)
    cand_df = pd.DataFrame(candidate_detail)

    diag_df.to_csv(OUT / "diagnosis.csv", index=False)
    grid_df.to_csv(OUT / "filter_grid.csv", index=False)
    ctrl_df.to_csv(OUT / "epl_control_protected.csv", index=False)
    if len(cand_df):
        cand_df.to_csv(OUT / "candidate_seasons.csv", index=False)

    # Rank candidates
    if len(grid_df):
        g = grid_df.dropna(subset=["roi"]).copy()
        g = g[g["n"] >= 80]
        g["stability"] = g.apply(
            lambda r: (r["seasons_pos"] / r["seasons_n"]) if r["seasons_n"] else 0.0,
            axis=1,
        )
        # Prefer primary-role modern models
        g["role_rank"] = g["role"].map({"primary": 0, "stale_goals_only": 2, "stale_pre_intercept": 1}).fillna(3)
        top = g.sort_values(
            ["role_rank", "roi", "stability", "n"],
            ascending=[True, False, False, False],
        ).head(40)
        top.to_csv(OUT / "top_candidates.csv", index=False)
    else:
        top = pd.DataFrame()

    # Markdown report
    lines = [
        "# Iter19 — Other-league totals hunt",
        "",
        "EPL packs **untouched**. Control scored only.",
        "",
        "## EPL control (protected)",
        "",
        "| Pack | n | ROI | Hit | Seasons + |",
        "|------|--:|----:|----:|----------:|",
    ]
    for _, r in ctrl_df.iterrows():
        lines.append(
            f"| {r['pack']} | {r['n']} | {100*(r['roi'] or 0):+.1f}% | "
            f"{100*(r['hit'] or 0):.1f}% | {r['seasons_pos']}/{r['seasons_n']} |"
        )

    lines += [
        "",
        "## Diagnosis (why other leagues are hard)",
        "",
        "| League | Model | n | Goals bias (λ−act) | corr(λ,goals) | Over% act/model | Brier OU | corr(edge,over) |",
        "|--------|-------|--:|-------------------:|--------------:|----------------:|---------:|----------------:|",
    ]
    for _, r in diag_df.iterrows():
        ce = r.get("corr_edge_vs_actual_over")
        ce_s = f"{ce:.3f}" if ce is not None and pd.notna(ce) else "—"
        lines.append(
            f"| {r['league']} | {r['label']} | {r['n']} | {r['goals_bias']:+.3f} | "
            f"{r['corr_lambda_goals']:.3f} | "
            f"{100*r['over_rate_actual']:.1f}/{100*r['over_rate_model']:.1f} | "
            f"{r['brier_ou25']:.4f} | {ce_s} |"
        )

    lines += [
        "",
        "### Reading the diagnosis",
        "",
        "- **Goals bias > 0**: model over-projects totals (Unders-friendly if priced).",
        "- **Goals bias < 0**: model under-projects (Overs-friendly if priced).",
        "- **corr(λ,goals)** low → weak ranking; filters cannot invent edge.",
        "- **corr(edge,over)** near 0 → claimed edges do not track outcomes.",
        "",
        "## Top filter candidates (n≥80, ROI≥+2%, multi-season majority)",
        "",
    ]
    if len(top):
        lines += [
            "| League | Model | Side | Band | Edge | n | ROI | Hit | Seasons + | Role |",
            "|--------|-------|------|------|-----:|--:|----:|----:|----------:|------|",
        ]
        for _, r in top.head(25).iterrows():
            lines.append(
                f"| {r['league']} | {r['label']} | {r['side']} | "
                f"{r['min_odds']:.2f}–{r['max_odds']:.2f} | {100*r['edge']:.0f}% | "
                f"{int(r['n'])} | {100*r['roi']:+.1f}% | {100*(r['hit'] or 0):.1f}% | "
                f"{int(r['seasons_pos'])}/{int(r['seasons_n'])} | {r['role']} |"
            )
    else:
        lines.append("_No candidates met the stability bar on available artifacts._")

    # Per-league best
    lines += ["", "## Best per league (any role, n≥80)", ""]
    if len(grid_df):
        g2 = grid_df.dropna(subset=["roi"])
        g2 = g2[g2["n"] >= 80]
        lines += [
            "| League | Best | Side | Band | e | n | ROI | Seasons + |",
            "|--------|------|------|------|--:|--:|----:|----------:|",
        ]
        for league, gg in g2.groupby("league"):
            r = gg.sort_values("roi", ascending=False).iloc[0]
            lines.append(
                f"| {league} | {r['label']} | {r['side']} | "
                f"{r['min_odds']:.2f}–{r['max_odds']:.2f} | {100*r['edge']:.0f}% | "
                f"{int(r['n'])} | {100*r['roi']:+.1f}% | "
                f"{int(r['seasons_pos'])}/{int(r['seasons_n'])} |"
            )

    lines += [
        "",
        "## Honest verdict / next steps",
        "",
        "- Championship & Serie A preds are **pre-iter14 stack** (stale). Treat positive filters as hypotheses until fresh WF.",
        "- Bundesliga / La Liga use modern vol06 / intercept stacks — filter results are decision-grade.",
        "- Recommend: (1) promote any D1/SP1 candidate with ≥6/10 seasons; "
        "(2) run fresh Championship + Serie A WF with totals intercept + allow stack; "
        "(3) never merge with EPL packs.",
        "",
        f"Artifacts: `{OUT.as_posix()}`",
        "",
    ]

    report = "\n".join(lines)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    sys.stdout.buffer.write(report.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
