#!/usr/bin/env python
"""
Iter20 — Deep Bundesliga Unders investigation.

Primary system: Unders 1.70–2.50 @ edge >= 10% on thresh05.
Also ranks nearby variations (vol06, edges, bands) and writes one report.

Does NOT touch EPL packs or configs.
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

OUT = ROOT / "experiments" / "iter20_bundesliga_deep"
PRIMARY = {
    "label": "thresh05",
    "experiment_id": "20260810T174732Z_iter16_Bundesliga_int_thresh05",
    "side": "under",
    "min_odds": 1.70,
    "max_odds": 2.50,
    "edge": 0.10,
}
MODELS = [
    {
        "label": "thresh05",
        "experiment_id": "20260810T174732Z_iter16_Bundesliga_int_thresh05",
    },
    {
        "label": "vol06",
        "experiment_id": "20260810T214911Z_iter17_Bundesliga_vol06",
    },
]
ALIGNED = "matches_aligned_D1.parquet"

UNDER_BANDS = [
    (1.60, 2.40),
    (1.70, 2.50),
    (1.70, 2.80),
    (1.80, 2.50),
    (1.80, 3.00),
    (2.00, 3.00),
    (2.00, 4.00),
    (2.20, 3.50),
]
OVER_BANDS = [
    (1.40, 2.00),
    (1.50, 2.20),
    (1.60, 2.50),
    (1.70, 2.80),
]
EDGES = [0.05, 0.08, 0.10, 0.12, 0.15]


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


def _enrich(bets: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    if len(bets) == 0:
        return bets
    m = matches.set_index("match_id")
    rows = []
    for _, r in bets.iterrows():
        mid = r["match_id"]
        if mid not in m.index:
            continue
        match = m.loc[mid]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        tg = match.get("total_goals")
        date = pd.to_datetime(match.get("date"))
        # early = first half of season months (Aug–Dec for European leagues)
        month = int(date.month) if pd.notna(date) else None
        early = month in (8, 9, 10, 11, 12) if month else None
        rows.append(
            {
                **r.to_dict(),
                "total_goals": float(tg) if pd.notna(tg) else np.nan,
                "home_goals": float(match.get("home_goals")) if pd.notna(match.get("home_goals")) else np.nan,
                "away_goals": float(match.get("away_goals")) if pd.notna(match.get("away_goals")) else np.nan,
                "month": month,
                "early_season": early,
                "date": date,
            }
        )
    return pd.DataFrame(rows)


def seasonal_table(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cum_u = 0.0
    cum_st = 0.0
    for season, g in bets.groupby("season"):
        st = float(g["stake"].sum())
        pr = float(g["profit"].sum())
        roi = pr / st if st else 0.0
        cum_u += pr
        cum_st += st
        edge_col = "edge" if "edge" in g.columns else None
        rows.append(
            {
                "season": season,
                "n": len(g),
                "hit": _hit(g),
                "roi": roi,
                "units": pr,
                "cum_roi": cum_u / cum_st if cum_st else 0.0,
                "cum_units": cum_u,
                "avg_odds": float(g["close_odds"].mean()) if "close_odds" in g.columns else np.nan,
                "avg_edge": float(g[edge_col].mean()) if edge_col else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bucket_cut(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=False)


def deep_cuts(bets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if len(bets) == 0:
        return out

    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n": len(g),
                "hit": _hit(g),
                "roi": _roi(g),
                "avg_odds": float(g["close_odds"].mean()) if "close_odds" in g.columns else np.nan,
                "avg_edge": float(g["edge"].mean()) if "edge" in g.columns else np.nan,
                "units": float(g["profit"].sum()),
            }
        )

    # Odds buckets inside 1.70–2.50
    odds = bets["close_odds"].astype(float)
    bets = bets.copy()
    bets["odds_bucket"] = bucket_cut(
        odds, [1.70, 1.90, 2.10, 2.30, 2.50, 2.51], ["1.70–1.90", "1.90–2.10", "2.10–2.30", "2.30–2.50", "2.50+"]
    )
    out["by_odds_bucket"] = bets.groupby("odds_bucket", observed=False).apply(agg, include_groups=False).reset_index()

    # Edge size
    if "edge" in bets.columns:
        bets["edge_bucket"] = bucket_cut(
            bets["edge"].astype(float),
            [0.10, 0.12, 0.15, 0.20, 1.0],
            ["10–12%", "12–15%", "15–20%", "20%+"],
        )
        out["by_edge_bucket"] = (
            bets.groupby("edge_bucket", observed=False).apply(agg, include_groups=False).reset_index()
        )

    # Home vs away under — under is match total, so use favorite/home goal share proxies
    # Classify by whether home team is favorite on 1X2 if available; else by λ
    if "lambda_home" in bets.columns and "lambda_away" in bets.columns:
        bets["fav_side"] = np.where(bets["lambda_home"] >= bets["lambda_away"], "home_fav", "away_fav")
        out["by_favorite_side"] = (
            bets.groupby("fav_side", observed=False).apply(agg, include_groups=False).reset_index()
        )

    if "early_season" in bets.columns:
        bets["season_half"] = bets["early_season"].map({True: "early_Aug-Dec", False: "late_Jan-May"})
        out["by_season_half"] = (
            bets.dropna(subset=["season_half"])
            .groupby("season_half", observed=False)
            .apply(agg, include_groups=False)
            .reset_index()
        )

    if "total_goals" in bets.columns:
        bets["env"] = np.where(
            bets["total_goals"] <= 1.5,
            "actual_low_0-1",
            np.where(bets["total_goals"] >= 4, "actual_high_4+", "actual_mid_2-3"),
        )
        # For environment BEFORE the match use model λ sum if present
        if "lambda_home" in bets.columns:
            bets["pred_tot"] = bets["lambda_home"].astype(float) + bets["lambda_away"].astype(float)
            bets["pred_env"] = bucket_cut(
                bets["pred_tot"],
                [0, 2.3, 2.7, 3.2, 10],
                ["pred_low_<2.3", "pred_mid_2.3-2.7", "pred_high_2.7-3.2", "pred_vhigh_>3.2"],
            )
            out["by_pred_total_env"] = (
                bets.groupby("pred_env", observed=False).apply(agg, include_groups=False).reset_index()
            )

        # Failure pattern: losses
        losses = bets[bets["won"].astype(float) < 0.5].copy()
        wins = bets[bets["won"].astype(float) >= 0.5].copy()
        out["failure_vs_win"] = pd.DataFrame(
            [
                {
                    "outcome": "loss",
                    "n": len(losses),
                    "avg_total_goals": float(losses["total_goals"].mean()) if len(losses) else np.nan,
                    "pct_ge4": float((losses["total_goals"] >= 4).mean()) if len(losses) else np.nan,
                    "pct_le1": float((losses["total_goals"] <= 1).mean()) if len(losses) else np.nan,
                    "avg_odds": float(losses["close_odds"].mean()) if len(losses) else np.nan,
                    "avg_edge": float(losses["edge"].mean()) if "edge" in losses.columns and len(losses) else np.nan,
                },
                {
                    "outcome": "win",
                    "n": len(wins),
                    "avg_total_goals": float(wins["total_goals"].mean()) if len(wins) else np.nan,
                    "pct_ge4": float((wins["total_goals"] >= 4).mean()) if len(wins) else np.nan,
                    "pct_le1": float((wins["total_goals"] <= 1).mean()) if len(wins) else np.nan,
                    "avg_odds": float(wins["close_odds"].mean()) if len(wins) else np.nan,
                    "avg_edge": float(wins["edge"].mean()) if "edge" in wins.columns and len(wins) else np.nan,
                },
            ]
        )

    return out


def variation_grid(preds, matches, label: str) -> pd.DataFrame:
    rows = []
    for side, bands in [("under", UNDER_BANDS), ("over", OVER_BANDS)]:
        for lo, hi in bands:
            for edge in EDGES:
                bets = _score_ou(preds, matches, side=side, min_odds=lo, max_odds=hi, edge=edge)
                roi = _roi(bets)
                hit = _hit(bets)
                spos, sn = _season_pos(bets)
                rows.append(
                    {
                        "model": label,
                        "side": side,
                        "min_odds": lo,
                        "max_odds": hi,
                        "edge": edge,
                        "n": len(bets),
                        "roi": roi,
                        "hit": hit,
                        "seasons_pos": spos,
                        "seasons_n": sn,
                        "avg_odds": float(bets["close_odds"].mean()) if len(bets) else np.nan,
                        "avg_edge": float(bets["edge"].mean()) if len(bets) and "edge" in bets.columns else np.nan,
                        "units": float(bets["profit"].sum()) if len(bets) else 0.0,
                        "tag": f"{label}_{side}_{lo}-{hi}_e{int(edge*100)}",
                    }
                )
    return pd.DataFrame(rows)


def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100*float(x):+.1f}%"


def _fmt_pct_plain(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100*float(x):.1f}%"


def write_report(
    *,
    primary_bets: pd.DataFrame,
    seasonal: pd.DataFrame,
    cuts: dict[str, pd.DataFrame],
    grid: pd.DataFrame,
    primary_meta: dict,
) -> str:
    lines: list[str] = []
    lines.append("# Iteration 20 — Bundesliga Unders Deep Dive")
    lines.append("")
    lines.append("Date: 2026-08-11")
    lines.append("")
    lines.append("EPL packs **untouched**. This report covers Bundesliga only.")
    lines.append("")
    lines.append("## Primary system (paper)")
    lines.append("")
    lines.append(
        f"- **Pack:** `Bundesliga_unders_short_exp`  \n"
        f"- **Model:** `{primary_meta['label']}` (`{primary_meta['experiment_id']}`)  \n"
        f"- **Rules:** Unders **{primary_meta['min_odds']:.2f}–{primary_meta['max_odds']:.2f}** @ edge **≥{100*primary_meta['edge']:.0f}%**"
    )
    lines.append("")
    n = len(primary_bets)
    roi = _roi(primary_bets)
    hit = _hit(primary_bets)
    spos, sn = _season_pos(primary_bets)
    units = float(primary_bets["profit"].sum()) if n else 0.0
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| n | {n} |")
    lines.append(f"| Hit rate | {_fmt_pct_plain(hit)} |")
    lines.append(f"| ROI | {_fmt_pct(roi)} |")
    lines.append(f"| Units | {units:+.2f}u |")
    lines.append(
        f"| Avg odds | {float(primary_bets['close_odds'].mean()):.3f} |" if n else "| Avg odds | — |"
    )
    lines.append(
        f"| Avg edge | {_fmt_pct_plain(float(primary_bets['edge'].mean()))} |"
        if n and "edge" in primary_bets.columns
        else "| Avg edge | — |"
    )
    lines.append(f"| Seasons + | {spos}/{sn} |")
    lines.append("")

    lines.append("## Season-by-season (primary)")
    lines.append("")
    lines.append("| Season | n | Hit | ROI | Units | Cum ROI | Cum u | Avg odds | Avg edge |")
    lines.append("|-------:|--:|----:|----:|------:|--------:|------:|---------:|---------:|")
    for _, r in seasonal.iterrows():
        lines.append(
            f"| {int(r['season'])} | {int(r['n'])} | {_fmt_pct_plain(r['hit'])} | {_fmt_pct(r['roi'])} | "
            f"{r['units']:+.2f} | {_fmt_pct(r['cum_roi'])} | {r['cum_units']:+.2f} | "
            f"{r['avg_odds']:.3f} | {_fmt_pct_plain(r['avg_edge'])} |"
        )
    lines.append("")

    # Stability notes
    red = seasonal[seasonal["roi"] < 0]
    lines.append("### Stability notes")
    lines.append("")
    if len(red):
        lines.append(
            f"- Losing seasons: {', '.join(str(int(s)) for s in red['season'])} "
            f"({len(red)}/{len(seasonal)})."
        )
        worst = red.loc[red["roi"].idxmin()]
        lines.append(
            f"- Worst season: **{int(worst['season'])}** ROI {_fmt_pct(worst['roi'])} "
            f"(n={int(worst['n'])})."
        )
    else:
        lines.append("- No losing seasons.")
    if len(seasonal):
        lines.append(
            f"- Final cum ROI {_fmt_pct(seasonal.iloc[-1]['cum_roi'])} "
            f"({seasonal.iloc[-1]['cum_units']:+.2f}u)."
        )
        # Drawdown of cum units
        cum = seasonal["cum_units"].values
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        lines.append(f"- Max cum-units drawdown: **{dd.min():+.2f}u**.")
    lines.append("")

    def _table(title: str, df: pd.DataFrame) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if df is None or len(df) == 0:
            lines.append("_No data._")
            lines.append("")
            return
        cols = list(df.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---:"] * len(cols)) + "|")
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if c in ("roi", "hit", "avg_edge") or "pct" in c:
                    cells.append(_fmt_pct(v) if c == "roi" else _fmt_pct_plain(v))
                elif isinstance(v, float):
                    cells.append(f"{v:.3f}" if abs(v) < 20 else f"{v:.2f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    for key, title in [
        ("by_odds_bucket", "Odds buckets (inside band)"),
        ("by_edge_bucket", "Edge size buckets"),
        ("by_favorite_side", "Home-favorite vs Away-favorite (λ)"),
        ("by_season_half", "Early vs late season"),
        ("by_pred_total_env", "Predicted totals environment"),
        ("failure_vs_win", "Failure patterns (wins vs losses)"),
    ]:
        if key in cuts:
            _table(title, cuts[key])

    # Variation ranking
    lines.append("## Nearby variations (ranked)")
    lines.append("")
    lines.append("Criteria for **decision-grade**: n≥80, ROI≥+3%, seasons+ ≥ majority.")
    lines.append("")
    g = grid.copy()
    g = g[g["roi"].notna() & (g["n"] >= 40)].sort_values("roi", ascending=False)
    decision = g[
        (g["n"] >= 80)
        & (g["roi"] >= 0.03)
        & (g["seasons_pos"] >= (g["seasons_n"] // 2 + 1))
    ].head(25)
    lines.append("### Decision-grade candidates")
    lines.append("")
    if len(decision) == 0:
        lines.append("_None beyond primary band criteria._")
        lines.append("")
    else:
        lines.append("| Rank | Tag | n | ROI | Hit | Seasons+ | Units |")
        lines.append("|-----:|-----|--:|----:|----:|---------:|------:|")
        for i, (_, r) in enumerate(decision.iterrows(), 1):
            lines.append(
                f"| {i} | `{r['tag']}` | {int(r['n'])} | {_fmt_pct(r['roi'])} | "
                f"{_fmt_pct_plain(r['hit'])} | {int(r['seasons_pos'])}/{int(r['seasons_n'])} | "
                f"{r['units']:+.1f} |"
            )
        lines.append("")

    # Primary neighborhood focus
    neigh = g[
        (g["side"] == "under")
        & (g["min_odds"].between(1.60, 1.90))
        & (g["max_odds"].between(2.40, 3.00))
    ].sort_values(["model", "edge", "min_odds"])
    lines.append("### Unders neighborhood (1.60–3.00 bands)")
    lines.append("")
    lines.append("| Model | Band | Edge | n | ROI | Hit | Seasons+ |")
    lines.append("|-------|------|-----:|--:|----:|----:|---------:|")
    for _, r in neigh.iterrows():
        mark = " **← primary**" if (
            r["model"] == PRIMARY["label"]
            and float(r["min_odds"]) == PRIMARY["min_odds"]
            and float(r["max_odds"]) == PRIMARY["max_odds"]
            and float(r["edge"]) == PRIMARY["edge"]
        ) else ""
        lines.append(
            f"| {r['model']} | {r['min_odds']:.2f}–{r['max_odds']:.2f} | "
            f"{100*r['edge']:.0f}% | {int(r['n'])} | {_fmt_pct(r['roi'])} | "
            f"{_fmt_pct_plain(r['hit'])} | {int(r['seasons_pos'])}/{int(r['seasons_n'])} |{mark}"
        )
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "Keep **`Bundesliga_unders_short_exp`** as the paper pack: Unders **1.70–2.50 @ e≥10%** "
        "on **thresh05**. Prefer e10% over e12% for sample size and seasonal coverage. "
        "Do **not** merge with EPL packs. Paper-trade separately on D1 fixtures + sharp odds."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Directory: `{OUT.as_posix()}`")
    lines.append("- `primary_bets.parquet`, `seasonal_primary.csv`, cut CSVs, `variation_grid.csv`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    setup_logging("WARNING")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    matches = load_aligned(data_dir / "interim" / ALIGNED)

    # Primary
    p_preds = pd.read_parquet(ROOT / "experiments" / PRIMARY["experiment_id"] / "predictions.parquet")
    primary_bets = _score_ou(
        p_preds,
        matches,
        side=PRIMARY["side"],
        min_odds=PRIMARY["min_odds"],
        max_odds=PRIMARY["max_odds"],
        edge=PRIMARY["edge"],
    )
    # Attach lambdas from preds
    pred_idx = p_preds.set_index("match_id")
    for col in ("lambda_home", "lambda_away", "p_over25", "p_under25"):
        if col in pred_idx.columns:
            primary_bets[col] = primary_bets["match_id"].map(pred_idx[col])
    primary_bets = _enrich(primary_bets, matches)
    seasonal = seasonal_table(primary_bets)
    cuts = deep_cuts(primary_bets)

    primary_bets.to_parquet(OUT / "primary_bets.parquet", index=False)
    seasonal.to_csv(OUT / "seasonal_primary.csv", index=False)
    for name, df in cuts.items():
        df.to_csv(OUT / f"{name}.csv", index=False)

    # Variation grids
    grids = []
    for m in MODELS:
        preds = pd.read_parquet(ROOT / "experiments" / m["experiment_id"] / "predictions.parquet")
        print(f"Grid {m['label']} n_pred={len(preds)}")
        g = variation_grid(preds, matches, m["label"])
        grids.append(g)
    grid = pd.concat(grids, ignore_index=True)
    grid.to_csv(OUT / "variation_grid.csv", index=False)
    grid.sort_values("roi", ascending=False).head(40).to_csv(OUT / "variation_top40.csv", index=False)

    report = write_report(
        primary_bets=primary_bets,
        seasonal=seasonal,
        cuts=cuts,
        grid=grid,
        primary_meta=PRIMARY,
    )
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report[:2500])
    print(f"\nWrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
