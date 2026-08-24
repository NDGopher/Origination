"""
Deep performance autopsy: segment ROI / hit-rate / t-stats for betting diagnostics.

Segments cover favorites/dogs, odds buckets, home/away, over/under, edge size,
and optional residual-correction magnitude (when base predictions are supplied).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


def _t_ci(profits: np.ndarray) -> tuple[float, float, float]:
    n = len(profits)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(profits))
    se = float(np.std(profits, ddof=1) / np.sqrt(n))
    if se <= 0:
        return mean, float("nan"), float("nan")
    t = mean / se
    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
    return t, float(ci[0]), float(ci[1])


def summarize_bets(df: pd.DataFrame, *, min_n: int = 1) -> dict[str, Any]:
    if df is None or len(df) < min_n:
        return {
            "n_bets": 0 if df is None else int(len(df)),
            "hit_rate": np.nan,
            "roi": np.nan,
            "avg_odds": np.nan,
            "avg_edge": np.nan,
            "units_profit": np.nan,
            "stake_total": np.nan,
            "t_stat": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    stake = float(df["stake"].sum())
    profit = float(df["profit"].sum())
    t, lo, hi = _t_ci(df["profit"].astype(float).values)
    return {
        "n_bets": int(len(df)),
        "hit_rate": float(df["won"].mean()),
        "roi": profit / stake if stake else np.nan,
        "avg_odds": float(df["close_odds"].mean()),
        "avg_edge": float(df["edge"].mean()),
        "units_profit": profit,
        "stake_total": stake,
        "t_stat": t,
        "profit_ci95_low": lo,
        "profit_ci95_high": hi,
    }


def _odds_bucket(odds: float) -> str:
    if not np.isfinite(odds):
        return "na"
    if odds < 1.50:
        return "big_fav_<1.50"
    if odds < 2.00:
        return "mild_fav_1.50-2.00"
    if odds < 2.70:
        return "pickem_2.00-2.70"
    if odds < 4.00:
        return "mild_dog_2.70-4.00"
    return "big_dog_4.00+"


def _fav_dog_label(row: pd.Series) -> str:
    """1X2: fav = shortest close price among H/D/A on that match side vs market."""
    odds = float(row["close_odds"])
    if odds < 2.0:
        return "favorite"
    if odds < 2.7:
        return "pickem"
    return "underdog"


def _home_away_side(row: pd.Series) -> str:
    side = str(row["side"])
    if row["market"] == "1x2":
        if side == "H":
            return "home"
        if side == "A":
            return "away"
        return "draw"
    if row["market"] == "ou25":
        return side  # over / under
    if row["market"] == "ah":
        return "ah_home" if side in ("H", "home", "ahh") else "ah_away"
    return side


def enrich_bets_for_autopsy(
    bets: pd.DataFrame,
    matches: pd.DataFrame | None = None,
    *,
    base_preds: pd.DataFrame | None = None,
    resid_preds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add diagnostic columns used by segment tables."""
    out = bets.copy()
    out["odds_bucket"] = out["close_odds"].map(_odds_bucket)
    out["fav_dog"] = out.apply(_fav_dog_label, axis=1)
    out["side_role"] = out.apply(_home_away_side, axis=1)

    if matches is not None and len(matches):
        m = matches.set_index("match_id")
        # Market favorite for 1X2: lowest close odds side
        fav_side = []
        for mid in out["match_id"]:
            if mid not in m.index:
                fav_side.append(np.nan)
                continue
            row = m.loc[mid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            odds = {
                "H": float(row.get("close_h", np.nan)),
                "D": float(row.get("close_d", np.nan)),
                "A": float(row.get("close_a", np.nan)),
            }
            finite = {k: v for k, v in odds.items() if np.isfinite(v)}
            fav_side.append(min(finite, key=finite.get) if finite else np.nan)
        out["market_fav_side"] = fav_side
        out["bet_on_market_fav"] = (
            (out["market"] == "1x2") & (out["side"] == out["market_fav_side"])
        )
        out["bet_on_market_dog"] = (
            (out["market"] == "1x2")
            & out["side"].isin(["H", "A", "D"])
            & (out["side"] != out["market_fav_side"])
        )

    # Residual correction magnitude
    out["resid_l1"] = np.nan
    if (
        base_preds is not None
        and resid_preds is not None
        and len(base_preds)
        and len(resid_preds)
    ):
        b = base_preds.set_index("match_id")
        r = resid_preds.set_index("match_id")
        common = b.index.intersection(r.index)
        deltas = {}
        for mid in common:
            pb = b.loc[mid]
            pr = r.loc[mid]
            if isinstance(pb, pd.DataFrame):
                pb = pb.iloc[0]
            if isinstance(pr, pd.DataFrame):
                pr = pr.iloc[0]
            d = (
                abs(float(pr["p_home"]) - float(pb["p_home"]))
                + abs(float(pr["p_draw"]) - float(pb["p_draw"]))
                + abs(float(pr["p_away"]) - float(pb["p_away"]))
            )
            if "p_over25" in pr.index and "p_over25" in pb.index:
                d += abs(float(pr["p_over25"]) - float(pb["p_over25"]))
            deltas[mid] = d
        out["resid_l1"] = out["match_id"].map(deltas)
        q = out["resid_l1"].dropna()
        if len(q) and float(q.nunique()) >= 3 and float(q.std()) > 1e-12:
            lo, hi = q.quantile([0.33, 0.67])
            if lo < hi:
                out["resid_size"] = pd.cut(
                    out["resid_l1"],
                    bins=[-np.inf, lo, hi, np.inf],
                    labels=["small", "medium", "large"],
                    duplicates="drop",
                ).astype(str)
            else:
                out["resid_size"] = "uniform"
        elif len(q) and float(q.max()) > 0:
            med = float(q.median())
            out["resid_size"] = np.where(out["resid_l1"] <= med, "small", "large")
        else:
            out["resid_size"] = "none"
    else:
        out["resid_size"] = "na"

    return out


def _segment_table(df: pd.DataFrame, col: str, *, market: str | None = None) -> pd.DataFrame:
    sub = df if market is None else df[df["market"] == market]
    rows = []
    for key, g in sub.groupby(col, dropna=False):
        s = summarize_bets(g)
        rows.append({"segment": col, "value": key, "market": market or "all", **s})
    return pd.DataFrame(rows)


def build_autopsy_tables(
    bets: pd.DataFrame,
    matches: pd.DataFrame | None = None,
    *,
    label: str = "model",
    base_preds: pd.DataFrame | None = None,
    resid_preds: pd.DataFrame | None = None,
    edge_threshold: float = 0.03,
) -> dict[str, pd.DataFrame]:
    """Build full segment diagnostic tables for one league/model."""
    b = bets[bets["edge_threshold"] == edge_threshold].copy() if "edge_threshold" in bets.columns else bets.copy()
    # If bets.parquet already filtered at one threshold, edge_threshold may be constant
    if "edge_threshold" not in b.columns:
        b["edge_threshold"] = edge_threshold
    else:
        # walk-forward bets are usually single-threshold
        pass

    enriched = enrich_bets_for_autopsy(
        b, matches, base_preds=base_preds, resid_preds=resid_preds
    )
    tables: dict[str, pd.DataFrame] = {}

    # Overall by market
    ov = []
    for mkt, g in enriched.groupby("market"):
        ov.append({"segment": "overall", "value": mkt, "market": mkt, **summarize_bets(g)})
    ov.append({"segment": "overall", "value": "all", "market": "all", **summarize_bets(enriched)})
    tables["overall"] = pd.DataFrame(ov)

    for col in ["odds_bucket", "fav_dog", "side_role", "edge_bucket", "resid_size"]:
        if col not in enriched.columns:
            continue
        parts = [_segment_table(enriched, col)]
        for mkt in enriched["market"].unique():
            parts.append(_segment_table(enriched, col, market=mkt))
        tables[col] = pd.concat(parts, ignore_index=True)

    # 1X2 bet on market fav vs dog
    if "bet_on_market_fav" in enriched.columns:
        rows = []
        for flag, name in [(True, "on_market_fav"), (False, "not_on_market_fav")]:
            g = enriched[(enriched["market"] == "1x2") & (enriched["bet_on_market_fav"] == flag)]
            rows.append({"segment": "1x2_fav_alignment", "value": name, "market": "1x2", **summarize_bets(g)})
        g_dog = enriched[(enriched["market"] == "1x2") & (enriched["bet_on_market_dog"])]
        rows.append({"segment": "1x2_fav_alignment", "value": "on_market_dog", "market": "1x2", **summarize_bets(g_dog)})
        tables["fav_alignment"] = pd.DataFrame(rows)

    # OU over vs under already in side_role; explicit
    if (enriched["market"] == "ou25").any():
        tables["ou_side"] = _segment_table(enriched[enriched["market"] == "ou25"], "side", market="ou25")

    # Rank drags / near-positive
    all_seg = pd.concat(
        [t for t in tables.values() if len(t)],
        ignore_index=True,
    )
    all_seg["label"] = label
    viable = all_seg[all_seg["n_bets"] >= 50].copy()
    tables["biggest_drags"] = viable.sort_values("roi").head(25)
    tables["closest_positive"] = viable.sort_values("roi", ascending=False).head(25)
    tables["all_segments"] = all_seg

    return tables


def save_autopsy(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    logger.info("Wrote autopsy tables to {}", out_dir)


def write_autopsy_summary(tables: dict[str, pd.DataFrame], path: Path, *, label: str) -> None:
    """Human-readable failure-mode summary."""
    lines = [f"# Performance autopsy — {label}", ""]
    ov = tables.get("overall")
    if ov is not None and len(ov):
        lines.append("## Overall")
        lines.append("")
        lines.append(ov.to_string(index=False))
        lines.append("")

    drags = tables.get("biggest_drags")
    if drags is not None and len(drags):
        lines.append("## Biggest ROI drags (n≥50)")
        lines.append("")
        cols = ["market", "segment", "value", "n_bets", "hit_rate", "roi", "avg_odds", "avg_edge", "t_stat"]
        lines.append(drags[cols].head(15).to_string(index=False))
        lines.append("")

    pos = tables.get("closest_positive")
    if pos is not None and len(pos):
        lines.append("## Closest to breakeven / positive (n≥50)")
        lines.append("")
        cols = ["market", "segment", "value", "n_bets", "hit_rate", "roi", "avg_odds", "avg_edge", "t_stat"]
        lines.append(pos[cols].head(15).to_string(index=False))
        lines.append("")

    # Key narratives
    lines.append("## Key failure modes (auto)")
    lines.append("")
    for key in ["odds_bucket", "fav_dog", "ou_side", "resid_size", "fav_alignment"]:
        t = tables.get(key)
        if t is None or not len(t):
            continue
        # prefer market-specific rows for 1x2/ou
        sub = t.copy()
        if "market" in sub.columns:
            pref = sub[sub["market"].isin(["1x2", "ou25"])]
            if len(pref):
                sub = pref
        worst = sub.loc[sub["n_bets"] >= 50].sort_values("roi").head(3) if (sub["n_bets"] >= 50).any() else sub.head(0)
        best = sub.loc[sub["n_bets"] >= 50].sort_values("roi", ascending=False).head(3) if (sub["n_bets"] >= 50).any() else sub.head(0)
        lines.append(f"### {key}")
        if len(worst):
            lines.append("Worst: " + "; ".join(
                f"{r['market']}/{r['value']} n={r['n_bets']} ROI={r['roi']:.1%} hit={r['hit_rate']:.1%}"
                for _, r in worst.iterrows()
            ))
        if len(best):
            lines.append("Best: " + "; ".join(
                f"{r['market']}/{r['value']} n={r['n_bets']} ROI={r['roi']:.1%} hit={r['hit_rate']:.1%}"
                for _, r in best.iterrows()
            ))
        lines.append("")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote autopsy summary {}", path)
