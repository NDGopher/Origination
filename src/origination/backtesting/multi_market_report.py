"""
Comprehensive multi-market walk-forward performance tables.

Produces per-market / per-threshold / per-season ROI, claimed edge, hit rate,
and simple significance (t-stat + 95% CI on per-bet ROI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from origination.backtesting.walk_forward import evaluate_predictions, _summarize
THRESHOLDS = [0.02, 0.03, 0.04, 0.05]


def _roi_significance(profits: np.ndarray, stakes: np.ndarray) -> dict[str, Any]:
    """t-stat / CI on per-bet ROI (profit/stake), flat-stake friendly."""
    if len(profits) == 0 or np.nansum(stakes) <= 0:
        return {
            "n": 0,
            "roi": None,
            "roi_mean_per_bet": None,
            "roi_std_per_bet": None,
            "t_stat": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    r = profits / np.where(stakes > 0, stakes, np.nan)
    r = r[np.isfinite(r)]
    n = len(r)
    mean = float(np.mean(r)) if n else None
    std = float(np.std(r, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else np.nan
    t = mean / se if n > 1 and se > 0 else None
    ci_lo = mean - 1.96 * se if n > 1 and np.isfinite(se) else None
    ci_hi = mean + 1.96 * se if n > 1 and np.isfinite(se) else None
    stake_sum = float(np.nansum(stakes))
    profit_sum = float(np.nansum(profits))
    return {
        "n": n,
        "roi": profit_sum / stake_sum if stake_sum > 0 else None,
        "roi_mean_per_bet": mean,
        "roi_std_per_bet": std if n > 1 else None,
        "t_stat": float(t) if t is not None else None,
        "ci95_low": float(ci_lo) if ci_lo is not None else None,
        "ci95_high": float(ci_hi) if ci_hi is not None else None,
    }


def _agg_bets(bets: pd.DataFrame) -> dict[str, Any]:
    if len(bets) == 0:
        return {
            "n_bets": 0,
            "roi": None,
            "avg_claimed_edge": None,
            "avg_clv_prob": None,
            "hit_rate": None,
            "units_profit": 0.0,
            "stake_total": 0.0,
            "t_stat": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    settled = bets.dropna(subset=["profit"])
    sig = _roi_significance(
        settled["profit"].astype(float).values,
        settled["stake"].astype(float).values,
    )
    won = settled["won"].astype(float)
    # Hit rate: count full/half wins as won>0.5 contribution
    hit = float((won >= 0.75).mean()) if len(won) else None
    return {
        "n_bets": int(len(bets)),
        "roi": sig["roi"],
        "avg_claimed_edge": float(bets["edge"].mean()),
        "avg_clv_prob": float(bets["clv_prob"].mean()),
        "hit_rate": hit,
        "units_profit": float(settled["profit"].sum()) if len(settled) else 0.0,
        "stake_total": float(settled["stake"].sum()) if len(settled) else 0.0,
        "t_stat": sig["t_stat"],
        "ci95_low": sig["ci95_low"],
        "ci95_high": sig["ci95_high"],
    }


def evaluate_market_bets(
    preds: pd.DataFrame,
    matches: pd.DataFrame,
    bt_cfg: dict[str, Any],
    market: str,
    edge_threshold: float,
) -> pd.DataFrame:
    cfg = dict(bt_cfg)
    cfg["markets"] = [market]
    return evaluate_predictions(preds, matches, cfg, edge_threshold=edge_threshold)


def ht_calibration_table(preds: pd.DataFrame, matches: pd.DataFrame, *, label: str = "") -> pd.DataFrame:
    """
    First-half model calibration vs outcomes.

    Football-data EPL extract has HT goals / HTR but no HT closing odds, so
    ROI / CLV are not available — report log-loss / Brier / favorite hit-rate.
    """
    if "fold_season" in preds.columns:
        season_key = "fold_season"
    elif "season" in preds.columns:
        season_key = "season"
    else:
        preds = preds.merge(matches[["match_id", "season"]], on="match_id", how="left")
        season_key = "season"

    rows: list[dict[str, Any]] = []
    for season, g in preds.groupby(season_key):
        row = ht_calibration_overall(g, matches)
        row["season"] = season
        row["label"] = label
        rows.append(row)
    overall = ht_calibration_overall(preds, matches)
    overall["label"] = label
    rows.append(overall)
    return pd.DataFrame(rows)


def ht_calibration_overall(preds: pd.DataFrame, matches: pd.DataFrame) -> dict[str, Any]:
    m = matches.set_index("match_id")
    ll_1x2: list[float] = []
    brier: list[float] = []
    fav_hit: list[float] = []
    ll_ou15: list[float] = []
    for _, row in preds.iterrows():
        if row["match_id"] not in m.index:
            continue
        match = m.loc[row["match_id"]]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        if pd.isna(match.get("ht_home_goals")) or pd.isna(match.get("ht_away_goals")):
            continue
        hg = float(match["ht_home_goals"])
        ag = float(match["ht_away_goals"])
        htr = "H" if hg > ag else ("A" if ag > hg else "D")
        probs = np.array(
            [row.get("p_ht_home", np.nan), row.get("p_ht_draw", np.nan), row.get("p_ht_away", np.nan)],
            dtype=float,
        )
        if not np.all(np.isfinite(probs)):
            continue
        probs = np.clip(probs / probs.sum(), 1e-6, 1.0)
        outcome = {"H": 0, "D": 1, "A": 2}[htr]
        ll_1x2.append(float(-np.log(probs[outcome])))
        y = np.zeros(3)
        y[outcome] = 1.0
        brier.append(float(np.sum((probs - y) ** 2)))
        fav_hit.append(1.0 if int(np.argmax(probs)) == outcome else 0.0)
        tot = hg + ag
        po = float(row.get("p_ht_over15", np.nan))
        if np.isfinite(po):
            y_over = 1.0 if tot > 1.5 else 0.0
            p = np.clip(po, 1e-6, 1 - 1e-6)
            ll_ou15.append(float(-(y_over * np.log(p) + (1 - y_over) * np.log(1 - p))))
    return {
        "season": "ALL",
        "market": "ht_1x2_calib",
        "n": len(ll_1x2),
        "log_loss": float(np.mean(ll_1x2)) if ll_1x2 else None,
        "brier": float(np.mean(brier)) if brier else None,
        "favorite_hit_rate": float(np.mean(fav_hit)) if fav_hit else None,
        "ht_ou15_log_loss": float(np.mean(ll_ou15)) if ll_ou15 else None,
        "note": "no HT closing odds in football-data — calibration only",
    }


def build_multi_market_report(
    preds: pd.DataFrame,
    matches: pd.DataFrame,
    bt_cfg: dict[str, Any],
    *,
    label: str,
    thresholds: list[float] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build summary / by_season / by_edge_bucket / high_edge tables for
    1x2, ou25, ah (+ HT calibration).
    """
    thresholds = thresholds or list(THRESHOLDS)
    markets = ["1x2", "ou25", "ah"]
    summary_rows: list[dict[str, Any]] = []
    season_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []

    # 1X2 market log-loss (primary)
    ll_summary = _summarize(pd.DataFrame(), preds, matches)

    for market in markets:
        for thr in thresholds:
            bets = evaluate_market_bets(preds, matches, bt_cfg, market, thr)
            agg = _agg_bets(bets)
            summary_rows.append(
                {
                    "label": label,
                    "market": market,
                    "edge_threshold": thr,
                    "log_loss_1x2": ll_summary.get("log_loss_1x2") if market == "1x2" else None,
                    "log_loss_market_1x2": ll_summary.get("log_loss_market_1x2") if market == "1x2" else None,
                    **agg,
                }
            )
            if len(bets) == 0:
                continue
            for season, g in bets.groupby("season"):
                sagg = _agg_bets(g)
                season_rows.append(
                    {
                        "label": label,
                        "market": market,
                        "edge_threshold": thr,
                        "season": season,
                        **sagg,
                    }
                )
            for bucket, g in bets.groupby("edge_bucket"):
                bagg = _agg_bets(g)
                bucket_rows.append(
                    {
                        "label": label,
                        "market": market,
                        "edge_threshold": thr,
                        "edge_bucket": bucket,
                        **bagg,
                    }
                )
            if market == "ah" and "ah_line" in bets.columns:
                for line, g in bets.groupby("ah_line"):
                    lagg = _agg_bets(g)
                    line_rows.append(
                        {
                            "label": label,
                            "market": "ah",
                            "edge_threshold": thr,
                            "ah_line": line,
                            **lagg,
                        }
                    )

    # HT calibration (no odds)
    ht_df = ht_calibration_table(preds, matches, label=label)

    # Positive expectancy flags
    pos = [
        r
        for r in summary_rows
        if r.get("roi") is not None and r["roi"] > 0 and r.get("n_bets", 0) >= 30
    ]
    pos_season = [
        r
        for r in season_rows
        if r.get("roi") is not None and r["roi"] > 0 and r.get("n_bets", 0) >= 20
    ]

    return {
        "summary": pd.DataFrame(summary_rows),
        "by_season": pd.DataFrame(season_rows),
        "by_edge_bucket": pd.DataFrame(bucket_rows),
        "ah_by_line": pd.DataFrame(line_rows),
        "ht_calibration": ht_df,
        "positive_roi_cells": pd.DataFrame(pos),
        "positive_roi_seasons": pd.DataFrame(pos_season),
        "ll_summary": pd.DataFrame(
            [{"label": label, **{k: v for k, v in ll_summary.items() if k != "by_threshold"}}]
        ),
    }


def save_multi_market_report(tables: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame) and len(df):
            df.to_csv(out_dir / f"{name}.csv", index=False)
    logger.info("Wrote multi-market report CSVs to {}", out_dir)
    return out_dir
