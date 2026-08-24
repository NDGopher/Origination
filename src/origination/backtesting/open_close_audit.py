"""
Open vs close odds audit — is the model ahead of the open but behind the close?

Attaches opening odds columns and compares model probabilities / betting
performance at open vs at close for 1X2, O/U 2.5, and AH.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from origination.backtesting.multi_market_report import _agg_bets, _roi_significance
from origination.backtesting.walk_forward import evaluate_predictions, _summarize
from origination.utils.odds import fair_probs, two_way_fair


def _first_valid(row: pd.Series, cols: list[str]) -> float:
    for c in cols:
        if c in row.index and pd.notna(row.get(c)):
            try:
                v = float(row[c])
            except (TypeError, ValueError):
                continue
            if v > 1.0:
                return v
    return float("nan")


def attach_opening_odds(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Derive opening-line columns from football-data raw odds.

    Opening = non-closing books (no ``C`` suffix). Closing remains on ``close_*``.
    """
    out = matches.copy()
    out["open_h"] = out.apply(
        lambda r: _first_valid(r, ["PSH", "B365H", "AvgH", "BbAvH", "MaxH"]), axis=1
    )
    out["open_d"] = out.apply(
        lambda r: _first_valid(r, ["PSD", "B365D", "AvgD", "BbAvD", "MaxD"]), axis=1
    )
    out["open_a"] = out.apply(
        lambda r: _first_valid(r, ["PSA", "B365A", "AvgA", "BbAvA", "MaxA"]), axis=1
    )
    out["open_over25"] = out.apply(
        lambda r: _first_valid(r, ["P>2.5", "B365>2.5", "Avg>2.5", "BbAv>2.5", "Max>2.5"]),
        axis=1,
    )
    out["open_under25"] = out.apply(
        lambda r: _first_valid(r, ["P<2.5", "B365<2.5", "Avg<2.5", "BbAv<2.5", "Max<2.5"]),
        axis=1,
    )
    # Prefer explicit closing AH if present; opening uses non-C books
    out["open_ahh"] = out.apply(
        lambda r: _first_valid(r, ["B365AHH", "PAHH", "AvgAHH", "BbAvAHH", "MaxAHH"]), axis=1
    )
    out["open_aha"] = out.apply(
        lambda r: _first_valid(r, ["B365AHA", "PAHA", "AvgAHA", "BbAvAHA", "MaxAHA"]), axis=1
    )
    # True closing AH (C-suffix) for fair open/close compare when available
    out["true_close_ahh"] = out.apply(
        lambda r: _first_valid(r, ["B365CAHH", "PCAHH", "AvgCAHH", "MaxCAHH"]), axis=1
    )
    out["true_close_aha"] = out.apply(
        lambda r: _first_valid(r, ["B365CAHA", "PCAHA", "AvgCAHA", "MaxCAHA"]), axis=1
    )
    out["true_close_ah_line"] = out.apply(
        lambda r: _first_valid(r, ["AHCh"]) if "AHCh" in out.columns else float("nan"),
        axis=1,
    )
    # If AHCh missing, keep ah_line for both
    if "ah_line" in out.columns:
        out["true_close_ah_line"] = out["true_close_ah_line"].fillna(out["ah_line"])
    return out


def _matches_as_odds_source(matches: pd.DataFrame, source: str) -> pd.DataFrame:
    """Copy matches with close_* remapped to open or true-close for reuse of evaluate_predictions."""
    m = matches.copy()
    if source == "open":
        m["close_h"] = m["open_h"]
        m["close_d"] = m["open_d"]
        m["close_a"] = m["open_a"]
        m["close_over25"] = m["open_over25"]
        m["close_under25"] = m["open_under25"]
        m["close_ahh"] = m["open_ahh"]
        m["close_aha"] = m["open_aha"]
    elif source == "true_close_ah":
        # Keep 1x2/ou close_*; swap AH to C-suffix closes when available
        m["close_ahh"] = m["true_close_ahh"].fillna(m["close_ahh"])
        m["close_aha"] = m["true_close_aha"].fillna(m["close_aha"])
        if "true_close_ah_line" in m.columns:
            m["ah_line"] = m["true_close_ah_line"].fillna(m.get("ah_line"))
    # source == "close" → leave as-is
    return m


def _log_loss_1x2(preds: pd.DataFrame, matches: pd.DataFrame, odds_h: str, odds_d: str, odds_a: str) -> dict[str, Any]:
    m = matches.set_index("match_id")
    ll_model, ll_mkt, n = [], [], 0
    for _, row in preds.iterrows():
        if row["match_id"] not in m.index:
            continue
        match = m.loc[row["match_id"]]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        probs = np.array([row["p_home"], row["p_draw"], row["p_away"]], dtype=float)
        if not np.all(np.isfinite(probs)):
            continue
        probs = np.clip(probs / probs.sum(), 1e-6, 1.0)
        outcome = {"H": 0, "D": 1, "A": 2}[str(match["ftr"])]
        ll_model.append(float(-np.log(probs[outcome])))
        odds = np.array([match.get(odds_h), match.get(odds_d), match.get(odds_a)], dtype=float)
        if np.all(np.isfinite(odds)) and np.all(odds > 1.0):
            fair = np.clip(fair_probs(odds, method="power"), 1e-6, 1.0)
            ll_mkt.append(float(-np.log(fair[outcome])))
            n += 1
    out = {
        "n": len(ll_model),
        "n_with_odds": n,
        "log_loss_model": float(np.mean(ll_model)) if ll_model else None,
        "log_loss_market": float(np.mean(ll_mkt)) if ll_mkt else None,
    }
    if ll_model and ll_mkt and len(ll_mkt) == len(ll_model):
        out["gap_model_minus_market"] = out["log_loss_model"] - out["log_loss_market"]
    elif ll_model and ll_mkt:
        # Restrict to overlapping
        out["gap_model_minus_market"] = float(np.mean(ll_model[: len(ll_mkt)])) - out["log_loss_market"]
    return out


def _log_loss_ou(preds: pd.DataFrame, matches: pd.DataFrame, odds_o: str, odds_u: str) -> dict[str, Any]:
    m = matches.set_index("match_id")
    ll_model, ll_mkt = [], []
    for _, row in preds.iterrows():
        if row["match_id"] not in m.index:
            continue
        match = m.loc[row["match_id"]]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        if pd.isna(match.get("total_goals")) or not np.isfinite(float(row.get("p_over25", np.nan))):
            continue
        y = 1.0 if float(match["total_goals"]) > 2.5 else 0.0
        po = float(np.clip(row["p_over25"], 1e-6, 1 - 1e-6))
        ll_model.append(float(-(y * np.log(po) + (1 - y) * np.log(1 - po))))
        o, u = match.get(odds_o), match.get(odds_u)
        if pd.notna(o) and pd.notna(u) and float(o) > 1.0 and float(u) > 1.0:
            fo, _ = two_way_fair(float(o), float(u), method="power")
            pm = float(np.clip(fo, 1e-6, 1 - 1e-6))
            ll_mkt.append(float(-(y * np.log(pm) + (1 - y) * np.log(1 - pm))))
    return {
        "n_model": len(ll_model),
        "n_with_odds": len(ll_mkt),
        "log_loss_model": float(np.mean(ll_model)) if ll_model else None,
        "log_loss_market": float(np.mean(ll_mkt)) if ll_mkt else None,
        "gap_model_minus_market": (
            float(np.mean(ll_model) - np.mean(ll_mkt)) if ll_model and ll_mkt else None
        ),
    }


def _line_move_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """How much the market moves open → close (fair-prob space)."""
    rows = []
    # 1X2
    sub = matches.dropna(subset=["open_h", "open_d", "open_a", "close_h", "close_d", "close_a"])
    if len(sub):
        d_home, d_draw, d_away = [], [], []
        for _, r in sub.iterrows():
            fo = fair_probs([r["open_h"], r["open_d"], r["open_a"]], method="power")
            fc = fair_probs([r["close_h"], r["close_d"], r["close_a"]], method="power")
            d_home.append(fc[0] - fo[0])
            d_draw.append(fc[1] - fo[1])
            d_away.append(fc[2] - fo[2])
        rows.append(
            {
                "market": "1x2",
                "n": len(sub),
                "mean_abs_move_home": float(np.mean(np.abs(d_home))),
                "mean_abs_move_draw": float(np.mean(np.abs(d_draw))),
                "mean_abs_move_away": float(np.mean(np.abs(d_away))),
                "mean_abs_move": float(
                    np.mean(np.abs(d_home) + np.abs(d_draw) + np.abs(d_away)) / 3.0
                ),
            }
        )
    # OU
    sub = matches.dropna(subset=["open_over25", "open_under25", "close_over25", "close_under25"])
    if len(sub):
        moves = []
        for _, r in sub.iterrows():
            fo, _ = two_way_fair(float(r["open_over25"]), float(r["open_under25"]), method="power")
            fc, _ = two_way_fair(float(r["close_over25"]), float(r["close_under25"]), method="power")
            moves.append(abs(fc - fo))
        rows.append(
            {
                "market": "ou25",
                "n": len(sub),
                "mean_abs_move_home": float(np.mean(moves)),
                "mean_abs_move_draw": None,
                "mean_abs_move_away": None,
                "mean_abs_move": float(np.mean(moves)),
            }
        )
    return pd.DataFrame(rows)


def run_open_close_audit(
    preds: pd.DataFrame,
    matches: pd.DataFrame,
    bt_cfg: dict[str, Any],
    *,
    label: str = "best",
    thresholds: list[float] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Full open vs close audit tables.
    """
    thresholds = thresholds or [0.02, 0.03, 0.04, 0.05]
    m = attach_opening_odds(matches)
    # Restrict preds to matches that have opening 1X2 (always) for fair compare
    coverage = {
        "open_1x2": float(m[["open_h", "open_d", "open_a"]].notna().all(axis=1).mean()),
        "open_ou25": float(m[["open_over25", "open_under25"]].notna().all(axis=1).mean()),
        "open_ah": float(m[["open_ahh", "open_aha"]].notna().all(axis=1).mean()),
        "true_close_ah": float(m[["true_close_ahh", "true_close_aha"]].notna().all(axis=1).mean()),
    }
    logger.info("Open/close coverage: {}", coverage)

    ll_rows = []
    ll_open = _log_loss_1x2(preds, m, "open_h", "open_d", "open_a")
    ll_close = _log_loss_1x2(preds, m, "close_h", "close_d", "close_a")
    ll_rows.append({"label": label, "market": "1x2", "line": "open", **ll_open})
    ll_rows.append({"label": label, "market": "1x2", "line": "close", **ll_close})
    ll_rows.append(
        {
            "label": label,
            "market": "ou25",
            "line": "open",
            **_log_loss_ou(preds, m, "open_over25", "open_under25"),
        }
    )
    ll_rows.append(
        {
            "label": label,
            "market": "ou25",
            "line": "close",
            **_log_loss_ou(preds, m, "close_over25", "close_under25"),
        }
    )

    # Betting at open vs close
    bet_rows = []
    for source, tag in [("open", "open"), ("close", "close")]:
        m_src = _matches_as_odds_source(m, source)
        for market in ["1x2", "ou25", "ah"]:
            for thr in thresholds:
                bets = evaluate_predictions(
                    preds, m_src, {**bt_cfg, "markets": [market]}, edge_threshold=thr
                )
                agg = _agg_bets(bets)
                bet_rows.append(
                    {
                        "label": label,
                        "odds_source": tag,
                        "market": market,
                        "edge_threshold": thr,
                        **agg,
                    }
                )

    # Pairwise interpretation: same bets selected at open edge, settled at open odds vs close odds
    # (CLV of open-selected edges)
    clv_rows = []
    m_open = _matches_as_odds_source(m, "open")
    for market in ["1x2", "ou25", "ah"]:
        for thr in thresholds:
            open_bets = evaluate_predictions(
                preds, m_open, {**bt_cfg, "markets": [market]}, edge_threshold=thr
            )
            if len(open_bets) == 0:
                continue
            # Recompute profit at close for same match/side
            close_lookup = m.set_index("match_id")
            profits_open = []
            profits_close = []
            edges_open = []
            edges_close = []
            for _, b in open_bets.iterrows():
                mid = b["match_id"]
                if mid not in close_lookup.index:
                    continue
                match = close_lookup.loc[mid]
                if isinstance(match, pd.DataFrame):
                    match = match.iloc[0]
                stake = float(b["stake"])
                profits_open.append(float(b["profit"]))
                edges_open.append(float(b["edge"]))
                # Close fair edge for same side
                if market == "1x2":
                    odds_map = {"H": "close_h", "D": "close_d", "A": "close_a"}
                    c_odds = float(match[odds_map[b["side"]]])
                    fair = fair_probs(
                        [match["close_h"], match["close_d"], match["close_a"]], method="power"
                    )
                    fp = {"H": fair[0], "D": fair[1], "A": fair[2]}[b["side"]]
                    won = float(b["won"])
                    profit_c = stake * ((c_odds - 1.0) if won else -1.0)
                    edges_close.append(float(b["model_prob"]) - float(fp))
                    profits_close.append(profit_c)
                elif market == "ou25":
                    side = b["side"]
                    c_odds = float(match["close_over25"] if side == "over" else match["close_under25"])
                    fo, fu = two_way_fair(
                        float(match["close_over25"]), float(match["close_under25"]), method="power"
                    )
                    fp = fo if side == "over" else fu
                    won = float(b["won"])
                    profit_c = stake * ((c_odds - 1.0) if won else -1.0)
                    edges_close.append(float(b["model_prob"]) - float(fp))
                    profits_close.append(profit_c)
                else:
                    # AH: compare open selection edge vs close fair if both exist
                    continue
            if not profits_open:
                continue
            so = _roi_significance(np.array(profits_open), np.ones(len(profits_open)) * open_bets["stake"].iloc[0])
            # use actual stakes
            stakes = open_bets["stake"].astype(float).values[: len(profits_open)]
            sig_o = _roi_significance(np.asarray(profits_open), stakes)
            sig_c = _roi_significance(np.asarray(profits_close), stakes)
            clv_rows.append(
                {
                    "label": label,
                    "market": market,
                    "edge_threshold": thr,
                    "n_bets_open_selected": len(profits_open),
                    "roi_at_open": sig_o["roi"],
                    "roi_same_bets_at_close": sig_c["roi"],
                    "avg_edge_vs_open": float(np.mean(edges_open)),
                    "avg_edge_vs_close": float(np.mean(edges_close)) if edges_close else None,
                    "edge_decay_open_to_close": (
                        float(np.mean(edges_open) - np.mean(edges_close)) if edges_close else None
                    ),
                    "t_stat_open": sig_o["t_stat"],
                    "t_stat_close_settlement": sig_c["t_stat"],
                }
            )

    interpretation = _interpret(ll_rows, bet_rows, clv_rows, coverage)

    return {
        "coverage": pd.DataFrame([{"label": label, **coverage}]),
        "log_loss": pd.DataFrame(ll_rows),
        "bets_by_source": pd.DataFrame(bet_rows),
        "open_selected_clv": pd.DataFrame(clv_rows),
        "line_moves": _line_move_stats(m),
        "interpretation": pd.DataFrame([interpretation]),
    }


def _interpret(ll_rows, bet_rows, clv_rows, coverage) -> dict[str, Any]:
    ll = pd.DataFrame(ll_rows)
    def gap(market, line):
        r = ll[(ll["market"] == market) & (ll["line"] == line)]
        if len(r) == 0:
            return None
        return r.iloc[0].get("gap_model_minus_market")

    g1o, g1c = gap("1x2", "open"), gap("1x2", "close")
    gouo, gouc = gap("ou25", "open"), gap("ou25", "close")

    ahead_of_open = g1o is not None and g1o < 0  # model better than open market
    # gap is model - market; negative means model better (lower LL)
    # Wait: log_loss gap_model_minus_market = model_ll - market_ll. Positive means model worse.
    model_worse_than_open = g1o is not None and g1o > 0
    model_worse_than_close = g1c is not None and g1c > 0
    gap_widens = (
        g1o is not None and g1c is not None and g1c > g1o
    )  # market pulls away further by close

    if model_worse_than_open and model_worse_than_close and gap_widens:
        verdict = (
            "FUNDAMENTAL + LATE INFO: model already behind the open; close widens the gap further. "
            "Primary need is stronger probabilities, not only late signals."
        )
        weakness = "mostly_fundamental"
    elif model_worse_than_open and model_worse_than_close and not gap_widens:
        verdict = (
            "FUNDAMENTAL: model behind both open and close; late move does not explain the deficit."
        )
        weakness = "mostly_fundamental"
    elif (not model_worse_than_open) and model_worse_than_close:
        verdict = (
            "LATE MARKET INFO: model competitive with (or ahead of) the open but behind the close. "
            "Closing line incorporates information the current feature set lacks."
        )
        weakness = "mostly_late_market"
    else:
        verdict = "MIXED / CHECK NUMBERS — see log_loss and open_selected_clv tables."
        weakness = "mixed"

    return {
        "gap_1x2_vs_open": g1o,
        "gap_1x2_vs_close": g1c,
        "gap_ou_vs_open": gouo,
        "gap_ou_vs_close": gouc,
        "gap_widens_open_to_close_1x2": gap_widens,
        "weakness_class": weakness,
        "verdict": verdict,
        "open_1x2_coverage": coverage.get("open_1x2"),
        "open_ou_coverage": coverage.get("open_ou25"),
    }


def save_open_close_audit(tables: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(out_dir / f"{name}.csv", index=False)
    logger.info("Wrote open/close audit to {}", out_dir)
    return out_dir
