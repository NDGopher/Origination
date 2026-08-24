#!/usr/bin/env python
"""
Iter18 — Deep post-mortem of the two separate EPL systems on vol06 + thresh.

Books (DO NOT MERGE):
  1) EPL_aggressive — Unders 2.00–4.00 @ edge ≥ 8% (+ 1X2/AH pack rules)
  2) EPL_overs_short_exp — Overs 1.60–2.50 @ edge ≥ 10%

Outputs seasonal tables, conflict analysis, odds/edge diagnostics.
"""

from __future__ import annotations

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

# Best model: iter17 EPL vol06 (preferred over iter16)
PRED_PATH = ROOT / "experiments" / "20260810T201358Z_iter17_EPL_vol06" / "predictions.parquet"
OUT_DIR = ROOT / "experiments" / "iter18_epl_two_books"

EPL_AGGRESSIVE = {
    "edge_threshold": 0.03,
    "edge_threshold_by_market": {"ou25": 0.08, "ah": 0.05},
    "bet_filters": {
        "enabled": True,
        "rules": [
            {"markets": ["1x2"], "max_odds": 1.80},
            {"markets": ["ou25"], "min_odds": 2.00, "max_odds": 4.00, "allow_sides": ["under"]},
            {"markets": ["ah"], "max_odds": 1.90},
        ],
    },
}

EPL_OVERS_SHORT = {
    "edge_threshold": 0.03,
    "edge_threshold_by_market": {"ou25": 0.10, "ah": 0.05},
    "bet_filters": {
        "enabled": True,
        "rules": [
            {"markets": ["1x2"], "max_odds": 2.00},
            {"markets": ["ou25"], "min_odds": 1.60, "max_odds": 2.50, "allow_sides": ["over"]},
        ],
    },
}

BT_BASE = {
    "markets": ["1x2", "ou25", "ah"],
    "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
    "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
}


def _roi(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None
    st = float(df["stake"].sum())
    return float(df["profit"].sum()) / st if st else None


def _hit(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None
    # treat push (0.5) as non-win for hit rate; use won > 0.5
    w = df["won"].astype(float)
    return float((w > 0.5).mean())


def _season_table(bets: pd.DataFrame, *, label: str) -> pd.DataFrame:
    rows = []
    if len(bets) == 0:
        return pd.DataFrame()
    b = bets.sort_values("date").copy()
    cum_profit = 0.0
    cum_stake = 0.0
    for season, g in b.groupby("season", sort=True):
        st = float(g["stake"].sum())
        pr = float(g["profit"].sum())
        cum_profit += pr
        cum_stake += st
        rows.append(
            {
                "book": label,
                "season": int(season),
                "n": int(len(g)),
                "hit_rate": _hit(g),
                "roi": pr / st if st else None,
                "units_profit": pr,
                "avg_odds": float(g["close_odds"].mean()),
                "avg_edge": float(g["edge"].mean()),
                "cum_n": int(cum_stake),  # placeholder overwritten below
                "cum_roi": cum_profit / cum_stake if cum_stake else None,
                "cum_units": cum_profit,
            }
        )
    out = pd.DataFrame(rows)
    # proper cumulative n
    out["cum_n"] = out["n"].cumsum()
    return out


def _odds_bucket(odds: float) -> str:
    if odds < 1.60:
        return "<1.60"
    if odds < 2.00:
        return "1.60-2.00"
    if odds < 2.50:
        return "2.00-2.50"
    if odds < 3.00:
        return "2.50-3.00"
    if odds < 4.00:
        return "3.00-4.00"
    return ">=4.00"


def _edge_bucket(edge: float) -> str:
    if edge < 0.08:
        return "0.05-0.08"
    if edge < 0.10:
        return "0.08-0.10"
    if edge < 0.12:
        return "0.10-0.12"
    if edge < 0.15:
        return "0.12-0.15"
    return ">=0.15"


def _bucket_table(bets: pd.DataFrame, *, by: str, label: str) -> pd.DataFrame:
    if len(bets) == 0:
        return pd.DataFrame()
    b = bets.copy()
    if by == "odds":
        b["bucket"] = b["close_odds"].map(_odds_bucket)
        order = ["<1.60", "1.60-2.00", "2.00-2.50", "2.50-3.00", "3.00-4.00", ">=4.00"]
    else:
        b["bucket"] = b["edge"].map(_edge_bucket)
        order = ["0.05-0.08", "0.08-0.10", "0.10-0.12", "0.12-0.15", ">=0.15"]
    rows = []
    for bucket, g in b.groupby("bucket"):
        rows.append(
            {
                "book": label,
                "bucket_type": by,
                "bucket": bucket,
                "n": int(len(g)),
                "hit_rate": _hit(g),
                "roi": _roi(g),
                "avg_odds": float(g["close_odds"].mean()),
                "avg_edge": float(g["edge"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["bucket"] = pd.Categorical(out["bucket"], categories=order, ordered=True)
    return out.sort_values("bucket")


def _score(preds: pd.DataFrame, matches: pd.DataFrame, pack: dict) -> pd.DataFrame:
    bt = {**BT_BASE, **pack}
    return evaluate_predictions(preds, matches, bt)


def _conflict_analysis(
    under_ou: pd.DataFrame, over_ou: pd.DataFrame, matches: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Games flagged Under by pack1 and Over by pack2."""
    u = under_ou[under_ou["side"] == "under"][["match_id", "close_odds", "edge", "won", "profit", "stake"]].rename(
        columns={
            "close_odds": "under_odds",
            "edge": "under_edge",
            "won": "under_won",
            "profit": "under_profit",
            "stake": "under_stake",
        }
    )
    o = over_ou[over_ou["side"] == "over"][["match_id", "close_odds", "edge", "won", "profit", "stake"]].rename(
        columns={
            "close_odds": "over_odds",
            "edge": "over_edge",
            "won": "over_won",
            "profit": "over_profit",
            "stake": "over_stake",
        }
    )
    conf = u.merge(o, on="match_id", how="inner")
    m = matches.set_index("match_id")
    if len(conf):
        conf["date"] = conf["match_id"].map(m["date"])
        conf["season"] = conf["match_id"].map(m["season"])
        conf["home_team"] = conf["match_id"].map(m["home_team"])
        conf["away_team"] = conf["match_id"].map(m["away_team"])
        conf["total_goals"] = conf["match_id"].map(m["total_goals"])
        # which side won: under_won and over_won are mutually exclusive for OU2.5 (no push)
        conf["winner"] = np.where(
            conf["under_won"].astype(float) > 0.5,
            "under",
            np.where(conf["over_won"].astype(float) > 0.5, "over", "push"),
        )
        conf["net_if_both"] = conf["under_profit"] + conf["over_profit"]
        conf["net_under_only"] = conf["under_profit"]
        conf["net_over_only"] = conf["over_profit"]

    summary = {
        "n_conflicts": int(len(conf)),
        "under_wins": int((conf["winner"] == "under").sum()) if len(conf) else 0,
        "over_wins": int((conf["winner"] == "over").sum()) if len(conf) else 0,
        "pushes": int((conf["winner"] == "push").sum()) if len(conf) else 0,
        "net_if_both_books_bet": float(conf["net_if_both"].sum()) if len(conf) else 0.0,
        "net_under_side_only": float(conf["net_under_only"].sum()) if len(conf) else 0.0,
        "net_over_side_only": float(conf["net_over_only"].sum()) if len(conf) else 0.0,
        "avg_under_odds": float(conf["under_odds"].mean()) if len(conf) else None,
        "avg_over_odds": float(conf["over_odds"].mean()) if len(conf) else None,
        "avg_under_edge": float(conf["under_edge"].mean()) if len(conf) else None,
        "avg_over_edge": float(conf["over_edge"].mean()) if len(conf) else None,
    }
    return conf, summary


def _failure_patterns(bets: pd.DataFrame, matches: pd.DataFrame, *, label: str) -> dict:
    """Notable loss clusters: by season, by odds, high-edge misses."""
    if len(bets) == 0:
        return {"book": label, "n": 0}
    lost = bets[bets["won"].astype(float) < 0.5]
    won = bets[bets["won"].astype(float) > 0.5]
    m = matches.set_index("match_id")
    tg = bets["match_id"].map(m["total_goals"])
    return {
        "book": label,
        "n": int(len(bets)),
        "n_wins": int(len(won)),
        "n_losses": int(len(lost)),
        "worst_season": (
            bets.groupby("season").apply(lambda g: _roi(g), include_groups=False).idxmin()
            if bets["season"].nunique()
            else None
        ),
        "worst_season_roi": (
            float(bets.groupby("season").apply(lambda g: _roi(g), include_groups=False).min())
            if bets["season"].nunique()
            else None
        ),
        "best_season": (
            bets.groupby("season").apply(lambda g: _roi(g), include_groups=False).idxmax()
            if bets["season"].nunique()
            else None
        ),
        "best_season_roi": (
            float(bets.groupby("season").apply(lambda g: _roi(g), include_groups=False).max())
            if bets["season"].nunique()
            else None
        ),
        "avg_total_goals_on_wins": float(tg[won.index].mean()) if len(won) else None,
        "avg_total_goals_on_losses": float(tg[lost.index].mean()) if len(lost) else None,
        "high_edge_misses_n": int(
            len(lost[lost["edge"] >= 0.12])
        ),
        "high_edge_miss_roi_contribution": float(lost[lost["edge"] >= 0.12]["profit"].sum())
        if len(lost)
        else 0.0,
        "seasons_positive": int(
            sum(
                1
                for _, g in bets.groupby("season")
                if (_roi(g) or 0) > 0
            )
        ),
        "seasons_total": int(bets["season"].nunique()),
    }


def main() -> None:
    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    preds = pd.read_parquet(PRED_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Full packs
    bets_agg = _score(preds, matches, EPL_AGGRESSIVE)
    bets_over = _score(preds, matches, EPL_OVERS_SHORT)

    # OU-only slices for the two "systems" as defined
    under_ou = bets_agg[(bets_agg["market"] == "ou25") & (bets_agg["side"] == "under")].copy()
    over_ou = bets_over[(bets_over["market"] == "ou25") & (bets_over["side"] == "over")].copy()

    # Season tables
    season_under = _season_table(under_ou, label="EPL_aggressive_unders")
    season_over = _season_table(over_ou, label="EPL_overs_short_exp")
    season_agg_all = _season_table(bets_agg, label="EPL_aggressive_ALL")
    season_all = pd.concat([season_under, season_over, season_agg_all], ignore_index=True)
    season_all.to_csv(OUT_DIR / "seasonal_roi.csv", index=False)

    # Bucket diagnostics
    buckets = pd.concat(
        [
            _bucket_table(under_ou, by="odds", label="unders"),
            _bucket_table(over_ou, by="odds", label="short_overs"),
            _bucket_table(under_ou, by="edge", label="unders"),
            _bucket_table(over_ou, by="edge", label="short_overs"),
        ],
        ignore_index=True,
    )
    buckets.to_csv(OUT_DIR / "odds_edge_buckets.csv", index=False)

    # Conflicts
    conf, conf_summary = _conflict_analysis(under_ou, over_ou, matches)
    if len(conf):
        conf.to_csv(OUT_DIR / "conflicts.csv", index=False)
    with open(OUT_DIR / "conflict_summary.json", "w", encoding="utf-8") as f:
        json.dump(conf_summary, f, indent=2)

    # Failure patterns
    failures = [
        _failure_patterns(under_ou, matches, label="unders"),
        _failure_patterns(over_ou, matches, label="short_overs"),
    ]
    with open(OUT_DIR / "failure_patterns.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, default=str)

    # Headline summary
    headline = {
        "model": "20260810T201358Z_iter17_EPL_vol06",
        "unders": {
            "n": int(len(under_ou)),
            "roi": _roi(under_ou),
            "hit_rate": _hit(under_ou),
            "avg_odds": float(under_ou["close_odds"].mean()) if len(under_ou) else None,
            "avg_edge": float(under_ou["edge"].mean()) if len(under_ou) else None,
            "units": float(under_ou["profit"].sum()) if len(under_ou) else 0.0,
            "seasons_pos": failures[0]["seasons_positive"],
            "seasons_n": failures[0]["seasons_total"],
        },
        "short_overs": {
            "n": int(len(over_ou)),
            "roi": _roi(over_ou),
            "hit_rate": _hit(over_ou),
            "avg_odds": float(over_ou["close_odds"].mean()) if len(over_ou) else None,
            "avg_edge": float(over_ou["edge"].mean()) if len(over_ou) else None,
            "units": float(over_ou["profit"].sum()) if len(over_ou) else 0.0,
            "seasons_pos": failures[1]["seasons_positive"],
            "seasons_n": failures[1]["seasons_total"],
        },
        "aggressive_ALL": {
            "n": int(len(bets_agg)),
            "roi": _roi(bets_agg),
            "units": float(bets_agg["profit"].sum()) if len(bets_agg) else 0.0,
        },
        "conflicts": conf_summary,
    }
    with open(OUT_DIR / "headline.json", "w", encoding="utf-8") as f:
        json.dump(headline, f, indent=2)

    # Markdown report
    lines = [
        "# Iter18 — EPL two-book post-mortem (vol06)",
        "",
        f"Model: `{PRED_PATH.parent.name}`",
        "",
        "## Headline",
        "",
        "| Book | n | ROI | Hit | Avg odds | Avg edge | Seasons + |",
        "|------|--:|----:|----:|---------:|---------:|----------:|",
    ]
    for key, name in [
        ("unders", "EPL_aggressive Unders"),
        ("short_overs", "EPL_overs_short_exp"),
    ]:
        h = headline[key]
        lines.append(
            f"| {name} | {h['n']} | {100*(h['roi'] or 0):+.2f}% | "
            f"{100*(h['hit_rate'] or 0):.1f}% | {h['avg_odds']:.3f} | "
            f"{100*(h['avg_edge'] or 0):.1f}% | {h['seasons_pos']}/{h['seasons_n']} |"
        )
    ha = headline["aggressive_ALL"]
    lines.append(
        f"| EPL_aggressive ALL | {ha['n']} | {100*(ha['roi'] or 0):+.2f}% | — | — | — | — |"
    )
    lines += [
        "",
        "## Season-by-season — Unders (OU under only)",
        "",
        "| Season | n | Hit | ROI | Cum n | Cum ROI | Cum u |",
        "|-------:|--:|----:|----:|------:|--------:|------:|",
    ]
    for _, r in season_under.iterrows():
        lines.append(
            f"| {int(r['season'])} | {int(r['n'])} | {100*r['hit_rate']:.1f}% | "
            f"{100*r['roi']:+.1f}% | {int(r['cum_n'])} | {100*r['cum_roi']:+.1f}% | "
            f"{r['cum_units']:+.1f} |"
        )
    lines += [
        "",
        "## Season-by-season — Short Overs",
        "",
        "| Season | n | Hit | ROI | Cum n | Cum ROI | Cum u |",
        "|-------:|--:|----:|----:|------:|--------:|------:|",
    ]
    for _, r in season_over.iterrows():
        lines.append(
            f"| {int(r['season'])} | {int(r['n'])} | {100*r['hit_rate']:.1f}% | "
            f"{100*r['roi']:+.1f}% | {int(r['cum_n'])} | {100*r['cum_roi']:+.1f}% | "
            f"{r['cum_units']:+.1f} |"
        )
    cs = conf_summary
    lines += [
        "",
        "## Conflict analysis (same match: Under pack vs Over pack)",
        "",
        f"- Conflicting games: **{cs['n_conflicts']}**",
        f"- Under side won: **{cs['under_wins']}**",
        f"- Over side won: **{cs['over_wins']}**",
        f"- Net if both books bet same match: **{cs['net_if_both_books_bet']:+.2f} u**",
        f"- Net Under-only on conflicts: **{cs['net_under_side_only']:+.2f} u**",
        f"- Net Over-only on conflicts: **{cs['net_over_side_only']:+.2f} u**",
        "",
        "Books remain separate — conflicts are risk, not a merge signal.",
        "",
        "## Failure patterns",
        "",
    ]
    for fp in failures:
        lines.append(
            f"- **{fp['book']}**: worst season {fp['worst_season']} "
            f"({100*(fp['worst_season_roi'] or 0):+.1f}%), "
            f"best {fp['best_season']} ({100*(fp['best_season_roi'] or 0):+.1f}%); "
            f"high-edge (>=12%) misses n={fp['high_edge_misses_n']} "
            f"(contrib {fp['high_edge_miss_roi_contribution']:+.1f} u); "
            f"avg goals win/loss {fp['avg_total_goals_on_wins']:.2f}/"
            f"{fp['avg_total_goals_on_losses']:.2f}"
        )
    lines += ["", f"Artifacts: `{OUT_DIR.as_posix()}`", ""]
    report = "\n".join(lines)
    (OUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    sys.stdout.buffer.write(report.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
