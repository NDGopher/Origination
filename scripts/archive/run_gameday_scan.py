#!/usr/bin/env python
"""
Multi-league gameday scan across the 5 active systems.

Systems (rules frozen):
  1. EPL Unders          — under 2.00–4.00 @ e≥8%
  2. EPL short Overs     — over  1.60–2.50 @ e≥10%
  3. Bundesliga Unders   — under 1.70–2.50 @ e≥10%
  4. La Liga Home ML     — home  max 1.80 @ e≥8%
  5. Serie A Away ML     — away  max 2.00 @ e≥3%
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.utils.odds import model_edge_vs_odds, model_edge_vs_two_way

OUT = ROOT / "experiments" / "gameday_scan"
OUT.mkdir(parents=True, exist_ok=True)

LEAGUES = ["EPL", "Bundesliga", "LaLiga", "SerieA"]

SYSTEMS = [
    {
        "name": "EPL Unders",
        "league": "EPL",
        "market": "OU Under",
        "side": "under",
        "min_odds": 2.00,
        "max_odds": 4.00,
        "edge_thr": 0.08,
        "odds_col": "pin_under25",
        "edge_col": "edge_under_vs_pinnacle",
        "prob_col": "p_under25",
    },
    {
        "name": "EPL short Overs",
        "league": "EPL",
        "market": "OU Over",
        "side": "over",
        "min_odds": 1.60,
        "max_odds": 2.50,
        "edge_thr": 0.10,
        "odds_col": "pin_over25",
        "edge_col": "edge_over_vs_pinnacle",
        "prob_col": "p_over25",
    },
    {
        "name": "Bundesliga Unders",
        "league": "Bundesliga",
        "market": "OU Under",
        "side": "under",
        "min_odds": 1.70,
        "max_odds": 2.50,
        "edge_thr": 0.10,
        "odds_col": "pin_under25",
        "edge_col": "edge_under_vs_pinnacle",
        "prob_col": "p_under25",
    },
    {
        "name": "La Liga Home ML",
        "league": "LaLiga",
        "market": "1X2 Home",
        "side": "H",
        "min_odds": 1.01,
        "max_odds": 1.80,
        "edge_thr": 0.08,
        "odds_col": "odds_1x2_h",
        "edge_col": "edge_1x2_h",
        "prob_col": "p_home",
    },
    {
        "name": "Serie A Away ML",
        "league": "SerieA",
        "market": "1X2 Away",
        "side": "A",
        "min_odds": 1.01,
        "max_odds": 2.00,
        "edge_thr": 0.03,
        "odds_col": "odds_1x2_a",
        "edge_col": "edge_1x2_a",
        "prob_col": "p_away",
    },
]


def _py() -> Path:
    return ROOT / ".venv" / "Scripts" / "python.exe"


def run_league_sheet(league: str) -> Path:
    out = ROOT / "data" / "processed" / f"gameday_sheet_{league}.csv"
    if league == "EPL":
        out = ROOT / "data" / "processed" / "gameday_sheet.csv"
    cmd = [
        str(_py()),
        str(ROOT / "scripts" / "run_gameday_sheet.py"),
        "--league",
        league,
        "--refresh-fixtures",
        "--refresh-odds",
        "--fast",
        "--out",
        str(out),
        "--log-level",
        "WARNING",
    ]
    print(f"\n=== {league} ===", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return out


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def evaluate_row(sys_: dict, row: pd.Series) -> dict:
    odds = _f(row.get(sys_["odds_col"]))
    if odds is None and sys_["odds_col"] == "odds_1x2_h":
        odds = _f(row.get("pin_h"))
    if odds is None and sys_["odds_col"] == "odds_1x2_a":
        odds = _f(row.get("pin_a"))
    edge = _f(row.get(sys_["edge_col"]))
    # recompute edge if missing but we have probs+odds
    if edge is None:
        prob = _f(row.get(sys_["prob_col"]))
        if sys_["market"].startswith("OU") and odds is not None and prob is not None:
            # prefer two-way if both sides present
            o = _f(row.get("pin_over25"))
            u = _f(row.get("pin_under25"))
            if o and u:
                edge = model_edge_vs_two_way(
                    prob, o, u, side=sys_["side"], method="power"
                )
            else:
                edge = model_edge_vs_odds(prob, odds)
        elif odds is not None and prob is not None:
            edge = model_edge_vs_odds(prob, odds)

    thr = sys_["edge_thr"]
    lo, hi = sys_["min_odds"], sys_["max_odds"]
    in_band = odds is not None and lo <= odds <= hi
    edge_ok = edge is not None and edge >= thr
    qualifies = bool(in_band and edge_ok)

    # Near-miss: edge within 2pp of thr with odds in band, OR edge OK with odds within 0.10 of band
    near = False
    near_reason = ""
    if not qualifies and odds is not None and edge is not None:
        if in_band and (thr - 0.02) <= edge < thr:
            near = True
            near_reason = f"edge {100*edge:.1f}% just below {100*thr:.0f}%"
        elif edge_ok and odds < lo and odds >= lo - 0.10:
            near = True
            near_reason = f"odds {odds:.2f} just below band {lo:.2f}"
        elif edge_ok and odds > hi and odds <= hi + 0.10:
            near = True
            near_reason = f"odds {odds:.2f} just above band {hi:.2f}"
        elif (not in_band) and (thr - 0.02) <= edge < thr and (
            (lo - 0.10) <= odds <= (hi + 0.10)
        ):
            near = True
            near_reason = "edge+odds both close"

    return {
        "system": sys_["name"],
        "league": sys_["league"],
        "market": sys_["market"],
        "side": sys_["side"],
        "date": str(row.get("date", ""))[:10],
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "match_id": row.get("match_id"),
        "proj_home": _f(row.get("proj_home_goals")),
        "proj_away": _f(row.get("proj_away_goals")),
        "proj_total": _f(row.get("proj_total_goals")),
        "model_prob": _f(row.get(sys_["prob_col"])),
        "pin_odds": odds,
        "edge": edge,
        "edge_thr": thr,
        "odds_band": f"{lo:.2f}–{hi:.2f}",
        "qualifies": qualifies,
        "near_miss": near,
        "near_reason": near_reason,
        "recommendation": "PLAY" if qualifies else ("WATCH" if near else "NO PLAY"),
        "pin_over25": _f(row.get("pin_over25")),
        "pin_under25": _f(row.get("pin_under25")),
        "pin_h": _f(row.get("pin_h")),
        "pin_d": _f(row.get("pin_d")),
        "pin_a": _f(row.get("pin_a")),
        "p_home": _f(row.get("p_home")),
        "p_draw": _f(row.get("p_draw")),
        "p_away": _f(row.get("p_away")),
        "p_over25": _f(row.get("p_over25")),
        "p_under25": _f(row.get("p_under25")),
    }


def main() -> None:
    sheets: dict[str, pd.DataFrame] = {}
    for lg in LEAGUES:
        path = run_league_sheet(lg)
        if not path.exists():
            print(f"WARNING: missing sheet {path}")
            continue
        df = pd.read_csv(path)
        sheets[lg] = df
        print(f"  sheet rows={len(df)} cols={len(df.columns)}", flush=True)

    plays = []
    nears = []
    all_eval = []
    for sys_ in SYSTEMS:
        df = sheets.get(sys_["league"])
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            ev = evaluate_row(sys_, row)
            all_eval.append(ev)
            if ev["qualifies"]:
                plays.append(ev)
            elif ev["near_miss"]:
                nears.append(ev)

    plays_df = pd.DataFrame(plays)
    nears_df = pd.DataFrame(nears)
    all_df = pd.DataFrame(all_eval)

    plays_path = OUT / "QUALIFIED_PLAYS.csv"
    nears_path = OUT / "NEAR_MISSES.csv"
    all_path = OUT / "all_system_evals.csv"
    plays_df.to_csv(plays_path, index=False)
    nears_df.to_csv(nears_path, index=False)
    all_df.to_csv(all_path, index=False)

    # Per-league sheets already written under data/processed
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Multi-league gameday scan — {ts}",
        "",
        "Active systems (rules unchanged):",
        "1. EPL Unders — Under 2.00-4.00 @ edge >=8%",
        "2. EPL short Overs — Over 1.60-2.50 @ edge >=10%",
        "3. Bundesliga Unders — Under 1.70-2.50 @ edge >=10%",
        "4. La Liga Home ML — Home @ edge >=8%, max odds 1.80",
        "5. Serie A Away ML — Away @ edge >=3%, max odds 2.00",
        "",
        f"## Qualified plays: **{len(plays_df)}**",
        "",
    ]
    if len(plays_df) == 0:
        lines.append("_No games currently qualify across the 5 systems._")
    else:
        lines.append(
            "| Rec | System | League | Date | Match | Side | Proj (H-A / Tot) | Model p | Pin odds | Edge | Band |"
        )
        lines.append(
            "|-----|--------|--------|------|-------|------|------------------|--------:|---------:|-----:|------|"
        )
        for _, r in plays_df.sort_values(["date", "league", "system"]).iterrows():
            proj = (
                f"{r['proj_home']:.2f}-{r['proj_away']:.2f} / {r['proj_total']:.2f}"
                if r["proj_home"] is not None
                else "—"
            )
            lines.append(
                f"| **PLAY** | {r['system']} | {r['league']} | {r['date']} | "
                f"{r['home_team']} vs {r['away_team']} | {r['side']} | {proj} | "
                f"{100*(r['model_prob'] or 0):.1f}% | {r['pin_odds']:.2f} | "
                f"{100*(r['edge'] or 0):+.1f}% | {r['odds_band']} |"
            )

    lines += ["", f"## Near misses: **{len(nears_df)}**", ""]
    if len(nears_df) == 0:
        lines.append("_None within ~2pp edge or ~0.10 odds of the filters._")
    else:
        lines.append(
            "| Rec | System | Date | Match | Pin odds | Edge | Why close |"
        )
        lines.append("|-----|--------|------|-------|---------:|-----:|-----------|")
        for _, r in nears_df.sort_values(["date", "system"]).iterrows():
            lines.append(
                f"| WATCH | {r['system']} | {r['date']} | "
                f"{r['home_team']} vs {r['away_team']} | "
                f"{(r['pin_odds'] or 0):.2f} | {100*(r['edge'] or 0):+.1f}% | {r['near_reason']} |"
            )

    lines += [
        "",
        "## Coverage",
        "",
    ]
    for lg in LEAGUES:
        df = sheets.get(lg)
        n = len(df) if df is not None else 0
        if df is None or n == 0:
            lines.append(f"- **{lg}**: no sheet / 0 fixtures")
            continue
        if "odds_status" in df.columns:
            miss = int((df["odds_status"] == "MISSING").sum())
            ou = int(df["has_pin_ou"].sum()) if "has_pin_ou" in df.columns else 0
            ml = int(df["has_pin_1x2"].sum()) if "has_pin_1x2" in df.columns else 0
        else:
            ou = int(df["pin_over25"].notna().sum()) if "pin_over25" in df.columns else 0
            ml = int(df["odds_1x2_h"].notna().sum()) if "odds_1x2_h" in df.columns else 0
            miss = max(0, n - max(ou, ml))
        flagged = 0
        if "flag_any" in df.columns:
            flagged = int(df["flag_any"].fillna(False).astype(bool).sum())
        elif "systems_flagged" in df.columns:
            flagged = int(df["systems_flagged"].astype(str).str.len().gt(0).sum())
        lines.append(
            f"- **{lg}**: {n} fixtures · OU={ou} · 1X2={ml} · missing={miss} · flagged={flagged}"
        )

    lines += [
        "",
        "## Files",
        "",
        f"- Qualified: `{plays_path}`",
        f"- Near misses: `{nears_path}`",
        f"- Full eval: `{all_path}`",
        "- League sheets: `data/processed/gameday_sheet*.csv`",
        "",
    ]
    report = OUT / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
