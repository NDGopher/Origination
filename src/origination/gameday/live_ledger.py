"""Append-only live-system performance ledger.

Records qualified PLAYS from the daily scan and settles them when
final scores are known. Does not change pack rules.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from origination.utils.system_registry import live_systems

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "data" / "gameday" / "live_ledger.csv"
ARCHIVE = ROOT / "experiments" / "gameday_scan" / "history"
RESULTS = ROOT / "data" / "gameday" / "settled_results.csv"
SCAN = ROOT / "experiments" / "gameday_scan"
PERF_JSON = ROOT / "data" / "gameday" / "system_performance.json"

# Date each pack entered the default live scan (metadata only — not a betting rule).
LIVE_SINCE = {
    "EPL_unders": "2026-08-11",
    "EPL_overs_short": "2026-08-11",
    "Bundesliga_unders": "2026-08-11",
    "LaLiga_home_ml": "2026-08-11",
    "SerieA_away_ml": "2026-08-11",
    "Primeira_ah_e12": "2026-08-14",
}
LEDGER_START = "2026-08-14"  # first complete flagged-play log
RECENT_N = 15

LEDGER_COLS = [
    "play_id",
    "recorded_at",
    "system_id",
    "system",
    "league",
    "match_id",
    "date",
    "home_team",
    "away_team",
    "market",
    "side",
    "ah_line",
    "pin_odds",
    "edge_vs_pin",
    "edge_thr",
    "fair_odds",
    "book_odds",
    "recommendation",
    "stake_u",
    "status",  # open | settled | void | postponed
    "actual_home",
    "actual_away",
    "actual_total",
    "won",
    "profit_u",
    "settled_at",
    "settle_note",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _play_id(row: pd.Series) -> str:
    return "|".join(
        [
            str(row.get("system_id") or ""),
            str(row.get("match_id") or ""),
            str(row.get("side") or ""),
            str(row.get("market") or ""),
        ]
    )


def load_ledger() -> pd.DataFrame:
    if not LEDGER.is_file():
        return pd.DataFrame(columns=LEDGER_COLS)
    df = pd.read_csv(LEDGER)
    for c in LEDGER_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[LEDGER_COLS]
    for c in (
        "play_id",
        "recorded_at",
        "system_id",
        "system",
        "league",
        "match_id",
        "date",
        "home_team",
        "away_team",
        "market",
        "side",
        "recommendation",
        "status",
        "settled_at",
        "settle_note",
    ):
        df[c] = df[c].astype("string")
    return df


def save_ledger(df: pd.DataFrame) -> Path:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    out = df[LEDGER_COLS].copy() if len(df) else pd.DataFrame(columns=LEDGER_COLS)
    out.to_csv(LEDGER, index=False)
    return LEDGER


def archive_scan() -> Path | None:
    plays = SCAN / "QUALIFIED_PLAYS.csv"
    if not plays.is_file():
        return None
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = ARCHIVE / f"{stamp}_QUALIFIED_PLAYS.csv"
    dest.write_bytes(plays.read_bytes())
    nears = SCAN / "NEAR_MISSES.csv"
    if nears.is_file():
        (ARCHIVE / f"{stamp}_NEAR_MISSES.csv").write_bytes(nears.read_bytes())
    return dest


def record_from_scan(plays: pd.DataFrame | None = None, *, stake_u: float = 1.0) -> int:
    """Append new qualified PLAYS. Returns number added."""
    if plays is None:
        p = SCAN / "QUALIFIED_PLAYS.csv"
        if not p.is_file():
            return 0
        plays = pd.read_csv(p)
    if plays is None or len(plays) == 0:
        return 0
    if "qualifies" in plays.columns:
        plays = plays[plays["qualifies"] == True]
    if len(plays) == 0:
        return 0

    ledger = load_ledger()
    existing = set(ledger["play_id"].astype(str)) if len(ledger) else set()
    added = 0
    rows = [] if len(ledger) == 0 else [ledger]
    new_rows = []
    for _, r in plays.iterrows():
        pid = _play_id(r)
        if pid in existing:
            continue
        new_rows.append(
            {
                "play_id": pid,
                "recorded_at": _now(),
                "system_id": r.get("system_id"),
                "system": r.get("system"),
                "league": r.get("league"),
                "match_id": r.get("match_id"),
                "date": str(r.get("date") or "")[:10],
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
                "market": r.get("market"),
                "side": r.get("side"),
                "ah_line": r.get("ah_line"),
                "pin_odds": r.get("pin_odds"),
                "edge_vs_pin": r.get("edge_vs_pin"),
                "edge_thr": r.get("edge_thr"),
                "fair_odds": r.get("fair_odds"),
                "book_odds": r.get("book_odds"),
                "recommendation": r.get("recommendation") or "PLAY",
                "stake_u": stake_u,
                "status": "open",
                "actual_home": pd.NA,
                "actual_away": pd.NA,
                "actual_total": pd.NA,
                "won": pd.NA,
                "profit_u": pd.NA,
                "settled_at": pd.NA,
                "settle_note": "",
            }
        )
        existing.add(pid)
        added += 1
    if new_rows:
        extra = pd.DataFrame(new_rows)
        ledger = pd.concat(rows + [extra], ignore_index=True) if rows else extra
        save_ledger(ledger)
    archive_scan()
    return added


def load_results_table() -> pd.DataFrame:
    frames = []
    if RESULTS.is_file():
        frames.append(pd.read_csv(RESULTS))
    retro = ROOT / "experiments" / "weekend_retro" / "actuals.csv"
    if retro.is_file():
        frames.append(pd.read_csv(retro))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["actual_home", "actual_away"], how="any")
    return df.drop_duplicates(subset=["match_id"], keep="last")


def _ah_cover(home: float, away: float, line: float, side: str) -> str:
    """Return win / lose / push for AH. `line` is the home line (e.g. -1.5)."""
    gd = home - away
    if "away" in str(side).lower():
        margin = -gd - line  # away gets opposite of home line
    else:
        margin = gd + line
    if abs(margin) < 1e-9:
        return "push"
    return "win" if margin > 0 else "lose"


def _settle_one(row: pd.Series, home: float, away: float) -> dict[str, Any]:
    total = home + away
    market = str(row.get("market") or "")
    side = str(row.get("side") or "").lower()
    odds = float(row["pin_odds"]) if pd.notna(row.get("pin_odds")) else None
    stake = float(row.get("stake_u") or 1.0)
    note = ""
    outcome = None

    if market.startswith("OU") or side in ("under", "over"):
        if total > 2.5:
            outcome = "win" if "over" in side else "lose"
        elif total < 2.5:
            outcome = "win" if "under" in side else "lose"
        else:
            outcome = "push"
    elif market.startswith("1X2") or side in ("h", "a", "d", "home", "away"):
        if home > away:
            res = "h"
        elif away > home:
            res = "a"
        else:
            res = "d"
        want = {"home": "h", "h": "h", "away": "a", "a": "a", "d": "d", "draw": "d"}.get(side, side[:1])
        outcome = "win" if want == res else "lose"
    elif market == "AH" or "ah" in side:
        try:
            line = float(row.get("ah_line"))
        except (TypeError, ValueError):
            return {"status": "open", "settle_note": "missing AH line"}
        outcome = _ah_cover(home, away, line, side)
    else:
        return {"status": "open", "settle_note": f"unknown market {market}"}

    if outcome == "push" or odds is None:
        profit = 0.0
        won = 0 if outcome != "win" else 1
        if outcome == "push":
            note = "push — stake returned"
            won = pd.NA
    elif outcome == "win":
        profit = stake * (odds - 1.0)
        won = 1
    else:
        profit = -stake
        won = 0

    return {
        "status": "settled",
        "actual_home": home,
        "actual_away": away,
        "actual_total": total,
        "won": won,
        "profit_u": round(float(profit), 4),
        "settled_at": _now(),
        "settle_note": note or outcome,
    }


def settle_open() -> int:
    ledger = load_ledger()
    if len(ledger) == 0:
        return 0
    results = load_results_table()
    if len(results) == 0:
        return 0
    by_id = {str(r["match_id"]): r for _, r in results.iterrows()}
    n = 0
    for i, row in ledger.iterrows():
        if str(row.get("status")) != "open":
            continue
        mid = str(row.get("match_id") or "")
        res = by_id.get(mid)
        if res is None:
            continue
        if str(res.get("notes") or "").upper().find("POSTPONED") >= 0:
            ledger.at[i, "status"] = "postponed"
            ledger.at[i, "settled_at"] = _now()
            ledger.at[i, "settle_note"] = "postponed"
            n += 1
            continue
        try:
            h = float(res["actual_home"])
            a = float(res["actual_away"])
        except (TypeError, ValueError):
            continue
        upd = _settle_one(row, h, a)
        for k, v in upd.items():
            ledger.loc[i, k] = v
        n += 1
    save_ledger(ledger)
    return n


def _max_dd_u(profits: pd.Series) -> float | None:
    p = pd.to_numeric(profits, errors="coerce").fillna(0.0)
    if len(p) == 0:
        return None
    equity = p.cumsum()
    dd = equity - equity.cummax()
    return round(float(dd.min()), 2)


def _form_str(won: pd.Series, n: int = 10) -> str:
    marks = []
    for v in won.tolist()[-n:]:
        if pd.isna(v):
            marks.append("P")
        else:
            marks.append("W" if int(v) == 1 else "L")
    return " ".join(marks) if marks else "—"


def _live_slice(g: pd.DataFrame) -> dict[str, Any]:
    if g is None or len(g) == 0:
        return {
            "n": 0,
            "n_open": 0,
            "n_settled": 0,
            "n_decided": 0,
            "wins": 0,
            "hit": None,
            "units": 0.0,
            "roi": None,
            "max_dd_u": None,
            "form": "—",
        }
    settled = g[g["status"] == "settled"]
    decided = settled[pd.notna(settled["won"])] if len(settled) else settled
    stake = float(pd.to_numeric(settled.get("stake_u"), errors="coerce").fillna(0).sum()) if len(settled) else 0.0
    profit = float(pd.to_numeric(settled.get("profit_u"), errors="coerce").fillna(0).sum()) if len(settled) else 0.0
    wins = int(pd.to_numeric(decided["won"], errors="coerce").fillna(0).sum()) if len(decided) else 0
    n_dec = int(len(decided))
    ordered = settled.sort_values(["date", "recorded_at"], na_position="last") if len(settled) else settled
    return {
        "n": int(len(g)),
        "n_open": int((g["status"] == "open").sum()),
        "n_settled": int(len(settled)),
        "n_decided": n_dec,
        "wins": wins,
        "hit": None if n_dec == 0 else round(wins / n_dec, 4),
        "units": round(profit, 2),
        "roi": None if stake <= 0 else round(profit / stake, 4),
        "max_dd_u": _max_dd_u(ordered["profit_u"]) if len(ordered) else None,
        "form": _form_str(ordered["won"], 10) if len(ordered) else "—",
    }


def performance_snapshot(*, recent_n: int = RECENT_N) -> dict[str, Any]:
    """Backtest (registry) + live ledger + recent form. Does not change pack rules."""
    df = load_ledger()
    systems = []
    for sys_ in live_systems():
        sid = sys_["id"]
        hist = sys_.get("history") or {}
        g = df[df["system_id"] == sid] if len(df) else df
        live = _live_slice(g)
        settled = g[g["status"] == "settled"] if len(g) else g
        if len(settled):
            settled = settled.sort_values(["date", "recorded_at"], na_position="last")
            recent = _live_slice(settled.tail(recent_n))
        else:
            recent = _live_slice(pd.DataFrame(columns=LEDGER_COLS))
        systems.append(
            {
                "system_id": sid,
                "system": sys_["name"],
                "pack": sys_["pack"],
                "status": sys_.get("status"),
                "rules": sys_.get("rules_text"),
                "live_since": LIVE_SINCE.get(sid, LEDGER_START),
                "backtest": {
                    "n": hist.get("n"),
                    "roi": hist.get("roi"),
                    "hit": hist.get("hit"),
                    "units": None,
                    "max_dd_u": hist.get("max_dd_u"),
                    "t_stat": hist.get("t_stat"),
                    "seasons_pos": hist.get("seasons_pos"),
                    "seasons_n": hist.get("seasons_n"),
                    "source": hist.get("source"),
                },
                "live": live,
                "recent": recent,
            }
        )
    plays = []
    if len(df):
        show = df.sort_values(["date", "recorded_at"], na_position="last")
        for _, r in show.iterrows():
            won = r.get("won")
            if str(r.get("status")) == "open":
                result = "open"
            elif str(r.get("status")) == "postponed":
                result = "postponed"
            elif pd.isna(won):
                result = "push"
            else:
                result = "W" if int(won) == 1 else "L"
            plays.append(
                {
                    "date": str(r.get("date") or "")[:10],
                    "system": r.get("system"),
                    "system_id": r.get("system_id"),
                    "match": f"{r.get('home_team')} vs {r.get('away_team')}",
                    "home_team": r.get("home_team"),
                    "away_team": r.get("away_team"),
                    "market": r.get("market"),
                    "side": r.get("side"),
                    "ah_line": r.get("ah_line") if pd.notna(r.get("ah_line")) else None,
                    "pin_odds": r.get("pin_odds") if pd.notna(r.get("pin_odds")) else None,
                    "edge_vs_pin": r.get("edge_vs_pin") if pd.notna(r.get("edge_vs_pin")) else None,
                    "status": r.get("status"),
                    "result": result,
                    "actual": (
                        f"{r.get('actual_home')}-{r.get('actual_away')}"
                        if pd.notna(r.get("actual_home"))
                        else None
                    ),
                    "profit_u": None if pd.isna(r.get("profit_u")) else float(r.get("profit_u")),
                }
            )
    snap = {
        "updated_at": _now(),
        "ledger_start": LEDGER_START,
        "recent_n": recent_n,
        "n_open": int((df["status"] == "open").sum()) if len(df) else 0,
        "n_settled": int((df["status"] == "settled").sum()) if len(df) else 0,
        "n_total": int(len(df)),
        "systems": systems,
        "plays": plays,
        "note": (
            "Live ledger records every PLAY the scan flags, whether or not it was bet. "
            f"Complete log from {LEDGER_START}. Backtest figures are signed walk-forward, not live."
        ),
    }
    PERF_JSON.parent.mkdir(parents=True, exist_ok=True)
    PERF_JSON.write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")
    return snap


def summary() -> dict[str, Any]:
    snap = performance_snapshot()
    return {
        "updated_at": snap["updated_at"],
        "n_open": snap["n_open"],
        "n_settled": snap["n_settled"],
        "n_total": snap["n_total"],
        "systems": [
            {
                "system_id": s["system_id"],
                "system": s["system"],
                "n": s["live"]["n"],
                "n_decided": s["live"]["n_decided"],
                "wins": s["live"]["wins"],
                "units": s["live"]["units"],
                "roi": s["live"]["roi"],
                "open": s["live"]["n_open"],
            }
            for s in snap["systems"]
            if s["live"]["n"] > 0
        ],
    }


def write_report() -> Path:
    snap = performance_snapshot()
    df = load_ledger()
    lines = [
        "# Live systems — performance ledger",
        "",
        f"Updated: {snap['updated_at']}",
        "",
        "Protected pack **rules are unchanged**. This is a record of flagged plays only, whether or not they were bet.",
        "",
        f"Live log starts **{snap['ledger_start']}**. ROI on small n is **not** a system evaluation — walk-forward history still governs promotion.",
        "",
        f"Open: **{snap['n_open']}** · Settled: **{snap['n_settled']}** · Total logged: **{snap['n_total']}**",
        "",
        "## By system (backtest vs live vs recent)",
        "",
        "| System | Backtest n | BT ROI | BT DD | Live n | Live W-L | Live units | Live ROI | Live DD | Recent form | Open |",
        "|--------|-----------:|-------:|------:|-------:|----------|-----------:|---------:|--------:|-------------|-----:|",
    ]
    for s in snap["systems"]:
        bt = s["backtest"]
        lv = s["live"]
        rec = s["recent"]
        bt_roi = "—" if bt.get("roi") is None else f"{100*bt['roi']:+.1f}%"
        bt_dd = "—" if bt.get("max_dd_u") is None else f"{bt['max_dd_u']:+.1f}u"
        live_roi = "—" if lv.get("roi") is None else f"{100*lv['roi']:+.1f}%"
        live_dd = "—" if lv.get("max_dd_u") is None else f"{lv['max_dd_u']:+.1f}u"
        wl = f"{lv['wins']}-{lv['n_decided']-lv['wins']}" if lv["n_decided"] else "—"
        units_s = "—" if lv["n_settled"] == 0 else f"{lv['units']:+.2f}u"
        lines.append(
            f"| {s['system']} | {bt.get('n') or 0} | {bt_roi} | {bt_dd} | "
            f"{lv['n']} | {wl} | {units_s} | {live_roi} | {live_dd} | "
            f"{rec.get('form') or '—'} | {lv['n_open']} |"
        )
    lines += ["", "## Open plays", ""]
    open_ = df[df["status"] == "open"] if len(df) else df
    if len(open_) == 0:
        lines.append("_None._")
    else:
        for _, r in open_.iterrows():
            edge = r.get("edge_vs_pin")
            edge_s = "" if pd.isna(edge) else f" · edge {100*float(edge):+.1f}%"
            lines.append(
                f"- {r.get('date')} · **{r.get('system')}** · {r.get('home_team')} vs {r.get('away_team')} · "
                f"{r.get('side')} @ {r.get('pin_odds')}{edge_s}"
            )
    lines += ["", "## Settled", ""]
    settled = df[df["status"] == "settled"] if len(df) else df
    if len(settled) == 0:
        lines.append("_None yet._")
    else:
        for _, r in settled.iterrows():
            won = r.get("won")
            mark = "P" if pd.isna(won) else ("W" if int(won) == 1 else "L")
            profit = r.get("profit_u")
            profit_s = "—" if pd.isna(profit) else f"{float(profit):+.3f}u"
            lines.append(
                f"- {mark} {r.get('date')} · **{r.get('system')}** · {r.get('home_team')} vs {r.get('away_team')} "
                f"{r.get('actual_home')}-{r.get('actual_away')} · {r.get('side')} @ {r.get('pin_odds')} · "
                f"{profit_s}"
            )
    out = ROOT / "docs" / "LIVE_LEDGER.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (SCAN / "LIVE_LEDGER.md").write_text("\n".join(lines), encoding="utf-8")
    (LEDGER.parent / "live_ledger_summary.json").write_text(
        json.dumps({k: snap[k] for k in ("updated_at", "n_open", "n_settled", "n_total", "systems")}, indent=2, default=str),
        encoding="utf-8",
    )
    return out
