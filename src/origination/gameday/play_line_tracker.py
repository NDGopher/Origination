"""
Track Pinnacle line movement for flagged PLAYS / WATCH from first radar → close.

Append one row per daily scan observation. At kickoff (or ledger settle), freeze
"close" as the last pre-kickoff Pin price and compute CLV vs first sight / entry.

Positive odds-space CLV = (bet_odds / close_odds) - 1 on the bet side:
  you got a longer price than where the line closed (steam toward your pick).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HISTORY = ROOT / "data" / "gameday" / "play_line_history.csv"
SUMMARY = ROOT / "data" / "gameday" / "play_line_summary.csv"
LEDGER = ROOT / "data" / "gameday" / "live_ledger.csv"
SCAN_HISTORY = ROOT / "experiments" / "gameday_scan" / "history"
REPORT = ROOT / "docs" / "LINE_MOVES.md"
SCAN_LOG = ROOT / "data" / "gameday" / "line_scan_log.jsonl"
DAILY_RUNS = ROOT / "docs" / "daily_runs"

HISTORY_COLS = [
    "observed_at",
    "scan_stamp",
    "play_id",
    "tier",
    "match_id",
    "system_id",
    "system",
    "league",
    "market",
    "side",
    "home_team",
    "away_team",
    "match_date",
    "kickoff_utc",
    "hours_to_kick",
    "pin_odds",
    "edge_vs_pin",
    "model_prob",
    "fair_odds",
    "recommendation",
]

SUMMARY_COLS = [
    "play_id",
    "match_id",
    "system_id",
    "system",
    "league",
    "market",
    "side",
    "home_team",
    "away_team",
    "match_date",
    "kickoff_utc",
    "first_seen_at",
    "first_tier",
    "first_pin_odds",
    "first_edge",
    "entry_pin_odds",
    "entry_at",
    "last_seen_at",
    "last_pin_odds",
    "last_edge",
    "close_pin_odds",
    "close_observed_at",
    "n_observations",
    "clv_first_pct",
    "clv_entry_pct",
    "clv_last_pct",
    "steam_vs_first",
    "status",
    "timing_note",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _play_id(row: pd.Series | dict) -> str:
    if isinstance(row, dict):
        row = pd.Series(row)
    return "|".join(
        [
            str(row.get("system_id") or ""),
            str(row.get("match_id") or ""),
            str(row.get("side") or ""),
            str(row.get("market") or ""),
        ]
    )


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def odds_clv_pct(bet_odds: float | None, close_odds: float | None) -> float | None:
    """Percent CLV in decimal-odds space for the bet side."""
    if bet_odds is None or close_odds is None or close_odds <= 1.0 or bet_odds <= 1.0:
        return None
    return round((bet_odds / close_odds - 1.0) * 100.0, 2)


def steam_label(first_odds: float | None, close_odds: float | None, *, flat_pp: float = 1.0) -> str:
    clv = odds_clv_pct(first_odds, close_odds)
    if clv is None:
        return "unknown"
    if clv > flat_pp:
        return "toward_us"
    if clv < -flat_pp:
        return "against_us"
    return "flat"


def bet_timing_action(
    first_odds: float | None,
    last_odds: float | None,
    *,
    clv_last_pct: float | None = None,
    n_obs: int = 1,
    tier: str = "WATCH",
) -> str:
    """Actionable bet-timing label for live deployment."""
    if first_odds is None or last_odds is None or n_obs < 2:
        return "MONITOR" if tier == "WATCH" else "INSUFFICIENT_DATA"
    clv = clv_last_pct if clv_last_pct is not None else odds_clv_pct(first_odds, last_odds)
    if clv is None:
        return "MONITOR"
    if clv >= 2.0:
        return "BET_NOW"
    if clv <= -2.0:
        return "WAIT"
    if tier == "PLAY" and clv >= 0:
        return "BET_NOW"
    if tier == "PLAY" and clv < -1.0:
        return "WAIT"
    return "MONITOR"


def timing_note(first: float | None, entry: float | None, close: float | None) -> str:
    cf = odds_clv_pct(first, close)
    ce = odds_clv_pct(entry, close) if entry else None
    if cf is None:
        return "insufficient line history"
    if cf > 2.0:
        return "early entry rewarded — line steamed toward us by close"
    if cf < -2.0:
        return "early entry hurt — wait-for-close would have been better"
    if ce is not None and entry and first and abs(entry - first) / first > 0.02:
        if ce > cf + 1.0:
            return "ledger entry beat first radar — late add was fine"
        if ce < cf - 1.0:
            return "first radar was best price — bet when flagged"
    return "flat market — early vs close similar"


def _load_kickoffs() -> dict[str, str]:
    out: dict[str, str] = {}
    interim = ROOT / "data" / "interim"
    for p in interim.glob("fixtures_upcoming_*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        if "match_id" not in df.columns:
            continue
        kcol = "kickoff_utc" if "kickoff_utc" in df.columns else None
        if not kcol:
            continue
        for _, r in df.iterrows():
            mid = str(r["match_id"])
            k = r.get(kcol)
            if pd.notna(k):
                out[mid] = str(k)
    return out


def load_history() -> pd.DataFrame:
    if not HISTORY.is_file():
        return pd.DataFrame(columns=HISTORY_COLS)
    df = pd.read_csv(HISTORY)
    for c in HISTORY_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[HISTORY_COLS]


def save_history(df: pd.DataFrame) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    df[HISTORY_COLS].to_csv(HISTORY, index=False)


def load_summary() -> pd.DataFrame:
    if not SUMMARY.is_file():
        return pd.DataFrame(columns=SUMMARY_COLS)
    df = pd.read_csv(SUMMARY)
    for c in SUMMARY_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[SUMMARY_COLS]


def save_summary(df: pd.DataFrame) -> None:
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    df[SUMMARY_COLS].to_csv(SUMMARY, index=False)


def _hours_to_kick(kickoff_utc: str | None, observed_at: str) -> float | None:
    if not kickoff_utc:
        return None
    try:
        k = pd.Timestamp(kickoff_utc)
        if k.tzinfo is None:
            k = k.tz_localize("UTC")
        o = pd.Timestamp(observed_at)
        if o.tzinfo is None:
            o = o.tz_localize("UTC")
        return round(float((k - o).total_seconds() / 3600.0), 2)
    except Exception:  # noqa: BLE001
        return None


def _row_from_eval(row: pd.Series, *, tier: str, kickoffs: dict[str, str], stamp: str) -> dict:
    observed = _now()
    mid = str(row.get("match_id") or "")
    k = kickoffs.get(mid)
    return {
        "observed_at": observed,
        "scan_stamp": stamp,
        "play_id": _play_id(row),
        "tier": tier,
        "match_id": mid,
        "system_id": row.get("system_id"),
        "system": row.get("system"),
        "league": row.get("league"),
        "market": row.get("market"),
        "side": row.get("side"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "match_date": str(row.get("date") or "")[:10],
        "kickoff_utc": k,
        "hours_to_kick": _hours_to_kick(k, observed),
        "pin_odds": _f(row.get("pin_odds")),
        "edge_vs_pin": _f(row.get("edge_vs_pin")),
        "model_prob": _f(row.get("model_prob")),
        "fair_odds": _f(row.get("fair_odds")),
        "recommendation": row.get("recommendation") or tier,
    }


def _refresh_summary_from_history(hist: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if len(hist) == 0:
        return summary

    ledger_entry: dict[str, tuple[float | None, str | None]] = {}
    if LEDGER.is_file():
        lg = pd.read_csv(LEDGER)
        for _, r in lg.iterrows():
            pid = str(r.get("play_id") or "")
            ledger_entry[pid] = (_f(r.get("pin_odds")), str(r.get("recorded_at") or "")[:25])

    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for pid, g in hist.groupby("play_id", sort=False):
        g = g.sort_values("observed_at")
        first = g.iloc[0]
        last = g.iloc[-1]
        kick = first.get("kickoff_utc") or last.get("kickoff_utc")
        kick_ts = None
        if kick and pd.notna(kick):
            try:
                kick_ts = pd.Timestamp(kick)
                if kick_ts.tzinfo is None:
                    kick_ts = kick_ts.tz_localize("UTC")
            except Exception:  # noqa: BLE001
                kick_ts = None
        if kick_ts is None and first.get("match_date") and pd.notna(first.get("match_date")):
            try:
                md = pd.Timestamp(str(first.get("match_date")))
                if md.tzinfo is None:
                    md = md.tz_localize("UTC")
                # Default 20:00 UTC on match day when kickoff not on file (finished fixtures drop off)
                kick_ts = md + pd.Timedelta(hours=20)
            except Exception:  # noqa: BLE001
                kick_ts = None

        pre = g.copy()
        if kick_ts is not None:
            pre["_obs"] = pd.to_datetime(pre["observed_at"], utc=True, errors="coerce")
            pre = pre[pre["_obs"] <= kick_ts]
        close_row = pre.iloc[-1] if len(pre) else last
        close_odds = _f(close_row.get("pin_odds"))
        close_at = str(close_row.get("observed_at") or "")

        entry_odds, entry_at = ledger_entry.get(str(pid), (None, None))
        if entry_odds is None:
            play_rows = g[g["tier"] == "PLAY"]
            if len(play_rows):
                entry_odds = _f(play_rows.iloc[0].get("pin_odds"))
                entry_at = str(play_rows.iloc[0].get("observed_at") or "")[:25]

        first_odds = _f(first.get("pin_odds"))
        last_odds = _f(last.get("pin_odds"))
        status = "tracking"
        if kick_ts is not None and now >= kick_ts:
            status = "closed"
        prev = summary[summary["play_id"].astype(str) == str(pid)]
        if len(prev) and str(prev.iloc[0].get("status") or "") == "settled":
            status = "settled"

        rows.append(
            {
                "play_id": pid,
                "match_id": first.get("match_id"),
                "system_id": first.get("system_id"),
                "system": first.get("system"),
                "league": first.get("league"),
                "market": first.get("market"),
                "side": first.get("side"),
                "home_team": first.get("home_team"),
                "away_team": first.get("away_team"),
                "match_date": first.get("match_date"),
                "kickoff_utc": kick,
                "first_seen_at": first.get("observed_at"),
                "first_tier": first.get("tier"),
                "first_pin_odds": first_odds,
                "first_edge": _f(first.get("edge_vs_pin")),
                "entry_pin_odds": entry_odds,
                "entry_at": entry_at,
                "last_seen_at": last.get("observed_at"),
                "last_pin_odds": last_odds,
                "last_edge": _f(last.get("edge_vs_pin")),
                "close_pin_odds": close_odds if status in ("closed", "settled") else pd.NA,
                "close_observed_at": close_at if status in ("closed", "settled") else pd.NA,
                "n_observations": int(len(g)),
                "clv_first_pct": odds_clv_pct(first_odds, close_odds)
                if status in ("closed", "settled")
                else pd.NA,
                "clv_entry_pct": odds_clv_pct(entry_odds, close_odds)
                if status in ("closed", "settled") and entry_odds
                else pd.NA,
                "clv_last_pct": odds_clv_pct(last_odds, close_odds)
                if status in ("closed", "settled")
                else odds_clv_pct(first_odds, last_odds),
                "steam_vs_first": steam_label(first_odds, close_odds)
                if status in ("closed", "settled")
                else steam_label(first_odds, last_odds),
                "status": status,
                "timing_note": timing_note(first_odds, entry_odds, close_odds)
                if status in ("closed", "settled")
                else timing_note(first_odds, first_odds, last_odds),
            }
        )
    return pd.DataFrame(rows)


def _prev_odds_by_play(hist: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if len(hist) == 0:
        return out
    for pid, g in hist.groupby("play_id", sort=False):
        g = g.sort_values("observed_at")
        out[str(pid)] = _f(g.iloc[-1].get("pin_odds"))
    return out


def append_scan_log(
    stamp: str,
    new_rows: list[dict],
    *,
    prev_odds: dict[str, float | None],
) -> None:
    """Append one JSONL record per scan — elite audit trail with per-play deltas."""
    moves: list[dict[str, Any]] = []
    for row in new_rows:
        pid = str(row.get("play_id") or "")
        now_odds = _f(row.get("pin_odds"))
        prev = prev_odds.get(pid)
        delta = round(now_odds - prev, 4) if now_odds is not None and prev is not None else None
        clv_vs_prev = odds_clv_pct(prev, now_odds) if prev and now_odds else None
        moves.append(
            {
                "play_id": pid,
                "tier": row.get("tier"),
                "match": f"{row.get('home_team')} vs {row.get('away_team')}",
                "system": row.get("system"),
                "pin_odds": now_odds,
                "prev_pin_odds": prev,
                "delta_odds": delta,
                "clv_vs_prev_pct": clv_vs_prev,
                "edge_vs_pin": _f(row.get("edge_vs_pin")),
                "hours_to_kick": row.get("hours_to_kick"),
            }
        )
    payload = {
        "scan_stamp": stamp,
        "observed_at": _now(),
        "kind": "live_plays",
        "n_plays": sum(1 for m in moves if m.get("tier") == "PLAY"),
        "n_watch": sum(1 for m in moves if m.get("tier") == "WATCH"),
        "n_observations": len(moves),
        "moves": moves,
    }
    SCAN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SCAN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def record_scan_observations(
    plays: pd.DataFrame | None,
    watches: pd.DataFrame | None = None,
    *,
    scan_stamp: str | None = None,
) -> int:
    """Append today's PLAYS + WATCH rows to line history. Returns rows added."""
    stamp = scan_stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kickoffs = _load_kickoffs()
    hist = load_history()
    prev_odds = _prev_odds_by_play(hist)
    new_rows: list[dict] = []

    if plays is not None and len(plays):
        for _, r in plays.iterrows():
            new_rows.append(_row_from_eval(r, tier="PLAY", kickoffs=kickoffs, stamp=stamp))
    if watches is not None and len(watches):
        for _, r in watches.iterrows():
            new_rows.append(_row_from_eval(r, tier="WATCH", kickoffs=kickoffs, stamp=stamp))

    if not new_rows:
        return 0

    append_scan_log(stamp, new_rows, prev_odds=prev_odds)
    extra = pd.DataFrame(new_rows)
    hist = pd.concat([hist, extra], ignore_index=True) if len(hist) else extra
    save_history(hist)
    save_summary(_refresh_summary_from_history(hist, load_summary()))
    return len(new_rows)


def close_past_fixtures() -> int:
    """Recompute summary so past match_dates get close lines frozen."""
    hist = load_history()
    if len(hist) == 0:
        return 0
    before = load_summary()
    save_summary(_refresh_summary_from_history(hist, before))
    after = load_summary()
    return int((after["status"].astype(str) != "tracking").sum() - (before["status"].astype(str) != "tracking").sum())


def mark_settled(play_ids: list[str]) -> int:
    """Mark summary rows settled (after live ledger settle)."""
    if not play_ids:
        return 0
    sm = load_summary()
    if len(sm) == 0:
        return 0
    n = 0
    for pid in play_ids:
        m = sm["play_id"].astype(str) == str(pid)
        if m.any():
            sm.loc[m, "status"] = "settled"
            n += int(m.sum())
    if n:
        save_summary(sm)
    return n


def backfill_from_scan_history(*, dry_run: bool = False) -> int:
    """Import archived QUALIFIED_PLAYS + NEAR_MISSES snapshots into line history."""
    if not SCAN_HISTORY.is_dir():
        return 0
    kickoffs = _load_kickoffs()
    hist = load_history()
    existing = set(zip(hist["scan_stamp"].astype(str), hist["play_id"].astype(str))) if len(hist) else set()
    added = 0
    files = sorted(SCAN_HISTORY.glob("*_QUALIFIED_PLAYS.csv"))
    for p in files:
        stamp = p.name.split("_QUALIFIED_PLAYS")[0]
        try:
            plays = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        near_path = SCAN_HISTORY / f"{stamp}_NEAR_MISSES.csv"
        watches = pd.read_csv(near_path) if near_path.is_file() else pd.DataFrame()
        # synthetic observed_at from stamp
        try:
            obs = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            obs = _now()

        def _inject(df: pd.DataFrame, tier: str) -> None:
            nonlocal added, hist
            if df is None or len(df) == 0:
                return
            for _, r in df.iterrows():
                pid = _play_id(r)
                key = (stamp, pid)
                if key in existing:
                    continue
                row = _row_from_eval(r, tier=tier, kickoffs=kickoffs, stamp=stamp)
                row["observed_at"] = obs
                hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
                existing.add(key)
                added += 1

        _inject(plays, "PLAY")
        _inject(watches, "WATCH")

    if added and not dry_run:
        save_history(hist)
        save_summary(_refresh_summary_from_history(hist, load_summary()))
    return added


def open_plays_line_table() -> pd.DataFrame:
    sm = load_summary()
    if len(sm) == 0:
        return sm
    open_ = sm[sm["status"].astype(str) == "tracking"].copy()
    if len(open_) == 0:
        return open_
    open_ = open_.sort_values("clv_last_pct", ascending=False, na_position="last")
    return open_


def closed_plays_line_table(*, limit: int = 30) -> pd.DataFrame:
    sm = load_summary()
    if len(sm) == 0:
        return sm
    done = sm[sm["status"].astype(str).isin(["closed", "settled"])].copy()
    if len(done) == 0:
        return done
    done = done.sort_values("close_observed_at", ascending=False).head(limit)
    return done


def _observation_timeline(hist: pd.DataFrame, play_id: str) -> list[str]:
    g = hist[hist["play_id"].astype(str) == str(play_id)].sort_values("observed_at")
    lines: list[str] = []
    prev: float | None = None
    for _, r in g.iterrows():
        odds = _f(r.get("pin_odds"))
        if odds is None:
            continue
        delta_s = ""
        if prev is not None:
            d = odds - prev
            delta_s = f" ({d:+.3f})"
        hrs = r.get("hours_to_kick")
        hrs_s = f" · {float(hrs):.0f}h to KO" if pd.notna(hrs) else ""
        ts = str(r.get("observed_at") or "")[:16]
        lines.append(f"  - {ts} · {odds:.3f}{delta_s}{hrs_s}")
        prev = odds
    return lines


def _system_stats(sm: pd.DataFrame) -> pd.DataFrame:
    if len(sm) == 0 or "system" not in sm.columns:
        return pd.DataFrame()
    rows = []
    for sys_name, g in sm.groupby("system", sort=False):
        closed = g[g["status"].astype(str).isin(["closed", "settled"])]
        tracking = g[g["status"].astype(str) == "tracking"]
        clv = pd.to_numeric(closed["clv_first_pct"], errors="coerce") if len(closed) else pd.Series(dtype=float)
        rows.append(
            {
                "system": sys_name,
                "open": int(len(tracking)),
                "closed": int(len(closed)),
                "mean_clv_first": float(clv.mean()) if len(clv) else None,
                "pct_toward_us": float((closed["steam_vs_first"] == "toward_us").mean())
                if len(closed)
                else None,
            }
        )
    return pd.DataFrame(rows)


def format_scan_section() -> str:
    """Short block for daily scan stdout / decision card."""
    open_ = open_plays_line_table()
    if len(open_) == 0:
        return ""
    lines = [
        "",
        "----------------------------------------------------------------",
        "  LINE MOVES (since first radar — bet timing / CLV vs now)",
        "----------------------------------------------------------------",
    ]
    for _, r in open_.head(12).iterrows():
        match = f"{r.get('home_team')} vs {r.get('away_team')}"
        first = _f(r.get("first_pin_odds"))
        last = _f(r.get("last_pin_odds"))
        clv = r.get("clv_last_pct")
        steam = r.get("steam_vs_first") or "?"
        n = int(r.get("n_observations") or 0)
        tier = str(r.get("first_tier") or "WATCH")
        clv_f = float(clv) if pd.notna(clv) else None
        action = bet_timing_action(first, last, clv_last_pct=clv_f, n_obs=n, tier=tier)
        clv_s = f"{clv_f:+.1f}%" if clv_f is not None else "—"
        first_s = f"{first:.3f}" if first else "—"
        last_s = f"{last:.3f}" if last else "—"
        lines.append(
            f"  [{action:<6}] {match[:32]:<32} {first_s}→{last_s}  CLV {clv_s}  "
            f"{steam} n={n}"
        )
    lines.append("  Full report → docs/LINE_MOVES.md · scan log → data/gameday/line_scan_log.jsonl")
    return "\n".join(lines)


def write_daily_snapshot() -> Path | None:
    """Archive today's line-move report under docs/daily_runs/."""
    if not REPORT.is_file():
        return None
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dest = DAILY_RUNS / f"{day}_LINE_MOVES.md"
    DAILY_RUNS.mkdir(parents=True, exist_ok=True)
    dest.write_text(REPORT.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def write_report() -> Path:
    """Write docs/LINE_MOVES.md from summary + history."""
    sm = load_summary()
    hist = load_history()
    lines = [
        "# Line movement — PLAYS / WATCH radar → close",
        "",
        f"Updated: {_now()}",
        "",
        "Tracks Pin prices on **every daily scan** from first radar (PLAY or WATCH) ",
        "through kickoff. **CLV%** = `(first_or_entry_odds / close_odds) - 1` on the bet side. ",
        "Positive = line steamed **toward** our pick by close (early entry rewarded).",
        "",
        "Backtests use **closing** Pin; live CLV here tells you whether to bet early or wait.",
        "",
        "**Actions:** `BET_NOW` = steam toward us · `WAIT` = line moving against · ",
        "`MONITOR` = flat/thin history · `INSUFFICIENT_DATA` = first observation only.",
        "",
        f"**Audit trail:** `{SCAN_LOG.relative_to(ROOT)}` (append-only JSONL per scan).",
        "",
    ]

    if len(sm) == 0:
        lines.append("_No line history yet — runs accumulate after each daily scan._")
    else:
        tracking = sm[sm["status"].astype(str) == "tracking"]
        closed = sm[sm["status"].astype(str).isin(["closed", "settled"])]
        lines.append(f"**Tracking:** {len(tracking)} open · **Closed/settled:** {len(closed)}")
        lines.append("")

        sys_stats = _system_stats(sm)
        if len(sys_stats):
            lines.append("## By system")
            lines.append("")
            lines.append("| System | Open | Closed | Avg CLV first | % toward us |")
            lines.append("|--------|-----:|-------:|--------------:|------------:|")
            for _, r in sys_stats.iterrows():
                avg = r.get("mean_clv_first")
                pct = r.get("pct_toward_us")
                avg_s = f"{float(avg):+.1f}%" if pd.notna(avg) else "—"
                pct_s = f"{float(pct) * 100:.0f}%" if pd.notna(pct) else "—"
                lines.append(
                    f"| {r.get('system')} | {int(r.get('open') or 0)} | "
                    f"{int(r.get('closed') or 0)} | {avg_s} | {pct_s} |"
                )
            lines.append("")

        if len(tracking):
            lines.append("## Open — CLV vs last scan (bet now or wait?)")
            lines.append("")
            lines.append("| Action | Match | System | First | Now | CLV vs now | Steam | Obs |")
            lines.append("|--------|-------|--------|------:|----:|-----------:|-------|----:|")
            for _, r in tracking.sort_values("match_date").iterrows():
                match = f"{r.get('home_team')} vs {r.get('away_team')}"
                clv = r.get("clv_last_pct")
                clv_f = float(clv) if pd.notna(clv) else None
                clv_s = f"{clv_f:+.1f}%" if clv_f is not None else "—"
                first = _f(r.get("first_pin_odds"))
                last = _f(r.get("last_pin_odds"))
                n = int(r.get("n_observations") or 0)
                tier = str(r.get("first_tier") or "WATCH")
                action = bet_timing_action(first, last, clv_last_pct=clv_f, n_obs=n, tier=tier)
                lines.append(
                    f"| **{action}** | {match} | {r.get('system')} | "
                    f"{r.get('first_pin_odds'):.3f} | {r.get('last_pin_odds'):.3f} | "
                    f"{clv_s} | {r.get('steam_vs_first')} | {n} |"
                )
            lines.append("")

            lines.append("## Open — observation timelines")
            lines.append("")
            for _, r in tracking.sort_values("match_date").iterrows():
                match = f"{r.get('home_team')} vs {r.get('away_team')}"
                lines.append(f"### {match} ({r.get('system')})")
                for tl in _observation_timeline(hist, str(r.get("play_id"))):
                    lines.append(tl)
                lines.append("")

        if len(closed):
            pos = closed[pd.to_numeric(closed["clv_first_pct"], errors="coerce") > 2]
            neg = closed[pd.to_numeric(closed["clv_first_pct"], errors="coerce") < -2]
            lines.append("## Closed — CLV vs kickoff close")
            lines.append("")
            if len(closed):
                avg = pd.to_numeric(closed["clv_first_pct"], errors="coerce").mean()
                lines.append(
                    f"Average first→close CLV: **{avg:+.1f}%** · "
                    f"toward us (>2%): **{len(pos)}** · against us (<-2%): **{len(neg)}**"
                )
            lines.append("")
            lines.append("| Match | First | Close | CLV first | CLV entry | Steam | Timing |")
            lines.append("|-------|------:|------:|----------:|----------:|-------|--------|")
            for _, r in closed.head(40).iterrows():
                match = f"{r.get('home_team')} vs {r.get('away_team')}"
                cf = r.get("clv_first_pct")
                ce = r.get("clv_entry_pct")
                cf_s = f"{float(cf):+.1f}%" if pd.notna(cf) else "—"
                ce_s = f"{float(ce):+.1f}%" if pd.notna(ce) else "—"
                lines.append(
                    f"| {match} | {r.get('first_pin_odds'):.3f} | {r.get('close_pin_odds'):.3f} | "
                    f"{cf_s} | {ce_s} | {r.get('steam_vs_first')} | {r.get('timing_note')} |"
                )
            lines.append("")

    stats_path = ROOT / "data" / "gameday" / "line_move_stats.json"
    closed_df = sm[sm["status"].astype(str).isin(["closed", "settled"])] if len(sm) else pd.DataFrame()
    tracking_df = sm[sm["status"].astype(str) == "tracking"] if len(sm) else pd.DataFrame()
    payload: dict[str, Any] = {
        "updated_at": _now(),
        "n_tracking": int(len(tracking_df)),
        "n_closed": int(len(closed_df)),
        "n_history_rows": int(len(hist)),
        "n_scan_log_entries": 0,
    }
    if SCAN_LOG.is_file():
        payload["n_scan_log_entries"] = sum(1 for _ in SCAN_LOG.open(encoding="utf-8"))
    if len(closed_df):
        payload["mean_clv_first_pct"] = float(
            pd.to_numeric(closed_df["clv_first_pct"], errors="coerce").mean()
        )
        payload["pct_toward_us"] = float((closed_df["steam_vs_first"] == "toward_us").mean())
    if len(tracking_df):
        actions = []
        for _, r in tracking_df.iterrows():
            actions.append(
                bet_timing_action(
                    _f(r.get("first_pin_odds")),
                    _f(r.get("last_pin_odds")),
                    clv_last_pct=_f(r.get("clv_last_pct")),
                    n_obs=int(r.get("n_observations") or 0),
                    tier=str(r.get("first_tier") or "WATCH"),
                )
            )
        payload["open_bet_now"] = int(sum(1 for a in actions if a == "BET_NOW"))
        payload["open_wait"] = int(sum(1 for a in actions if a == "WAIT"))
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    write_daily_snapshot()
    return REPORT
