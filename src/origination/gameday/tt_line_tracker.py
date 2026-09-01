"""
Track Pinnacle TT line movement for paper TRACK / CONFLICT candidates.

Mirrors play_line_tracker: append one row per score build, freeze close at kickoff,
compute CLV vs first sight / ledger entry for team-total overs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from origination.gameday.play_line_tracker import (
    DAILY_RUNS,
    _f,
    _now,
    bet_timing_action,
    odds_clv_pct,
    steam_label,
    timing_note,
)

ROOT = Path(__file__).resolve().parents[3]
HISTORY = ROOT / "data" / "gameday" / "tt_line_history.csv"
SUMMARY = ROOT / "data" / "gameday" / "tt_line_summary.csv"
LEDGER = ROOT / "data" / "gameday" / "tt_ledger.csv"
SCAN_LOG = ROOT / "data" / "gameday" / "line_scan_log.jsonl"
REPORT = ROOT / "docs" / "TT_LINE_MOVES.md"

HISTORY_COLS = [
    "observed_at",
    "scan_stamp",
    "play_id",
    "tier",
    "match_id",
    "league",
    "team",
    "side",
    "tt_line",
    "tt_lean",
    "match",
    "match_date",
    "kickoff_local",
    "pin_odds",
    "tt_edge_pp",
    "tt_p_over",
    "tt_proj",
]

SUMMARY_COLS = [
    "play_id",
    "match_id",
    "league",
    "team",
    "tt_line",
    "tt_lean",
    "match",
    "match_date",
    "first_seen_at",
    "first_tier",
    "first_pin_odds",
    "first_edge_pp",
    "entry_pin_odds",
    "entry_at",
    "last_seen_at",
    "last_pin_odds",
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


def _play_id(row: pd.Series | dict) -> str:
    if isinstance(row, dict):
        row = pd.Series(row)
    return "|".join(
        [
            str(row.get("match_id") or ""),
            str(row.get("side") or ""),
            str(row.get("tt_line") or ""),
            str(row.get("tt_lean") or ""),
        ]
    )


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


def _row_from_candidate(row: pd.Series, *, stamp: str) -> dict:
    observed = _now()
    return {
        "observed_at": observed,
        "scan_stamp": stamp,
        "play_id": _play_id(row),
        "tier": str(row.get("tier") or "TRACK"),
        "match_id": str(row.get("match_id") or ""),
        "league": row.get("league"),
        "team": row.get("team"),
        "side": row.get("side"),
        "tt_line": row.get("tt_line"),
        "tt_lean": row.get("tt_lean"),
        "match": row.get("match"),
        "match_date": str(row.get("date") or "")[:10],
        "kickoff_local": row.get("kickoff_local"),
        "pin_odds": _f(row.get("pin_odds")),
        "tt_edge_pp": _f(row.get("tt_lean_pp") or row.get("tt_edge_over_pp")),
        "tt_p_over": _f(row.get("tt_p_over")),
        "tt_proj": _f(row.get("tt_proj")),
    }


def _refresh_summary(hist: pd.DataFrame) -> pd.DataFrame:
    if len(hist) == 0:
        return pd.DataFrame(columns=SUMMARY_COLS)

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
        md = first.get("match_date")
        kick_ts = None
        if md and pd.notna(md):
            try:
                kick_ts = pd.Timestamp(str(md)).tz_localize("UTC") + pd.Timedelta(hours=20)
            except Exception:  # noqa: BLE001
                kick_ts = None

        close_row = last
        close_odds = _f(close_row.get("pin_odds"))
        close_at = str(close_row.get("observed_at") or "")

        entry_odds, entry_at = ledger_entry.get(str(pid), (None, None))
        if entry_odds is None:
            track_rows = g[g["tier"] == "TRACK"]
            if len(track_rows):
                entry_odds = _f(track_rows.iloc[0].get("pin_odds"))
                entry_at = str(track_rows.iloc[0].get("observed_at") or "")[:25]

        first_odds = _f(first.get("pin_odds"))
        last_odds = _f(last.get("pin_odds"))
        status = "tracking"
        if kick_ts is not None and now >= kick_ts:
            status = "closed"
        prev = load_summary()
        if len(prev):
            old = prev[prev["play_id"].astype(str) == str(pid)]
            if len(old) and str(old.iloc[0].get("status") or "") == "settled":
                status = "settled"

        rows.append(
            {
                "play_id": pid,
                "match_id": first.get("match_id"),
                "league": first.get("league"),
                "team": first.get("team"),
                "tt_line": first.get("tt_line"),
                "tt_lean": first.get("tt_lean"),
                "match": first.get("match"),
                "match_date": first.get("match_date"),
                "first_seen_at": first.get("observed_at"),
                "first_tier": first.get("tier"),
                "first_pin_odds": first_odds,
                "first_edge_pp": _f(first.get("tt_edge_pp")),
                "entry_pin_odds": entry_odds,
                "entry_at": entry_at,
                "last_seen_at": last.get("observed_at"),
                "last_pin_odds": last_odds,
                "close_pin_odds": close_odds if status in ("closed", "settled") else pd.NA,
                "close_observed_at": close_at if status in ("closed", "settled") else pd.NA,
                "n_observations": int(len(g)),
                "clv_first_pct": odds_clv_pct(first_odds, close_odds)
                if status in ("closed", "settled")
                else pd.NA,
                "clv_entry_pct": odds_clv_pct(entry_odds, close_odds)
                if status in ("closed", "settled") and entry_odds
                else pd.NA,
                "clv_last_pct": odds_clv_pct(first_odds, last_odds)
                if status == "tracking"
                else odds_clv_pct(last_odds, close_odds),
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


def _append_scan_log(stamp: str, new_rows: list[dict], prev_odds: dict[str, float | None]) -> None:
    moves = []
    for row in new_rows:
        pid = str(row.get("play_id") or "")
        now_odds = _f(row.get("pin_odds"))
        prev = prev_odds.get(pid)
        delta = round(now_odds - prev, 4) if now_odds is not None and prev is not None else None
        moves.append(
            {
                "play_id": pid,
                "tier": row.get("tier"),
                "team": row.get("team"),
                "tt_line": row.get("tt_line"),
                "tt_lean": row.get("tt_lean"),
                "pin_odds": now_odds,
                "prev_pin_odds": prev,
                "delta_odds": delta,
                "clv_vs_prev_pct": odds_clv_pct(prev, now_odds) if prev and now_odds else None,
                "tt_edge_pp": _f(row.get("tt_edge_pp")),
            }
        )
    payload = {
        "scan_stamp": stamp,
        "observed_at": _now(),
        "kind": "tt_overs",
        "n_track": sum(1 for m in moves if m.get("tier") == "TRACK"),
        "n_conflict": sum(1 for m in moves if m.get("tier") == "CONFLICT_WATCH"),
        "n_observations": len(moves),
        "moves": moves,
    }
    SCAN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SCAN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def record_tt_observations(
    candidates: pd.DataFrame | None,
    *,
    scan_stamp: str | None = None,
) -> int:
    """Append TT candidate rows from today's score build. Returns rows added."""
    if candidates is None or len(candidates) == 0:
        return 0
    stamp = scan_stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    hist = load_history()
    prev_odds: dict[str, float | None] = {}
    if len(hist):
        for pid, g in hist.groupby("play_id", sort=False):
            g = g.sort_values("observed_at")
            prev_odds[str(pid)] = _f(g.iloc[-1].get("pin_odds"))

    new_rows = [_row_from_candidate(r, stamp=stamp) for _, r in candidates.iterrows()]
    _append_scan_log(stamp, new_rows, prev_odds)
    extra = pd.DataFrame(new_rows)
    hist = pd.concat([hist, extra], ignore_index=True) if len(hist) else extra
    save_history(hist)
    save_summary(_refresh_summary(hist))
    return len(new_rows)


def mark_settled(play_ids: list[str]) -> int:
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


def write_report() -> Path:
    sm = load_summary()
    hist = load_history()
    lines = [
        "# TT line movement — TRACK / CONFLICT radar → close",
        "",
        f"Updated: {_now()}",
        "",
        "Tracks Pin TT prices on **every score build** from first TRACK/CONFLICT flag ",
        "through kickoff. Same CLV logic as live PLAYS — positive = steam toward our lean.",
        "",
        f"**Audit trail:** `{SCAN_LOG.relative_to(ROOT)}` (kind=`tt_overs`).",
        "",
    ]

    if len(sm) == 0:
        lines.append("_No TT line history yet — accumulates after each score build._")
    else:
        tracking = sm[sm["status"].astype(str) == "tracking"]
        closed = sm[sm["status"].astype(str).isin(["closed", "settled"])]
        lines.append(f"**Tracking:** {len(tracking)} open · **Closed/settled:** {len(closed)}")
        lines.append("")

        if len(tracking):
            lines.append("## Open TT — bet timing")
            lines.append("")
            lines.append("| Action | Team | Line | First | Now | CLV vs now | Steam | Obs |")
            lines.append("|--------|------|-----:|------:|----:|-----------:|-------|----:|")
            for _, r in tracking.sort_values("match_date").iterrows():
                clv_f = _f(r.get("clv_last_pct"))
                clv_s = f"{clv_f:+.1f}%" if clv_f is not None else "—"
                first = _f(r.get("first_pin_odds"))
                last = _f(r.get("last_pin_odds"))
                n = int(r.get("n_observations") or 0)
                tier = str(r.get("first_tier") or "TRACK")
                action = bet_timing_action(first, last, clv_last_pct=clv_f, n_obs=n, tier=tier)
                first_s = f"{first:.3f}" if first else "—"
                last_s = f"{last:.3f}" if last else "—"
                lines.append(
                    f"| **{action}** | {r.get('team')} {r.get('tt_lean')} {r.get('tt_line')} | "
                    f"{r.get('tt_line')} | {first_s} | {last_s} | {clv_s} | "
                    f"{r.get('steam_vs_first')} | {n} |"
                )
            lines.append("")

        if len(closed):
            avg = pd.to_numeric(closed["clv_first_pct"], errors="coerce").mean()
            lines.append("## Closed TT — CLV vs kickoff")
            lines.append("")
            lines.append(f"Average first→close CLV: **{avg:+.1f}%**")
            lines.append("")
            lines.append("| Team | First | Close | CLV first | CLV entry | Steam | Note |")
            lines.append("|------|------:|------:|----------:|----------:|-------|------|")
            for _, r in closed.head(30).iterrows():
                cf = r.get("clv_first_pct")
                ce = r.get("clv_entry_pct")
                cf_s = f"{float(cf):+.1f}%" if pd.notna(cf) else "—"
                ce_s = f"{float(ce):+.1f}%" if pd.notna(ce) else "—"
                label = f"{r.get('team')} {r.get('tt_lean')} {r.get('tt_line')}"
                lines.append(
                    f"| {label} | {r.get('first_pin_odds'):.3f} | {r.get('close_pin_odds'):.3f} | "
                    f"{cf_s} | {ce_s} | {r.get('steam_vs_first')} | {r.get('timing_note')} |"
                )
            lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snap = DAILY_RUNS / f"{day}_TT_LINE_MOVES.md"
    DAILY_RUNS.mkdir(parents=True, exist_ok=True)
    snap.write_text("\n".join(lines), encoding="utf-8")
    return REPORT
