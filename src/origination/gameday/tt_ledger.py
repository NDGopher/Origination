"""Paper ledger for Score Predictions team totals (not a live pack).

Records model-vs-Pin TT candidates each day, settles when finals are known.
Rules for the 6 protected live systems are untouched.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCAN = ROOT / "experiments" / "gameday_scan"
LEDGER = ROOT / "data" / "gameday" / "tt_ledger.csv"
ARCHIVE = SCAN / "tt_history"
RESULTS = ROOT / "data" / "gameday" / "settled_results.csv"
REPORT = ROOT / "docs" / "TT_LEDGER.md"
SUMMARY_JSON = ROOT / "data" / "gameday" / "tt_ledger_summary.json"

# Paper TRACK: team OVER only (0.5 / 1.5 / 2.5). Unders are not TRACK.
EDGE_TRACK_PP = 10.0
EDGE_CONFLICT_PP = 15.0
MIN_P_OVER = 0.55
# Prefer attacking sides / big games — week-1 Pin sample was better with cushion
MIN_PROJ_OVER_LINE = 0.3
TRACK_LINES = {0.5, 1.5, 2.5}
TRACK_LEAN = "OVER"
# Soft-exclude thin / weak early-season leagues from TRACK
TRACK_SKIP_LEAGUES = {"Championship", "MLS", "Turkey", "Scotland"}
FOCUS_ONLY = True
MAX_PER_DAY = 12

LEDGER_COLS = [
    "play_id",
    "recorded_at",
    "tier",  # TRACK | CONFLICT_WATCH
    "league",
    "match_id",
    "date",
    "kickoff_local",
    "match",
    "side",
    "team",
    "tt_line",
    "tt_lean",
    "tt_proj",
    "tt_p_over",
    "tt_p_under",
    "tt_pin_over",
    "tt_pin_under",
    "tt_edge_over_pp",
    "tt_edge_under_pp",
    "tt_lean_pp",
    "tt_vs_pin",
    "pin_odds",  # odds on the lean side
    "why",
    "data_grade",
    "stake_u",
    "status",
    "actual_home",
    "actual_away",
    "actual_team_goals",
    "won",
    "profit_u",
    "settled_at",
    "settle_note",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _play_id(row: pd.Series | dict) -> str:
    return "|".join(
        [
            str(row.get("match_id") or ""),
            str(row.get("side") or ""),
            str(row.get("tt_line") or ""),
            str(row.get("tt_lean") or ""),
        ]
    )


def load_ledger() -> pd.DataFrame:
    if not LEDGER.is_file():
        return pd.DataFrame(columns=LEDGER_COLS)
    df = pd.read_csv(LEDGER)
    for c in LEDGER_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[LEDGER_COLS]


def save_ledger(df: pd.DataFrame) -> Path:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    out = df[LEDGER_COLS].copy() if len(df) else pd.DataFrame(columns=LEDGER_COLS)
    out.to_csv(LEDGER, index=False)
    return LEDGER


def archive_tt_snapshot(tt: pd.DataFrame | None = None) -> Path | None:
    """Save dated copy of SCORE_TEAM_TOTALS for future closing-edge research."""
    src = SCAN / "SCORE_TEAM_TOTALS.csv"
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if tt is not None and len(tt):
        dest = ARCHIVE / f"{stamp}_SCORE_TEAM_TOTALS.csv"
        tt.to_csv(dest, index=False)
        return dest
    if src.is_file():
        dest = ARCHIVE / f"{stamp}_SCORE_TEAM_TOTALS.csv"
        dest.write_bytes(src.read_bytes())
        return dest
    return None


def _lean_pin_odds(r: pd.Series) -> float | None:
    lean = str(r.get("tt_lean") or "").upper()
    try:
        if lean == "OVER":
            return float(r.get("tt_pin_over"))
        if lean == "UNDER":
            return float(r.get("tt_pin_under"))
    except (TypeError, ValueError):
        return None
    return None


def _why_text(r: pd.Series, score_why: str = "") -> str:
    bits = [
        f"proj {r.get('tt_proj')} goals",
        f"line {r.get('tt_line')}",
        f"model O/U {100*float(r['tt_p_over']):.0f}/{100*float(r['tt_p_under']):.0f}%"
        if pd.notna(r.get("tt_p_over"))
        else "",
        str(r.get("tt_vs_pin") or ""),
    ]
    if score_why:
        bits.append(str(score_why)[:80])
    return " | ".join(b for b in bits if b)


def select_candidates(
    tt: pd.DataFrame,
    *,
    score_df: pd.DataFrame | None = None,
    edge_pp: float = EDGE_TRACK_PP,
    conflict_pp: float = EDGE_CONFLICT_PP,
    focus_only: bool = FOCUS_ONLY,
    max_n: int = MAX_PER_DAY,
) -> pd.DataFrame:
    """Pick paper TRACK + CONFLICT_WATCH rows (OVER 1.5 / 2.5 focus)."""
    if tt is None or len(tt) == 0:
        return pd.DataFrame()
    df = tt.copy()
    if "has_pin_tt" in df.columns:
        df = df[df["has_pin_tt"] == True]
    if focus_only and "in_focus" in df.columns:
        df = df[df["in_focus"] == True]
    if len(df) == 0:
        return df

    why_map: dict[str, str] = {}
    if score_df is not None and len(score_df) and "match_id" in score_df.columns:
        for _, s in score_df.iterrows():
            why_map[str(s.get("match_id"))] = str(s.get("why") or "")

    rows = []
    for _, r in df.iterrows():
        try:
            line = float(r.get("tt_line"))
        except (TypeError, ValueError):
            continue
        # Hard ban 0.5 (and any non-focus line)
        if line not in TRACK_LINES:
            continue
        lean = str(r.get("tt_lean") or "").upper()
        if lean != TRACK_LEAN:
            continue
        try:
            p_over = float(r.get("tt_p_over"))
        except (TypeError, ValueError):
            p_over = float("nan")
        if not np.isfinite(p_over) or p_over < MIN_P_OVER:
            continue
        try:
            proj = float(r.get("tt_proj"))
        except (TypeError, ValueError):
            proj = float("nan")
        if not np.isfinite(proj) or proj < line + MIN_PROJ_OVER_LINE:
            continue

        # Edge on the Over side vs Pin
        try:
            eo = float(r.get("tt_edge_over_pp"))
        except (TypeError, ValueError):
            eo = float(r.get("tt_lean_pp") or 0)
        if not np.isfinite(eo):
            eo = 0.0
        if eo < edge_pp:
            continue

        league = str(r.get("league") or "")
        conflict = bool(r.get("tt_pin_conflict")) or eo >= conflict_pp
        soft_skip = league in TRACK_SKIP_LEAGUES
        if soft_skip and not conflict:
            # Keep as CONFLICT_WATCH-style caution rather than TRACK
            tier = "CONFLICT_WATCH"
        else:
            tier = "CONFLICT_WATCH" if conflict else "TRACK"

        pin_odds = _lean_pin_odds(r)
        rows.append(
            {
                "tier": tier,
                "league": r.get("league"),
                "match_id": r.get("match_id"),
                "date": str(r.get("date") or "")[:10],
                "kickoff_local": r.get("kickoff_local"),
                "match": r.get("match"),
                "side": r.get("side"),
                "team": r.get("team"),
                "tt_line": line,
                "tt_lean": lean,
                "tt_proj": r.get("tt_proj"),
                "tt_p_over": r.get("tt_p_over"),
                "tt_p_under": r.get("tt_p_under"),
                "tt_pin_over": r.get("tt_pin_over"),
                "tt_pin_under": r.get("tt_pin_under"),
                "tt_edge_over_pp": eo,
                "tt_edge_under_pp": r.get("tt_edge_under_pp"),
                "tt_lean_pp": eo,
                "tt_vs_pin": r.get("tt_vs_pin") or f"O{eo:+.1f}pp",
                "pin_odds": pin_odds,
                "why": _why_text(r, why_map.get(str(r.get("match_id")), "")),
                "data_grade": r.get("data_grade"),
                "abs_edge": eo,
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out["tier_rank"] = out["tier"].map({"TRACK": 0, "CONFLICT_WATCH": 1}).fillna(2)
    out = out.sort_values(["tier_rank", "abs_edge"], ascending=[True, False]).reset_index(drop=True)
    tracks = out[out["tier"] == "TRACK"].head(max_n)
    conflicts = out[out["tier"] == "CONFLICT_WATCH"].head(max(6, max_n // 2))
    return pd.concat([tracks, conflicts], ignore_index=True)


def write_today_card(cands: pd.DataFrame) -> Path:
    """Human-readable daily TT card."""
    lines = [
        "# Team totals - today's paper card",
        "",
        f"Updated: {_now()}",
        "",
        "Paper TRACK only -- **not** one of the 6 live packs.",
        f"Focus: **team OVER** on lines {sorted(TRACK_LINES)} (Unders not TRACK).",
        f"TRACK = Over edge >={EDGE_TRACK_PP:.0f}pp, p_over>={MIN_P_OVER:.2f}, "
        f"proj≥line+{MIN_PROJ_OVER_LINE}, <{EDGE_CONFLICT_PP:.0f}pp conflict. "
        f"Soft-skip: {', '.join(sorted(TRACK_SKIP_LEAGUES))}.",
        "",
    ]
    tracks = cands[cands["tier"] == "TRACK"] if len(cands) else cands
    confs = cands[cands["tier"] == "CONFLICT_WATCH"] if len(cands) else cands
    lines += ["## TRACK (best paper TT)", ""]
    if len(tracks) == 0:
        lines.append("_None today._")
    else:
        for i, (_, r) in enumerate(tracks.iterrows(), start=1):
            lines.append(
                f"{i}. **{r.get('team')}** ({r.get('side')}) | {r.get('match')} | "
                f"**{r.get('tt_lean')} {r.get('tt_line')}** @ {r.get('pin_odds')} | "
                f"edge {r.get('tt_vs_pin')} | {r.get('kickoff_local')} | {r.get('league')}"
            )
            lines.append(f"   why: {r.get('why')}")
    lines += ["", f"## CONFLICT_WATCH (>={EDGE_CONFLICT_PP:.0f}pp - caution)", ""]
    if len(confs) == 0:
        lines.append("_None._")
    else:
        for _, r in confs.iterrows():
            lines.append(
                f"- {r.get('team')} | {r.get('tt_lean')} {r.get('tt_line')} | {r.get('tt_vs_pin')} | "
                f"{r.get('match')} ({r.get('league')})"
            )
    out = SCAN / "TT_TODAY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    cands_path = SCAN / "TT_CANDIDATES.csv"
    if len(cands):
        cands.to_csv(cands_path, index=False)
    else:
        pd.DataFrame(columns=["tier", "team", "tt_lean", "tt_line"]).to_csv(cands_path, index=False)
    return out


def record_candidates(
    cands: pd.DataFrame | None = None,
    *,
    stake_u: float = 1.0,
    track_conflicts: bool = True,
) -> int:
    """Append new TRACK (and optional CONFLICT_WATCH) rows. Returns n added."""
    if cands is None:
        p = SCAN / "TT_CANDIDATES.csv"
        if not p.is_file():
            return 0
        cands = pd.read_csv(p)
    if cands is None or len(cands) == 0:
        return 0
    if not track_conflicts:
        cands = cands[cands["tier"] == "TRACK"]
    ledger = load_ledger()
    existing = set(ledger["play_id"].astype(str)) if len(ledger) else set()
    new_rows = []
    for _, r in cands.iterrows():
        pid = _play_id(r)
        if pid in existing:
            continue
        new_rows.append(
            {
                "play_id": pid,
                "recorded_at": _now(),
                "tier": r.get("tier") or "TRACK",
                "league": r.get("league"),
                "match_id": r.get("match_id"),
                "date": str(r.get("date") or "")[:10],
                "kickoff_local": r.get("kickoff_local"),
                "match": r.get("match"),
                "side": r.get("side"),
                "team": r.get("team"),
                "tt_line": r.get("tt_line"),
                "tt_lean": r.get("tt_lean"),
                "tt_proj": r.get("tt_proj"),
                "tt_p_over": r.get("tt_p_over"),
                "tt_p_under": r.get("tt_p_under"),
                "tt_pin_over": r.get("tt_pin_over"),
                "tt_pin_under": r.get("tt_pin_under"),
                "tt_edge_over_pp": r.get("tt_edge_over_pp"),
                "tt_edge_under_pp": r.get("tt_edge_under_pp"),
                "tt_lean_pp": r.get("tt_lean_pp"),
                "tt_vs_pin": r.get("tt_vs_pin"),
                "pin_odds": r.get("pin_odds"),
                "why": r.get("why"),
                "data_grade": r.get("data_grade"),
                "stake_u": stake_u,
                "status": "open",
                "actual_home": pd.NA,
                "actual_away": pd.NA,
                "actual_team_goals": pd.NA,
                "won": pd.NA,
                "profit_u": pd.NA,
                "settled_at": pd.NA,
                "settle_note": "",
            }
        )
        existing.add(pid)
    if not new_rows:
        return 0
    extra = pd.DataFrame(new_rows)
    ledger = pd.concat([ledger, extra], ignore_index=True) if len(ledger) else extra
    save_ledger(ledger)
    return len(new_rows)


def load_results_table() -> pd.DataFrame:
    frames = []
    if RESULTS.is_file():
        frames.append(pd.read_csv(RESULTS))
    retro = ROOT / "experiments" / "weekend_retro" / "actuals.csv"
    if retro.is_file():
        frames.append(pd.read_csv(retro))
    # Also pull from aligned current season when match_id matches
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["actual_home", "actual_away"], how="any")
    return df.drop_duplicates(subset=["match_id"], keep="last")


def _settle_tt(row: pd.Series, home: float, away: float) -> dict[str, Any]:
    side = str(row.get("side") or "").lower()
    lean = str(row.get("tt_lean") or "").upper()
    try:
        line = float(row.get("tt_line"))
    except (TypeError, ValueError):
        return {"status": "open", "settle_note": "missing line"}
    goals = home if side == "home" else away
    # Half-lines: over if goals > line, under if goals < line (no push on *.5)
    if abs(line - round(line)) < 1e-9:
        # integer line — push possible
        if goals == line:
            outcome = "push"
        elif goals > line:
            outcome = "win" if lean == "OVER" else "lose"
        else:
            outcome = "win" if lean == "UNDER" else "lose"
    else:
        if goals > line:
            outcome = "win" if lean == "OVER" else "lose"
        else:
            outcome = "win" if lean == "UNDER" else "lose"

    odds = float(row["pin_odds"]) if pd.notna(row.get("pin_odds")) else None
    stake = float(row.get("stake_u") or 1.0)
    note = outcome
    if outcome == "push" or odds is None:
        profit = 0.0
        won = pd.NA if outcome == "push" else 0
        if outcome == "push":
            note = "push — stake returned"
        elif odds is None:
            note = f"{outcome} (no pin odds — hit only)"
            won = 1 if outcome == "win" else 0
            profit = 0.0
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
        "actual_team_goals": goals,
        "won": won,
        "profit_u": round(float(profit), 4),
        "settled_at": _now(),
        "settle_note": note,
    }


def settle_open() -> int:
    ledger = load_ledger()
    if len(ledger) == 0:
        return 0
    # Avoid float64 coercion on string settlement fields (pandas setitem traps).
    for c in ("status", "settled_at", "settle_note", "why", "tier", "match", "team", "tt_lean", "tt_vs_pin"):
        if c in ledger.columns:
            ledger[c] = ledger[c].astype("object")
    results = load_results_table()
    if len(results) == 0:
        # try aligned parquets by match_id for finished games
        results = _results_from_aligned(ledger)
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
        try:
            h = float(res["actual_home"])
            a = float(res["actual_away"])
        except (TypeError, ValueError, KeyError):
            continue
        upd = _settle_tt(row, h, a)
        for k, v in upd.items():
            ledger.at[i, k] = v
        n += 1
    save_ledger(ledger)
    return n


def _results_from_aligned(ledger: pd.DataFrame) -> pd.DataFrame:
    """Fallback: look up finals in league aligned files for open match_ids."""
    from origination.utils.league_registry import get_league

    open_ids = set(ledger.loc[ledger["status"].astype(str) == "open", "match_id"].astype(str))
    if not open_ids:
        return pd.DataFrame()
    rows = []
    leagues = ledger.loc[ledger["status"].astype(str) == "open", "league"].dropna().unique()
    for lg in leagues:
        try:
            info = get_league(str(lg))
        except KeyError:
            continue
        p = ROOT / "data" / "interim" / info["aligned"]
        if not p.is_file():
            continue
        try:
            df = pd.read_parquet(p, columns=["match_id", "home_goals", "away_goals"])
        except Exception:  # noqa: BLE001
            continue
        df = df[df["match_id"].astype(str).isin(open_ids)]
        df = df.dropna(subset=["home_goals", "away_goals"])
        for _, r in df.iterrows():
            rows.append(
                {
                    "match_id": r["match_id"],
                    "actual_home": float(r["home_goals"]),
                    "actual_away": float(r["away_goals"]),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["match_id"], keep="last") if rows else pd.DataFrame()


def _slice_stats(g: pd.DataFrame) -> dict[str, Any]:
    if g is None or len(g) == 0:
        return {"n": 0, "n_open": 0, "n_settled": 0, "n_decided": 0, "wins": 0, "hit": None, "units": 0.0, "roi": None}
    settled = g[g["status"] == "settled"]
    decided = settled[pd.notna(settled["won"])] if len(settled) else settled
    stake = float(pd.to_numeric(settled.get("stake_u"), errors="coerce").fillna(0).sum()) if len(settled) else 0.0
    profit = float(pd.to_numeric(settled.get("profit_u"), errors="coerce").fillna(0).sum()) if len(settled) else 0.0
    wins = int(pd.to_numeric(decided["won"], errors="coerce").fillna(0).sum()) if len(decided) else 0
    n_dec = int(len(decided))
    return {
        "n": int(len(g)),
        "n_open": int((g["status"] == "open").sum()),
        "n_settled": int(len(settled)),
        "n_decided": n_dec,
        "wins": wins,
        "hit": None if n_dec == 0 else round(wins / n_dec, 4),
        "units": round(profit, 2),
        "roi": None if stake <= 0 else round(profit / stake, 4),
    }


def write_report() -> Path:
    df = load_ledger()
    track = df[df["tier"] == "TRACK"] if len(df) and "tier" in df.columns else df
    conflict = df[df["tier"] == "CONFLICT_WATCH"] if len(df) and "tier" in df.columns else df.iloc[0:0]
    st = _slice_stats(track)
    sc = _slice_stats(conflict)
    roi_s = "n/a" if st["roi"] is None else f"{100*st['roi']:+.1f}%"
    hit_s = "n/a" if sc["hit"] is None else f"{100*sc['hit']:.0f}%"
    lines = [
        "# Team totals - paper ledger",
        "",
        f"Updated: {_now()}",
        "",
        "Not a live pack. TRACK = team **OVER** on 0.5 / 1.5 / 2.5 (Unders not TRACK).",
        f"TRACK threshold >={EDGE_TRACK_PP:.0f}pp vs Pin | p_over>={MIN_P_OVER:.2f} | "
        f"CONFLICT_WATCH >={EDGE_CONFLICT_PP:.0f}pp.",
        "",
        f"Research: [`TT_OVER_PIN_VALUE.md`](../experiments/score_predictions/TT_OVER_PIN_VALUE.md).",
        "",
        f"**TRACK:** open {st['n_open']} | settled {st['n_settled']} | "
        f"W-L {st['wins']}-{(st['n_decided']-st['wins']) if st['n_decided'] else 0} | "
        f"units {st['units']:+.2f}u | ROI {roi_s}",
        "",
        f"**CONFLICT_WATCH:** open {sc['n_open']} | settled {sc['n_settled']} | hit {hit_s}",
        "",
        "Units use **real Pin Over odds** from the day logged. Flat-2.00 historical ROIs are not used for promotion.",
        "",
        "## Open TRACK",
        "",
    ]
    open_t = track[track["status"] == "open"] if len(track) else track
    if len(open_t) == 0:
        lines.append("_None._")
    else:
        for _, r in open_t.sort_values("date").iterrows():
            lines.append(
                f"- {r.get('date')} | **{r.get('team')}** {r.get('tt_lean')} {r.get('tt_line')} "
                f"@ {r.get('pin_odds')} | {r.get('tt_vs_pin')} | {r.get('match')} ({r.get('league')})"
            )
            if r.get("why"):
                lines.append(f"  - why: {r.get('why')}")
    lines += ["", "## Settled TRACK", ""]
    settled = track[track["status"] == "settled"] if len(track) else track
    if len(settled) == 0:
        lines.append(
            "_None yet. After games finish, use Settle finished games "
            "(or `scripts/update_tt_ledger.py`) — scores come from weekend actuals / aligned results._"
        )
    else:
        for _, r in settled.sort_values("date").iterrows():
            won = r.get("won")
            mark = "P" if pd.isna(won) else ("W" if int(won) == 1 else "L")
            profit = r.get("profit_u")
            profit_s = "n/a" if pd.isna(profit) else f"{float(profit):+.3f}u"
            lines.append(
                f"- {mark} {r.get('date')} | **{r.get('team')}** {r.get('tt_lean')} {r.get('tt_line')} "
                f"goals={r.get('actual_team_goals')} | @ {r.get('pin_odds')} | {profit_s}"
            )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    (SCAN / "TT_LEDGER.md").write_text("\n".join(lines), encoding="utf-8")
    SUMMARY_JSON.write_text(
        json.dumps(
            {"updated_at": _now(), "track": st, "conflict_watch": sc, "n_total": int(len(df))},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return REPORT


def run_daily_from_score(
    score_df: pd.DataFrame,
    tt_df: pd.DataFrame,
    *,
    record: bool = True,
) -> dict[str, Any]:
    """Called after Score Predictions build: card + archive + optional ledger append."""
    archive_tt_snapshot(tt_df)
    cands = select_candidates(tt_df, score_df=score_df)
    card = write_today_card(cands)
    n_add = record_candidates(cands) if record else 0
    report = write_report()
    return {
        "n_candidates": int(len(cands)),
        "n_recorded": int(n_add),
        "card": str(card),
        "report": str(report),
    }
