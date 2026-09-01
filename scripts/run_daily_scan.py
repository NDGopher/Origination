#!/usr/bin/env python
"""
Daily scan — evaluate all protected + paper systems and emit idiot-proof play cards.

Prefer the UI (Launch_Gameday.bat): Update Data → Update Odds → Run Scan.
CLI: use --no-refresh after those steps, or --refresh to pull fixtures+odds inside the scan.

Protected (rules frozen):
  1–5 EPL Unders/Overs, Bundesliga Unders, La Liga Home ML, Serie A Away ML
  6   Primeira Liga AH short (paper)
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.utils.odds import (
    decimal_to_american,
    fair_decimal_odds,
    model_edge_vs_odds,
    model_edge_vs_two_way,
)
from origination.utils.system_registry import (
    PAPER_SYSTEMS,
    PROTECTED_SYSTEMS,
    history_summary,
    live_systems,
)

OUT = ROOT / "experiments" / "gameday_scan"
OUT.mkdir(parents=True, exist_ok=True)
USER_ODDS = ROOT / "data" / "gameday" / "odds.csv"
FRESHNESS = OUT / "data_freshness.json"

LIVE_LEAGUES = ["EPL", "Bundesliga", "LaLiga", "SerieA", "PrimeiraLiga"]


def _py() -> Path:
    win = ROOT / ".venv" / "Scripts" / "python.exe"
    unix = ROOT / ".venv" / "bin" / "python"
    if win.exists():
        return win
    if unix.exists():
        return unix
    return Path(sys.executable)


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _amer(dec: float | None) -> int | None:
    if dec is None:
        return None
    return decimal_to_american(dec)


def run_league_sheet(league: str, *, refresh: bool = True) -> Path:
    out = ROOT / "data" / "processed" / f"gameday_sheet_{league}.csv"
    if league == "EPL":
        out = ROOT / "data" / "processed" / "gameday_sheet.csv"
    cmd = [
        str(_py()),
        str(ROOT / "scripts" / "run_gameday_sheet.py"),
        "--league",
        league,
        "--fast",
        "--out",
        str(out),
        "--log-level",
        "WARNING",
    ]
    if refresh:
        cmd += ["--refresh-fixtures", "--refresh-odds"]
    if USER_ODDS.is_file():
        cmd += ["--odds-file", str(USER_ODDS)]
    print(f"\n=== {league} (refresh={refresh}) ===", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return out


def check_freshness(sheets: dict[str, pd.DataFrame]) -> dict:
    """Basic pipeline health for protected leagues."""
    report: dict = {"checked_at": datetime.now(timezone.utc).isoformat(), "leagues": {}}
    ok_all = True
    for lg in LIVE_LEAGUES:
        info: dict = {"sheet_rows": 0, "issues": []}
        df = sheets.get(lg)
        fx = ROOT / "data" / "interim" / (
            "fixtures_upcoming_EPL.csv" if lg == "EPL" else f"fixtures_upcoming_{lg}.csv"
        )
        pin_meta = ROOT / "data" / "interim" / (
            "pinnacle_ou25_EPL.meta.json"
            if lg == "EPL"
            else f"pinnacle_ou25_{lg}.meta.json"
        )
        if not fx.is_file():
            info["issues"].append("missing fixtures")
            ok_all = False
        if not pin_meta.is_file():
            info["issues"].append("missing pinnacle meta")
            ok_all = False
        else:
            try:
                meta = json.loads(pin_meta.read_text(encoding="utf-8"))
                info["pinnacle_fetched_at"] = meta.get("fetched_at")
                info["n_with_ou25"] = meta.get("n_with_ou25")
                info["n_with_1x2"] = meta.get("n_with_1x2")
                info["n_with_ah"] = meta.get("n_with_ah")
            except Exception as exc:  # noqa: BLE001
                info["issues"].append(f"bad pin meta: {exc}")
                ok_all = False
        if df is None or len(df) == 0:
            info["issues"].append("empty sheet")
            ok_all = False
        else:
            info["sheet_rows"] = int(len(df))
            if "odds_status" in df.columns:
                miss = int((df["odds_status"] == "MISSING").sum())
                if miss == len(df):
                    info["issues"].append("all fixtures missing odds")
                    ok_all = False
                info["missing_odds"] = miss
        info["ok"] = len(info["issues"]) == 0
        if not info["ok"]:
            ok_all = False
        report["leagues"][lg] = info
    report["ok"] = ok_all
    FRESHNESS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_system_row(sys_: dict, row: pd.Series) -> list[dict]:
    """Return 0–2 eval dicts (AH can fire on either side)."""
    market = sys_["market"]
    results = []

    if market == "AH":
        for side, odds_c, edge_c, prob_c, fair_c, book_c, book_e, amer_c in [
            (
                "ah_home",
                "pin_ahh",
                "edge_ah_home",
                "p_ah_home",
                "fair_odds_ah_home",
                "book_ahh",
                "edge_ah_home_vs_book",
                "pin_ahh_american",
            ),
            (
                "ah_away",
                "pin_aha",
                "edge_ah_away",
                "p_ah_away",
                "fair_odds_ah_away",
                "book_aha",
                "edge_ah_away_vs_book",
                "pin_aha_american",
            ),
        ]:
            ev = _eval_side(sys_, row, side, odds_c, edge_c, prob_c, fair_c, book_c, book_e, amer_c)
            if ev is not None:
                results.append(ev)
        return results

    ev = _eval_side(
        sys_,
        row,
        sys_["side"],
        sys_["odds_col"],
        sys_["edge_col"],
        sys_["prob_col"],
        sys_.get("fair_odds_col") or "",
        sys_.get("book_odds_col") or "",
        sys_.get("book_edge_col") or "",
        sys_.get("american_col") or "",
    )
    return [ev] if ev is not None else []


def _eval_side(
    sys_: dict,
    row: pd.Series,
    side: str,
    odds_col: str,
    edge_col: str,
    prob_col: str,
    fair_col: str,
    book_col: str,
    book_edge_col: str,
    amer_col: str,
) -> dict | None:
    odds = _f(row.get(odds_col))
    if odds is None and odds_col == "odds_1x2_h":
        odds = _f(row.get("pin_h"))
    if odds is None and odds_col == "odds_1x2_a":
        odds = _f(row.get("pin_a"))

    edge = _f(row.get(edge_col))
    prob = _f(row.get(prob_col))
    if edge is None and odds is not None and prob is not None:
        if sys_["market"].startswith("OU"):
            o = _f(row.get("pin_over25"))
            u = _f(row.get("pin_under25"))
            if o and u:
                edge = model_edge_vs_two_way(prob, o, u, side=sys_["side"], method="power")
            else:
                edge = model_edge_vs_odds(prob, odds)
        else:
            edge = model_edge_vs_odds(prob, odds)

    thr = float(sys_["edge_thr"])
    lo, hi = float(sys_["min_odds"]), float(sys_["max_odds"])
    in_band = odds is not None and lo <= odds <= hi
    edge_ok = edge is not None and edge >= thr
    qualifies = bool(in_band and edge_ok)

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

    if not qualifies and not near:
        return None

    fair = _f(row.get(fair_col)) if fair_col else None
    if fair is None and prob is not None:
        fair = fair_decimal_odds(prob)

    book = _f(row.get(book_col)) if book_col else None
    edge_book = _f(row.get(book_edge_col)) if book_edge_col else None
    if edge_book is None and book is not None and prob is not None:
        edge_book = model_edge_vs_odds(prob, book)

    # User book still +EV vs model even if worse than Pin
    book_meets_thr = edge_book is not None and edge_book >= thr
    book_plus_ev = edge_book is not None and edge_book > 0
    if book is None:
        book_verdict = "NO BOOK"
    elif book_meets_thr:
        book_verdict = "BOOK PLAY"
    elif book_plus_ev:
        book_verdict = "BOOK +EV (below thr)"
    else:
        book_verdict = "BOOK SKIP"

    pin_am = None
    if amer_col:
        try:
            pin_am = int(float(row.get(amer_col))) if pd.notna(row.get(amer_col)) else None
        except (TypeError, ValueError):
            pin_am = None
    if pin_am is None:
        pin_am = _amer(odds)
    fair_am = _amer(fair)
    book_am = _amer(book)

    hist = sys_.get("history") or {}
    return {
        "recommendation": "PLAY" if qualifies else "WATCH",
        "system": sys_["name"],
        "system_id": sys_["id"],
        "system_status": sys_.get("status"),
        "rules": sys_.get("rules_text"),
        "history_summary": history_summary(sys_),
        "hist_n": hist.get("n"),
        "hist_roi": hist.get("roi"),
        "hist_seasons_pos": hist.get("seasons_pos"),
        "hist_seasons_n": hist.get("seasons_n"),
        "hist_max_dd_u": hist.get("max_dd_u"),
        "league": sys_["league"],
        "market": sys_["market"],
        "side": side,
        "date": str(row.get("date", ""))[:10],
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "match_id": row.get("match_id"),
        "proj_home": _f(row.get("proj_home_goals")),
        "proj_away": _f(row.get("proj_away_goals")),
        "proj_total": _f(row.get("proj_total_goals")),
        "ah_line": _f(row.get("ah_line")),
        "model_prob": prob,
        "fair_odds": fair,
        "fair_american": fair_am,
        "pin_odds": odds,
        "pin_american": pin_am,
        "edge_vs_pin": edge,
        "edge_thr": thr,
        "odds_band": f"{lo:.2f}–{hi:.2f}",
        "book_odds": book,
        "book_american": book_am,
        "edge_vs_book": edge_book,
        "book_verdict": book_verdict,
        "book_meets_system_thr": book_meets_thr,
        "book_still_plus_ev": book_plus_ev,
        "near_miss": near and not qualifies,
        "near_reason": near_reason,
        "qualifies": qualifies,
    }


def _fmt_amer(a):
    if a is None:
        return "—"
    return f"{a:+d}" if a > 0 else str(a)


def _side_label(r):
    side = str(r.get("side") or "")
    market = str(r.get("market") or "")
    ah = r.get("ah_line")
    if market.startswith("OU") or side in ("under", "over"):
        return f"{side.upper()} 2.5"
    if market.startswith("1X2") or side in ("H", "A", "D"):
        return {"H": "HOME ML", "A": "AWAY ML", "D": "DRAW ML"}.get(side, side)
    if "ah" in side.lower() or market == "AH":
        which = "HOME" if "home" in side.lower() else "AWAY"
        try:
            line = float(ah)
            return f"AH {which} {line:+.2f}"
        except (TypeError, ValueError):
            return f"AH {which}"
    return side


def _odds_amer(dec, am) -> str:
    try:
        d = f"{float(dec):.3f}"
    except (TypeError, ValueError):
        d = "?"
    try:
        a = int(am)
        a_s = f"{a:+d}" if a > 0 else str(a)
    except (TypeError, ValueError):
        a_s = "—"
    return f"{d}  ({a_s})"


def write_simple_plays(plays, nears):
    lines = []
    lines.append("=" * 64)
    lines.append("  WHAT TO BET TODAY")
    lines.append("=" * 64)
    lines.append("")
    if plays is None or len(plays) == 0:
        lines.append("  NO PLAYS right now.")
        lines.append("  (Nothing cleared the system filters on this slate.)")
        lines.append("  ACTION:  DO NOTHING")
        lines.append("")
    else:
        ordered = plays
        if "date" in plays.columns and "league" in plays.columns:
            ordered = plays.sort_values(["date", "league"])
        for i, (_, r) in enumerate(ordered.iterrows(), 1):
            bet = _side_label(r)
            pin, fair, book = r.get("pin_odds"), r.get("fair_odds"), r.get("book_odds")
            pin_am, fair_am, book_am = r.get("pin_american"), r.get("fair_american"), r.get("book_american")

            lines.append(f"  >>> PLAY #{i} <<<")
            lines.append(f"  BET:     {r.get('system')}  →  {bet}")
            lines.append(
                f"  MATCH:   {r.get('home_team')} vs {r.get('away_team')}  ({str(r.get('date',''))[:10]})"
            )
            lines.append(f"  PIN:     {_odds_amer(pin, pin_am)}")
            lines.append(f"  FAIR:    {_odds_amer(fair, fair_am)}")
            if book is not None and str(book) not in ("", "nan"):
                lines.append(f"  MY BOOK: {_odds_amer(book, book_am)}  →  {r.get('book_verdict')}")
            else:
                lines.append("  MY BOOK: (not entered — optional)")
            try:
                lines.append(
                    f"  EDGE:    {100*float(r.get('edge_vs_pin')):+.1f}% vs Pin "
                    f"(need >={100*float(r.get('edge_thr')):.0f}%)"
                )
            except (TypeError, ValueError):
                pass
            lines.append(f"  TRACK:   {r.get('history_summary')}")
            lines.append("  ACTION:  PLACE THIS BET")
            lines.append("")
    if nears is not None and len(nears):
        lines.append("-" * 64)
        lines.append("  WATCH LIST (close, but not quite — usually SKIP)")
        lines.append("-" * 64)
        for _, r in nears.iterrows():
            lines.append(
                f"  WATCH · {r.get('system')} · {r.get('home_team')} vs {r.get('away_team')} · "
                f"{_side_label(r)} · {r.get('near_reason')}"
            )
            lines.append("  ACTION:  DO NOT BET (watch only)")
        lines.append("")
    lines.append("=" * 64)
    path = OUT / "PLAYS_SIMPLE.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_decision_report(plays, nears, fresh):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# What to bet — {ts}",
        "",
        f"**Data freshness:** {'OK' if fresh.get('ok') else 'CHECK WARNINGS — refresh Data / Odds'}",
        "",
        f"## PLAYS: **{len(plays)}**  (do these)",
        "",
    ]
    if len(plays) == 0:
        lines.append("_No qualified plays. Do nothing, or check WATCH list._")
    else:
        ordered = plays
        if "date" in plays.columns and "league" in plays.columns:
            ordered = plays.sort_values(["date", "league"])
        for i, (_, r) in enumerate(ordered.iterrows(), 1):
            bet = _side_label(r)
            lines += [
                f"### PLAY #{i} — {r.get('home_team')} vs {r.get('away_team')}",
                "",
                "| | |",
                "|--|--|",
                "| **Recommendation** | **PLAY** |",
                f"| **System** | {r.get('system')} |",
                f"| **Bet exactly** | **{bet}** |",
                f"| **Date** | {str(r.get('date',''))[:10]} |",
                f"| **Pinnacle** | {r.get('pin_odds')} ({r.get('pin_american')}) |",
                f"| **Model fair** | {r.get('fair_odds')} ({r.get('fair_american')}) |",
                f"| **Edge vs Pin** | {100*(r.get('edge_vs_pin') or 0):+.1f}% (need >={100*(r.get('edge_thr') or 0):.0f}%) |",
                f"| **Your book** | {r.get('book_odds') or '—'} → **{r.get('book_verdict')}** |",
                f"| **Track record** | {r.get('history_summary')} |",
                "",
                "**→ Place this bet.**",
                "",
            ]
    lines += ["", f"## WATCH: **{len(nears)}**  (usually skip)", ""]
    if len(nears) == 0:
        lines.append("_None._")
    else:
        for _, r in nears.iterrows():
            lines.append(
                f"- **WATCH** — {r.get('system')}: {r.get('home_team')} vs {r.get('away_team')} · "
                f"{_side_label(r)} · {r.get('near_reason')}"
            )
    lines += ["", "## Freshness", ""]
    for lg, info in (fresh.get("leagues") or {}).items():
        status = "OK" if info.get("ok") else "ISSUE"
        issues = ", ".join(info.get("issues") or []) or "—"
        lines.append(f"- **{lg}** [{status}]: {issues}")
    path = OUT / "DECISION_CARD.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run protected-system gameday scan")
    ap.add_argument(
        "--no-refresh",
        action="store_true",
        help="Use fixtures/odds already on disk (UI Scan button)",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh fixtures+odds inside the scan",
    )
    args = ap.parse_args()
    do_refresh = True
    if args.no_refresh:
        do_refresh = False
    if args.refresh:
        do_refresh = True

    print(
        f"Scan pipeline — leagues={', '.join(LIVE_LEAGUES)}  refresh={do_refresh}",
        flush=True,
    )
    sheets = {}
    for lg in LIVE_LEAGUES:
        try:
            path = run_league_sheet(lg, refresh=do_refresh)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {lg}: {exc}", flush=True)
            continue
        if path.exists():
            sheets[lg] = pd.read_csv(path)
            print(f"  sheet rows={len(sheets[lg])}", flush=True)

    fresh = check_freshness(sheets)
    print(f"Freshness ok={fresh.get('ok')}", flush=True)

    plays = []
    nears = []
    all_eval = []

    for sys_ in live_systems():
        df = sheets.get(sys_["league"])
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            for ev in evaluate_system_row(sys_, row):
                all_eval.append(ev)
                if ev["qualifies"]:
                    plays.append(ev)
                elif ev.get("near_miss"):
                    nears.append(ev)

    plays_df = pd.DataFrame(plays)
    nears_df = pd.DataFrame(nears)
    all_df = pd.DataFrame(all_eval)

    plays_df.to_csv(OUT / "QUALIFIED_PLAYS.csv", index=False)
    plays_df.to_csv(OUT / "PLAYS_DECISION.csv", index=False)
    nears_df.to_csv(OUT / "NEAR_MISSES.csv", index=False)
    all_df.to_csv(OUT / "all_system_evals.csv", index=False)

    simple = write_simple_plays(plays_df, nears_df)
    report = write_decision_report(plays_df, nears_df, fresh)
    print("\n" + simple.read_text(encoding="utf-8"))
    print(f"\nWrote {simple}", flush=True)
    print(f"Wrote {report}", flush=True)
    print(f"PLAYS={len(plays_df)}  WATCH={len(nears_df)}", flush=True)

    try:
        from origination.gameday.live_ledger import record_from_scan, settle_open, write_report
        from origination.gameday.play_line_tracker import (
            close_past_fixtures,
            format_scan_section,
            mark_settled,
            record_scan_observations,
            write_report as write_line_report,
        )

        n_line = record_scan_observations(
            plays_df if len(plays_df) else None,
            nears_df if len(nears_df) else None,
        )
        n_add = record_from_scan(plays_df if len(plays_df) else None)
        n_set = settle_open()
        if n_set:
            settled = pd.read_csv(ROOT / "data" / "gameday" / "live_ledger.csv")
            settled = settled[settled["status"].astype(str) == "settled"]
            recent = settled.sort_values("settled_at").tail(n_set)
            mark_settled(recent["play_id"].astype(str).tolist())
        close_past_fixtures()
        line_path = write_line_report()
        ledger_path = write_report()
        line_section = format_scan_section()
        if line_section:
            print(line_section, flush=True)
        print(
            f"Line tracker +{n_line} obs → {line_path}",
            flush=True,
        )
        print(f"Ledger +{n_add} recorded, {n_set} settled → {ledger_path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING live ledger update skipped: {exc}", flush=True)


if __name__ == "__main__":
    main()
