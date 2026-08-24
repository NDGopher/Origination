#!/usr/bin/env python
"""
Week-1 gauntlet: score Aug-21 predictions + settle live/TT paper ledgers.

Does NOT change the 6 protected live pack rules.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.utils.league_registry import LEAGUES, list_league_keys  # noqa: E402

OUT = ROOT / "experiments" / "week1_gauntlet"
PRED = ROOT / "experiments" / "gameday_scan" / "SCORE_PREDICTIONS_20260821.csv"
TT_SNAP = ROOT / "experiments" / "gameday_scan" / "SCORE_TEAM_TOTALS_20260821.csv"
SETTLED = ROOT / "data" / "gameday" / "settled_results.csv"
WEEKEND_ACT = ROOT / "experiments" / "weekend_retro" / "actuals.csv"

# Predictions were made Fri 21 Aug for focus through ~Sun 23
DATE_MIN = "2026-08-15"
DATE_MAX = "2026-08-24"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(s: str) -> str:
    import re
    import unicodedata

    t = unicodedata.normalize("NFKD", str(s or ""))
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


def collect_actuals() -> pd.DataFrame:
    """Pull finished matches from aligned tables + prior weekend actuals."""
    rows: list[dict] = []
    for key in list_league_keys():
        info = LEAGUES[key]
        path = ROOT / "data" / "interim" / info["aligned"]
        if not path.is_file():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {key}: {exc}", flush=True)
            continue
        need = {"match_id", "date", "home_team", "away_team", "home_goals", "away_goals"}
        if not need.issubset(set(df.columns)):
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        mask = (d["date"] >= DATE_MIN) & (d["date"] <= DATE_MAX)
        d = d.loc[mask].dropna(subset=["home_goals", "away_goals"])
        src = d["result_source"].astype(str) if "result_source" in d.columns else "aligned"
        for _, r in d.iterrows():
            rows.append(
                {
                    "match_id": str(r["match_id"]),
                    "league": key,
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "actual_home": float(r["home_goals"]),
                    "actual_away": float(r["away_goals"]),
                    "date": str(r["date"].date()),
                    "source": src if isinstance(src, str) else str(r.get("result_source", "aligned")),
                    "notes": "",
                }
            )
        print(f"  {key}: {len(d)} results in window", flush=True)

    if WEEKEND_ACT.is_file():
        prior = pd.read_csv(WEEKEND_ACT)
        for _, r in prior.iterrows():
            rows.append(
                {
                    "match_id": str(r["match_id"]),
                    "league": r.get("league"),
                    "home_team": r.get("home_team"),
                    "away_team": r.get("away_team"),
                    "actual_home": float(r["actual_home"]),
                    "actual_away": float(r["actual_away"]),
                    "date": str(r.get("date") or "")[:10],
                    "source": r.get("source") or "weekend_retro",
                    "notes": r.get("notes") or "",
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["match_id"], keep="last")
    return out.sort_values(["date", "league", "match_id"]).reset_index(drop=True)


def _fuzzy_match_ids(pred: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    """Attach actuals; fall back to date+team fuzzy when match_id differs."""
    m = pred.merge(act, on="match_id", how="left", suffixes=("", "_act"))
    missing = m["actual_home"].isna()
    if not missing.any():
        return m

    act2 = act.copy()
    act2["_hk"] = act2["home_team"].map(_norm)
    act2["_ak"] = act2["away_team"].map(_norm)
    act2["_dk"] = act2["date"].astype(str).str[:10]
    lookup = {
        (r["_dk"], r["_hk"], r["_ak"]): r
        for _, r in act2.iterrows()
    }
    # also reverse date window ±1d is overkill; try without date
    lookup_teams = {}
    for _, r in act2.iterrows():
        lookup_teams.setdefault((r["_hk"], r["_ak"]), r)

    filled = 0
    for i in m.index[missing]:
        row = m.loc[i]
        hk = _norm(row.get("home_team") or str(row.get("match", "")).split(" vs ")[0])
        ak = _norm(row.get("away_team") or (str(row.get("match", "")).split(" vs ")[-1] if " vs " in str(row.get("match", "")) else ""))
        dk = str(row.get("date") or "")[:10]
        hit = lookup.get((dk, hk, ak)) or lookup_teams.get((hk, ak))
        if hit is None:
            continue
        m.at[i, "actual_home"] = hit["actual_home"]
        m.at[i, "actual_away"] = hit["actual_away"]
        m.at[i, "source"] = hit.get("source")
        if pd.isna(m.at[i, "match_id"]) or not str(m.at[i, "match_id"]):
            m.at[i, "match_id"] = hit["match_id"]
        filled += 1
    print(f"  fuzzy-filled {filled} prediction rows", flush=True)
    return m


def analyze_scores(pred: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    m = _fuzzy_match_ids(pred, act)
    m = m.dropna(subset=["actual_home", "actual_away"]).copy()
    if len(m) == 0:
        return m
    m["actual_total"] = m["actual_home"] + m["actual_away"]
    m["actual_ou"] = m["actual_total"].map(lambda t: "OVER" if t > 2.5 else "UNDER")
    m["lean_hit"] = m["lean"].astype(str).str.upper() == m["actual_ou"]
    m["proj_total"] = pd.to_numeric(m.get("proj_total"), errors="coerce")
    m["total_error"] = m["actual_total"] - m["proj_total"]
    m["abs_total_error"] = m["total_error"].abs()
    pin_over = pd.to_numeric(m.get("pin_over_pct"), errors="coerce")
    m["pin_lean"] = pin_over.map(
        lambda p: "OVER" if pd.notna(p) and p >= 50 else ("UNDER" if pd.notna(p) else None)
    )
    m["pin_hit"] = m["pin_lean"] == m["actual_ou"]
    over_gap = pd.to_numeric(m.get("model_minus_pin_over_pp"), errors="coerce").abs()
    under_gap = pd.to_numeric(m.get("model_minus_pin_under_pp"), errors="coerce").abs()
    gap = pd.concat([over_gap, under_gap], axis=1).max(axis=1)
    flagged = m["pin_conflict"].fillna(False).astype(bool) if "pin_conflict" in m.columns else False
    m["conflict_15pp"] = flagged | (gap.fillna(0) >= 15)
    m["score_profile"] = m.get("score_profile", pd.Series([""] * len(m))).astype(str).str.upper()
    return m


def analyze_tt(tt: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    if tt is None or len(tt) == 0:
        return pd.DataFrame()
    a = act.copy()
    a["_hk"] = a["home_team"].map(_norm)
    a["_ak"] = a["away_team"].map(_norm)
    by_id = {str(r["match_id"]): r for _, r in a.iterrows()}
    by_teams = {(r["_hk"], r["_ak"]): r for _, r in a.iterrows()}

    rows = []
    for _, r in tt.iterrows():
        mid = str(r.get("match_id") or "")
        res = by_id.get(mid)
        if res is None:
            match = str(r.get("match") or "")
            if " vs " in match:
                h, aw = match.split(" vs ", 1)
                res = by_teams.get((_norm(h), _norm(aw)))
        if res is None:
            continue
        side = str(r.get("side") or "").lower()
        goals = float(res["actual_home"] if side == "home" else res["actual_away"])
        try:
            line = float(r.get("tt_line"))
        except (TypeError, ValueError):
            continue
        lean = str(r.get("tt_lean") or "").upper()
        if abs(line - round(line)) < 1e-9:
            if goals == line:
                outcome = "push"
            elif goals > line:
                outcome = "win" if lean == "OVER" else "lose"
            else:
                outcome = "win" if lean == "UNDER" else "lose"
        else:
            outcome = (
                "win"
                if ((goals > line and lean == "OVER") or (goals < line and lean == "UNDER"))
                else "lose"
            )
        # paper profit at Pin lean odds
        pin_o = r.get("tt_pin_over")
        pin_u = r.get("tt_pin_under")
        try:
            odds = float(pin_o if lean == "OVER" else pin_u)
        except (TypeError, ValueError):
            odds = None
        if outcome == "push" or odds is None:
            profit = 0.0
            won = None if outcome == "push" else (1 if outcome == "win" else 0)
        elif outcome == "win":
            profit = odds - 1.0
            won = 1
        else:
            profit = -1.0
            won = 0
        rows.append(
            {
                **{
                    k: r.get(k)
                    for k in (
                        "when",
                        "league",
                        "match",
                        "match_id",
                        "side",
                        "team",
                        "tt_line",
                        "tt_lean",
                        "tt_proj",
                        "tt_lean_pp",
                        "tt_vs_pin",
                        "tt_pin_over",
                        "tt_pin_under",
                        "has_pin_tt",
                        "tt_pin_conflict",
                    )
                },
                "actual_team_goals": goals,
                "actual_home": float(res["actual_home"]),
                "actual_away": float(res["actual_away"]),
                "outcome": outcome,
                "won": won,
                "profit_u": round(profit, 4),
                "pin_odds": odds,
            }
        )
    return pd.DataFrame(rows)


def write_settled(act: pd.DataFrame) -> Path:
    SETTLED.parent.mkdir(parents=True, exist_ok=True)
    # merge with any existing
    frames = [act[["match_id", "actual_home", "actual_away", "league", "home_team", "away_team", "notes"]].copy()]
    if SETTLED.is_file():
        frames.insert(0, pd.read_csv(SETTLED))
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["match_id"], keep="last")
    out.to_csv(SETTLED, index=False)
    # also append to weekend_retro actuals for ledger fallback
    WEEKEND_ACT.parent.mkdir(parents=True, exist_ok=True)
    wa = act.copy()
    if WEEKEND_ACT.is_file():
        prior = pd.read_csv(WEEKEND_ACT)
        wa = pd.concat([prior, wa], ignore_index=True).drop_duplicates(subset=["match_id"], keep="last")
    wa.to_csv(WEEKEND_ACT, index=False)
    return SETTLED


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100 * x:.0f}%"


def _hit_rate(s: pd.Series) -> float | None:
    s = s.dropna()
    if len(s) == 0:
        return None
    return float(s.astype(bool).mean())


def build_report(
    act: pd.DataFrame,
    scores: pd.DataFrame,
    tt: pd.DataFrame,
    live_summary: dict,
    tt_summary: dict,
    stamp: dict,
) -> str:
    lines = [
        "# Week 1 gauntlet — Score Predictions, TT paper, live ledger",
        "",
        f"Updated: {_now()}",
        "",
        "Protected live pack **rules unchanged**. This is a post-week-1 performance review.",
        "",
        f"Window: **{DATE_MIN} → {DATE_MAX}**. Predictions snap: `SCORE_PREDICTIONS_20260821.csv`.",
        "",
        "## 1. Inputs / freshness",
        "",
    ]
    model = (stamp or {}).get("model") or {}
    cur = (model.get("current_season") or {}) if isinstance(model, dict) else {}
    if cur:
        lines.append("| League | n current | latest | FD | extra | xG |")
        lines.append("|--------|----------:|--------|---:|------:|---:|")
        for lg, info in sorted(cur.items()):
            if not isinstance(info, dict):
                continue
            lines.append(
                f"| {lg} | {info.get('n', 0)} | {info.get('latest') or '—'} | "
                f"{info.get('n_football_data', 0)} | {info.get('n_extra', 0)} | {info.get('n_xg', 0)} |"
            )
    else:
        lines.append("_No model stamp yet — run Full Model Refresh._")
    lines += [
        "",
        f"Aligned results in window: **{len(act)}** matches written to settled_results + weekend actuals.",
        "",
        "## 2. Score Predictions (match O/U lean)",
        "",
    ]
    if len(scores) == 0:
        lines.append("_No joined prediction/actual rows. Check match_id alignment after refresh._")
    else:
        n = len(scores)
        mh = _hit_rate(scores["lean_hit"])
        pin_known = scores.dropna(subset=["pin_lean"])
        ph = _hit_rate(pin_known["pin_hit"]) if len(pin_known) else None
        mae = float(scores["abs_total_error"].mean())
        bias = float(scores["total_error"].mean())
        lines += [
            f"- Joined **{n}** fixtures from the Aug 21 snap that have finals.",
            f"- Model O/U lean hit: **{_pct(mh)}** ({int(scores['lean_hit'].sum())}/{n})",
            f"- Pin O/U lean hit: **{_pct(ph)}**"
            + (f" ({int(pin_known['pin_hit'].sum())}/{len(pin_known)})" if len(pin_known) else ""),
            f"- Total goals MAE: **{mae:.2f}** · bias (actual−proj): **{bias:+.2f}**",
            "",
        ]
        # CONFLICT
        if "conflict_15pp" in scores.columns:
            c = scores[scores["conflict_15pp"] == True]
            a = scores[scores["conflict_15pp"] != True]
            if len(c):
                lines.append(
                    f"- CONFLICT ≥15pp vs Pin: model hit {_pct(_hit_rate(c['lean_hit']))} "
                    f"({int(c['lean_hit'].sum())}/{len(c)}) — keep flagging, do not bet from Score tab"
                )
            if len(a):
                lines.append(
                    f"- Aligned (&lt;15pp): model hit {_pct(_hit_rate(a['lean_hit']))} "
                    f"({int(a['lean_hit'].sum())}/{len(a)})"
                )
        # HIGH/LOW
        for prof in ("HIGH", "LOW"):
            sub = scores[scores["score_profile"] == prof]
            if len(sub):
                lines.append(
                    f"- {prof} profile: lean hit {_pct(_hit_rate(sub['lean_hit']))} "
                    f"(n={len(sub)}, MAE={sub['abs_total_error'].mean():.2f})"
                )
        # by league
        lines += ["", "| League | n | Model hit | Pin hit | MAE | Bias |", "|--------|--:|----------:|--------:|----:|-----:|"]
        for lg, g in scores.groupby("league"):
            pk = g.dropna(subset=["pin_lean"])
            lines.append(
                f"| {lg} | {len(g)} | {_pct(_hit_rate(g['lean_hit']))} | "
                f"{_pct(_hit_rate(pk['pin_hit'])) if len(pk) else 'n/a'} | "
                f"{g['abs_total_error'].mean():.2f} | {g['total_error'].mean():+.2f} |"
            )
        # worst misses
        worst = scores.sort_values("abs_total_error", ascending=False).head(8)
        lines += ["", "### Largest total misses", ""]
        for _, r in worst.iterrows():
            lines.append(
                f"- {r.get('league')} {r.get('match')}: proj {r.get('proj_total')} → "
                f"{int(r['actual_home'])}-{int(r['actual_away'])} "
                f"(err {r['total_error']:+.1f}) lean={r.get('lean')} "
                f"{'HIT' if r['lean_hit'] else 'MISS'}"
            )

        lines += ["", "## 3. Team totals (paper TRACK / CONFLICT)", ""]
    if len(tt) == 0:
        lines.append("_No TT rows joined to finals yet._")
    else:
        lean_pp = pd.to_numeric(tt.get("tt_lean_pp"), errors="coerce").fillna(0)
        conflict = tt.get("tt_pin_conflict", pd.Series([False] * len(tt)))
        if not isinstance(conflict, pd.Series):
            conflict = pd.Series([False] * len(tt))
        conflict = conflict.fillna(False).astype(bool)
        slices = [
            ("All joined TT", pd.Series([True] * len(tt), index=tt.index)),
            ("Edge >=8pp", lean_pp >= 8),
            ("TRACK-style (>=8pp, not conflict)", (lean_pp >= 8) & (~conflict)),
            ("CONFLICT >=15pp", conflict | (lean_pp >= 15)),
        ]
        for label, mask in slices:
            sub = tt.loc[mask]
            decided = sub[sub["won"].notna()] if len(sub) else sub
            hit = _hit_rate(decided["won"] == 1) if len(decided) else None
            units = (
                float(pd.to_numeric(decided["profit_u"], errors="coerce").fillna(0).sum())
                if len(decided) and "profit_u" in decided.columns
                else 0.0
            )
            lines.append(
                f"- **{label}**: hit {_pct(hit)} (n={len(decided)}) · paper units {units:+.2f}u @ Pin"
            )

        tracks = tt.loc[(lean_pp >= 8) & (~conflict)].copy()
        if len(tracks):
            lines += ["", "### TRACK-style (>=8pp, not conflict)", ""]
            for _, r in tracks.sort_values("tt_lean_pp", ascending=False).head(20).iterrows():
                w = r.get("won")
                mark = (
                    "P"
                    if w is None or (isinstance(w, float) and np.isnan(w))
                    else ("W" if int(w) == 1 else "L")
                )
                pu = r.get("profit_u")
                pu_s = f"{float(pu):+.2f}u" if pu is not None and pd.notna(pu) else "n/a"
                lines.append(
                    f"- {mark} **{r.get('team')}** {r.get('tt_lean')} {r.get('tt_line')} "
                    f"goals={r.get('actual_team_goals')} | {r.get('tt_vs_pin')} | {r.get('match')} "
                    f"| {pu_s}"
                )

        lines += ["", "### By lean side (all joined TT)", ""]
        for lean in ("OVER", "UNDER"):
            sub = tt[tt["tt_lean"].astype(str).str.upper() == lean]
            decided = sub[sub["won"].notna()]
            if len(decided) == 0:
                continue
            units = float(pd.to_numeric(decided["profit_u"], errors="coerce").fillna(0).sum())
            lines.append(
                f"- {lean}: hit {_pct(_hit_rate(decided['won'] == 1))} (n={len(decided)}) · {units:+.2f}u"
            )

    lines += [
        "",
        "## 4. Live packs (flagged PLAY ledger)",
        "",
        f"Open after settle: **{live_summary.get('n_open')}** · "
        f"Settled: **{live_summary.get('n_settled')}** · Total logged: **{live_summary.get('n_total')}**",
        "",
    ]
    # Play-by-play from ledger CSV
    live_csv = ROOT / "data" / "gameday" / "live_ledger.csv"
    if live_csv.is_file():
        ll = pd.read_csv(live_csv)
        lines += ["", "### Play-by-play", ""]
        for _, r in ll.sort_values("date").iterrows():
            w = r.get("won")
            if str(r.get("status")) != "settled":
                mark = "OPEN"
            elif pd.isna(w):
                mark = "P"
            else:
                mark = "W" if int(w) == 1 else "L"
            act = (
                f"{r.get('actual_home')}-{r.get('actual_away')}"
                if pd.notna(r.get("actual_home"))
                else "—"
            )
            pu = r.get("profit_u")
            pu_s = f"{float(pu):+.2f}u" if pd.notna(pu) else ""
            lines.append(
                f"- {mark} {r.get('date')} **{r.get('system')}** "
                f"{r.get('home_team')} vs {r.get('away_team')} {r.get('side')} "
                f"→ {act} {pu_s}"
            )
        settled = ll[ll["status"] == "settled"]
        if len(settled):
            total_u = float(pd.to_numeric(settled["profit_u"], errors="coerce").fillna(0).sum())
            lines.append("")
            lines.append(f"**Ledger total (all settled flags):** {total_u:+.2f}u")
    lines.append("")
    for s in live_summary.get("systems") or []:
        roi = s.get("roi")
        roi_s = "n/a" if roi is None else f"{100 * float(roi):+.1f}%"
        lines.append(
            f"- **{s.get('system')}**: n={s.get('n')} decided={s.get('n_decided')} "
            f"W={s.get('wins')} units={s.get('units')} ROI={roi_s} open={s.get('open')}"
        )

    lines += [
        "",
        "## 5. TT paper ledger summary",
        "",
        f"```json\n{json.dumps(tt_summary, indent=2, default=str)}\n```",
        "",
        "## 6. Recommendations (no live rule changes)",
        "",
        "### Keep doing",
        "- Score Predictions stay **information only** — week-1 lean hit vs Pin still does not clear a promotion bar.",
        "- Keep CONFLICT ≥15pp flags on the Score tab.",
        "- Keep TT paper TRACK logging; settle after each Full Model Refresh.",
        "- Live pack rules: **unchanged**. Week-1 live n is tiny (4 new EPL flags) — do not retune edges.",
        "",
        "### Inputs to watch",
        "- **Bundesliga** still 0 current-season rows — season may not have started / Understat empty; refresh again when matchday 1 lands.",
        "- **Primeira / Championship / Turkey / Eredivisie / Belgium** FD files lag (many still end ~16–17 Aug). Weekend form for those leagues is thin until FD catches up.",
        "- **MLS refresh failed** this run — fix or skip; do not trust MLS Score/TT until aligned.",
        "- EPL / Serie A / La Liga / Ligue 1: Understat extras are feeding form+xG — good; re-run Score Predictions before next slate.",
        "",
        "### Optional process tweaks (not pack rules)",
        "- Archive dated `SCORE_PREDICTIONS_YYYYMMDD.csv` automatically on each Score refresh (done manually for Aug 21).",
        "- Expand Full Model Refresh default leagues to include Score slate leagues (Ligue1, etc.) so week-end retros are one click.",
        "- Hull match Under PLAY won while Hull TT Under 0.5 lost (Hull scored 2) — treat match O/U and TT as separate signals.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Collecting actuals from aligned tables...", flush=True)
    act = collect_actuals()
    print(f"  total unique finals: {len(act)}", flush=True)
    act.to_csv(OUT / "actuals_week1.csv", index=False)
    write_settled(act)

    if not PRED.is_file():
        # fall back to current
        pred_path = ROOT / "experiments" / "gameday_scan" / "SCORE_PREDICTIONS.csv"
    else:
        pred_path = PRED
    pred = pd.read_csv(pred_path)
    # ensure home/away columns
    if "home_team" not in pred.columns and "match" in pred.columns:
        parts = pred["match"].astype(str).str.split(" vs ", n=1, expand=True)
        pred["home_team"] = parts[0]
        pred["away_team"] = parts[1] if parts.shape[1] > 1 else ""

    print("Scoring predictions...", flush=True)
    scores = analyze_scores(pred, act)
    scores.to_csv(OUT / "score_vs_actual.csv", index=False)
    print(f"  joined scores: {len(scores)}", flush=True)

    tt = pd.DataFrame()
    if TT_SNAP.is_file():
        print("Scoring team totals snap...", flush=True)
        tt_raw = pd.read_csv(TT_SNAP)
        tt = analyze_tt(tt_raw, act)
        # also try fuzzy for unmatched
        if len(tt) < len(tt_raw) * 0.5:
            # enrich act index with fuzzy keys for remaining
            pass
        tt.to_csv(OUT / "tt_vs_actual.csv", index=False)
        print(f"  joined TT: {len(tt)}", flush=True)

    # Settle ledgers
    print("Settling live + TT ledgers...", flush=True)
    from origination.gameday.live_ledger import settle_open as live_settle
    from origination.gameday.live_ledger import summary as live_summary_fn
    from origination.gameday.live_ledger import write_report as live_report
    from origination.gameday.tt_ledger import settle_open as tt_settle
    from origination.gameday.tt_ledger import write_report as tt_report

    n_live = live_settle()
    n_tt = tt_settle()
    print(f"  settled live={n_live} tt={n_tt}", flush=True)
    live_path = live_report()
    tt_path = tt_report()

    stamp = {}
    stamp_path = ROOT / "data" / "gameday" / "last_data_update.json"
    if stamp_path.is_file():
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))

    # TT summary from file
    tt_sum = {}
    ts = ROOT / "data" / "gameday" / "tt_ledger_summary.json"
    if ts.is_file():
        tt_sum = json.loads(ts.read_text(encoding="utf-8"))

    report = build_report(act, scores, tt, live_summary_fn(), tt_sum, stamp)
    out_md = OUT / "REPORT.md"
    out_md.write_text(report, encoding="utf-8")
    (ROOT / "docs" / "WEEK1_GAUNTLET.md").write_text(report, encoding="utf-8")
    print(f"Wrote {out_md}", flush=True)
    print(f"Wrote docs/WEEK1_GAUNTLET.md", flush=True)
    print(f"Live report: {live_path}", flush=True)
    print(f"TT report: {tt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
