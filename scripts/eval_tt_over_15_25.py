#!/usr/bin/env python
"""
Extensive test: team-total OVER 1.5 and OVER 2.5 only (no 0.5 lines).

Uses historical_oos.parquet lambdas vs actual team goals.
Also grades the Aug-21 live Pin snap / week-1 actuals when present.

Information / paper only — not a live pack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.models.poisson import team_total_over_prob  # noqa: E402

OOS = ROOT / "experiments" / "score_predictions" / "historical_oos.parquet"
OUT = ROOT / "experiments" / "score_predictions" / "TT_OVER_15_25.md"
WEEK1_TT = ROOT / "experiments" / "week1_gauntlet" / "tt_vs_actual.csv"
WEEK1_ACT = ROOT / "experiments" / "week1_gauntlet" / "actuals_week1.csv"
SNAP_TT = ROOT / "experiments" / "gameday_scan" / "SCORE_TEAM_TOTALS_20260821.csv"

FOCUS_LINES = (1.5, 2.5)
# Model confidence thresholds for "lean OVER"
P_CUTS = (0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70)
# Edge vs flat 50% in pp (proxy when no Pin)
EDGE_CUTS = (0, 5, 8, 10, 12, 15)


def _sides(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, lam_c, g_c in (
        ("home", "lambda_home", "home_goals"),
        ("away", "lambda_away", "away_goals"),
    ):
        sub = df[["league", "match_id", "season", lam_c, g_c]].copy()
        sub = sub.rename(columns={lam_c: "lam", g_c: "goals"})
        sub["side"] = side
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def _enrich(sides: pd.DataFrame, line: float) -> pd.DataFrame:
    out = sides.copy()
    out["line"] = line
    out["p_over"] = out["lam"].map(
        lambda x: team_total_over_prob(float(x), line) if np.isfinite(x) else np.nan
    )
    out["actual_over"] = out["goals"] > line
    out["edge_vs_50_pp"] = 100 * (out["p_over"] - 0.5)
    # Paper unit @ flat decimal odds implied by 50/50 (2.0) — naive
    out["profit_flat2"] = np.where(out["actual_over"], 1.0, -1.0)
    # Paper @ model-fair odds (zero-EV if calibrated) — for comparison only
    fair = 1.0 / out["p_over"].clip(0.05, 0.95)
    out["profit_at_fair"] = np.where(out["actual_over"], fair - 1.0, -1.0)
    return out


def _slice_stats(g: pd.DataFrame, label: str) -> dict:
    g = g.dropna(subset=["p_over", "goals"])
    n = len(g)
    if n == 0:
        return {"label": label, "n": 0}
    hit = float(g["actual_over"].mean())  # over rate among selected (we only select overs)
    # For OVER bets, hit = actual_over rate
    base = None
    units_flat = float(g["profit_flat2"].sum())
    roi_flat = units_flat / n
    # CLV-style: did we beat 50%?
    edge_realized = hit - 0.5
    return {
        "label": label,
        "n": n,
        "over_hit": hit,
        "vs_50_pp": 100 * edge_realized,
        "units_@2.0": round(units_flat, 1),
        "roi_@2.0": roi_flat,
        "mean_p": float(g["p_over"].mean()),
        "mean_lam": float(g["lam"].mean()),
    }


def _fmt(r: dict) -> str:
    if r["n"] == 0:
        return f"| {r['label']} | 0 | — | — | — | — |"
    return (
        f"| {r['label']} | {r['n']} | {100*r['over_hit']:.1f}% | "
        f"{r['vs_50_pp']:+.1f} | {r['units_@2.0']:+.0f}u | {100*r['roi_@2.0']:+.1f}% |"
    )


def hist_report(sides: pd.DataFrame) -> list[str]:
    lines = [
        "# Team totals — OVER 1.5 / OVER 2.5 extensive test",
        "",
        "Paper research only. **No 0.5 lines.** Focus: betting the Over.",
        "",
        "Source: `historical_oos.parquet` (last ~3 seasons OOS Dixon–Coles path).",
        "No historical Pin TT closes — units below assume **flat 2.00** on every Over "
        "(= need >50% hit to profit). Real Pin Overs are usually shorter than 2.00, "
        "so these ROIs are **optimistic upper bounds**.",
        "",
    ]
    for line in FOCUS_LINES:
        en = _enrich(sides, line)
        base_rate = float(en["actual_over"].mean())
        lines += [
            f"## Line {line} — base Over rate {100*base_rate:.1f}%",
            "",
            "### By model P(Over) threshold (bet Over when p ≥ cut)",
            "",
            "| Cut | n | Over hit | vs 50% (pp) | Units @2.00 | ROI @2.00 |",
            "|-----|--:|---------:|------------:|------------:|----------:|",
        ]
        for cut in P_CUTS:
            sub = en[en["p_over"] >= cut]
            lines.append(_fmt(_slice_stats(sub, f"p≥{cut:.2f}")))

        lines += [
            "",
            "### By edge vs 50% (pp)",
            "",
            "| Edge | n | Over hit | vs 50% (pp) | Units @2.00 | ROI @2.00 |",
            "|------|--:|---------:|------------:|------------:|----------:|",
        ]
        for e in EDGE_CUTS:
            sub = en[en["edge_vs_50_pp"] >= e]
            lines.append(_fmt(_slice_stats(sub, f"≥{e}pp")))

        # Strong projection slices
        lines += ["", f"### HIGH λ slices @ Over {line}", ""]
        for lo, hi_lab in ((1.6, "λ≥1.6"), (1.8, "λ≥1.8"), (2.0, "λ≥2.0"), (2.2, "λ≥2.2")):
            if line == 2.5 and lo < 2.0:
                continue
            sub = en[(en["lam"] >= lo) & (en["p_over"] >= 0.55)]
            lines.append(_fmt(_slice_stats(sub, hi_lab + " & p≥0.55")))

        lines += [
            "",
            f"### By league — Over {line}, p≥0.58",
            "",
            "| League | n | Over hit | vs 50% | Units @2.00 | ROI |",
            "|--------|--:|---------:|-------:|------------:|----:|",
        ]
        for lg, g in en.groupby("league"):
            sub = g[g["p_over"] >= 0.58]
            lines.append(_fmt(_slice_stats(sub, str(lg))))

        # Season stability for best cut
        best_cut = 0.58
        lines += [
            "",
            f"### Season stability — Over {line}, p≥{best_cut}",
            "",
            "| Season | n | Hit | ROI @2.00 |",
            "|--------|--:|----:|----------:|",
        ]
        sub_all = en[en["p_over"] >= best_cut]
        if "season" in sub_all.columns:
            for ssn, g in sub_all.groupby("season"):
                st = _slice_stats(g, str(ssn))
                if st["n"] == 0:
                    continue
                lines.append(
                    f"| {ssn} | {st['n']} | {100*st['over_hit']:.1f}% | {100*st['roi_@2.0']:+.1f}% |"
                )
        lines.append("")
    return lines


def week1_pin_report() -> list[str]:
    lines = [
        "## Week-1 live Pin snap — OVER 1.5 / 2.5 only",
        "",
        "From `SCORE_TEAM_TOTALS_20260821` joined to week-1 finals. Real Pin odds.",
        "",
    ]
    if not WEEK1_TT.is_file() and not SNAP_TT.is_file():
        lines.append("_No week-1 TT file._")
        return lines

    if WEEK1_TT.is_file():
        tt = pd.read_csv(WEEK1_TT)
    else:
        tt = pd.read_csv(SNAP_TT)
        # need settle from actuals
        if WEEK1_ACT.is_file():
            act = pd.read_csv(WEEK1_ACT).set_index("match_id")
            rows = []
            for _, r in tt.iterrows():
                mid = str(r.get("match_id"))
                if mid not in act.index:
                    continue
                a = act.loc[mid]
                if isinstance(a, pd.DataFrame):
                    a = a.iloc[-1]
                side = str(r.get("side")).lower()
                goals = float(a["actual_home"] if side == "home" else a["actual_away"])
                line = float(r.get("tt_line"))
                lean = str(r.get("tt_lean")).upper()
                won = 1 if ((goals > line and lean == "OVER") or (goals < line and lean == "UNDER")) else 0
                if abs(line - round(line)) < 1e-9 and goals == line:
                    won = np.nan
                rows.append({**r.to_dict(), "actual_team_goals": goals, "won": won})
            tt = pd.DataFrame(rows)

    # Filter focus
    line = pd.to_numeric(tt.get("tt_line"), errors="coerce")
    lean = tt.get("tt_lean", pd.Series(dtype=str)).astype(str).str.upper()
    focus = tt[(line.isin([1.5, 2.5])) & (lean == "OVER")].copy()
    if len(focus) == 0:
        lines.append("_No Over 1.5/2.5 rows in week-1 join._")
        return lines

    def pin_units(g: pd.DataFrame) -> tuple[int, float, float]:
        g = g.dropna(subset=["won"])
        n = len(g)
        if n == 0:
            return 0, 0.0, float("nan")
        hits = int(g["won"].sum())
        # use pin over odds
        odds = pd.to_numeric(g.get("tt_pin_over"), errors="coerce")
        if "profit_u" in g.columns and g["profit_u"].notna().any():
            # only count rows that were OVER leans
            profit = float(pd.to_numeric(g["profit_u"], errors="coerce").fillna(0).sum())
        else:
            profit = 0.0
            for i, r in g.iterrows():
                o = float(r["tt_pin_over"]) if pd.notna(r.get("tt_pin_over")) else None
                if o is None:
                    continue
                profit += (o - 1.0) if int(r["won"]) == 1 else -1.0
        return n, hits / n, profit

    lines += [
        "| Slice | n | Hit | Pin units |",
        "|-------|--:|----:|----------:|",
    ]
    for lab, mask in (
        ("All Over 1.5/2.5", pd.Series(True, index=focus.index)),
        ("Over 1.5", line.loc[focus.index] == 1.5),
        ("Over 2.5", line.loc[focus.index] == 2.5),
        ("Edge ≥8pp", pd.to_numeric(focus.get("tt_lean_pp"), errors="coerce").fillna(0) >= 8),
        ("Edge ≥10pp", pd.to_numeric(focus.get("tt_lean_pp"), errors="coerce").fillna(0) >= 10),
        ("Edge ≥12pp", pd.to_numeric(focus.get("tt_lean_pp"), errors="coerce").fillna(0) >= 12),
    ):
        sub = focus.loc[mask]
        n, hit, u = pin_units(sub)
        hit_s = "—" if n == 0 or hit != hit else f"{100*hit:.0f}%"
        lines.append(f"| {lab} | {n} | {hit_s} | {u:+.2f}u |")

    # List settled
    lines += ["", "### Week-1 Over 1.5 / 2.5 detail", ""]
    for _, r in focus.sort_values("tt_lean_pp", ascending=False).iterrows():
        w = r.get("won")
        mark = "P" if pd.isna(w) else ("W" if int(w) == 1 else "L")
        lines.append(
            f"- {mark} **{r.get('team')}** OVER {r.get('tt_line')} "
            f"goals={r.get('actual_team_goals')} | {r.get('tt_vs_pin')} | {r.get('match')}"
        )
    lines.append("")
    return lines


def takeaways(hist_sides: pd.DataFrame) -> list[str]:
    # Summarize best realistic cuts
    bits = []
    for line in FOCUS_LINES:
        en = _enrich(hist_sides, line)
        for cut in (0.55, 0.58, 0.60, 0.62):
            st = _slice_stats(en[en["p_over"] >= cut], f"{line}/p≥{cut}")
            if st["n"] >= 500:
                bits.append(st)

    lines = [
        "## Verdict for tracker design",
        "",
        "- **Drop all 0.5 lines** from TRACK (agreed).",
        "- **Primary TRACK = OVER 1.5 and OVER 2.5 only** (no Unders in the main card for now).",
        "- Historical Over hit when model is confident (p≥0.58–0.62) clears 50% at flat 2.00 — "
        "**but Pin Overs are shorter**, so do **not** promote to a live pack yet.",
        "- Raise paper edge vs Pin to **≥10pp** (week-1 soft 8pp Overs struggled).",
        "- Prefer big-5 / Primeira when logging; soft-weight Championship/MLS/Turkey until form denser.",
        "- Keep CONFLICT ≥15pp as watch-only.",
        "",
        "### Suggested TRACK rules (paper)",
        "",
        "1. Lean **OVER** only",
        "2. Line ∈ **{1.5, 2.5}**",
        "3. Model−Pin edge **≥10pp** and **<15pp** (else CONFLICT_WATCH)",
        "4. Focus window (next 24h + through tomorrow)",
        "5. Optional: require model p_over ≥ 0.55",
        "",
    ]
    if bits:
        lines += ["### Historical snapshot (flat 2.00 — optimistic)", "",
                  "| Cell | n | Hit | ROI @2.00 |",
                  "|------|--:|----:|----------:|"]
        for st in bits:
            lines.append(
                f"| {st['label']} | {st['n']} | {100*st['over_hit']:.1f}% | {100*st['roi_@2.0']:+.1f}% |"
            )
        lines.append("")
    return lines


def main() -> int:
    if not OOS.is_file():
        print(f"Missing {OOS}", flush=True)
        return 1
    df = pd.read_parquet(OOS)
    # seasons column may exist
    if "season" not in df.columns and "date" in df.columns:
        df["season"] = pd.to_datetime(df["date"]).dt.year
    sides = _sides(df)
    report = hist_report(sides)
    report += week1_pin_report()
    report += takeaways(sides)
    OUT.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)
    # print key tables
    print(OUT.read_text(encoding="utf-8")[-4000:], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
