#!/usr/bin/env python
"""
Team-total OVER value using real Pin odds (not flat 2.00).

Joins live SCORE_TEAM_TOTALS snaps (+ week-1 finals) and grades model Overs
at Pin prices. Also summarizes model-only calibration honestly (hit rates only).

Paper research — not a live pack.
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
OUT = ROOT / "experiments" / "score_predictions" / "TT_OVER_PIN_VALUE.md"
WEEK1 = ROOT / "experiments" / "week1_gauntlet" / "tt_vs_actual.csv"
SNAP_DIR = ROOT / "experiments" / "gameday_scan"
ACT = ROOT / "experiments" / "week1_gauntlet" / "actuals_week1.csv"

LINES = (0.5, 1.5, 2.5)


def _sides(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, lam_c, g_c in (
        ("home", "lambda_home", "home_goals"),
        ("away", "lambda_away", "away_goals"),
    ):
        sub = df[["league", "match_id", lam_c, g_c]].copy()
        sub = sub.rename(columns={lam_c: "lam", g_c: "goals"})
        sub["side"] = side
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def calib_table(sides: pd.DataFrame) -> list[str]:
    """Honest calibration — hit rates only, no fake odds."""
    lines = [
        "# Team totals OVER — Pin-priced value + model filters",
        "",
        "Paper only. **Unders are out of TRACK.** Overs on 0.5 / 1.5 / 2.5 are in scope.",
        "",
        "## 1. Why flat 2.00 ROIs were wrong",
        "",
        "Historical tables that assume every Over pays **2.00** inflate ROI. "
        "On good teams / big games, Pin team Over 1.5 is often **~1.4–1.8**, and Over 0.5 is often **~1.2–1.5**. "
        "You need a much higher hit rate to beat short prices. Below we use **real Pin Over odds** where we have them.",
        "",
        "## 2. Model calibration only (no odds — hit rates)",
        "",
        "Source: `historical_oos.parquet`. Useful for filters (λ, p_over), not for claiming EV.",
        "",
    ]
    for line in LINES:
        p = sides["lam"].map(lambda x, ln=line: team_total_over_prob(float(x), ln) if np.isfinite(x) else np.nan)
        over = sides["goals"] > line
        base = float(over.mean())
        lines += [f"### Over {line} — base rate {100*base:.1f}%", "",
                  "| Filter | n | Hit | vs base (pp) |",
                  "|--------|--:|----:|-------------:|"]
        for lab, mask in (
            ("all", p.notna()),
            ("p≥0.55", p >= 0.55),
            ("p≥0.60", p >= 0.60),
            ("p≥0.65", p >= 0.65),
            ("λ ≥ line+0.3", sides["lam"] >= line + 0.3),
            ("λ ≥ line+0.5", sides["lam"] >= line + 0.5),
            ("λ ≥ line+0.3 & p≥0.55", (sides["lam"] >= line + 0.3) & (p >= 0.55)),
            ("λ ≥ line+0.5 & p≥0.60", (sides["lam"] >= line + 0.5) & (p >= 0.60)),
        ):
            m = mask & p.notna() & sides["goals"].notna()
            n = int(m.sum())
            if n < 30:
                lines.append(f"| {lab} | {n} | — | — |")
                continue
            hit = float(over[m].mean())
            lines.append(f"| {lab} | {n} | {100*hit:.1f}% | {100*(hit-base):+.1f} |")
        lines.append("")
    return lines


def _load_pin_joined() -> pd.DataFrame:
    """Week-1 TT vs actual with Pin overs; prefer week1 file."""
    if WEEK1.is_file():
        tt = pd.read_csv(WEEK1)
    else:
        return pd.DataFrame()
    lean = tt.get("tt_lean", pd.Series(dtype=str)).astype(str).str.upper()
    line = pd.to_numeric(tt.get("tt_line"), errors="coerce")
    tt = tt[(lean == "OVER") & (line.isin(list(LINES)))].copy()
    tt["tt_line"] = line.loc[tt.index]
    tt["pin_over"] = pd.to_numeric(tt.get("tt_pin_over"), errors="coerce")
    tt["p_over"] = pd.to_numeric(tt.get("tt_p_over"), errors="coerce")
    tt["proj"] = pd.to_numeric(tt.get("tt_proj"), errors="coerce")
    tt["edge_pp"] = pd.to_numeric(tt.get("tt_lean_pp"), errors="coerce")
    tt["won"] = pd.to_numeric(tt.get("won"), errors="coerce")
    # rebuild profit at pin if needed
    profits = []
    for _, r in tt.iterrows():
        if pd.isna(r["won"]) or pd.isna(r["pin_over"]) or r["pin_over"] <= 1:
            profits.append(np.nan)
        elif int(r["won"]) == 1:
            profits.append(float(r["pin_over"]) - 1.0)
        else:
            profits.append(-1.0)
    tt["profit_pin"] = profits
    tt["implied"] = 1.0 / tt["pin_over"]
    tt["model_edge_vs_pin_pp"] = 100 * (tt["p_over"] - tt["implied"])
    return tt


def pin_value_section(tt: pd.DataFrame) -> list[str]:
    lines = [
        "## 3. Live Pin-priced Overs (week-1 snap → finals)",
        "",
        "Real Pin Over decimal odds. This is the honest sample (small n).",
        "",
    ]
    if len(tt) == 0:
        lines.append("_No joined Pin TT overs yet._")
        return lines

    def row(lab: str, g: pd.DataFrame) -> str:
        g = g.dropna(subset=["won", "pin_over"])
        n = len(g)
        if n == 0:
            return f"| {lab} | 0 | — | — | — | — |"
        hit = float(g["won"].mean())
        units = float(g["profit_pin"].sum())
        avg_odds = float(g["pin_over"].mean())
        be = 1.0 / avg_odds  # break-even hit at avg odds
        return (
            f"| {lab} | {n} | {100*hit:.0f}% | {avg_odds:.2f} | "
            f"need {100*be:.0f}% | {units:+.2f}u |"
        )

    lines += [
        "| Slice | n | Hit | Avg Pin Over | Break-even | Pin units |",
        "|-------|--:|----:|-------------:|-----------:|----------:|",
        row("All Over 0.5/1.5/2.5", tt),
    ]
    for ln in LINES:
        lines.append(row(f"Over {ln}", tt[tt["tt_line"] == ln]))

    lines += ["", "### Filters on Pin sample (value hunt)", "",
              "| Slice | n | Hit | Avg Pin | BE | Units |",
              "|-------|--:|----:|--------:|---:|------:|"]
    for lab, mask in (
        ("edge≥8pp", tt["edge_pp"] >= 8),
        ("edge≥10pp", tt["edge_pp"] >= 10),
        ("edge≥12pp", tt["edge_pp"] >= 12),
        ("p≥0.60", tt["p_over"] >= 0.60),
        ("p≥0.65", tt["p_over"] >= 0.65),
        ("proj ≥ line+0.3", tt["proj"] >= tt["tt_line"] + 0.3),
        ("proj ≥ line+0.5", tt["proj"] >= tt["tt_line"] + 0.5),
        ("edge≥10 & proj≥line+0.3", (tt["edge_pp"] >= 10) & (tt["proj"] >= tt["tt_line"] + 0.3)),
        ("edge≥10 & p≥0.60", (tt["edge_pp"] >= 10) & (tt["p_over"] >= 0.60)),
        ("Pin Over ≥1.70 (longer)", tt["pin_over"] >= 1.70),
        ("Pin Over ≥1.90", tt["pin_over"] >= 1.90),
        ("Pin Over <1.50 (short fav)", tt["pin_over"] < 1.50),
        ("big-5+Primeira", tt["league"].isin(["EPL", "SerieA", "LaLiga", "Bundesliga", "Ligue1", "PrimeiraLiga"])),
        ("big-5 edge≥10", tt["league"].isin(["EPL", "SerieA", "LaLiga", "Ligue1", "Bundesliga"]) & (tt["edge_pp"] >= 10)),
    ):
        lines.append(row(lab, tt.loc[mask]))

    # Odds distribution insight
    lines += [
        "",
        "### What Pin actually priced on week-1 Overs",
        "",
        f"- Median Pin Over: **{tt['pin_over'].median():.2f}** · mean **{tt['pin_over'].mean():.2f}**",
        f"- Over 0.5 median: **{tt.loc[tt.tt_line==0.5,'pin_over'].median() if (tt.tt_line==0.5).any() else float('nan'):.2f}**",
        f"- Over 1.5 median: **{tt.loc[tt.tt_line==1.5,'pin_over'].median() if (tt.tt_line==1.5).any() else float('nan'):.2f}**",
        f"- Over 2.5 median: **{tt.loc[tt.tt_line==2.5,'pin_over'].median() if (tt.tt_line==2.5).any() else float('nan'):.2f}**",
        "",
    ]
    return lines


def today_candidates() -> list[str]:
    """Current SCORE_TEAM_TOTALS Overs that clear paper TRACK-ish filters."""
    path = SNAP_DIR / "SCORE_TEAM_TOTALS.csv"
    lines = ["## 4. Current slate — Over value scan (model vs Pin)", ""]
    if not path.is_file():
        lines.append("_No SCORE_TEAM_TOTALS.csv_")
        return lines
    tt = pd.read_csv(path)
    if "has_pin_tt" in tt.columns:
        tt = tt[tt["has_pin_tt"] == True]
    lean = tt.get("tt_lean", pd.Series(dtype=str)).astype(str).str.upper()
    line = pd.to_numeric(tt.get("tt_line"), errors="coerce")
    edge = pd.to_numeric(tt.get("tt_edge_over_pp"), errors="coerce")
    if edge.isna().all():
        edge = pd.to_numeric(tt.get("tt_lean_pp"), errors="coerce")
    p = pd.to_numeric(tt.get("tt_p_over"), errors="coerce")
    proj = pd.to_numeric(tt.get("tt_proj"), errors="coerce")
    pin = pd.to_numeric(tt.get("tt_pin_over"), errors="coerce")
    focus = tt.get("in_focus", True)
    m = (
        (lean == "OVER")
        & (line.isin(list(LINES)))
        & (edge >= 10)
        & (p >= 0.55)
        & (focus == True if hasattr(focus, "__eq__") else True)
    )
    # also require proj cushion for "good team" feel
    m2 = m & (proj >= line + 0.3)
    show = tt.loc[m].copy()
    show["edge_o"] = edge.loc[show.index]
    show["pin_o"] = pin.loc[show.index]
    show["proj"] = proj.loc[show.index]
    show["line"] = line.loc[show.index]
    show["p_o"] = p.loc[show.index]
    show["be"] = 1.0 / show["pin_o"]
    show["ev_proxy"] = show["p_o"] - show["be"]
    show = show.sort_values("ev_proxy", ascending=False)

    lines.append(
        f"Focus Overs with edge≥10pp & p≥0.55: **{len(show)}**. "
        f"With proj ≥ line+0.3: **{int(m2.sum())}**."
    )
    lines += ["", "| Team | Line | Proj | p_over | Pin | BE | model−BE | Edge pp | Match |",
              "|------|-----:|-----:|-------:|----:|---:|---------:|--------:|-------|"]
    for _, r in show.head(15).iterrows():
        lines.append(
            f"| {r.get('team')} | {r['line']} | {r['proj']} | {100*r['p_o']:.0f}% | "
            f"{r['pin_o']:.2f} | {100*r['be']:.0f}% | {100*r['ev_proxy']:+.1f}pp | "
            f"{r['edge_o']:+.1f} | {r.get('match')} ({r.get('league')}) |"
        )
    if len(show) == 0:
        lines.append("_None on the current focus slate._")
    lines.append("")
    return lines


def verdict() -> list[str]:
    return [
        "## 5. Tracker / research verdict",
        "",
        "- **TRACK = Overs only** on 0.5 / 1.5 / 2.5 (Unders off the main card).",
        "- Judge paper P&L only at **Pin Over odds**, never flat 2.00.",
        "- Model filters worth keeping while we learn: **p_over ≥ 0.55**, **edge vs Pin ≥ 10pp**, "
        "prefer **proj ≥ line + 0.3** (attacking side / big game).",
        "- Over 1.5 is the main hunt; Over 0.5 needs very high hit rates vs short Pin; "
        "Over 2.5 is rare and noisy — keep secondary.",
        "- Soft-skip Championship / MLS / Turkey / Scotland until form denser.",
        "- **Do not promote** until forward Pin ledger is clearly positive.",
        "",
    ]


def main() -> int:
    lines: list[str] = []
    if OOS.is_file():
        sides = _sides(pd.read_parquet(OOS))
        lines += calib_table(sides)
    else:
        lines += ["# Team totals OVER — Pin value", "", "_Missing historical_oos.parquet_", ""]
    tt = _load_pin_joined()
    lines += pin_value_section(tt)
    lines += today_candidates()
    lines += verdict()
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)
    print("\n".join(lines[-80:]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
