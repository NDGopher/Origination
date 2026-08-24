#!/usr/bin/env python
"""
Team-total calibration from Score Predictions historical OOS (no closing TT market on FD).

Uses lambda_home / lambda_away from experiments/score_predictions/historical_oos.parquet.
Information only — not a live pack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.models.poisson import team_total_over_prob

OOS = ROOT / "experiments" / "score_predictions" / "historical_oos.parquet"
OUT = ROOT / "experiments" / "score_predictions" / "TEAM_TOTALS.md"
LINES = (0.5, 1.5, 2.5)


def _side_frame(df: pd.DataFrame) -> pd.DataFrame:
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


def hit_rate(lam: pd.Series, goals: pd.Series, line: float) -> dict:
    p = lam.map(lambda x: team_total_over_prob(float(x), line) if np.isfinite(x) else np.nan)
    actual = goals > line
    lean = p >= 0.5
    mask = p.notna() & goals.notna()
    if mask.sum() == 0:
        return {"n": 0, "hit": None, "bias": None, "mae": None, "brier": None}
    return {
        "n": int(mask.sum()),
        "hit": float((lean[mask] == actual[mask]).mean()),
        "bias": float((lam[mask] - goals[mask]).mean()),
        "mae": float((lam[mask] - goals[mask]).abs().mean()),
        "brier": float(((p[mask] - actual[mask].astype(float)) ** 2).mean()),
    }


def main() -> int:
    if not OOS.is_file():
        print(f"Missing {OOS} — run eval_score_predictions_historical.py first", flush=True)
        return 1
    df = pd.read_parquet(OOS)
    sides = _side_frame(df)
    lines_out = ["# Team totals — historical Score path", "", 
                 "Source: `historical_oos.parquet` (Dixon–Coles + intercept + temperature).",
                 "No football-data closing team-total odds — this is **model calibration vs actual goals**, not EV vs Pin.",
                 "Live Pin TT comparison is in `SCORE_TEAM_TOTALS.csv` (pre-match only).",
                 "",
                 "## Pooled (home + away)",
                 "",
                 "| Line | n | Model O/U lean hit | Bias (proj−act) | MAE | Brier |",
                 "|-----:|--:|-------------------:|----------------:|----:|------:|"]
    for line in LINES:
        s = hit_rate(sides["lam"], sides["goals"], line)
        lines_out.append(
            f"| {line} | {s['n']} | {100*s['hit']:.1f}% | {s['bias']:+.2f} | {s['mae']:.2f} | {s['brier']:.3f} |"
            if s["hit"] is not None
            else f"| {line} | 0 | — | — | — | — |"
        )

    lines_out += ["", "## By league @ 1.5", "",
                  "| League | n | Hit | Bias | MAE |",
                  "|--------|--:|----:|-----:|----:|"]
    for lg, g in sides.groupby("league"):
        s = hit_rate(g["lam"], g["goals"], 1.5)
        if s["hit"] is None:
            continue
        lines_out.append(
            f"| {lg} | {s['n']} | {100*s['hit']:.1f}% | {s['bias']:+.2f} | {s['mae']:.2f} |"
        )

    # High projected team goals slice
    lines_out += ["", "## HIGH team projections (lam ≥ 1.8) @ main 1.5", ""]
    hi = sides[sides["lam"] >= 1.8]
    s = hit_rate(hi["lam"], hi["goals"], 1.5)
    lines_out.append(
        f"n={s['n']}  hit={100*s['hit']:.1f}%  bias={s['bias']:+.2f}  MAE={s['mae']:.2f}"
        if s["hit"] is not None
        else "n=0"
    )
    lo = sides[sides["lam"] <= 0.9]
    s2 = hit_rate(lo["lam"], lo["goals"], 0.5)
    lines_out += ["", "## LOW team projections (lam ≤ 0.9) @ 0.5", ""]
    lines_out.append(
        f"n={s2['n']}  hit={100*s2['hit']:.1f}%  bias={s2['bias']:+.2f}  MAE={s2['mae']:.2f}"
        if s2["hit"] is not None
        else "n=0"
    )

    lines_out += [
        "",
        "## Takeaways",
        "",
        "- Team totals can be sharper informationally than match O/U when one side drives the total.",
        "- Without historical Pin TT closes we **cannot** claim a betting edge yet — only calibration.",
        "- Use live `SCORE_TEAM_TOTALS.csv` for model vs Pin value; keep CONFLICT discipline on match O/U.",
        "- Do **not** promote team totals to a live pack from this alone.",
        "",
    ]
    OUT.write_text("\n".join(lines_out), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
