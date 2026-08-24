#!/usr/bin/env python
"""
Score Predictions calibration — information tool, not a betting system.

Reads the weekend retro table (and any later graded rows) and reports:
  - O/U lean hit rates vs Pinnacle
  - mean total error (model too high/low)
  - large model–Pin gaps
  - Brier on Over 2.5

Does not change live pack rules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RETRO = ROOT / "experiments" / "weekend_retro" / "prediction_vs_actual.csv"
OUT_DIR = ROOT / "experiments" / "score_predictions"
OUT_MD = OUT_DIR / "CALIBRATION.md"


def _bool(s) -> bool:
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in {"true", "1", "yes"}


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    model_hits = int(df["lean_hit"].map(_bool).sum()) if "lean_hit" in df.columns else 0
    pin_hits = int(df["pin_hit"].map(_bool).sum()) if "pin_hit" in df.columns else 0
    tot_err = pd.to_numeric(df.get("total_error"), errors="coerce")
    proj = pd.to_numeric(df.get("proj_total"), errors="coerce")
    actual = pd.to_numeric(df.get("actual_total"), errors="coerce")
    bias = float((proj - actual).mean()) if proj.notna().any() else float("nan")
    mae = float((proj - actual).abs().mean()) if proj.notna().any() else float("nan")

    over_p = pd.to_numeric(df.get("over_pct"), errors="coerce") / 100.0
    is_over = df.get("actual_ou").astype(str).str.upper().eq("OVER")
    brier = float(((over_p - is_over.astype(float)) ** 2).mean()) if over_p.notna().any() else float("nan")

    under_gap = pd.to_numeric(df.get("model_minus_pin_under_pp"), errors="coerce")
    over_gap = pd.to_numeric(df.get("model_minus_pin_over_pp"), errors="coerce")
    lean = df.get("lean").astype(str).str.upper()
    conflict15 = ((lean.eq("UNDER") & under_gap.abs().ge(15)) | (lean.eq("OVER") & over_gap.abs().ge(15)))
    conflict20 = df["conflict_20pp"].map(_bool) if "conflict_20pp" in df.columns else conflict15

    c15 = df.loc[conflict15]
    c20 = df.loc[conflict20]
    aligned = df.loc[~conflict15]

    under_vs_pin_over = df[
        lean.eq("UNDER")
        & (pd.to_numeric(df.get("pin_over_pct"), errors="coerce") >= 50)
        & under_gap.abs().ge(15)
    ]

    return {
        "n": n,
        "model_hits": model_hits,
        "model_rate": model_hits / n if n else 0,
        "pin_hits": pin_hits,
        "pin_rate": pin_hits / n if n else 0,
        "bias_proj_minus_actual": bias,
        "mae_total": mae,
        "brier_over25": brier,
        "n_conflict15": int(conflict15.sum()),
        "conflict15_hits": int(c15["lean_hit"].map(_bool).sum()) if len(c15) else 0,
        "n_conflict20": int(conflict20.sum()),
        "conflict20_hits": int(c20["lean_hit"].map(_bool).sum()) if len(c20) else 0,
        "aligned_n": int(len(aligned)),
        "aligned_hits": int(aligned["lean_hit"].map(_bool).sum()) if len(aligned) else 0,
        "under_vs_pin_over_n": int(len(under_vs_pin_over)),
        "under_vs_pin_over_hits": int(under_vs_pin_over["lean_hit"].map(_bool).sum())
        if len(under_vs_pin_over)
        else 0,
        "mean_actual_total": float(actual.mean()) if actual.notna().any() else float("nan"),
        "mean_proj_total": float(proj.mean()) if proj.notna().any() else float("nan"),
    }


def render(s: dict) -> str:
    def pct(h, n):
        return f"{h}/{n} ({100 * h / n:.0f}%)" if n else "—"

    bias = s["bias_proj_minus_actual"]
    bias_txt = (
        f"model totals **{abs(bias):.2f} too {'high' if bias > 0 else 'low'}** on average"
        if pd.notna(bias)
        else "n/a"
    )
    return f"""# Score Predictions calibration

**Sample:** weekend 14–16 Aug 2026 snapshot vs finals (`prediction_vs_actual.csv`).  
**n = {s['n']}.** This is a small sample. Do **not** promote Score Predictions to a live pack.

## Headline

| Slice | Result |
|-------|--------|
| Model O/U lean | {pct(s['model_hits'], s['n'])} |
| Pinnacle O/U lean | {pct(s['pin_hits'], s['n'])} |
| Model vs Pin ≥15pp | {pct(s['conflict15_hits'], s['n_conflict15'])} |
| Model vs Pin ≥20pp | {pct(s['conflict20_hits'], s['n_conflict20'])} |
| Model aligned with Pin (<15pp) | {pct(s['aligned_hits'], s['aligned_n'])} |
| Model Under vs Pin Over (≥15pp) | {pct(s['under_vs_pin_over_hits'], s['under_vs_pin_over_n'])} |

## Totals bias

- Mean projected total: **{s['mean_proj_total']:.2f}**
- Mean actual total: **{s['mean_actual_total']:.2f}**
- {bias_txt}
- MAE: **{s['mae_total']:.2f}** goals
- Brier (Over 2.5): **{s['brier_over25']:.3f}** (0.25 = coin-flip)

## What this means

Pinnacle was slightly sharper than the model on this weekend. The failure mode to watch is **strong model Unders that fight Pin Overs by ~15–20pp** (Sporting, Willem II, Excelsior). Those are now flagged as **CONFLICT** in the Score Predictions table and ranked lower.

Score Predictions stay an **information tool**. Live betting still comes only from the 6 protected systems.

Source: [`experiments/weekend_retro/REPORT.md`](../weekend_retro/REPORT.md)
"""


def main() -> int:
    p = argparse.ArgumentParser(description="Score Predictions calibration report")
    p.add_argument("--csv", type=Path, default=RETRO)
    args = p.parse_args()
    if not args.csv.is_file():
        print(f"Missing {args.csv}")
        return 1
    df = pd.read_csv(args.csv)
    stats = summarize(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = render(stats)
    OUT_MD.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
