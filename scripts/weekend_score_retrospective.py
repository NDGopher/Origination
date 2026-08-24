#!/usr/bin/env python
"""Join Friday Score Predictions snapshot to weekend actuals. Information only."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "experiments" / "gameday_scan" / "SCORE_PREDICTIONS_20260814.csv"
ACT = ROOT / "experiments" / "weekend_retro" / "actuals.csv"
OUT = ROOT / "experiments" / "weekend_retro"


def main() -> int:
    pred = pd.read_csv(SNAP)
    act = pd.read_csv(ACT)
    m = pred.merge(act, on="match_id", how="inner", suffixes=("", "_act"))
    m = m.dropna(subset=["actual_home", "actual_away"]).copy()
    m["actual_total"] = m["actual_home"] + m["actual_away"]
    m["actual_ou"] = m["actual_total"].map(lambda t: "OVER" if t > 2.5 else "UNDER")
    m["lean_hit"] = m["lean"] == m["actual_ou"]
    m["total_error"] = m["actual_total"] - m["proj_total"]
    m["abs_total_error"] = m["total_error"].abs()
    pin_over = pd.to_numeric(m.get("pin_over_pct"), errors="coerce")
    m["pin_lean"] = pin_over.map(lambda p: "OVER" if pd.notna(p) and p >= 50 else ("UNDER" if pd.notna(p) else None))
    m["pin_hit"] = m["pin_lean"] == m["actual_ou"]
    m["conflict_20pp"] = m["model_minus_pin_under_pp"].abs() >= 20
    cols = [
        "league",
        "match",
        "data_grade",
        "data_strength",
        "proj_score",
        "proj_total",
        "over_pct",
        "under_pct",
        "lean",
        "lean_pp",
        "pin_over_pct",
        "pin_under_pct",
        "model_minus_pin_over_pp",
        "model_minus_pin_under_pp",
        "actual_home",
        "actual_away",
        "actual_total",
        "actual_ou",
        "lean_hit",
        "pin_lean",
        "pin_hit",
        "total_error",
        "conflict_20pp",
        "notes",
    ]
    keep = [c for c in cols if c in m.columns]
    out = m[keep].sort_values(["data_grade", "league", "match"])
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "prediction_vs_actual.csv"
    out.to_csv(csv_path, index=False)

    n = len(out)
    hits = int(out["lean_hit"].sum())
    pin_known = out.dropna(subset=["pin_lean"])
    print(f"n={n}  model_lean_hit={hits}/{n} ({100*hits/n:.0f}%)", flush=True)
    if len(pin_known):
        ph = int(pin_known["pin_hit"].sum())
        print(f"pin_lean_hit={ph}/{len(pin_known)} ({100*ph/len(pin_known):.0f}%)", flush=True)
    conflict = out[out["conflict_20pp"] == True]
    if len(conflict):
        ch = int(conflict["lean_hit"].sum())
        print(f"model-pin |gap|>=20pp: model hit {ch}/{len(conflict)}", flush=True)
        aligned = out[out["conflict_20pp"] != True]
        ah = int(aligned["lean_hit"].sum())
        print(f"aligned (<20pp): model hit {ah}/{len(aligned)}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
