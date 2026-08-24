#!/usr/bin/env python
"""
Historical Score Predictions test (information tool — not a live pack).

Mirrors the gameday --fast path: Dixon–Coles + totals intercept + temperature
calibration. Residual is skipped (Score Predictions rebuild uses --fast).

Seasonal walk-forward on the last 3 complete seasons. Does not change live packs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.data_ingestion.align import load_aligned
from origination.features.store import build_feature_matrix
from origination.models.calibration import build_calibrator
from origination.models.poisson import apply_totals_intercept, build_model
from origination.utils import load_config, resolve_data_dir, setup_logging
from origination.utils.league_registry import get_league
from origination.utils.odds import two_way_fair

OUT = ROOT / "experiments" / "score_predictions"
LEAGUES = [
    "EPL",
    "Bundesliga",
    "LaLiga",
    "SerieA",
    "Ligue1",
    "PrimeiraLiga",
    "Eredivisie",
    "Belgium",
    "Championship",
    "Scotland",
    "MLS",
]
GRADE = {
    "EPL": "A",
    "Bundesliga": "A",
    "LaLiga": "A",
    "SerieA": "A",
    "Ligue1": "A",
    "PrimeiraLiga": "B",
    "Eredivisie": "B",
    "Belgium": "B",
    "Scotland": "B",
    "Championship": "C",
    "MLS": "C",
}


def _pin_probs(over, under) -> tuple[float | None, float | None]:
    try:
        o, u = float(over), float(under)
    except (TypeError, ValueError):
        return None, None
    if not np.isfinite(o) or not np.isfinite(u) or o <= 1.0 or u <= 1.0:
        return None, None
    fo, fu = two_way_fair(o, u, method="power")
    return float(fo), float(fu)


def _features_path(key: str) -> Path:
    return ROOT / "data" / "processed" / f"features_{key}.parquet"


def load_league(key: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    info = get_league(key)
    cfg = load_config(ROOT / info["config"])
    data_dir = resolve_data_dir(cfg)
    matches = load_aligned(data_dir / "interim" / info["aligned"])
    matches = matches.dropna(subset=["home_goals", "away_goals", "home_team", "away_team"]).copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches["total_goals"] = matches["home_goals"] + matches["away_goals"]
    fp = _features_path(key)
    if fp.is_file():
        feat = pd.read_parquet(fp)
    else:
        print(f"    building features [{key}]...", flush=True)
        feat = build_feature_matrix(matches, cfg.get("features", {}))
    return matches, feat, cfg


def oos_predict(matches: pd.DataFrame, feat: pd.DataFrame, cfg: dict, test_seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in test_seasons:
        train = matches[matches["season"] < season]
        test = matches[matches["season"] == season]
        if len(train) < 400 or len(test) < 40:
            continue
        seasons = sorted(train["season"].dropna().unique())
        calib_seasons = set(seasons[-1:]) if len(seasons) >= 3 else set()
        fit_df = train[~train["season"].isin(calib_seasons)] if calib_seasons else train
        calib_df = train[train["season"].isin(calib_seasons)] if calib_seasons else train
        if len(fit_df) < 200:
            fit_df, calib_df = train, train

        model = build_model(cfg)
        model.fit(fit_df)
        feat_fit = feat[feat["match_id"].isin(fit_df["match_id"])]
        apply_totals_intercept(model, fit_df, feat_fit, cfg)

        cal = build_calibrator(cfg)
        if len(calib_df) >= 80:
            raw_c = model.predict_dataframe(
                calib_df, features=feat[feat["match_id"].isin(calib_df["match_id"])]
            )
            cal.fit(raw_c, calib_df)

        raw = model.predict_dataframe(test, features=feat[feat["match_id"].isin(test["match_id"])])
        preds = cal.transform(raw)
        keep = test[
            [
                c
                for c in (
                    "match_id",
                    "date",
                    "season",
                    "home_team",
                    "away_team",
                    "home_goals",
                    "away_goals",
                    "total_goals",
                    "close_over25",
                    "close_under25",
                )
                if c in test.columns
            ]
        ]
        frames.append(preds.merge(keep, on="match_id", how="left"))
        print(f"      season {season}: n={len(test)}", flush=True)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame, key: str) -> dict:
    n = len(df)
    tot = pd.to_numeric(df["total_goals"], errors="coerce")
    proj = pd.to_numeric(df["lambda_home"], errors="coerce") + pd.to_numeric(
        df["lambda_away"], errors="coerce"
    )
    p_o = pd.to_numeric(df["p_over25"], errors="coerce")
    actual_over = tot > 2.5
    lean_over = p_o >= 0.5
    model_hit = (lean_over == actual_over) & p_o.notna() & tot.notna()
    pin_o, pin_u = [], []
    for _, r in df.iterrows():
        fo, _fu = _pin_probs(r.get("close_over25"), r.get("close_under25"))
        pin_o.append(fo)
    pin_o = pd.Series(pin_o, index=df.index)
    pin_lean_over = pin_o >= 0.5
    pin_hit = (pin_lean_over == actual_over) & pin_o.notna() & tot.notna()
    gap = (p_o - pin_o) * 100
    brier = float(((p_o - actual_over.astype(float)) ** 2).mean()) if p_o.notna().any() else np.nan

    def rate(mask: pd.Series) -> dict:
        sub = df.loc[mask]
        if len(sub) == 0:
            return {"n": 0, "model": None, "pin": None, "bias": None, "mae": None}
        t = tot.loc[mask]
        pr = proj.loc[mask]
        mh = model_hit.loc[mask]
        ph = pin_hit.loc[mask]
        return {
            "n": int(len(sub)),
            "model": float(mh.mean()) if mh.notna().any() else None,
            "pin": float(ph.mean()) if ph.notna().any() else None,
            "bias": float((pr - t).mean()) if pr.notna().any() else None,
            "mae": float((pr - t).abs().mean()) if pr.notna().any() else None,
        }

    conflict15 = gap.abs() >= 15
    under_vs_pin_over = (p_o < 0.5) & (pin_o >= 0.5) & (gap.abs() >= 15)
    high = proj >= 3.15
    low = proj <= 2.25
    return {
        "league": key,
        "grade": GRADE.get(key, "?"),
        "n": n,
        "model_hit": float(model_hit.mean()) if n else None,
        "pin_hit": float(pin_hit.mean()) if n else None,
        "bias": float((proj - tot).mean()) if n else None,
        "mae": float((proj - tot).abs().mean()) if n else None,
        "brier": brier,
        "mean_proj": float(proj.mean()) if n else None,
        "mean_actual": float(tot.mean()) if n else None,
        "conflict15": rate(conflict15.fillna(False)),
        "aligned": rate((gap.abs() < 15).fillna(False)),
        "under_vs_pin_over": rate(under_vs_pin_over.fillna(False)),
        "high_proj": rate(high.fillna(False)),
        "low_proj": rate(low.fillna(False)),
        "gap_0_8": rate((gap.abs() < 8).fillna(False)),
        "gap_8_15": rate((gap.abs() >= 8) & (gap.abs() < 15)),
        "gap_15p": rate(conflict15.fillna(False)),
    }


def render(rows: list[dict], weekend: dict | None) -> str:
    def pct(x):
        return "—" if x is None else f"{100 * x:.1f}%"

    def num(x, nd=2):
        return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"

    lines = [
        "# Score Predictions — historical test",
        "",
        "Path: Dixon–Coles + totals intercept + temperature (same as gameday `--fast`).",
        "Test: last 3 complete seasons, seasonal walk-forward. **Not a live pack.**",
        "",
        "## By league",
        "",
        "| League | Grade | n | Model O/U | Pin O/U | Bias (proj−act) | MAE | Brier |",
        "|--------|:-----:|--:|----------:|--------:|----------------:|----:|------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['league']} | {r['grade']} | {r['n']} | {pct(r['model_hit'])} | "
            f"{pct(r['pin_hit'])} | {num(r['bias'])} | {num(r['mae'])} | {num(r['brier'], 3)} |"
        )

    # Pooled by grade
    lines += ["", "## By data grade (pooled)", ""]
    by_g: dict[str, list] = {}
    for r in rows:
        by_g.setdefault(r["grade"], []).append(r)
    lines.append("| Grade | n | Model O/U | Pin O/U | Bias |")
    lines.append("|:-----:|--:|----------:|--------:|-----:|")
    for g in ("A", "B", "C"):
        chunk = by_g.get(g, [])
        if not chunk:
            continue
        n = sum(x["n"] for x in chunk)
        mh = sum(x["model_hit"] * x["n"] for x in chunk if x["model_hit"] is not None) / n
        ph = sum(x["pin_hit"] * x["n"] for x in chunk if x["pin_hit"] is not None) / n
        bias = sum(x["bias"] * x["n"] for x in chunk if x["bias"] is not None) / n
        lines.append(f"| {g} | {n} | {pct(mh)} | {pct(ph)} | {num(bias)} |")

    lines += ["", "## Model−Pin gap (pooled, leagues with closing OU)", ""]
    lines.append("| Gap | n | Model O/U | Pin O/U |")
    lines.append("|-----|--:|----------:|--------:|")
    for label, key in [("0–8pp", "gap_0_8"), ("8–15pp", "gap_8_15"), ("≥15pp CONFLICT", "gap_15p")]:
        n = sum(r[key]["n"] for r in rows)
        if n == 0:
            continue
        mh = sum((r[key]["model"] or 0) * r[key]["n"] for r in rows) / n
        ph = sum((r[key]["pin"] or 0) * r[key]["n"] for r in rows) / n
        lines.append(f"| {label} | {n} | {pct(mh)} | {pct(ph)} |")

    lines += ["", "## High vs low projected totals", ""]
    lines.append("| Slice | n | Model O/U | Bias | MAE |")
    lines.append("|-------|--:|----------:|-----:|----:|")
    for label, key in [("proj ≥ 3.15 (HIGH)", "high_proj"), ("proj ≤ 2.25 (LOW)", "low_proj")]:
        n = sum(r[key]["n"] for r in rows)
        if n == 0:
            continue
        mh = sum((r[key]["model"] or 0) * r[key]["n"] for r in rows) / n
        bias = sum((r[key]["bias"] or 0) * r[key]["n"] for r in rows) / n
        mae = sum((r[key]["mae"] or 0) * r[key]["n"] for r in rows) / n
        lines.append(f"| {label} | {n} | {pct(mh)} | {num(bias)} | {num(mae)} |")

    lines += [
        "",
        "## Under vs Pin Over (≥15pp)",
        "",
    ]
    n = sum(r["under_vs_pin_over"]["n"] for r in rows)
    if n:
        mh = sum((r["under_vs_pin_over"]["model"] or 0) * r["under_vs_pin_over"]["n"] for r in rows) / n
        ph = sum((r["under_vs_pin_over"]["pin"] or 0) * r["under_vs_pin_over"]["n"] for r in rows) / n
        lines.append(f"n={n}  model hit {pct(mh)}  Pin hit {pct(ph)}.")
        lines.append("This is the weekend failure mode. Keep CONFLICT flags. Do not bet these from the Score tab.")
    else:
        lines.append("No rows.")

    if weekend:
        lines += [
            "",
            "## Weekend 14–16 Aug (n=23) — still the only live-slate grade",
            "",
            f"Model {weekend.get('model_hits')}/{weekend.get('n')} vs Pin {weekend.get('pin_hits')}/{weekend.get('n')}.",
            "Small sample; historical table above is the real test.",
        ]

    lines += [
        "",
        "## Recommendations",
        "",
        "- Do **not** promote Score Predictions to a live pack.",
        "- Pin is the better O/U lean when the gap is large; CONFLICT ranking stays.",
        "- Display drivers (form / xG / Elo) so a HIGH/LOW lean is explainable.",
        "- Any totals intercept / residual change for **live packs** needs a full walk-forward, not this script.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    setup_logging("WARNING")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    all_preds = []
    for key in LEAGUES:
        print(f"\n=== {key} ===", flush=True)
        try:
            matches, feat, cfg = load_league(key)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP: {exc}", flush=True)
            continue
        seasons = sorted(int(s) for s in matches["season"].dropna().unique())
        complete = [s for s in seasons if s < 2026]
        test = complete[-3:] if len(complete) >= 3 else complete[-2:]
        print(f"  test seasons {test}  hist={len(matches)}", flush=True)
        preds = oos_predict(matches, feat, cfg, test)
        if len(preds) == 0:
            print("  no OOS rows", flush=True)
            continue
        preds["league"] = key
        all_preds.append(preds)
        s = summarize(preds, key)
        rows.append(s)
        print(
            f"  model {s['model_hit']:.1%}  pin {s['pin_hit']:.1%}  "
            f"bias {s['bias']:+.2f}  n={s['n']}",
            flush=True,
        )

    weekend = None
    wpath = ROOT / "experiments" / "weekend_retro" / "prediction_vs_actual.csv"
    if wpath.is_file():
        w = pd.read_csv(wpath)
        weekend = {
            "n": int(len(w)),
            "model_hits": int(w["lean_hit"].astype(str).str.lower().isin(["true", "1"]).sum()),
            "pin_hits": int(w["pin_hit"].astype(str).str.lower().isin(["true", "1"]).sum()),
        }

    text = render(rows, weekend)
    (OUT / "HISTORICAL.md").write_text(text, encoding="utf-8")
    (OUT / "historical_by_league.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if all_preds:
        pd.concat(all_preds, ignore_index=True).to_parquet(OUT / "historical_oos.parquet", index=False)
    print(text)
    print(f"Wrote {OUT / 'HISTORICAL.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
