#!/usr/bin/env python
"""
Iteration 28 — careful hunt on unused leagues (live packs frozen).

Uses seasonal Dixon–Coles + intercept + temperature (same as Score Predictions
--fast). This is NOT the full residual walk-forward used to promote the 6 live
systems. Anything that clears the bar here is a watch list only until a full
WF with residual is run.

Protected packs are not modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.features.store import build_feature_matrix
from origination.models.calibration import build_calibrator
from origination.models.poisson import apply_totals_intercept, build_model
from origination.utils import load_config, resolve_data_dir, setup_logging
from origination.utils.league_registry import get_league
from origination.utils.system_registry import live_systems

OUT = ROOT / "experiments" / "iter28"
HUNT_LEAGUES = ["Eredivisie", "Belgium", "Scotland", "Championship", "Ligue1"]

UNDER_BANDS = [(1.50, 2.20), (1.60, 2.40), (1.70, 2.50), (1.80, 2.50), (2.00, 4.00)]
OVER_BANDS = [(1.40, 2.00), (1.50, 2.20), (1.60, 2.50), (1.70, 2.80)]
EDGES = [0.05, 0.08, 0.10, 0.12]
ML_EDGES = [0.03, 0.05, 0.08, 0.10]
ML_MAX = [1.80, 2.00, 2.20]
AH_EDGES = [0.08, 0.10, 0.12]
AH_MAX = [1.90, 2.00]


def _bt(edge: float, markets: list[str]) -> dict:
    return {
        "markets": markets,
        "edge_threshold": edge,
        "edge_threshold_by_market": {m: edge for m in markets},
        "bet_filters": {"enabled": False, "rules": []},
        "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
        "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
    }


def season_pos(bets: pd.DataFrame) -> tuple[int, int]:
    if bets is None or len(bets) == 0 or "season" not in bets.columns:
        return 0, 0
    pos = n = 0
    for _, g in bets.groupby("season"):
        st = float(g["stake"].sum())
        if st <= 0:
            continue
        n += 1
        if float(g["profit"].sum()) / st > 0:
            pos += 1
    return pos, n


def last3_pos(bets: pd.DataFrame) -> tuple[int, int]:
    if bets is None or len(bets) == 0 or "season" not in bets.columns:
        return 0, 0
    seasons = sorted(bets["season"].dropna().unique())[-3:]
    pos = n = 0
    for s in seasons:
        g = bets[bets["season"] == s]
        st = float(g["stake"].sum())
        if st <= 0:
            continue
        n += 1
        if float(g["profit"].sum()) / st > 0:
            pos += 1
    return pos, n


def max_dd(bets: pd.DataFrame) -> float | None:
    if bets is None or len(bets) == 0:
        return None
    b = bets.sort_values("date") if "date" in bets.columns else bets
    equity = b["profit"].astype(float).cumsum()
    dd = equity - equity.cummax()
    return float(dd.min()) if len(dd) else None


def sig_fast(bets: pd.DataFrame) -> dict:
    if bets is None or len(bets) < 40:
        return {"n": 0}
    r = (bets["profit"] / bets["stake"].replace(0, np.nan)).astype(float).dropna()
    n = len(r)
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else np.nan
    t = mean / se if n > 1 and se > 0 else np.nan
    spos, sn = season_pos(bets)
    l3p, l3n = last3_pos(bets)
    return {
        "n": n,
        "roi": float(bets["profit"].sum() / bets["stake"].sum()),
        "hit": float((bets["won"].astype(float) >= 0.5).mean()),
        "t_stat": float(t) if np.isfinite(t) else None,
        "units": float(bets["profit"].sum()),
        "seasons_pos": spos,
        "seasons_n": sn,
        "last3_pos": l3p,
        "last3_n": l3n,
        "max_dd_u": max_dd(bets),
    }


def bootstrap_roi(bets: pd.DataFrame, n_boot: int = 1500) -> dict:
    profits = bets["profit"].astype(float).values
    stakes = bets["stake"].astype(float).values
    n = len(profits)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        st = stakes[idx].sum()
        if st > 0:
            boots.append(profits[idx].sum() / st)
    boots = np.array(boots)
    return {
        "boot_ci95_lo": float(np.quantile(boots, 0.025)),
        "boot_ci95_hi": float(np.quantile(boots, 0.975)),
        "pct_boot_pos": float((boots > 0).mean()),
    }


def dc_oos(key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    info = get_league(key)
    cfg = load_config(ROOT / info["config"])
    data_dir = resolve_data_dir(cfg)
    matches = load_aligned(data_dir / "interim" / info["aligned"])
    matches = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    matches["date"] = pd.to_datetime(matches["date"])
    print(f"  features [{key}]...", flush=True)
    feat = build_feature_matrix(matches, cfg.get("features", {}))
    seasons = sorted(int(s) for s in matches["season"].dropna().unique() if int(s) < 2026)
    frames = []
    for season in seasons:
        train = matches[matches["season"] < season]
        test = matches[matches["season"] == season]
        if len(train) < 400 or len(test) < 40:
            continue
        print(f"    fit {season} train={len(train)} test={len(test)}", flush=True)
        model = build_model(cfg)
        model.fit(train)
        apply_totals_intercept(model, train, feat[feat["match_id"].isin(train["match_id"])], cfg)
        cal = build_calibrator(cfg)
        seas = sorted(train["season"].dropna().unique())
        calib = train[train["season"] == seas[-1]] if len(seas) >= 3 else train
        if len(calib) >= 80:
            raw_c = model.predict_dataframe(calib, features=feat[feat["match_id"].isin(calib["match_id"])])
            cal.fit(raw_c, calib)
        raw = model.predict_dataframe(test, features=feat[feat["match_id"].isin(test["match_id"])])
        frames.append(cal.transform(raw))
    if not frames:
        return pd.DataFrame(), matches
    return pd.concat(frames, ignore_index=True), matches


def hunt(preds: pd.DataFrame, matches: pd.DataFrame, league: str) -> list[dict]:
    uni = evaluate_predictions(preds, matches, _bt(0.03, ["ou25", "ah", "1x2"]), edge_threshold=0.03)
    rows: list[dict] = []
    ou = uni[uni["market"] == "ou25"]
    ah = uni[uni["market"] == "ah"]
    ml = uni[uni["market"] == "1x2"]
    for side, bands in [("under", UNDER_BANDS), ("over", OVER_BANDS)]:
        base = ou[ou["side"] == side]
        for lo, hi in bands:
            band = base[(base["close_odds"] >= lo) & (base["close_odds"] <= hi)]
            for e in EDGES:
                bets = band[band["edge"] >= e]
                s = sig_fast(bets)
                if s.get("n", 0) < 80:
                    continue
                rows.append(
                    {
                        "league": league,
                        "market": "ou25",
                        "side": side,
                        "min_odds": lo,
                        "max_odds": hi,
                        "edge": e,
                        **s,
                        "tag": f"{league}_ou25_{side}_{lo}-{hi}_e{int(e*100)}",
                    }
                )
    for e in AH_EDGES:
        for max_o in AH_MAX:
            bets = ah[(ah["edge"] >= e) & (ah["close_odds"] <= max_o)]
            s = sig_fast(bets)
            if s.get("n", 0) < 80:
                continue
            rows.append(
                {
                    "league": league,
                    "market": "ah",
                    "side": "best",
                    "min_odds": 1.01,
                    "max_odds": max_o,
                    "edge": e,
                    **s,
                    "tag": f"{league}_ah_e{int(e*100)}_max{max_o}",
                }
            )
    for side in ("H", "A"):
        base = ml[ml["side"] == side]
        for e in ML_EDGES:
            for max_o in ML_MAX:
                bets = base[(base["edge"] >= e) & (base["close_odds"] <= max_o)]
                s = sig_fast(bets)
                if s.get("n", 0) < 80:
                    continue
                rows.append(
                    {
                        "league": league,
                        "market": "1x2",
                        "side": side,
                        "min_odds": 1.01,
                        "max_odds": max_o,
                        "edge": e,
                        **s,
                        "tag": f"{league}_1x2_{side}_e{int(e*100)}_max{max_o}",
                    }
                )
    return rows


def clears_bar(r: dict) -> bool:
    if (r.get("n") or 0) < 120:
        return False
    if (r.get("t_stat") or 0) < 2.0:
        return False
    sn = r.get("seasons_n") or 0
    sp = r.get("seasons_pos") or 0
    if sn < 8 or (sp / sn) < 0.70:
        return False
    if (r.get("last3_n") or 0) < 3 or (r.get("last3_pos") or 0) < 2:
        return False
    if (r.get("boot_ci95_lo") or -1) <= 0:
        return False
    if (r.get("pct_boot_pos") or 0) < 0.95:
        return False
    if (r.get("max_dd_u") or -99) <= -8.0:
        return False
    if (r.get("roi") or 0) < 0.05:
        return False
    return True


def main() -> int:
    setup_logging("WARNING")
    OUT.mkdir(parents=True, exist_ok=True)
    live_packs = [s["pack"] for s in live_systems()]
    lines = [
        "# Iteration 28 — unused-league hunt",
        "",
        "Method: seasonal Dixon–Coles + intercept + temperature (**no residual**).",
        "Live packs frozen: " + ", ".join(live_packs),
        "",
        "A cell that clears the bar here is still **not** wired — it needs a full residual walk-forward first.",
        "",
    ]
    all_rows: list[dict] = []
    for key in HUNT_LEAGUES:
        print(f"\n=== Hunt {key} ===", flush=True)
        try:
            preds, matches = dc_oos(key)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"## {key}: SKIP — {exc}")
            lines.append("")
            continue
        if len(preds) == 0:
            lines.append(f"## {key}: SKIP — no OOS preds")
            lines.append("")
            continue
        preds.to_parquet(OUT / f"preds_{key}.parquet", index=False)
        rows = hunt(preds, matches, key)
        enriched = []
        for r in rows:
            if (r.get("n") or 0) < 100 or (r.get("t_stat") or 0) < 1.5:
                enriched.append(r)
                continue
            uni = evaluate_predictions(
                preds, matches, _bt(r["edge"], [r["market"]]), edge_threshold=r["edge"]
            )
            bets = uni[uni["market"] == r["market"]]
            if r["market"] == "ou25":
                bets = bets[
                    (bets["side"] == r["side"])
                    & (bets["close_odds"] >= r["min_odds"])
                    & (bets["close_odds"] <= r["max_odds"])
                    & (bets["edge"] >= r["edge"])
                ]
            elif r["market"] == "1x2":
                bets = bets[
                    (bets["side"] == r["side"])
                    & (bets["close_odds"] <= r["max_odds"])
                    & (bets["edge"] >= r["edge"])
                ]
            else:
                bets = bets[(bets["close_odds"] <= r["max_odds"]) & (bets["edge"] >= r["edge"])]
            if len(bets) >= 80:
                r.update(bootstrap_roi(bets))
            enriched.append(r)
        enriched.sort(key=lambda x: (x.get("t_stat") or 0), reverse=True)
        all_rows.extend(enriched)
        lines.append(f"## {key} (DC seasonal OOS, n_pred={len(preds)})")
        lines.append("")
        lines.append("| tag | n | ROI | t | seasons+ | DD | boot lo | clear? |")
        lines.append("|-----|--:|----:|--:|---------:|---:|--------:|:------:|")
        for r in enriched[:8]:
            clear = "YES" if clears_bar(r) else "no"
            lo = r.get("boot_ci95_lo")
            lo_s = f"{100*lo:.1f}%" if lo is not None else "—"
            dd = r.get("max_dd_u")
            dd_s = f"{dd:.1f}u" if dd is not None else "—"
            lines.append(
                f"| `{r['tag']}` | {r['n']} | {100*(r.get('roi') or 0):+.1f}% | "
                f"{r.get('t_stat') or 0:.2f} | {r.get('seasons_pos')}/{r.get('seasons_n')} | "
                f"{dd_s} | {lo_s} | {clear} |"
            )
        lines.append("")

    cleared = [r for r in all_rows if clears_bar(r)]
    lines += [
        "## Cleared promotion bar",
        "",
        f"Count: **{len(cleared)}** (DC-only; still do not wire without residual WF).",
        "",
    ]
    if not cleared:
        lines.append("Nothing cleared. Closest cells are in the per-league tables above.")
        lines.append("")
    else:
        for r in cleared:
            lines.append(f"- `{r['tag']}` n={r['n']} ROI={100*r['roi']:+.1f}% t={r['t_stat']:.2f}")
        lines.append("")
    lines += [
        "## Recommendations",
        "",
        "- Do not change the 6 live packs.",
        "- Score Predictions stay an information tool.",
        "- Scotland Over 1.4–2.0 @ e5 remains the iter26 watch (DD failed on full WF).",
        "",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    if all_rows:
        pd.DataFrame(all_rows).to_csv(OUT / "cells.csv", index=False)
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Wrote {OUT / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
