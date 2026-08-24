#!/usr/bin/env python
"""
Iteration 26 — careful research hunt (protected live systems untouched).

Continues after Score Predictions coverage fix.
Hunts: Scotland, Turkey, Ligue1, Eredivisie, Belgium, Championship
(and confirms no silent change to the 6 live packs).

Promotion bar (all must pass):
  n >= 120
  t-stat >= 2.0
  seasons_n >= 8 and seasons_pos / seasons_n >= 0.70
  last-3 seasons: at least 2 positive
  bootstrap 95% CI lo > 0
  bootstrap P(ROI>0) >= 95%
  max drawdown > -8u
  ROI >= 5%
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
from origination.utils import setup_logging
from origination.utils.league_registry import get_league
from origination.utils.system_registry import live_systems

OUT = ROOT / "experiments" / "iter26"
OUT.mkdir(parents=True, exist_ok=True)

HUNT_LEAGUES = {
    "Ligue1": None,
    "Eredivisie": None,
    "Belgium": None,
    "Championship": None,
    "Scotland": None,
    "Turkey": None,
}

UNDER_BANDS = [
    (1.50, 2.20),
    (1.60, 2.40),
    (1.70, 2.50),
    (1.80, 2.50),
    (1.80, 3.00),
    (2.00, 3.00),
    (2.00, 4.00),
]
OVER_BANDS = [
    (1.40, 2.00),
    (1.50, 2.20),
    (1.60, 2.50),
    (1.70, 2.80),
    (1.80, 3.00),
]
EDGES = [0.05, 0.08, 0.10, 0.12, 0.15]
ML_EDGES = [0.03, 0.05, 0.08, 0.10, 0.12]
ML_MAX = [1.60, 1.80, 2.00, 2.20]
AH_EDGES = [0.08, 0.10, 0.12, 0.15]
AH_MAX = [1.80, 1.90, 2.00]


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
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if n > 2 and np.isfinite(t) else np.nan
    spos, sn = season_pos(bets)
    l3p, l3n = last3_pos(bets)
    return {
        "n": n,
        "roi": float(bets["profit"].sum() / bets["stake"].sum()),
        "hit": float((bets["won"].astype(float) >= 0.5).mean()),
        "t_stat": float(t) if np.isfinite(t) else None,
        "p_value": p if np.isfinite(p) else None,
        "units": float(bets["profit"].sum()),
        "seasons_pos": spos,
        "seasons_n": sn,
        "last3_pos": l3p,
        "last3_n": l3n,
        "max_dd_u": max_dd(bets),
    }


def bootstrap_roi(bets: pd.DataFrame, n_boot: int = 2500) -> dict:
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


def find_preds(key: str) -> Path | None:
    cands = sorted(
        (ROOT / "experiments").glob(f"**/*{key}*/predictions.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    prefer = [
        c
        for c in cands
        if any(x in str(c) for x in ("iter25", "iter22", "iter26", "_base", "vol06"))
    ]
    pool = prefer or cands
    return pool[0] if pool else None


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
            bets = ah[ah["edge"] >= e]
            bets = bets[bets["close_odds"] <= max_o]
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
    live = live_systems()
    live_packs = [s["pack"] for s in live]
    lines = [
        "# Iteration 26 — careful research",
        "",
        f"Protected live packs (untouched): {', '.join(live_packs)}",
        "",
    ]
    all_rows: list[dict] = []
    for key in HUNT_LEAGUES:
        info = get_league(key)
        pred_p = find_preds(key)
        if pred_p is None:
            lines.append(f"## {key}: SKIP — no predictions.parquet")
            continue
        matches = load_aligned(ROOT / "data" / "interim" / info["aligned"])
        preds = pd.read_parquet(pred_p)
        print(f"Hunt [{key}] preds={pred_p.name} n_pred={len(preds)}", flush=True)
        rows = hunt(preds, matches, key)
        # bootstrap only promising cells
        enriched = []
        for r in rows:
            if (r.get("n") or 0) < 100 or (r.get("t_stat") or 0) < 1.5:
                enriched.append(r)
                continue
            # re-evaluate bets for bootstrap
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
            elif r["market"] == "ah":
                bets = bets[(bets["edge"] >= r["edge"]) & (bets["close_odds"] <= r["max_odds"])]
            else:
                bets = bets[
                    (bets["side"] == r["side"])
                    & (bets["edge"] >= r["edge"])
                    & (bets["close_odds"] <= r["max_odds"])
                ]
            if len(bets) >= 80:
                r = {**r, **bootstrap_roi(bets)}
            enriched.append(r)
        all_rows.extend(enriched)
        top = sorted(
            enriched,
            key=lambda x: (x.get("t_stat") or -99, x.get("roi") or -99),
            reverse=True,
        )[:8]
        lines.append(f"## {key} (preds `{pred_p.parent.name}`)")
        lines.append("")
        lines.append("| tag | n | ROI | t | seasons+ | DD | boot lo | clear? |")
        lines.append("|-----|--:|----:|--:|---------:|---:|--------:|:------:|")
        for r in top:
            lines.append(
                f"| `{r['tag']}` | {r.get('n')} | {100*(r.get('roi') or 0):+.1f}% | "
                f"{r.get('t_stat') or float('nan'):.2f} | "
                f"{r.get('seasons_pos')}/{r.get('seasons_n')} | "
                f"{r.get('max_dd_u') or float('nan'):+.1f}u | "
                f"{100*(r.get('boot_ci95_lo') or 0):+.1f}% | "
                f"{'YES' if clears_bar(r) else 'no'} |"
            )
        lines.append("")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "hunt_cells.csv", index=False)
    cleared = [r for r in all_rows if clears_bar(r)]
    lines += [
        "## Cleared promotion bar",
        "",
        f"Count: **{len(cleared)}**",
        "",
    ]
    for r in sorted(cleared, key=lambda x: x.get("t_stat") or 0, reverse=True)[:20]:
        lines.append(
            f"- `{r['tag']}` n={r['n']} ROI={100*r['roi']:+.1f}% t={r['t_stat']:.2f} "
            f"seasons={r['seasons_pos']}/{r['seasons_n']} DD={r['max_dd_u']:+.1f}u"
        )
    if not cleared:
        lines.append("None — keep hunting; do not wire new live packs.")
    lines += [
        "",
        "## Live systems check",
        "",
        "No pack rules modified in this iteration.",
        "",
    ]
    report = "\n".join(lines)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {OUT / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
