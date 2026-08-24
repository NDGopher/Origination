#!/usr/bin/env python
"""
Iteration 24 — careful multi-league hunt.

Protected live systems are NEVER modified.
Uses existing walk-forward predictions (no re-fit unless missing).

Promotion bar (all must pass):
  n >= 120
  t-stat >= 2.0
  seasons_n >= 8 and seasons_pos / seasons_n >= 0.70
  last-3 seasons: at least 2 positive (anti-overfit)
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

OUT = ROOT / "experiments" / "iter24"
OUT.mkdir(parents=True, exist_ok=True)

# Research expansion — not the 6 protected live packs
HUNT_LEAGUES = {
    "Ligue1": "20260812T182646Z_iter22_Ligue1_base",
    "Eredivisie": "20260812T184347Z_iter22_Eredivisie_base",
    "Belgium": "20260812T185650Z_iter22_Belgium_base",
    "Championship": "20260811T193046Z_iter20_Championship_shots_vol",
    "PrimeiraLiga": "20260812T185040Z_iter22_PrimeiraLiga_base",  # confirm sibling only
}

PROTECTED_TAGS = {
    "EPL_unders",
    "EPL_overs_short",
    "Bundesliga_unders",
    "LaLiga_home_ml",
    "SerieA_away_ml",
    "Primeira_ah_short",
}

UNDER_BANDS = [
    (1.50, 2.20),
    (1.60, 2.40),
    (1.70, 2.50),
    (1.80, 2.50),
    (1.80, 3.00),
    (2.00, 3.00),
    (2.00, 4.00),
    (2.20, 3.50),
]
OVER_BANDS = [
    (1.40, 2.00),
    (1.50, 2.20),
    (1.60, 2.50),
    (1.70, 2.80),
    (1.80, 3.00),
    (2.00, 3.50),
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
    named = HUNT_LEAGUES.get(key)
    if named:
        p = ROOT / "experiments" / named / "predictions.parquet"
        if p.is_file():
            return p
    cands = sorted(
        (ROOT / "experiments").glob(f"*iter22_{key}*/predictions.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


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
            if max_o is not None:
                bets = bets[bets["close_odds"] <= max_o]
            s = sig_fast(bets)
            if s.get("n", 0) < 80:
                continue
            rows.append(
                {
                    "league": league,
                    "market": "ah",
                    "side": "all",
                    "min_odds": None,
                    "max_odds": max_o,
                    "edge": e,
                    **s,
                    "tag": f"{league}_ah_e{int(e*100)}_max{max_o}",
                }
            )

    for e in ML_EDGES:
        for max_o in ML_MAX:
            for sides, stag in [(["H"], "H"), (["A"], "A"), (None, "all")]:
                bets = ml[(ml["edge"] >= e) & (ml["close_odds"] <= max_o)]
                if sides:
                    bets = bets[bets["side"].isin(sides)]
                s = sig_fast(bets)
                if s.get("n", 0) < 80:
                    continue
                rows.append(
                    {
                        "league": league,
                        "market": "1x2",
                        "side": stag,
                        "min_odds": None,
                        "max_odds": max_o,
                        "edge": e,
                        **s,
                        "tag": f"{league}_1x2_{stag}_e{int(e*100)}_max{max_o}",
                    }
                )
    return rows


def slice_bets(uni: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if row["market"] == "ou25":
        return uni[
            (uni["market"] == "ou25")
            & (uni["side"] == row["side"])
            & (uni["close_odds"] >= float(row["min_odds"]))
            & (uni["close_odds"] <= float(row["max_odds"]))
            & (uni["edge"] >= float(row["edge"]))
        ]
    if row["market"] == "ah":
        bets = uni[(uni["market"] == "ah") & (uni["edge"] >= float(row["edge"]))]
        if not pd.isna(row["max_odds"]):
            bets = bets[bets["close_odds"] <= float(row["max_odds"])]
        return bets
    bets = uni[
        (uni["market"] == "1x2")
        & (uni["edge"] >= float(row["edge"]))
        & (uni["close_odds"] <= float(row["max_odds"]))
    ]
    if row["side"] != "all":
        bets = bets[bets["side"].isin(list(row["side"]))]
    return bets


def main() -> None:
    setup_logging("ERROR")
    hunt_rows: list[dict] = []
    pred_map: dict[str, tuple[Path, Path]] = {}
    quality = []

    for key in HUNT_LEAGUES:
        info = get_league(key)
        aligned = ROOT / "data" / "interim" / info["aligned"]
        pred = find_preds(key)
        q = {
            "league": key,
            "aligned_exists": aligned.is_file(),
            "preds_exists": pred is not None and pred.is_file(),
            "n_aligned": 0,
            "has_xg": bool(info.get("understat")),
        }
        if aligned.is_file():
            try:
                q["n_aligned"] = int(len(pd.read_parquet(aligned, columns=["match_id"])))
            except Exception:  # noqa: BLE001
                q["n_aligned"] = 0
        quality.append(q)
        if not aligned.exists() or pred is None:
            print(f"SKIP {key}: aligned={aligned.exists()} preds={pred}", flush=True)
            continue
        pred_map[key] = (pred, aligned)
        print(f"HUNT {key} preds={pred.parent.name}", flush=True)
        preds = pd.read_parquet(pred)
        matches = load_aligned(aligned)
        hunt_rows.extend(hunt(preds, matches, key))

    grid = pd.DataFrame(hunt_rows)
    grid.to_csv(OUT / "full_grid.csv", index=False)
    pd.DataFrame(quality).to_csv(OUT / "data_quality.csv", index=False)
    print(f"grid rows={len(grid)}", flush=True)

    if len(grid) == 0:
        (OUT / "REPORT.md").write_text("# Iter24 — no hunt rows (missing preds)\n", encoding="utf-8")
        return

    short = grid[
        (grid["n"] >= 120)
        & (grid["roi"] >= 0.05)
        & (grid["seasons_n"] >= 8)
        & (grid["seasons_pos"] >= (grid["seasons_n"] * 0.70))
        & (grid["last3_n"].fillna(0) >= 2)
        & (grid["last3_pos"].fillna(0) >= 2)
        & (grid["t_stat"].fillna(0) >= 2.0)
        & (grid["max_dd_u"].fillna(-99) > -8.0)
    ].copy()
    short.to_csv(OUT / "shortlist.csv", index=False)
    print(f"shortlist={len(short)}", flush=True)

    boot_rows = []
    for _, row in short.iterrows():
        key = row["league"]
        if key not in pred_map:
            continue
        pred_path, aligned_path = pred_map[key]
        preds = pd.read_parquet(pred_path)
        matches = load_aligned(aligned_path)
        uni = evaluate_predictions(preds, matches, _bt(0.03, ["ou25", "ah", "1x2"]), edge_threshold=0.03)
        bets = slice_bets(uni, row)
        if len(bets) < 120:
            continue
        b = bootstrap_roi(bets)
        boot_rows.append({**row.to_dict(), **b, "max_dd_u": max_dd(bets)})

    boot = pd.DataFrame(boot_rows)
    if len(boot):
        prom = boot[
            (boot["boot_ci95_lo"].fillna(-1) > 0)
            & (boot["pct_boot_pos"].fillna(0) >= 0.95)
            & (boot["t_stat"].fillna(0) >= 2.0)
        ].sort_values(["t_stat", "roi"], ascending=False)
    else:
        prom = pd.DataFrame()
    boot.to_csv(OUT / "bootstrap_shortlist.csv", index=False)
    prom.to_csv(OUT / "promotion_candidates.csv", index=False)

    # Near-misses for transparency (t>=1.6, n>=100, not promoted)
    near = grid[
        (grid["n"] >= 100)
        & (grid["roi"] >= 0.04)
        & (grid["t_stat"].fillna(0) >= 1.6)
        & (grid["seasons_pos"].fillna(0) >= 6)
    ].sort_values("t_stat", ascending=False)
    near.to_csv(OUT / "near_misses.csv", index=False)

    lines = [
        "# Iteration 24 — Careful multi-league hunt",
        "",
        "Protected systems **untouched** (6 live/paper packs).",
        "",
        "## Promotion bar",
        "",
        "- Walk-forward only (existing OOS predictions)",
        "- n ≥ 120 · t ≥ 2.0 · seasons+ ≥ 70% of ≥8 seasons",
        "- Last 3 seasons: ≥2 positive",
        "- Bootstrap 95% CI lo > 0 · P(ROI>0) ≥ 95%",
        "- Max DD > −8u · ROI ≥ 5%",
        "",
        f"## New candidates that clear the bar: **{len(prom)}**",
        "",
    ]
    if len(prom) == 0:
        lines.append("_None. No new system is promoted to paper or live._")
    else:
        lines += [
            "| Tag | n | ROI | t | Boot CI lo | Seasons+ | Last3 | Max DD |",
            "|-----|--:|----:|--:|-----------:|---------:|------:|-------:|",
        ]
        for _, r in prom.head(15).iterrows():
            dd = r.get("max_dd_u")
            dd_s = f"{dd:+.1f}u" if dd is not None and pd.notna(dd) else "—"
            lines.append(
                f"| `{r['tag']}` | {int(r['n'])} | {100*r['roi']:+.1f}% | "
                f"{r['t_stat']:.2f} | {100*r['boot_ci95_lo']:+.1f}% | "
                f"{int(r['seasons_pos'])}/{int(r['seasons_n'])} | "
                f"{int(r['last3_pos'])}/{int(r['last3_n'])} | {dd_s} |"
            )
        lines.append("")
        lines.append(
            "Even if a cell clears the bar, it is **research only** until a separate "
            "paper-live decision. Protected rules stay frozen."
        )

    lines += ["", "## Near-misses (not promoted)", ""]
    if len(near) == 0:
        lines.append("_None worth listing._")
    else:
        lines += [
            "| Tag | n | ROI | t | Seasons+ | Why held |",
            "|-----|--:|----:|--:|---------:|----------|",
        ]
        shown = 0
        for _, r in near.head(12).iterrows():
            if len(prom) and r["tag"] in set(prom["tag"]):
                continue
            why = []
            if r.get("t_stat", 0) < 2.0:
                why.append("t<2.0")
            if r.get("n", 0) < 120:
                why.append("thin n")
            if r.get("last3_pos", 0) < 2:
                why.append("last-3 weak")
            if not why:
                why.append("failed bootstrap / DD / already paper")
            lines.append(
                f"| `{r['tag']}` | {int(r['n'])} | {100*r['roi']:+.1f}% | "
                f"{r['t_stat']:.2f} | {int(r['seasons_pos'])}/{int(r['seasons_n'])} | "
                f"{', '.join(why)} |"
            )
            shown += 1
            if shown >= 10:
                break

    lines += [
        "",
        "## Data freshness (season-long)",
        "",
        "Live path can stay fresh without re-downloading history:",
        "",
        "1. **Update Data Sources** — fixtures daily; optional current-season results (cache for old seasons)",
        "2. **Full Model Refresh** — force current-season FD + Understat, rebuild align + feature store + context",
        "3. **Update Odds** — Pinnacle OU / 1X2 / AH",
        "4. **Run Scan** — evaluate frozen systems only",
        "",
        "Aligned tables on disk:",
        "",
    ]
    for q in quality:
        xg = "xG yes" if q["has_xg"] else "goals-only"
        lines.append(
            f"- **{q['league']}**: aligned={q['n_aligned']} · {xg} · "
            f"preds={'yes' if q['preds_exists'] else 'NO'}"
        )

    lines += [
        "",
        "## Status",
        "",
        "| System | Status |",
        "|--------|--------|",
        "| EPL Unders / short Overs | Production — unchanged |",
        "| Bundesliga Unders / La Liga Home / Serie A Away | Paper live — unchanged |",
        "| Primeira AH short (e10) | Paper live — unchanged |",
        "| Primeira AH e12 sibling | Paper backtest only — not live |",
        "| New iter24 candidates | See above |",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "MASTER_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT / "REPORT.md", flush=True)
    print(f"PROMOTION_CANDIDATES={len(prom)}", flush=True)


if __name__ == "__main__":
    main()
