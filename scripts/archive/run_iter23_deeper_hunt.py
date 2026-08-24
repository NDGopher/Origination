#!/usr/bin/env python
"""
Iteration 23 — deeper OU / 1X2 / AH hunt on expansion leagues.

Protected systems NEVER modified.
Uses existing iter22 walk-forward predictions (no re-WF unless missing).
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

OUT = ROOT / "experiments" / "iter23"
OUT.mkdir(parents=True, exist_ok=True)

LEAGUES = ["Ligue1", "Eredivisie", "PrimeiraLiga", "Belgium"]

UNDER_BANDS = [
    (1.50, 2.20),
    (1.60, 2.40),
    (1.70, 2.50),
    (1.80, 2.50),
    (1.80, 3.00),
    (2.00, 3.00),
    (2.00, 4.00),
    (2.20, 3.50),
    (2.50, 4.00),
]
OVER_BANDS = [
    (1.40, 2.00),
    (1.50, 2.20),
    (1.60, 2.50),
    (1.70, 2.80),
    (1.80, 3.00),
    (2.00, 3.50),
]
EDGES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18]
ML_EDGES = [0.03, 0.05, 0.08, 0.10, 0.12]
ML_MAX = [1.50, 1.60, 1.70, 1.80, 2.00, 2.20, 2.50]
AH_EDGES = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
AH_MAX = [1.70, 1.80, 1.90, 2.00, 2.20, None]


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


def max_dd(bets: pd.DataFrame) -> float | None:
    if bets is None or len(bets) == 0:
        return None
    b = bets.sort_values("date") if "date" in bets.columns else bets
    equity = b["profit"].astype(float).cumsum()
    peak = equity.cummax()
    dd = equity - peak
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
    return {
        "n": n,
        "roi": float(bets["profit"].sum() / bets["stake"].sum()),
        "hit": float((bets["won"].astype(float) >= 0.5).mean()),
        "t_stat": float(t) if np.isfinite(t) else None,
        "p_value": p if np.isfinite(p) else None,
        "units": float(bets["profit"].sum()),
        "seasons_pos": spos,
        "seasons_n": sn,
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
        (ROOT / "experiments").glob(f"*iter22_{key}*/predictions.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def hunt(preds: pd.DataFrame, matches: pd.DataFrame, league: str) -> list[dict]:
    uni = evaluate_predictions(preds, matches, _bt(0.03, ["ou25", "ah", "1x2"]), edge_threshold=0.03)
    rows = []
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
                if s.get("n", 0) < 60:
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


def main() -> None:
    setup_logging("ERROR")
    hunt_rows: list[dict] = []
    pred_map: dict[str, tuple[Path, Path]] = {}

    for key in LEAGUES:
        info = get_league(key)
        aligned = ROOT / "data" / "interim" / info["aligned"]
        pred = find_preds(key)
        if not aligned.exists() or pred is None:
            print(f"SKIP {key}: aligned={aligned.exists()} preds={pred}")
            continue
        pred_map[key] = (pred, aligned)
        print(f"HUNT {key} preds={pred.parent.name}", flush=True)
        preds = pd.read_parquet(pred)
        matches = load_aligned(aligned)
        # Prefer Pin closes when present for AH re-score quality note
        pin_share = 0.0
        if "pin_close_ahh" in matches.columns:
            pin_share = float(matches["pin_close_ahh"].notna().mean())
        print(f"  pin_ah_close_coverage={100*pin_share:.0f}%", flush=True)
        hunt_rows.extend(hunt(preds, matches, key))

    grid = pd.DataFrame(hunt_rows)
    grid.to_csv(OUT / "full_grid.csv", index=False)

    if len(grid) == 0:
        print("No hunt rows")
        return

    short = grid[
        (grid["n"] >= 100)
        & (grid["roi"] >= 0.04)
        & (grid["seasons_n"] >= 6)
        & (grid["seasons_pos"] >= (grid["seasons_n"] // 2 + 1))
        & (grid["t_stat"].fillna(0) >= 1.6)
    ].copy()

    boot_rows = []
    for _, row in short.iterrows():
        key = row["league"]
        if key not in pred_map:
            continue
        pred_path, aligned_path = pred_map[key]
        preds = pd.read_parquet(pred_path)
        matches = load_aligned(aligned_path)
        uni = evaluate_predictions(
            preds, matches, _bt(0.03, ["ou25", "ah", "1x2"]), edge_threshold=0.03
        )
        if row["market"] == "ou25":
            bets = uni[
                (uni["market"] == "ou25")
                & (uni["side"] == row["side"])
                & (uni["close_odds"] >= float(row["min_odds"]))
                & (uni["close_odds"] <= float(row["max_odds"]))
                & (uni["edge"] >= float(row["edge"]))
            ]
        elif row["market"] == "ah":
            bets = uni[(uni["market"] == "ah") & (uni["edge"] >= float(row["edge"]))]
            if not pd.isna(row["max_odds"]):
                bets = bets[bets["close_odds"] <= float(row["max_odds"])]
        else:
            bets = uni[
                (uni["market"] == "1x2")
                & (uni["edge"] >= float(row["edge"]))
                & (uni["close_odds"] <= float(row["max_odds"]))
            ]
            if row["side"] != "all":
                bets = bets[bets["side"].isin(list(row["side"]))]
        if len(bets) < 80:
            continue
        b = bootstrap_roi(bets)
        boot_rows.append({**row.to_dict(), **b, "max_dd_u": max_dd(bets)})

    boot = pd.DataFrame(boot_rows)
    if len(boot):
        prom = boot[
            (boot["boot_ci95_lo"].fillna(-1) > -0.03)
            & (boot["pct_boot_pos"].fillna(0) >= 0.88)
            & (boot["t_stat"].fillna(0) >= 1.8)
        ].sort_values(["roi", "t_stat"], ascending=False)
    else:
        prom = pd.DataFrame()
    boot.to_csv(OUT / "bootstrap_shortlist.csv", index=False)
    prom.to_csv(OUT / "promotion_candidates.csv", index=False)

    # Primeira AH protected paper re-check
    lines = [
        "# Iteration 23 — Deeper hunt (expansion leagues)",
        "",
        "Protected systems untouched (5 live + Primeira AH paper).",
        "",
        f"## Promotion candidates: **{len(prom)}**",
        "",
    ]
    if len(prom) == 0:
        lines.append("_None cleared the stricter iter23 bar (t≥1.8, boot+≥88%, CI lo > −3%)._")
    else:
        lines.append("| Tag | n | ROI | t | Boot CI lo | Seasons+ | Max DD |")
        lines.append("|-----|--:|----:|--:|-----------:|---------:|-------:|")
        for _, r in prom.head(20).iterrows():
            dd = r.get("max_dd_u")
            dd_s = f"{dd:+.1f}u" if dd is not None and pd.notna(dd) else "—"
            lines.append(
                f"| `{r['tag']}` | {int(r['n'])} | {100*r['roi']:+.1f}% | "
                f"{r['t_stat']:.2f} | {100*r['boot_ci95_lo']:+.1f}% | "
                f"{int(r['seasons_pos'])}/{int(r['seasons_n'])} | {dd_s} |"
            )

    # Explicit Primeira AH paper metrics refresh
    lines += ["", "## Primeira AH paper (protected) — refresh", ""]
    if "PrimeiraLiga" in pred_map:
        pred_path, aligned_path = pred_map["PrimeiraLiga"]
        preds = pd.read_parquet(pred_path)
        matches = load_aligned(aligned_path)
        uni = evaluate_predictions(
            preds, matches, _bt(0.03, ["ah"]), edge_threshold=0.03
        )
        bets = uni[
            (uni["market"] == "ah")
            & (uni["edge"] >= 0.10)
            & (uni["close_odds"] <= 1.90)
        ]
        s = sig_fast(bets)
        b = bootstrap_roi(bets) if len(bets) >= 50 else {}
        lines.append(
            f"- Rules frozen: AH @ e≥10% max 1.90 · n={s.get('n')} · "
            f"ROI={100*s.get('roi',0):+.1f}% · t={s.get('t_stat')} · "
            f"seasons+={s.get('seasons_pos')}/{s.get('seasons_n')} · "
            f"maxDD={s.get('max_dd_u')} · "
            f"boot CI lo={100*b.get('boot_ci95_lo', float('nan')):+.1f}% · "
            f"boot+={100*b.get('pct_boot_pos', 0):.0f}%"
        )
        lines.append(
            "- Historical closes prefer Pinnacle (PAHH/PAHA) when present; "
            "live scan uses Pin guest AH main line."
        )
        pd.DataFrame([{**s, **b, "tag": "PrimeiraLiga_ah_short_exp"}]).to_csv(
            OUT / "primeira_ah_refresh.csv", index=False
        )

    lines += [
        "",
        "## Status",
        "",
        "| System | Status |",
        "|--------|--------|",
        "| 5 protected live | Unchanged |",
        "| Primeira AH short | Paper (live-scannable with Pin AH) |",
        "| New iter23 candidates | See above |",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "MASTER_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT / "REPORT.md")
    print(f"Candidates: {len(prom)}")


if __name__ == "__main__":
    main()
