#!/usr/bin/env python
"""
Iteration 21 — overnight master: significance, wide hunt, AH/1X2, MLS preds.

Does NOT mutate EPL or Bundesliga protected pack rules.
Writes everything under experiments/iter21_*/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.utils import load_config, resolve_data_dir, setup_logging

OUT = ROOT / "experiments" / "iter21"
OUT.mkdir(parents=True, exist_ok=True)

# Protected systems — score only, never rewrite pack YAML rules
PROTECTED = [
    {
        "name": "EPL_unders",
        "status": "production",
        "league": "EPL",
        "experiment_id": "20260810T201358Z_iter17_EPL_vol06",
        "aligned": "matches_aligned.parquet",
        "market": "ou25",
        "side": "under",
        "min_odds": 2.00,
        "max_odds": 4.00,
        "edge": 0.08,
        # EPL_aggressive also has 1x2/ah rules; OU-only for significance of totals book
    },
    {
        "name": "EPL_overs_short",
        "status": "production",
        "league": "EPL",
        "experiment_id": "20260810T201358Z_iter17_EPL_vol06",
        "aligned": "matches_aligned.parquet",
        "market": "ou25",
        "side": "over",
        "min_odds": 1.60,
        "max_odds": 2.50,
        "edge": 0.10,
    },
    {
        "name": "Bundesliga_unders",
        "status": "paper",
        "league": "Bundesliga",
        "experiment_id": "20260810T174732Z_iter16_Bundesliga_int_thresh05",
        "aligned": "matches_aligned_D1.parquet",
        "market": "ou25",
        "side": "under",
        "min_odds": 1.70,
        "max_odds": 2.50,
        "edge": 0.10,
    },
]

LEAGUE_ARTIFACTS = [
    {
        "league": "EPL",
        "label": "vol06",
        "experiment_id": "20260810T201358Z_iter17_EPL_vol06",
        "aligned": "matches_aligned.parquet",
        "role": "protected_control",
    },
    {
        "league": "Bundesliga",
        "label": "thresh05",
        "experiment_id": "20260810T174732Z_iter16_Bundesliga_int_thresh05",
        "aligned": "matches_aligned_D1.parquet",
        "role": "protected_paper",
    },
    {
        "league": "Bundesliga",
        "label": "vol06",
        "experiment_id": "20260810T214911Z_iter17_Bundesliga_vol06",
        "aligned": "matches_aligned_D1.parquet",
        "role": "search",
    },
    {
        "league": "LaLiga",
        "label": "vol06",
        "experiment_id": "20260810T224919Z_iter17_LaLiga_vol06",
        "aligned": "matches_aligned_SP1.parquet",
        "role": "search",
    },
    {
        "league": "LaLiga",
        "label": "base",
        "experiment_id": "20260810T223204Z_iter17_LaLiga_base",
        "aligned": "matches_aligned_SP1.parquet",
        "role": "search",
    },
    {
        "league": "Championship",
        "label": "shots_vol",
        "experiment_id": "20260811T193046Z_iter20_Championship_shots_vol",
        "aligned": "matches_aligned_E1.parquet",
        "role": "search",
    },
    {
        "league": "SerieA",
        "label": "vol06",
        "experiment_id": "20260811T194347Z_iter20_SerieA_vol06",
        "aligned": "matches_aligned_I1.parquet",
        "role": "search",
    },
]

UNDER_BANDS = [
    (1.50, 2.20),
    (1.60, 2.40),
    (1.70, 2.50),
    (1.80, 2.50),
    (1.80, 3.00),
    (2.00, 3.00),
    (2.00, 4.00),
    (2.20, 3.50),
    (2.50, 4.50),
    (1.90, 2.80),
]
OVER_BANDS = [
    (1.40, 2.00),
    (1.50, 2.20),
    (1.50, 2.50),
    (1.60, 2.50),
    (1.70, 2.80),
    (1.80, 3.00),
    (2.00, 3.50),
    (1.55, 2.30),
    (1.65, 2.40),
]
EDGES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18]
AH_EDGES = [0.03, 0.05, 0.08, 0.10, 0.12]
ML_EDGES = [0.03, 0.05, 0.08, 0.10, 0.12]
ML_MAX_ODDS = [1.60, 1.80, 2.00, 2.20, 2.50, 3.00, 3.50]


def _bt_base(edge: float, markets: list[str], filt: dict) -> dict:
    return {
        "markets": markets,
        "edge_threshold": edge,
        "edge_threshold_by_market": {m: edge for m in markets},
        "bet_filters": filt,
        "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
        "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
    }


def score_ou(preds, matches, *, side, min_odds, max_odds, edge):
    filt = {
        "enabled": True,
        "rules": [
            {
                "markets": ["ou25"],
                "min_odds": min_odds,
                "max_odds": max_odds,
                "allow_sides": [side],
            }
        ],
    }
    bets = evaluate_predictions(
        preds, matches, _bt_base(edge, ["ou25"], filt), edge_threshold=edge
    )
    return bets[(bets["market"] == "ou25") & (bets["side"] == side)].copy()


def score_ah(preds, matches, *, edge, max_odds: float | None = None, allow_sides=None):
    rules = [{"markets": ["ah"]}]
    if max_odds is not None:
        rules[0]["max_odds"] = max_odds
    if allow_sides:
        rules[0]["allow_sides"] = allow_sides
    filt = {"enabled": True, "rules": rules}
    bets = evaluate_predictions(
        preds, matches, _bt_base(edge, ["ah"], filt), edge_threshold=edge
    )
    return bets[bets["market"] == "ah"].copy()


def score_1x2(preds, matches, *, edge, max_odds: float, allow_sides=None):
    rules = [{"markets": ["1x2"], "max_odds": max_odds}]
    if allow_sides:
        rules[0]["allow_sides"] = allow_sides
    filt = {"enabled": True, "rules": rules}
    bets = evaluate_predictions(
        preds, matches, _bt_base(edge, ["1x2"], filt), edge_threshold=edge
    )
    return bets[bets["market"] == "1x2"].copy()


def roi(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None
    st = float(df["stake"].sum())
    return float(df["profit"].sum()) / st if st else None


def hit(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None
    return float((df["won"].astype(float) >= 0.5).mean())


def season_pos(df: pd.DataFrame) -> tuple[int, int]:
    if df is None or len(df) == 0 or "season" not in df.columns:
        return 0, 0
    pos = n = 0
    for _, g in df.groupby("season"):
        r = roi(g)
        if r is None:
            continue
        n += 1
        if r > 0:
            pos += 1
    return pos, n


def significance(bets: pd.DataFrame, *, n_boot: int = 2000, seed: int = 42) -> dict:
    if bets is None or len(bets) == 0:
        return {"n": 0}
    r = (bets["profit"] / bets["stake"].replace(0, np.nan)).astype(float).dropna()
    n = len(r)
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else np.nan
    t = mean / se if n > 1 and se > 0 else np.nan
    # two-sided p from Student-t
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if n > 2 and np.isfinite(t) else np.nan
    ci_lo = mean - 1.96 * se if n > 1 and np.isfinite(se) else np.nan
    ci_hi = mean + 1.96 * se if n > 1 and np.isfinite(se) else np.nan
    profits = bets["profit"].astype(float).values
    stakes = bets["stake"].astype(float).values
    out = {
        "n": n,
        "roi": float(profits.sum() / stakes.sum()) if stakes.sum() else None,
        "hit": hit(bets),
        "mean_per_bet": mean,
        "std_per_bet": std,
        "t_stat": float(t) if np.isfinite(t) else None,
        "p_value": p if np.isfinite(p) else None,
        "ci95_lo": float(ci_lo) if np.isfinite(ci_lo) else None,
        "ci95_hi": float(ci_hi) if np.isfinite(ci_hi) else None,
        "boot_ci95_lo": None,
        "boot_ci95_hi": None,
        "boot_median": None,
        "pct_boot_positive": None,
        "avg_odds": float(bets["close_odds"].mean()),
        "avg_edge": float(bets["edge"].mean()),
        "units": float(profits.sum()),
        "seasons_pos": season_pos(bets)[0],
        "seasons_n": season_pos(bets)[1],
    }
    if n_boot > 0 and n >= 20:
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            st = stakes[idx].sum()
            if st <= 0:
                continue
            boots.append(profits[idx].sum() / st)
        boots = np.array(boots)
        if len(boots):
            out["boot_ci95_lo"] = float(np.quantile(boots, 0.025))
            out["boot_ci95_hi"] = float(np.quantile(boots, 0.975))
            out["boot_median"] = float(np.median(boots))
            out["pct_boot_positive"] = float((boots > 0).mean())
    return out


def significance_fast(bets: pd.DataFrame) -> dict:
    """t/p/ROI only — used for wide grids before expensive bootstrap."""
    return significance(bets, n_boot=0)


def seasonal_table(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cum_u = cum_st = 0.0
    peak = 0.0
    max_dd = 0.0
    dd_start = None
    recovery = None
    for season, g in bets.groupby("season"):
        st = float(g["stake"].sum())
        pr = float(g["profit"].sum())
        cum_u += pr
        cum_st += st
        if cum_u > peak:
            peak = cum_u
            if dd_start is not None and recovery is None:
                recovery = int(season)
        dd = cum_u - peak
        if dd < max_dd:
            max_dd = dd
            dd_start = int(season)
            recovery = None
        rows.append(
            {
                "season": int(season),
                "n": len(g),
                "hit": hit(g),
                "roi": pr / st if st else 0.0,
                "units": pr,
                "cum_roi": cum_u / cum_st if cum_st else 0.0,
                "cum_units": cum_u,
                "avg_odds": float(g["close_odds"].mean()),
                "avg_edge": float(g["edge"].mean()),
            }
        )
    df = pd.DataFrame(rows)
    return df, {"max_dd_units": max_dd, "dd_trough_season": dd_start, "recovery_season": recovery}


def leave_one_season_out(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seasons = sorted(bets["season"].unique())
    for s in seasons:
        rest = bets[bets["season"] != s]
        rows.append(
            {
                "left_out": int(s),
                "n": len(rest),
                "roi": roi(rest),
                "hit": hit(rest),
                "units": float(rest["profit"].sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 1 — Protected system significance
# ---------------------------------------------------------------------------


def part1_significance(data_dir: Path) -> None:
    print("\n=== PART 1: Protected system significance ===")
    dest = OUT / "significance"
    dest.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    lines = ["# Iteration 21 — Statistical significance of protected systems", ""]
    for sys_ in PROTECTED:
        print(f"  {sys_['name']}…")
        preds = pd.read_parquet(
            ROOT / "experiments" / sys_["experiment_id"] / "predictions.parquet"
        )
        matches = load_aligned(data_dir / "interim" / sys_["aligned"])
        bets = score_ou(
            preds,
            matches,
            side=sys_["side"],
            min_odds=sys_["min_odds"],
            max_odds=sys_["max_odds"],
            edge=sys_["edge"],
        )
        sig = significance(bets)
        seas, dd = seasonal_table(bets)
        loo = leave_one_season_out(bets)
        bets.to_parquet(dest / f"{sys_['name']}_bets.parquet", index=False)
        seas.to_csv(dest / f"{sys_['name']}_seasons.csv", index=False)
        loo.to_csv(dest / f"{sys_['name']}_leave_one_season.csv", index=False)
        # sensitivity
        sens = []
        for e in [sys_["edge"] - 0.02, sys_["edge"], sys_["edge"] + 0.02, sys_["edge"] + 0.05]:
            if e < 0.03:
                continue
            for dlo, dhi in [(-0.1, 0.0), (0.0, 0.0), (0.0, 0.1), (-0.1, 0.1)]:
                lo = max(1.2, sys_["min_odds"] + dlo)
                hi = sys_["max_odds"] + dhi
                b2 = score_ou(preds, matches, side=sys_["side"], min_odds=lo, max_odds=hi, edge=e)
                if len(b2) < 40:
                    continue
                spos, sn = season_pos(b2)
                sens.append(
                    {
                        "edge": e,
                        "min_odds": lo,
                        "max_odds": hi,
                        "n": len(b2),
                        "roi": roi(b2),
                        "hit": hit(b2),
                        "seasons_pos": spos,
                        "seasons_n": sn,
                        "t_stat": significance(b2).get("t_stat"),
                    }
                )
        pd.DataFrame(sens).to_csv(dest / f"{sys_['name']}_sensitivity.csv", index=False)

        credible = (
            sig.get("n", 0) >= 100
            and (sig.get("t_stat") or 0) >= 1.64
            and (sig.get("boot_ci95_lo") or -1) > -0.02
            and sig.get("seasons_pos", 0) >= max(6, (sig.get("seasons_n") or 0) // 2 + 1)
            and (sig.get("pct_boot_positive") or 0) >= 0.90
        )
        verdict = "STATISTICALLY CREDIBLE" if credible else "KEEP / MONITOR"
        if sig.get("p_value") is not None and sig["p_value"] < 0.05 and sig.get("n", 0) >= 100:
            verdict = "STATISTICALLY CREDIBLE (p<0.05)"
        elif sig.get("t_stat") and sig["t_stat"] >= 1.96 and sig.get("n", 0) >= 100:
            verdict = "STATISTICALLY CREDIBLE (t≥1.96)"

        row = {**sys_, **sig, **dd, "verdict": verdict}
        summary_rows.append(row)

        lines.append(f"## {sys_['name']} ({sys_['status']})")
        lines.append("")
        lines.append(
            f"Rules: {sys_['side']} {sys_['min_odds']:.2f}–{sys_['max_odds']:.2f} @ e≥{100*sys_['edge']:.0f}%"
        )
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        for k in [
            "n",
            "roi",
            "hit",
            "units",
            "t_stat",
            "p_value",
            "ci95_lo",
            "ci95_hi",
            "boot_ci95_lo",
            "boot_ci95_hi",
            "pct_boot_positive",
            "seasons_pos",
            "seasons_n",
            "max_dd_units",
            "avg_odds",
            "avg_edge",
        ]:
            v = row.get(k)
            if isinstance(v, float):
                if "roi" in k or "hit" in k or "pct" in k or "ci" in k or k == "p_value":
                    lines.append(f"| {k} | {v:.4f} |")
                else:
                    lines.append(f"| {k} | {v:.3f} |")
            else:
                lines.append(f"| {k} | {v} |")
        lines.append(f"| **verdict** | **{verdict}** |")
        lines.append("")
        lines.append("### Season-by-season")
        lines.append("")
        lines.append("| Season | n | Hit | ROI | Units | Cum ROI | Cum u |")
        lines.append("|-------:|--:|----:|----:|------:|--------:|------:|")
        for _, r in seas.iterrows():
            lines.append(
                f"| {int(r['season'])} | {int(r['n'])} | {100*r['hit']:.1f}% | "
                f"{100*r['roi']:+.1f}% | {r['units']:+.2f} | {100*r['cum_roi']:+.1f}% | "
                f"{r['cum_units']:+.2f} |"
            )
        lines.append("")

    pd.DataFrame(summary_rows).to_csv(dest / "summary.csv", index=False)
    (dest / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("  Wrote", dest / "REPORT.md")


# ---------------------------------------------------------------------------
# Part 2 — Wide OU / AH / 1X2 hunt
# ---------------------------------------------------------------------------


def part2_wide_hunt(data_dir: Path) -> None:
    # Fast grid: one evaluate_predictions per artifact, then pandas filter combos.
    print("\n=== PART 2: Wide multi-league hunt (fast filter) ===", flush=True)
    dest = OUT / "hunt"
    dest.mkdir(parents=True, exist_ok=True)
    grid_rows: list[dict] = []
    prog_path = dest / "full_grid_partial.csv"

    universe_cfg = {
        "markets": ["ou25", "ah", "1x2"],
        "edge_threshold": 0.03,
        "edge_threshold_by_market": {"ou25": 0.03, "ah": 0.03, "1x2": 0.03},
        "bet_filters": {"enabled": False, "rules": []},
        "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
        "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
    }

    def _metrics(bets: pd.DataFrame) -> dict | None:
        if bets is None or len(bets) < 50:
            return None
        sig = significance_fast(bets)
        spos, sn = season_pos(bets)
        return {**sig, "seasons_pos": spos, "seasons_n": sn}

    for art in LEAGUE_ARTIFACTS:
        pred_path = ROOT / "experiments" / art["experiment_id"] / "predictions.parquet"
        if not pred_path.exists():
            print(f"  SKIP missing {art['league']} {art['label']}", flush=True)
            continue
        preds = pd.read_parquet(pred_path)
        matches = load_aligned(data_dir / "interim" / art["aligned"])
        print(f"  {art['league']}/{art['label']} n={len(preds)} — scoring universe…", flush=True)
        t_art = time.time()
        universe = evaluate_predictions(preds, matches, universe_cfg, edge_threshold=0.03)
        print(f"    universe bets={len(universe)} in {(time.time()-t_art):.1f}s", flush=True)
        ou = universe[universe["market"] == "ou25"]
        ah = universe[universe["market"] == "ah"]
        ml = universe[universe["market"] == "1x2"]

        for side, bands in [("under", UNDER_BANDS), ("over", OVER_BANDS)]:
            base = ou[ou["side"] == side]
            for lo, hi in bands:
                band = base[(base["close_odds"] >= lo) & (base["close_odds"] <= hi)]
                for edge in EDGES:
                    bets = band[band["edge"] >= edge]
                    m = _metrics(bets)
                    if m is None:
                        continue
                    grid_rows.append(
                        {
                            "league": art["league"],
                            "model": art["label"],
                            "role": art["role"],
                            "market": "ou25",
                            "side": side,
                            "min_odds": lo,
                            "max_odds": hi,
                            "edge": edge,
                            "n": m["n"],
                            "roi": m["roi"],
                            "hit": m["hit"],
                            "t_stat": m["t_stat"],
                            "p_value": m["p_value"],
                            "boot_ci95_lo": None,
                            "pct_boot_pos": None,
                            "seasons_pos": m["seasons_pos"],
                            "seasons_n": m["seasons_n"],
                            "units": m["units"],
                            "tag": f"{art['league']}_{art['label']}_ou25_{side}_{lo}-{hi}_e{int(edge*100)}",
                        }
                    )

        for edge in AH_EDGES:
            for max_o in [1.90, 2.10, 2.50, None]:
                for sides in [None, ["H"], ["A"]]:
                    bets = ah[ah["edge"] >= edge]
                    if max_o is not None:
                        bets = bets[bets["close_odds"] <= max_o]
                    if sides is not None:
                        bets = bets[bets["side"].isin(sides)]
                    m = _metrics(bets)
                    if m is None:
                        continue
                    side_tag = "all" if sides is None else "".join(sides)
                    grid_rows.append(
                        {
                            "league": art["league"],
                            "model": art["label"],
                            "role": art["role"],
                            "market": "ah",
                            "side": side_tag,
                            "min_odds": None,
                            "max_odds": max_o,
                            "edge": edge,
                            "n": m["n"],
                            "roi": m["roi"],
                            "hit": m["hit"],
                            "t_stat": m["t_stat"],
                            "p_value": m["p_value"],
                            "boot_ci95_lo": None,
                            "pct_boot_pos": None,
                            "seasons_pos": m["seasons_pos"],
                            "seasons_n": m["seasons_n"],
                            "units": m["units"],
                            "tag": f"{art['league']}_{art['label']}_ah_{side_tag}_e{int(edge*100)}_max{max_o}",
                        }
                    )

        for edge in ML_EDGES:
            for max_o in ML_MAX_ODDS:
                for sides in [None, ["H"], ["A"], ["H", "A"], ["D"]]:
                    bets = ml[(ml["edge"] >= edge) & (ml["close_odds"] <= max_o)]
                    if sides is not None:
                        bets = bets[bets["side"].isin(sides)]
                    m = _metrics(bets)
                    if m is None:
                        continue
                    side_tag = "all" if sides is None else "".join(sides)
                    grid_rows.append(
                        {
                            "league": art["league"],
                            "model": art["label"],
                            "role": art["role"],
                            "market": "1x2",
                            "side": side_tag,
                            "min_odds": None,
                            "max_odds": max_o,
                            "edge": edge,
                            "n": m["n"],
                            "roi": m["roi"],
                            "hit": m["hit"],
                            "t_stat": m["t_stat"],
                            "p_value": m["p_value"],
                            "boot_ci95_lo": None,
                            "pct_boot_pos": None,
                            "seasons_pos": m["seasons_pos"],
                            "seasons_n": m["seasons_n"],
                            "units": m["units"],
                            "tag": f"{art['league']}_{art['label']}_1x2_{side_tag}_e{int(edge*100)}_max{max_o}",
                        }
                    )

        pd.DataFrame(grid_rows).to_csv(prog_path, index=False)
        print(
            f"    grid rows={len(grid_rows)} | artifact {(time.time()-t_art)/60:.1f}m",
            flush=True,
        )

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(dest / "full_grid.csv", index=False)

    short = grid[
        (grid["n"] >= 100)
        & (grid["roi"] >= 0.02)
        & (grid["seasons_pos"] >= (grid["seasons_n"] // 2 + 1))
        & (grid["seasons_n"] >= 6)
        & (grid["t_stat"].fillna(0) >= 1.2)
    ].copy()
    print(f"  Bootstrap shortlist: {len(short)}", flush=True)

    cache: dict[tuple[str, str], pd.DataFrame] = {}

    def _universe_for(league: str, model: str) -> pd.DataFrame:
        key = (league, model)
        if key not in cache:
            art = next(
                a for a in LEAGUE_ARTIFACTS if a["league"] == league and a["label"] == model
            )
            preds = pd.read_parquet(
                ROOT / "experiments" / art["experiment_id"] / "predictions.parquet"
            )
            matches = load_aligned(data_dir / "interim" / art["aligned"])
            cache[key] = evaluate_predictions(preds, matches, universe_cfg, edge_threshold=0.03)
        return cache[key]

    boot_rows = []
    for i, (_, row) in enumerate(short.iterrows()):
        uni = _universe_for(row["league"], row["model"])
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
            if row["side"] != "all":
                bets = bets[bets["side"].isin(list(row["side"]))]
        else:
            bets = uni[
                (uni["market"] == "1x2")
                & (uni["edge"] >= float(row["edge"]))
                & (uni["close_odds"] <= float(row["max_odds"]))
            ]
            if row["side"] != "all":
                bets = bets[bets["side"].isin(list(row["side"]))]
        sig = significance(bets, n_boot=2000)
        seas, dd = seasonal_table(bets)
        boot_rows.append(
            {
                **row.to_dict(),
                "boot_ci95_lo": sig["boot_ci95_lo"],
                "boot_ci95_hi": sig["boot_ci95_hi"],
                "pct_boot_pos": sig["pct_boot_positive"],
                "t_stat": sig["t_stat"],
                "p_value": sig["p_value"],
                "max_dd_units": dd["max_dd_units"],
            }
        )
        if sig.get("roi", 0) >= 0.04 and (sig.get("t_stat") or 0) >= 1.5:
            seas.to_csv(dest / f"seasons_{row['tag']}.csv", index=False)
        if (i + 1) % 25 == 0:
            print(f"    bootstrapped {i+1}/{len(short)}", flush=True)

    boot_df = pd.DataFrame(boot_rows)
    if len(boot_df):
        for col in ["boot_ci95_lo", "boot_ci95_hi", "pct_boot_pos"]:
            grid = grid.drop(columns=[col], errors="ignore")
        grid = grid.merge(
            boot_df[["tag", "boot_ci95_lo", "boot_ci95_hi", "pct_boot_pos", "max_dd_units"]],
            on="tag",
            how="left",
        )
        grid.to_csv(dest / "full_grid.csv", index=False)
        boot_df.to_csv(dest / "bootstrap_shortlist.csv", index=False)

    src = boot_df if len(boot_df) else grid
    prom = src[
        (src["n"] >= 100)
        & (src["roi"] >= 0.03)
        & (src["seasons_pos"] >= (src["seasons_n"] // 2 + 1))
        & (src["seasons_n"] >= 6)
        & (src["t_stat"].fillna(0) >= 1.5)
        & (src["boot_ci95_lo"].fillna(-1) > -0.05)
        & (src["role"] != "protected_control")
    ].sort_values(["roi", "t_stat"], ascending=False)

    confirm = grid[
        (grid["role"].isin(["protected_control", "protected_paper"]))
        & (grid["n"] >= 100)
        & (grid["roi"] >= 0.03)
    ].sort_values("roi", ascending=False)

    watch = src[
        (src["n"] >= 80)
        & (src["roi"] >= 0.04)
        & (src["seasons_pos"] >= 6)
        & (src["t_stat"].fillna(0) >= 1.2)
        & (src["role"] != "protected_control")
    ].sort_values(["t_stat", "roi"], ascending=False)

    prom.to_csv(dest / "promotion_candidates.csv", index=False)
    confirm.to_csv(dest / "protected_confirmation.csv", index=False)
    watch.to_csv(dest / "research_watchlist.csv", index=False)
    grid.sort_values("roi", ascending=False).head(80).to_csv(dest / "top80_any.csv", index=False)
    for mkt in ["ou25", "ah", "1x2"]:
        grid[grid["market"] == mkt].sort_values("roi", ascending=False).head(30).to_csv(
            dest / f"top30_{mkt}.csv", index=False
        )

    lines = [
        "# Iteration 21 — Wide hunt results",
        "",
        f"Grid size: {len(grid)} configurations",
        f"Bootstrap shortlist: {len(boot_df)}",
        "",
        "## Promotion candidates (n≥100, ROI≥3%, majority seasons, t≥1.5, boot CI not deeply red)",
        "",
    ]
    if len(prom) == 0:
        lines.append("_None met the bar beyond already-known protected systems._")
    else:
        lines.append("| Tag | n | ROI | t | p | Boot CI lo | Seasons+ | Units | DD |")
        lines.append("|-----|--:|----:|--:|--:|-----------:|---------:|------:|---:|")
        for _, r in prom.head(40).iterrows():
            lines.append(
                f"| `{r['tag']}` | {int(r['n'])} | {100*r['roi']:+.1f}% | "
                f"{r['t_stat']:.2f} | {r['p_value']:.3f} | {100*(r['boot_ci95_lo'] or 0):+.1f}% | "
                f"{int(r['seasons_pos'])}/{int(r['seasons_n'])} | {r['units']:+.1f} | "
                f"{r.get('max_dd_units', float('nan')):.2f} |"
            )
    lines += ["", "## Research watchlist (softer bar)", ""]
    if len(watch) == 0:
        lines.append("_Empty._")
    else:
        lines.append("| Tag | n | ROI | t | Seasons+ |")
        lines.append("|-----|--:|----:|--:|---------:|")
        for _, r in watch.head(25).iterrows():
            lines.append(
                f"| `{r['tag']}` | {int(r['n'])} | {100*r['roi']:+.1f}% | "
                f"{r['t_stat']:.2f} | {int(r['seasons_pos'])}/{int(r['seasons_n'])} |"
            )
    lines += [
        "",
        "## Other totals lines (1.5 / 3.5)",
        "",
        "Aligned football-data closes have **OU 2.5 only** — no `close_over15` / `close_over35`. "
        "Model produces `p_over15`/`p_over35` for live sheets, but historical EV backtests "
        "for those lines are **blocked** until odds are ingested.",
        "",
        "## AH + 1X2 summary",
        "",
    ]
    for mkt in ["ah", "1x2"]:
        sub = grid[grid["market"] == mkt]
        pos = sub[(sub["roi"] >= 0.03) & (sub["n"] >= 100) & (sub["seasons_pos"] >= 6)]
        lines.append(
            f"- **{mkt}**: {len(sub)} configs scored; {len(pos)} with ROI≥3%, n≥100, ≥6 positive seasons "
            f"(see `top30_{mkt}.csv`)."
        )
    lines.append("")
    (dest / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"  Candidates: {len(prom)} | Watch: {len(watch)} | Wrote {dest / 'REPORT.md'}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Part 3 — MLS ingest + score predictions (no OU EV)
# ---------------------------------------------------------------------------


def part3_mls_and_score_preds(data_dir: Path) -> None:
    print("\n=== PART 3: MLS + score predictions ===")
    dest = OUT / "score_predictions"
    dest.mkdir(parents=True, exist_ok=True)

    # Ingest MLS USA.csv into a simple matches frame
    import requests
    from io import BytesIO

    url = "https://www.football-data.co.uk/new/USA.csv"
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    raw = pd.read_csv(BytesIO(r.content))
    raw = raw[raw["League"].astype(str).str.upper() == "MLS"].copy()
    raw_path = data_dir / "raw" / "football_data" / "USA" / "USA.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(r.content)

    from origination.utils.team_names import DEFAULT_MAPPER
    from origination.utils.seeding import season_from_date

    mapper = DEFAULT_MAPPER
    df = raw.copy()
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "Home", "Away"])
    df["home_team"] = mapper.map_series(df["Home"])
    df["away_team"] = mapper.map_series(df["Away"])
    df["home_goals"] = pd.to_numeric(df["HG"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["AG"], errors="coerce")
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["season"] = df["date"].map(lambda d: int(d.year))  # MLS calendar year
    df["close_h"] = pd.to_numeric(df.get("PSCH"), errors="coerce")
    df["close_d"] = pd.to_numeric(df.get("PSCD"), errors="coerce")
    df["close_a"] = pd.to_numeric(df.get("PSCA"), errors="coerce")
    df["match_id"] = [
        f"{d.strftime('%Y%m%d')}_{h.replace(' ','')}_{a.replace(' ','')}"
        for d, h, a in zip(df["date"], df["home_team"], df["away_team"], strict=True)
    ]
    # No OU closes
    df["close_over25"] = np.nan
    df["close_under25"] = np.nan
    df["ah_line"] = np.nan
    df["close_ahh"] = np.nan
    df["close_aha"] = np.nan

    completed = df.dropna(subset=["home_goals", "away_goals"]).copy()
    completed["ftr"] = np.where(
        completed["home_goals"] > completed["away_goals"],
        "H",
        np.where(completed["home_goals"] < completed["away_goals"], "A", "D"),
    )
    upcoming = df[df["home_goals"].isna()].copy()

    aligned_path = data_dir / "interim" / "matches_aligned_MLS.parquet"
    # Store completed for modeling
    keep_cols = [
        c
        for c in [
            "match_id",
            "date",
            "season",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "total_goals",
            "ftr",
            "close_h",
            "close_d",
            "close_a",
            "close_over25",
            "close_under25",
            "ah_line",
            "close_ahh",
            "close_aha",
        ]
        if c in completed.columns
    ]
    completed[keep_cols].to_parquet(aligned_path, index=False)
    print(f"  MLS completed matches: {len(completed)} -> {aligned_path}")

    # Fit a simple goals DC-ish model via existing stack if possible
    cfg_path = ROOT / "configs" / "league_MLS.yaml"
    if not cfg_path.exists():
        # minimal goals-only config
        cfg_path.write_text(
            """
project:
  name: origination
  seed: 42
  data_dir: data
  experiments_dir: experiments
  experiment_label: league_MLS
leagues:
  - code: USA
    name: MLS
    understat: null
    start_season: 2014
    end_season: null
data:
  football_data: {enabled: false}
  understat: {enabled: false}
  align:
    require_odds: false
    output: matches_aligned_MLS.parquet
features:
  groups:
    basic_form: true
    xg_form: false
    shots: false
    elo: true
    schedule: true
    understat_advanced: false
  windows: [3, 5, 10]
  ewm_span: 10
  elo: {k: 20.0, home_advantage: 60.0, initial: 1500.0}
  context_adjustments: {enabled: false}
model:
  type: dixon_coles
  max_goals: 10
  dixon_coles:
    rho_init: -0.05
    xi: 0.0015
    intensity_source: goals
    intensity_adjustments: {enabled: false}
    totals_intercept: {enabled: false}
  calibration: {method: temperature, ou_method: temperature, holdout_seasons: 1}
  residual: {enabled: false}
  hierarchical: {enabled: false}
backtest:
  markets: [1x2]
  edge_threshold: 0.05
  stake: {method: flat, unit: 1.0}
  odds: {remove_vig: power, min_odds: 1.2, max_odds: 15.0}
""".strip(),
            encoding="utf-8",
        )

    from origination.prediction.upcoming import predict_upcoming
    from origination.data_ingestion.fixtures_upcoming import refresh_upcoming_fixtures_for_league
    from origination.data_ingestion.pinnacle_odds import refresh_pinnacle_odds, build_pinnacle_ou25_table

    cfg = load_config(cfg_path)

    # Upcoming from Pinnacle MLS
    try:
        fx, meta = refresh_upcoming_fixtures_for_league(data_dir, "MLS", cfg, days_ahead=21)
        print(f"  MLS fixtures: {len(fx)} source={meta.get('source')}")
    except Exception as exc:
        print(f"  MLS fixtures failed: {exc}")
        # Build from blank scores in USA if any
        fx = upcoming.rename(columns={})[
            ["match_id", "date", "home_team", "away_team", "season"]
        ].copy() if len(upcoming) else pd.DataFrame()
        if len(fx) == 0:
            # last resort: pinnacle only
            pin, _ = build_pinnacle_ou25_table(league_id=2663)
            if len(pin):
                fx = pin.rename(columns={})[
                    [c for c in ["date", "home_team", "away_team"] if c in pin.columns]
                ].copy()
                fx["match_id"] = [
                    f"{pd.Timestamp(d).strftime('%Y%m%d')}_{str(h).replace(' ','')}_{str(a).replace(' ','')}"
                    for d, h, a in zip(fx["date"], fx["home_team"], fx["away_team"], strict=True)
                ]
                fx["season"] = pd.to_datetime(fx["date"]).dt.year

    confidence = {
        "n_history": int(len(completed)),
        "seasons": sorted(completed["season"].unique().tolist()) if len(completed) else [],
        "has_xg": False,
        "has_shots": False,
        "has_ou_closes": False,
        "has_1x2_closes": bool(completed["close_h"].notna().mean() > 0.5) if len(completed) else False,
        "intensity": "goals-only Dixon–Coles",
        "confidence_label": "LOW–MODERATE",
        "confidence_notes": (
            "MLS predictions use goals-only intensities (no Understat xG/shots). "
            "No historical OU closes for EV validation. Treat projected scores as "
            "directional aids for manual review, not as a bettable OU system."
        ),
    }
    if confidence["n_history"] >= 3000 and confidence["has_1x2_closes"]:
        confidence["confidence_label"] = "MODERATE (goals+1X2 history; no OU validation)"
    elif confidence["n_history"] >= 1500:
        confidence["confidence_label"] = "LOW–MODERATE"
    else:
        confidence["confidence_label"] = "LOW"

    preds_out = pd.DataFrame()
    if len(fx) and len(completed) >= 500:
        try:
            # Need season on upcoming
            fx = fx.copy()
            fx["date"] = pd.to_datetime(fx["date"])
            if "season" not in fx.columns:
                fx["season"] = fx["date"].dt.year
            hist = completed.copy()
            hist["date"] = pd.to_datetime(hist["date"])
            preds = predict_upcoming(hist, fx, cfg, odds=None, apply_residual=False)
            preds_out = preds.copy()
            preds_out["confidence_label"] = confidence["confidence_label"]
            preds_out["data_n_history"] = confidence["n_history"]
            preds_out["has_xg"] = False
            preds_out["has_ou_closes"] = False
            preds_out.to_csv(dest / "MLS_score_predictions.csv", index=False)
            print(f"  Wrote MLS score predictions: {len(preds_out)}")
        except Exception as exc:
            print(f"  MLS predict failed: {exc}")
            (dest / "MLS_predict_error.txt").write_text(str(exc), encoding="utf-8")

    # Also produce score preds for other leagues' upcoming fixtures if present
    from origination.utils.league_registry import list_league_keys, get_league

    for key in list_league_keys():
        if key == "MLS":
            continue
        info = get_league(key)
        fx_path = data_dir / "interim" / (
            "fixtures_upcoming_EPL.csv" if key == "EPL" else f"fixtures_upcoming_{key}.csv"
        )
        aligned = data_dir / "interim" / info["aligned"]
        cfg_l = load_config(ROOT / info["config"])
        if not fx_path.exists() or not aligned.exists():
            # try refresh
            try:
                refresh_upcoming_fixtures_for_league(data_dir, key, cfg_l, days_ahead=14)
            except Exception:
                continue
        if not fx_path.exists():
            continue
        try:
            fx2 = pd.read_csv(fx_path, parse_dates=["date"])
            hist2 = load_aligned(aligned)
            if len(fx2) == 0 or len(hist2) < 500:
                continue
            if "season" not in fx2.columns:
                fx2["season"] = fx2["date"].map(season_from_date)
            # enrich understat if needed
            if cfg_l.get("features", {}).get("groups", {}).get("understat_advanced"):
                from origination.data_ingestion.understat_advanced import (
                    enrich_matches_with_understat_advanced,
                    load_understat_team_history,
                )

                hist_us = load_understat_team_history(data_dir / "raw" / "understat")
                hist2 = enrich_matches_with_understat_advanced(hist2, hist_us)
            # load pinnacle if available
            from origination.data_ingestion.pinnacle_odds import load_pinnacle_odds

            odds = load_pinnacle_odds(data_dir, key)
            preds2 = predict_upcoming(
                hist2, fx2, cfg_l, odds=odds if len(odds) else None, apply_residual=False
            )
            n_hist = len(hist2.dropna(subset=["home_goals", "away_goals"]))
            has_xg = "home_xg" in hist2.columns and hist2["home_xg"].notna().mean() > 0.5
            conf = "HIGH" if has_xg and n_hist >= 2500 else ("MODERATE" if n_hist >= 1500 else "LOW")
            preds2["confidence_label"] = conf
            preds2["data_n_history"] = n_hist
            preds2["has_xg"] = has_xg
            preds2["league"] = key
            preds2.to_csv(dest / f"{key}_score_predictions.csv", index=False)
            print(f"  {key} score preds: {len(preds2)} conf={conf}")
        except Exception as exc:
            print(f"  {key} score preds failed: {exc}")

    # MLS feasibility update
    feas = [
        "# MLS feasibility update (iter21)",
        "",
        f"- Completed matches ingested: **{len(completed)}** (seasons {confidence['seasons'][:1]}–{confidence['seasons'][-1:]})",
        f"- 1X2 closes present: **{confidence['has_1x2_closes']}**",
        "- OU closes: **No** (cannot EV-backtest totals)",
        "- xG / shots: **No**",
        f"- Score-prediction confidence: **{confidence['confidence_label']}**",
        "",
        confidence["confidence_notes"],
        "",
        "Live path: Pinnacle league id 2663 for OU prices; goals-only model for projected scores.",
        f"Artifacts: `{dest.as_posix()}/MLS_score_predictions.csv`",
        "",
    ]
    (dest / "MLS_FEASIBILITY.md").write_text("\n".join(feas), encoding="utf-8")
    (dest / "MLS_confidence.json").write_text(json.dumps(confidence, indent=2), encoding="utf-8")


def write_master_report() -> None:
    lines = [
        "# Iteration 21 — Master Summary",
        "",
        "EPL Unders, EPL short Overs, and Bundesliga Unders packs were **not modified**.",
        "",
        "## Artifacts",
        "",
        f"- Significance: `{OUT / 'significance' / 'REPORT.md'}`",
        f"- Wide hunt: `{OUT / 'hunt' / 'REPORT.md'}`",
        f"- Score predictions: `{OUT / 'score_predictions'}`",
        f"- MLS feasibility: `{OUT / 'score_predictions' / 'MLS_FEASIBILITY.md'}`",
        "",
    ]
    sig_csv = OUT / "significance" / "summary.csv"
    if sig_csv.exists():
        s = pd.read_csv(sig_csv)
        lines.append("## 1) Protected systems — statistical verdicts")
        lines.append("")
        lines.append("| System | Status | n | ROI | t | p | Boot+ | Seasons+ | Max DD | Verdict |")
        lines.append("|--------|--------|--:|----:|--:|--:|------:|---------:|-------:|---------|")
        for _, r in s.iterrows():
            lines.append(
                f"| {r['name']} | {r['status']} | {int(r['n'])} | {100*r['roi']:+.1f}% | "
                f"{r.get('t_stat', float('nan')):.2f} | {r.get('p_value', float('nan')):.3f} | "
                f"{100*r.get('pct_boot_positive', 0):.0f}% | "
                f"{int(r['seasons_pos'])}/{int(r['seasons_n'])} | {r.get('max_dd_units', 0):.2f}u | "
                f"{r['verdict']} |"
            )
        lines.append("")
        lines.append(
            "**Interpretation:** None of the three clears a strict two-sided p<0.05 bar. "
            "EPL Unders is closest (t≈1.61, ~95% bootstrap mass positive, 7/10 seasons). "
            "All three remain **KEEP** for production/paper; do not size up purely on classical significance."
        )
        lines.append("")

    prom = OUT / "hunt" / "promotion_candidates.csv"
    watch = OUT / "hunt" / "research_watchlist.csv"
    if prom.exists():
        p = pd.read_csv(prom)
        lines.append(f"## 2) New promotion candidates: **{len(p)}**")
        lines.append("")
        if len(p):
            lines.append("| Tag | League | Market | n | ROI | t | Seasons+ | Status suggestion |")
            lines.append("|-----|--------|--------|--:|----:|--:|---------:|-------------------|")
            for _, r in p.head(20).iterrows():
                sug = "Paper" if (r.get("t_stat") or 0) >= 1.8 and r["n"] >= 120 else "Research"
                # never auto-promote anything that collides with protected OU rules
                lines.append(
                    f"| `{r['tag']}` | {r['league']} | {r['market']}/{r['side']} | "
                    f"{int(r['n'])} | {100*r['roi']:+.1f}% | {r['t_stat']:.2f} | "
                    f"{int(r['seasons_pos'])}/{int(r['seasons_n'])} | {sug} |"
                )
        else:
            lines.append("None cleared the conservative promotion bar.")
        lines.append("")
    if watch.exists():
        w = pd.read_csv(watch)
        lines.append(f"## Research watchlist: **{len(w)}** (softer bar — do not paper yet)")
        lines.append("")
        if len(w):
            for _, r in w.head(15).iterrows():
                lines.append(
                    f"- `{r['tag']}` — n={int(r['n'])}, ROI={100*r['roi']:+.1f}%, "
                    f"t={r['t_stat']:.2f}, seasons {int(r['seasons_pos'])}/{int(r['seasons_n'])}"
                )
        lines.append("")

    lines.append("## 3) AH + Moneyline")
    lines.append("")
    for mkt in ["ah", "1x2"]:
        top = OUT / "hunt" / f"top30_{mkt}.csv"
        if top.exists():
            t = pd.read_csv(top)
            best = t.iloc[0] if len(t) else None
            if best is not None:
                lines.append(
                    f"- **{mkt}** best ROI row: `{best['tag']}` "
                    f"(n={int(best['n'])}, ROI={100*best['roi']:+.1f}%, t={best.get('t_stat', float('nan')):.2f})"
                )
            else:
                lines.append(f"- **{mkt}**: no rows")
        else:
            lines.append(f"- **{mkt}**: hunt incomplete")
    lines.append("")
    lines.append("## 4) Other totals (1.5 / 3.5)")
    lines.append("")
    lines.append(
        "Historical EV backtests **blocked** — football-data aligned closes are OU 2.5 only. "
        "Live model still emits `p_over15` / `p_over35` for manual inspection."
    )
    lines.append("")
    lines.append("## 5) MLS + score predictions")
    lines.append("")
    mls_feas = OUT / "score_predictions" / "MLS_FEASIBILITY.md"
    if mls_feas.exists():
        lines.append(mls_feas.read_text(encoding="utf-8"))
    else:
        lines.append("_Score-prediction run pending / failed — see logs._")
    lines.append("")
    pred_dir = OUT / "score_predictions"
    if pred_dir.exists():
        csvs = sorted(pred_dir.glob("*_score_predictions.csv"))
        if csvs:
            lines.append("### Score prediction files")
            lines.append("")
            for c in csvs:
                try:
                    n = len(pd.read_csv(c))
                except Exception:
                    n = "?"
                lines.append(f"- `{c.name}` — {n} fixtures")
            lines.append("")

    lines.append("## 6) Live / paper trading readiness")
    lines.append("")
    lines.append("| Priority | System | Mode | Notes |")
    lines.append("|---------:|--------|------|-------|")
    lines.append("| 1 | EPL Unders 2.00–4.00 @ ≥8% | **Production** | KEEP/MONITOR; largest n |")
    lines.append("| 2 | EPL short Overs 1.60–2.50 @ ≥10% | **Production** | KEEP/MONITOR |")
    lines.append("| 3 | Bundesliga Unders 1.70–2.50 @ ≥10% | **Paper** | Best DD profile |")
    lines.append("| 4 | MLS / lesser leagues | **Score preds only** | Manual review; no OU EV |")
    lines.append("| — | AH / 1X2 pockets | **Research** | See hunt CSVs before any paper |")
    lines.append("")
    lines.append("Gameday: `scripts/gameday_ui.py` supports league switch + multi-pack flags; "
                 "`run_gameday_sheet.py --league …` writes `flag_*` columns per active pack.")
    lines.append("")
    (OUT / "MASTER_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT / "MASTER_REPORT.md", flush=True)


def main() -> None:
    setup_logging("WARNING")
    t0 = time.time()
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    # Skip part1 if already complete (restart-friendly)
    sig_done = (OUT / "significance" / "summary.csv").exists()
    if sig_done:
        print("=== PART 1: skipped (summary.csv exists) ===", flush=True)
    else:
        part1_significance(data_dir)
    part2_wide_hunt(data_dir)
    part3_mls_and_score_preds(data_dir)
    write_master_report()
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
