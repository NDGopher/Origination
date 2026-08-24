#!/usr/bin/env python
"""
Iteration 22 — expand to new leagues + hunt systems.

Protected packs NEVER modified:
  EPL Unders, EPL short Overs, Bundesliga Unders, La Liga Home ML, Serie A Away ML.

New leagues (research): Ligue1 (F1), Eredivisie (N1), Primeira (P1), Belgium (B1).
Also re-hunts Championship with existing preds.
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

from origination.backtesting import run_walk_forward
from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.utils import (
    load_config,
    resolve_data_dir,
    resolve_experiments_dir,
    set_global_seed,
    setup_logging,
)
from origination.utils.league_registry import get_league

OUT = ROOT / "experiments" / "iter22"
OUT.mkdir(parents=True, exist_ok=True)

NEW_LEAGUES = ["Ligue1", "Eredivisie", "PrimeiraLiga", "Belgium"]

UNDER_BANDS = [
    (1.60, 2.40),
    (1.70, 2.50),
    (1.80, 2.50),
    (2.00, 3.00),
    (2.00, 4.00),
    (2.20, 3.50),
]
OVER_BANDS = [
    (1.50, 2.20),
    (1.60, 2.50),
    (1.70, 2.80),
    (1.80, 3.00),
]
EDGES = [0.05, 0.08, 0.10, 0.12, 0.15]
ML_EDGES = [0.03, 0.05, 0.08, 0.10]
ML_MAX = [1.60, 1.80, 2.00, 2.20, 2.50]
AH_EDGES = [0.03, 0.05, 0.08, 0.10]


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


def sig_fast(bets: pd.DataFrame) -> dict:
    if bets is None or len(bets) < 50:
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
    }


def bootstrap_roi(bets: pd.DataFrame, n_boot: int = 2000) -> dict:
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


def diagnose(preds: pd.DataFrame, matches: pd.DataFrame) -> dict:
    m = matches.set_index("match_id")
    rows = []
    for _, r in preds.iterrows():
        mid = r["match_id"]
        if mid not in m.index:
            continue
        match = m.loc[mid]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        tg = match.get("total_goals")
        lam = float(r.get("lambda_home", np.nan)) + float(r.get("lambda_away", np.nan))
        if pd.notna(tg) and np.isfinite(lam):
            rows.append({"lam": lam, "goals": float(tg), "p_over": float(r.get("p_over25", np.nan))})
    if not rows:
        return {"n": 0}
    d = pd.DataFrame(rows)
    corr = float(d["lam"].corr(d["goals"]))
    bias = float((d["lam"] - d["goals"]).mean())
    return {"n": len(d), "corr_lam_goals": corr, "bias_lam_minus_goals": bias}


def ingest_league(key: str, *, force: bool = False) -> Path | None:
    from origination.data_ingestion import (
        build_aligned_from_config,
        ingest_football_data_from_config,
        ingest_understat_from_config,
    )

    info = get_league(key)
    cfg = load_config(ROOT / info["config"])
    data_dir = resolve_data_dir(cfg)
    out = data_dir / "interim" / info["aligned"]
    if out.exists() and not force:
        print(f"\n=== INGEST {key} (skip, aligned exists: {out.name}) ===", flush=True)
        return out
    print(f"\n=== INGEST {key} ===", flush=True)
    try:
        fd = ingest_football_data_from_config(cfg, data_dir)
        print(f"  FD rows={len(fd)}", flush=True)
    except Exception as exc:
        print(f"  FD FAILED: {exc}", flush=True)
        return None
    us = None
    if cfg.get("data", {}).get("understat", {}).get("enabled"):
        try:
            us = ingest_understat_from_config(cfg, data_dir)
            print(f"  Understat rows={len(us) if us is not None else 0}", flush=True)
        except Exception as exc:
            print(f"  Understat failed (continue goals-only): {exc}", flush=True)
    try:
        aligned = build_aligned_from_config(cfg, data_dir, fd, us, None)
        print(f"  Aligned -> {out} n={len(aligned)}", flush=True)
        return out
    except Exception as exc:
        print(f"  Align FAILED: {exc}", flush=True)
        return None


def walk_forward_league(key: str, *, force: bool = False) -> Path | None:
    info = get_league(key)
    cfg = load_config(ROOT / info["config"])
    set_global_seed(int(cfg.get("project", {}).get("seed", 42)))
    data_dir = resolve_data_dir(cfg)
    exp_dir = resolve_experiments_dir(cfg)
    aligned_path = data_dir / "interim" / info["aligned"]
    if not aligned_path.exists():
        print(f"  SKIP WF {key}: no aligned")
        return None
    # Resume: reuse newest iter22 preds for this league
    if not force:
        existing = sorted(
            exp_dir.glob(f"*iter22_{key}*/predictions.parquet"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if existing:
            print(f"\n=== WF {key} (skip, preds exist: {existing[0].parent.name}) ===", flush=True)
            return existing[0]
    matches = load_aligned(aligned_path)
    if cfg.get("features", {}).get("groups", {}).get("understat_advanced"):
        try:
            hist_us = load_understat_team_history(data_dir / "raw" / "understat")
            matches = enrich_matches_with_understat_advanced(matches, hist_us)
        except Exception as exc:
            print(f"  understat enrich skip: {exc}")
    cfg.setdefault("project", {})["experiment_label"] = f"iter22_{key}_base"
    print(f"\n=== WF {key} n={len(matches)} ===", flush=True)
    t0 = time.time()
    result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
    eid = getattr(result, "experiment_id", None)
    pred_path = None
    if eid:
        cand = exp_dir / str(eid) / "predictions.parquet"
        if cand.exists():
            pred_path = cand
    if pred_path is None and getattr(result, "predictions", None) is not None:
        tmp = OUT / f"preds_{key}.parquet"
        result.predictions.to_parquet(tmp, index=False)
        pred_path = tmp
    if pred_path is None:
        cands = sorted(
            exp_dir.glob(f"*iter22_{key}*/predictions.parquet"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if cands:
            pred_path = cands[0]
    print(f"  DONE {(time.time()-t0)/60:.1f}m preds={pred_path}", flush=True)
    return pred_path


def hunt(preds: pd.DataFrame, matches: pd.DataFrame, league: str, model: str) -> list[dict]:
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
                if s.get("n", 0) < 50:
                    continue
                rows.append(
                    {
                        "league": league,
                        "model": model,
                        "market": "ou25",
                        "side": side,
                        "min_odds": lo,
                        "max_odds": hi,
                        "edge": e,
                        **s,
                        "tag": f"{league}_{model}_ou25_{side}_{lo}-{hi}_e{int(e*100)}",
                    }
                )

    for e in AH_EDGES:
        for max_o in [1.90, 2.10, 2.50, None]:
            bets = ah[ah["edge"] >= e]
            if max_o is not None:
                bets = bets[bets["close_odds"] <= max_o]
            s = sig_fast(bets)
            if s.get("n", 0) < 80:
                continue
            rows.append(
                {
                    "league": league,
                    "model": model,
                    "market": "ah",
                    "side": "all",
                    "min_odds": None,
                    "max_odds": max_o,
                    "edge": e,
                    **s,
                    "tag": f"{league}_{model}_ah_e{int(e*100)}_max{max_o}",
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
                        "model": model,
                        "market": "1x2",
                        "side": stag,
                        "min_odds": None,
                        "max_odds": max_o,
                        "edge": e,
                        **s,
                        "tag": f"{league}_{model}_1x2_{stag}_e{int(e*100)}_max{max_o}",
                    }
                )
    return rows


def main() -> None:
    setup_logging("ERROR")
    quality_rows = []
    hunt_rows = []
    pred_map = {}

    # 1) Ingest + WF new leagues
    for key in NEW_LEAGUES:
        aligned = ingest_league(key)
        info = get_league(key)
        data_dir = resolve_data_dir(load_config(ROOT / info["config"]))
        ap = data_dir / "interim" / info["aligned"]
        q = {"league": key, "aligned_exists": ap.exists()}
        if ap.exists():
            df = load_aligned(ap)
            q.update(
                {
                    "n": len(df),
                    "seasons": f"{int(df['season'].min())}-{int(df['season'].max())}",
                    "pct_1x2": float(df["close_h"].notna().mean()) if "close_h" in df.columns else 0,
                    "pct_ou": float(df["close_over25"].notna().mean())
                    if "close_over25" in df.columns
                    else 0,
                    "pct_ah": float(df["close_ahh"].notna().mean()) if "close_ahh" in df.columns else 0,
                    "pct_xg": float(df["home_xg"].notna().mean()) if "home_xg" in df.columns else 0,
                }
            )
        quality_rows.append(q)
        if not ap.exists():
            continue
        pred_path = walk_forward_league(key)
        if pred_path:
            pred_map[key] = (pred_path, ap)

    # Also include Championship existing if present
    champ = get_league("Championship")
    champ_aligned = ROOT / "data" / "interim" / champ["aligned"]
    champ_preds = sorted(
        (ROOT / "experiments").glob("*Championship*/predictions.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if champ_aligned.exists() and champ_preds:
        pred_map["Championship"] = (champ_preds[0], champ_aligned)
        df = load_aligned(champ_aligned)
        quality_rows.append(
            {
                "league": "Championship",
                "aligned_exists": True,
                "n": len(df),
                "seasons": f"{int(df['season'].min())}-{int(df['season'].max())}",
                "pct_1x2": float(df["close_h"].notna().mean()),
                "pct_ou": float(df["close_over25"].notna().mean()),
                "pct_ah": float(df["close_ahh"].notna().mean()) if "close_ahh" in df.columns else 0,
                "pct_xg": float(df["home_xg"].notna().mean()) if "home_xg" in df.columns else 0,
            }
        )

    # 2) Diagnose + hunt
    diag_rows = []
    for key, (pred_path, aligned_path) in pred_map.items():
        preds = pd.read_parquet(pred_path)
        matches = load_aligned(aligned_path)
        d = diagnose(preds, matches)
        d["league"] = key
        d["pred_path"] = str(pred_path)
        diag_rows.append(d)
        print(
            f"DIAG {key}: corr={d.get('corr_lam_goals')} bias={d.get('bias_lam_minus_goals')}",
            flush=True,
        )
        model = pred_path.parent.name.split("_")[-1] if pred_path else "base"
        hunt_rows.extend(hunt(preds, matches, key, model))

    quality = pd.DataFrame(quality_rows)
    diag = pd.DataFrame(diag_rows)
    grid = pd.DataFrame(hunt_rows)
    quality.to_csv(OUT / "league_data_quality.csv", index=False)
    diag.to_csv(OUT / "ranking_diagnosis.csv", index=False)
    grid.to_csv(OUT / "full_grid.csv", index=False)

    # 3) Promotion shortlist + bootstrap
    if len(grid):
        short = grid[
            (grid["n"] >= 100)
            & (grid["roi"] >= 0.03)
            & (grid["seasons_pos"] >= (grid["seasons_n"] // 2 + 1))
            & (grid["seasons_n"] >= 6)
            & (grid["t_stat"].fillna(0) >= 1.5)
        ].copy()
    else:
        short = pd.DataFrame()

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
        boot_rows.append({**row.to_dict(), **b})

    boot = pd.DataFrame(boot_rows)
    if len(boot):
        prom = boot[
            (boot["boot_ci95_lo"].fillna(-1) > -0.05)
            & (boot["pct_boot_pos"].fillna(0) >= 0.85)
        ].sort_values(["roi", "t_stat"], ascending=False)
    else:
        prom = pd.DataFrame()
    boot.to_csv(OUT / "bootstrap_shortlist.csv", index=False)
    prom.to_csv(OUT / "promotion_candidates.csv", index=False)

    # 4) Report — never touch protected packs
    lines = [
        "# Iteration 22 — League expansion",
        "",
        "Protected systems untouched: EPL Unders, EPL short Overs, Bundesliga Unders, "
        "La Liga Home ML, Serie A Away ML.",
        "",
        "## Data quality",
        "",
    ]
    if len(quality):
        lines.append("| League | n | Seasons | 1X2 | OU | AH | xG |")
        lines.append("|--------|--:|--------:|----:|---:|---:|---:|")
        for _, r in quality.iterrows():
            if not r.get("aligned_exists"):
                lines.append(f"| {r['league']} | — | — | FAIL | FAIL | FAIL | FAIL |")
                continue
            lines.append(
                f"| {r['league']} | {int(r['n'])} | {r['seasons']} | "
                f"{100*r.get('pct_1x2',0):.0f}% | {100*r.get('pct_ou',0):.0f}% | "
                f"{100*r.get('pct_ah',0):.0f}% | {100*r.get('pct_xg',0):.0f}% |"
            )
    lines += ["", "## Ranking diagnosis", ""]
    if len(diag):
        lines.append("| League | n | corr(λ,goals) | bias |")
        lines.append("|--------|--:|--------------:|-----:|")
        for _, r in diag.iterrows():
            lines.append(
                f"| {r['league']} | {int(r.get('n',0))} | "
                f"{r.get('corr_lam_goals', float('nan')):.3f} | "
                f"{r.get('bias_lam_minus_goals', float('nan')):+.3f} |"
            )
    lines += ["", f"## Promotion candidates: **{len(prom)}**", ""]
    if len(prom) == 0:
        lines.append("_None met the conservative bar. New leagues stay Research._")
    else:
        lines.append("| Tag | n | ROI | t | Boot CI lo | Seasons+ |")
        lines.append("|-----|--:|----:|--:|-----------:|---------:|")
        for _, r in prom.head(25).iterrows():
            lines.append(
                f"| `{r['tag']}` | {int(r['n'])} | {100*r['roi']:+.1f}% | "
                f"{r['t_stat']:.2f} | {100*r['boot_ci95_lo']:+.1f}% | "
                f"{int(r['seasons_pos'])}/{int(r['seasons_n'])} |"
            )
    lines += [
        "",
        "## Status board",
        "",
        "| System | Status |",
        "|--------|--------|",
        "| EPL Unders | **Production** |",
        "| EPL short Overs | **Production** |",
        "| Bundesliga Unders | **Paper** |",
        "| La Liga Home ML | **Paper** |",
        "| Serie A Away ML | **Paper** |",
        "| Ligue1 / Eredivisie / Primeira / Belgium | **Research** (new data) |",
        "| Championship OU packs | **Research** |",
        "| MLS | **Score preds only** |",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "MASTER_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT / "REPORT.md", flush=True)
    print(f"Candidates: {len(prom)}", flush=True)


if __name__ == "__main__":
    main()
