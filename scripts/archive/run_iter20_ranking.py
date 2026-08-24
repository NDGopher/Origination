#!/usr/bin/env python
"""
Iter20 — Ranking improvement WF for Championship + Serie A.

Championship: shot-volume / allow via FD shots proxy + signed intercept (min_abs_raw=0).
Serie A: enable vol06 on existing xG stack.

Reports corr(λ, goals) vs iter19 baseline. Does not touch EPL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger

from origination.backtesting import run_walk_forward, save_experiment
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.utils import load_config, resolve_data_dir, set_global_seed, setup_logging
from origination.utils.odds import two_way_fair

OUT = ROOT / "experiments" / "iter20_ranking"

JOBS = [
    {
        "name": "Championship",
        "config": "configs/league_E1_championship.yaml",
        "aligned": "matches_aligned_E1.parquet",
        "label": "iter20_Championship_shots_vol",
        "understat": False,
        "baseline_id": "20260811T164035Z_iter19_Championship_thresh_intercept",
    },
    {
        "name": "SerieA",
        "config": "configs/league_I1_serie_a.yaml",
        "aligned": "matches_aligned_I1.parquet",
        "label": "iter20_SerieA_vol06",
        "understat": True,
        "baseline_id": "20260811T165028Z_iter19_SerieA_signed_intercept",
    },
]


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
        if pd.isna(tg):
            continue
        lam = float(r.get("lambda_home", np.nan))
        mu = float(r.get("lambda_away", np.nan))
        if not (np.isfinite(lam) and np.isfinite(mu)):
            continue
        p_o = float(r["p_over25"])
        o_odds = match.get("close_over25")
        u_odds = match.get("close_under25")
        fair_o = np.nan
        if pd.notna(o_odds) and pd.notna(u_odds) and float(o_odds) > 1 and float(u_odds) > 1:
            fair_o, _ = two_way_fair(float(o_odds), float(u_odds), method="power")
        rows.append(
            {
                "sum_lambda": lam + mu,
                "total_goals": float(tg),
                "p_over25": p_o,
                "actual_over": float(tg) > 2.5,
                "edge_over": p_o - fair_o if np.isfinite(fair_o) else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    edge = df.dropna(subset=["edge_over"])
    corr_edge = (
        float(np.corrcoef(edge["edge_over"], edge["actual_over"].astype(float))[0, 1])
        if len(edge) > 50
        else None
    )
    return {
        "n": int(len(df)),
        "goals_bias": float(df["sum_lambda"].mean() - df["total_goals"].mean()),
        "corr_lambda_goals": float(np.corrcoef(df["sum_lambda"], df["total_goals"])[0, 1]),
        "brier_ou25": float(((df["p_over25"] - df["actual_over"].astype(float)) ** 2).mean()),
        "corr_edge_vs_actual_over": corr_edge,
        "over_rate_actual": float(df["actual_over"].mean()),
        "over_rate_model": float(df["p_over25"].mean()),
    }


def main() -> None:
    setup_logging("INFO")
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    rows = []
    for job in JOBS:
        if only and job["name"] not in only and job["label"] not in only:
            continue
        cfg = load_config(ROOT / job["config"])
        set_global_seed(int(cfg.get("project", {}).get("seed", 42)))
        data_dir = resolve_data_dir(cfg)
        exp_dir = ROOT / "experiments"
        matches = load_aligned(data_dir / "interim" / job["aligned"])
        if job["understat"] and cfg.get("features", {}).get("groups", {}).get(
            "understat_advanced", False
        ):
            hist = load_understat_team_history(data_dir / "raw" / "understat")
            matches = enrich_matches_with_understat_advanced(matches, hist)

        cfg.setdefault("project", {})["experiment_label"] = job["label"]
        logger.info("=== WF {} (n={}) ===", job["name"], len(matches))
        result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
        path = save_experiment(result, exp_dir)
        logger.info("Saved {} -> {}", job["name"], path)

        diag = diagnose(result.predictions, matches)
        diag["league"] = job["name"]
        diag["label"] = job["label"]
        diag["experiment_id"] = Path(path).name if path else None
        # Baseline compare
        base_path = exp_dir / job["baseline_id"] / "predictions.parquet"
        if base_path.exists():
            base_preds = pd.read_parquet(base_path)
            base = diagnose(base_preds, matches)
            diag["baseline_corr_lambda_goals"] = base["corr_lambda_goals"]
            diag["baseline_corr_edge"] = base["corr_edge_vs_actual_over"]
            diag["delta_corr_lambda"] = diag["corr_lambda_goals"] - base["corr_lambda_goals"]
            diag["delta_corr_edge"] = (
                (diag["corr_edge_vs_actual_over"] or 0) - (base["corr_edge_vs_actual_over"] or 0)
            )
        rows.append(diag)
        logger.info(
            "  corr_lambda={:.4f} (delta={:+.4f}) corr_edge={} bias={:+.3f}",
            diag["corr_lambda_goals"],
            diag.get("delta_corr_lambda", float("nan")),
            diag["corr_edge_vs_actual_over"],
            diag["goals_bias"],
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ranking_diagnosis.csv", index=False)
    (OUT / "ranking_diagnosis.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    lines = ["# Iter20 — Championship / Serie A ranking", ""]
    for _, r in df.iterrows():
        lines.append(f"## {r['league']} (`{r['label']}`)")
        lines.append("")
        lines.append(f"- corr(λ, goals): **{r['corr_lambda_goals']:.4f}** "
                     f"(baseline {r.get('baseline_corr_lambda_goals', float('nan')):.4f}, "
                     f"Δ {r.get('delta_corr_lambda', float('nan')):+.4f})")
        lines.append(f"- corr(edge, over): {r['corr_edge_vs_actual_over']}")
        lines.append(f"- goals bias: {r['goals_bias']:+.3f}")
        lines.append(f"- experiment: `{r['experiment_id']}`")
        lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
