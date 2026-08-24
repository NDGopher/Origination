#!/usr/bin/env python
"""
Iter14: multi-league totals modeling study.

Every variant is walk-forwarded on EPL + Bundesliga + La Liga.
Focus metrics: OU LL, short-band (1.60–2.50) LL gap vs market, mild-book OU ROI,
and 1X2 LL (must not collapse).

Variants (alone + combos):
- baseline (sum_* residual features always on via code)
- shot_volume intensity
- xg_allow intensity
- OU platt calibration
- higher alpha_ou
- volume + allow
- volume + alpha_ou
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting import run_walk_forward
from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.features.possession_value import enrich_matches_with_possession_value
from origination.utils import (
    load_config,
    resolve_data_dir,
    resolve_experiments_dir,
    set_global_seed,
    setup_logging,
)
from origination.utils.odds import two_way_fair


LEAGUES = [
    {
        "label": "EPL",
        "config": "configs/default.yaml",
        "aligned": "matches_aligned.parquet",
    },
    {
        "label": "Bundesliga",
        "config": "configs/league_D1_bundesliga.yaml",
        "aligned": "matches_aligned_D1.parquet",
    },
    {
        "label": "LaLiga",
        "config": "configs/league_SP1_la_liga.yaml",
        "aligned": "matches_aligned_SP1.parquet",
    },
]

VARIANTS = [
    {"label": "base"},
    {"label": "vol06", "shot_volume_coef": 0.06},
    {"label": "allow06", "xg_allow_coef": 0.06},
    {"label": "vol_allow", "shot_volume_coef": 0.06, "xg_allow_coef": 0.06},
    {"label": "ou_platt", "ou_method": "platt"},
    {"label": "aou15", "alpha_ou": 0.15},
    {"label": "vol_aou15", "shot_volume_coef": 0.06, "alpha_ou": 0.15},
]

MILD_FILTERS = {
    "enabled": True,
    "rules": [
        {"markets": ["1x2"], "max_odds": 2.00},
        {"markets": ["ou25"], "min_odds": 1.50, "max_odds": 3.00},
        {"markets": ["ah"], "min_odds": 1.50, "max_odds": 3.00},
    ],
}
MILD_EDGE = {"ou25": 0.05, "ah": 0.05}


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def short_band_metrics(preds: pd.DataFrame, matches: pd.DataFrame) -> dict:
    m = matches.set_index("match_id")
    pm, pk, y, short_odds = [], [], [], []
    for _, r in preds.iterrows():
        mid = r["match_id"]
        if mid not in m.index:
            continue
        match = m.loc[mid]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        o, u = match.get("close_over25"), match.get("close_under25")
        if not (pd.notna(o) and pd.notna(u)):
            continue
        o, u = float(o), float(u)
        try:
            fair_o, _ = two_way_fair(o, u, method="power")
        except Exception:
            continue
        s = min(o, u)
        if not (1.60 <= s <= 2.50):
            continue
        pm.append(float(r["p_over25"]))
        pk.append(float(fair_o))
        y.append(1.0 if float(match["total_goals"]) > 2.5 else 0.0)
        short_odds.append(s)
    if len(y) < 40:
        return {"short_n": len(y)}
    pm_a, pk_a, y_a = np.asarray(pm), np.asarray(pk), np.asarray(y)
    return {
        "short_n": int(len(y)),
        "short_ll_model": _ll(pm_a, y_a),
        "short_ll_mkt": _ll(pk_a, y_a),
        "short_ll_gap": _ll(pm_a, y_a) - _ll(pk_a, y_a),
        "short_bias": float((pm_a - y_a).mean()),
    }


def _roi(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None
    st = float(df["stake"].sum())
    return float(df["profit"].sum()) / st if st else None


def mild_ou_roi(preds, matches) -> dict:
    bt = {
        "markets": ["1x2", "ou25", "ah"],
        "edge_threshold": 0.03,
        "edge_threshold_by_market": MILD_EDGE,
        "bet_filters": MILD_FILTERS,
        "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
        "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
    }
    bets = evaluate_predictions(preds, matches, bt, edge_threshold=0.03)
    ou = bets[bets["market"] == "ou25"]
    short = ou[(ou["close_odds"] >= 1.60) & (ou["close_odds"] <= 2.50)]
    return {
        "mild_n_ou": int(len(ou)),
        "mild_roi_ou": _roi(ou),
        "mild_n_short": int(len(short)),
        "mild_roi_short": _roi(short),
        "mild_roi_all": _roi(bets),
        "mild_n_all": int(len(bets)),
    }


def apply_variant(cfg: dict, v: dict) -> dict:
    c = copy.deepcopy(cfg)
    adj = c.setdefault("model", {}).setdefault("dixon_coles", {}).setdefault(
        "intensity_adjustments", {}
    )
    adj["shot_volume_coef"] = float(v.get("shot_volume_coef", 0.0))
    adj["xg_allow_coef"] = float(v.get("xg_allow_coef", 0.0))
    adj["shot_volume_center"] = 2.4
    adj["xg_allow_center"] = 2.4
    # Keep PV off for this study
    adj["pv_coef"] = 0.0
    if "ou_method" in v:
        c.setdefault("model", {}).setdefault("calibration", {})["ou_method"] = v["ou_method"]
    if "alpha_ou" in v:
        c.setdefault("model", {}).setdefault("residual", {})["alpha_ou"] = float(v["alpha_ou"])
    # Mild filters for WF internal book
    c.setdefault("backtest", {})["bet_filters"] = MILD_FILTERS
    c["backtest"]["edge_threshold_by_market"] = MILD_EDGE
    return c


def main() -> None:
    setup_logging("INFO")
    set_global_seed(42)
    data_dir = resolve_data_dir(load_config(ROOT / "configs" / "default.yaml"))
    exp_dir = resolve_experiments_dir(load_config(ROOT / "configs" / "default.yaml"))
    out_csv = exp_dir / "iter14_totals_multileague.csv"
    hist = load_understat_team_history(data_dir / "raw" / "understat")
    pv_path = data_dir / "interim" / "understat_possession_value.parquet"
    pv = pd.read_parquet(pv_path) if pv_path.exists() else None

    rows = []
    if out_csv.exists():
        rows = pd.read_csv(out_csv).to_dict(orient="records")

    done = {(r["league"], r["label"]) for r in rows}

    for league in LEAGUES:
        base_cfg = load_config(ROOT / league["config"])
        matches = load_aligned(data_dir / "interim" / league["aligned"])
        if base_cfg.get("features", {}).get("groups", {}).get("understat_advanced", False):
            matches = enrich_matches_with_understat_advanced(matches, hist)
        if pv is not None and base_cfg.get("features", {}).get("groups", {}).get(
            "possession_value", False
        ):
            matches = enrich_matches_with_possession_value(matches, pv)

        for v in VARIANTS:
            key = (league["label"], v["label"])
            if key in done:
                print("SKIP done", key)
                continue
            cfg = apply_variant(base_cfg, v)
            cfg.setdefault("project", {})["experiment_label"] = (
                f"iter14_{league['label']}_{v['label']}"
            )
            print(f"\n=== {league['label']} / {v['label']} ===")
            result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
            s = result.summary
            short = short_band_metrics(result.predictions, matches)
            mild = mild_ou_roi(result.predictions, matches)
            row = {
                "league": league["label"],
                "label": v["label"],
                "experiment_id": result.experiment_id,
                "log_loss_1x2": s.get("log_loss_1x2"),
                "log_loss_ou25": s.get("log_loss_ou25"),
                "ou_gap": s.get("log_loss_edge_vs_market_ou25"),
                **short,
                **mild,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(json.dumps(row, indent=2, default=str))
            done.add(key)

    # Cross-league rank by mean OU LL improvement vs base per league
    df = pd.DataFrame(rows)
    base_ll = df[df["label"] == "base"].set_index("league")["log_loss_ou25"]
    df["d_ou_ll_vs_base"] = df.apply(
        lambda r: float(r["log_loss_ou25"]) - float(base_ll.get(r["league"], np.nan)),
        axis=1,
    )
    df["d_short_gap_vs_base"] = df.apply(
        lambda r: (
            float(r.get("short_ll_gap", np.nan))
            - float(
                df[(df["league"] == r["league"]) & (df["label"] == "base")][
                    "short_ll_gap"
                ].iloc[0]
            )
            if r["label"] != "base"
            and "short_ll_gap" in r
            and pd.notna(r.get("short_ll_gap"))
            and len(df[(df["league"] == r["league"]) & (df["label"] == "base")])
            else 0.0
        ),
        axis=1,
    )
    df.to_csv(out_csv, index=False)

    rank = (
        df.groupby("label")
        .agg(
            mean_ou_ll=("log_loss_ou25", "mean"),
            mean_d_ou_ll=("d_ou_ll_vs_base", "mean"),
            mean_short_gap=("short_ll_gap", "mean"),
            mean_mild_ou=("mild_roi_ou", "mean"),
            mean_1x2_ll=("log_loss_1x2", "mean"),
            n_leagues=("league", "nunique"),
        )
        .sort_values("mean_d_ou_ll")
    )
    rank.to_csv(exp_dir / "iter14_totals_rank.csv")
    print(rank)
    print("Wrote", out_csv)


if __name__ == "__main__":
    main()
