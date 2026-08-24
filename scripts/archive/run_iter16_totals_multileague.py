#!/usr/bin/env python
"""
Iter16: Bundesliga-safe totals intercept + re-test sidelined layers on new base.

New base = xg_allow=0.06 + totals_intercept signed on EPL/La Liga (D1 off in configs).
EPL_aggressive pack scored on EPL only; rules never modified.
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
    {"label": "EPL", "config": "configs/default.yaml", "aligned": "matches_aligned.parquet"},
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

# base = config files as-is (EPL/SP1 signed intercept on; D1 off)
VARIANTS = [
    {"label": "base"},
    # D1-safe intercept modes (force on all leagues to measure portability)
    {"label": "int_lift", "totals_intercept": True, "totals_mode": "lift_only"},
    {
        "label": "int_asym",
        "totals_intercept": True,
        "totals_mode": "asymmetric",
        "dampen_shrink": 1.0,
    },
    {
        "label": "int_thresh05",
        "totals_intercept": True,
        "totals_mode": "signed",
        "min_abs_raw": 0.05,
    },
    # Sidelined layers re-tested on config base
    {"label": "vol06", "shot_volume_coef": 0.06},
    {"label": "aou15", "alpha_ou": 0.15},
    {"label": "suppress04", "suppress_resid_coef": 0.04},
    {"label": "shortw2", "ou_short_weight": 2.0},
    # Combos with asymmetric intercept (candidate universal)
    {
        "label": "asym_aou",
        "totals_intercept": True,
        "totals_mode": "asymmetric",
        "dampen_shrink": 1.0,
        "alpha_ou": 0.15,
    },
    {
        "label": "asym_vol",
        "totals_intercept": True,
        "totals_mode": "asymmetric",
        "dampen_shrink": 1.0,
        "shot_volume_coef": 0.06,
    },
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

EPL_PACK = {
    "enabled": True,
    "rules": [
        {"markets": ["1x2"], "max_odds": 1.80},
        {"markets": ["ou25"], "min_odds": 2.00, "max_odds": 4.00, "allow_sides": ["under"]},
        {"markets": ["ah"], "max_odds": 1.90},
    ],
}
EPL_EDGE = {"ou25": 0.08, "ah": 0.05}


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _roi(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None
    st = float(df["stake"].sum())
    return float(df["profit"].sum()) / st if st else None


def join_ou(preds: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.set_index("match_id")
    rows = []
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
        y = 1.0 if float(match["total_goals"]) > 2.5 else 0.0
        lam = float(r.get("lambda_home", np.nan)) + float(r.get("lambda_away", np.nan))
        rows.append(
            {
                "y": y,
                "p": float(r["p_over25"]),
                "fair": float(fair_o),
                "short": min(o, u),
                "sum_lambda": lam,
                "goals": float(match["total_goals"]),
                "odds_over": o,
            }
        )
    return pd.DataFrame(rows)


def short_and_corr(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 40:
        return {"n_ou": 0 if df is None else int(len(df))}
    y, pm, pk = df["y"].to_numpy(), df["p"].to_numpy(), df["fair"].to_numpy()
    short = df[(df["short"] >= 1.60) & (df["short"] <= 2.50)]
    overs = df[df["odds_over"] <= df["short"] + 1e-9]  # over is short side
    out = {
        "n_ou": int(len(df)),
        "ll_gap": _ll(pm, y) - _ll(pk, y),
        "bias": float((pm - y).mean()),
        "goals_err": float((df["sum_lambda"] - df["goals"]).mean()),
        "corr_lam_goals": float(df["sum_lambda"].corr(df["goals"])),
    }
    if len(short) >= 40:
        ys, pms, pks = short["y"].to_numpy(), short["p"].to_numpy(), short["fair"].to_numpy()
        out.update(
            {
                "short_n": int(len(short)),
                "short_ll_gap": _ll(pms, ys) - _ll(pks, ys),
                "short_bias": float((pms - ys).mean()),
                "short_corr_lam_goals": float(short["sum_lambda"].corr(short["goals"])),
            }
        )
    if len(overs) >= 40:
        yo, po, ko = overs["y"].to_numpy(), overs["p"].to_numpy(), overs["fair"].to_numpy()
        out["over_fav_ll_gap"] = _ll(po, yo) - _ll(ko, yo)
        out["over_fav_n"] = int(len(overs))
    return out


def score_book(preds, matches, filt, edge) -> dict:
    bt = {
        "markets": ["1x2", "ou25", "ah"],
        "edge_threshold": 0.03,
        "edge_threshold_by_market": edge,
        "bet_filters": filt,
        "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
        "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
    }
    bets = evaluate_predictions(preds, matches, bt, edge_threshold=0.03)
    ou = bets[bets["market"] == "ou25"]
    short = ou[(ou["close_odds"] >= 1.60) & (ou["close_odds"] <= 2.50)]
    overs = ou[ou["side"] == "over"]
    return {
        "n_ou": int(len(ou)),
        "roi_ou": _roi(ou),
        "n_short": int(len(short)),
        "roi_short": _roi(short),
        "n_over": int(len(overs)),
        "roi_over": _roi(overs),
        "roi_all": _roi(bets),
        "n_all": int(len(bets)),
    }


def apply_variant(cfg: dict, v: dict) -> dict:
    c = copy.deepcopy(cfg)
    adj = c.setdefault("model", {}).setdefault("dixon_coles", {}).setdefault(
        "intensity_adjustments", {}
    )
    if "shot_volume_coef" in v:
        adj["shot_volume_coef"] = float(v["shot_volume_coef"])
    if "suppress_resid_coef" in v:
        adj["suppress_resid_coef"] = float(v["suppress_resid_coef"])
    if "tempo_ppda_coef" in v:
        adj["tempo_ppda_coef"] = float(v["tempo_ppda_coef"])
    if "pv_open_orth_coef" in v:
        adj["pv_open_orth_coef"] = float(v["pv_open_orth_coef"])
    if "xg_allow_coef" in v:
        adj["xg_allow_coef"] = float(v["xg_allow_coef"])

    if "totals_intercept" in v or "totals_mode" in v:
        ti = c["model"]["dixon_coles"].setdefault("totals_intercept", {})
        if "totals_intercept" in v:
            ti["enabled"] = bool(v["totals_intercept"])
        if "totals_mode" in v:
            ti["mode"] = str(v["totals_mode"])
        if "dampen_shrink" in v:
            ti["dampen_shrink"] = float(v["dampen_shrink"])
        if "min_abs_raw" in v:
            ti["min_abs_raw"] = float(v["min_abs_raw"])
        if "totals_shrink" in v:
            ti["shrink"] = float(v["totals_shrink"])
        c.setdefault("model", {}).setdefault("hierarchical", {})["totals_intercept"] = ti.get(
            "enabled", False
        )

    if "alpha_ou" in v:
        c.setdefault("model", {}).setdefault("residual", {})["alpha_ou"] = float(v["alpha_ou"])
    if "ou_short_weight" in v:
        c.setdefault("model", {}).setdefault("residual", {})["ou_short_weight"] = float(
            v["ou_short_weight"]
        )
    if v.get("possession_value"):
        c.setdefault("features", {}).setdefault("groups", {})["possession_value"] = True

    c.setdefault("backtest", {})["bet_filters"] = MILD_FILTERS
    c["backtest"]["edge_threshold_by_market"] = MILD_EDGE
    return c


def main() -> None:
    setup_logging("INFO")
    set_global_seed(42)
    base_cfg0 = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(base_cfg0)
    exp_dir = resolve_experiments_dir(base_cfg0)
    out_csv = exp_dir / "iter16_totals_multileague.csv"
    hist = load_understat_team_history(data_dir / "raw" / "understat")
    pv_path = data_dir / "interim" / "understat_possession_value.parquet"
    pv = pd.read_parquet(pv_path) if pv_path.exists() else None

    rows = pd.read_csv(out_csv).to_dict(orient="records") if out_csv.exists() else []
    done = {(r["league"], r["label"]) for r in rows}

    for league in LEAGUES:
        base_cfg = load_config(ROOT / league["config"])
        matches = load_aligned(data_dir / "interim" / league["aligned"])
        if base_cfg.get("features", {}).get("groups", {}).get("understat_advanced", False):
            matches = enrich_matches_with_understat_advanced(matches, hist)

        for v in VARIANTS:
            key = (league["label"], v["label"])
            if key in done:
                print("SKIP done", key)
                continue
            cfg = apply_variant(base_cfg, v)
            if cfg.get("features", {}).get("groups", {}).get("possession_value") and pv is not None:
                matches_run = enrich_matches_with_possession_value(matches.copy(), pv)
            else:
                matches_run = matches
            cfg.setdefault("project", {})["experiment_label"] = (
                f"iter16_{league['label']}_{v['label']}"
            )
            print(f"\n=== {league['label']} / {v['label']} ===")
            result = run_walk_forward(matches_run, cfg, experiments_dir=exp_dir)
            s = result.summary
            diag = short_and_corr(join_ou(result.predictions, matches_run))
            mild = score_book(result.predictions, matches_run, MILD_FILTERS, MILD_EDGE)
            row = {
                "league": league["label"],
                "label": v["label"],
                "experiment_id": result.experiment_id,
                "log_loss_1x2": s.get("log_loss_1x2"),
                "log_loss_ou25": s.get("log_loss_ou25"),
                "ou_gap": s.get("log_loss_edge_vs_market_ou25"),
                **{f"diag_{k}": val for k, val in diag.items()},
                **{f"mild_{k}": val for k, val in mild.items()},
            }
            if league["label"] == "EPL":
                epl = score_book(result.predictions, matches_run, EPL_PACK, EPL_EDGE)
                row.update({f"eplpack_{k}": val for k, val in epl.items()})
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(json.dumps(row, indent=2, default=str))
            done.add(key)

    df = pd.DataFrame(rows)
    base_ll = df[df["label"] == "base"].set_index("league")["log_loss_ou25"]
    base_gap = df[df["label"] == "base"].set_index("league")["diag_short_ll_gap"]
    df["d_ou_ll_vs_base"] = df.apply(
        lambda r: float(r["log_loss_ou25"]) - float(base_ll.get(r["league"], np.nan)),
        axis=1,
    )
    df["d_short_gap_vs_base"] = df.apply(
        lambda r: (
            float(r["diag_short_ll_gap"]) - float(base_gap.get(r["league"], np.nan))
            if pd.notna(r.get("diag_short_ll_gap"))
            else np.nan
        ),
        axis=1,
    )
    df.to_csv(out_csv, index=False)
    rank = (
        df.groupby("label")
        .agg(
            mean_ou_ll=("log_loss_ou25", "mean"),
            mean_d_ou_ll=("d_ou_ll_vs_base", "mean"),
            mean_short_gap=("diag_short_ll_gap", "mean"),
            mean_d_short=("d_short_gap_vs_base", "mean"),
            mean_mild_ou=("mild_roi_ou", "mean"),
            mean_mild_over=("mild_roi_over", "mean"),
            mean_corr=("diag_corr_lam_goals", "mean"),
            mean_1x2_ll=("log_loss_1x2", "mean"),
            n_leagues=("league", "nunique"),
        )
        .sort_values("mean_d_short")
    )
    rank.to_csv(exp_dir / "iter16_totals_rank.csv")
    print(rank)
    print("Wrote", out_csv)


if __name__ == "__main__":
    main()
