#!/usr/bin/env python
"""Iter12: refine EPL OU pocket, AH filters, league rule packs (fast re-score)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting.performance_autopsy import (
    build_autopsy_tables,
    enrich_bets_for_autopsy,
    save_autopsy,
    summarize_bets,
    write_autopsy_summary,
)
from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.utils import load_config, resolve_data_dir, setup_logging


SPECS = [
    ("EPL", "20260805T212804Z_iter10_hier_baseline", "matches_aligned.parquet"),
    ("Championship", "20260805T200446Z_league_E1_champ_iter8", "matches_aligned_E1.parquet"),
    ("Bundesliga", "20260805T193621Z_league_D1_xg_resid", "matches_aligned_D1.parquet"),
    ("SerieA", "20260805T194400Z_league_I1_serie_a", "matches_aligned_I1.parquet"),
    ("LaLiga", "20260805T195244Z_league_SP1_la_liga", "matches_aligned_SP1.parquet"),
]

# 1X2 short filter always on for portfolio realism
BASE_1X2 = {"markets": ["1x2"], "max_odds": 1.80}


def _roi(df: pd.DataFrame) -> float:
    if df is None or len(df) == 0:
        return float("nan")
    st = float(df["stake"].sum())
    return float(df["profit"].sum()) / st if st else float("nan")


def _pack(rules: list, edge_by: dict | None = None) -> tuple[dict, dict]:
    filt = {"enabled": True, "rules": rules}
    return filt, dict(edge_by or {})


# --- EPL OU refinement candidates ---
EPL_OU_VARIANTS = [
    ("iter11_promote", [BASE_1X2, {"markets": ["ou25"], "min_odds": 2.00}], {"ou25": 0.08, "ah": 0.05}),
    ("ou_under_only", [BASE_1X2, {"markets": ["ou25"], "min_odds": 2.00, "allow_sides": ["under"]}], {"ou25": 0.08, "ah": 0.05}),
    ("ou_under_max4", [BASE_1X2, {"markets": ["ou25"], "min_odds": 2.00, "max_odds": 4.00, "allow_sides": ["under"]}], {"ou25": 0.08, "ah": 0.05}),
    ("ou_under_dogs", [BASE_1X2, {"markets": ["ou25"], "min_odds": 2.70, "max_odds": 4.00, "allow_sides": ["under"]}], {"ou25": 0.08, "ah": 0.05}),
    ("ou_edge08_12", [BASE_1X2, {"markets": ["ou25"], "min_odds": 2.00, "max_odds": 4.00}], {"ou25": 0.08, "ah": 0.05}),  # then post-cap edge<=0.12
    ("ou_edge10", [BASE_1X2, {"markets": ["ou25"], "min_odds": 2.00}], {"ou25": 0.10, "ah": 0.05}),
    ("ou_under_edge10", [BASE_1X2, {"markets": ["ou25"], "min_odds": 2.00, "max_odds": 4.00, "allow_sides": ["under"]}], {"ou25": 0.10, "ah": 0.05}),
    ("ou_under_edge08", [BASE_1X2, {"markets": ["ou25"], "min_odds": 2.00, "max_odds": 4.00, "allow_sides": ["under"]}], {"ou25": 0.08, "ah": 0.05}),
    ("ou_min2p2_under", [BASE_1X2, {"markets": ["ou25"], "min_odds": 2.20, "max_odds": 4.00, "allow_sides": ["under"]}], {"ou25": 0.08, "ah": 0.05}),
    ("ou_dogs_both", [BASE_1X2, {"markets": ["ou25"], "min_odds": 2.70, "max_odds": 4.00}], {"ou25": 0.08, "ah": 0.05}),
]

# AH variants on top of best-looking OU (filled after OU grid; also standalone)
AH_VARIANTS = [
    ("ah_edge05", {"ah": 0.05}),
    ("ah_edge06", {"ah": 0.06}),
    ("ah_edge08", {"ah": 0.08}),
    ("ah_edge10", {"ah": 0.10}),
    ("ah_max_1.95", {"ah": 0.05}),  # + max_odds rule
    ("ah_max_1.90", {"ah": 0.05}),
    ("ah_edge08_max195", {"ah": 0.08}),
    ("ah_off", {}),  # drop AH from markets
]

# League-specific OU packs to grid
LEAGUE_OU_GRIDS = {
    "EPL": [
        ("promote", [{"markets": ["ou25"], "min_odds": 2.00}], {"ou25": 0.08}),
        ("under_max4", [{"markets": ["ou25"], "min_odds": 2.00, "max_odds": 4.00, "allow_sides": ["under"]}], {"ou25": 0.08}),
        ("under_dogs", [{"markets": ["ou25"], "min_odds": 2.70, "max_odds": 4.00, "allow_sides": ["under"]}], {"ou25": 0.08}),
    ],
    "Championship": [
        ("raw_ou", [], {"ou25": 0.03}),
        ("ou_edge08", [], {"ou25": 0.08}),
        ("ou_min2", [{"markets": ["ou25"], "min_odds": 2.00}], {"ou25": 0.08}),
        ("ou_under", [{"markets": ["ou25"], "allow_sides": ["under"]}], {"ou25": 0.05}),
        ("ou_over", [{"markets": ["ou25"], "allow_sides": ["over"]}], {"ou25": 0.05}),
        ("ou_edge10", [], {"ou25": 0.10}),
    ],
    "Bundesliga": [
        ("raw_ou", [], {"ou25": 0.03}),
        ("ou_over", [{"markets": ["ou25"], "allow_sides": ["over"]}], {"ou25": 0.05}),
        ("ou_over_edge08", [{"markets": ["ou25"], "allow_sides": ["over"]}], {"ou25": 0.08}),
        ("ou_edge08", [], {"ou25": 0.08}),
        ("ou_min2", [{"markets": ["ou25"], "min_odds": 2.00}], {"ou25": 0.08}),
        ("ou_under", [{"markets": ["ou25"], "allow_sides": ["under"]}], {"ou25": 0.08}),
    ],
    "SerieA": [
        ("raw_ou", [], {"ou25": 0.03}),
        ("ou_edge08", [], {"ou25": 0.08}),
        ("ou_min2", [{"markets": ["ou25"], "min_odds": 2.00}], {"ou25": 0.08}),
        ("ou_under", [{"markets": ["ou25"], "allow_sides": ["under"]}], {"ou25": 0.08}),
        ("ou_over", [{"markets": ["ou25"], "allow_sides": ["over"]}], {"ou25": 0.08}),
        ("ou_edge10", [], {"ou25": 0.10}),
    ],
    "LaLiga": [
        ("raw_ou", [], {"ou25": 0.03}),
        ("ou_edge08", [], {"ou25": 0.08}),
        ("ou_under", [{"markets": ["ou25"], "allow_sides": ["under"]}], {"ou25": 0.05}),
        ("ou_over", [{"markets": ["ou25"], "allow_sides": ["over"]}], {"ou25": 0.08}),
        ("ou_min2", [{"markets": ["ou25"], "min_odds": 2.00}], {"ou25": 0.08}),
        ("ah_focus", [], {"ou25": 0.10, "ah": 0.03}),  # keep OU tight, AH looser
    ],
}

LA_LIGA_AH = [
    ("ah_raw_e03", {"ah": 0.03}),
    ("ah_e05", {"ah": 0.05}),
    ("ah_e06", {"ah": 0.06}),
    ("ah_e08", {"ah": 0.08}),
    ("ah_max195_e05", {"ah": 0.05}),
    ("ah_max190_e05", {"ah": 0.05}),
]


def eval_book(preds, matches, cfg, rules, edge_by, *, markets=None, max_edge_ou=None):
    bt = copy.deepcopy(cfg.get("backtest", {}))
    bt["bet_filters"] = {"enabled": True, "rules": rules}
    bt["edge_threshold_by_market"] = edge_by
    if markets is not None:
        bt["markets"] = markets
    bets = evaluate_predictions(preds, matches, bt, edge_threshold=0.03)
    if max_edge_ou is not None and len(bets):
        mask = ~((bets["market"] == "ou25") & (bets["edge"] > max_edge_ou))
        bets = bets[mask].copy()
    return bets


def main() -> None:
    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    out = ROOT / "experiments" / "iter12_totals_refine"
    out.mkdir(parents=True, exist_ok=True)

    # Load EPL
    epl_preds = pd.read_parquet(ROOT / "experiments" / SPECS[0][1] / "predictions.parquet")
    epl_m = load_aligned(data_dir / "interim" / SPECS[0][2])

    # ---- 1) Deep autopsy of current positive OU ----
    bets0 = eval_book(
        epl_preds,
        epl_m,
        cfg,
        [BASE_1X2, {"markets": ["ou25"], "min_odds": 2.00}],
        {"ou25": 0.08, "ah": 0.05},
    )
    ou0 = bets0[bets0["market"] == "ou25"]
    tables = build_autopsy_tables(ou0, epl_m, label="epl_ou_positive_pocket")
    save_autopsy(tables, out / "autopsy_ou_pocket")
    write_autopsy_summary(tables, out / "autopsy_ou_pocket" / "SUMMARY.md", label="EPL OU positive pocket")
    en = enrich_bets_for_autopsy(ou0, epl_m)
    season = (
        en.groupby("season")
        .apply(
            lambda g: pd.Series(
                {"n": len(g), "roi": g["profit"].sum() / g["stake"].sum(), "hit": g["won"].mean()}
            ),
            include_groups=False,
        )
        .reset_index()
    )
    season.to_csv(out / "ou_pocket_by_season.csv", index=False)

    # ---- 2) EPL OU refinement grid ----
    ou_rows = []
    for label, rules, edge_by in EPL_OU_VARIANTS:
        max_e = 0.12 if label == "ou_edge08_12" else None
        # include AH at edge 0.05 for ALL unless label says otherwise
        full_rules = list(rules)
        bets = eval_book(epl_preds, epl_m, cfg, full_rules, edge_by, max_edge_ou=max_e)
        bo = bets[bets["market"] == "ou25"]
        ba = bets[bets["market"] == "ah"]
        b1 = bets[bets["market"] == "1x2"]
        row = {
            "label": label,
            "n_ou": len(bo),
            "roi_ou": _roi(bo),
            "hit_ou": float(bo["won"].mean()) if len(bo) else np.nan,
            "avg_odds_ou": float(bo["close_odds"].mean()) if len(bo) else np.nan,
            "t_ou": summarize_bets(bo).get("t_stat"),
            "n_ah": len(ba),
            "roi_ah": _roi(ba),
            "n_1x2": len(b1),
            "roi_1x2": _roi(b1),
            "n_all": len(bets),
            "roi_all": _roi(bets),
        }
        ou_rows.append(row)
        print(
            f"OU {label:20s} n={row['n_ou']:4d} ROI={row['roi_ou']:+.2%} t={row['t_ou']:.2f} "
            f"ALL={row['roi_all']:+.2%}"
            if row["n_ou"]
            else f"OU {label}: empty"
        )
    ou_df = pd.DataFrame(ou_rows).sort_values("roi_ou", ascending=False)
    ou_df.to_csv(out / "epl_ou_refine_grid.csv", index=False)

    best_ou_label = str(ou_df.iloc[0]["label"])
    best_ou_rules = next(r for lab, r, _ in EPL_OU_VARIANTS if lab == best_ou_label)
    best_ou_edge = next(e for lab, _, e in EPL_OU_VARIANTS if lab == best_ou_label)
    print("Best OU label:", best_ou_label)

    # ---- 3) AH grid on best OU stack ----
    ah_rows = []
    for label, edge_extra in AH_VARIANTS:
        edge_by = {**best_ou_edge, **edge_extra}
        rules = list(best_ou_rules)
        markets = ["1x2", "ou25", "ah"]
        if label == "ah_off":
            markets = ["1x2", "ou25"]
        if "max_1.95" in label or label.endswith("max195"):
            rules = rules + [{"markets": ["ah"], "max_odds": 1.95}]
        if "max_1.90" in label or label.endswith("max190"):
            rules = rules + [{"markets": ["ah"], "max_odds": 1.90}]
        bets = eval_book(epl_preds, epl_m, cfg, rules, edge_by, markets=markets)
        bo = bets[bets["market"] == "ou25"]
        ba = bets[bets["market"] == "ah"]
        row = {
            "label": label,
            "n_ou": len(bo),
            "roi_ou": _roi(bo),
            "n_ah": len(ba),
            "roi_ah": _roi(ba),
            "hit_ah": float(ba["won"].mean()) if len(ba) else np.nan,
            "n_all": len(bets),
            "roi_all": _roi(bets),
            "t_ah": summarize_bets(ba).get("t_stat") if len(ba) else np.nan,
        }
        ah_rows.append(row)
        print(
            f"AH {label:20s} AH n={row['n_ah']:4d} ROI={row['roi_ah']:+.2%} ALL={row['roi_all']:+.2%}"
            if row["n_ah"] or label == "ah_off"
            else f"AH {label}: empty"
        )
    ah_df = pd.DataFrame(ah_rows).sort_values("roi_all", ascending=False)
    ah_df.to_csv(out / "epl_ah_refine_grid.csv", index=False)

    # AH autopsy on current promoted stack
    ah_bets = bets0[bets0["market"] == "ah"]
    ah_tables = build_autopsy_tables(ah_bets, epl_m, label="epl_ah_filtered")
    save_autopsy(ah_tables, out / "autopsy_ah")
    write_autopsy_summary(ah_tables, out / "autopsy_ah" / "SUMMARY.md", label="EPL AH on filtered stack")

    # ---- 4) League-specific packs ----
    league_rows = []
    for league, eid, aligned in SPECS:
        exp = ROOT / "experiments" / eid
        if not (exp / "predictions.parquet").exists():
            continue
        preds = pd.read_parquet(exp / "predictions.parquet")
        matches = load_aligned(data_dir / "interim" / aligned)
        for ou_lab, ou_rules, ou_edge in LEAGUE_OU_GRIDS.get(league, []):
            rules = [BASE_1X2] + ou_rules
            # AH: La Liga keep mild; others edge 0.05 default if ah in edge
            edge_by = {"ah": 0.05, **ou_edge}
            if league == "LaLiga" and "ah" not in ou_edge:
                edge_by["ah"] = 0.05
            bets = eval_book(preds, matches, cfg, rules, edge_by)
            bo = bets[bets["market"] == "ou25"]
            ba = bets[bets["market"] == "ah"]
            league_rows.append(
                {
                    "league": league,
                    "pack": ou_lab,
                    "n_ou": len(bo),
                    "roi_ou": _roi(bo),
                    "hit_ou": float(bo["won"].mean()) if len(bo) else np.nan,
                    "n_ah": len(ba),
                    "roi_ah": _roi(ba),
                    "roi_all": _roi(bets),
                    "n_all": len(bets),
                }
            )
        # La Liga AH focus
        if league == "LaLiga":
            for ah_lab, ah_edge in LA_LIGA_AH:
                rules = [BASE_1X2]
                edge_by = {"ou25": 0.10, **ah_edge}
                extra = []
                if "max195" in ah_lab:
                    extra = [{"markets": ["ah"], "max_odds": 1.95}]
                if "max190" in ah_lab:
                    extra = [{"markets": ["ah"], "max_odds": 1.90}]
                bets = eval_book(preds, matches, cfg, rules + extra, edge_by)
                ba = bets[bets["market"] == "ah"]
                league_rows.append(
                    {
                        "league": league,
                        "pack": ah_lab,
                        "n_ou": len(bets[bets.market == "ou25"]),
                        "roi_ou": _roi(bets[bets.market == "ou25"]),
                        "hit_ou": np.nan,
                        "n_ah": len(ba),
                        "roi_ah": _roi(ba),
                        "roi_all": _roi(bets),
                        "n_all": len(bets),
                    }
                )

    lg_df = pd.DataFrame(league_rows)
    lg_df.to_csv(out / "league_rule_packs.csv", index=False)

    # Best pack per league by OU ROI (n>=80) else ALL
    best_lines = ["# Iter12 league rule packs\n"]
    for league, g in lg_df.groupby("league"):
        viable = g[g["n_ou"] >= 80].sort_values("roi_ou", ascending=False)
        if len(viable) == 0:
            viable = g.sort_values("roi_all", ascending=False)
        best = viable.iloc[0]
        best_lines.append(
            f"- **{league}**: `{best['pack']}` OU={best['roi_ou']:+.2%} (n={int(best['n_ou'])}) "
            f"AH={best['roi_ah']:+.2%} ALL={best['roi_all']:+.2%}\n"
        )
        print(best_lines[-1].strip())
    (out / "LEAGUE_PACKS.md").write_text("".join(best_lines), encoding="utf-8")

    # Summary
    summary = {
        "best_epl_ou": ou_df.iloc[0].to_dict(),
        "best_epl_all_from_ah_grid": ah_df.iloc[0].to_dict(),
        "iter11_ou_baseline": next(r for r in ou_rows if r["label"] == "iter11_promote"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
