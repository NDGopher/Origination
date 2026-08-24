#!/usr/bin/env python
"""
Iteration 5 grid: OU calibration + congestion/rest totals intensity.

Runs on top of current best stack (card_bias, coaching, motivation).
Primary rank: 1X2 LL. Secondary: OU LL + OU ROI.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from loguru import logger

from origination.backtesting import run_walk_forward
from origination.backtesting.multi_market_report import (
    build_multi_market_report,
    save_multi_market_report,
)
from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.utils import load_config, resolve_data_dir, resolve_experiments_dir, set_global_seed, setup_logging


def _run(cfg0, matches, exp_dir, label: str, *, ou_method: str = "none", cong: float = 0.0, rest: float = 0.0):
    cfg = copy.deepcopy(cfg0)
    cfg["project"]["experiment_label"] = label
    cfg.setdefault("model", {}).setdefault("calibration", {})["ou_method"] = ou_method
    adj = cfg.setdefault("model", {}).setdefault("dixon_coles", {}).setdefault("intensity_adjustments", {})
    adj["enabled"] = True
    adj["congestion_coef"] = cong
    adj["rest_coef"] = rest
    # ensure ppda/deep kept
    adj.setdefault("ppda_coef", 0.01)
    adj.setdefault("deep_coef", 0.03)
    cfg.setdefault("backtest", {})["markets"] = ["1x2", "ou25", "ah"]

    logger.warning("=== {} | ou={} cong={} rest={} ===", label, ou_method, cong, rest)
    result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
    s = result.summary

    # Per-market OU ROI @ 3%
    test_m = matches[matches["match_id"].isin(result.predictions["match_id"])]
    bt = dict(cfg["backtest"])
    ou_bets = evaluate_predictions(result.predictions, test_m, {**bt, "markets": ["ou25"]}, edge_threshold=0.03)
    if len(ou_bets):
        stake = float(ou_bets["stake"].sum())
        profit = float(ou_bets["profit"].sum())
        ou_roi = profit / stake if stake > 0 else None
        ou_n = len(ou_bets)
        ou_edge = float(ou_bets["edge"].mean())
    else:
        ou_roi, ou_n, ou_edge = None, 0, None

    row = {
        "label": label,
        "ou_method": ou_method,
        "congestion_coef": cong,
        "rest_coef": rest,
        "log_loss_1x2": s.get("log_loss_1x2"),
        "log_loss_edge_vs_market": s.get("log_loss_edge_vs_market"),
        "log_loss_ou25": s.get("log_loss_ou25"),
        "log_loss_market_ou25": s.get("log_loss_market_ou25"),
        "log_loss_edge_vs_market_ou25": s.get("log_loss_edge_vs_market_ou25"),
        "roi_mixed": s.get("roi"),
        "ou_roi_3pct": ou_roi,
        "ou_n_bets_3pct": ou_n,
        "ou_avg_edge_3pct": ou_edge,
        "avg_clv_prob": s.get("avg_clv_prob"),
        "n_bets_mixed": s.get("n_bets"),
        "experiment_id": result.experiment_id,
    }
    print(json.dumps(row, indent=2, default=str))
    return row, result


def main() -> None:
    setup_logging("WARNING")
    set_global_seed(42)
    cfg0 = load_config(ROOT / "configs" / "default.yaml")
    # Force baseline OU/congestion off for clean grid
    cfg0.setdefault("model", {}).setdefault("calibration", {})["ou_method"] = "none"
    cfg0.setdefault("model", {}).setdefault("dixon_coles", {}).setdefault(
        "intensity_adjustments", {}
    ).update({"congestion_coef": 0.0, "rest_coef": 0.0})

    data_dir = resolve_data_dir(cfg0)
    exp_dir = resolve_experiments_dir(cfg0)
    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    matches = enrich_matches_with_understat_advanced(
        matches, load_understat_team_history(data_dir / "raw" / "understat")
    )

    specs = [
        ("iter5_baseline", {"ou_method": "none", "cong": 0.0, "rest": 0.0}),
        ("iter5_ou_temp", {"ou_method": "temperature", "cong": 0.0, "rest": 0.0}),
        ("iter5_ou_platt", {"ou_method": "platt", "cong": 0.0, "rest": 0.0}),
        ("iter5_cong_m0p05", {"ou_method": "none", "cong": -0.05, "rest": 0.0}),
        ("iter5_cong_m0p10", {"ou_method": "none", "cong": -0.10, "rest": 0.0}),
        ("iter5_cong_p0p05", {"ou_method": "none", "cong": 0.05, "rest": 0.0}),
        ("iter5_cong_p0p10", {"ou_method": "none", "cong": 0.10, "rest": 0.0}),
        ("iter5_rest_m0p05", {"ou_method": "none", "cong": 0.0, "rest": -0.05}),
        ("iter5_rest_p0p05", {"ou_method": "none", "cong": 0.0, "rest": 0.05}),
        ("iter5_rest_p0p10", {"ou_method": "none", "cong": 0.0, "rest": 0.10}),
        ("iter5_cong_m0p05_rest_p0p05", {"ou_method": "none", "cong": -0.05, "rest": 0.05}),
    ]

    rows = []
    results_by_label = {}
    for label, kw in specs:
        row, result = _run(cfg0, matches, exp_dir, label, **kw)
        rows.append(row)
        results_by_label[label] = result

    cmp = pd.DataFrame(rows).sort_values("log_loss_1x2")
    out = exp_dir / "ou_congestion_comparison_iter5.csv"
    cmp.to_csv(out, index=False)
    print("\n=== RANKED BY 1X2 LL ===")
    print(cmp.to_string(index=False))
    print("\n=== RANKED BY OU LL ===")
    print(cmp.sort_values("log_loss_ou25").to_string(index=False))
    print("Wrote", out)

    # Pick OU-calib winner (best OU LL among ou_method != none, 1X2 LL not worse than baseline+0.001)
    baseline_ll = float(cmp.loc[cmp["label"] == "iter5_baseline", "log_loss_1x2"].iloc[0])
    ou_cands = cmp[cmp["ou_method"] != "none"].copy()
    ou_cands = ou_cands[ou_cands["log_loss_1x2"] <= baseline_ll + 0.001]
    # Congestion cands: improve OU LL or OU ROI without hurting 1X2 much
    cong_cands = cmp[(cmp["congestion_coef"] != 0) | (cmp["rest_coef"] != 0)].copy()
    cong_cands = cong_cands[cong_cands["log_loss_1x2"] <= baseline_ll + 0.001]

    promote_ou = None
    if len(ou_cands):
        promote_ou = ou_cands.sort_values("log_loss_ou25").iloc[0]
        print("\nOU-calib candidate:", promote_ou["label"], "OU_LL", promote_ou["log_loss_ou25"])

    promote_cong = None
    if len(cong_cands):
        # Prefer better OU LL, then better OU ROI
        promote_cong = cong_cands.sort_values(["log_loss_ou25", "ou_roi_3pct"], ascending=[True, False]).iloc[0]
        print("Congestion candidate:", promote_cong["label"], "OU_LL", promote_cong["log_loss_ou25"])

    # Combined: best OU method + best cong if both help — run one more if needed
    if promote_ou is not None and promote_cong is not None:
        ou_m = str(promote_ou["ou_method"])
        cg = float(promote_cong["congestion_coef"])
        rs = float(promote_cong["rest_coef"])
        if ou_m != "none" and (cg != 0 or rs != 0):
            label = f"iter5_combo_{ou_m}_c{str(cg).replace('-','m').replace('.','p')}_r{str(rs).replace('-','m').replace('.','p')}"
            row, result = _run(cfg0, matches, exp_dir, label, ou_method=ou_m, cong=cg, rest=rs)
            rows.append(row)
            results_by_label[label] = result
            cmp = pd.DataFrame(rows).sort_values("log_loss_1x2")
            cmp.to_csv(out, index=False)
            print("\nCombo:", json.dumps(row, indent=2, default=str))

    # Multi-market reports for baseline + best overall by composite
    # Best: minimize 1X2 LL among those with OU LL <= baseline OU LL (or best OU ROI if LL tie)
    baseline_ou = float(cmp.loc[cmp["label"] == "iter5_baseline", "log_loss_ou25"].iloc[0])
    report_labels = ["iter5_baseline"]
    best_1x2 = cmp.iloc[0]["label"]
    if best_1x2 not in report_labels:
        report_labels.append(best_1x2)
    best_ou = cmp.sort_values("log_loss_ou25").iloc[0]["label"]
    if best_ou not in report_labels:
        report_labels.append(best_ou)
    # also combo if present
    for lab in cmp["label"]:
        if str(lab).startswith("iter5_combo") and lab not in report_labels:
            report_labels.append(lab)

    bt = dict(cfg0.get("backtest", {}))
    bt["markets"] = ["1x2", "ou25", "ah"]
    report_root = exp_dir / "multi_market_iter5"
    all_sum = []
    for lab in report_labels:
        if lab not in results_by_label:
            continue
        result = results_by_label[lab]
        tables = build_multi_market_report(result.predictions, matches, bt, label=lab)
        save_multi_market_report(tables, report_root / lab)
        save_multi_market_report(tables, exp_dir / result.experiment_id / "multi_market")
        all_sum.append(tables["summary"])
    if all_sum:
        pd.concat(all_sum, ignore_index=True).to_csv(report_root / "combined_summary.csv", index=False)
        print("Wrote multi-market reports to", report_root)


if __name__ == "__main__":
    main()
