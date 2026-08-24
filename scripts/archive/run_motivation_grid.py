#!/usr/bin/env python
"""Grid table-position motivation intensity coefs on top of current best stack."""

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
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.utils import load_config, resolve_data_dir, resolve_experiments_dir, set_global_seed, setup_logging


MOTIVATION_BASE = {
    "enabled": True,
    "season_length": 38,
    "safety_rank": 17,
    "title_pts_gap": 6.0,
    "releg_pts_gap": 6.0,
    "euro_pts_gap": 6.0,
    "min_games": 8,
    "late_games_left": 12,
    "title_coef": 0.0,
    "releg_coef": 0.0,
    "dead_rubber_coef": 0.0,
    "stakes_coef": 0.0,
    "motivation_diff_coef": 0.0,
}


def _run(cfg0, matches, exp_dir, label: str, motiv: dict) -> dict:
    cfg = copy.deepcopy(cfg0)
    cfg["project"]["experiment_label"] = label
    ctx = cfg.setdefault("features", {}).setdefault("context_adjustments", {})
    ctx["enabled"] = True
    # Keep promoted stack
    ctx.setdefault(
        "referee",
        {"enabled": True, "min_prior_games": 5, "tempo_coef": 0.0, "card_bias_coef": 0.5},
    )
    ctx.setdefault(
        "coaching_change",
        {
            "enabled": True,
            "new_coach_days": 60,
            "new_coach_games": 8,
            "bounce_coef": 0.05,
            "changes_path": "data/interim/coaching_changes.csv",
        },
    )
    ctx["motivation"] = {**MOTIVATION_BASE, **motiv}
    for k in ("injuries", "lineups", "formations", "weather", "travel"):
        ctx.setdefault(k, {"enabled": False})
    logger.warning("=== {} | motiv={} ===", label, motiv)
    result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
    s = result.summary
    return {
        "label": label,
        **{k: motiv.get(k) for k in (
            "title_coef", "releg_coef", "dead_rubber_coef", "stakes_coef", "motivation_diff_coef"
        )},
        "log_loss_1x2": s.get("log_loss_1x2"),
        "log_loss_edge_vs_market": s.get("log_loss_edge_vs_market"),
        "roi": s.get("roi"),
        "avg_clv_prob": s.get("avg_clv_prob"),
        "n_bets": s.get("n_bets"),
        "experiment_id": result.experiment_id,
    }


def main() -> None:
    setup_logging("WARNING")
    set_global_seed(42)
    cfg0 = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg0)
    exp_dir = resolve_experiments_dir(cfg0)

    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    matches = enrich_matches_with_understat_advanced(
        matches, load_understat_team_history(data_dir / "raw" / "understat")
    )

    # Baseline: motivation off (current best)
    rows = []
    rows.append(
        _run(
            cfg0,
            matches,
            exp_dir,
            "motiv_off",
            {"enabled": False},
        )
    )
    # Features only (coefs 0) — should match off for LL
    rows.append(_run(cfg0, matches, exp_dir, "motiv_feat_only", {}))

    grid = [
        ("motiv_stakes_0p02", {"stakes_coef": 0.02}),
        ("motiv_stakes_0p05", {"stakes_coef": 0.05}),
        ("motiv_stakes_0p10", {"stakes_coef": 0.10}),
        ("motiv_dead_0p05", {"dead_rubber_coef": 0.05}),
        ("motiv_dead_0p10", {"dead_rubber_coef": 0.10}),
        ("motiv_title_0p05", {"title_coef": 0.05}),
        ("motiv_releg_0p05", {"releg_coef": 0.05}),
        ("motiv_diff_0p05", {"motivation_diff_coef": 0.05}),
        ("motiv_diff_0p10", {"motivation_diff_coef": 0.10}),
        ("motiv_stakes_dead", {"stakes_coef": 0.05, "dead_rubber_coef": 0.05}),
        ("motiv_diff_m0p05", {"motivation_diff_coef": -0.05}),
    ]
    for label, motiv in grid:
        rows.append(_run(cfg0, matches, exp_dir, label, motiv))
        print(json.dumps(rows[-1], indent=2))

    cmp = pd.DataFrame(rows).sort_values("log_loss_1x2")
    out = exp_dir / "motivation_comparison_iter4.csv"
    cmp.to_csv(out, index=False)
    print("\n=== RANKED BY LL ===")
    print(cmp.to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
