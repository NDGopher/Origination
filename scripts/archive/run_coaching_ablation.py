#!/usr/bin/env python
"""Coaching-change ablation on best λ stack (after card_bias decision)."""

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

    variants = [
        ("coach_off", False, 0.0),
        ("coach_feat_bounce0", True, 0.0),
        ("coach_bounce_p02", True, 0.02),
        ("coach_bounce_p05", True, 0.05),
        ("coach_bounce_m02", True, -0.02),
    ]
    rows = []
    for label, enabled, bounce in variants:
        cfg = copy.deepcopy(cfg0)
        cfg["project"]["experiment_label"] = label
        # Keep referee features live, card_bias at whatever default is (0 unless promoted)
        ctx = cfg.setdefault("features", {}).setdefault("context_adjustments", {})
        ctx["enabled"] = True
        ctx.setdefault(
            "referee",
            {
                "enabled": True,
                "tempo_coef": 0.0,
                "card_bias_coef": 0.5,
                "min_prior_games": 5,
            },
        )
        ctx["coaching_change"] = {
            "enabled": enabled,
            "new_coach_days": 60,
            "new_coach_games": 8,
            "bounce_coef": bounce,
            "changes_path": "data/interim/coaching_changes.csv",
        }
        for k in ("injuries", "lineups", "formations", "motivation", "weather", "travel"):
            ctx.setdefault(k, {"enabled": False})
        if not enabled:
            # pure baseline with context master on but coaching off
            pass
        logger.warning("=== {} ===", label)
        result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
        s = result.summary
        row = {
            "label": label,
            "bounce_coef": bounce,
            "enabled": enabled,
            "log_loss_1x2": s.get("log_loss_1x2"),
            "log_loss_edge_vs_market": s.get("log_loss_edge_vs_market"),
            "roi": s.get("roi"),
            "avg_clv_prob": s.get("avg_clv_prob"),
            "n_bets": s.get("n_bets"),
            "experiment_id": result.experiment_id,
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    cmp = pd.DataFrame(rows).sort_values("log_loss_1x2")
    out = exp_dir / "coaching_comparison_iter3.csv"
    cmp.to_csv(out, index=False)
    print(cmp.to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
