#!/usr/bin/env python
"""Grid asymmetric referee card_bias_coef on top of best λ stack."""

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

    # Small coef grid including 0 baseline and sign flip (negative = reverse hypothesis)
    coefs = [0.0, 0.02, 0.05, 0.10, 0.20, -0.05]
    rows = []
    for coef in coefs:
        label = f"card_bias_{coef}".replace("-", "m").replace(".", "p")
        cfg = copy.deepcopy(cfg0)
        cfg["project"]["experiment_label"] = label
        ctx = cfg.setdefault("features", {}).setdefault("context_adjustments", {})
        ctx["enabled"] = True
        ctx["referee"] = {
            "enabled": True,
            "min_prior_games": 5,
            "tempo_coef": 0.0,
            "card_bias_coef": coef,
        }
        for k in ("injuries", "lineups", "formations", "motivation", "weather", "coaching_change", "travel"):
            ctx.setdefault(k, {"enabled": False})
        logger.warning("=== {} (card_bias_coef={}) ===", label, coef)
        result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
        s = result.summary
        row = {
            "label": label,
            "card_bias_coef": coef,
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
    out = exp_dir / "card_bias_comparison_iter3.csv"
    cmp.to_csv(out, index=False)
    print("\n=== RANKED BY LL ===")
    print(cmp.to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
