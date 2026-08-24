#!/usr/bin/env python
"""
Focused grid / ablation runner for this iteration.

Loads aligned data once, enriches Understat advanced once, then runs walk-forward
for each candidate config variant. Writes a comparison CSV at the end.
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
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.utils import load_config, resolve_data_dir, resolve_experiments_dir, set_global_seed, setup_logging


def base_cfg() -> dict:
    return load_config(ROOT / "configs" / "default.yaml")


def with_label(cfg: dict, label: str) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("project", {})["experiment_label"] = label
    return cfg


def set_intensity(cfg: dict, *, source: str, blend: float | None, ppda: float, deep: float, adj_on: bool) -> dict:
    cfg = copy.deepcopy(cfg)
    dc = cfg.setdefault("model", {}).setdefault("dixon_coles", {})
    dc["intensity_source"] = source
    if blend is not None:
        dc["blend_xg_weight"] = blend
    dc["intensity_adjustments"] = {
        "enabled": adj_on,
        "ppda_coef": ppda,
        "deep_coef": deep,
    }
    return cfg


def set_referee(cfg: dict, *, enabled: bool, tempo: float = 0.0, min_prior: int = 5) -> dict:
    cfg = copy.deepcopy(cfg)
    ctx = cfg.setdefault("features", {}).setdefault("context_adjustments", {})
    ctx["enabled"] = enabled
    ctx["referee"] = {"enabled": enabled, "tempo_coef": tempo, "min_prior_games": min_prior}
    # keep other scaffolds off
    for k in ("injuries", "lineups", "formations", "motivation", "weather", "coaching_change", "travel"):
        ctx.setdefault(k, {"enabled": False})
    return cfg


def main() -> None:
    setup_logging("WARNING")
    set_global_seed(42)
    cfg0 = base_cfg()
    data_dir = resolve_data_dir(cfg0)
    exp_dir = resolve_experiments_dir(cfg0)

    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    hist = load_understat_team_history(data_dir / "raw" / "understat")
    matches = enrich_matches_with_understat_advanced(matches, hist)
    logger.info("Matches ready: {}", len(matches))

    variants: list[tuple[str, dict]] = []

    # --- Phase 1: λ coefficient grid (xg + understat_advanced) ---
    coef_grid = [
        ("grid_off", 0.0, 0.0, False),
        ("grid_p0_d01", 0.0, 0.01, True),
        ("grid_p005_d01", 0.005, 0.01, True),
        ("grid_p005_d015", 0.005, 0.015, True),
        ("grid_p01_d01", 0.01, 0.01, True),
        ("grid_p01_d02", 0.01, 0.02, True),  # prior default
        ("grid_p01_d03", 0.01, 0.03, True),
    ]
    for label, ppda, deep, on in coef_grid:
        c = set_intensity(cfg0, source="xg", blend=None, ppda=ppda, deep=deep, adj_on=on)
        c = set_referee(c, enabled=False)
        variants.append((label, with_label(c, label)))

    # --- Phase 1b: blend + advanced (use mild coeffs; refine after coef winner) ---
    for w in (0.6, 0.7, 0.8):
        label = f"grid_blend{w}_p005_d01"
        c = set_intensity(cfg0, source="blend", blend=w, ppda=0.005, deep=0.01, adj_on=True)
        c = set_referee(c, enabled=False)
        variants.append((label, with_label(c, label)))

    rows = []
    for label, cfg in variants:
        logger.warning("=== RUN {} ===", label)
        result = run_walk_forward(matches, cfg, experiments_dir=exp_dir)
        s = result.summary
        rows.append(
            {
                "label": label,
                "log_loss_1x2": s.get("log_loss_1x2"),
                "log_loss_edge_vs_market": s.get("log_loss_edge_vs_market"),
                "roi": s.get("roi"),
                "avg_clv_prob": s.get("avg_clv_prob"),
                "n_bets": s.get("n_bets"),
                "experiment_id": result.experiment_id,
            }
        )
        print(json.dumps(rows[-1], indent=2))

    cmp = pd.DataFrame(rows).sort_values("log_loss_1x2")
    out = exp_dir / "grid_comparison_iter2.csv"
    cmp.to_csv(out, index=False)
    print("\n=== GRID RANKED BY LL ===")
    print(cmp.to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
