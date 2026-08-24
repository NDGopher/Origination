#!/usr/bin/env python
"""
Iter17 Phase A: Overs filter / selection search on existing preds (fast).

Scores Overs vs Unders separately across leagues and model artifacts.
Also scores intact EPL_aggressive pack. Does NOT modify pack rules.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.utils import load_config, resolve_data_dir, setup_logging


# Best / interesting iter16 artifacts per league
ARTIFACTS = [
    {
        "league": "EPL",
        "label": "thresh05",
        "experiment_id": "20260810T164655Z_iter16_EPL_int_thresh05",
        "aligned": "matches_aligned.parquet",
    },
    {
        "league": "EPL",
        "label": "vol06",
        "experiment_id": "20260810T165447Z_iter16_EPL_vol06",
        "aligned": "matches_aligned.parquet",
    },
    {
        "league": "Bundesliga",
        "label": "thresh05",
        "experiment_id": "20260810T174732Z_iter16_Bundesliga_int_thresh05",
        "aligned": "matches_aligned_D1.parquet",
    },
    {
        "league": "Bundesliga",
        "label": "vol06",
        "experiment_id": "20260810T175007Z_iter16_Bundesliga_vol06",
        "aligned": "matches_aligned_D1.parquet",
    },
    {
        "league": "LaLiga",
        "label": "base_signed",
        "experiment_id": "20260810T181342Z_iter16_LaLiga_base",
        "aligned": "matches_aligned_SP1.parquet",
    },
    {
        "league": "LaLiga",
        "label": "vol06",
        "experiment_id": "20260810T184751Z_iter16_LaLiga_vol06",
        "aligned": "matches_aligned_SP1.parquet",
    },
]

EPL_PACK = {
    "enabled": True,
    "rules": [
        {"markets": ["1x2"], "max_odds": 1.80},
        {"markets": ["ou25"], "min_odds": 2.00, "max_odds": 4.00, "allow_sides": ["under"]},
        {"markets": ["ah"], "max_odds": 1.90},
    ],
}
EPL_EDGE = {"ou25": 0.08, "ah": 0.05}


def _roi(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None
    st = float(df["stake"].sum())
    return float(df["profit"].sum()) / st if st else None


def score(preds, matches, filt, edge_ou: float, edge_default: float = 0.03) -> dict:
    bt = {
        "markets": ["1x2", "ou25", "ah"],
        "edge_threshold": edge_default,
        "edge_threshold_by_market": {"ou25": edge_ou, "ah": 0.05},
        "bet_filters": filt,
        "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
        "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
    }
    bets = evaluate_predictions(preds, matches, bt, edge_threshold=edge_default)
    ou = bets[bets["market"] == "ou25"]
    overs = ou[ou["side"] == "over"]
    unders = ou[ou["side"] == "under"]
    short_o = overs[(overs["close_odds"] >= 1.60) & (overs["close_odds"] <= 2.50)]
    long_o = overs[(overs["close_odds"] >= 2.00) & (overs["close_odds"] <= 3.50)]
    out = {
        "n_ou": int(len(ou)),
        "roi_ou": _roi(ou),
        "n_over": int(len(overs)),
        "roi_over": _roi(overs),
        "n_under": int(len(unders)),
        "roi_under": _roi(unders),
        "n_over_short": int(len(short_o)),
        "roi_over_short": _roi(short_o),
        "n_over_long": int(len(long_o)),
        "roi_over_long": _roi(long_o),
        "roi_all": _roi(bets),
        "n_all": int(len(bets)),
    }
    if "season" in ou.columns and len(unders) >= 40:
        by = unders.groupby("season").apply(
            lambda g: g["profit"].sum() / g["stake"].sum() if g["stake"].sum() else None
        )
        out["under_seasons_pos"] = int((by > 0).sum())
        out["under_seasons_n"] = int(by.notna().sum())
    if "season" in ou.columns and len(overs) >= 40:
        by = overs.groupby("season").apply(
            lambda g: g["profit"].sum() / g["stake"].sum() if g["stake"].sum() else None
        )
        out["over_seasons_pos"] = int((by > 0).sum())
        out["over_seasons_n"] = int(by.notna().sum())
    return out


def over_packs() -> list[tuple[str, dict, float]]:
    """(name, bet_filters, ou_edge)."""
    packs = []
    # Raw overs at edges
    for e in (0.03, 0.05, 0.08, 0.10, 0.12):
        packs.append(
            (
                f"over_only_e{int(e*100):02d}",
                {
                    "enabled": True,
                    "rules": [
                        {"markets": ["1x2"], "max_odds": 2.00},
                        {"markets": ["ou25"], "allow_sides": ["over"]},
                    ],
                },
                e,
            )
        )
    # Short overs
    for e, lo, hi in itertools.product(
        (0.05, 0.08, 0.10),
        (1.50, 1.60),
        (2.00, 2.20, 2.50),
    ):
        if lo >= hi:
            continue
        packs.append(
            (
                f"over_short_{lo}_{hi}_e{int(e*100):02d}",
                {
                    "enabled": True,
                    "rules": [
                        {"markets": ["1x2"], "max_odds": 2.00},
                        {
                            "markets": ["ou25"],
                            "allow_sides": ["over"],
                            "min_odds": lo,
                            "max_odds": hi,
                        },
                    ],
                },
                e,
            )
        )
    # Mid/long overs (mirror under pocket)
    for e, lo, hi in ((0.08, 2.00, 3.00), (0.08, 2.00, 4.00), (0.10, 2.20, 3.50), (0.08, 1.80, 2.70)):
        packs.append(
            (
                f"over_band_{lo}_{hi}_e{int(e*100):02d}",
                {
                    "enabled": True,
                    "rules": [
                        {"markets": ["1x2"], "max_odds": 2.00},
                        {
                            "markets": ["ou25"],
                            "allow_sides": ["over"],
                            "min_odds": lo,
                            "max_odds": hi,
                        },
                    ],
                },
                e,
            )
        )
    # Mild both sides (reference)
    packs.append(
        (
            "mild_both_1p5_3_e05",
            {
                "enabled": True,
                "rules": [
                    {"markets": ["1x2"], "max_odds": 2.00},
                    {"markets": ["ou25"], "min_odds": 1.50, "max_odds": 3.00},
                ],
            },
            0.05,
        )
    )
    # Under control (should stay healthy on EPL)
    packs.append(
        (
            "under_2_4_e08",
            {
                "enabled": True,
                "rules": [
                    {"markets": ["1x2"], "max_odds": 1.80},
                    {
                        "markets": ["ou25"],
                        "min_odds": 2.00,
                        "max_odds": 4.00,
                        "allow_sides": ["under"],
                    },
                ],
            },
            0.08,
        )
    )
    return packs


def main() -> None:
    setup_logging("WARNING")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    exp = ROOT / "experiments"
    out = exp / "iter17_overs_filter_grid"
    out.mkdir(parents=True, exist_ok=True)

    packs = over_packs()
    rows = []
    for art in ARTIFACTS:
        pred_path = exp / art["experiment_id"] / "predictions.parquet"
        if not pred_path.exists():
            print("SKIP", art["experiment_id"])
            continue
        preds = pd.read_parquet(pred_path)
        matches = load_aligned(data_dir / "interim" / art["aligned"])
        # Always score protected EPL pack on EPL artifacts
        if art["league"] == "EPL":
            ep = score(preds, matches, EPL_PACK, EPL_EDGE["ou25"])
            rows.append(
                {
                    "league": art["league"],
                    "model": art["label"],
                    "pack": "EPL_aggressive",
                    **ep,
                }
            )
        for name, filt, e_ou in packs:
            s = score(preds, matches, filt, e_ou)
            rows.append(
                {
                    "league": art["league"],
                    "model": art["label"],
                    "pack": name,
                    **s,
                }
            )
        print("done", art["league"], art["label"])

    df = pd.DataFrame(rows)
    df.to_csv(out / "all_packs.csv", index=False)

    # Best overs packs with n>=80
    overs = df[df["pack"].str.startswith("over")].copy()
    overs = overs[overs["n_over"] >= 80]
    best = overs.sort_values("roi_over", ascending=False)
    best.to_csv(out / "overs_ranked.csv", index=False)

    # Cross-league: packs that are +EV overs on ≥2 leagues (same pack name, any model)
    pivot = (
        overs.groupby(["pack", "league"])["roi_over"]
        .max()
        .unstack("league")
    )
    if pivot is not None and len(pivot):
        pivot["n_pos"] = (pivot > 0).sum(axis=1)
        pivot["mean_roi"] = pivot.mean(axis=1)
        pivot.sort_values(["n_pos", "mean_roi"], ascending=False).to_csv(
            out / "overs_cross_league.csv"
        )

    # Under health check
    und = df[df["pack"] == "under_2_4_e08"][
        ["league", "model", "n_under", "roi_under", "under_seasons_pos", "under_seasons_n"]
    ]
    und.to_csv(out / "under_health.csv", index=False)

    lines = [
        "# Iter17 Overs filter search",
        "",
        "## Top Overs packs (n_over≥80)",
        "",
    ]
    top = best.head(25)
    lines.append("| league | model | pack | n | ROI over | seasons+ |")
    lines.append("|--------|-------|------|--:|---------:|---------:|")
    for _, r in top.iterrows():
        lines.append(
            f"| {r['league']} | {r['model']} | {r['pack']} | {int(r['n_over'])} | "
            f"{r['roi_over']:+.1%} | {r.get('over_seasons_pos', '')}/{r.get('over_seasons_n', '')} |"
        )
    lines += ["", "## Under health (under_2_4_e08)", ""]
    lines.append(und.to_string(index=False))
    lines += ["", "## EPL_aggressive", ""]
    epl_a = df[df["pack"] == "EPL_aggressive"][
        ["model", "roi_all", "roi_ou", "roi_under", "n_under", "n_over"]
    ]
    lines.append(epl_a.to_string(index=False))
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out)
    print(top.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
