#!/usr/bin/env python
"""
Iter18 expansion research — other lines / leagues / AH — without touching EPL packs.

Uses existing prediction artifacts only (no retrain). Reports:
  - EPL AH under EPL_aggressive (intact pack AH slice)
  - Mild OU / short overs on D1, SP1, and EPL for contrast
  - Synthetic OU 1.5 / 3.5 hit-rate study from λ,μ (no book odds in FD)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.models.poisson import score_matrix
from origination.utils import load_config, resolve_data_dir, setup_logging

OUT = ROOT / "experiments" / "iter18_expansion"

ARTIFACTS = [
    {
        "league": "EPL",
        "label": "vol06",
        "preds": "experiments/20260810T201358Z_iter17_EPL_vol06/predictions.parquet",
        "aligned": "matches_aligned.parquet",
    },
    {
        "league": "Bundesliga",
        "label": "vol06",
        "preds": "experiments/20260810T175007Z_iter16_Bundesliga_vol06/predictions.parquet",
        "aligned": "matches_aligned_D1.parquet",
    },
    {
        "league": "LaLiga",
        "label": "vol06",
        "preds": "experiments/20260810T184751Z_iter16_LaLiga_vol06/predictions.parquet",
        "aligned": "matches_aligned_SP1.parquet",
    },
]

MILD = {
    "enabled": True,
    "rules": [
        {"markets": ["1x2"], "max_odds": 2.00},
        {"markets": ["ou25"], "min_odds": 1.50, "max_odds": 3.00},
        {"markets": ["ah"], "min_odds": 1.50, "max_odds": 3.00},
    ],
}
EPL_AGG = {
    "enabled": True,
    "rules": [
        {"markets": ["1x2"], "max_odds": 1.80},
        {"markets": ["ou25"], "min_odds": 2.00, "max_odds": 4.00, "allow_sides": ["under"]},
        {"markets": ["ah"], "max_odds": 1.90},
    ],
}
SHORT_OVER = {
    "enabled": True,
    "rules": [
        {"markets": ["ou25"], "min_odds": 1.60, "max_odds": 2.50, "allow_sides": ["over"]},
    ],
}


def _roi(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None
    st = float(df["stake"].sum())
    return float(df["profit"].sum()) / st if st else None


def _score(preds, matches, filt, edge_ou: float) -> pd.DataFrame:
    bt = {
        "markets": ["1x2", "ou25", "ah"],
        "edge_threshold": 0.03,
        "edge_threshold_by_market": {"ou25": edge_ou, "ah": 0.05},
        "bet_filters": filt,
        "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
        "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
    }
    return evaluate_predictions(preds, matches, bt)


def _ou_line_from_lambda(preds: pd.DataFrame, matches: pd.DataFrame, line: float) -> dict:
    """Hit-rate when model favors over/under at `line` with p - 0.5 >= thr (no odds)."""
    m = matches.set_index("match_id")
    rows = []
    for _, r in preds.iterrows():
        mid = r["match_id"]
        if mid not in m.index:
            continue
        lam = float(r["lambda_home"])
        mu = float(r["lambda_away"])
        if not (np.isfinite(lam) and np.isfinite(mu)):
            continue
        mat = score_matrix(max(lam, 1e-6), max(mu, 1e-6), max_goals=10, rho=0.0, dixon_coles=False)
        goals = np.arange(mat.shape[0])
        total = goals[:, None] + goals[None, :]
        p_over = float(mat[total > line].sum())
        tg = float(m.loc[mid, "total_goals"])
        rows.append({"match_id": mid, "p_over": p_over, "actual_over": tg > line})
    df = pd.DataFrame(rows)
    out = {"line": line, "n": int(len(df))}
    for thr in (0.05, 0.08, 0.10):
        overs = df[df["p_over"] - 0.5 >= thr]
        unders = df[(1.0 - df["p_over"]) - 0.5 >= thr]
        out[f"over_e{int(thr*100)}_n"] = int(len(overs))
        out[f"over_e{int(thr*100)}_hit"] = float(overs["actual_over"].mean()) if len(overs) else None
        out[f"under_e{int(thr*100)}_n"] = int(len(unders))
        out[f"under_e{int(thr*100)}_hit"] = (
            float((~unders["actual_over"]).mean()) if len(unders) else None
        )
    return out


def main() -> None:
    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    OUT.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    line_rows = []

    for art in ARTIFACTS:
        pred_path = ROOT / art["preds"]
        if not pred_path.exists():
            summary_rows.append({**art, "error": "missing_preds"})
            continue
        aligned_path = data_dir / "interim" / art["aligned"]
        if not aligned_path.exists():
            summary_rows.append({**art, "error": "missing_aligned"})
            continue
        preds = pd.read_parquet(pred_path)
        matches = load_aligned(aligned_path)

        # Mild OU @ e5 / e8
        for edge, tag in [(0.05, "mild_e5"), (0.08, "mild_e8")]:
            bets = _score(preds, matches, {"enabled": True, "rules": MILD["rules"]}, edge)
            ou = bets[bets["market"] == "ou25"]
            summary_rows.append(
                {
                    "league": art["league"],
                    "model": art["label"],
                    "pack": tag,
                    "n_ou": int(len(ou)),
                    "roi_ou": _roi(ou),
                    "roi_over": _roi(ou[ou["side"] == "over"]),
                    "roi_under": _roi(ou[ou["side"] == "under"]),
                    "n_ah": int(len(bets[bets["market"] == "ah"])),
                    "roi_ah": _roi(bets[bets["market"] == "ah"]),
                    "roi_all": _roi(bets),
                }
            )

        # Short overs @ e10
        bets_so = _score(preds, matches, {"enabled": True, "rules": SHORT_OVER["rules"]}, 0.10)
        ou_so = bets_so[(bets_so["market"] == "ou25") & (bets_so["side"] == "over")]
        summary_rows.append(
            {
                "league": art["league"],
                "model": art["label"],
                "pack": "short_over_e10",
                "n_ou": int(len(ou_so)),
                "roi_ou": _roi(ou_so),
                "roi_over": _roi(ou_so),
                "roi_under": None,
                "n_ah": 0,
                "roi_ah": None,
                "roi_all": _roi(ou_so),
            }
        )

        if art["league"] == "EPL":
            bets_agg = _score(preds, matches, {"enabled": True, "rules": EPL_AGG["rules"]}, 0.08)
            ah = bets_agg[bets_agg["market"] == "ah"]
            summary_rows.append(
                {
                    "league": "EPL",
                    "model": art["label"],
                    "pack": "EPL_aggressive_AH_only",
                    "n_ou": 0,
                    "roi_ou": None,
                    "roi_over": None,
                    "roi_under": None,
                    "n_ah": int(len(ah)),
                    "roi_ah": _roi(ah),
                    "roi_all": _roi(ah),
                }
            )
            # Confirm Under pack untouched
            und = bets_agg[(bets_agg["market"] == "ou25") & (bets_agg["side"] == "under")]
            summary_rows.append(
                {
                    "league": "EPL",
                    "model": art["label"],
                    "pack": "EPL_aggressive_unders_check",
                    "n_ou": int(len(und)),
                    "roi_ou": _roi(und),
                    "roi_over": None,
                    "roi_under": _roi(und),
                    "n_ah": int(len(ah)),
                    "roi_ah": _roi(ah),
                    "roi_all": _roi(bets_agg),
                }
            )
            for line in (1.5, 3.5):
                rec = _ou_line_from_lambda(preds, matches, line)
                rec.update({"league": "EPL", "model": art["label"]})
                line_rows.append(rec)

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(OUT / "expansion_summary.csv", index=False)
    if line_rows:
        pd.DataFrame(line_rows).to_csv(OUT / "ou15_35_hitrate.csv", index=False)

    # Markdown
    lines = [
        "# Iter18 expansion research",
        "",
        "EPL Under / short-Over packs **unchanged**. This is discovery only.",
        "",
        "## Cross-league packs (vol06 artifacts)",
        "",
        "| League | Pack | n OU | ROI OU | ROI Over | ROI Under | n AH | ROI AH |",
        "|--------|------|-----:|-------:|---------:|----------:|-----:|-------:|",
    ]
    for _, r in sdf.iterrows():
        if r.get("error"):
            continue
        def fmt(x):
            return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*float(x):+.1f}%"
        lines.append(
            f"| {r['league']} | {r['pack']} | {r.get('n_ou', 0)} | {fmt(r.get('roi_ou'))} | "
            f"{fmt(r.get('roi_over'))} | {fmt(r.get('roi_under'))} | {r.get('n_ah', 0)} | {fmt(r.get('roi_ah'))} |"
        )
    if line_rows:
        lines += [
            "",
            "## EPL OU 1.5 / 3.5 (no book odds — model hit-rate only)",
            "",
            "Not +EV claims; FD has no closing 1.5/3.5 books in aligned data.",
            "",
        ]
        for rec in line_rows:
            lines.append(
                f"- Line {rec['line']}: over@e10 hit={rec.get('over_e10_hit')} "
                f"(n={rec.get('over_e10_n')}); under@e10 hit={rec.get('under_e10_hit')} "
                f"(n={rec.get('under_e10_n')})"
            )
    lines += ["", f"Artifacts: `{OUT.as_posix()}`", ""]
    report = "\n".join(lines)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    # Avoid Windows console encoding issues
    sys.stdout.buffer.write(report.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
