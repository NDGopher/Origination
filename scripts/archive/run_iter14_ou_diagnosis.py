#!/usr/bin/env python
"""
Iter14: deep OU *probability* diagnosis (not selection ROI).

For each league artifact, join preds + matches and measure:
- model vs market log-loss / Brier overall and by short-price band
- calibration (reliability) of p_over25
- signed bias by odds band and by over/under favorite
- mean-goals error (λ+μ vs actual) by OU price band

Artifacts → experiments/iter14_ou_diagnosis/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.data_ingestion.align import load_aligned
from origination.utils import load_config, resolve_data_dir, setup_logging
from origination.utils.odds import two_way_fair


SPECS = [
    {
        "label": "EPL",
        "experiment_id": "20260806T160826Z_iter13_L_base",
        "aligned": "matches_aligned.parquet",
    },
    {
        "label": "Championship",
        "experiment_id": "20260805T200446Z_league_E1_champ_iter8",
        "aligned": "matches_aligned_E1.parquet",
    },
    {
        "label": "Bundesliga",
        "experiment_id": "20260805T193621Z_league_D1_xg_resid",
        "aligned": "matches_aligned_D1.parquet",
    },
    {
        "label": "SerieA",
        "experiment_id": "20260805T194400Z_league_I1_serie_a",
        "aligned": "matches_aligned_I1.parquet",
    },
    {
        "label": "LaLiga",
        "experiment_id": "20260805T195244Z_league_SP1_la_liga",
        "aligned": "matches_aligned_SP1.parquet",
    },
]

# Shorter listed OU price (favorite side of the 2.5 market)
ODDS_BANDS = [
    ("1.40-1.60", 1.40, 1.60),
    ("1.60-1.80", 1.60, 1.80),
    ("1.80-2.00", 1.80, 2.00),
    ("2.00-2.20", 2.00, 2.20),
    ("2.20-2.50", 2.20, 2.50),
    ("2.50-3.00", 2.50, 3.00),
    ("3.00+", 3.00, 99.0),
]


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _prepare(preds: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.set_index("match_id")
    rows = []
    for _, r in preds.iterrows():
        mid = r["match_id"]
        if mid not in m.index:
            continue
        match = m.loc[mid]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        o = match.get("close_over25")
        u = match.get("close_under25")
        if not (pd.notna(o) and pd.notna(u)):
            continue
        o, u = float(o), float(u)
        try:
            fair_o, fair_u = two_way_fair(o, u, method="power")
        except Exception:
            continue
        y = 1.0 if float(match["total_goals"]) > 2.5 else 0.0
        p = float(np.clip(r["p_over25"], 1e-6, 1.0 - 1e-6))
        short = min(o, u)
        fav_side = "over" if o <= u else "under"
        lam = float(r.get("lambda_home", np.nan)) + float(r.get("lambda_away", np.nan))
        rows.append(
            {
                "match_id": mid,
                "season": match.get("season"),
                "y_over": y,
                "total_goals": float(match["total_goals"]),
                "p_over": p,
                "p_under": 1.0 - p,
                "fair_over": float(fair_o),
                "fair_under": float(fair_u),
                "odds_over": o,
                "odds_under": u,
                "short_odds": short,
                "fav_side": fav_side,
                "sum_lambda": lam,
                "err_p": p - y,
                "err_vs_mkt": p - float(fair_o),
                "goals_err": lam - float(match["total_goals"]) if np.isfinite(lam) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _band_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, lo, hi in ODDS_BANDS:
        sub = df[(df["short_odds"] >= lo) & (df["short_odds"] < hi)]
        if len(sub) < 30:
            continue
        y = sub["y_over"].to_numpy()
        pm = sub["p_over"].to_numpy()
        pk = sub["fair_over"].to_numpy()
        # Model edge on the short (favorite) side
        short_is_over = sub["odds_over"] <= sub["odds_under"]
        p_short = np.where(short_is_over, pm, 1.0 - pm)
        y_short = np.where(short_is_over, y, 1.0 - y)
        fair_short = np.where(short_is_over, pk, 1.0 - pk)
        out.append(
            {
                "band": name,
                "n": int(len(sub)),
                "rate_over": float(y.mean()),
                "mean_p_model": float(pm.mean()),
                "mean_p_mkt": float(pk.mean()),
                "bias_model": float((pm - y).mean()),
                "bias_mkt": float((pk - y).mean()),
                "ll_model": _ll(pm, y),
                "ll_mkt": _ll(pk, y),
                "ll_gap": _ll(pm, y) - _ll(pk, y),
                "brier_model": _brier(pm, y),
                "brier_mkt": _brier(pk, y),
                "short_ll_model": _ll(p_short, y_short),
                "short_ll_mkt": _ll(fair_short, y_short),
                "short_bias": float((p_short - y_short).mean()),
                "mean_sum_lambda": float(sub["sum_lambda"].mean()),
                "mean_goals": float(sub["total_goals"].mean()),
                "mean_goals_err": float(sub["goals_err"].mean()),
            }
        )
    return pd.DataFrame(out)


def _reliability(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    d = df.copy()
    d["bin"] = pd.qcut(d["p_over"], q=n_bins, duplicates="drop")
    rows = []
    for b, sub in d.groupby("bin", observed=True):
        rows.append(
            {
                "bin": str(b),
                "n": int(len(sub)),
                "mean_p": float(sub["p_over"].mean()),
                "rate_over": float(sub["y_over"].mean()),
                "gap": float(sub["p_over"].mean() - sub["y_over"].mean()),
                "ll": _ll(sub["p_over"].to_numpy(), sub["y_over"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def _fav_side_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, sub in df.groupby("fav_side"):
        y = sub["y_over"].to_numpy()
        pm = sub["p_over"].to_numpy()
        pk = sub["fair_over"].to_numpy()
        # Restrict to short-price favorites in 1.60–2.50
        short = sub[(sub["short_odds"] >= 1.60) & (sub["short_odds"] <= 2.50)]
        rows.append(
            {
                "fav_side": side,
                "n": int(len(sub)),
                "ll_gap": _ll(pm, y) - _ll(pk, y),
                "bias": float((pm - y).mean()),
                "n_short_1p6_2p5": int(len(short)),
                "ll_gap_short": (
                    _ll(short["p_over"].to_numpy(), short["y_over"].to_numpy())
                    - _ll(short["fair_over"].to_numpy(), short["y_over"].to_numpy())
                    if len(short) >= 30
                    else np.nan
                ),
                "bias_short": float((short["p_over"] - short["y_over"]).mean()) if len(short) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _focus_short(df: pd.DataFrame) -> dict:
    short = df[(df["short_odds"] >= 1.60) & (df["short_odds"] <= 2.50)]
    if len(short) < 50:
        return {"n": int(len(short))}
    y = short["y_over"].to_numpy()
    pm = short["p_over"].to_numpy()
    pk = short["fair_over"].to_numpy()
    return {
        "n": int(len(short)),
        "ll_model": _ll(pm, y),
        "ll_mkt": _ll(pk, y),
        "ll_gap": _ll(pm, y) - _ll(pk, y),
        "brier_model": _brier(pm, y),
        "brier_mkt": _brier(pk, y),
        "bias_model": float((pm - y).mean()),
        "bias_mkt": float((pk - y).mean()),
        "mean_goals_err": float(short["goals_err"].mean()),
        "corr_sumlam_goals": float(short["sum_lambda"].corr(short["total_goals"])),
    }


def main() -> None:
    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    exp_root = ROOT / "experiments"
    out = exp_root / "iter14_ou_diagnosis"
    out.mkdir(parents=True, exist_ok=True)

    rollup = []
    band_parts = []
    fav_parts = []

    for spec in SPECS:
        eid = exp_root / spec["experiment_id"]
        pred_path = eid / "predictions.parquet"
        if not pred_path.exists():
            print("SKIP missing", eid)
            continue
        preds = pd.read_parquet(pred_path)
        matches = load_aligned(data_dir / "interim" / spec["aligned"])
        df = _prepare(preds, matches)
        label = spec["label"]
        league_dir = out / label
        league_dir.mkdir(parents=True, exist_ok=True)

        bands = _band_table(df)
        rel = _reliability(df)
        fav = _fav_side_table(df)
        focus = _focus_short(df)
        overall = {
            "league": label,
            "n": int(len(df)),
            "ll_model": _ll(df["p_over"].to_numpy(), df["y_over"].to_numpy()),
            "ll_mkt": _ll(df["fair_over"].to_numpy(), df["y_over"].to_numpy()),
            "ll_gap": _ll(df["p_over"].to_numpy(), df["y_over"].to_numpy())
            - _ll(df["fair_over"].to_numpy(), df["y_over"].to_numpy()),
            "brier_model": _brier(df["p_over"].to_numpy(), df["y_over"].to_numpy()),
            "brier_mkt": _brier(df["fair_over"].to_numpy(), df["y_over"].to_numpy()),
            "bias": float((df["p_over"] - df["y_over"]).mean()),
            "mean_goals_err": float(df["goals_err"].mean()),
            **{f"short_{k}": v for k, v in focus.items()},
        }
        rollup.append(overall)
        bands.assign(league=label).to_csv(league_dir / "by_odds_band.csv", index=False)
        rel.to_csv(league_dir / "reliability.csv", index=False)
        fav.assign(league=label).to_csv(league_dir / "by_fav_side.csv", index=False)
        band_parts.append(bands.assign(league=label))
        fav_parts.append(fav.assign(league=label))
        print(json.dumps(overall, indent=2, default=str))

    roll = pd.DataFrame(rollup)
    roll.to_csv(out / "rollup.csv", index=False)
    if band_parts:
        pd.concat(band_parts, ignore_index=True).to_csv(out / "by_odds_band_all.csv", index=False)
    if fav_parts:
        pd.concat(fav_parts, ignore_index=True).to_csv(out / "by_fav_side_all.csv", index=False)

    # Markdown summary
    lines = [
        "# Iter14 OU probability diagnosis",
        "",
        "## Overall model vs market (OU 2.5)",
        "",
        "| League | n | LL model | LL mkt | LL gap | bias | short 1.60–2.50 LL gap |",
        "|--------|--:|---------:|-------:|-------:|-----:|-----------------------:|",
    ]
    for _, r in roll.iterrows():
        lines.append(
            f"| {r['league']} | {int(r['n'])} | {r['ll_model']:.4f} | {r['ll_mkt']:.4f} | "
            f"**{r['ll_gap']:+.4f}** | {r['bias']:+.3f} | "
            f"{r.get('short_ll_gap', float('nan')):+.4f} |"
        )
    lines += [
        "",
        "Positive LL gap = model worse than market. Short band is the focus weakness.",
        "",
        "See `by_odds_band_all.csv` for band-level bias / goals-error.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
