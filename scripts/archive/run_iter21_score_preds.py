#!/usr/bin/env python
"""Iteration 21 — robust multi-league score predictions with confidence labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.fixtures_upcoming import refresh_upcoming_fixtures_for_league
from origination.data_ingestion.pinnacle_odds import load_pinnacle_odds
from origination.prediction.upcoming import predict_upcoming
from origination.utils import load_config, resolve_data_dir, setup_logging
from origination.utils.league_registry import get_league, list_league_keys
from origination.utils.seeding import season_from_date
from origination.utils.team_names import DEFAULT_MAPPER

OUT = ROOT / "experiments" / "iter21" / "score_predictions"
OUT.mkdir(parents=True, exist_ok=True)


def _confidence(hist: pd.DataFrame) -> dict:
    n = int(len(hist.dropna(subset=["home_goals", "away_goals"])))
    has_xg = "home_xg" in hist.columns and float(hist["home_xg"].notna().mean()) > 0.5
    has_ou = "close_over25" in hist.columns and float(hist["close_over25"].notna().mean()) > 0.5
    has_1x2 = "close_h" in hist.columns and float(hist["close_h"].notna().mean()) > 0.5
    if has_xg and has_ou and n >= 2500:
        label = "HIGH"
    elif has_xg and n >= 2000:
        label = "MODERATE–HIGH"
    elif n >= 3000 and has_1x2:
        label = "MODERATE"
    elif n >= 1500:
        label = "LOW–MODERATE"
    else:
        label = "LOW"
    return {
        "n_history": n,
        "has_xg": has_xg,
        "has_ou_closes": has_ou,
        "has_1x2_closes": has_1x2,
        "confidence_label": label,
    }


def _ensure_mls(data_dir: Path) -> Path:
    aligned = data_dir / "interim" / "matches_aligned_MLS.parquet"
    if aligned.exists() and aligned.stat().st_size > 1000:
        return aligned
    import requests
    from io import BytesIO

    url = "https://www.football-data.co.uk/new/USA.csv"
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    raw = pd.read_csv(BytesIO(r.content))
    raw = raw[raw["League"].astype(str).str.upper() == "MLS"].copy()
    (data_dir / "raw" / "football_data" / "USA").mkdir(parents=True, exist_ok=True)
    (data_dir / "raw" / "football_data" / "USA" / "USA.csv").write_bytes(r.content)
    df = raw.copy()
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "Home", "Away"])
    df["home_team"] = DEFAULT_MAPPER.map_series(df["Home"])
    df["away_team"] = DEFAULT_MAPPER.map_series(df["Away"])
    df["home_goals"] = pd.to_numeric(df["HG"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["AG"], errors="coerce")
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["season"] = df["date"].dt.year
    df["close_h"] = pd.to_numeric(df.get("PSCH"), errors="coerce")
    df["close_d"] = pd.to_numeric(df.get("PSCD"), errors="coerce")
    df["close_a"] = pd.to_numeric(df.get("PSCA"), errors="coerce")
    df["match_id"] = [
        f"{d.strftime('%Y%m%d')}_{str(h).replace(' ', '')}_{str(a).replace(' ', '')}"
        for d, h, a in zip(df["date"], df["home_team"], df["away_team"], strict=True)
    ]
    for c in ("close_over25", "close_under25", "ah_line", "close_ahh", "close_aha"):
        df[c] = np.nan
    completed = df.dropna(subset=["home_goals", "away_goals"]).copy()
    completed["ftr"] = np.where(
        completed["home_goals"] > completed["away_goals"],
        "H",
        np.where(completed["home_goals"] < completed["away_goals"], "A", "D"),
    )
    keep = [
        c
        for c in [
            "match_id",
            "date",
            "season",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "total_goals",
            "ftr",
            "close_h",
            "close_d",
            "close_a",
            "close_over25",
            "close_under25",
            "ah_line",
            "close_ahh",
            "close_aha",
        ]
        if c in completed.columns
    ]
    completed[keep].to_parquet(aligned, index=False)
    return aligned


def run_league(data_dir: Path, key: str) -> Path | None:
    info = get_league(key)
    cfg = load_config(ROOT / info["config"])
    if key == "MLS":
        aligned_path = _ensure_mls(data_dir)
        # Ensure MLS config exists (written by master if present)
        if not (ROOT / info["config"]).exists():
            print(f"  SKIP {key}: no config")
            return None
    else:
        aligned_path = data_dir / "interim" / info["aligned"]
        if not aligned_path.exists():
            print(f"  SKIP {key}: no aligned data")
            return None

    hist = load_aligned(aligned_path) if key != "MLS" else pd.read_parquet(aligned_path)
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)
    conf = _confidence(hist)

    try:
        fx, meta = refresh_upcoming_fixtures_for_league(data_dir, key, cfg, days_ahead=21)
        print(f"  {key} fixtures: {len(fx)} source={meta.get('source')}")
    except Exception as exc:
        print(f"  {key} fixture refresh failed: {exc}")
        fx_path = data_dir / "interim" / (
            "fixtures_upcoming_EPL.csv" if key == "EPL" else f"fixtures_upcoming_{key}.csv"
        )
        if not fx_path.exists():
            return None
        fx = pd.read_csv(fx_path, parse_dates=["date"])

    if fx is None or len(fx) == 0:
        print(f"  {key}: no upcoming fixtures")
        return None

    fx = fx.copy()
    fx["date"] = pd.to_datetime(fx["date"])
    fx["home_team"] = DEFAULT_MAPPER.map_series(fx["home_team"])
    fx["away_team"] = DEFAULT_MAPPER.map_series(fx["away_team"])
    if "season" not in fx.columns or fx["season"].isna().any():
        if key == "MLS":
            fx["season"] = fx["date"].dt.year
        else:
            fx["season"] = fx["date"].map(season_from_date)
    if "match_id" not in fx.columns:
        fx["match_id"] = [
            f"{d.strftime('%Y%m%d')}_{str(h).replace(' ', '')}_{str(a).replace(' ', '')}"
            for d, h, a in zip(fx["date"], fx["home_team"], fx["away_team"], strict=True)
        ]

    # Enrich understat when configured
    if cfg.get("features", {}).get("groups", {}).get("understat_advanced"):
        try:
            from origination.data_ingestion.understat_advanced import (
                enrich_matches_with_understat_advanced,
                load_understat_team_history,
            )

            hist_us = load_understat_team_history(data_dir / "raw" / "understat")
            hist = enrich_matches_with_understat_advanced(hist, hist_us)
        except Exception as exc:
            print(f"  {key} understat enrich skipped: {exc}")

    odds = load_pinnacle_odds(data_dir, key)
    try:
        preds = predict_upcoming(
            hist,
            fx,
            cfg,
            odds=odds if len(odds) else None,
            apply_residual=False,
        )
    except Exception as exc:
        print(f"  {key} predict failed: {exc}")
        (OUT / f"{key}_predict_error.txt").write_text(str(exc), encoding="utf-8")
        return None

    preds["league"] = key
    preds["confidence_label"] = conf["confidence_label"]
    preds["data_n_history"] = conf["n_history"]
    preds["has_xg"] = conf["has_xg"]
    preds["has_ou_closes"] = conf["has_ou_closes"]
    preds["has_1x2_closes"] = conf["has_1x2_closes"]
    # Friendly score columns
    if "lambda_home" in preds.columns:
        preds["proj_home_goals"] = preds["lambda_home"]
        preds["proj_away_goals"] = preds["lambda_away"]
        preds["proj_total_goals"] = preds["lambda_home"] + preds["lambda_away"]

    out = OUT / f"{key}_score_predictions.csv"
    preds.to_csv(out, index=False)
    (OUT / f"{key}_confidence.json").write_text(json.dumps(conf, indent=2), encoding="utf-8")
    print(f"  Wrote {out.name} n={len(preds)} conf={conf['confidence_label']}")
    return out


def main() -> None:
    setup_logging("WARNING")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    # Prefer MLS first (user priority), then others
    keys = ["MLS"] + [k for k in list_league_keys() if k != "MLS"]
    written = []
    for key in keys:
        print(f"=== {key} ===")
        p = run_league(data_dir, key)
        if p:
            written.append(p.name)
    # Feasibility note
    mls_conf_path = OUT / "MLS_confidence.json"
    conf = json.loads(mls_conf_path.read_text(encoding="utf-8")) if mls_conf_path.exists() else {}
    feas = [
        "# MLS feasibility update (iter21)",
        "",
        f"- Completed matches: **{conf.get('n_history', '?')}**",
        f"- 1X2 closes: **{conf.get('has_1x2_closes')}**",
        f"- OU closes: **{conf.get('has_ou_closes')}** (cannot EV-backtest totals)",
        f"- xG / shots: **{conf.get('has_xg')}**",
        f"- Score-prediction confidence: **{conf.get('confidence_label', '?')}**",
        "",
        "Use `MLS_score_predictions.csv` for projected scores + model probs; "
        "compare manually to live Pinnacle (league id 2663). Not a bettable OU system.",
        "",
        f"Files written: {', '.join(written) if written else 'none'}",
        "",
    ]
    (OUT / "MLS_FEASIBILITY.md").write_text("\n".join(feas), encoding="utf-8")
    print("DONE", written)


if __name__ == "__main__":
    main()
