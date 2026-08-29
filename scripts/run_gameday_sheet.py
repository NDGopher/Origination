#!/usr/bin/env python
"""
Gameday / live prediction sheet — production CLI for EPL totals books.

Same code path as walk-forward: features → Dixon–Coles → totals intercept →
calibration → residual → markets. Pack flags for EPL_aggressive Unders and
EPL_overs_short_exp stay completely separate.

Examples
--------
  # Refresh data then sheet for fixtures CSV + optional odds
  .venv\\Scripts\\python.exe scripts/run_gameday_sheet.py --update-data \\
      --fixtures path/to/fixtures.csv --odds-file path/to/odds.csv

  # Fast (skip residual OOS fit)
  .venv\\Scripts\\python.exe scripts/run_gameday_sheet.py --fixtures fixtures.csv --fast

Edge
----
  Two-way (preferred, matches backtest):
      edge = model_prob - power_devig_fair(side | over_odds, under_odds)
  Single-price fallback:
      edge = model_prob - 1/book_odds
  Consensus: mean of all *over25* / *under25* columns, then same formulas.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from loguru import logger

from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.prediction.upcoming import (
    apply_late_info_to_features,
    build_gameday_sheet,
    load_late_info_csv,
    load_odds_csv,
    predict_upcoming,
)
from origination.features.store import build_feature_matrix
from origination.utils import load_config, resolve_data_dir, set_global_seed, setup_logging
from origination.utils.seeding import season_from_date


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build gameday prediction sheet")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--league",
        default="EPL",
        help="League key: EPL, Bundesliga, Championship, SerieA, LaLiga, MLS",
    )
    default_fx = ROOT / "data" / "interim" / "fixtures_upcoming_EPL.csv"
    p.add_argument(
        "--fixtures",
        default=None,
        help="CSV: match_id,date,home_team,away_team "
        "(default: data/interim/fixtures_upcoming_{league}.csv)",
    )
    p.add_argument(
        "--odds-file",
        default=None,
        help="CSV keyed by match_id with book_over25/book_under25 and/or "
        "ref_over25/ref_under25 and optional extra book columns for consensus",
    )
    p.add_argument(
        "--late-info",
        default=None,
        help="Optional CSV: confirmed XI / injuries / lam_mult overrides",
    )
    p.add_argument("--out", default=None, help="Output CSV (default data/processed/gameday_sheet_{league}.csv)")
    p.add_argument("--update-data", action="store_true", help="Run scripts/update_data.py first")
    p.add_argument(
        "--refresh-fixtures",
        action="store_true",
        help="Refresh upcoming fixtures for --league before running",
    )
    p.add_argument(
        "--refresh-odds",
        action="store_true",
        help="Refresh Pinnacle OU 2.5 odds for --league before running",
    )
    p.add_argument(
        "--no-pinnacle",
        action="store_true",
        help="Do not auto-load Pinnacle sharp reference odds",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Skip residual OOS fit (faster; base+calib+intercept only)",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from origination.utils.league_registry import get_league
    from origination.data_ingestion.fixtures_upcoming import (
        fixtures_paths,
        refresh_upcoming_fixtures_for_league,
    )
    from origination.data_ingestion.pinnacle_odds import (
        load_pinnacle_meta,
        load_pinnacle_odds,
        refresh_pinnacle_odds,
    )

    league = get_league(args.league)
    league_key = league["key"]
    cfg_path = Path(args.config if args.config != "configs/default.yaml" else league["config"])
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    setup_logging(args.log_level)
    set_global_seed(int(cfg.get("project", {}).get("seed", 42)))
    data_dir = resolve_data_dir(cfg)

    win = ROOT / ".venv" / "Scripts" / "python.exe"
    unix = ROOT / ".venv" / "bin" / "python"
    if win.exists():
        py_exe = str(win)
    elif unix.exists():
        py_exe = str(unix)
    else:
        py_exe = sys.executable

    if args.update_data and league_key == "EPL":
        logger.info("Updating data pipeline…")
        cmd = [py_exe, str(ROOT / "scripts" / "update_data.py"), "--config", str(cfg_path)]
        subprocess.check_call(cmd, cwd=str(ROOT))
    else:
        if args.refresh_fixtures:
            logger.info("Refreshing upcoming fixtures for {}…", league_key)
            refresh_upcoming_fixtures_for_league(data_dir, league_key, cfg)
        if args.refresh_odds:
            logger.info("Refreshing Pinnacle odds for {}…", league_key)
            refresh_pinnacle_odds(data_dir, cfg=cfg, league_key=league_key)

    fx_default, meta_default, _ = fixtures_paths(data_dir, league_key)
    fixtures_path = Path(args.fixtures) if args.fixtures else fx_default
    if not fixtures_path.is_absolute():
        fixtures_path = ROOT / fixtures_path
    if not fixtures_path.is_file():
        raise SystemExit(
            f"Fixtures not found: {fixtures_path}\n"
            f"Run: python scripts/run_gameday_sheet.py --league {league_key} --refresh-fixtures"
        )

    # Refuse placeholder examples
    sample = fixtures_path.read_text(encoding="utf-8", errors="ignore").lower()
    if "20260816_arsenal_liverpool" in sample:
        raise SystemExit(
            "Refusing placeholder/example fixtures. "
            "Refresh real fixtures first."
        )

    meta_path = meta_default
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info(
            "Fixtures [{}] source={} fetched_at={} n={}",
            league_key,
            meta.get("source"),
            meta.get("fetched_at"),
            meta.get("n_fixtures"),
        )

    aligned_name = league["aligned"]
    history = load_aligned(data_dir / "interim" / aligned_name)
    if cfg.get("features", {}).get("groups", {}).get("understat_advanced", False):
        hist_us = load_understat_team_history(data_dir / "raw" / "understat")
        history = enrich_matches_with_understat_advanced(history, hist_us)

    upcoming = pd.read_csv(fixtures_path, parse_dates=["date"])
    for col in ("match_id", "date", "home_team", "away_team"):
        if col not in upcoming.columns:
            raise SystemExit(f"fixtures CSV missing column: {col}")
    if "season" not in upcoming.columns:
        upcoming["season"] = upcoming["date"].map(season_from_date)

    logger.info("Using {} fixtures from {}", len(upcoming), fixtures_path)
    for _, r in upcoming.iterrows():
        logger.info(
            "  {}  {} vs {}",
            pd.Timestamp(r["date"]).date(),
            r["home_team"],
            r["away_team"],
        )

    odds: pd.DataFrame | None = None
    if not args.no_pinnacle:
        pin = load_pinnacle_odds(data_dir, league_key)
        if len(pin):
            odds = pin.copy()
            pmeta = load_pinnacle_meta(data_dir, league_key) or {}
            logger.info(
                "Pinnacle sharp odds [{}]: {} rows (fetched_at={})",
                league_key,
                len(pin),
                pmeta.get("fetched_at"),
            )
        else:
            logger.warning(
                "No Pinnacle odds on disk for {} — pack flags need sharp prices. "
                "Pass --refresh-odds",
                league_key,
            )

    if args.odds_file:
        user = load_odds_csv(Path(args.odds_file))
        # Map accidental ref_/pin_ columns in user file → book_*
        rename = {}
        cols_l = {c.lower(): c for c in user.columns}
        if "book_over25" not in cols_l:
            for alias in ("ref_over25", "pin_over25", "over25"):
                if alias in cols_l:
                    rename[cols_l[alias]] = "book_over25"
                    break
        if "book_under25" not in cols_l:
            for alias in ("ref_under25", "pin_under25", "under25"):
                if alias in cols_l:
                    rename[cols_l[alias]] = "book_under25"
                    break
        if rename:
            user = user.rename(columns=rename)
            logger.info("Mapped user odds columns {} → book_*", rename)
        keep_u = [c for c in user.columns if c == "match_id" or c.startswith("book_") or c.startswith("user_")]
        user = user[keep_u]
        if odds is None:
            odds = user
        else:
            odds = odds.merge(user, on="match_id", how="outer")
        logger.info("User/book odds file: {} ({} rows)", args.odds_file, len(user))

    late = load_late_info_csv(Path(args.late_info)) if args.late_info else None

    # Pre-build features so late-info can merge before predict
    hist = history.dropna(subset=["home_goals", "away_goals"]).copy()
    combo = pd.concat([hist, upcoming], ignore_index=True, sort=False)
    for col in ("home_goals", "away_goals"):
        if col not in combo.columns:
            combo[col] = pd.NA
        mask = combo["match_id"].isin(upcoming["match_id"])
        combo.loc[mask, col] = pd.NA
    if "season" not in combo.columns or combo["season"].isna().any():
        combo["season"] = combo["date"].map(season_from_date)
    features = build_feature_matrix(combo, cfg.get("features", {}))
    if late is not None:
        features = apply_late_info_to_features(features, late, cfg)
        logger.info("Applied late info for {} rows", len(late))

    preds = predict_upcoming(
        history,
        upcoming,
        cfg,
        odds=odds,
        features=features,
        apply_residual=not args.fast,
    )
    active_packs = list(league.get("packs") or [])
    sheet = build_gameday_sheet(preds, cfg, active_packs=active_packs or None)

    default_out = (
        data_dir / "processed" / "gameday_sheet.csv"
        if league_key == "EPL"
        else data_dir / "processed" / f"gameday_sheet_{league_key}.csv"
    )
    out_path = Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out_path, index=False)
    logger.info("Wrote {}", out_path)

    # Compact console view
    show = [
        "date",
        "home_team",
        "away_team",
        "odds_status",
        "systems_flagged",
        "proj_total_goals",
        "fair_odds_under25",
        "fair_odds_over25",
        "pin_under25",
        "pin_over25",
        "odds_1x2_h",
        "odds_1x2_a",
        "book_under25",
        "book_over25",
        "edge_under_vs_pinnacle",
        "edge_over_vs_pinnacle",
        "edge_1x2_h",
        "edge_1x2_a",
        "edge_under_vs_book",
        "edge_over_vs_book",
    ]
    show += [c for c in sheet.columns if str(c).startswith("flag_") and c != "flag_any"]
    show = [c for c in show if c in sheet.columns]
    with pd.option_context("display.max_columns", 24, "display.width", 200):
        print(sheet[show].to_string(index=False))

    flag_cols = [c for c in sheet.columns if str(c).startswith("flag_") and c != "flag_any"]
    parts = [f"{c}={int(sheet[c].sum())}" for c in flag_cols] if len(sheet) else []
    print(f"\nFlags [{league_key}]: " + (" | ".join(parts) if parts else "none"))
    if "odds_status" in sheet.columns and len(sheet):
        missing = int((sheet["odds_status"] == "MISSING").sum())
        ou_n = int(sheet["has_pin_ou"].sum()) if "has_pin_ou" in sheet.columns else 0
        ml_n = int(sheet["has_pin_1x2"].sum()) if "has_pin_1x2" in sheet.columns else 0
        print(
            f"Odds coverage [{league_key}]: OU={ou_n}/{len(sheet)}  "
            f"1X2={ml_n}/{len(sheet)}  missing={missing}"
        )
    if "systems_flagged" in sheet.columns and len(sheet):
        fired = sheet.loc[sheet["systems_flagged"].astype(str).str.len() > 0]
        if len(fired):
            print("\nQualified this slate:")
            for _, r in fired.iterrows():
                print(
                    f"  {str(r.get('date',''))[:10]}  {r.get('home_team')} vs {r.get('away_team')}  "
                    f"-> {r.get('systems_flagged')}  [{r.get('odds_status')}]"
                )
    print(f"Full sheet: {out_path}")


if __name__ == "__main__":
    main()
