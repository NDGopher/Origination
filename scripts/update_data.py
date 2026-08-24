#!/usr/bin/env python
"""Download / refresh all configured data sources and write aligned Parquet.

Also refreshes upcoming EPL fixtures (Premier League Pulse API) for gameday.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger

from origination.data_ingestion import (
    build_aligned_from_config,
    ingest_fbref_from_config,
    ingest_football_data_from_config,
    ingest_pinnacle_odds_from_config,
    ingest_understat_from_config,
    ingest_upcoming_fixtures_from_config,
)
from origination.utils import load_config, resolve_data_dir, set_global_seed, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update Origination data pipeline")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seasons", default=None, help="Comma-separated start years, e.g. 2014,2015")
    p.add_argument("--force", action="store_true", help="Re-download raw files")
    p.add_argument(
        "--fixtures-only",
        action="store_true",
        help="Only refresh upcoming EPL fixtures (skip FD/Understat/align)",
    )
    p.add_argument(
        "--odds-only",
        action="store_true",
        help="Only refresh Pinnacle OU 2.5 odds (requires fixtures on disk)",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    setup_logging(args.log_level)
    set_global_seed(int(cfg.get("project", {}).get("seed", 42)))
    data_dir = resolve_data_dir(cfg)

    if args.seasons:
        years = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
        for league in cfg.get("leagues", []):
            league["start_season"] = min(years)
            league["end_season"] = max(years)

    if not args.fixtures_only and not args.odds_only:
        logger.info("=== football-data.co.uk ===")
        fd = ingest_football_data_from_config(cfg, data_dir)
        fd_path = data_dir / "interim" / "football_data.parquet"
        fd_path.parent.mkdir(parents=True, exist_ok=True)
        fd.to_parquet(fd_path, index=False)
        logger.info("Saved {}", fd_path)

        logger.info("=== Understat ===")
        understat = ingest_understat_from_config(cfg, data_dir)
        if understat is not None:
            us_path = data_dir / "interim" / "understat.parquet"
            understat.to_parquet(us_path, index=False)
            logger.info("Saved {}", us_path)

        logger.info("=== FBref ===")
        fbref = ingest_fbref_from_config(cfg, data_dir)

        logger.info("=== Align ===")
        aligned = build_aligned_from_config(cfg, data_dir, fd, understat, fbref)
        logger.info("Aligned matches: {}", len(aligned))

    if not args.odds_only:
        logger.info("=== Upcoming EPL fixtures ===")
        try:
            fixtures = ingest_upcoming_fixtures_from_config(cfg, data_dir)
            logger.info("Upcoming fixtures in window: {}", len(fixtures))
            if len(fixtures):
                show = fixtures[["date", "home_team", "away_team"]].head(20)
                for _, r in show.iterrows():
                    logger.info(
                        "  {}  {} vs {}",
                        r["date"].date() if hasattr(r["date"], "date") else r["date"],
                        r["home_team"],
                        r["away_team"],
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error("Upcoming fixtures refresh FAILED: {}", exc)
            if args.fixtures_only:
                raise SystemExit(1) from exc
            logger.error(
                "Gameday will warn until fixtures refresh succeeds. "
                "Retry: python scripts/update_data.py --fixtures-only"
            )

    logger.info("=== Pinnacle OU 2.5 (sharp reference) ===")
    try:
        pin = ingest_pinnacle_odds_from_config(cfg, data_dir)
        n_px = int(pin["pin_over25"].notna().sum()) if len(pin) and "pin_over25" in pin.columns else 0
        logger.info("Pinnacle-matched fixtures with OU2.5: {}", n_px)
        if len(pin) and "pin_over25" in pin.columns:
            for _, r in pin[pin["pin_over25"].notna()].head(15).iterrows():
                logger.info(
                    "  {}  {} vs {}  O {} / U {}",
                    r.get("date"),
                    r.get("home_team"),
                    r.get("away_team"),
                    r.get("pin_over25"),
                    r.get("pin_under25"),
                )
    except Exception as exc:  # noqa: BLE001
        logger.error("Pinnacle odds refresh FAILED: {}", exc)
        if args.odds_only:
            raise SystemExit(1) from exc
        logger.error(
            "Gameday can still run without Pinnacle; sharp edges/pack flags need a retry. "
            "Retry: python scripts/update_data.py --odds-only"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
