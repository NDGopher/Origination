#!/usr/bin/env python
"""
Refresh gameday DATA sources (not odds).

Smart defaults:
  - Upcoming fixtures for all live leagues (always — these change)
  - Optional light results/xG refresh WITHOUT --force (cached seasons skipped)
  - Does NOT re-download immutable historical seasons

Does not touch Pinnacle odds (use refresh_gameday_odds.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.utils import load_config, resolve_data_dir, setup_logging
from origination.utils.league_registry import get_league

LIVE_LEAGUES = ["EPL", "Bundesliga", "LaLiga", "SerieA", "PrimeiraLiga"]
STAMP = ROOT / "data" / "gameday" / "last_data_update.json"


def _py_stamp(kind: str, detail: dict) -> None:
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": kind,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **detail,
    }
    # merge with prior odds stamp if present
    prior = {}
    if STAMP.exists():
        try:
            prior = json.loads(STAMP.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prior = {}
    prior["data"] = payload
    prior["updated_at_data"] = payload["updated_at"]
    STAMP.write_text(json.dumps(prior, indent=2), encoding="utf-8")


def refresh_fixtures(leagues: list[str]) -> dict[str, int]:
    from origination.data_ingestion.fixtures_upcoming import refresh_upcoming_fixtures_for_league

    counts = {}
    for key in leagues:
        info = get_league(key)
        cfg = load_config(ROOT / info["config"])
        data_dir = resolve_data_dir(cfg)
        print(f"  Fixtures [{key}]…", flush=True)
        df, meta = refresh_upcoming_fixtures_for_league(data_dir, key, cfg)
        n = int(len(df)) if df is not None else 0
        counts[key] = n
        print(f"    → {n} matches  source={meta.get('source')}", flush=True)
    return counts


def light_results_refresh(leagues: list[str]) -> None:
    """Pull current-season results using cache (no --force)."""
    from origination.data_ingestion import (
        build_aligned_from_config,
        ingest_football_data_from_config,
        ingest_understat_from_config,
    )

    for key in leagues:
        info = get_league(key)
        cfg = load_config(ROOT / info["config"])
        data_dir = resolve_data_dir(cfg)
        print(f"  Results/align [{key}] (cache-friendly)…", flush=True)
        try:
            fd = ingest_football_data_from_config(cfg, data_dir)
            us = None
            if cfg.get("data", {}).get("understat", {}).get("enabled"):
                try:
                    us = ingest_understat_from_config(cfg, data_dir)
                except Exception as exc:  # noqa: BLE001
                    print(f"    Understat skip: {exc}", flush=True)
            build_aligned_from_config(cfg, data_dir, fd, us, None)
            print(f"    → aligned OK ({len(fd)} FD rows)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"    WARN: {exc}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Refresh gameday data sources")
    p.add_argument(
        "--fixtures-only",
        action="store_true",
        help="Only upcoming fixtures (fastest; recommended most days)",
    )
    p.add_argument(
        "--with-results",
        action="store_true",
        help="Also refresh recent results/xG using cache (no full re-download)",
    )
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    setup_logging(args.log_level)

    print("=== Update Data Sources ===", flush=True)
    print("Live leagues:", ", ".join(LIVE_LEAGUES), flush=True)
    counts = refresh_fixtures(LIVE_LEAGUES)
    if args.with_results and not args.fixtures_only:
        light_results_refresh(LIVE_LEAGUES)
    elif not args.fixtures_only and args.with_results is False:
        # Default daily: fixtures only (idiot-proof fast path)
        print("  (Skipping historical re-download — use --with-results if you need new scores/xG)", flush=True)

    _py_stamp(
        "data",
        {
            "fixtures_only": bool(args.fixtures_only or not args.with_results),
            "with_results": bool(args.with_results),
            "fixture_counts": counts,
            "leagues": LIVE_LEAGUES,
        },
    )
    print("Done. Stamp →", STAMP, flush=True)


if __name__ == "__main__":
    main()
