#!/usr/bin/env python
"""
Refresh Pinnacle odds (OU 2.5 + 1X2 + AH) for all live gameday leagues.

Requires fixtures on disk (run refresh_gameday_data.py first if missing).
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
# Also refresh odds for score-prediction leagues (does not change live pack rules)
SCORE_EXTRA_LEAGUES = [
    "Championship",
    "Ligue1",
    "Eredivisie",
    "Belgium",
    "MLS",
    "Scotland",
    "Turkey",
]
STAMP = ROOT / "data" / "gameday" / "last_data_update.json"


def main() -> None:
    p = argparse.ArgumentParser(description="Refresh Pinnacle odds for live leagues")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    setup_logging(args.log_level)

    from origination.data_ingestion.fixtures_upcoming import load_upcoming_fixtures
    from origination.data_ingestion.pinnacle_odds import refresh_pinnacle_odds

    print("=== Update Odds (Pinnacle OU / 1X2 / AH) ===", flush=True)
    summary = {}
    for key in LIVE_LEAGUES + SCORE_EXTRA_LEAGUES:
        info = get_league(key)
        cfg = load_config(ROOT / info["config"])
        data_dir = resolve_data_dir(cfg)
        print(f"  Odds [{key}]…", flush=True)
        try:
            fx = load_upcoming_fixtures(data_dir, league_key=key)
            if fx is None or len(fx) == 0:
                print("    SKIP — no fixtures on disk. Run Update Data Sources first.", flush=True)
                summary[key] = {"ok": False, "error": "no_fixtures"}
                continue
            matched, meta = refresh_pinnacle_odds(
                data_dir, fixtures=fx, cfg=cfg, league_key=key
            )
            summary[key] = {
                "ok": True,
                "n_fixtures": int(len(matched)) if matched is not None else 0,
                "n_with_ou25": meta.get("n_with_ou25"),
                "n_with_1x2": meta.get("n_with_1x2"),
                "n_with_ah": meta.get("n_with_ah"),
                "fetched_at": meta.get("fetched_at")
                or datetime.now(timezone.utc).isoformat(),
            }
            print(
                f"    -> OU={summary[key]['n_with_ou25']}  "
                f"1X2={summary[key]['n_with_1x2']}  AH={summary[key]['n_with_ah']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR: {exc}", flush=True)
            summary[key] = {"ok": False, "error": str(exc)}

    STAMP.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if STAMP.exists():
        try:
            prior = json.loads(STAMP.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prior = {}
    prior["odds"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "leagues": summary,
    }
    prior["updated_at_odds"] = prior["odds"]["updated_at"]
    STAMP.write_text(json.dumps(prior, indent=2), encoding="utf-8")
    print("Done. Stamp ->", STAMP, flush=True)


if __name__ == "__main__":
    main()
