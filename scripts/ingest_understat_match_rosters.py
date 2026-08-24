#!/usr/bin/env python
"""Ingest Understat match rosters for aligned EPL matches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.understat_match_rosters import ingest_match_rosters
from origination.utils import load_config, resolve_data_dir, setup_logging


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    cache = data_dir / "raw" / "understat" / "match_rosters"
    parquet = data_dir / "interim" / "understat_match_rosters.parquet"
    df = ingest_match_rosters(
        matches,
        cache_dir=cache,
        parquet_path=parquet,
        max_workers=args.workers,
        limit=args.limit,
        force=args.force,
    )
    print(f"appearances={len(df)} unique_matches={df['understat_id'].nunique() if len(df) else 0}")


if __name__ == "__main__":
    main()
