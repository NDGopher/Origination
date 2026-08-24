#!/usr/bin/env python
"""Build possession-value team-match parquet from cached Understat match JSON."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.data_ingestion.align import load_aligned
from origination.features.possession_value import build_possession_value_table
from origination.utils import load_config, resolve_data_dir, setup_logging


def main() -> None:
    setup_logging("INFO")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data_dir = resolve_data_dir(cfg)
    matches = load_aligned(data_dir / "interim" / "matches_aligned.parquet")
    cache = data_dir / "raw" / "understat" / "match_rosters"
    out = data_dir / "interim" / "understat_possession_value.parquet"
    df = build_possession_value_table(cache, matches, max_workers=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} rows={len(df)} mean_obv={df['pv_obv'].mean():.3f}")


if __name__ == "__main__":
    main()
