"""
FBref (StatsBomb-powered) advanced stats ingestion — scaffold.

Full scrape is rate-limited and brittle; this module provides:
- Configurable enable flag
- Team-match log download hooks
- Placeholder schema for progressive actions, SCA/GCA, pressures

Enable in config once football-data + Understat baseline CLV is solid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class FBrefIngester:
    """Scaffold for FBref team match logs."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch_and_load(
        self,
        league_slug: str,
        start_season: int,
        end_season: int | None = None,
    ) -> pd.DataFrame:
        """
        Returns empty frame with expected schema if not yet implemented.

        Expected future columns (team-match grain, one row per team per match):
        progressive_passes, progressive_carries, sca, gca, possession,
        touches_att_3rd, pressures, tackles, set_piece_xg, ...
        """
        logger.warning(
            "FBref ingestion is scaffolded only (league={}). Returning empty frame.",
            league_slug,
        )
        cols = [
            "date",
            "team",
            "opponent",
            "venue",
            "progressive_passes",
            "progressive_carries",
            "sca",
            "gca",
            "possession",
            "touches_att_3rd",
            "pressures",
            "season",
            "league_fbref",
        ]
        return pd.DataFrame(columns=cols)


def ingest_fbref_from_config(cfg: dict[str, Any], data_dir: Path) -> pd.DataFrame | None:
    fb_cfg = cfg.get("data", {}).get("fbref", {})
    if not fb_cfg.get("enabled", False):
        logger.info("FBref disabled in config")
        return None
    ingester = FBrefIngester(raw_dir=data_dir / "raw" / "fbref")
    frames: list[pd.DataFrame] = []
    for league in cfg.get("leagues", []):
        slug = league.get("fbref")
        if not slug:
            continue
        frames.append(
            ingester.fetch_and_load(
                league_slug=slug,
                start_season=int(league["start_season"]),
                end_season=league.get("end_season"),
            )
        )
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)
