"""
Understat.com match-level xG ingestion via the getLeagueData JSON API.

Understat no longer embeds datesData in the league HTML; the SPA loads
https://understat.com/getLeagueData/{league}/{season} with XHR headers.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from origination.utils.team_names import DEFAULT_MAPPER, TeamNameMapper


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get_json(url: str, referer: str) -> dict[str, Any]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": referer,
    }
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _season_years(start: int, end: int | None) -> list[int]:
    if end is None:
        today = datetime.utcnow()
        # Current season start year; file may 404 early in August before data exists
        end = today.year if today.month >= 8 else today.year - 1
    return list(range(start, end + 1))


class UnderstatIngester:
    """Fetch and cache Understat league JSON → match-level xG (+ team history)."""

    def __init__(
        self,
        raw_dir: Path,
        base_url: str = "https://understat.com",
        mapper: TeamNameMapper | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.base_url = base_url.rstrip("/")
        self.mapper = mapper or DEFAULT_MAPPER

    def season_json_path(self, league: str, season_start: int) -> Path:
        return self.raw_dir / league / f"{season_start}_league.json"

    def fetch_season(self, league: str, season_start: int, *, force: bool = False) -> Path | None:
        dest = self.season_json_path(league, season_start)
        if dest.exists() and not force:
            return dest
        url = f"{self.base_url}/getLeagueData/{league}/{season_start}"
        referer = f"{self.base_url}/league/{league}/{season_start}"
        try:
            data = _get_json(url, referer=referer)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Understat fetch failed {} {}: {}", league, season_start, exc)
            return None
        if not data.get("dates"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(data), encoding="utf-8")
            logger.info(
                "Understat {} {} has no fixtures yet (season not started) -> {}",
                league,
                season_start,
                dest,
            )
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data), encoding="utf-8")
        logger.info("Fetched Understat {} {} -> {} ({} matches)", league, season_start, dest, len(data["dates"]))
        return dest

    def parse_season(self, league: str, season_start: int) -> pd.DataFrame:
        path = self.season_json_path(league, season_start)
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        for m in data.get("dates", []):
            try:
                home = m["h"]["title"]
                away = m["a"]["title"]
                if not m.get("isResult"):
                    continue
                rows.append(
                    {
                        "understat_id": int(m["id"]),
                        "is_result": True,
                        "date": pd.to_datetime(m["datetime"]),
                        "home_team_raw": home,
                        "away_team_raw": away,
                        "home_team": self.mapper.canonicalize(home),
                        "away_team": self.mapper.canonicalize(away),
                        "home_goals": float(m["goals"]["h"]),
                        "away_goals": float(m["goals"]["a"]),
                        "home_xg": float(m["xG"]["h"]),
                        "away_xg": float(m["xG"]["a"]),
                        "league_understat": league,
                        "season": season_start,
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Skip understat match: {}", exc)

        # Team-match history (PPDA, deep completions, npxG) — valuable for features
        hist_rows: list[dict[str, Any]] = []
        for team in (data.get("teams") or {}).values():
            title = self.mapper.canonicalize(team["title"])
            for match in team.get("history", []):
                hist_rows.append(
                    {
                        "date": pd.to_datetime(match["date"]),
                        "team": title,
                        "is_home": match.get("h_a") == "h",
                        "xg_for": float(match.get("xG", 0) or 0),
                        "xg_against": float(match.get("xGA", 0) or 0),
                        "npxg_for": float(match.get("npxG", 0) or 0),
                        "npxg_against": float(match.get("npxGA", 0) or 0),
                        "deep": float(match.get("deep", 0) or 0),
                        "deep_allowed": float(match.get("deep_allowed", 0) or 0),
                        "ppda_att": float((match.get("ppda") or {}).get("att") or 0),
                        "ppda_def": float((match.get("ppda") or {}).get("def") or 0),
                        "scored": float(match.get("scored", 0) or 0),
                        "missed": float(match.get("missed", 0) or 0),
                        "season": season_start,
                        "league_understat": league,
                    }
                )
        if hist_rows:
            hist = pd.DataFrame(hist_rows)
            hist["ppda"] = hist.apply(
                lambda r: (r["ppda_att"] / r["ppda_def"]) if r["ppda_def"] else float("nan"),
                axis=1,
            )
            hist_path = self.raw_dir / league / f"{season_start}_team_history.parquet"
            hist.to_parquet(hist_path, index=False)

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values("date").reset_index(drop=True)
        parquet = self.raw_dir / league / f"{season_start}_matches.parquet"
        df.to_parquet(parquet, index=False)
        return df

    def fetch_and_load(
        self,
        league: str,
        start_season: int,
        end_season: int | None = None,
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for y in _season_years(start_season, end_season):
            path = self.fetch_season(league, y, force=force)
            if path is None:
                continue
            try:
                parsed = self.parse_season(league, y)
                if parsed is None or len(parsed) == 0:
                    logger.info("Understat {} {}: 0 results", league, y)
                    continue
                frames.append(parsed)
            except Exception as exc:  # noqa: BLE001
                logger.error("Understat parse {} {}: {}", league, y, exc)
        if not frames:
            raise RuntimeError(f"No Understat data for {league}")
        df = pd.concat(frames, ignore_index=True)
        logger.info("Understat {}: {} matches", league, len(df))
        return df


def ingest_understat_from_config(cfg: dict[str, Any], data_dir: Path) -> pd.DataFrame | None:
    us_cfg = cfg.get("data", {}).get("understat", {})
    if not us_cfg.get("enabled", False):
        logger.info("Understat disabled in config")
        return None
    ingester = UnderstatIngester(
        raw_dir=data_dir / "raw" / "understat",
        base_url=us_cfg.get("base_url", "https://understat.com"),
    )
    frames: list[pd.DataFrame] = []
    for league in cfg.get("leagues", []):
        us_league = league.get("understat")
        if not us_league:
            continue
        try:
            frames.append(
                ingester.fetch_and_load(
                    league=us_league,
                    start_season=int(league["start_season"]),
                    end_season=league.get("end_season"),
                )
            )
        except RuntimeError as exc:
            logger.error("{}", exc)
    if not frames:
        logger.warning("Understat enabled but no data loaded — continuing without xG")
        return None
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
