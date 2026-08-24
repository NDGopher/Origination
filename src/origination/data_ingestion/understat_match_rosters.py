"""Understat match-level roster ingest via getMatchData/{understat_id}.

Caches raw JSON per match and a flattened appearances parquet.
Leakage note: minutes / xG on match t are post-match — providers must only
use appearances with date < target match date (or prior understat fixtures).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger


def _float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def fetch_match_rosters(
    understat_id: int | str,
    *,
    base_url: str = "https://understat.com",
    session: requests.Session | None = None,
    timeout: float = 45.0,
) -> dict[str, Any]:
    uid = int(understat_id)
    sess = session or requests.Session()
    url = f"{base_url.rstrip('/')}/getMatchData/{uid}"
    r = sess.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; Origination/1.0)",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{base_url.rstrip('/')}/match/{uid}",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def roster_json_to_rows(
    understat_id: int,
    payload: dict[str, Any],
    *,
    match_date: Any = None,
    home_team: str | None = None,
    away_team: str | None = None,
    season: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rosters = payload.get("rosters") or {}
    for side_key, team_name in (("h", home_team), ("a", away_team)):
        block = rosters.get(side_key) or {}
        if not isinstance(block, dict):
            continue
        for _rid, p in block.items():
            if not isinstance(p, dict):
                continue
            minutes = _float(p.get("time"))
            rows.append(
                {
                    "understat_id": int(understat_id),
                    "date": match_date,
                    "season": season,
                    "side": side_key,
                    "team": team_name,
                    "team_id": str(p.get("team_id") or ""),
                    "player_id": str(p.get("player_id") or ""),
                    "player_name": p.get("player"),
                    "position": str(p.get("position") or ""),
                    "minutes": minutes,
                    "starter": minutes >= 45.0,
                    "goals": _float(p.get("goals")),
                    "assists": _float(p.get("assists")),
                    "shots": _float(p.get("shots")),
                    "xG": _float(p.get("xG")),
                    "xA": _float(p.get("xA")),
                    "xGChain": _float(p.get("xGChain")),
                    "xGBuildup": _float(p.get("xGBuildup")),
                    "yellow_card": _float(p.get("yellow_card")),
                    "red_card": _float(p.get("red_card")),
                }
            )
    return rows


def ingest_match_rosters(
    matches: pd.DataFrame,
    *,
    cache_dir: Path,
    parquet_path: Path | None = None,
    base_url: str = "https://understat.com",
    max_workers: int = 12,
    sleep_s: float = 0.0,
    limit: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download/cache Understat rosters for matches with ``understat_id``.

    ``matches`` should include understat_id, date, home_team, away_team, season.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = Path(parquet_path) if parquet_path else cache_dir.parent / "match_rosters.parquet"

    need = matches.dropna(subset=["understat_id"]).copy()
    need["understat_id"] = need["understat_id"].astype(int)
    need = need.drop_duplicates("understat_id")
    if limit is not None:
        need = need.head(int(limit))

    meta = {
        int(r.understat_id): r
        for r in need.itertuples(index=False)
    }
    ids = list(meta.keys())
    logger.info("Ingesting {} Understat match rosters → {}", len(ids), cache_dir)

    session_local = __import__("threading").local()

    def _session() -> requests.Session:
        s = getattr(session_local, "s", None)
        if s is None:
            s = requests.Session()
            session_local.s = s
        return s

    def _one(uid: int) -> tuple[int, list[dict[str, Any]] | None, str | None]:
        path = cache_dir / f"{uid}.json"
        try:
            if path.exists() and not force:
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                if sleep_s > 0:
                    time.sleep(sleep_s)
                payload = fetch_match_rosters(uid, base_url=base_url, session=_session())
                path.write_text(json.dumps(payload), encoding="utf-8")
            m = meta[uid]
            rows = roster_json_to_rows(
                uid,
                payload,
                match_date=getattr(m, "date", None),
                home_team=getattr(m, "home_team", None),
                away_team=getattr(m, "away_team", None),
                season=int(getattr(m, "season")) if getattr(m, "season", None) is not None else None,
            )
            return uid, rows, None
        except Exception as exc:  # noqa: BLE001
            return uid, None, str(exc)

    all_rows: list[dict[str, Any]] = []
    errors = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, uid) for uid in ids]
        for fut in as_completed(futs):
            uid, rows, err = fut.result()
            done += 1
            if err:
                errors += 1
                if errors <= 5:
                    logger.warning("Roster fetch failed id={}: {}", uid, err)
            elif rows:
                all_rows.extend(rows)
            if done % 250 == 0 or done == len(ids):
                logger.info("Roster progress {}/{} (errors={})", done, len(ids), errors)

    df = pd.DataFrame(all_rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        logger.info("Wrote {} appearances → {}", len(df), parquet_path)
    else:
        logger.warning("No roster rows ingested")
    return df


def load_match_rosters(parquet_path: Path) -> pd.DataFrame:
    path = Path(parquet_path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
