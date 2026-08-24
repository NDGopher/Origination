"""
Upcoming EPL fixtures — automatic scrape for gameday.

Primary source: Premier League Pulse API (footballapi.pulselive.com)
  — official schedule including future kickoffs (statuses U/L).

Fallbacks (when Pulse has no upcoming rows):
  1. football-data.co.uk E0 season CSV rows with blank scores (Div must be E0)
  2. Understat getLeagueData dates with isResult=False

Writes:
  data/interim/fixtures_upcoming_EPL.csv
  data/interim/fixtures_upcoming_EPL.meta.json
  data/gameday/fixtures.csv  (synced copy for the UI default)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from origination.utils.seeding import season_from_date
from origination.utils.team_names import DEFAULT_MAPPER, TeamNameMapper

PULSE_BASE = "https://footballapi.pulselive.com/football"
PULSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Origin": "https://www.premierleague.com",
    "Referer": "https://www.premierleague.com/fixtures",
    "Accept": "application/json",
}

FIXTURES_CSV_NAME = "fixtures_upcoming_EPL.csv"
FIXTURES_META_NAME = "fixtures_upcoming_EPL.meta.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _match_id(date: pd.Timestamp, home: str, away: str) -> str:
    return (
        f"{date.strftime('%Y%m%d')}_"
        f"{home.replace(' ', '')}_"
        f"{away.replace(' ', '')}"
    )


def _finalize_frame(rows: list[dict[str, Any]], mapper: TeamNameMapper) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "match_id",
                "date",
                "kickoff_utc",
                "home_team",
                "away_team",
                "season",
                "gameweek",
                "status",
                "source",
            ]
        )
    df = pd.DataFrame(rows)
    df["home_team"] = mapper.map_series(df["home_team_raw"])
    df["away_team"] = mapper.map_series(df["away_team_raw"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["season"] = df["date"].map(season_from_date)
    df["match_id"] = [
        _match_id(d, h, a)
        for d, h, a in zip(df["date"], df["home_team"], df["away_team"], strict=True)
    ]
    cols = [
        "match_id",
        "date",
        "kickoff_utc",
        "home_team",
        "away_team",
        "season",
        "gameweek",
        "status",
        "source",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[cols].drop_duplicates(subset=["match_id"]).sort_values(["date", "kickoff_utc", "match_id"])
    return df.reset_index(drop=True)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _pulse_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{PULSE_BASE}/{path.lstrip('/')}"
    resp = requests.get(url, params=params or {}, headers=PULSE_HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _latest_comp_season_id() -> tuple[int, str]:
    data = _pulse_get("competitions/1/compseasons", {"page": 0, "pageSize": 20})
    content = data.get("content") or []
    if not content:
        raise RuntimeError("Premier League Pulse: no competition seasons returned")
    # API returns newest first
    top = content[0]
    return int(top["id"]), str(top.get("label") or "")


def fetch_pulse_upcoming(
    *,
    days_ahead: int = 21,
    include_live: bool = True,
    mapper: TeamNameMapper | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch unplayed (and optionally live) EPL fixtures from the official Pulse API."""
    mapper = mapper or DEFAULT_MAPPER
    season_id, season_label = _latest_comp_season_id()
    statuses = "U,L" if include_live else "U"
    now = _utc_now()
    horizon = now + timedelta(days=int(days_ahead))

    rows: list[dict[str, Any]] = []
    page = 0
    num_pages = 1
    while page < num_pages:
        data = _pulse_get(
            "fixtures",
            {
                "comps": 1,
                "compSeasons": season_id,
                "page": page,
                "pageSize": 50,
                "sort": "asc",
                "statuses": statuses,
            },
        )
        info = data.get("pageInfo") or {}
        num_pages = int(info.get("numPages") or 1)
        for m in data.get("content") or []:
            teams = m.get("teams") or []
            if len(teams) < 2:
                continue
            home = (teams[0].get("team") or {}).get("name")
            away = (teams[1].get("team") or {}).get("name")
            if not home or not away:
                continue
            kick = m.get("kickoff") or m.get("provisionalKickoff") or {}
            millis = kick.get("millis")
            if millis is None:
                continue
            kickoff = datetime.fromtimestamp(float(millis) / 1000.0, tz=timezone.utc)
            # Keep fixtures from start-of-today UTC through horizon
            if kickoff < (now - timedelta(hours=6)):
                # allow slight overlap for in-progress; skip older
                if (m.get("status") or "") not in ("L", "LIVE"):
                    continue
            if kickoff > horizon:
                continue
            gw = m.get("gameweek") or {}
            rows.append(
                {
                    "home_team_raw": home,
                    "away_team_raw": away,
                    "date": kickoff.date().isoformat(),
                    "kickoff_utc": kickoff.isoformat(),
                    "gameweek": int(gw.get("gameweek") or 0) or pd.NA,
                    "status": m.get("status") or "U",
                    "source": "premierleague_pulse",
                    "pulse_fixture_id": m.get("id"),
                }
            )
        page += 1

    df = _finalize_frame(rows, mapper)
    meta = {
        "source": "premierleague_pulse",
        "source_detail": f"footballapi.pulselive.com comps=1 compSeasons={season_id}",
        "season_label": season_label,
        "comp_season_id": season_id,
        "days_ahead": days_ahead,
        "n_raw_rows": len(rows),
    }
    return df, meta


def _pulse_team_score(team_obj: dict[str, Any]) -> float | None:
    s = team_obj.get("score")
    if s is None:
        return None
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
    if isinstance(s, dict):
        for k in ("current", "ft", "fullTime", "score", "goals"):
            v = s.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def fetch_pulse_completed(
    *,
    mapper: TeamNameMapper | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Finished Premier League results from Pulse (statuses=C). Used when FD E0 2627 is missing."""
    mapper = mapper or DEFAULT_MAPPER
    season_id, season_label = _latest_comp_season_id()
    rows: list[dict[str, Any]] = []
    page = 0
    num_pages = 1
    while page < num_pages:
        data = _pulse_get(
            "fixtures",
            {
                "comps": 1,
                "compSeasons": season_id,
                "page": page,
                "pageSize": 50,
                "sort": "asc",
                "statuses": "C",
            },
        )
        info = data.get("pageInfo") or {}
        num_pages = int(info.get("numPages") or 1)
        for m in data.get("content") or []:
            teams = m.get("teams") or []
            if len(teams) < 2:
                continue
            home = (teams[0].get("team") or {}).get("name")
            away = (teams[1].get("team") or {}).get("name")
            hg = _pulse_team_score(teams[0])
            ag = _pulse_team_score(teams[1])
            if not home or not away or hg is None or ag is None:
                continue
            kick = m.get("kickoff") or m.get("provisionalKickoff") or {}
            millis = kick.get("millis")
            if millis is None:
                continue
            kickoff = datetime.fromtimestamp(float(millis) / 1000.0, tz=timezone.utc)
            home_c = mapper.canonicalize(home)
            away_c = mapper.canonicalize(away)
            rows.append(
                {
                    "date": kickoff.date().isoformat(),
                    "home_team": home_c,
                    "away_team": away_c,
                    "home_team_raw": home,
                    "away_team_raw": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "home_xg": pd.NA,
                    "away_xg": pd.NA,
                    "season": season_from_date(kickoff),
                    "source": "premierleague_pulse",
                    "is_result": True,
                }
            )
        page += 1

    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    meta = {
        "source": "premierleague_pulse_completed",
        "season_label": season_label,
        "comp_season_id": season_id,
        "n": int(len(df)),
    }
    return df, meta


def fetch_football_data_upcoming(
    data_dir: Path,
    *,
    league_code: str = "E0",
    base_url: str = "https://www.football-data.co.uk/mmz4281",
    mapper: TeamNameMapper | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Blank-score E0 rows from the current season CSV (when published)."""
    mapper = mapper or DEFAULT_MAPPER
    today = datetime.utcnow()
    season_start = today.year if today.month >= 8 else today.year - 1
    yy = f"{season_start % 100:02d}{(season_start + 1) % 100:02d}"
    url = f"{base_url.rstrip('/')}/{yy}/{league_code}.csv"
    raw_path = Path(data_dir) / "raw" / "football_data" / league_code / f"{yy}_fixtures_probe.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60, allow_redirects=True)
    if resp.status_code in (300, 404):
        return _finalize_frame([], mapper), {
            "source": "football_data",
            "url": url,
            "error": f"http_{resp.status_code}",
        }
    resp.raise_for_status()
    final = (resp.url or url).split("?")[0]
    if f"/{league_code}.csv".lower() not in final.lower():
        logger.warning("football-data fixtures {} redirected to {} — skip", url, final)
        return _finalize_frame([], mapper), {
            "source": "football_data",
            "url": url,
            "error": f"redirect:{final}",
        }
    raw_path.write_bytes(resp.content)
    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    if "Div" not in df.columns or "HomeTeam" not in df.columns:
        return _finalize_frame([], mapper), {"source": "football_data", "error": "bad_schema"}
    # Reject wrong division (e.g. EC mistakenly served as E0)
    divs = set(df["Div"].dropna().astype(str).unique())
    if league_code not in divs:
        logger.warning(
            "football-data {}/{} unexpected Div={} (wanted {}); skipping",
            yy,
            league_code,
            divs,
            league_code,
        )
        return _finalize_frame([], mapper), {
            "source": "football_data",
            "url": url,
            "error": f"wrong_div:{sorted(divs)}",
        }

    blank = df["FTHG"].isna() | (df["FTHG"].astype(str).str.strip() == "")
    upcoming = df.loc[blank].copy()
    if upcoming.empty:
        return _finalize_frame([], mapper), {"source": "football_data", "url": url, "n": 0}

    upcoming["Date"] = pd.to_datetime(upcoming["Date"], dayfirst=True, errors="coerce")
    upcoming = upcoming.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    rows = []
    for _, r in upcoming.iterrows():
        rows.append(
            {
                "home_team_raw": r["HomeTeam"],
                "away_team_raw": r["AwayTeam"],
                "date": r["Date"].date().isoformat(),
                "kickoff_utc": pd.Timestamp(r["Date"]).tz_localize("UTC").isoformat(),
                "gameweek": pd.NA,
                "status": "U",
                "source": "football_data",
            }
        )
    out = _finalize_frame(rows, mapper)
    return out, {"source": "football_data", "url": url, "season_yy": yy, "n": len(out)}


def fetch_understat_upcoming(
    data_dir: Path,
    *,
    league: str = "EPL",
    mapper: TeamNameMapper | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Understat dates with isResult=False for the current season start year."""
    mapper = mapper or DEFAULT_MAPPER
    today = datetime.utcnow()
    season_start = today.year if today.month >= 8 else today.year - 1
    url = f"https://understat.com/getLeagueData/{league}/{season_start}"
    headers = {
        "User-Agent": PULSE_HEADERS["User-Agent"],
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": f"https://understat.com/league/{league}/{season_start}",
    }
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for m in data.get("dates") or []:
        if m.get("isResult"):
            continue
        home = (m.get("h") or {}).get("title")
        away = (m.get("a") or {}).get("title")
        if not home or not away:
            continue
        dt = pd.to_datetime(m.get("datetime"), utc=True, errors="coerce")
        if pd.isna(dt):
            continue
        rows.append(
            {
                "home_team_raw": home,
                "away_team_raw": away,
                "date": dt.date().isoformat(),
                "kickoff_utc": dt.isoformat(),
                "gameweek": pd.NA,
                "status": "U",
                "source": "understat",
            }
        )
    out = _finalize_frame(rows, mapper)
    return out, {
        "source": "understat",
        "url": url,
        "season_start": season_start,
        "n": len(out),
    }


def fixtures_paths(data_dir: Path, league_key: str = "EPL") -> tuple[Path, Path, Path]:
    """EPL keeps legacy filenames; other leagues use fixtures_upcoming_{KEY}.csv."""
    interim = Path(data_dir) / "interim"
    gameday = Path(data_dir) / "gameday"
    key = (league_key or "EPL").strip()
    if key in ("EPL", "E0", ""):
        csv_name, meta_name, gd_name = FIXTURES_CSV_NAME, FIXTURES_META_NAME, "fixtures.csv"
    else:
        csv_name = f"fixtures_upcoming_{key}.csv"
        meta_name = f"fixtures_upcoming_{key}.meta.json"
        gd_name = f"fixtures_{key}.csv"
    return interim / csv_name, interim / meta_name, gameday / gd_name


def save_fixtures(
    df: pd.DataFrame,
    data_dir: Path,
    meta: dict[str, Any],
    *,
    league_key: str = "EPL",
) -> tuple[Path, Path, Path]:
    csv_path, meta_path, gameday_path = fixtures_paths(data_dir, league_key)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    gameday_path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    if len(out) and "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(csv_path, index=False)

    # UI sheet format
    slim = out[["match_id", "date", "home_team", "away_team"]].copy() if len(out) else pd.DataFrame(
        columns=["match_id", "date", "home_team", "away_team"]
    )
    slim.to_csv(gameday_path, index=False)

    payload = {
        **meta,
        "league_key": league_key,
        "fetched_at": _utc_now().isoformat(),
        "n_fixtures": int(len(out)),
        "csv_path": str(csv_path.as_posix()),
        "gameday_path": str(gameday_path.as_posix()),
        "next_kickoff": str(out["kickoff_utc"].iloc[0]) if len(out) and "kickoff_utc" in out.columns else None,
        "last_kickoff": str(out["kickoff_utc"].iloc[-1]) if len(out) and "kickoff_utc" in out.columns else None,
        "fixture_dates": sorted(out["date"].dropna().unique().tolist()) if len(out) else [],
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Wrote {} upcoming fixtures (source={}) -> {} | synced {}",
        len(out),
        meta.get("source"),
        csv_path,
        gameday_path,
    )
    return csv_path, meta_path, gameday_path


def load_fixtures_meta(data_dir: Path, league_key: str = "EPL") -> dict[str, Any] | None:
    _, meta_path, _ = fixtures_paths(data_dir, league_key)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def load_upcoming_fixtures(data_dir: Path, league_key: str = "EPL") -> pd.DataFrame:
    csv_path, _, _ = fixtures_paths(data_dir, league_key)
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


def fixtures_health(
    data_dir: Path,
    *,
    max_age_hours: float = 48.0,
    league_key: str = "EPL",
) -> dict[str, Any]:
    """Assess whether stored fixtures are usable for live gameday."""
    csv_path, meta_path, gameday_path = fixtures_paths(data_dir, league_key)
    meta = load_fixtures_meta(data_dir, league_key)
    df = load_upcoming_fixtures(data_dir, league_key) if csv_path.exists() else pd.DataFrame()
    now = _utc_now()
    issues: list[str] = []
    ok = True

    if meta is None or not csv_path.exists():
        return {
            "ok": False,
            "issues": ["No automatic fixtures file — run Update Data."],
            "meta": meta,
            "n": 0,
            "path": str(csv_path),
        }

    fetched = meta.get("fetched_at")
    age_h = None
    if fetched:
        try:
            ft = datetime.fromisoformat(str(fetched).replace("Z", "+00:00"))
            if ft.tzinfo is None:
                ft = ft.replace(tzinfo=timezone.utc)
            age_h = (now - ft).total_seconds() / 3600.0
            if age_h > max_age_hours:
                issues.append(f"Fixtures look stale (fetched {age_h:.1f}h ago). Re-run Update Data.")
                ok = False
        except Exception:  # noqa: BLE001
            issues.append("Could not parse fixtures fetched_at timestamp.")
            ok = False
    else:
        issues.append("Fixtures meta missing fetched_at.")
        ok = False

    if len(df) == 0:
        issues.append("Fixtures file is empty — no upcoming matches in window.")
        ok = False
    else:
        # Placeholder / example detection
        sample = " ".join(df["match_id"].astype(str).head(5).tolist()).lower()
        if "arsenal_liverpool" in sample and "20260816" in sample:
            issues.append("Placeholder/example fixtures detected — refusing to use them.")
            ok = False
        today = pd.Timestamp(now.date())
        max_date = pd.to_datetime(df["date"]).max()
        if pd.notna(max_date) and max_date < today - pd.Timedelta(days=1):
            issues.append(f"All stored fixtures are in the past (last date {max_date.date()}).")
            ok = False

    return {
        "ok": ok,
        "issues": issues,
        "meta": meta,
        "n": int(len(df)),
        "age_hours": age_h,
        "path": str(csv_path),
        "gameday_path": str(gameday_path),
        "meta_path": str(meta_path),
    }


def refresh_upcoming_fixtures(
    data_dir: Path,
    cfg: dict[str, Any] | None = None,
    *,
    days_ahead: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Pull upcoming EPL fixtures (Pulse → football-data → Understat) and persist.
    Raises RuntimeError if no real upcoming fixtures can be obtained.
    """
    cfg = cfg or {}
    fcfg = cfg.get("fixtures") or {}
    days = int(days_ahead if days_ahead is not None else fcfg.get("days_ahead", 21))
    mapper = DEFAULT_MAPPER

    errors: list[str] = []
    df = pd.DataFrame()
    meta: dict[str, Any] = {}

    try:
        df, meta = fetch_pulse_upcoming(days_ahead=days, mapper=mapper)
        logger.info("Pulse upcoming fixtures: {}", len(df))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pulse:{exc}")
        logger.warning("Pulse fixtures failed: {}", exc)

    if len(df) == 0:
        try:
            df, meta = fetch_football_data_upcoming(
                data_dir,
                league_code="E0",
                base_url=str(
                    (cfg.get("data") or {}).get("football_data", {}).get(
                        "base_url", "https://www.football-data.co.uk/mmz4281"
                    )
                ),
                mapper=mapper,
            )
            logger.info("football-data upcoming fixtures: {}", len(df))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"football_data:{exc}")
            logger.warning("football-data fixtures failed: {}", exc)

    if len(df) == 0:
        try:
            df, meta = fetch_understat_upcoming(data_dir, league="EPL", mapper=mapper)
            # filter to days_ahead
            if len(df):
                horizon = pd.Timestamp(_utc_now().date()) + pd.Timedelta(days=days)
                df = df[pd.to_datetime(df["date"]) <= horizon].copy()
            logger.info("Understat upcoming fixtures: {}", len(df))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"understat:{exc}")
            logger.warning("Understat fixtures failed: {}", exc)

    if len(df) == 0:
        raise RuntimeError(
            "Could not refresh upcoming EPL fixtures from Pulse, football-data, or Understat. "
            f"Errors: {errors}"
        )

    meta = {**meta, "fallback_errors": errors, "days_ahead": days, "league_key": "EPL"}
    save_fixtures(df, data_dir, meta, league_key="EPL")
    return df, meta


def refresh_upcoming_fixtures_for_league(
    data_dir: Path,
    league_key: str,
    cfg: dict[str, Any] | None = None,
    *,
    days_ahead: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Refresh upcoming fixtures for any registered league.

    EPL: Pulse → football-data E0 → Understat (unchanged).
    Other European leagues: football-data blank-score rows (mmz4281/{yy}/{code}.csv).
    MLS: blank-score rows from football-data.co.uk/new/USA.csv when available.
    """
    from origination.utils.league_registry import get_league

    cfg = cfg or {}
    info = get_league(league_key)
    key = info["key"]
    if key == "EPL":
        return refresh_upcoming_fixtures(data_dir, cfg, days_ahead=days_ahead)

    fcfg = cfg.get("fixtures") or {}
    days = int(days_ahead if days_ahead is not None else fcfg.get("days_ahead", 21))
    mapper = DEFAULT_MAPPER
    errors: list[str] = []
    df = pd.DataFrame()
    meta: dict[str, Any] = {}

    # Understat upcoming when available
    ust = info.get("understat")
    if ust:
        try:
            df, meta = fetch_understat_upcoming(data_dir, league=str(ust), mapper=mapper)
            if len(df):
                horizon = pd.Timestamp(_utc_now().date()) + pd.Timedelta(days=days)
                df = df[pd.to_datetime(df["date"]) <= horizon].copy()
            logger.info("Understat upcoming [{}]: {}", key, len(df))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"understat:{exc}")
            logger.warning("Understat fixtures [{}] failed: {}", key, exc)

    if len(df) == 0 and info.get("fd_code") and key != "MLS":
        try:
            df, meta = fetch_football_data_upcoming(
                data_dir,
                league_code=str(info["fd_code"]),
                base_url=str(
                    (cfg.get("data") or {}).get("football_data", {}).get(
                        "base_url", "https://www.football-data.co.uk/mmz4281"
                    )
                ),
                mapper=mapper,
            )
            if len(df):
                horizon = pd.Timestamp(_utc_now().date()) + pd.Timedelta(days=days)
                df = df[pd.to_datetime(df["date"]) <= horizon].copy()
            logger.info("football-data upcoming [{}]: {}", key, len(df))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"football_data:{exc}")
            logger.warning("football-data fixtures [{}] failed: {}", key, exc)

    if len(df) == 0 and key == "MLS":
        try:
            df, meta = fetch_mls_upcoming_from_usa_csv(data_dir, days_ahead=days, mapper=mapper)
            logger.info("MLS USA.csv upcoming: {}", len(df))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"mls_usa:{exc}")
            logger.warning("MLS fixtures failed: {}", exc)

    if len(df) == 0:
        # Fallback: Pinnacle matchup list as fixtures (live window)
        try:
            from origination.data_ingestion.pinnacle_odds import (
                PINNACLE_LEAGUE_IDS,
                build_pinnacle_ou25_table,
            )

            pin_id = int(
                (cfg.get("pinnacle") or {}).get("leagues", {}).get(
                    key,
                    info.get("pinnacle_league_id")
                    or PINNACLE_LEAGUE_IDS.get(key, 0),
                )
                or 0
            )
            if pin_id:
                pin, pin_meta = build_pinnacle_ou25_table(league_id=pin_id, mapper=mapper)
                if len(pin):
                    rows = []
                    today = pd.Timestamp(_utc_now().date())
                    horizon = today + pd.Timedelta(days=days)
                    for _, r in pin.iterrows():
                        dt = pd.to_datetime(r.get("date"), errors="coerce")
                        if pd.isna(dt):
                            continue
                        if getattr(dt, "tzinfo", None) is not None:
                            dt = dt.tz_localize(None)
                        dt = pd.Timestamp(dt).normalize()
                        if dt > horizon or dt < today - pd.Timedelta(days=1):
                            continue
                        ko = r.get("kickoff_utc")
                        rows.append(
                            {
                                "home_team_raw": r.get("home_team_raw") or r.get("home_team"),
                                "away_team_raw": r.get("away_team_raw") or r.get("away_team"),
                                "date": dt.date().isoformat(),
                                "kickoff_utc": str(ko) if ko is not None else dt.isoformat() + "+00:00",
                                "gameweek": pd.NA,
                                "status": "U",
                                "source": "pinnacle_matchups",
                            }
                        )
                    df = _finalize_frame(rows, mapper)
                    meta = {
                        **pin_meta,
                        "source": "pinnacle_matchups",
                        "n": len(df),
                    }
                    logger.info("Pinnacle matchup fixtures [{}]: {}", key, len(df))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pinnacle_fixtures:{exc}")
            logger.warning("Pinnacle fixture fallback [{}] failed: {}", key, exc)

    if len(df) == 0:
        raise RuntimeError(
            f"Could not refresh upcoming fixtures for {key}. Errors: {errors}"
        )

    meta = {**meta, "fallback_errors": errors, "days_ahead": days, "league_key": key}
    save_fixtures(df, data_dir, meta, league_key=key)
    return df, meta


def fetch_mls_upcoming_from_usa_csv(
    data_dir: Path,
    *,
    days_ahead: int = 21,
    mapper: TeamNameMapper | None = None,
    url: str = "https://www.football-data.co.uk/new/USA.csv",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Blank-score MLS rows from football-data new/USA.csv (1X2 closes only historically)."""
    mapper = mapper or DEFAULT_MAPPER
    raw_path = Path(data_dir) / "raw" / "football_data" / "USA" / "USA.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    raw_path.write_bytes(resp.content)
    df = pd.read_csv(raw_path)
    if "League" in df.columns:
        df = df[df["League"].astype(str).str.upper() == "MLS"].copy()
    # Upcoming = missing scores
    hg = df["HG"] if "HG" in df.columns else df.get("FTHG")
    blank = hg.isna() | (hg.astype(str).str.strip() == "")
    upcoming = df.loc[blank].copy()
    if upcoming.empty:
        return _finalize_frame([], mapper), {"source": "football_data_usa", "url": url, "n": 0}
    home_col = "Home" if "Home" in upcoming.columns else "HomeTeam"
    away_col = "Away" if "Away" in upcoming.columns else "AwayTeam"
    upcoming["Date"] = pd.to_datetime(upcoming["Date"], dayfirst=True, errors="coerce")
    upcoming = upcoming.dropna(subset=["Date", home_col, away_col])
    horizon = pd.Timestamp(_utc_now().date()) + pd.Timedelta(days=days_ahead)
    upcoming = upcoming[upcoming["Date"] <= horizon]
    rows = []
    for _, r in upcoming.iterrows():
        rows.append(
            {
                "home_team_raw": r[home_col],
                "away_team_raw": r[away_col],
                "date": r["Date"].date().isoformat(),
                "kickoff_utc": pd.Timestamp(r["Date"]).tz_localize("UTC").isoformat(),
                "gameweek": pd.NA,
                "status": "U",
                "source": "football_data_usa",
            }
        )
    out = _finalize_frame(rows, mapper)
    return out, {"source": "football_data_usa", "url": url, "n": len(out)}


def ingest_upcoming_fixtures_from_config(cfg: dict[str, Any], data_dir: Path) -> pd.DataFrame:
    """Config entry-point used by update_data."""
    fcfg = cfg.get("fixtures") or {}
    if fcfg.get("enabled", True) is False:
        logger.info("Upcoming fixtures refresh disabled in config")
        return pd.DataFrame()
    df, _meta = refresh_upcoming_fixtures(data_dir, cfg)
    return df
