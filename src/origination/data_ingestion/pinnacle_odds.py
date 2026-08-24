"""
Pinnacle Guest API — EPL Over/Under 2.5 odds for gameday.

Endpoints (public guest API):
  https://guest.api.arcadia.pinnacle.com/0.1/leagues/1980/matchups
  https://guest.api.arcadia.pinnacle.com/0.1/leagues/1980/markets/straight

League 1980 = England Premier League.

Writes:
  data/interim/pinnacle_ou25_EPL.csv
  data/interim/pinnacle_ou25_EPL.meta.json
  data/gameday/odds_pinnacle.csv  (match_id,ref_over25,ref_under25,…)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from origination.utils.odds import american_to_decimal
from origination.utils.team_names import DEFAULT_MAPPER, TeamNameMapper

PINNACLE_LEAGUE_ID = 1980
PINNACLE_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
PINNACLE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.pinnacle.com/",
    "Origin": "https://www.pinnacle.com",
}

ODDS_CSV_NAME = "pinnacle_ou25_EPL.csv"
ODDS_META_NAME = "pinnacle_ou25_EPL.meta.json"

# Pinnacle soccer league IDs (guest API /sports/29/leagues)
PINNACLE_LEAGUE_IDS = {
    "EPL": 1980,
    "Championship": 1977,
    "Bundesliga": 1842,
    "SerieA": 2436,
    "LaLiga": 2196,
    "MLS": 2663,
    "Ligue1": 2036,
    "Eredivisie": 1928,
    "PrimeiraLiga": 2386,
    "Belgium": 1817,
    "Scotland": 2421,  # Premiership (was 1975 — HTTP dead / wrong)
    "Turkey": 2592,  # Super League (1843 is German 2.Bundesliga — do not use)
    "Austria": 1838,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _get_json(url: str) -> Any:
    resp = requests.get(url, headers=PINNACLE_HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_pinnacle_matchups(league_id: int = PINNACLE_LEAGUE_ID) -> list[dict[str, Any]]:
    url = f"{PINNACLE_BASE}/leagues/{league_id}/matchups"
    data = _get_json(url)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Pinnacle matchups payload: {type(data)}")
    return data


def fetch_pinnacle_markets(league_id: int = PINNACLE_LEAGUE_ID) -> list[dict[str, Any]]:
    url = f"{PINNACLE_BASE}/leagues/{league_id}/markets/straight"
    data = _get_json(url)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Pinnacle markets payload: {type(data)}")
    return data


def _parse_matchups(
    matchups: list[dict[str, Any]],
    mapper: TeamNameMapper,
) -> pd.DataFrame:
    rows = []
    for m in matchups:
        if m.get("type") != "matchup":
            continue
        if m.get("parentId") is not None:
            continue
        parts = m.get("participants") or []
        home_raw = next((p.get("name") for p in parts if p.get("alignment") == "home"), None)
        away_raw = next((p.get("name") for p in parts if p.get("alignment") == "away"), None)
        if not home_raw or not away_raw:
            continue
        start = m.get("startTime")
        kickoff = pd.to_datetime(start, utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue
        rows.append(
            {
                "pin_matchup_id": int(m["id"]),
                "kickoff_utc": kickoff,
                "date": kickoff.tz_convert("UTC").normalize(),
                "home_team_raw": home_raw,
                "away_team_raw": away_raw,
                "home_team": mapper.canonicalize(home_raw),
                "away_team": mapper.canonicalize(away_raw),
                "is_live": bool(m.get("isLive")),
            }
        )
    return pd.DataFrame(rows)


def _extract_ou25(markets: list[dict[str, Any]]) -> pd.DataFrame:
    """Extract full-game Over/Under 2.5 American prices per matchupId."""
    rows = []
    for mk in markets:
        if mk.get("type") != "total":
            continue
        period = mk.get("period")
        if period is None or int(period) != 0:
            continue
        key = str(mk.get("key") or "")
        # Prefer explicit 2.5 market
        if key != "s;0;ou;2.5" and not key.endswith(";ou;2.5"):
            continue
        status = mk.get("status")
        if status in ("closed", "suspended"):
            continue
        mid = mk.get("matchupId")
        if mid is None:
            continue
        over_am = under_am = None
        for pr in mk.get("prices") or []:
            des = str(pr.get("designation") or "").lower()
            pts = pr.get("points")
            if pts is not None and abs(float(pts) - 2.5) > 1e-9:
                continue
            price = pr.get("price")
            if price is None:
                continue
            if des == "over":
                over_am = float(price)
            elif des == "under":
                under_am = float(price)
        if over_am is None or under_am is None:
            continue
        over_dec = american_to_decimal(over_am)
        under_dec = american_to_decimal(under_am)
        if over_dec is None or under_dec is None:
            continue
        rows.append(
            {
                "pin_matchup_id": int(mid),
                "pin_over25_american": int(over_am),
                "pin_under25_american": int(under_am),
                "pin_over25": round(float(over_dec), 4),
                "pin_under25": round(float(under_dec), 4),
                "pin_line": 2.5,
                "pin_market_key": key,
            }
        )
    return pd.DataFrame(rows)


def _extract_moneyline(markets: list[dict[str, Any]]) -> pd.DataFrame:
    """Extract full-game 1X2 (moneyline) American prices per matchupId."""
    rows = []
    for mk in markets:
        if mk.get("type") != "moneyline":
            continue
        period = mk.get("period")
        if period is None or int(period) != 0:
            continue
        key = str(mk.get("key") or "")
        if key not in ("s;0;m",) and not key.endswith(";m"):
            # still allow main ML if period 0
            if ";m" not in key:
                continue
        status = mk.get("status")
        if status in ("closed", "suspended"):
            continue
        mid = mk.get("matchupId")
        if mid is None:
            continue
        home_am = draw_am = away_am = None
        for pr in mk.get("prices") or []:
            des = str(pr.get("designation") or "").lower()
            price = pr.get("price")
            if price is None:
                continue
            if des == "home":
                home_am = float(price)
            elif des == "draw":
                draw_am = float(price)
            elif des == "away":
                away_am = float(price)
        if home_am is None or away_am is None:
            continue
        home_dec = american_to_decimal(home_am)
        away_dec = american_to_decimal(away_am)
        draw_dec = american_to_decimal(draw_am) if draw_am is not None else None
        if home_dec is None or away_dec is None:
            continue
        rows.append(
            {
                "pin_matchup_id": int(mid),
                "pin_h": round(float(home_dec), 4),
                "pin_d": round(float(draw_dec), 4) if draw_dec is not None else pd.NA,
                "pin_a": round(float(away_dec), 4),
                "pin_h_american": int(home_am),
                "pin_d_american": int(draw_am) if draw_am is not None else pd.NA,
                "pin_a_american": int(away_am),
            }
        )
    return pd.DataFrame(rows)


def _extract_team_totals_main(markets: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Extract full-game team-total main lines for home and away.

    Main line per side = open period-0 team_total whose Over/Under American
    prices are closest to balanced (min |over_am + under_am|). Prefers common
    soccer lines 0.5 / 1.5 / 2.5 when tied.
    """
    by_mid: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for mk in markets:
        if mk.get("type") != "team_total":
            continue
        period = mk.get("period")
        if period is None or int(period) != 0:
            continue
        status = mk.get("status")
        if status in ("closed", "suspended"):
            continue
        mid = mk.get("matchupId")
        if mid is None:
            continue
        side = str(mk.get("side") or "").lower()
        if side not in ("home", "away"):
            # some payloads only encode side in the key: s;0;tt;1.5;home
            key = str(mk.get("key") or "")
            if key.endswith(";home"):
                side = "home"
            elif key.endswith(";away"):
                side = "away"
            else:
                continue
        over_am = under_am = None
        line = None
        for pr in mk.get("prices") or []:
            des = str(pr.get("designation") or "").lower()
            price = pr.get("price")
            pts = pr.get("points")
            if price is None or pts is None:
                continue
            if des == "over":
                over_am = float(price)
                line = float(pts)
            elif des == "under":
                under_am = float(price)
                if line is None:
                    line = float(pts)
        if over_am is None or under_am is None or line is None:
            continue
        by_mid.setdefault(int(mid), {}).setdefault(side, []).append(
            {
                "line": float(line),
                "over_am": over_am,
                "under_am": under_am,
                "balance": abs(over_am + under_am),
                "common": 0 if float(line) in (0.5, 1.5, 2.5) else 1,
                "key": str(mk.get("key") or ""),
            }
        )

    rows: list[dict[str, Any]] = []
    for mid, sides in by_mid.items():
        row: dict[str, Any] = {"pin_matchup_id": mid}
        for side, cands in sides.items():
            cands.sort(key=lambda c: (c["balance"], c["common"], abs(c["line"] - 1.5)))
            best = cands[0]
            over_dec = american_to_decimal(best["over_am"])
            under_dec = american_to_decimal(best["under_am"])
            if over_dec is None or under_dec is None:
                continue
            prefix = "pin_tt_home" if side == "home" else "pin_tt_away"
            row[f"{prefix}_line"] = round(float(best["line"]), 3)
            row[f"{prefix}_over"] = round(float(over_dec), 4)
            row[f"{prefix}_under"] = round(float(under_dec), 4)
            row[f"{prefix}_over_american"] = int(best["over_am"])
            row[f"{prefix}_under_american"] = int(best["under_am"])
            row[f"{prefix}_market_key"] = best["key"]
        if len(row) > 1:
            rows.append(row)
    return pd.DataFrame(rows)


def _extract_ah_main(markets: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Extract full-game Asian Handicap (spread) main line per matchup.

    Main line = open period-0 spread whose home/away American prices are
    closest to balanced (min |home_am + away_am|), matching typical Pin main.
    Home points = football-data AHh convention (negative ⇒ home favorite).
    """
    by_mid: dict[int, list[dict[str, Any]]] = {}
    for mk in markets:
        if mk.get("type") != "spread":
            continue
        period = mk.get("period")
        if period is None or int(period) != 0:
            continue
        status = mk.get("status")
        if status in ("closed", "suspended"):
            continue
        mid = mk.get("matchupId")
        if mid is None:
            continue
        home_am = away_am = None
        home_pts = None
        for pr in mk.get("prices") or []:
            des = str(pr.get("designation") or "").lower()
            price = pr.get("price")
            pts = pr.get("points")
            if price is None or pts is None:
                continue
            if des == "home":
                home_am = float(price)
                home_pts = float(pts)
            elif des == "away":
                away_am = float(price)
        if home_am is None or away_am is None or home_pts is None:
            continue
        by_mid.setdefault(int(mid), []).append(
            {
                "home_am": home_am,
                "away_am": away_am,
                "home_pts": home_pts,
                "key": str(mk.get("key") or ""),
                "balance": abs(home_am + away_am),
            }
        )

    rows = []
    for mid, cands in by_mid.items():
        cands.sort(key=lambda c: (c["balance"], abs(c["home_pts"])))
        best = cands[0]
        home_dec = american_to_decimal(best["home_am"])
        away_dec = american_to_decimal(best["away_am"])
        if home_dec is None or away_dec is None:
            continue
        rows.append(
            {
                "pin_matchup_id": mid,
                "pin_ah_line": round(float(best["home_pts"]), 3),
                "pin_ahh": round(float(home_dec), 4),
                "pin_aha": round(float(away_dec), 4),
                "pin_ahh_american": int(best["home_am"]),
                "pin_aha_american": int(best["away_am"]),
                "pin_ah_market_key": best["key"],
            }
        )
    return pd.DataFrame(rows)


def build_pinnacle_ou25_table(
    *,
    league_id: int = PINNACLE_LEAGUE_ID,
    mapper: TeamNameMapper | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch matchups + markets and join into a Pinnacle OU/ML/AH table."""
    mapper = mapper or DEFAULT_MAPPER
    matchups = fetch_pinnacle_matchups(league_id)
    markets = fetch_pinnacle_markets(league_id)
    mdf = _parse_matchups(matchups, mapper)
    odf = _extract_ou25(markets)
    mldf = _extract_moneyline(markets)
    ahdf = _extract_ah_main(markets)
    ttdf = _extract_team_totals_main(markets)
    if len(mdf) == 0:
        return mdf, {
            "source": "pinnacle_guest",
            "league_id": league_id,
            "error": "no_matchups",
            "n_matchups_raw": len(matchups),
        }
    out = mdf.copy()
    if len(odf):
        out = out.merge(odf, on="pin_matchup_id", how="left")
    else:
        logger.warning("Pinnacle: no OU 2.5 markets found for league {}", league_id)
        for c in (
            "pin_over25",
            "pin_under25",
            "pin_over25_american",
            "pin_under25_american",
            "pin_line",
            "pin_market_key",
        ):
            out[c] = pd.NA
    if len(mldf):
        out = out.merge(mldf, on="pin_matchup_id", how="left")
    else:
        logger.warning("Pinnacle: no moneyline markets found for league {}", league_id)
        for c in ("pin_h", "pin_d", "pin_a", "pin_h_american", "pin_d_american", "pin_a_american"):
            out[c] = pd.NA
    if len(ahdf):
        out = out.merge(ahdf, on="pin_matchup_id", how="left")
    else:
        logger.warning("Pinnacle: no AH/spread markets found for league {}", league_id)
        for c in (
            "pin_ah_line",
            "pin_ahh",
            "pin_aha",
            "pin_ahh_american",
            "pin_aha_american",
            "pin_ah_market_key",
        ):
            out[c] = pd.NA
    if len(ttdf):
        out = out.merge(ttdf, on="pin_matchup_id", how="left")
    else:
        logger.warning("Pinnacle: no team-total markets found for league {}", league_id)
        for c in (
            "pin_tt_home_line",
            "pin_tt_home_over",
            "pin_tt_home_under",
            "pin_tt_home_over_american",
            "pin_tt_home_under_american",
            "pin_tt_home_market_key",
            "pin_tt_away_line",
            "pin_tt_away_over",
            "pin_tt_away_under",
            "pin_tt_away_over_american",
            "pin_tt_away_under_american",
            "pin_tt_away_market_key",
        ):
            out[c] = pd.NA

    meta = {
        "source": "pinnacle_guest",
        "league_id": league_id,
        "n_matchups": int(len(mdf)),
        "n_with_ou25": int(out["pin_over25"].notna().sum()) if "pin_over25" in out.columns else 0,
        "n_with_1x2": int(out["pin_h"].notna().sum()) if "pin_h" in out.columns else 0,
        "n_with_ah": int(out["pin_ahh"].notna().sum()) if "pin_ahh" in out.columns else 0,
        "n_with_tt": int(out["pin_tt_home_line"].notna().sum())
        if "pin_tt_home_line" in out.columns
        else 0,
        "matchup_url": f"{PINNACLE_BASE}/leagues/{league_id}/matchups",
        "markets_url": f"{PINNACLE_BASE}/leagues/{league_id}/markets/straight",
    }
    return out, meta


def match_pinnacle_to_fixtures(
    pin: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Match Pinnacle rows to internal fixtures by canonical home/away + date.

    Returns fixtures left-joined with Pinnacle odds columns, plus match report.
    """
    if fixtures is None or len(fixtures) == 0:
        return pd.DataFrame(), {"matched": 0, "unmatched_fixtures": [], "unmatched_pinnacle": []}

    fx = fixtures.copy()
    fx["date"] = pd.to_datetime(fx["date"]).dt.normalize()
    pin = pin.copy()
    if len(pin):
        pin["date"] = pd.to_datetime(pin["date"]).dt.tz_localize(None).dt.normalize()

    # Primary key: date + home + away
    merged = fx.merge(
        pin[
            [
                c
                for c in pin.columns
                if c
                in {
                    "date",
                    "home_team",
                    "away_team",
                    "pin_matchup_id",
                    "kickoff_utc",
                    "pin_over25",
                    "pin_under25",
                    "pin_over25_american",
                    "pin_under25_american",
                    "pin_line",
                    "pin_h",
                    "pin_d",
                    "pin_a",
                    "pin_h_american",
                    "pin_d_american",
                    "pin_a_american",
                    "pin_ah_line",
                    "pin_ahh",
                    "pin_aha",
                    "pin_ahh_american",
                    "pin_aha_american",
                    "pin_ah_market_key",
                    "pin_tt_home_line",
                    "pin_tt_home_over",
                    "pin_tt_home_under",
                    "pin_tt_home_over_american",
                    "pin_tt_home_under_american",
                    "pin_tt_home_market_key",
                    "pin_tt_away_line",
                    "pin_tt_away_over",
                    "pin_tt_away_under",
                    "pin_tt_away_over_american",
                    "pin_tt_away_under_american",
                    "pin_tt_away_market_key",
                    "home_team_raw",
                    "away_team_raw",
                }
            ]
        ],
        on=["date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_pin"),
    )

    matched_mask = (
        merged["pin_over25"].notna()
        if "pin_over25" in merged.columns
        else pd.Series(False, index=merged.index)
    )
    if "pin_h" in merged.columns:
        matched_mask = matched_mask | merged["pin_h"].notna()
    if "pin_ahh" in merged.columns:
        matched_mask = matched_mask | merged["pin_ahh"].notna()
    if "pin_tt_home_line" in merged.columns:
        matched_mask = matched_mask | merged["pin_tt_home_line"].notna()
    if "pin_tt_away_line" in merged.columns:
        matched_mask = matched_mask | merged["pin_tt_away_line"].notna()
    unmatched_fx = merged.loc[~matched_mask, ["match_id", "date", "home_team", "away_team"]].copy()

    # Pinnacle rows not in fixtures
    if len(pin):
        pin_keys = set(zip(pin["date"].astype(str), pin["home_team"], pin["away_team"], strict=False))
        fx_keys = set(zip(fx["date"].astype(str), fx["home_team"], fx["away_team"], strict=False))
        orphan = pin_keys - fx_keys
        unmatched_pin = [
            {"date": d, "home_team": h, "away_team": a} for d, h, a in sorted(orphan)
        ]
    else:
        unmatched_pin = []

    report = {
        "n_fixtures": int(len(fx)),
        "n_pinnacle": int(len(pin)),
        "matched": int(matched_mask.sum()),
        "unmatched_fixtures": unmatched_fx.to_dict(orient="records"),
        "unmatched_pinnacle": unmatched_pin,
    }
    for u in report["unmatched_fixtures"]:
        logger.warning(
            "Pinnacle unmatched fixture: {} {} vs {}",
            u.get("date"),
            u.get("home_team"),
            u.get("away_team"),
        )
    for u in unmatched_pin:
        logger.warning(
            "Pinnacle matchup not in fixtures window: {} {} vs {}",
            u.get("date"),
            u.get("home_team"),
            u.get("away_team"),
        )
    logger.info(
        "Pinnacle match: {}/{} fixtures with OU 2.5 ({} pin matchups)",
        report["matched"],
        report["n_fixtures"],
        report["n_pinnacle"],
    )
    return merged, report


def pinnacle_paths(data_dir: Path, league_key: str = "EPL") -> tuple[Path, Path, Path]:
    """Artifact paths. EPL keeps legacy filenames for backward compatibility."""
    interim = Path(data_dir) / "interim"
    gameday = Path(data_dir) / "gameday"
    key = (league_key or "EPL").strip()
    if key in ("EPL", "E0", ""):
        csv_name, meta_name, gd_name = ODDS_CSV_NAME, ODDS_META_NAME, "odds_pinnacle.csv"
    else:
        csv_name = f"pinnacle_ou25_{key}.csv"
        meta_name = f"pinnacle_ou25_{key}.meta.json"
        gd_name = f"odds_pinnacle_{key}.csv"
    return interim / csv_name, interim / meta_name, gameday / gd_name


def save_pinnacle_odds(
    matched: pd.DataFrame,
    data_dir: Path,
    meta: dict[str, Any],
    *,
    league_key: str = "EPL",
) -> tuple[Path, Path, Path]:
    csv_path, meta_path, gameday_path = pinnacle_paths(data_dir, league_key)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    gameday_path.parent.mkdir(parents=True, exist_ok=True)

    out = matched.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(csv_path, index=False)

    slim_cols = [
        "match_id",
        "ref_over25",
        "ref_under25",
        "pin_over25",
        "pin_under25",
        "pin_over25_american",
        "pin_under25_american",
        "pin_h",
        "pin_d",
        "pin_a",
        "pin_h_american",
        "pin_d_american",
        "pin_a_american",
        "pin_ah_line",
        "pin_ahh",
        "pin_aha",
        "pin_ahh_american",
        "pin_aha_american",
        "pin_tt_home_line",
        "pin_tt_home_over",
        "pin_tt_home_under",
        "pin_tt_home_over_american",
        "pin_tt_home_under_american",
        "pin_tt_away_line",
        "pin_tt_away_over",
        "pin_tt_away_under",
        "pin_tt_away_over_american",
        "pin_tt_away_under_american",
        "pin_matchup_id",
    ]
    if len(out) and "match_id" in out.columns:
        slim = out.copy()
        if "pin_over25" in slim.columns:
            slim["ref_over25"] = slim["pin_over25"]
            slim["ref_under25"] = slim["pin_under25"]
        keep = [c for c in slim_cols if c in slim.columns]
        slim = slim[keep].dropna(subset=["match_id"])
        # Keep rows with OU and/or 1X2 and/or AH
        has_ou = (
            slim["ref_over25"].notna() & slim["ref_under25"].notna()
            if "ref_over25" in slim.columns
            else pd.Series(False, index=slim.index)
        )
        has_ml = slim["pin_h"].notna() if "pin_h" in slim.columns else pd.Series(False, index=slim.index)
        has_ah = slim["pin_ahh"].notna() if "pin_ahh" in slim.columns else pd.Series(False, index=slim.index)
        has_tt = (
            slim["pin_tt_home_line"].notna() | slim["pin_tt_away_line"].notna()
            if "pin_tt_home_line" in slim.columns
            else pd.Series(False, index=slim.index)
        )
        slim = slim[has_ou | has_ml | has_ah | has_tt]
    else:
        slim = pd.DataFrame(columns=slim_cols)
    slim.to_csv(gameday_path, index=False)

    payload = {
        **meta,
        "league_key": league_key,
        "fetched_at": _utc_now().isoformat(),
        "csv_path": str(csv_path.as_posix()),
        "gameday_path": str(gameday_path.as_posix()),
        "n_rows": int(len(out)),
        "n_ref_prices": int(len(slim)),
    }
    meta_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Wrote Pinnacle OU2.5 [{}] for {} fixtures ({} with prices) -> {}",
        league_key,
        len(out),
        len(slim),
        csv_path,
    )
    return csv_path, meta_path, gameday_path


def load_pinnacle_odds(data_dir: Path, league_key: str = "EPL") -> pd.DataFrame:
    csv_path, _, gameday_path = pinnacle_paths(data_dir, league_key)
    if gameday_path.exists():
        return pd.read_csv(gameday_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "pin_over25" in df.columns and "ref_over25" not in df.columns:
            df = df.rename(columns={"pin_over25": "ref_over25", "pin_under25": "ref_under25"})
        return df
    return pd.DataFrame()


def load_pinnacle_meta(data_dir: Path, league_key: str = "EPL") -> dict[str, Any] | None:
    _, meta_path, _ = pinnacle_paths(data_dir, league_key)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def refresh_pinnacle_odds(
    data_dir: Path,
    fixtures: pd.DataFrame | None = None,
    cfg: dict[str, Any] | None = None,
    *,
    league_key: str = "EPL",
    league_id: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Pull Pinnacle OU 2.5, match to upcoming fixtures, persist artifacts.

    If fixtures is None, loads data/interim/fixtures_upcoming_{league}.csv.
    """
    cfg = cfg or {}
    pcfg = cfg.get("pinnacle") or {}
    key = league_key or "EPL"
    if league_id is None:
        league_id = int(PINNACLE_LEAGUE_IDS.get(key, pcfg.get("league_id", PINNACLE_LEAGUE_ID)))
    leagues_cfg = pcfg.get("leagues") or {}
    if key in leagues_cfg:
        league_id = int(leagues_cfg[key])

    if fixtures is None:
        fx_name = "fixtures_upcoming_EPL.csv" if key == "EPL" else f"fixtures_upcoming_{key}.csv"
        fx_path = Path(data_dir) / "interim" / fx_name
        if not fx_path.exists():
            raise RuntimeError(
                f"No fixtures at {fx_path}; refresh fixtures before Pinnacle odds."
            )
        fixtures = pd.read_csv(fx_path, parse_dates=["date"])

    try:
        pin, meta = build_pinnacle_ou25_table(league_id=league_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Pinnacle fetch failed: {}", exc)
        raise RuntimeError(f"Pinnacle odds fetch failed: {exc}") from exc

    matched, report = match_pinnacle_to_fixtures(pin, fixtures)
    meta = {**meta, "match_report": report, "league_key": key}
    save_pinnacle_odds(matched, data_dir, meta, league_key=key)
    return matched, meta


def ingest_pinnacle_odds_from_config(cfg: dict[str, Any], data_dir: Path) -> pd.DataFrame:
    pcfg = cfg.get("pinnacle") or {}
    if pcfg.get("enabled", True) is False:
        logger.info("Pinnacle odds refresh disabled in config")
        return pd.DataFrame()
    league_key = str(pcfg.get("league_key", "EPL"))
    matched, _meta = refresh_pinnacle_odds(data_dir, cfg=cfg, league_key=league_key)
    return matched
