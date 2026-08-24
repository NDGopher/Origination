"""
football-data.co.uk ingestion for English Premier League (E0) and Big 5.

Downloads seasonal CSVs, normalizes columns, parses dates, and extracts
closing odds preferred for CLV measurement.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from origination.utils.seeding import season_label
from origination.utils.team_names import DEFAULT_MAPPER, TeamNameMapper

# Columns we always try to keep when present
RESULT_COLS = [
    "Div",
    "Date",
    "Time",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
    "Referee",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]

# Odds column families (open/close variants)
ODDS_PREFIXES = [
    "B365",
    "BW",
    "IW",
    "PS",
    "WH",
    "VC",
    "Avg",
    "Max",
]
ODDS_SUFFIXES_1X2 = ["H", "D", "A"]
ODDS_CLOSE_MARKERS = ["C"]  # B365CH, PSCH, AvgCH, etc.


def _season_years(start: int, end: int | None) -> list[int]:
    if end is None:
        # Through current football season start year
        today = datetime.utcnow()
        end = today.year if today.month >= 8 else today.year - 1
    return list(range(start, end + 1))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _download_csv(url: str, dest: Path, *, expected_code: str | None = None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60, allow_redirects=True)
    if resp.status_code in (300, 404):
        logger.warning("Not available ({}): {}", resp.status_code, url)
        return False
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    text_head = resp.content[:200].decode("utf-8", errors="ignore").lower()
    if "html" in content_type.lower() or text_head.strip().startswith("<!"):
        logger.warning("Got HTML instead of CSV for {}", url)
        return False
    final = (resp.url or url).split("?")[0]
    if expected_code:
        want = f"/{expected_code}.csv"
        if want.lower() not in final.lower():
            logger.warning(
                "Rejected football-data {} — redirected to {} (wrong league file)",
                url,
                final,
            )
            return False
    dest.write_bytes(resp.content)
    logger.info("Downloaded {} -> {}", url, dest)
    return True


def _parse_date(series: pd.Series) -> pd.Series:
    """football-data dates are mixed DD/MM/YY and DD/MM/YYYY."""
    parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    return parsed


def _normalize_frame(
    df: pd.DataFrame,
    league_code: str,
    season_start: int,
    mapper: TeamNameMapper,
) -> pd.DataFrame:
    df = df.copy()
    # Drop fully empty trailing columns / rows
    df = df.dropna(how="all")
    if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        raise ValueError("Missing HomeTeam/AwayTeam columns")

    df["Date"] = _parse_date(df["Date"])
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    if "Div" in df.columns:
        divs = df["Div"].dropna().astype(str).str.upper().unique().tolist()
        if divs and any(d != str(league_code).upper() for d in divs):
            raise ValueError(
                f"football-data {league_code} season {season_start} has Div={divs} "
                f"(wrong league file — not loaded)"
            )
    df["home_team"] = mapper.map_series(df["HomeTeam"])
    df["away_team"] = mapper.map_series(df["AwayTeam"])
    df["league"] = league_code
    df["season"] = season_start
    df["match_id"] = (
        df["Date"].dt.strftime("%Y%m%d")
        + "_"
        + df["home_team"].str.replace(" ", "", regex=False)
        + "_"
        + df["away_team"].str.replace(" ", "", regex=False)
    )

    # Standardize result columns
    rename = {
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "ftr",
        "HTHG": "ht_home_goals",
        "HTAG": "ht_away_goals",
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_sot",
        "AST": "away_sot",
        "HC": "home_corners",
        "AC": "away_corners",
        "HF": "home_fouls",
        "AF": "away_fouls",
        "HY": "home_yellow",
        "AY": "away_yellow",
        "HR": "home_red",
        "AR": "away_red",
        "Referee": "referee",
        "HxG": "home_xg_fd",
        "AxG": "away_xg_fd",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for col in ["home_goals", "away_goals", "home_xg_fd", "away_xg_fd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coerce all odds-like columns to numeric (handles object/string dtypes)
    for c in df.columns:
        if any(c.startswith(p) for p in ODDS_PREFIXES) or c.endswith(("H", "D", "A")):
            if c in ("home_team", "away_team", "ftr", "HTR", "Date", "Time", "div"):
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("Date").reset_index(drop=True)


def select_closing_1x2(
    df: pd.DataFrame,
    preferred_books: list[str],
) -> pd.DataFrame:
    """
    Attach closing (or best available) 1X2 odds columns:
    close_h, close_d, close_a, close_book.
    """
    out = df.copy()
    close_h: list[float] = []
    close_d: list[float] = []
    close_a: list[float] = []
    close_book: list[str] = []

    # Candidate column triples per book family
    # PSC -> PSCH, PSCD, PSCA; B365C -> B365CH,...; AvgC -> AvgCH,...
    # Also non-close: B365H/D/A, PSH/PSD/PSA, AvgH/AvgD/AvgA
    def triples_for(book: str) -> tuple[str, str, str]:
        if book.endswith("C") and len(book) > 1:
            # Closing family: B365C -> B365CH, or PSC -> PSCH
            base = book  # already includes C
            return f"{base}H", f"{base}D", f"{base}A"
        return f"{book}H", f"{book}D", f"{book}A"

    for _, row in out.iterrows():
        chosen = None
        for book in preferred_books:
            h_col, d_col, a_col = triples_for(book)
            if h_col in out.columns and d_col in out.columns and a_col in out.columns:
                h, d, a = row.get(h_col), row.get(d_col), row.get(a_col)
                if pd.notna(h) and pd.notna(d) and pd.notna(a) and min(h, d, a) > 1.0:
                    chosen = (float(h), float(d), float(a), book)
                    break
        if chosen is None:
            close_h.append(float("nan"))
            close_d.append(float("nan"))
            close_a.append(float("nan"))
            close_book.append("")
        else:
            close_h.append(chosen[0])
            close_d.append(chosen[1])
            close_a.append(chosen[2])
            close_book.append(chosen[3])

    out["close_h"] = close_h
    out["close_d"] = close_d
    out["close_a"] = close_a
    out["close_book"] = close_book
    return out


def select_closing_ou25(df: pd.DataFrame) -> pd.DataFrame:
    """Attach O/U 2.5 closing odds when available (AvgC>2.5 / AvgC<2.5 etc.)."""
    out = df.copy()
    # football-data column naming varies; try common patterns
    over_candidates = [
        "AvgC>2.5",
        "B365C>2.5",
        "PC>2.5",
        "Avg>2.5",
        "B365>2.5",
        "P>2.5",
    ]
    under_candidates = [
        "AvgC<2.5",
        "B365C<2.5",
        "PC<2.5",
        "Avg<2.5",
        "B365<2.5",
        "P<2.5",
    ]
    # Also Max / BbAv patterns from older seasons
    over_candidates += ["BbAv>2.5", "BbMx>2.5", "Max>2.5"]
    under_candidates += ["BbAv<2.5", "BbMx<2.5", "Max<2.5"]

    def first_valid(row: pd.Series, cols: list[str]) -> float:
        for c in cols:
            if c in out.columns and pd.notna(row.get(c)) and float(row[c]) > 1.0:
                return float(row[c])
        return float("nan")

    out["close_over25"] = out.apply(lambda r: first_valid(r, over_candidates), axis=1)
    out["close_under25"] = out.apply(lambda r: first_valid(r, under_candidates), axis=1)
    return out


def select_ah_main(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Asian handicap main line + odds when present.

    Prefer Pinnacle closing (PAH / PAHH / PAHA) when available for CLV fidelity.
    """
    out = df.copy()
    # Prefer Pinnacle line + closes, then B365 / Avg / Betbrain
    line_cols = ["PAH", "AHh", "BbAHh"]
    home_odds_cols = ["PAHH", "B365AHH", "AvgAHH", "BbAvAHH"]
    away_odds_cols = ["PAHA", "B365AHA", "AvgAHA", "BbAvAHA"]

    def pick(row: pd.Series, cols: list[str]) -> float:
        for c in cols:
            if c in out.columns and pd.notna(row.get(c)):
                try:
                    return float(row[c])
                except (TypeError, ValueError):
                    continue
        return float("nan")

    out["ah_line"] = out.apply(lambda r: pick(r, line_cols), axis=1)
    out["close_ahh"] = out.apply(lambda r: pick(r, home_odds_cols), axis=1)
    out["close_aha"] = out.apply(lambda r: pick(r, away_odds_cols), axis=1)
    # Explicit Pin aliases for audits
    if "PAHH" in out.columns:
        out["pin_close_ahh"] = pd.to_numeric(out["PAHH"], errors="coerce")
    if "PAHA" in out.columns:
        out["pin_close_aha"] = pd.to_numeric(out["PAHA"], errors="coerce")
    if "PAH" in out.columns:
        out["pin_close_ah_line"] = pd.to_numeric(out["PAH"], errors="coerce")
    return out


class FootballDataIngester:
    """Download and normalize football-data.co.uk seasonal CSVs."""

    def __init__(
        self,
        raw_dir: Path,
        base_url: str = "https://www.football-data.co.uk/mmz4281",
        mapper: TeamNameMapper | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.base_url = base_url.rstrip("/")
        self.mapper = mapper or DEFAULT_MAPPER

    def season_path(self, league_code: str, season_start: int) -> Path:
        label = season_label(season_start)
        return self.raw_dir / league_code / f"{label}.csv"

    def fetch_season(
        self,
        league_code: str,
        season_start: int,
        *,
        force: bool = False,
    ) -> Path | None:
        dest = self.season_path(league_code, season_start)
        if dest.exists() and not force:
            logger.debug("Using cached {}", dest)
            return dest
        label = season_label(season_start)
        url = f"{self.base_url}/{label}/{league_code}.csv"
        ok = _download_csv(url, dest, expected_code=league_code)
        if not ok:
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
            return None
        try:
            peek = pd.read_csv(dest, encoding="utf-8-sig", nrows=12)
            if "Div" in peek.columns:
                divs = peek["Div"].dropna().astype(str).str.upper().unique().tolist()
                if divs and any(d != str(league_code).upper() for d in divs):
                    logger.warning(
                        "Rejected {} — Div={} (wanted {})",
                        dest,
                        divs,
                        league_code,
                    )
                    dest.unlink()
                    return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not validate football-data {}: {}", dest, exc)
            try:
                dest.unlink()
            except OSError:
                pass
            return None
        return dest

    def load_season(self, league_code: str, season_start: int) -> pd.DataFrame:
        path = self.season_path(league_code, season_start)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")
        return _normalize_frame(df, league_code, season_start, self.mapper)

    def fetch_and_load(
        self,
        league_code: str,
        start_season: int,
        end_season: int | None = None,
        *,
        force: bool = False,
        preferred_close_books: list[str] | None = None,
    ) -> pd.DataFrame:
        preferred = preferred_close_books or ["PSC", "B365C", "AvgC", "PS", "B365", "Avg"]
        frames: list[pd.DataFrame] = []
        for y in _season_years(start_season, end_season):
            path = self.fetch_season(league_code, y, force=force)
            if path is None:
                continue
            try:
                frames.append(self.load_season(league_code, y))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load {} {}: {}", league_code, y, exc)
                path = self.season_path(league_code, y)
                if path.exists() and "wrong league" in str(exc).lower():
                    try:
                        path.unlink()
                        logger.warning("Removed poisoned cache {}", path)
                    except OSError:
                        pass
        if not frames:
            raise RuntimeError(f"No football-data seasons loaded for {league_code}")
        df = pd.concat(frames, ignore_index=True)
        # Re-coerce odds columns after concat (mixed dtypes across seasons)
        for c in df.columns:
            if c in ("home_team", "away_team", "ftr", "HTR", "Date", "Time", "div", "close_book"):
                continue
            if any(c.startswith(p) for p in ODDS_PREFIXES) or c.endswith(("H", "D", "A")):
                if not pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == object:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
        df = select_closing_1x2(df, preferred)
        df = select_closing_ou25(df)
        df = select_ah_main(df)
        df = df.sort_values(["Date", "home_team", "away_team"]).reset_index(drop=True)
        logger.info(
            "football-data {}: {} matches ({}–{})",
            league_code,
            len(df),
            df["season"].min(),
            df["season"].max(),
        )
        return df


def ingest_football_data_from_config(cfg: dict[str, Any], data_dir: Path) -> pd.DataFrame:
    fd_cfg = cfg.get("data", {}).get("football_data", {})
    if not fd_cfg.get("enabled", True):
        raise RuntimeError("football_data disabled in config")
    ingester = FootballDataIngester(
        raw_dir=data_dir / "raw" / "football_data",
        base_url=fd_cfg.get("base_url", "https://www.football-data.co.uk/mmz4281"),
    )
    frames: list[pd.DataFrame] = []
    for league in cfg.get("leagues", []):
        frames.append(
            ingester.fetch_and_load(
                league_code=league["code"],
                start_season=int(league["start_season"]),
                end_season=league.get("end_season"),
                preferred_close_books=fd_cfg.get("preferred_close_books"),
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
