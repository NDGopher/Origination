"""
Align football-data, Understat (and later FBref) on date + home + away.

Produces a single match-level table used by features / models / backtest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from origination.utils.seeding import season_from_date
from origination.utils.team_names import DEFAULT_MAPPER


def _date_key(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.normalize()


def _pair_key(dates, homes, aways) -> set[tuple]:
    return set(zip(pd.to_datetime(dates).dt.normalize(), homes.astype(str), aways.astype(str)))


def _ftr(hg: float, ag: float) -> str:
    if hg > ag:
        return "H"
    if ag > hg:
        return "A"
    return "D"


def _fill_xg_from_football_data(base: pd.DataFrame) -> pd.DataFrame:
    """Primeira 2026/27 (and similar) ship HxG/AxG on football-data when Understat is absent."""
    out = base.copy()
    if "home_xg" not in out.columns:
        out["home_xg"] = pd.NA
    if "away_xg" not in out.columns:
        out["away_xg"] = pd.NA
    if "home_xg_fd" in out.columns:
        out["home_xg"] = out["home_xg"].where(out["home_xg"].notna(), out["home_xg_fd"])
    if "away_xg_fd" in out.columns:
        out["away_xg"] = out["away_xg"].where(out["away_xg"].notna(), out["away_xg_fd"])
    return out


def _append_unmatched_results(
    base: pd.DataFrame,
    extra: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    """Append finished matches that football-data has not published yet (no closing odds)."""
    if extra is None or len(extra) == 0:
        return base
    us = extra.copy()
    date_col = "date" if "date" in us.columns else "Date"
    if date_col not in us.columns:
        return base
    us["date"] = _date_key(us[date_col])
    need = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    if any(c not in us.columns for c in need):
        return base
    us = us.dropna(subset=need)
    if len(base) == 0:
        have: set[tuple] = set()
    else:
        have = _pair_key(base["date"], base["home_team"], base["away_team"])
    mask = [
        (pd.Timestamp(d).normalize(), str(h), str(a)) not in have
        for d, h, a in zip(us["date"], us["home_team"], us["away_team"], strict=True)
    ]
    miss = us.loc[mask].copy()
    if miss.empty:
        return base

    rows: list[dict] = []
    for _, r in miss.iterrows():
        hg = float(r["home_goals"])
        ag = float(r["away_goals"])
        dt = pd.Timestamp(r["date"]).normalize()
        home = str(r["home_team"])
        away = str(r["away_team"])
        ftr = _ftr(hg, ag)
        xh = r["home_xg"] if "home_xg" in r.index and pd.notna(r.get("home_xg")) else pd.NA
        xa = r["away_xg"] if "away_xg" in r.index and pd.notna(r.get("away_xg")) else pd.NA
        season = r["season"] if "season" in r.index and pd.notna(r.get("season")) else season_from_date(dt)
        rows.append(
            {
                "Date": dt,
                "date": dt,
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
                "home_xg": xh,
                "away_xg": xa,
                "ftr": ftr,
                "result_1x2": ftr,
                "goal_diff": hg - ag,
                "total_goals": hg + ag,
                "match_id": (
                    f"{dt.strftime('%Y%m%d')}_"
                    f"{home.replace(' ', '')}_"
                    f"{away.replace(' ', '')}"
                ),
                "season": int(season),
                "result_source": source,
                "close_h": pd.NA,
                "close_d": pd.NA,
                "close_a": pd.NA,
            }
        )
    add = pd.DataFrame(rows)
    logger.info("Appended {} {}-only results not in football-data", len(add), source)
    return pd.concat([base, add], ignore_index=True, sort=False)


def align_matches(
    fd: pd.DataFrame,
    understat: pd.DataFrame | None = None,
    fbref: pd.DataFrame | None = None,
    *,
    require_odds: bool = True,
    min_date: str | None = "2014-08-01",
    extra_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Left-join Understat xG onto football-data rows (source of odds + results).

    Current-season results that exist on Understat (or Pulse) but not yet on
    football-data.co.uk are appended without closing odds so form / xG stay fresh.
    Those extra rows are never used as priced backtest bets.
    """
    base = fd.copy()
    base["date"] = _date_key(base["Date"])
    if min_date:
        base = base[base["date"] >= pd.Timestamp(min_date)]
    base["result_source"] = "football_data"

    if require_odds:
        before = len(base)
        base = base.dropna(subset=["close_h", "close_d", "close_a"])
        logger.info("Dropped {} matches without closing 1X2 odds", before - len(base))

    if understat is not None and len(understat):
        us = understat.copy()
        us["date"] = _date_key(us["date"])
        keep_cols = ["date", "home_team", "away_team", "home_xg", "away_xg"]
        if "understat_id" in us.columns:
            keep_cols.append("understat_id")
        us_keys = us[keep_cols].drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
        before = len(base)
        merged = base.merge(
            us_keys,
            on=["date", "home_team", "away_team"],
            how="left",
            suffixes=("", "_us"),
        )
        matched = merged["home_xg"].notna().sum()
        logger.info(
            "Understat join: {}/{} matches matched ({:.1%})",
            matched,
            before,
            matched / max(before, 1),
        )
        miss = merged[merged["home_xg"].isna()][["date", "home_team", "away_team"]].head(20)
        if len(miss):
            logger.debug("Sample unmatched vs Understat:\n{}", miss)
        base = merged
    else:
        base["home_xg"] = pd.NA
        base["away_xg"] = pd.NA

    base = _fill_xg_from_football_data(base)

    if understat is not None and len(understat):
        base = _append_unmatched_results(base, understat, source="understat")
    if extra_results is not None and len(extra_results):
        base = _append_unmatched_results(base, extra_results, source="pulse")

    # FBref join deferred — team-match grain needs pivoting; scaffold only
    if fbref is not None and len(fbref):
        logger.info("FBref frame present ({} rows) — join not yet implemented", len(fbref))

    # Derived targets
    base["goal_diff"] = base["home_goals"] - base["away_goals"]
    base["total_goals"] = base["home_goals"] + base["away_goals"]
    if "ftr" in base.columns:
        base["result_1x2"] = base["ftr"].map({"H": "H", "D": "D", "A": "A"})

    base = base.sort_values(["date", "home_team"]).reset_index(drop=True)
    unmapped = DEFAULT_MAPPER.unmapped(
        list(base["home_team"].unique()) + list(base["away_team"].unique())
    )
    if unmapped:
        logger.warning("Potentially unmapped / passthrough teams: {}", unmapped)

    return base


def save_aligned(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    # Parquet-safe: coerce leftover string odds columns
    skip = {
        "home_team",
        "away_team",
        "ftr",
        "HTR",
        "Date",
        "Time",
        "div",
        "close_book",
        "match_id",
        "result_1x2",
        "referee",
        "result_source",
    }
    for c in out.columns:
        if c in skip:
            continue
        if out[c].dtype == object or str(out[c].dtype) == "string":
            converted = pd.to_numeric(out[c], errors="coerce")
            # only replace if mostly numeric / null after coerce
            if converted.notna().sum() >= out[c].notna().sum() * 0.5:
                out[c] = converted
    out.to_parquet(path, index=False)
    logger.info("Wrote aligned matches -> {} ({} rows)", path, len(out))
    return path


def load_aligned(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_aligned_from_config(
    cfg: dict[str, Any],
    data_dir: Path,
    fd: pd.DataFrame,
    understat: pd.DataFrame | None,
    fbref: pd.DataFrame | None,
    extra_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    align_cfg = cfg.get("data", {}).get("align", {})
    df = align_matches(
        fd,
        understat=understat,
        fbref=fbref,
        require_odds=align_cfg.get("require_odds", True),
        min_date=align_cfg.get("min_date"),
        extra_results=extra_results,
    )
    out_name = align_cfg.get("output", "matches_aligned.parquet")
    out = data_dir / "interim" / out_name
    save_aligned(df, out)
    return df
