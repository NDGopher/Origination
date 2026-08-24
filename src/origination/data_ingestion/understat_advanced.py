"""
Load cached Understat team-history (PPDA, deep, npxG) and join onto match rows.

Values are post-match for that fixture; feature store must lag with shift(1).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def load_understat_team_history(raw_understat_dir: Path) -> pd.DataFrame:
    """Concatenate all ``*_team_history.parquet`` under the Understat raw tree."""
    root = Path(raw_understat_dir)
    files = sorted(root.rglob("*_team_history.parquet"))
    if not files:
        logger.warning("No Understat team_history parquet files under {}", root)
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    logger.info("Loaded Understat team history: {} rows from {} files", len(df), len(files))
    return df


def enrich_matches_with_understat_advanced(
    matches: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach home/away advanced metrics for the same match date+team.

    Adds: home_ppda, away_ppda, home_deep, away_deep, home_npxg, away_npxg,
    home_npxga, away_npxga, home_deep_allowed, away_deep_allowed.
    """
    if history is None or len(history) == 0:
        return matches

    out = matches.copy()
    out["_date_key"] = pd.to_datetime(out["date"]).dt.normalize()

    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
    cols = [
        "date",
        "team",
        "is_home",
        "ppda",
        "deep",
        "deep_allowed",
        "npxg_for",
        "npxg_against",
    ]
    cols = [c for c in cols if c in hist.columns]
    hist = hist[cols].drop_duplicates(subset=["date", "team", "is_home"], keep="last")

    home_hist = hist[hist["is_home"] == True].rename(  # noqa: E712
        columns={
            "team": "home_team",
            "ppda": "home_ppda",
            "deep": "home_deep",
            "deep_allowed": "home_deep_allowed",
            "npxg_for": "home_npxg",
            "npxg_against": "home_npxga",
            "date": "_date_key",
        }
    )
    away_hist = hist[hist["is_home"] == False].rename(  # noqa: E712
        columns={
            "team": "away_team",
            "ppda": "away_ppda",
            "deep": "away_deep",
            "deep_allowed": "away_deep_allowed",
            "npxg_for": "away_npxg",
            "npxg_against": "away_npxga",
            "date": "_date_key",
        }
    )
    drop_is = [c for c in ("is_home",) if c in home_hist.columns]
    home_hist = home_hist.drop(columns=drop_is, errors="ignore")
    away_hist = away_hist.drop(columns=[c for c in ("is_home",) if c in away_hist.columns], errors="ignore")

    before = len(out)
    out = out.merge(
        home_hist,
        on=["_date_key", "home_team"],
        how="left",
    ).merge(
        away_hist,
        on=["_date_key", "away_team"],
        how="left",
    )
    matched = out["home_ppda"].notna().sum() if "home_ppda" in out.columns else 0
    logger.info(
        "Understat advanced join: {}/{} matches with home_ppda ({:.1%})",
        matched,
        before,
        matched / max(before, 1),
    )
    return out.drop(columns=["_date_key"], errors="ignore")
