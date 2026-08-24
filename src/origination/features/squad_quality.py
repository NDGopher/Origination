"""
Squad-quality player strength from Understat season player aggregates.

v1 is NOT confirmed XI strength — it uses prior-season squad contribution
(top minutes players' npxG+xA / xGChain per 90). Leakage-free: match in
season S only sees season S-1 player tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from origination.utils.team_names import DEFAULT_MAPPER, TeamNameMapper


def parse_understat_players_season(
    path: Path,
    *,
    league: str,
    season: int,
    mapper: TeamNameMapper | None = None,
) -> pd.DataFrame:
    mapper = mapper or DEFAULT_MAPPER
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for p in data.get("players") or []:
        try:
            minutes = float(p.get("time") or 0)
            if minutes <= 0:
                continue
            team_raw = str(p.get("team_title") or "")
            # Transfers can appear as "TeamA,TeamB" — attribute to primary (first) club
            team_raw = team_raw.split(",")[0].strip()
            if not team_raw:
                continue
            team = mapper.canonicalize(team_raw)
            rows.append(
                {
                    "player_id": str(p.get("id")),
                    "player_name": p.get("player_name"),
                    "team": team,
                    "team_raw": team_raw,
                    "position": str(p.get("position") or ""),
                    "minutes": minutes,
                    "games": float(p.get("games") or 0),
                    "npxG": float(p.get("npxG") or 0),
                    "xA": float(p.get("xA") or 0),
                    "xGChain": float(p.get("xGChain") or 0),
                    "xGBuildup": float(p.get("xGBuildup") or 0),
                    "goals": float(p.get("goals") or 0),
                    "season": int(season),
                    "league_understat": league,
                }
            )
        except (TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


def load_understat_players(
    raw_dir: Path,
    *,
    leagues: list[str] | None = None,
    mapper: TeamNameMapper | None = None,
) -> pd.DataFrame:
    """Load/parse all season player tables under data/raw/understat."""
    raw_dir = Path(raw_dir)
    frames: list[pd.DataFrame] = []
    for league_dir in sorted(raw_dir.iterdir()):
        if not league_dir.is_dir():
            continue
        league = league_dir.name
        if leagues and league not in leagues:
            continue
        for jf in sorted(league_dir.glob("*_league.json")):
            try:
                season = int(jf.name.split("_")[0])
            except ValueError:
                continue
            cache = league_dir / f"{season}_players.parquet"
            if cache.exists():
                frames.append(pd.read_parquet(cache))
                continue
            df = parse_understat_players_season(jf, league=league, season=season, mapper=mapper)
            if len(df):
                df.to_parquet(cache, index=False)
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    logger.info("Loaded Understat players: {} rows from {} seasons/files", len(out), len(frames))
    return out


def build_team_season_squad_metrics(
    players: pd.DataFrame,
    *,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Per team-season: top-N by minutes → attack/buildup rates (per 90).
    """
    if players is None or len(players) == 0:
        return pd.DataFrame()
    rows = []
    for (league, season, team), g in players.groupby(["league_understat", "season", "team"]):
        g = g.sort_values("minutes", ascending=False).head(int(top_n))
        mins = float(g["minutes"].sum())
        if mins < 1:
            continue
        att = float((g["npxG"] + g["xA"]).sum()) / mins * 90.0
        chain = float(g["xGChain"].sum()) / mins * 90.0
        buildup = float(g["xGBuildup"].sum()) / mins * 90.0
        # defensive proxy: non-F positions buildup share (possession without finishing)
        def_mask = ~g["position"].str.upper().str.contains("F", na=False)
        def_mins = float(g.loc[def_mask, "minutes"].sum()) or mins
        def_build = float(g.loc[def_mask, "xGBuildup"].sum()) / def_mins * 90.0
        rows.append(
            {
                "league_understat": league,
                "season": int(season),
                "team": team,
                "squad_attack_p90": att,
                "squad_chain_p90": chain,
                "squad_buildup_p90": buildup,
                "squad_def_buildup_p90": def_build,
                "squad_minutes": mins,
            }
        )
    return pd.DataFrame(rows)


class UnderstatSquadQualityProvider:
    """
    Prior-season squad quality → attack/defence deltas (z-scored within league-season).

    Config keys:
      players_path: optional parquet of all players (else load from raw_dir)
      raw_dir: default data/raw/understat
      top_n: 15
      understat_league: EPL | Bundesliga | Serie_A | La_liga
    """

    def __init__(self, team_metrics: pd.DataFrame | None = None) -> None:
        self._metrics = team_metrics
        self._z: pd.DataFrame | None = None

    def _ensure(self, config: dict[str, Any] | None) -> None:
        if self._z is not None:
            return
        cfg = config or {}
        if self._metrics is None:
            raw = Path(cfg.get("raw_dir", "data/raw/understat"))
            players_path = cfg.get("players_path")
            if players_path and Path(players_path).exists():
                players = pd.read_parquet(players_path)
            else:
                leagues = cfg.get("leagues")
                players = load_understat_players(raw, leagues=leagues)
            self._metrics = build_team_season_squad_metrics(
                players, top_n=int(cfg.get("top_n", 15))
            )
        m = self._metrics.copy()
        if cfg.get("understat_league"):
            m = m[m["league_understat"] == cfg["understat_league"]]
        # z-score within league-season
        for col in ["squad_attack_p90", "squad_chain_p90", "squad_def_buildup_p90"]:
            m[f"z_{col}"] = m.groupby(["league_understat", "season"])[col].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-6)
            )
        self._z = m

    def lineup_strength(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        self._ensure(config)
        assert self._z is not None
        cfg = config or {}
        z = self._z.copy()
        z["avail_season"] = z["season"].astype(int) + 1  # prior season → next season matches
        home = (
            z.rename(
                columns={
                    "team": "home_team",
                    "z_squad_attack_p90": "z_att_home",
                    "z_squad_chain_p90": "z_chain_home",
                    "z_squad_def_buildup_p90": "z_def_home",
                }
            )[
                [
                    "home_team",
                    "avail_season",
                    "z_att_home",
                    "z_chain_home",
                    "z_def_home",
                ]
            ]
            .rename(columns={"avail_season": "season"})
        )
        away = (
            z.rename(
                columns={
                    "team": "away_team",
                    "z_squad_attack_p90": "z_att_away",
                    "z_squad_chain_p90": "z_chain_away",
                    "z_squad_def_buildup_p90": "z_def_away",
                }
            )[
                [
                    "away_team",
                    "avail_season",
                    "z_att_away",
                    "z_chain_away",
                    "z_def_away",
                ]
            ]
            .rename(columns={"avail_season": "season"})
        )

        out = matches[["match_id", "season", "home_team", "away_team"]].copy()
        out["season"] = out["season"].astype(int)
        out = out.merge(home, on=["home_team", "season"], how="left")
        out = out.merge(away, on=["away_team", "season"], how="left")

        # Attack delta: finishing+chain; defence delta: inverted buildup deficit
        # Positive attack_delta → higher λ; positive defence_delta → stronger defence (lower opp λ)
        scale = float(cfg.get("delta_scale", 0.15))
        out["attack_delta_home"] = scale * (
            out["z_att_home"].fillna(0.0) * 0.6 + out["z_chain_home"].fillna(0.0) * 0.4
        )
        out["attack_delta_away"] = scale * (
            out["z_att_away"].fillna(0.0) * 0.6 + out["z_chain_away"].fillna(0.0) * 0.4
        )
        out["defence_delta_home"] = scale * out["z_def_home"].fillna(0.0)
        out["defence_delta_away"] = scale * out["z_def_away"].fillna(0.0)
        out["lineup_strength_home"] = out["attack_delta_home"] - out["defence_delta_away"]
        out["lineup_strength_away"] = out["attack_delta_away"] - out["defence_delta_home"]
        out["lineup_confirmed"] = False
        # Coverage flag for diagnostics
        out["squad_quality_covered"] = out["z_att_home"].notna() & out["z_att_away"].notna()
        return out[
            [
                "match_id",
                "lineup_strength_home",
                "lineup_strength_away",
                "attack_delta_home",
                "attack_delta_away",
                "defence_delta_home",
                "defence_delta_away",
                "lineup_confirmed",
                "squad_quality_covered",
            ]
        ]
