"""
Match-level player strength from Understat per-match rosters.

Leakage-free: expected XI = previous fixture starters; player ratings from
appearances strictly before the target match. ``lineup_confirmed=False``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from origination.data_ingestion.understat_match_rosters import load_match_rosters


def _pos_group(pos: str) -> str:
    p = str(pos or "").upper()
    if p.startswith("G"):
        return "GK"
    if p.startswith("D"):
        return "DEF"
    if p.startswith("M"):
        return "MID"
    if p.startswith("F") or p.startswith("S"):
        return "FWD"
    return "UNK"


class MatchLevelPlayerStrengthProvider:
    """PlayerStrengthProvider backed by Understat match rosters."""

    def __init__(self) -> None:
        self._by_uid: pd.DataFrame | None = None  # understat_id → side strengths
        self._path: str | None = None

    def _ensure(self, config: dict[str, Any] | None) -> None:
        cfg = config or {}
        path = str(cfg.get("rosters_parquet", "data/interim/understat_match_rosters.parquet"))
        if self._by_uid is not None and self._path == path:
            return
        self._path = path
        app = load_match_rosters(Path(path))
        if app is None or len(app) == 0:
            logger.warning("Match rosters empty at {}; null strengths", path)
            self._by_uid = pd.DataFrame()
            return

        starter_min = float(cfg.get("starter_minutes", 45.0))
        xi_size = int(cfg.get("xi_size", 11))
        min_prior = int(cfg.get("min_prior_apps", 1))
        scale = float(cfg.get("delta_scale", 0.15))

        app = app.copy()
        app = app[app["player_id"].astype(str).str.len() > 0]
        app["minutes"] = pd.to_numeric(app["minutes"], errors="coerce").fillna(0.0)
        app = app[app["minutes"] > 0].copy()
        app["date"] = pd.to_datetime(app["date"], errors="coerce")
        app = app.dropna(subset=["date", "team", "understat_id"])
        app["understat_id"] = app["understat_id"].astype(int)
        app["pos_group"] = app["position"].map(_pos_group)
        app["att_p90"] = 90.0 * (
            app["xG"].astype(float) + app["xA"].astype(float)
        ) / app["minutes"].clip(lower=1.0)
        app["def_p90"] = 90.0 * app["xGBuildup"].astype(float) / app["minutes"].clip(lower=1.0)
        app = app.sort_values(["player_id", "date", "understat_id"]).reset_index(drop=True)

        g = app.groupby("player_id", sort=False)
        app["n_prior"] = g.cumcount()
        app["prior_att_p90"] = g["att_p90"].transform(lambda s: s.shift(1).expanding().mean())
        app["prior_def_p90"] = g["def_p90"].transform(lambda s: s.shift(1).expanding().mean())

        # League centering from all prior-available rows
        valid = app["n_prior"] >= min_prior
        league_att = float(app.loc[valid, "prior_att_p90"].mean()) if valid.any() else 0.0
        league_def = float(app.loc[valid, "prior_def_p90"].mean()) if valid.any() else 0.0
        league_att_std = float(app.loc[valid, "prior_att_p90"].std()) or 1.0
        league_def_std = float(app.loc[valid, "prior_def_p90"].std()) or 1.0

        app["att_z"] = ((app["prior_att_p90"] - league_att) / league_att_std).clip(-3.0, 3.0)
        app["def_z"] = ((app["prior_def_p90"] - league_def) / league_def_std).clip(-3.0, 3.0)

        # Team-match ordering for previous XI
        team_matches = (
            app.groupby(["team", "understat_id"], as_index=False)
            .agg(date=("date", "first"))
            .sort_values(["team", "date", "understat_id"])
        )
        team_matches["prev_uid"] = team_matches.groupby("team")["understat_id"].shift(1)

        starters = app[app["minutes"] >= starter_min].copy()
        starters = starters.sort_values(["understat_id", "team", "minutes"], ascending=[True, True, False])
        starters["rk"] = starters.groupby(["understat_id", "team"]).cumcount()
        starters = starters[starters["rk"] < xi_size]

        # Map current match → previous match's starter player rows with *current-match* prior ratings
        # Rating at match t for a player = prior_* on the row of match t (excludes t).
        # For expected XI from prev match, look up each player's rating as of match t.
        rating_at = app.loc[
            valid,
            ["understat_id", "player_id", "att_z", "def_z", "pos_group", "team"],
        ]

        # Build expected XI player list from prev_uid starters
        prev_starters = starters.rename(
            columns={"understat_id": "prev_uid", "player_id": "xi_player_id"}
        )[["prev_uid", "team", "xi_player_id"]]

        cur = team_matches.dropna(subset=["prev_uid"]).copy()
        cur["prev_uid"] = cur["prev_uid"].astype(int)
        cur = cur.merge(prev_starters, on=["prev_uid", "team"], how="left")
        # Attach player rating as of *current* understat_id
        cur = cur.merge(
            rating_at.rename(columns={"player_id": "xi_player_id"}),
            on=["understat_id", "xi_player_id", "team"],
            how="left",
        )

        rows_side: list[dict[str, Any]] = []
        for (uid, team), g in cur.dropna(subset=["xi_player_id"]).groupby(
            ["understat_id", "team"]
        ):
            if g["att_z"].notna().sum() == 0:
                rows_side.append(
                    {
                        "understat_id": int(uid),
                        "team": team,
                        "attack_delta": 0.0,
                        "defence_delta": 0.0,
                        "covered": False,
                    }
                )
                continue
            is_att = g["pos_group"].isin(["MID", "FWD", "UNK"])
            is_def = g["pos_group"].isin(["GK", "DEF", "MID"])
            atk = g.loc[is_att, "att_z"].mean() if is_att.any() else g["att_z"].mean()
            dfn = g.loc[is_def, "def_z"].mean() if is_def.any() else g["def_z"].mean()
            rows_side.append(
                {
                    "understat_id": int(uid),
                    "team": team,
                    "attack_delta": scale * float(atk),
                    "defence_delta": scale * float(dfn),
                    "covered": True,
                }
            )
        side = pd.DataFrame(rows_side)
        if side.empty:
            side = pd.DataFrame(
                columns=["understat_id", "team", "attack_delta", "defence_delta", "covered"]
            )

        # One row per understat_id with home/away via side letter from app
        side_map = app.groupby(["understat_id", "team"], as_index=False).agg(
            side=("side", "first")
        )
        side = side.merge(side_map, on=["understat_id", "team"], how="left")

        homes = side[side["side"] == "h"].rename(
            columns={
                "attack_delta": "attack_delta_home",
                "defence_delta": "defence_delta_home",
                "covered": "cov_h",
            }
        )[["understat_id", "attack_delta_home", "defence_delta_home", "cov_h"]]
        aways = side[side["side"] == "a"].rename(
            columns={
                "attack_delta": "attack_delta_away",
                "defence_delta": "defence_delta_away",
                "covered": "cov_a",
            }
        )[["understat_id", "attack_delta_away", "defence_delta_away", "cov_a"]]

        by_uid = homes.merge(aways, on="understat_id", how="outer")
        for c in (
            "attack_delta_home",
            "attack_delta_away",
            "defence_delta_home",
            "defence_delta_away",
        ):
            by_uid[c] = by_uid[c].fillna(0.0)
        by_uid["lineup_covered"] = by_uid.get("cov_h", False).fillna(False) & by_uid.get(
            "cov_a", False
        ).fillna(False)
        by_uid["lineup_strength_home"] = (
            by_uid["attack_delta_home"] - by_uid["defence_delta_away"]
        )
        by_uid["lineup_strength_away"] = (
            by_uid["attack_delta_away"] - by_uid["defence_delta_home"]
        )
        self._by_uid = by_uid
        logger.info(
            "MatchLevelPlayerStrength: precomputed {} matches (covered={})",
            len(by_uid),
            int(by_uid["lineup_covered"].sum()),
        )

    def lineup_strength(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        del as_of  # per-match leakage handled in precompute
        self._ensure(config)
        out = pd.DataFrame({"match_id": matches["match_id"].values})
        out["lineup_strength_home"] = np.nan
        out["lineup_strength_away"] = np.nan
        out["attack_delta_home"] = 0.0
        out["attack_delta_away"] = 0.0
        out["defence_delta_home"] = 0.0
        out["defence_delta_away"] = 0.0
        out["lineup_confirmed"] = False
        out["lineup_covered"] = False

        if self._by_uid is None or len(self._by_uid) == 0:
            return out
        if "understat_id" not in matches.columns:
            logger.warning("matches lack understat_id; match-level strengths unavailable")
            return out

        m = matches[["match_id", "understat_id"]].copy()
        m["understat_id"] = pd.to_numeric(m["understat_id"], errors="coerce")
        merged = m.merge(self._by_uid, on="understat_id", how="left")
        for col in (
            "attack_delta_home",
            "attack_delta_away",
            "defence_delta_home",
            "defence_delta_away",
            "lineup_strength_home",
            "lineup_strength_away",
        ):
            out[col] = merged[col].fillna(0.0 if "delta" in col else np.nan).values
        out["lineup_covered"] = merged["lineup_covered"].fillna(False).astype(bool).values
        out["lineup_confirmed"] = False
        return out
