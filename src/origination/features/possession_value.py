"""
Possession / on-ball value (v1) from Understat match shots + roster buildup.

Practical xT-lite using data already cached in ``getMatchData`` JSON:
- depth-weighted shot threat: sum(xG * X)  (X in [0,1] toward opponent goal)
- final-third threat: sum(xG | X >= 0.66)
- open-play threat: sum(xG) excluding FromCorner / SetPiece / DirectFreekick
- roster chain/buildup: sum(xGChain), sum(xGBuildup)

Post-match values are joined onto fixtures then **lagged** in the feature store
(same contract as PPDA/deep). Live path uses the same lagged features.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

SET_PIECE = {"FromCorner", "SetPiece", "DirectFreekick", "Penalty"}


def _float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _shots_list(block: Any) -> list[dict[str, Any]]:
    if block is None:
        return []
    if isinstance(block, list):
        return [s for s in block if isinstance(s, dict)]
    if isinstance(block, dict):
        return [s for s in block.values() if isinstance(s, dict)]
    return []


def _roster_sum(block: Any, key: str) -> float:
    if not isinstance(block, dict):
        return 0.0
    total = 0.0
    for p in block.values():
        if isinstance(p, dict):
            total += _float(p.get(key))
    return total


def aggregate_match_possession_value(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return per-side PV metrics for one match payload."""
    shots = payload.get("rosters") and payload.get("shots") or payload.get("shots") or {}
    rosters = payload.get("rosters") or {}
    out: dict[str, dict[str, float]] = {}
    for side in ("h", "a"):
        slist = _shots_list(shots.get(side) if isinstance(shots, dict) else None)
        xg_sum = 0.0
        xg_depth = 0.0
        xg_final = 0.0
        xg_open = 0.0
        n = 0
        depth_sum = 0.0
        for s in slist:
            xg = _float(s.get("xG"))
            x = _float(s.get("X"))
            sit = str(s.get("situation") or "")
            xg_sum += xg
            xg_depth += xg * x
            depth_sum += x
            n += 1
            if x >= 0.66:
                xg_final += xg
            if sit not in SET_PIECE:
                xg_open += xg
        rb = rosters.get(side) if isinstance(rosters, dict) else {}
        out[side] = {
            "pv_xg": xg_sum,
            "pv_depth_w": xg_depth,  # primary OBV-lite
            "pv_final3": xg_final,
            "pv_open": xg_open,
            "pv_shot_depth": (depth_sum / n) if n else 0.0,
            "pv_n_shots": float(n),
            "pv_chain": _roster_sum(rb, "xGChain"),
            "pv_buildup": _roster_sum(rb, "xGBuildup"),
        }
    return out


def build_possession_value_table(
    cache_dir: Path,
    matches: pd.DataFrame,
    *,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Flatten cached match JSON → one row per team-match with PV metrics.

    ``matches`` needs understat_id, date, home_team, away_team, season.
    """
    cache_dir = Path(cache_dir)
    need = matches.dropna(subset=["understat_id"]).copy()
    need["understat_id"] = need["understat_id"].astype(int)
    need = need.drop_duplicates("understat_id")
    meta = {int(r.understat_id): r for r in need.itertuples(index=False)}

    def _one(uid: int) -> list[dict[str, Any]]:
        path = cache_dir / f"{uid}.json"
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        pv = aggregate_match_possession_value(payload)
        m = meta[uid]
        rows = []
        for side, team_attr in (("h", "home_team"), ("a", "away_team")):
            metrics = pv.get(side) or {}
            rows.append(
                {
                    "understat_id": uid,
                    "date": getattr(m, "date", None),
                    "season": getattr(m, "season", None),
                    "team": getattr(m, team_attr, None),
                    "is_home": side == "h",
                    **metrics,
                }
            )
        return rows

    rows: list[dict[str, Any]] = []
    uids = list(meta.keys())
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, u) for u in uids]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.extend(fut.result())
            if i % 1000 == 0:
                logger.info("PV aggregate {}/{}", i, len(uids))

    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Composite on-ball value: depth-weighted open-play threat + mild buildup
        # v2: prefer open-play (less set-piece noise) as primary OBV
        df["pv_obv"] = df["pv_open"].astype(float) * df["pv_shot_depth"].astype(float) + 0.15 * df[
            "pv_buildup"
        ].astype(float)
        # Keep v1-style depth-weighted total as secondary
        df["pv_obv_v1"] = df["pv_depth_w"].astype(float) + 0.25 * df["pv_buildup"].astype(float)

    logger.info("Possession-value table: {} team-match rows", len(df))
    return df


def enrich_matches_with_possession_value(
    matches: pd.DataFrame,
    pv_table: pd.DataFrame,
) -> pd.DataFrame:
    """Attach home_/away_ pv_* columns for the same match (post-match; lag in store)."""
    if pv_table is None or len(pv_table) == 0:
        return matches
    if "understat_id" not in matches.columns:
        logger.warning("No understat_id on matches; PV enrich skipped")
        return matches

    keep = [
        c
        for c in (
            "pv_obv",
            "pv_obv_v1",
            "pv_depth_w",
            "pv_final3",
            "pv_open",
            "pv_buildup",
            "pv_chain",
        )
        if c in pv_table.columns
    ]
    if not keep:
        return matches

    out = matches.copy()
    out["understat_id"] = pd.to_numeric(out["understat_id"], errors="coerce")

    home = pv_table.loc[pv_table["is_home"] == True, ["understat_id"] + keep].copy()  # noqa: E712
    home = home.rename(columns={c: f"home_{c}" for c in keep})
    away = pv_table.loc[pv_table["is_home"] == False, ["understat_id"] + keep].copy()  # noqa: E712
    away = away.rename(columns={c: f"away_{c}" for c in keep})

    # Drop prior columns if re-running
    drop_cols = [c for c in out.columns if c.startswith("home_pv_") or c.startswith("away_pv_")]
    out = out.drop(columns=drop_cols, errors="ignore")
    out = out.merge(home, on="understat_id", how="left").merge(away, on="understat_id", how="left")

    n = int(out["home_pv_obv"].notna().sum()) if "home_pv_obv" in out.columns else 0
    logger.info("Possession-value join: {}/{} matches with home_pv_obv", n, len(out))
    return out
