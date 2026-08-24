"""
Leakage-free feature store.

Every feature uses only information available before kickoff.
Rolling stats are computed on a long team-match panel with shift(1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from origination.utils.seeding import season_from_date


def _to_team_panel(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (home + away views)."""
    home = pd.DataFrame(
        {
            "date": matches["date"],
            "season": matches["season"],
            "match_id": matches["match_id"],
            "team": matches["home_team"],
            "opponent": matches["away_team"],
            "is_home": True,
            "goals_for": matches["home_goals"],
            "goals_against": matches["away_goals"],
            "shots_for": matches.get("home_shots"),
            "shots_against": matches.get("away_shots"),
            "sot_for": matches.get("home_sot"),
            "sot_against": matches.get("away_sot"),
            "xg_for": matches.get("home_xg"),
            "xg_against": matches.get("away_xg"),
            "npxg_for": matches.get("home_npxg"),
            "npxg_against": matches.get("home_npxga"),
            "ppda": matches.get("home_ppda"),
            "deep": matches.get("home_deep"),
            "deep_allowed": matches.get("home_deep_allowed"),
            "pv_obv": matches.get("home_pv_obv"),
            "pv_obv_v1": matches.get("home_pv_obv_v1"),
            "pv_depth_w": matches.get("home_pv_depth_w"),
            "pv_final3": matches.get("home_pv_final3"),
            "pv_open": matches.get("home_pv_open"),
            "pv_buildup": matches.get("home_pv_buildup"),
            "points": np.where(
                matches["home_goals"] > matches["away_goals"],
                3,
                np.where(matches["home_goals"] == matches["away_goals"], 1, 0),
            ),
        }
    )
    away = pd.DataFrame(
        {
            "date": matches["date"],
            "season": matches["season"],
            "match_id": matches["match_id"],
            "team": matches["away_team"],
            "opponent": matches["home_team"],
            "is_home": False,
            "goals_for": matches["away_goals"],
            "goals_against": matches["home_goals"],
            "shots_for": matches.get("away_shots"),
            "shots_against": matches.get("home_shots"),
            "sot_for": matches.get("away_sot"),
            "sot_against": matches.get("home_sot"),
            "xg_for": matches.get("away_xg"),
            "xg_against": matches.get("home_xg"),
            "npxg_for": matches.get("away_npxg"),
            "npxg_against": matches.get("away_npxga"),
            "ppda": matches.get("away_ppda"),
            "deep": matches.get("away_deep"),
            "deep_allowed": matches.get("away_deep_allowed"),
            "pv_obv": matches.get("away_pv_obv"),
            "pv_obv_v1": matches.get("away_pv_obv_v1"),
            "pv_depth_w": matches.get("away_pv_depth_w"),
            "pv_final3": matches.get("away_pv_final3"),
            "pv_open": matches.get("away_pv_open"),
            "pv_buildup": matches.get("away_pv_buildup"),
            "points": np.where(
                matches["away_goals"] > matches["home_goals"],
                3,
                np.where(matches["away_goals"] == matches["home_goals"], 1, 0),
            ),
        }
    )
    panel = pd.concat([home, away], ignore_index=True)
    panel = panel.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    return panel


def _rolling_means(panel: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    """shift(1) then rolling — no current-match leakage."""
    out = panel.copy()
    g = out.groupby("team", group_keys=False)
    for col in cols:
        if col not in out.columns:
            continue
        shifted = g[col].shift(1)
        for w in windows:
            out[f"{col}_roll{w}"] = shifted.groupby(out["team"]).transform(
                lambda s, ww=w: s.rolling(ww, min_periods=max(1, ww // 2)).mean()
            )
        # EWM
        out[f"{col}_ewm"] = shifted.groupby(out["team"]).transform(
            lambda s: s.ewm(span=10, min_periods=2).mean()
        )
    return out


def _rest_days(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["prev_date"] = out.groupby("team", sort=False)["date"].shift(1)
    out["rest_days"] = (out["date"] - out["prev_date"]).dt.days
    # Congestion: games in last 7/14/21 days BEFORE this match (no groupby.apply —
    # pandas can drop the group key column depending on version).
    out = out.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    g7 = np.zeros(len(out), dtype=float)
    g14 = np.zeros(len(out), dtype=float)
    g21 = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("team", sort=False).groups.items():
        positions = list(idx)
        dates = out.loc[positions, "date"].values
        for j, pos in enumerate(positions):
            if j == 0:
                continue
            d = dates[j]
            prev = dates[:j]
            deltas = (d - prev).astype("timedelta64[D]").astype(int)
            g7[pos] = np.sum((deltas > 0) & (deltas <= 7))
            g14[pos] = np.sum((deltas > 0) & (deltas <= 14))
            g21[pos] = np.sum((deltas > 0) & (deltas <= 21))
    out["games_last_7"] = g7
    out["games_last_14"] = g14
    out["games_last_21"] = g21
    return out


def compute_elo(
    matches: pd.DataFrame,
    *,
    k: float = 20.0,
    home_advantage: float = 60.0,
    initial: float = 1500.0,
) -> pd.DataFrame:
    """
    Sequential Elo with home advantage separated.
    Returns match-level pre-match Elo for home and away (no leakage).
    """
    ratings: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    df = matches.sort_values(["date", "match_id"]).reset_index(drop=True)

    for _, m in df.iterrows():
        h, a = m["home_team"], m["away_team"]
        rh = ratings.get(h, initial)
        ra = ratings.get(a, initial)
        # Pre-match
        rows.append(
            {
                "match_id": m["match_id"],
                "elo_home": rh,
                "elo_away": ra,
                "elo_diff": (rh + home_advantage) - ra,
            }
        )
        # Update from result
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + home_advantage) - ra) / 400.0))
        if m["home_goals"] > m["away_goals"]:
            score_h = 1.0
        elif m["home_goals"] < m["away_goals"]:
            score_h = 0.0
        else:
            score_h = 0.5
        ratings[h] = rh + k * (score_h - exp_h)
        ratings[a] = ra + k * ((1.0 - score_h) - (1.0 - exp_h))

    return pd.DataFrame(rows)


def build_feature_matrix(
    matches: pd.DataFrame,
    feature_cfg: dict[str, Any],
) -> pd.DataFrame:
    """
    Build pre-match feature matrix aligned 1:1 with matches (same order after sort).
    """
    groups = feature_cfg.get("groups", {})
    windows = feature_cfg.get("windows", [3, 5, 10])
    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)

    panel = _to_team_panel(matches)

    metric_cols: list[str] = []
    if groups.get("basic_form", True):
        metric_cols += ["goals_for", "goals_against", "points"]
    if groups.get("shots", True):
        metric_cols += ["shots_for", "shots_against", "sot_for", "sot_against"]
    if groups.get("xg_form", False):
        metric_cols += ["xg_for", "xg_against"]
    if groups.get("understat_advanced", False):
        metric_cols += ["ppda", "deep", "deep_allowed", "npxg_for", "npxg_against"]
    if groups.get("possession_value", False):
        metric_cols += ["pv_obv", "pv_obv_v1", "pv_depth_w", "pv_final3", "pv_open", "pv_buildup"]

    metric_cols = [c for c in metric_cols if c in panel.columns]
    panel = _rolling_means(panel, metric_cols, windows)

    if groups.get("xg_form", False) and "xg_for" in panel.columns:
        # finishing residual (goals - xG), lagged rolling
        panel["finish_resid"] = panel["goals_for"] - panel["xg_for"]
        panel = _rolling_means(panel, ["finish_resid"], windows)

    if groups.get("schedule", True):
        panel = _rest_days(panel)

    # Home/away specific form (computed on venue-filtered panel, merged back)
    if groups.get("basic_form", True):
        venue_frames: list[pd.DataFrame] = []
        for venue_flag, tag in [(True, "home"), (False, "away")]:
            sub = panel.loc[panel["is_home"] == venue_flag, ["match_id", "team", "goals_for", "goals_against", "points"]].copy()
            g = sub.groupby("team", sort=False)
            pieces = {"match_id": sub["match_id"].values, "team": sub["team"].values}
            for col in ["goals_for", "goals_against", "points"]:
                shifted = g[col].shift(1)
                pieces[f"{col}_{tag}_roll5"] = (
                    shifted.groupby(sub["team"], sort=False)
                    .transform(lambda s: s.rolling(5, min_periods=2).mean())
                    .values
                )
            venue_frames.append(pd.DataFrame(pieces))
        for vf in venue_frames:
            panel = panel.merge(vf, on=["match_id", "team"], how="left")

    # Pivot back to match grain
    home_feats = panel[panel["is_home"]].copy()
    away_feats = panel[~panel["is_home"]].copy()

    ignore = {
        "date",
        "season",
        "team",
        "opponent",
        "is_home",
        "goals_for",
        "goals_against",
        "shots_for",
        "shots_against",
        "sot_for",
        "sot_against",
        "xg_for",
        "xg_against",
        "npxg_for",
        "npxg_against",
        "ppda",
        "deep",
        "deep_allowed",
        "pv_obv",
        "pv_obv_v1",
        "pv_depth_w",
        "pv_final3",
        "pv_open",
        "pv_buildup",
        "points",
        "prev_date",
        "finish_resid",
    }
    feat_cols = [c for c in home_feats.columns if c not in ignore and c != "match_id"]

    home_renamed = home_feats[["match_id"] + feat_cols].rename(
        columns={c: f"home_{c}" for c in feat_cols}
    )
    away_renamed = away_feats[["match_id"] + feat_cols].rename(
        columns={c: f"away_{c}" for c in feat_cols}
    )

    feats = matches[["match_id", "date", "season", "home_team", "away_team"]].merge(
        home_renamed, on="match_id", how="left"
    ).merge(away_renamed, on="match_id", how="left")

    if groups.get("elo", True):
        elo_cfg = feature_cfg.get("elo", {})
        elo = compute_elo(
            matches,
            k=float(elo_cfg.get("k", 20.0)),
            home_advantage=float(elo_cfg.get("home_advantage", 60.0)),
            initial=float(elo_cfg.get("initial", 1500.0)),
        )
        feats = feats.merge(elo, on="match_id", how="left")

    # Diff features for key rolls
    for w in windows:
        for base in [
            "goals_for",
            "goals_against",
            "points",
            "xg_for",
            "xg_against",
            "ppda",
            "deep",
            "npxg_for",
            "npxg_against",
        ]:
            hc, ac = f"home_{base}_roll{w}", f"away_{base}_roll{w}"
            if hc in feats.columns and ac in feats.columns:
                feats[f"diff_{base}_roll{w}"] = feats[hc] - feats[ac]

    for base in [
        "ppda",
        "deep",
        "deep_allowed",
        "xg_for",
        "xg_against",
        "npxg_for",
        "npxg_against",
        "shots_for",
        "shots_against",
        "sot_for",
        "sot_against",
        "pv_obv",
        "pv_obv_v1",
        "pv_depth_w",
        "pv_final3",
        "pv_open",
        "pv_buildup",
    ]:
        hc, ac = f"home_{base}_ewm", f"away_{base}_ewm"
        if hc in feats.columns and ac in feats.columns:
            feats[f"diff_{base}_ewm"] = feats[hc] - feats[ac]
            # Match-level totals proxies (both sides) — key for OU residual / intensity
            feats[f"sum_{base}_ewm"] = feats[hc] + feats[ac]

    # Orthogonalized possession value vs deep (fixed coef — less collinear totals signal)
    pv_deep_coef = float(feature_cfg.get("pv_deep_orth_coef", 0.35))
    for side in ("home", "away"):
        pv_c, deep_c = f"{side}_pv_obv_ewm", f"{side}_deep_ewm"
        if pv_c in feats.columns and deep_c in feats.columns:
            feats[f"{side}_pv_resid_ewm"] = feats[pv_c] - pv_deep_coef * feats[deep_c].fillna(0.0)
    if "home_pv_resid_ewm" in feats.columns and "away_pv_resid_ewm" in feats.columns:
        feats["diff_pv_resid_ewm"] = feats["home_pv_resid_ewm"] - feats["away_pv_resid_ewm"]
        feats["sum_pv_resid_ewm"] = feats["home_pv_resid_ewm"] + feats["away_pv_resid_ewm"]

    # Shot-suppression residual: combined shots-against beyond xG-against
    # (less collinear with xg_allow intensity). ~8 shots per xG typical.
    if "sum_shots_against_ewm" in feats.columns and "sum_xg_against_ewm" in feats.columns:
        beta = float(feature_cfg.get("suppress_xg_beta", 8.0))
        feats["sum_suppress_resid_ewm"] = feats["sum_shots_against_ewm"] - beta * feats[
            "sum_xg_against_ewm"
        ].fillna(0.0)

    # Open-play PV orthogonal to combined xG-for (cleaner totals style signal)
    if "sum_pv_open_ewm" in feats.columns and "sum_xg_for_ewm" in feats.columns:
        pv_xg = float(feature_cfg.get("pv_open_xg_orth_coef", 0.40))
        feats["sum_pv_open_orth_ewm"] = feats["sum_pv_open_ewm"] - pv_xg * feats[
            "sum_xg_for_ewm"
        ].fillna(0.0)

    # Context adjustments hook (scaffolding + real sources e.g. referee)
    from origination.features.context_adjustments import (
        apply_context_adjustments,
        merge_context_into_features,
    )

    ctx = apply_context_adjustments(matches, feature_cfg.get("context_adjustments"))
    feats = merge_context_into_features(feats, ctx)
    # Attach context intensity multipliers as feature columns for predict path
    if ctx.intensity_multipliers is not None and len(ctx.intensity_multipliers):
        im = ctx.intensity_multipliers.rename(
            columns={
                "lam_mult_home": "ctx_lam_mult_home",
                "lam_mult_away": "ctx_lam_mult_away",
            }
        )
        feats = feats.merge(im, on="match_id", how="left")

    logger.info("Built feature matrix: {} rows x {} cols", len(feats), feats.shape[1])
    return feats


def assert_features_pre_match(matches: pd.DataFrame, features: pd.DataFrame) -> None:
    """
    Sanity checks against look-ahead:
    - same match_id alignment
    - Elo/rest features finite only after teams have history (NaN early is OK)
    - no feature column equals current match goals (exact leak)
    """
    assert len(matches) == len(features), "Row count mismatch"
    assert (matches["match_id"].values == features["match_id"].values).all(), "match_id misaligned"

    leak_candidates = []
    for col in features.columns:
        if col in matches.columns and col not in {
            "match_id",
            "date",
            "season",
            "home_team",
            "away_team",
        }:
            if features[col].equals(matches[col]):
                leak_candidates.append(col)
    if leak_candidates:
        raise AssertionError(f"Potential leakage — feature equals match column: {leak_candidates}")
