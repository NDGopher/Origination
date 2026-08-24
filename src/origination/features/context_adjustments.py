"""
Context adjustments layer — scaffolding for adaptive originator features.

This module defines the interface and no-op / passthrough implementations so that
injuries, lineups, formations, motivation, weather, coaching changes, referee,
travel, etc. can be plugged in later without rewriting the feature store or
backtester.

Adding a new adjustment
-----------------------
1. Create a class implementing ``ContextAdjustment`` (see below).
2. Register it in ``ADJUSTMENT_REGISTRY`` and enable it under
   ``features.context_adjustments.<name>.enabled: true`` in YAML.
3. Ensure its ``apply`` method uses ONLY pre-match information (assert dates /
   as-of timestamps). Return additive feature columns and/or multiplicative
   intensity factors keyed by match_id.
4. Add a unit test that feeds synthetic pre-match context and checks:
   - columns appear when enabled
   - disabling the flag removes them
   - no future leakage (values for match t ignore events after kickoff)

The walk-forward / live prediction path both call
``apply_context_adjustments(...)`` after the base feature matrix is built.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd
from loguru import logger


@dataclass
class AdjustmentResult:
    """Output of a context adjustment."""

    # Extra pre-match feature columns (index-aligned or match_id keyed)
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Optional intensity multipliers: columns match_id, lam_mult_home, lam_mult_away
    intensity_multipliers: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: dict[str, Any] = field(default_factory=dict)


class ContextAdjustment(ABC):
    """Base class for a single contextual adjustment source."""

    name: str = "base"

    @abstractmethod
    def apply(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> AdjustmentResult:
        """
        Produce features / intensity multipliers using only info available
        before each match's kickoff (and before ``as_of`` when provided).
        """


class NullAdjustment(ContextAdjustment):
    """Passthrough — used when a source is disabled or not yet wired."""

    name = "null"

    def apply(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> AdjustmentResult:
        return AdjustmentResult()


class InjuriesAdjustment(ContextAdjustment):
    """
    Player absences / injuries / rotation (scaffold).

    Future: subtract estimated player contribution (xG chain, minutes-weighted)
    from team attack/defence ratings given a confirmed or probable unavailable list.
    """

    name = "injuries"

    def apply(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> AdjustmentResult:
        logger.debug("InjuriesAdjustment scaffold — no data wired yet")
        empty = pd.DataFrame({"match_id": matches["match_id"]})
        empty["injury_attack_delta_home"] = 0.0
        empty["injury_attack_delta_away"] = 0.0
        empty["injury_defence_delta_home"] = 0.0
        empty["injury_defence_delta_away"] = 0.0
        return AdjustmentResult(features=empty, meta={"wired": False})


class LineupsAdjustment(ContextAdjustment):
    """Confirmed / probable lineups via PlayerStrengthProvider (elite interface)."""

    name = "lineups"

    def apply(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> AdjustmentResult:
        from origination.features.elite import resolve_player_provider

        cfg = config or {}
        provider = resolve_player_provider(cfg.get("provider"))
        feats = provider.lineup_strength(matches, as_of=as_of, config=cfg)
        wired = type(provider).__name__ != "NullPlayerStrengthProvider"
        # Mild λ multipliers when wired strengths exist
        intens = pd.DataFrame({"match_id": feats["match_id"]})
        coef = float(cfg.get("strength_coef", 0.0))
        if wired and coef != 0.0 and "attack_delta_home" in feats.columns:
            # Attack raises own λ; opponent defence raises our μ (their scoring)
            intens["lam_mult_home"] = 1.0 + coef * (
                feats["attack_delta_home"].fillna(0.0) - feats["defence_delta_away"].fillna(0.0)
            )
            intens["lam_mult_away"] = 1.0 + coef * (
                feats["attack_delta_away"].fillna(0.0) - feats["defence_delta_home"].fillna(0.0)
            )
            # clip mild
            intens["lam_mult_home"] = intens["lam_mult_home"].clip(0.85, 1.15)
            intens["lam_mult_away"] = intens["lam_mult_away"].clip(0.85, 1.15)
        return AdjustmentResult(
            features=feats,
            intensity_multipliers=intens if "lam_mult_home" in intens.columns else pd.DataFrame(),
            meta={"wired": wired, "provider": type(provider).__name__},
        )


class FormationAdjustment(ContextAdjustment):
    """Formation labels / embeddings via FormationEncoder (elite interface)."""

    name = "formations"

    def apply(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> AdjustmentResult:
        from origination.features.elite import resolve_formation_encoder

        cfg = config or {}
        encoder = resolve_formation_encoder(cfg.get("provider"))
        feats = encoder.encode(matches, as_of=as_of, config=cfg)
        wired = type(encoder).__name__ != "NullFormationEncoder"
        return AdjustmentResult(
            features=feats,
            meta={"wired": wired, "provider": type(encoder).__name__},
        )


class MotivationAdjustment(ContextAdjustment):
    """
    Table-position motivation from historical results only (pre-kickoff).

    For each match, build the season table from prior fixtures in that season
    (points / GD / GF). Derive lagged features available before kickoff:

      - table_pos_{side}, points_{side}, gd_{side}, games_played_{side}
      - games_remaining_{side}  (season_length - games_played)
      - pts_from_top_{side}, pts_from_safety_{side}  (safety = 17th place pts)
      - title_race_{side}, relegation_battle_{side}, euro_race_{side}
      - mid_table_dead_{side}, stakes_{side}  (continuous [0,1] motivation)
      - dead_rubber (both sides mid-table secure)

    Optional intensity (YAML coefficients, default 0 = features only):
      - title_coef: λ_side *= exp(title_coef * title_race_side)
      - releg_coef: λ_side *= exp(releg_coef * relegation_battle_side)
      - dead_rubber_coef: λ_side *= exp(-dead_rubber_coef * mid_table_dead_side)
      - stakes_coef: λ_side *= exp(stakes_coef * stakes_side)
      - motivation_diff_coef: asymmetric nudge from (stakes_home - stakes_away)
          λ_home *= exp(+coef * diff), λ_away *= exp(-coef * diff)

    Intensity is applied only when games_played >= min_games (default 8)
    so early-season table noise does not move λ.
    """

    name = "motivation"

    def apply(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> AdjustmentResult:
        import numpy as np

        cfg = config or {}
        season_length = int(cfg.get("season_length", 38))
        safety_rank = int(cfg.get("safety_rank", 17))  # 17th = first above drop zone
        title_pts_gap = float(cfg.get("title_pts_gap", 6.0))
        releg_pts_gap = float(cfg.get("releg_pts_gap", 6.0))
        euro_pts_gap = float(cfg.get("euro_pts_gap", 6.0))
        min_games = int(cfg.get("min_games", 8))
        late_games_left = int(cfg.get("late_games_left", 12))
        title_coef = float(cfg.get("title_coef", 0.0))
        releg_coef = float(cfg.get("releg_coef", 0.0))
        dead_rubber_coef = float(cfg.get("dead_rubber_coef", 0.0))
        stakes_coef = float(cfg.get("stakes_coef", 0.0))
        motivation_diff_coef = float(cfg.get("motivation_diff_coef", 0.0))

        required = {"match_id", "date", "season", "home_team", "away_team", "home_goals", "away_goals"}
        if not required.issubset(matches.columns):
            logger.warning(
                "MotivationAdjustment: missing {}; scaffold zeros",
                required - set(matches.columns),
            )
            empty = pd.DataFrame({"match_id": matches["match_id"]})
            for col in [
                "must_win_home",
                "must_win_away",
                "dead_rubber",
                "title_race_home",
                "title_race_away",
                "relegation_battle_home",
                "relegation_battle_away",
            ]:
                empty[col] = 0.0
            return AdjustmentResult(features=empty, meta={"wired": False, "skipped": "missing_columns"})

        df = matches.sort_values(["season", "date", "match_id"]).copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        # Running standings per season: team -> [played, pts, gd, gf]
        standings: dict[Any, dict[str, list[float]]] = {}
        rows: list[dict[str, Any]] = []

        def _table_snapshot(season_key: Any) -> pd.DataFrame:
            st = standings.get(season_key, {})
            if not st:
                return pd.DataFrame(columns=["team", "played", "pts", "gd", "gf", "pos"])
            recs = [
                {"team": t, "played": v[0], "pts": v[1], "gd": v[2], "gf": v[3]}
                for t, v in st.items()
            ]
            tab = pd.DataFrame(recs)
            tab = tab.sort_values(["pts", "gd", "gf", "team"], ascending=[False, False, False, True])
            tab["pos"] = np.arange(1, len(tab) + 1)
            return tab.reset_index(drop=True)

        def _side_feats(tab: pd.DataFrame, team: str) -> dict[str, float]:
            if len(tab) == 0 or team not in set(tab["team"]):
                return {
                    "pos": np.nan,
                    "pts": np.nan,
                    "gd": np.nan,
                    "played": 0.0,
                    "remaining": float(season_length),
                    "pts_from_top": np.nan,
                    "pts_from_safety": np.nan,
                    "title_race": 0.0,
                    "relegation_battle": 0.0,
                    "euro_race": 0.0,
                    "mid_table_dead": 0.0,
                    "stakes": 0.0,
                    "must_win": 0.0,
                }
            row = tab.loc[tab["team"] == team].iloc[0]
            played = float(row["played"])
            pts = float(row["pts"])
            pos = float(row["pos"])
            remaining = float(max(0, season_length - played))
            top_pts = float(tab.iloc[0]["pts"])
            # Safety line: points of team in safety_rank (or last if fewer teams)
            safety_idx = min(safety_rank, len(tab)) - 1
            safety_pts = float(tab.iloc[safety_idx]["pts"])
            # Fourth place for European race reference
            euro_idx = min(4, len(tab)) - 1
            euro_pts = float(tab.iloc[euro_idx]["pts"])
            pts_from_top = top_pts - pts
            pts_from_safety = pts - safety_pts
            late = remaining <= late_games_left and played >= min_games

            title_race = float(
                late and pos <= 3 and pts_from_top <= title_pts_gap
            )
            # Soft continuous stakes components
            title_soft = 0.0
            if played >= min_games and pos <= 5:
                gap_factor = max(0.0, 1.0 - pts_from_top / max(title_pts_gap * 2.0, 1.0))
                late_factor = max(0.0, 1.0 - remaining / float(season_length))
                title_soft = gap_factor * (0.4 + 0.6 * late_factor)

            releg_battle = float(
                late and pos >= safety_rank and pts_from_safety <= releg_pts_gap
            )
            releg_soft = 0.0
            if played >= min_games and pos >= safety_rank - 2:
                # How close below/above safety (negative pts_from_safety = below line)
                gap_factor = max(0.0, 1.0 - abs(min(pts_from_safety, releg_pts_gap)) / max(releg_pts_gap * 2.0, 1.0))
                if pts_from_safety < releg_pts_gap:
                    late_factor = max(0.0, 1.0 - remaining / float(season_length))
                    releg_soft = gap_factor * (0.4 + 0.6 * late_factor)

            euro_race = float(
                late and 3 < pos <= 7 and (euro_pts - pts) <= euro_pts_gap
            )
            mid_dead = float(
                late
                and 8 <= pos <= 14
                and pts_from_top > title_pts_gap * 2
                and pts_from_safety > releg_pts_gap * 1.5
            )
            stakes = float(np.clip(max(title_soft, releg_soft) + 0.5 * euro_race, 0.0, 1.0))
            must_win = float(title_race or (releg_battle and remaining <= 6))

            return {
                "pos": pos,
                "pts": pts,
                "gd": float(row["gd"]),
                "played": played,
                "remaining": remaining,
                "pts_from_top": pts_from_top,
                "pts_from_safety": pts_from_safety,
                "title_race": title_race,
                "relegation_battle": releg_battle,
                "euro_race": euro_race,
                "mid_table_dead": mid_dead,
                "stakes": stakes,
                "must_win": must_win,
            }

        for _, m in df.iterrows():
            season = m["season"]
            if season not in standings:
                standings[season] = {}
            tab = _table_snapshot(season)
            h = _side_feats(tab, m["home_team"])
            a = _side_feats(tab, m["away_team"])
            dead_rubber = float(h["mid_table_dead"] and a["mid_table_dead"])
            rows.append(
                {
                    "match_id": m["match_id"],
                    "table_pos_home": h["pos"],
                    "table_pos_away": a["pos"],
                    "points_home": h["pts"],
                    "points_away": a["pts"],
                    "gd_home": h["gd"],
                    "gd_away": a["gd"],
                    "games_played_home": h["played"],
                    "games_played_away": a["played"],
                    "games_remaining_home": h["remaining"],
                    "games_remaining_away": a["remaining"],
                    "pts_from_top_home": h["pts_from_top"],
                    "pts_from_top_away": a["pts_from_top"],
                    "pts_from_safety_home": h["pts_from_safety"],
                    "pts_from_safety_away": a["pts_from_safety"],
                    "title_race_home": h["title_race"],
                    "title_race_away": a["title_race"],
                    "relegation_battle_home": h["relegation_battle"],
                    "relegation_battle_away": a["relegation_battle"],
                    "euro_race_home": h["euro_race"],
                    "euro_race_away": a["euro_race"],
                    "mid_table_dead_home": h["mid_table_dead"],
                    "mid_table_dead_away": a["mid_table_dead"],
                    "stakes_home": h["stakes"],
                    "stakes_away": a["stakes"],
                    "must_win_home": h["must_win"],
                    "must_win_away": a["must_win"],
                    "dead_rubber": dead_rubber,
                    "motivation_diff": h["stakes"] - a["stakes"],
                }
            )

            # Update standings AFTER emitting features (no leakage)
            hg = float(m["home_goals"]) if pd.notna(m["home_goals"]) else None
            ag = float(m["away_goals"]) if pd.notna(m["away_goals"]) else None
            if hg is None or ag is None:
                continue
            st = standings[season]
            for team, gf, ga in (
                (m["home_team"], hg, ag),
                (m["away_team"], ag, hg),
            ):
                if team not in st:
                    st[team] = [0.0, 0.0, 0.0, 0.0]
                st[team][0] += 1.0
                if gf > ga:
                    st[team][1] += 3.0
                elif gf == ga:
                    st[team][1] += 1.0
                st[team][2] += gf - ga
                st[team][3] += gf

        feat = pd.DataFrame(rows)

        # Intensity gate: zero out continuous drivers when either side has < min_games
        played_ok = (feat["games_played_home"].fillna(0) >= min_games) & (
            feat["games_played_away"].fillna(0) >= min_games
        )
        th = feat["title_race_home"].fillna(0).values * played_ok.values
        ta = feat["title_race_away"].fillna(0).values * played_ok.values
        rh = feat["relegation_battle_home"].fillna(0).values * played_ok.values
        ra = feat["relegation_battle_away"].fillna(0).values * played_ok.values
        dh = feat["mid_table_dead_home"].fillna(0).values * played_ok.values
        da = feat["mid_table_dead_away"].fillna(0).values * played_ok.values
        sh = feat["stakes_home"].fillna(0).values * played_ok.values
        sa = feat["stakes_away"].fillna(0).values * played_ok.values
        diff = (sh - sa)

        intens = pd.DataFrame({"match_id": feat["match_id"]})
        intens["lam_mult_home"] = np.exp(
            title_coef * th
            + releg_coef * rh
            - dead_rubber_coef * dh
            + stakes_coef * sh
            + motivation_diff_coef * diff
        )
        intens["lam_mult_away"] = np.exp(
            title_coef * ta
            + releg_coef * ra
            - dead_rubber_coef * da
            + stakes_coef * sa
            - motivation_diff_coef * diff
        )
        intens["lam_mult_home"] = np.clip(intens["lam_mult_home"], 0.85, 1.15)
        intens["lam_mult_away"] = np.clip(intens["lam_mult_away"], 0.85, 1.15)

        use_intens = any(
            abs(c) > 1e-12
            for c in (title_coef, releg_coef, dead_rubber_coef, stakes_coef, motivation_diff_coef)
        )
        n_title = int(((feat["title_race_home"] + feat["title_race_away"]) > 0).sum())
        n_releg = int(((feat["relegation_battle_home"] + feat["relegation_battle_away"]) > 0).sum())
        n_dead = int(feat["dead_rubber"].sum())
        logger.info(
            "MotivationAdjustment: title_flags={} releg_flags={} dead_rubber={} | coefs title={} releg={} dead={} stakes={} diff={}",
            n_title,
            n_releg,
            n_dead,
            title_coef,
            releg_coef,
            dead_rubber_coef,
            stakes_coef,
            motivation_diff_coef,
        )
        return AdjustmentResult(
            features=feat,
            intensity_multipliers=intens if use_intens else pd.DataFrame(),
            meta={
                "wired": True,
                "source": "season table from prior results",
                "season_length": season_length,
                "min_games": min_games,
                "title_coef": title_coef,
                "releg_coef": releg_coef,
                "dead_rubber_coef": dead_rubber_coef,
                "stakes_coef": stakes_coef,
                "motivation_diff_coef": motivation_diff_coef,
                "n_title_flag_matches": n_title,
                "n_releg_flag_matches": n_releg,
                "n_dead_rubber": n_dead,
            },
        )


class WeatherAdjustment(ContextAdjustment):
    """Weather conditions (scaffold)."""

    name = "weather"

    def apply(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> AdjustmentResult:
        empty = pd.DataFrame({"match_id": matches["match_id"]})
        empty["temp_c"] = float("nan")
        empty["wind_kph"] = float("nan")
        empty["precip_mm"] = float("nan")
        return AdjustmentResult(features=empty, meta={"wired": False})


class CoachingChangeAdjustment(ContextAdjustment):
    """
    New-manager flags + days/games since appointment (pre-match only).

    Data source
    -----------
    CSV at ``data/interim/coaching_changes.csv`` (or path from config
    ``changes_path``) with columns:
      team, change_date[, notes]

    Team names are canonicalized via ``TeamNameMapper``. Rebuild / extend the
    CSV when adding seasons; it is intentionally explicit and reviewable.

    Logic
    -----
    For each match and side, find the most recent change_date <= match date.
    Features (home and away):
      - coach_days_in_charge_{side}
      - coach_games_in_charge_{side}  (league matches for that team since change)
      - new_coach_{side}  = 1 if days_in_charge <= new_coach_days (default 60)
                            OR games_in_charge <= new_coach_games (default 8)

    Optional intensity (``bounce_coef``): for a side with new_coach=1,
    multiply that side's λ by exp(bounce_coef). Positive = new-manager bounce;
    negative = disruption. Default 0 (features only until measured).
    """

    name = "coaching_change"

    def apply(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> AdjustmentResult:
        import numpy as np
        from pathlib import Path

        from origination.utils.config import project_root
        from origination.utils.team_names import DEFAULT_MAPPER

        cfg = config or {}
        new_coach_days = int(cfg.get("new_coach_days", 60))
        new_coach_games = int(cfg.get("new_coach_games", 8))
        bounce_coef = float(cfg.get("bounce_coef", 0.0))
        rel = cfg.get("changes_path", "data/interim/coaching_changes.csv")
        path = Path(rel)
        if not path.is_absolute():
            path = project_root() / path

        empty_cols = [
            "match_id",
            "new_coach_home",
            "new_coach_away",
            "coach_days_in_charge_home",
            "coach_days_in_charge_away",
            "coach_games_in_charge_home",
            "coach_games_in_charge_away",
        ]

        if not path.exists():
            logger.warning("CoachingChangeAdjustment: missing {}; zeros only", path)
            empty = pd.DataFrame({"match_id": matches["match_id"]})
            for c in empty_cols[1:]:
                empty[c] = 0.0 if c.startswith("new_coach") else float("nan")
            return AdjustmentResult(features=empty, meta={"wired": True, "skipped": "missing_csv"})

        changes = pd.read_csv(path, parse_dates=["change_date"])
        changes["team"] = changes["team"].map(lambda t: DEFAULT_MAPPER.canonicalize(str(t)))
        changes = changes.dropna(subset=["team", "change_date"]).sort_values(["team", "change_date"])

        df = matches.sort_values(["date", "match_id"]).copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        # Precompute chronological team-match index for games-in-charge
        panel_rows = []
        for _, m in df.iterrows():
            panel_rows.append({"date": m["date"], "match_id": m["match_id"], "team": m["home_team"], "side": "home"})
            panel_rows.append({"date": m["date"], "match_id": m["match_id"], "team": m["away_team"], "side": "away"})
        panel = pd.DataFrame(panel_rows).sort_values(["team", "date", "match_id"]).reset_index(drop=True)
        panel["team_game_idx"] = panel.groupby("team").cumcount()

        # Attach latest change_date <= match date per team row
        def _attach_change(team: str, date: pd.Timestamp) -> pd.Timestamp | pd.NaT:
            sub = changes[(changes["team"] == team) & (changes["change_date"] <= date)]
            if len(sub) == 0:
                return pd.NaT
            return pd.Timestamp(sub.iloc[-1]["change_date"]).normalize()

        panel["change_date"] = [
            _attach_change(t, d) for t, d in zip(panel["team"], panel["date"], strict=True)
        ]
        panel["coach_days_in_charge"] = (panel["date"] - panel["change_date"]).dt.days

        # Games since change: count panel rows for team with date in (change_date, match_date]
        # Efficient approximation via merge of change game index
        change_game_idx = {}
        for _, row in panel.dropna(subset=["change_date"]).iterrows():
            key = (row["team"], pd.Timestamp(row["change_date"]))
            if key in change_game_idx:
                continue
            # first team game on/after change_date
            mask = (panel["team"] == row["team"]) & (panel["date"] >= row["change_date"])
            if mask.any():
                change_game_idx[key] = int(panel.loc[mask, "team_game_idx"].iloc[0])
            else:
                change_game_idx[key] = int(row["team_game_idx"])

        games = []
        for _, row in panel.iterrows():
            if pd.isna(row["change_date"]):
                games.append(np.nan)
                continue
            key = (row["team"], pd.Timestamp(row["change_date"]))
            base = change_game_idx.get(key, row["team_game_idx"])
            games.append(float(row["team_game_idx"] - base))
        panel["coach_games_in_charge"] = games
        panel["new_coach"] = (
            (panel["coach_days_in_charge"] <= new_coach_days)
            | (panel["coach_games_in_charge"] <= new_coach_games)
        ).astype(float)
        panel.loc[panel["change_date"].isna(), "new_coach"] = 0.0

        home = panel[panel["side"] == "home"][
            ["match_id", "new_coach", "coach_days_in_charge", "coach_games_in_charge"]
        ].rename(
            columns={
                "new_coach": "new_coach_home",
                "coach_days_in_charge": "coach_days_in_charge_home",
                "coach_games_in_charge": "coach_games_in_charge_home",
            }
        )
        away = panel[panel["side"] == "away"][
            ["match_id", "new_coach", "coach_days_in_charge", "coach_games_in_charge"]
        ].rename(
            columns={
                "new_coach": "new_coach_away",
                "coach_days_in_charge": "coach_days_in_charge_away",
                "coach_games_in_charge": "coach_games_in_charge_away",
            }
        )
        feat = home.merge(away, on="match_id", how="outer")

        intens = pd.DataFrame({"match_id": feat["match_id"]})
        intens["lam_mult_home"] = np.exp(bounce_coef * feat["new_coach_home"].fillna(0).values)
        intens["lam_mult_away"] = np.exp(bounce_coef * feat["new_coach_away"].fillna(0).values)
        intens["lam_mult_home"] = np.clip(intens["lam_mult_home"], 0.85, 1.15)
        intens["lam_mult_away"] = np.clip(intens["lam_mult_away"], 0.85, 1.15)

        n_flag = int((feat["new_coach_home"].fillna(0) + feat["new_coach_away"].fillna(0) > 0).sum())
        logger.info(
            "CoachingChangeAdjustment: {} matches with a new-coach flag | bounce_coef={}",
            n_flag,
            bounce_coef,
        )
        return AdjustmentResult(
            features=feat,
            intensity_multipliers=intens if bounce_coef != 0.0 else pd.DataFrame(),
            meta={
                "wired": True,
                "source": str(path),
                "new_coach_days": new_coach_days,
                "new_coach_games": new_coach_games,
                "bounce_coef": bounce_coef,
                "n_flagged_matches": n_flag,
            },
        )


class RefereeAdjustment(ContextAdjustment):
    """
    Referee tendencies from football-data.co.uk history (pre-match only).

    Data source
    -----------
    Match-level columns already on the aligned table:
      referee, home_yellow, away_yellow, home_red, away_red,
      home_fouls, away_fouls

    Logic (leakage-free)
    --------------------
    Sort matches by date. For each referee, compute expanding means of prior
    matches only (shift(1) then expanding, min_periods from config, default 5):
      - ref_cards_avg: mean (HY+AY+2*(HR+AR)) before this fixture
      - ref_fouls_avg: mean (HF+AF) before this fixture
      - ref_home_card_share: mean share of cards given to the home side
      - ref_games_prior: count of prior appearances in sample

    Optional intensity multipliers (YAML):
      - ``tempo_coef``: equal λ/μ bump from ref_cards_vs_league (O/U channel; measured ≈null)
      - ``card_bias_coef``: asymmetric 1X2 channel from ref_home_card_share.
        Let bias = ref_home_card_share - 0.5. Then:
          λ_home *= exp(-card_bias_coef * bias)
          λ_away *= exp(+card_bias_coef * bias)
        Interpretation: refs who historically card the home side more
        slightly suppress home attack intensity and lift away.

    Config example::
        features.context_adjustments:
          enabled: true
          referee:
            enabled: true
            min_prior_games: 5
            tempo_coef: 0.0
            card_bias_coef: 0.05
    """

    name = "referee"

    def apply(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> AdjustmentResult:
        import numpy as np

        cfg = config or {}
        min_prior = int(cfg.get("min_prior_games", 5))
        tempo_coef = float(cfg.get("tempo_coef", 0.0))
        card_bias_coef = float(cfg.get("card_bias_coef", 0.0))

        required = {"match_id", "date", "referee"}
        if not required.issubset(matches.columns):
            logger.warning("RefereeAdjustment: missing columns {}; skipping", required - set(matches.columns))
            return AdjustmentResult(meta={"wired": True, "skipped": "missing_columns"})

        card_cols = ["home_yellow", "away_yellow", "home_red", "away_red"]
        foul_cols = ["home_fouls", "away_fouls"]
        if not all(c in matches.columns for c in card_cols):
            logger.warning("RefereeAdjustment: card columns missing; skipping")
            return AdjustmentResult(meta={"wired": True, "skipped": "missing_cards"})

        df = matches.sort_values(["date", "match_id"]).copy()
        df["date"] = pd.to_datetime(df["date"])

        hy = df["home_yellow"].astype(float).fillna(0)
        ay = df["away_yellow"].astype(float).fillna(0)
        hr = df["home_red"].astype(float).fillna(0)
        ar = df["away_red"].astype(float).fillna(0)
        df["_cards"] = hy + ay + 2.0 * (hr + ar)
        df["_home_cards"] = hy + 2.0 * hr
        df["_card_share_home"] = np.where(
            df["_cards"] > 0, df["_home_cards"] / df["_cards"], np.nan
        )
        if all(c in df.columns for c in foul_cols):
            df["_fouls"] = (
                df["home_fouls"].astype(float).fillna(0) + df["away_fouls"].astype(float).fillna(0)
            )
        else:
            df["_fouls"] = np.nan

        df["_league_cards_avg"] = df["_cards"].shift(1).expanding(min_periods=min_prior).mean()

        df["ref_cards_avg"] = np.nan
        df["ref_fouls_avg"] = np.nan
        df["ref_home_card_share"] = np.nan
        df["ref_games_prior"] = 0.0

        for _, idx in df.groupby("referee", sort=False).groups.items():
            positions = list(idx)
            cards = df.loc[positions, "_cards"].values.astype(float)
            fouls = df.loc[positions, "_fouls"].values.astype(float)
            share = df.loc[positions, "_card_share_home"].values.astype(float)
            n = len(positions)
            c_avg = np.full(n, np.nan)
            f_avg = np.full(n, np.nan)
            s_avg = np.full(n, np.nan)
            prior = np.zeros(n)
            for i in range(n):
                prior[i] = float(i)
                if i < min_prior:
                    continue
                c_avg[i] = np.nanmean(cards[:i])
                f_avg[i] = np.nanmean(fouls[:i])
                s_avg[i] = np.nanmean(share[:i])
            df.loc[positions, "ref_cards_avg"] = c_avg
            df.loc[positions, "ref_fouls_avg"] = f_avg
            df.loc[positions, "ref_home_card_share"] = s_avg
            df.loc[positions, "ref_games_prior"] = prior

        df["ref_cards_vs_league"] = df["ref_cards_avg"] - df["_league_cards_avg"]
        # Centered home-card share used by asymmetric channel
        df["ref_home_card_bias"] = df["ref_home_card_share"] - 0.5

        feat = df[
            [
                "match_id",
                "ref_cards_avg",
                "ref_fouls_avg",
                "ref_home_card_share",
                "ref_home_card_bias",
                "ref_games_prior",
                "ref_cards_vs_league",
            ]
        ].copy()

        intens = pd.DataFrame({"match_id": df["match_id"].values})
        tempo_delta = df["ref_cards_vs_league"].fillna(0.0).astype(float).values
        tempo_mult = np.clip(np.exp(tempo_coef * tempo_delta), 0.85, 1.15)

        bias = df["ref_home_card_bias"].fillna(0.0).astype(float).values
        # Opposite nudges: high home-card share → suppress home λ, lift away λ
        lam_asym = np.clip(np.exp(-card_bias_coef * bias), 0.85, 1.15)
        mu_asym = np.clip(np.exp(+card_bias_coef * bias), 0.85, 1.15)

        intens["lam_mult_home"] = tempo_mult * lam_asym
        intens["lam_mult_away"] = tempo_mult * mu_asym

        use_intens = (tempo_coef != 0.0) or (card_bias_coef != 0.0)
        covered = int(feat["ref_cards_avg"].notna().sum())
        logger.info(
            "RefereeAdjustment: {}/{} with history | tempo={} card_bias={}",
            covered,
            len(feat),
            tempo_coef,
            card_bias_coef,
        )
        return AdjustmentResult(
            features=feat,
            intensity_multipliers=intens if use_intens else pd.DataFrame(),
            meta={
                "wired": True,
                "source": "football-data referee + cards/fouls",
                "min_prior_games": min_prior,
                "tempo_coef": tempo_coef,
                "card_bias_coef": card_bias_coef,
                "n_with_history": covered,
            },
        )


class TravelAdjustment(ContextAdjustment):
    """Travel distance / midweek away congestion interactions (scaffold)."""

    name = "travel"

    def apply(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> AdjustmentResult:
        empty = pd.DataFrame({"match_id": matches["match_id"]})
        empty["travel_km_away"] = float("nan")
        empty["midweek_away"] = 0.0
        return AdjustmentResult(features=empty, meta={"wired": False})


ADJUSTMENT_REGISTRY: dict[str, type[ContextAdjustment]] = {
    "injuries": InjuriesAdjustment,
    "lineups": LineupsAdjustment,
    "formations": FormationAdjustment,
    "motivation": MotivationAdjustment,
    "weather": WeatherAdjustment,
    "coaching_change": CoachingChangeAdjustment,
    "referee": RefereeAdjustment,
    "travel": TravelAdjustment,
}


def apply_context_adjustments(
    matches: pd.DataFrame,
    context_cfg: dict[str, Any] | None,
    *,
    as_of: pd.Timestamp | None = None,
) -> AdjustmentResult:
    """
    Run all enabled context adjustments and merge feature / intensity outputs.

    Config shape::
        features.context_adjustments:
          enabled: true
          injuries: {enabled: false}
          lineups: {enabled: false}
          ...
    """
    if not context_cfg or not context_cfg.get("enabled", False):
        return AdjustmentResult()

    feat_parts: list[pd.DataFrame] = []
    intens_parts: list[pd.DataFrame] = []
    meta: dict[str, Any] = {}

    for name, cls in ADJUSTMENT_REGISTRY.items():
        sub = context_cfg.get(name, {})
        if not sub.get("enabled", False):
            continue
        adj = cls()
        result = adj.apply(matches, as_of=as_of, config=sub)
        meta[name] = result.meta
        if len(result.features):
            feat_parts.append(result.features)
        if len(result.intensity_multipliers):
            intens_parts.append(result.intensity_multipliers)
        logger.info("Applied context adjustment: {}", name)

    features = _merge_on_match_id(feat_parts)
    intensity = _merge_intensity_multipliers(intens_parts)
    return AdjustmentResult(features=features, intensity_multipliers=intensity, meta=meta)


def _merge_on_match_id(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="match_id", how="outer")
    return out


def _merge_intensity_multipliers(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine intensity frames by multiplying lam_mult_* columns (default 1)."""
    if not parts:
        return pd.DataFrame()
    out = parts[0].copy()
    for col in ("lam_mult_home", "lam_mult_away"):
        if col not in out.columns:
            out[col] = 1.0
    for p in parts[1:]:
        tmp = p.copy()
        for col in ("lam_mult_home", "lam_mult_away"):
            if col not in tmp.columns:
                tmp[col] = 1.0
        out = out.merge(tmp, on="match_id", how="outer", suffixes=("", "_r"))
        out["lam_mult_home"] = out["lam_mult_home"].fillna(1.0) * out.get(
            "lam_mult_home_r", pd.Series(1.0, index=out.index)
        ).fillna(1.0)
        out["lam_mult_away"] = out["lam_mult_away"].fillna(1.0) * out.get(
            "lam_mult_away_r", pd.Series(1.0, index=out.index)
        ).fillna(1.0)
        out = out.drop(columns=[c for c in out.columns if c.endswith("_r")], errors="ignore")
    return out[["match_id", "lam_mult_home", "lam_mult_away"]]


def merge_context_into_features(
    features: pd.DataFrame,
    adjustment: AdjustmentResult,
) -> pd.DataFrame:
    """Left-join context feature columns onto the base feature matrix."""
    if adjustment.features is None or len(adjustment.features) == 0:
        return features
    cols = [c for c in adjustment.features.columns if c != "match_id"]
    if not cols:
        return features
    return features.merge(adjustment.features[["match_id"] + cols], on="match_id", how="left")
