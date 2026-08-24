"""
Elite-layer interfaces: player strength, formations, hierarchical league sharing.

These protocols keep the live / walk-forward path identical while data wiring
catches up. Providers default to Null* implementations (no-op / zeros).

YAML (context adjustments)::

    features.context_adjustments:
      lineups:
        enabled: true
        provider: null          # null | registry key
      formations:
        enabled: true
        provider: null
      player_embeddings:
        enabled: false          # reserved; wired via LineupsAdjustment

YAML (hierarchical prior on Dixon–Coles)::

    model.hierarchical:
      enabled: false
      share_attack: 0.0         # pull team attack toward league mean
      share_defence: 0.0
      cross_league: false       # future: pool Big-5 attack/defence residuals
      totals_intercept: false   # iter15: fold-safe league scoring-rate offset
      totals_shrink: 0.15       # 0 = full empirical offset, 1 = none
      totals_clip: 0.12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from loguru import logger


@runtime_checkable
class PlayerStrengthProvider(Protocol):
    """Pre-match XI → attack/defence deltas (leakage-free as-of)."""

    def lineup_strength(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Return columns at minimum:
          match_id, lineup_strength_home, lineup_strength_away,
          attack_delta_home, attack_delta_away,
          defence_delta_home, defence_delta_away,
          lineup_confirmed (bool)
        """
        ...


@runtime_checkable
class FormationEncoder(Protocol):
    """Map formation labels to numeric embedding / one-hot features."""

    def encode(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Return match_id + formation_home/away (str) + embedding dims
        ``form_emb_{side}_{i}`` (float).
        """
        ...


@runtime_checkable
class HierarchicalStrengthShare(Protocol):
    """
    Share team strengths toward league / cross-league priors.

    Intended call site: after Dixon–Coles attack/defence fit, before λ build.
    """

    def shrink(
        self,
        attack: dict[str, float],
        defence: dict[str, float],
        *,
        league: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        ...


class NullPlayerStrengthProvider:
    """No player data — NaN strengths, zero deltas."""

    def lineup_strength(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        out = pd.DataFrame({"match_id": matches["match_id"].values})
        out["lineup_strength_home"] = np.nan
        out["lineup_strength_away"] = np.nan
        out["attack_delta_home"] = 0.0
        out["attack_delta_away"] = 0.0
        out["defence_delta_home"] = 0.0
        out["defence_delta_away"] = 0.0
        out["lineup_confirmed"] = False
        return out


class NullFormationEncoder:
    """No formation data — empty embeddings."""

    def encode(
        self,
        matches: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        dim = int((config or {}).get("embedding_dim", 4))
        out = pd.DataFrame({"match_id": matches["match_id"].values})
        out["formation_home"] = None
        out["formation_away"] = None
        for side in ("home", "away"):
            for i in range(dim):
                out[f"form_emb_{side}_{i}"] = 0.0
        return out


@dataclass
class LeagueMeanShrinker:
    """
    Hierarchical component: shrink team attack/defence toward league means.

    ``share_*`` in [0, 1]: 0 = no shrink (identity), 1 = all teams → mean.
    Cross-league pooling is stubbed until multi-league joint fits exist.
    """

    share_attack: float = 0.0
    share_defence: float = 0.0
    cross_league: bool = False
    _global_attack_mean: float | None = None
    _global_defence_mean: float | None = None

    def shrink(
        self,
        attack: dict[str, float],
        defence: dict[str, float],
        *,
        league: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        cfg = config or {}
        sa = float(cfg.get("share_attack", self.share_attack))
        sd = float(cfg.get("share_defence", self.share_defence))
        sa = float(np.clip(sa, 0.0, 1.0))
        sd = float(np.clip(sd, 0.0, 1.0))
        if not attack and not defence:
            return attack, defence

        a_mean = float(np.mean(list(attack.values()))) if attack else 0.0
        d_mean = float(np.mean(list(defence.values()))) if defence else 0.0
        if cfg.get("cross_league", self.cross_league) and self._global_attack_mean is not None:
            a_mean = float(self._global_attack_mean)
            d_mean = float(self._global_defence_mean or 0.0)

        atk = {k: (1.0 - sa) * v + sa * a_mean for k, v in attack.items()}
        dfn = {k: (1.0 - sd) * v + sd * d_mean for k, v in defence.items()}
        return atk, dfn

    def update_global_priors(self, attack: dict[str, float], defence: dict[str, float]) -> None:
        if attack:
            self._global_attack_mean = float(np.mean(list(attack.values())))
        if defence:
            self._global_defence_mean = float(np.mean(list(defence.values())))


PROVIDER_REGISTRY: dict[str, type] = {
    "null_players": NullPlayerStrengthProvider,
    "null_formations": NullFormationEncoder,
    "league_mean_shrink": LeagueMeanShrinker,
    "understat_squad_quality": None,  # resolved lazily
    "understat_match_players": None,  # resolved lazily
}


def resolve_player_provider(name: str | None) -> PlayerStrengthProvider:
    if not name or name in ("null", "none"):
        return NullPlayerStrengthProvider()
    if str(name) in ("understat_squad_quality", "squad_quality"):
        from origination.features.squad_quality import UnderstatSquadQualityProvider

        return UnderstatSquadQualityProvider()
    if str(name) in (
        "understat_match_players",
        "match_player_strength",
        "match_level_players",
    ):
        from origination.features.match_player_strength import MatchLevelPlayerStrengthProvider

        return MatchLevelPlayerStrengthProvider()
    cls = PROVIDER_REGISTRY.get(str(name))
    if cls is None:
        logger.warning("Unknown player provider {!r}; using null", name)
        return NullPlayerStrengthProvider()
    return cls()  # type: ignore[return-value]


def resolve_formation_encoder(name: str | None) -> FormationEncoder:
    if not name or name in ("null", "none"):
        return NullFormationEncoder()
    cls = PROVIDER_REGISTRY.get(str(name))
    if cls is None:
        logger.warning("Unknown formation encoder {!r}; using null", name)
        return NullFormationEncoder()
    return cls()  # type: ignore[return-value]


def build_hierarchical_shrinker(cfg: dict[str, Any] | None) -> LeagueMeanShrinker | None:
    hcfg = (cfg or {}).get("model", {}).get("hierarchical", {})
    if not hcfg.get("enabled", False):
        return None
    return LeagueMeanShrinker(
        share_attack=float(hcfg.get("share_attack", 0.0)),
        share_defence=float(hcfg.get("share_defence", 0.0)),
        cross_league=bool(hcfg.get("cross_league", False)),
    )
