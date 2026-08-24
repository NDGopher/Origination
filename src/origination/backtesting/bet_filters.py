"""YAML-configurable bet filters (underdog / longshot / market-fav / side rules)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def market_favorite_side(close_h: float, close_d: float, close_a: float) -> str | None:
    odds = {"H": float(close_h), "D": float(close_d), "A": float(close_a)}
    finite = {k: v for k, v in odds.items() if np.isfinite(v) and v > 1.0}
    if not finite:
        return None
    return min(finite, key=finite.get)


def _rule_for_market(filt: dict[str, Any], market: str) -> dict[str, Any] | None:
    """
    Resolve effective filter for a market.

    Supports either flat config (legacy) or ``rules`` list::

        bet_filters:
          enabled: true
          rules:
            - markets: [1x2]
              max_odds: 1.80
            - markets: [ou25]
              max_odds: 2.20
              block_sides: [over]
            - markets: [ah]
              max_odds: 2.10
    """
    if not filt or not filt.get("enabled", False):
        return None
    rules = filt.get("rules")
    if rules:
        for rule in rules:
            mkts = set(rule.get("markets") or rule.get("apply_markets") or [])
            if market in mkts:
                return rule
        return None
    apply = set(filt.get("apply_markets") or filt.get("markets") or ["1x2"])
    if market not in apply:
        return None
    return filt


def passes_bet_filters(
    *,
    market: str,
    side: str,
    close_odds: float,
    match: pd.Series,
    filt: dict[str, Any] | None,
) -> bool:
    """
    Return True if the candidate bet is allowed.

    See module docstring / ``_rule_for_market`` for YAML shapes.
    """
    rule = _rule_for_market(filt or {}, market)
    if rule is None:
        return True

    max_odds = rule.get("max_odds")
    if max_odds is not None and float(close_odds) > float(max_odds):
        return False

    min_odds = rule.get("min_odds")
    if min_odds is not None and float(close_odds) < float(min_odds):
        return False

    min_odds_f = rule.get("min_favorite_odds")
    if min_odds_f is not None and float(close_odds) < float(min_odds_f):
        return False

    block_sides = set(rule.get("block_sides") or [])
    if side in block_sides:
        return False

    allow_sides = rule.get("allow_sides")
    if allow_sides is not None and side not in set(allow_sides):
        return False

    if market == "1x2" and rule.get("block_draws", False) and side == "D":
        return False

    if market == "1x2" and rule.get("require_market_favorite", False):
        fav = market_favorite_side(
            float(match.get("close_h", np.nan)),
            float(match.get("close_d", np.nan)),
            float(match.get("close_a", np.nan)),
        )
        if fav is None or side != fav:
            return False

    # OU: optional "require shorter price" (bet the market favorite side of O/U)
    if market == "ou25" and rule.get("require_ou_favorite", False):
        o = float(match.get("close_over25", np.nan))
        u = float(match.get("close_under25", np.nan))
        if not (np.isfinite(o) and np.isfinite(u)):
            return False
        fav = "over" if o <= u else "under"
        if side != fav:
            return False

    return True


def passes_edge_rule(edge: float, market: str, filt: dict[str, Any] | None) -> bool:
    """Optional per-rule max_edge (used with bet_filters.rules)."""
    rule = _rule_for_market(filt or {}, market)
    if rule is None:
        return True
    max_edge = rule.get("max_edge")
    if max_edge is not None and float(edge) > float(max_edge):
        return False
    min_edge = rule.get("min_edge")
    if min_edge is not None and float(edge) < float(min_edge):
        return False
    return True
