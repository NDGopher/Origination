"""
Odds utilities: implied probabilities, vig removal, CLV, staking.

Vig removal methods:
- multiplicative (normalize implied probs)
- power (Shin-like power method for better favorite-longshot handling)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

VigMethod = Literal["multiplicative", "power", "balanced"]


def implied_probs(odds: np.ndarray) -> np.ndarray:
    """Raw implied probabilities (include overround)."""
    odds = np.asarray(odds, dtype=float)
    return 1.0 / odds


def remove_vig_multiplicative(odds: np.ndarray) -> np.ndarray:
    """Normalize implied probabilities to sum to 1."""
    raw = implied_probs(odds)
    return raw / raw.sum()


def remove_vig_power(odds: np.ndarray, tol: float = 1e-9, max_iter: int = 100) -> np.ndarray:
    """
    Power method: find k such that sum((1/odds)^k) = 1.
    Commonly used for fair probability estimation from book odds.
    """
    odds = np.asarray(odds, dtype=float)
    raw = 1.0 / odds

    # Binary search for k
    lo, hi = 0.5, 2.5
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = float(np.sum(raw**mid))
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    fair = raw**k
    return fair / fair.sum()


def remove_vig_balanced(odds: np.ndarray) -> np.ndarray:
    """
    Balanced book method: equal margin on each outcome.
    fair_i = (1/o_i) / sum(1/o_j)  — same as multiplicative for equal margin assumption.
    For unequal, we use multiplicative as approximation; alias kept for config clarity.
    """
    return remove_vig_multiplicative(odds)


def fair_probs(
    odds: np.ndarray | list[float],
    method: VigMethod = "power",
) -> np.ndarray:
    odds_arr = np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(odds_arr)) or np.any(odds_arr <= 1.0):
        return np.full_like(odds_arr, np.nan, dtype=float)
    if method == "power":
        return remove_vig_power(odds_arr)
    if method == "balanced":
        return remove_vig_balanced(odds_arr)
    return remove_vig_multiplicative(odds_arr)


def fair_1x2_from_row(
    row: pd.Series,
    method: VigMethod = "power",
    h_col: str = "close_h",
    d_col: str = "close_d",
    a_col: str = "close_a",
) -> tuple[float, float, float]:
    probs = fair_probs([row[h_col], row[d_col], row[a_col]], method=method)
    return float(probs[0]), float(probs[1]), float(probs[2])


def clv_probability(model_prob: float, close_fair_prob: float) -> float:
    """
    Probability-space CLV: model_prob - closing_fair_prob for the selected side.
    Positive means we were sharper than the close (bought a higher true price).
    """
    return float(model_prob - close_fair_prob)


def clv_odds(model_prob: float, close_odds: float) -> float:
    """
    Odds-space CLV relative to closing price:
    expected value at close if model_prob is true: model_prob * close_odds - 1.
    """
    return float(model_prob * close_odds - 1.0)


def kelly_fraction(prob: float, odds: float, fraction: float = 1.0) -> float:
    """Fractional Kelly stake as fraction of bankroll."""
    if odds <= 1.0 or prob <= 0.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - prob
    edge = (b * prob - q) / b
    return max(0.0, edge * fraction)


def apply_stake(
    prob: float,
    odds: float,
    *,
    method: str = "flat",
    unit: float = 1.0,
    kelly_fraction_mult: float = 0.25,
    max_stake: float = 5.0,
) -> float:
    if method == "fractional_kelly":
        stake = kelly_fraction(prob, odds, kelly_fraction_mult) * 100.0  # % of bankroll units
        # Interpret as units relative to unit size
        stake = min(stake, max_stake)
        return float(max(0.0, stake))
    return float(min(unit, max_stake))


def two_way_fair(over_odds: float, under_odds: float, method: VigMethod = "power") -> tuple[float, float]:
    probs = fair_probs([over_odds, under_odds], method=method)
    return float(probs[0]), float(probs[1])


def decimal_to_american(decimal_odds: float) -> int | None:
    """Convert decimal odds to American (moneyline). Returns None if invalid."""
    if decimal_odds is None or not np.isfinite(decimal_odds) or decimal_odds <= 1.0:
        return None
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100))
    return int(round(-100.0 / (decimal_odds - 1.0)))


def american_to_decimal(american: float) -> float | None:
    """Convert American odds to decimal."""
    if american is None or not np.isfinite(american) or american == 0:
        return None
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def fair_decimal_odds(prob: float) -> float | None:
    """Fair decimal odds = 1 / probability."""
    if prob is None or not np.isfinite(prob) or prob <= 0.0:
        return None
    return float(1.0 / prob)


def model_edge_vs_odds(model_prob: float, book_odds: float) -> float | None:
    """
    Edge vs a single bookmaker price (no vig removal on the single price):
        edge = model_prob - 1/book_odds
    Positive ⇒ model thinks the price is long.
    """
    if book_odds is None or not np.isfinite(book_odds) or book_odds <= 1.0:
        return None
    if model_prob is None or not np.isfinite(model_prob):
        return None
    return float(model_prob - 1.0 / book_odds)


def model_edge_vs_two_way(
    model_prob: float,
    over_odds: float,
    under_odds: float,
    *,
    side: str,
    method: VigMethod = "power",
) -> float | None:
    """
    Edge vs a two-way market after vig removal (same as backtest):
        edge = model_prob - fair_market_prob(side)
    """
    if over_odds is None or under_odds is None:
        return None
    if not (np.isfinite(over_odds) and np.isfinite(under_odds)):
        return None
    if over_odds <= 1.0 or under_odds <= 1.0:
        return None
    fo, fu = two_way_fair(float(over_odds), float(under_odds), method=method)
    fp = fo if side == "over" else fu
    return float(model_prob - fp)
