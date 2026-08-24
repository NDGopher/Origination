# First-half / Live architecture notes

Prepared in iter13 so totals work stays compatible with later 1H and live 2H.

## Current path (full-time pre-match)

```
aligned matches → feature store (lagged) → Dixon–Coles λ/μ
  → calibration → residual → markets (1X2, OU 2.5, AH)
  → bet_filters / edge_threshold_by_market
```

Live and 1H should **reuse** the same prediction → filter → stake contract; only the intensity and state inputs change.

## First-half (1H) — data needed

| Item | Source options | Status |
|------|----------------|--------|
| 1H goals / result | Football-Data `HTHG/HTAG`, or Understat minute goals | FD columns often present; not yet first-class |
| 1H OU 0.5 / 1.5 odds | exchange / book API | not ingested |
| 1H xG cumulative | Understat shot minutes | shots cached in match JSON — usable |
| Same pre-match features | existing store | ready |

**Minimal 1H build:** treat HT scoreline as a second market family (`p_1h_*`) from a DC variant fit on HT goals or truncated Poisson, using existing lagged team strength. Shot-timeline 1H xG is the natural intensity upgrade.

## Live 2H — data needed

| Item | Source options | Status |
|------|----------------|--------|
| In-play score, minute | live feed (Betfair/Sportradar/etc.) | not wired |
| Remaining-time goal model | conditional Poisson / score matrix truncated by elapsed | design only |
| Live OU / AH / next-goal odds | same live feed | not wired |
| Red cards / injuries in-play | feed events | not wired |
| Pre-match λ prior | current model | ready as prior |

**Design sketch:** `lambda_live = f(lambda_prematch, score_state, minute, red_cards)` with leakage-free as-of = current minute. Filters stay YAML (`bet_filters`) so live can share mild universal packs.

## Code hooks to preserve

1. `evaluate_predictions` + `bet_filters.rules` — market-agnostic; add `ou15_1h` etc. later without new filter engine.
2. `intensity_adjustments` / `ctx_lam_mult_*` — live multipliers attach the same way as referee/PV.
3. `PlayerStrengthProvider` — confirmed XI matters more for live/1H; keep interface.
4. Experiment artifacts (`predictions.parquet`) — add columns, don’t replace schema.

## Out of scope for iter13

No live feed integration; no 1H backtest yet. This note is the contract for a later iteration.
