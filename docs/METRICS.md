# Design notes — keep honest about what metrics mean

## Closing-line evaluation modes

### Mode A — Bet at close (current MVP backtest)
We compare model probabilities to vig-removed **closing** odds and simulate
staking at those prices.

- `avg_clv_prob` / `avg_clv_odds` = model's **claimed** edge vs the close
- These are **not** classic early-bet CLV (line move from bet → close)
- Overconfident models produce large positive claimed edge **and** negative ROI
- Primary honesty checks:
  - Walk-forward ROI at close
  - `log_loss_1x2` vs `log_loss_market_1x2` (`log_loss_edge_vs_market` > 0 ⇒ sharper)
  - Calibration (Brier / reliability diagrams — later)

### Mode B — Early bet CLV (future)
When opening / mid-market odds are available:
CLV = value of the bet relative to the eventual close
(e.g. bet at 2.20, close 2.00 ⇒ positive CLV).

football-data often has both open and close columns (B365 vs B365C, etc.).
Wire Mode B once Mode A baseline is calibrated and beating market log-loss.

## Iteration rule
Only promote a model version if it improves **out-of-sample** market log-loss
and/or walk-forward ROI at close across multiple seasons — not one lucky year.
