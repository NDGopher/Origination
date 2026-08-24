# Motivation / table-position context

Source: **historical results only** (same aligned match table). No new external data.

## Features (pre-kickoff / lagged)

For each match, build the season table from **prior** fixtures only, then attach:

| Feature | Meaning |
|---------|---------|
| `table_pos_*`, `points_*`, `gd_*` | Standing before kickoff |
| `games_played_*`, `games_remaining_*` | Progress (`season_length` default 38) |
| `pts_from_top_*` | Gap to leader |
| `pts_from_safety_*` | Gap vs `safety_rank` (default 17th) |
| `title_race_*` | Late season, top-3, within `title_pts_gap` |
| `relegation_battle_*` | Late season, at/below safety, within `releg_pts_gap` |
| `euro_race_*` | Late season, positions 4–7 near 4th |
| `mid_table_dead_*` / `dead_rubber` | Secure mid-table late season |
| `stakes_*` | Continuous [0,1] blend of title/releg soft pressure |
| `must_win_*` | Hard flag for title race or late releg scramble |
| `motivation_diff` | `stakes_home - stakes_away` |

Intensity is gated by `min_games` (default 8) so early-season noise does not move λ.

## YAML intensity coefficients

```yaml
features.context_adjustments.motivation:
  enabled: true
  title_coef: 0.0          # λ *= exp(coef * title_race)
  releg_coef: 0.0          # λ *= exp(coef * releg_battle)
  dead_rubber_coef: 0.0    # λ *= exp(-coef * mid_table_dead)
  stakes_coef: 0.0         # λ *= exp(coef * stakes)
  motivation_diff_coef: 0.0  # asymmetric: home += coef*diff, away -= 
```

## Status (iter 4)

**Promoted:** `enabled: true`, `stakes_coef: 0.10` (other intensity coefs 0).  
Best walk-forward 1X2 LL: **0.98355** vs motiv-off **0.98406**.

## Ablation

`configs/ablation_motivation_off.yaml`
