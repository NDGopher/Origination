# Squad quality / player-strength layer (v1)

## What it is

`UnderstatSquadQualityProvider` — **prior-season** Understat player aggregates
(top-N by minutes: npxG+xA, xGChain, xGBuildup), z-scored within league-season,
mapped onto matches via `PlayerStrengthProvider`.

**Not** confirmed XI strength. `lineup_confirmed=False` always in v1.
Promoted/relegated sides without prior-season top-flight data get neutral (0) deltas.

## Config

```yaml
features.context_adjustments.lineups:
  enabled: true
  provider: understat_squad_quality
  strength_coef: 0.10          # scales λ multipliers; grid carefully
  understat_league: EPL
  raw_dir: data/raw/understat
  top_n: 15
  delta_scale: 0.15
```

Intensity: `lam_mult = clip(1 + coef · (attack_delta − opp_defence_delta), 0.85, 1.15)`.

## Leakage

Match in season S uses only season S−1 player tables. No same-season player stats.

## Code

- `src/origination/features/squad_quality.py`
- Registry: `elite.resolve_player_provider("understat_squad_quality")`
- Autopsy: `scripts/run_performance_autopsy.py`
