# Elite layers — player / formation / hierarchical interfaces

Leakage-free scaffolds shared by walk-forward and live prediction.

## Modules

| Module | Path | Status |
|--------|------|--------|
| Protocols + null providers | `src/origination/features/elite.py` | Ready |
| Lineups / formations adapters | `context_adjustments.py` | Calls providers when enabled |
| Hierarchical shrink on DC strengths | `elite.LeagueMeanShrinker` via `model.hierarchical` | YAML-gated, default off |

## Player embeddings / lineup strength

```yaml
features.context_adjustments.lineups:
  enabled: true
  provider: null          # or registry key implementing PlayerStrengthProvider
  strength_coef: 0.0      # λ multipliers from attack/defence deltas
```

Implement `PlayerStrengthProvider.lineup_strength(...)` with as-of timestamps only.
Register in `PROVIDER_REGISTRY`. Residual heads automatically consume numeric columns.

## Formation embeddings

```yaml
features.context_adjustments.formations:
  enabled: true
  provider: null
  embedding_dim: 4
```

## Hierarchical (cross-team / future cross-league)

```yaml
model.hierarchical:
  enabled: false
  share_attack: 0.05      # shrink DC attack toward league mean
  share_defence: 0.05
  cross_league: false     # stub until joint Big-5 fits exist
  totals_intercept: false # iter15: league-aware joint λ/μ offset
  totals_shrink: 0.15
  totals_clip: 0.12
```

Applied after Dixon–Coles MLE, before λ construction. Ablatable; measure 1X2 LL before promote.

**Iter9:** `enabled: true`, `share_attack=share_defence=0.05` promoted (EPL 1X2 LL 0.98202).

**Iter15 totals intercept:** after each fold fit, compare train mean(λ+μ) to mean(goals) and apply `exp(offset)` jointly to λ and μ (clipped). Same knob under `model.dixon_coles.totals_intercept`. Lets Bundesliga lift and La Liga dampen without a global volume coef. Measure multi-league before promote.
