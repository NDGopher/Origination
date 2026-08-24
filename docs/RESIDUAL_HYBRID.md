# Residual hybrid + multi-task learning

## v2 — Additive logit residual (promoted iter7)

Train corrections on OOS base **errors** in logit space:

```text
logit_final = logit_base + α · Δ̂
```

- **1X2:** 3 centered logit deltas (LightGBM regressors), then softmax
- **OU:** 1 Bernoulli logit delta, then sigmoid
- Targets: `Δ* = clip(log y_soft − log p_base)` then row-center (1X2)
- Fit only on expanding-season OOS base predictions inside the train window

```yaml
model.residual:
  enabled: true
  mode: additive          # additive | blend (v1 rejected)
  alpha_1x2: 0.10
  alpha_ou: 0.10
  interactions: false     # pairwise λ/p products (iter8 grid)
  label_smoothing: 0.02
  delta_clip: 2.0
  min_oos_rows: 300
  max_oos_seasons: 4
```

Grid winner (iter7): **α=0.10** (1X2 LL 0.98239 vs DC-only 0.98339). α=0.20 overfits.

### Iter8 enrichment

Optional `interactions: true` appends products such as `λ_h·λ_a`, `λ_h·p_h`, `p_over·λ_*`.
Optional deeper LGBM (`num_leaves=31`, `n_estimators=400`).

**Iter8 promote:** interactions **on** + deeper params (1X2 LL **0.98218**). Deeper alone without interactions hurt LL — do not use. Ablation: `configs/ablation_residual_shallow.yaml`.

## v1 — Parallel blend (rejected iter6)

`mode: blend` keeps the old probability-model blend for ablation only — do not use.
