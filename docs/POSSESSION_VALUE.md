# Possession / On-Ball Value

Leakage-free OBV-lite from Understat match shots + roster buildup.

## v2 signal (iter12)

| Metric | Definition |
|--------|------------|
| `pv_open` | Σ xG excluding set pieces |
| `pv_obv` | `pv_open · mean(X) + 0.15·pv_buildup` (open-play primary) |
| `pv_obv_v1` | `pv_depth_w + 0.25·pv_buildup` (legacy) |
| `pv_resid_ewm` | `pv_obv_ewm − 0.35·deep_ewm` (orthogonalized vs deep) |

Post-match values are joined then **shift(1) rolled**. Intensity channel prefers `pv_resid_ewm` (center ≈ 0).

## Wiring

```yaml
features.groups.possession_value: true
features.pv_deep_orth_coef: 0.35
model.dixon_coles.intensity_adjustments:
  pv_coef: 0.08
  pv_center: 0.0
# OU-specialist residual (optional experiment):
model.residual:
  alpha_1x2: 0.0
  alpha_ou: 0.15
```

Build: `python scripts/build_possession_value.py`

## Status

- v1 (iter11): mild OU LL help; not promoted
- v2 (iter12): see `ITERATION_NOTES.md` / `possession_value_v2_comparison_iter12.csv`
