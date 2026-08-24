# Totals intensity (congestion / rest / volume / allowance)

Joint λ/μ multipliers from pre-match schedule and style features (already in the feature store).

## Schedule (promoted)

- `home_games_last_7` / `away_games_last_7`
- `home_rest_days` / `away_rest_days`

```text
cong = 0.5*(g7_h + g7_a) - 1
rest_short = 7 - 0.5*(rest_h + rest_a)
delta = congestion_coef * cong + rest_coef * (rest_short / 7)
λ *= exp(delta);  μ *= exp(delta)   # clipped to [0.85, 1.15]
```

YAML (`model.dixon_coles.intensity_adjustments`):

```yaml
congestion_coef: 0.0
rest_coef: -0.05   # iter5: short rest → fewer goals
```

Negative `rest_coef` means short-rested sides suppress totals; well-rested sides lift totals slightly.

## Volume / defensive allowance (iter14 — measure)

Uses match-level **sums** of lagged EWM features (`sum_xg_for_ewm`, `sum_xg_against_ewm`):

```text
joint *= exp(shot_volume_coef * (sum_xg_for_ewm - 2.4))
joint *= exp(xg_allow_coef * (sum_xg_against_ewm - 2.4))
```

```yaml
shot_volume_coef: 0.06    # iter17: EPL default only — Under pack + overs filter help
shot_volume_center: 2.4
xg_allow_coef: 0.06       # iter14 promote: OU LL ↑ on EPL + Bundesliga + La Liga
xg_allow_center: 2.4
```

`xg_allow` remains the portable intensity channel. `shot_volume` is **EPL-promoted** (iter17) for Under-pack strength; keep **0.0 on D1/SP1** (hurts La Liga short-band).

## League-aware totals intercept (iter15)

Fold-safe scoring-rate offset after DC fit (train mean goals vs mean λ+μ):

```text
offset = clip((1 - shrink) * log(mean_goals / mean(λ+μ)), ±clip)
λ *= exp(offset);  μ *= exp(offset)
```

```yaml
model.dixon_coles.totals_intercept:
  enabled: true
  shrink: 0.15
  clip: 0.12
  mode: signed          # signed | lift_only | dampen_only | asymmetric
  dampen_shrink: 1.0    # asymmetric: 1.0 = block dampening (Bundesliga-safe)
  min_abs_raw: 0.0      # skip tiny noisy offsets
```

**Iter15:** intercept improved short-band LL gap on EPL+D1+SP1 and OU LL on EPL+La Liga; over-damps Bundesliga mean goals — promote league-conditionally.

**Iter16:** `lift_only` / `asymmetric` block needed EPL/SP1 dampening — not universal.
Promoted: `signed` + `min_abs_raw: 0.05` on **EPL and Bundesliga**; La Liga keeps full signed (`min_abs_raw: 0`).


## Extra style channels (iter15 — measure)

```yaml
tempo_ppda_coef: 0.0       # sum_ppda_ewm
suppress_resid_coef: 0.0   # shots-against − 8·xG-against
pv_open_orth_coef: 0.0     # open-play PV ⊥ xG-for
```
