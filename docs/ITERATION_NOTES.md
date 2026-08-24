# Iteration notes — calibration, xG intensities, Understat advanced

---

# Iteration 25 — Score 24h + e12 recommendation + new leagues

Date: 2026-08-13

**Protected rules unchanged in the live scan.**

## Delivered

1. Score Predictions: rolling **next 24h** + ranked strongest O/U leans + more leagues  
2. Scotland + Turkey ingested + walk-forward (no pack cleared the bar)  
3. Safe overlay study on 6 live systems — **no rule changes**  
4. Primeira AH e12 vs e10: **recommend promote e12 to main paper-live** (dropped e10 slice flat)  
5. Report: `experiments/iter25/REPORT.md` · Status: `docs/STATUS_BOARD.md`

---

# Iteration 24 — Live launcher + score tab + careful hunt

Date: 2026-08-13

**Protected systems unchanged.**

## Delivered

1. **`START_HERE_LIVE.bat`** — the only daily live launcher  
2. UI tab **Score Predictions** (proj score, O/U 2.5, data strength)  
3. Careful hunt on Ligue1 / Eredivisie / Belgium / Championship / Primeira  
4. **0 new systems** — Primeira AH e10/e12 reconfirmed; nothing else cleared the bar  
5. Report: `experiments/iter24/REPORT.md`

---

# Daily UX cleanup — idiot-proof 3-step UI

Date: 2026-08-12  

**Protected systems unchanged.** Usability + cleanup only.

## Delivered

1. **`Launch_Gameday.bat`** → simple UI with separate buttons: Data / Odds / Scan  
2. Freshness stamps + >24h warnings  
3. Idiot-proof `PLAYS_SIMPLE.txt` (PLAY / WATCH / ACTION)  
4. Experimental scripts moved to `scripts/archive/`  
5. Guide: `docs/DAILY_GUIDE.md`

---

# Iteration 23 — Pin AH + daily scan + deeper hunt

Date: 2026-08-12  

**Protected:** 5 live systems + Primeira AH short (e10 / max 1.90) — rules frozen.

## Delivered

1. Daily scan pipeline + decision cards (later split into explicit Data / Odds / Scan UI)  
2. Play cards: system history · Pin · fair · your-book EV  
3. **Pinnacle AH** live + historical Pin-prefer closes  
4. Primeira AH refresh: n=242, **+12.6%**, t=2.46, boot CI lo **+2.9%**, 8/10  
5. New sibling pack `PrimeiraLiga_ah_e12_exp` (not live): n=161, +19.2%, t=3.05, 10/10  
6. Status: `docs/STATUS_BOARD.md`

Master: `experiments/iter23/MASTER_REPORT.md`

---

# Iteration 22 — Multi-system gameday harden + league expansion

Date: 2026-08-12  

**Absolute rule:** the 5 live systems untouched (EPL Unders/Overs, Bundesliga Unders, La Liga Home ML, Serie A Away ML).

## Priority 1 — Gameday / UI

- Multi-league UI: **Scan All 5 Systems**, league switcher, qualified-plays panel
- Sheet: `systems_flagged`, `odds_status`, missing-odds coverage
- Docs: `docs/GAMEDAY_UI.md`, `docs/GAMEDAY_SHEET.md`, `docs/STATUS_BOARD.md`

## Priority 2 — New leagues

Ingested + WF + hunt: Ligue 1, Eredivisie, Primeira Liga, Belgium (+ Championship re-check).

| League | n | Ranking corr | Outcome |
|--------|--:|-------------:|---------|
| Ligue 1 | 4237 (76% xG) | 0.122 | Research — no promote |
| Eredivisie | 3607 | 0.147 | Research — no promote |
| Primeira | 3681 | 0.143 | **AH paper backtest** (not live) |
| Belgium | 3286 | 0.112 | Overs watch only |
| Championship | 6624 | 0.044 | Ranking still weak |

`PrimeiraLiga_ah_short_exp`: AH @ e≥10% max 1.90 — n=249, +10.2%, t=2.00, boot CI lo +0.1%, 9/10. Not in live packs (no Pin AH).

Master: `experiments/iter22/MASTER_REPORT.md` · Status: `docs/STATUS_BOARD.md`

---

# Iteration notes (historical) — calibration, xG intensities, Understat advanced

Date: 2026-08-04  
Primary ranking metric: walk-forward **market log-loss** (lower better), then claimed edge realism, then ROI @ close.

Market 1X2 log-loss (power-devig closing): **0.9486** (constant across runs).

## Baseline (prior iteration)

| Metric | Value |
|--------|-------|
| Config | goals DC, calibration=none |
| log_loss_1x2 | 0.9874 |
| log_loss_edge_vs_market | −0.0388 |
| ROI @ 3% edge | −6.9% |
| avg claimed edge (clv_prob) | +9.5% |

## Step 1 — Calibration (goals intensity)

| Method | LL | vs mkt | ROI@3% | Verdict |
|--------|-----|--------|--------|---------|
| temperature | 0.9957 | −0.0471 | −6.5% | **Best calibrator** (nearest LL; slight ROI help) |
| platt (OVR) | 1.0053 | −0.0567 | −5.2% | ROI better, LL worse — do not promote on LL |
| isotonic (OVR) | 1.0865 | −0.1379 | −6.7% | **Kill** — harms calibration badly |

Protocol fix applied mid-step: calibrator fit on holdout season OOS preds; **strengths refit on full train** before test. Without this, LL collapsed further.

**Promote:** `calibration.method: temperature` (not isotonic/platt OVR for 1X2).

## Step 2 — Intensity source (+ temperature)

| Source | LL | vs mkt | ROI@3% | Verdict |
|--------|-----|--------|--------|---------|
| xg | 0.9938 | −0.0452 | −6.3% | Improves vs temp+goals |
| blend (0.7 xG) | 0.9923 | −0.0437 | −6.2% | Slightly better than pure xG |

Both beat temperature+goals on LL and ROI. Blend edges xG.

## Step 3 — Understat advanced (PPDA / deep / npxG) + intensity multipliers

| Config | LL | vs mkt | ROI@3% | claimed edge |
|--------|-----|--------|--------|--------------|
| temp + xg + understat_adv + λ adj | **0.9856** | **−0.0370** | −7.3% | **+8.4%** |

**Best market log-loss of the iteration** — first config to beat the uncalibrated goals baseline on LL (0.9856 vs 0.9874). Claimed edge moved toward realism (9.5% → 8.4%). ROI slightly worse than blend; still deeply negative — not bettable.

### Edge threshold sweep (step 3)

| thr | n_bets | claimed edge | ROI |
|-----|--------|--------------|-----|
| 2% | 7334 | 7.6% | −6.5% |
| 3% | 6331 | 8.4% | −7.3% |
| 4% | 5480 | 9.2% | −7.4% |
| 5% | 4665 | 10.0% | −6.9% |

Raising the threshold does **not** rescue ROI — model is not sharp vs close yet; fewer bets ≠ edge.

## What improved / what did not

**Improved**
- Temperature scaling > OVR Platt/isotonic for this multiclass DC setup
- xG / blend intensity pseudo-likelihood improves LL vs goals
- Lagged PPDA/deep/npxG + mild λ multipliers → **best LL** and lower claimed edge
- Context adjustment scaffolding ready (injuries, lineups, formations, motivation, weather, coaching, referee, travel)
- Threshold sweeps logged every run; fold-safe calibration; tests expanded (20+)

**Did not improve**
- OVR isotonic/Platt (hurt LL)
- ROI still negative at all tested thresholds — **no positive EV at close yet**
- Still ~0.037 nats behind the closing market on log-loss

## Promoted default stack after iteration 1 (`configs/default.yaml` at that time)

```
model.type: dixon_coles
intensity_source: xg
calibration: temperature (holdout_seasons: 1)
features.understat_advanced: true
intensity_adjustments: enabled (ppda_coef 0.01, deep_coef 0.02)
```

---

# Iteration 2 — λ grid, blend+advanced, referee context

Date: 2026-08-04 (continued)  
Market LL benchmark: **0.9486**

## Phase 1 — λ coefficient grid (xg + understat_advanced + temperature)

Full table: `experiments/grid_comparison_iter2.csv`

| Rank | Label | LL | vs mkt | ROI@3% | claimed |
|------|-------|-----|--------|--------|---------|
| 1 | **ppda=0.01, deep=0.03** | **0.9847** | **−0.0361** | −7.1% | +8.3% |
| 2 | ppda=0.01, deep=0.02 (prior) | 0.9856 | −0.0370 | −7.3% | +8.4% |
| 3 | blend 0.6 + p005/d01 | 0.9862 | −0.0377 | −7.7% | +8.6% |
| 4 | blend 0.7 + p005/d01 | 0.9864 | −0.0378 | −8.1% | +8.6% |
| … | milder / off | worse LL | | | |
| last | λ OFF | 0.9938 | −0.0452 | −6.3% | +9.3% |

**Decisions**
- **Promote** `deep_coef: 0.03` (best LL; claimed edge slightly more realistic).
- **Do not promote** blend+advanced at tested weights — all worse LL than pure xg + λ.
- λ OFF has best ROI among grid (−6.3%) but much worse LL — keep as `configs/ablation_lambda_off.yaml` only.

## Phase 2 — Referee tendencies (first real context feature)

Implementation: `RefereeAdjustment` in `context_adjustments.py`  
Docs: `docs/REFEREE_CONTEXT.md`  
Source: football-data `referee` + yellow/red/fouls; lagged expanding means per referee (`min_prior_games=5`).

Comparison: `experiments/referee_comparison_iter2.csv` (on top of ppda=0.01, deep=0.03)

| Label | LL | ROI@3% | Notes |
|-------|-----|--------|-------|
| ref off | 0.984688 | −7.11% | baseline |
| ref features, tempo=0 | 0.984688 | −7.11% | **identical** — DC does not consume unused feature cols |
| ref + tempo=0.01 | 0.984687 | −7.11% | noise-level LL change |
| ref + tempo=0.02 | 0.984686 | −7.32% | tiny LL, worse ROI — **do not promote tempo** |

**Decisions**
- Referee feature pipeline is **real and live** (YAML-controlled, tested, leakage-free).
- Keep `tempo_coef: 0.0` — equal λ/μ tempo does not help 1X2 meaningfully.
- Enable `context_adjustments.referee.enabled: true` so features sit on the matrix for future consumers / asymmetric bias experiments.
- Ablation: `configs/ablation_referee_off.yaml`

## Promoted default after iteration 2

```
intensity_source: xg
calibration: temperature
understat_advanced: true
intensity_adjustments: ppda_coef=0.01, deep_coef=0.03
context_adjustments.enabled: true
referee.enabled: true, tempo_coef: 0.0, min_prior_games: 5
```

Best LL gap to market: **0.9847 − 0.9486 ≈ 0.0361** (was ~0.0370 after iter 1).

## Single best next action (end of iter 2)

Wire an **asymmetric** referee channel that can move 1X2 (e.g. `ref_home_card_share` → opposite λ/μ nudges with a tiny `card_bias_coef`), re-measure; if null, next context source should be **coaching-change flag**.

## Experiment folders (iter 2)

- `experiments/*_grid_*` + `grid_comparison_iter2.csv`
- `experiments/*_ref_*` + `referee_comparison_iter2.csv`

---

# Iteration 3 — asymmetric referee + coaching change

Date: 2026-08-05  
Market LL benchmark: **0.9486**

## Phase 1 — Asymmetric referee (`card_bias_coef`)

Channel: `bias = ref_home_card_share - 0.5`;  
`λ_home *= exp(-coef * bias)`, `λ_away *= exp(+coef * bias)`.

Full table: `experiments/card_bias_comparison_iter3.csv`

| coef | LL | ROI@3% | Verdict |
|------|-----|--------|---------|
| 0.0 | 0.984688 | −7.11% | prior default |
| 0.02 … 0.4 | monotonic LL ↓ | improves then flat | |
| **0.5** | **0.984155** | **−6.83%** | **best LL + better ROI** |
| −0.05 | 0.984787 | −7.06% | wrong sign — confirms direction |

**Promote:** `card_bias_coef: 0.5`. Tempo remains 0. λ ppda/deep unchanged (0.01 / 0.03).

## Phase 2 — Coaching-change flags

Source: `data/interim/coaching_changes.csv` (explicit appointment dates; see `docs/COACHING_CONTEXT.md`).  
Features: days/games in charge + `new_coach_*` (≤60 days or ≤8 games).  
Intensity: `bounce_coef` multiplies that side's λ when flagged.

On top of card_bias=0.5 — `experiments/coaching_comparison_iter3.csv`

| Label | LL | ROI@3% | Verdict |
|-------|-----|--------|---------|
| coach off / features only | 0.984155 | −6.83% | features alone do not move DC |
| bounce +0.02 | 0.984102 | −6.87% | improves LL |
| **bounce +0.05** | **0.984062** | −7.11% | **best LL** |
| bounce −0.02 | 0.984226 | −7.04% | harmful |

**Promote:** `coaching_change.enabled: true`, `bounce_coef: 0.05`.

## Promoted default after iteration 3

```
intensity_source: xg
calibration: temperature
understat_advanced: true
λ adj: ppda=0.01, deep=0.03
referee: enabled, tempo=0, card_bias_coef=0.5
coaching_change: enabled, bounce_coef=0.05, days=60, games=8
```

Best LL: **0.98406** (gap to market **≈ 0.0355**, was 0.0361).

Ablations: `configs/ablation_card_bias_off.yaml`, `configs/ablation_coaching_off.yaml`, `configs/ablation_lambda_off.yaml`.

## Single best next action

1. Optionally probe `card_bias_coef` in {0.55, 0.6} once (diminishing returns likely) **or** leave at 0.5.
2. Next context with 1X2 bite: **home/away motivation** (relegation battle / title race from table position before kickoff — fully derivable from results history, zero new data sources) **or** lineup/injury when a clean source exists.
3. Still **no GBM / FBref / Big 5** until gap shrinks further or open→close CLV appears.

## Experiment folders (iter 3)

- `experiments/*_card_bias_*` + `card_bias_comparison_iter3.csv`
- `experiments/*_coach_*` + `coaching_comparison_iter3.csv`

---

# Iteration 4 — table-position motivation + multi-market report

Date: 2026-08-05  
Market LL benchmark: **0.9486**

## Phase 1 — Motivation features (results-only table)

Implementation: `MotivationAdjustment` in `context_adjustments.py`  
Docs: `docs/MOTIVATION_CONTEXT.md`  
Source: season standings from **prior** fixtures only (points / GD / GF).  
Features: table pos, pts from top/safety, games remaining, title/releg/euro flags, mid-table dead rubber, continuous `stakes_*`, `motivation_diff`.  
Intensity coefs (YAML): `title_coef`, `releg_coef`, `dead_rubber_coef`, `stakes_coef`, `motivation_diff_coef` (gated by `min_games=8`).

Full table: `experiments/motivation_comparison_iter4.csv`

| Label | Coef | LL | vs mkt | ROI (mixed mkts @3%) | Verdict |
|-------|------|-----|--------|----------------------|---------|
| motiv_off / feat only | — | 0.984062 | −0.0355 | −6.47% | prior best |
| **stakes=0.10** | stakes | **0.983551** | **−0.0350** | −6.61% | **best LL** |
| stakes=0.05 | stakes | 0.983708 | −0.0351 | −6.80% | improves LL |
| diff=0.05 | asym | 0.983695 | −0.0351 | −6.84% | improves LL |
| title=0.05 | title | 0.983821 | −0.0352 | −6.57% | mild help |
| dead_rubber >0 | dead | ≥0.98422 | worse | | **do not promote** |
| releg=0.05 | releg | 0.984578 | worse | | harmful |
| diff=−0.05 | flip | 0.984912 | worse | | wrong sign |

**Promote:** `motivation.enabled: true`, `stakes_coef: 0.10` (other coefs 0).  
ROI still negative; promotion is on primary metric (market log-loss). Gap to market ≈ **0.0350** (was 0.0355).

Ablation: `configs/ablation_motivation_off.yaml`.

## Phase 2 — Multi-market / multi-season breakdown

Artifacts: `experiments/multi_market_iter4/`  
(`best_iter3_stack/` = motiv off; `motivation_stakes_0p10/` = promoted; plus `combined_summary.csv`)

Also copied under each experiment’s `multi_market/` folder.

Per-market rows include n_bets, ROI, avg claimed edge/CLV, hit rate, t-stat and 95% CI on per-bet ROI.

### Overall @ edge thresholds (promoted motivation stack)

| Market | thr | n | ROI | claimed edge | hit | t-stat | CI95 |
|--------|-----|---|-----|--------------|-----|--------|------|
| **1X2** | 2% | 4320 | −9.6% | +7.4% | 26.6% | −3.53 | [−15.0, −4.3]% |
| 1X2 | 3% | 3720 | −9.4% | +8.2% | 26.9% | −3.21 | [−15.2, −3.7]% |
| 1X2 | 4% | 3171 | −9.1% | +9.0% | 27.2% | −2.87 | [−15.4, −2.9]% |
| 1X2 | 5% | 2679 | −9.4% | +9.8% | 27.4% | −2.69 | [−16.2, −2.6]% |
| **OU 2.5** | 2% | 3054 | −2.9% | +7.9% | 49.1% | −1.54 | [−6.5, +0.8]% |
| OU 2.5 | 3% | 2690 | −3.8% | +8.7% | 48.4% | −1.91 | [−7.7, +0.1]% |
| OU 2.5 | 4% | 2346 | −3.5% | +9.4% | 48.3% | −1.62 | [−7.7, +0.7]% |
| OU 2.5 | 5% | 2027 | −2.9% | +10.2% | 48.5% | −1.23 | [−7.4, +1.7]% |
| **AH (main line)** | 2% | 3231 | −5.4% | +9.8% | 45.0% | −3.45 | [−8.5, −2.3]% |
| AH | 3% | 2981 | −5.6% | +10.5% | 44.9% | −3.46 | [−8.8, −2.4]% |
| AH | 4% | 2713 | −5.5% | +11.1% | 44.9% | −3.21 | [−8.8, −2.1]% |
| AH | 5% | 2437 | −4.0% | +11.9% | 45.6% | −2.22 | [−7.5, −0.5]% |

Baseline (motiv off) is similar: 1X2 ≈ −9%, OU ≈ −3 to −4.5%, AH ≈ −5%. Motivation improves LL; OU ROI slightly better at 2–3% thr; 1X2 ROI slightly worse.

**No overall market×threshold cell with positive ROI and n≥30.** Highest-edge buckets do not rescue 1X2/AH. OU **8–12% edge** bucket @3% is roughly flat/slightly positive (+1.4% to +1.9% ROI) — noisy, not actionable alone.

### Positive pockets (season × market)

clearest positive expectancy pockets (both models):

| Season | Market | Notes |
|--------|--------|-------|
| **2021** | **OU 2.5** | Strongest: ROI ~+10–16% across thresholds (n≈150–300); t-stat often >1.5 |
| 2018 | AH / 1X2@5% | Mild positive; CI wide |
| 2017 / 2023 / 2025 | AH or OU | Small positive slices |

Full tables: `positive_roi_seasons.csv` under each model folder.

### First-half markets

HT goals / HTR are available; **no HT closing odds** in the football-data EPL extract → no ROI/CLV for FH 1X2 or FH O/U.

Model HT calibration (λ_ht ≈ 0.45 · λ_ft approximation), motivation stack:

| Metric | Value |
|--------|-------|
| n | 3800 |
| HT 1X2 log-loss | ≈ 1.040 |
| HT 1X2 Brier | ≈ 0.630 |
| Favorite hit rate | ≈ 44.1% |
| HT O/U 1.5 log-loss | ≈ 0.654 |

## Promoted default after iteration 4

```
intensity_source: xg
calibration: temperature
understat_advanced: true
λ adj: ppda=0.01, deep=0.03
referee: card_bias_coef=0.5, tempo=0
coaching_change: bounce_coef=0.05
motivation: enabled, stakes_coef=0.10
backtest.markets: [1x2, ou25, ah]
```

Best LL: **0.98355** (gap ≈ **0.0350**).

## Single best next action

**OU-focused sharpening on the promoted stack** (still ranked by 1X2 LL for any global change): holdout-calibrate O/U 2.5 separately (binary temperature/Platt on over/under only), and/or add a totals-specific intensity knob (e.g. schedule congestion → joint λ/μ) aimed at the near-breakeven OU book. Multi-market evidence says totals is the only market with CI near zero and a clear positive season (2021). Keep 1X2 as primary; do **not** expand to GBM / FBref / Big 5 yet.

## Experiment folders (iter 4)

- `experiments/*_motiv_*` + `motivation_comparison_iter4.csv`
- `experiments/multi_market_iter4/{best_iter3_stack,motivation_stakes_0p10}/*.csv`
- `experiments/multi_market_iter4/combined_summary.csv`

---

# Iteration 5 — OU calibration + rest/congestion totals channel

Date: 2026-08-05  
Market 1X2 LL benchmark: **0.9486** | Market OU LL: **0.6745**

## Phase 1 — Independent O/U holdout calibration

Config: `calibration.ou_method: none | temperature | platt` (does not alter 1X2).  
Fit on same holdout season OOS preds as 1X2 temperature.

| Config | 1X2 LL | OU LL | OU gap | OU ROI@3% | claimed OU edge | Verdict |
|--------|--------|-------|--------|-----------|-----------------|---------|
| baseline (ou none) | 0.983551 | 0.68595 | −0.0114 | −3.81% | +8.7% | prior |
| **ou temperature** | 0.983551 | **0.68350** | **−0.0090** | −5.14% | **+7.8%** | **best OU LL**; 1X2 unchanged |
| ou platt | 0.983551 | 0.68713 | −0.0126 | −6.35% | +8.4% | **kill** — worse OU LL + ROI |

OU temperature improves probability quality and claimed-edge realism; flat-stake ROI worsens (fewer overconfident bets). Promote on secondary probability metric without harming primary 1X2 LL.

## Phase 2 — Totals intensity (joint λ & μ)

Channel: `totals_delta = congestion_coef * (mean(games_last_7)−1) + rest_coef * ((7−mean(rest_days))/7)`; both λ,μ *= exp(delta).

Full table: `experiments/ou_congestion_comparison_iter5.csv`

| Config | 1X2 LL | OU LL | OU ROI@3% | Verdict |
|--------|--------|-------|-----------|---------|
| baseline | 0.983551 | 0.68595 | −3.81% | |
| **rest=−0.05** | **0.983388** | 0.68692 | −3.54% | **best 1X2 LL**; mild OU ROI help |
| cong=−0.05 | 0.983503 | 0.68719 | −3.27% | slight 1X2 help; OU LL worse |
| cong=+0.05 | 0.983724 | 0.68623 | **−3.18%** | best OU ROI in grid; worse 1X2 |
| cong / rest |±0.10| | worse | worse | | too strong |
| combo ou_temp+cong+0.05 | 0.983724 | 0.68401 | −4.40% | script auto-combo; not best |

**Promote rest_coef=−0.05** (short rest → lower totals). Do **not** promote congestion_coef (no clean OU-LL win at signs that also help 1X2).

## Phase 3 — Promoted combo

`ou_method=temperature` + `rest_coef=−0.05` (+ all prior context):

| Metric | Baseline (iter4) | Promoted combo |
|--------|------------------|----------------|
| 1X2 LL | 0.98355 | **0.98339** |
| 1X2 gap | 0.0350 | **0.0348** |
| OU LL | 0.68595 | **0.68358** |
| OU gap | 0.0114 | **0.0091** |
| OU ROI@3% | −3.81% | −4.94% |
| OU claimed edge@3% | +8.7% | +8.0% |

### Multi-market @ 3% (promoted)

| Market | n | ROI | claimed | t-stat | CI95 |
|--------|---|-----|---------|--------|------|
| 1X2 | 3699 | −9.3% | +8.2% | −3.14 | [−15.0, −3.5]% |
| OU 2.5 | 2597 | −4.9% | +8.0% | −2.27 | [−9.2, −0.7]% |
| AH | 2989 | −5.4% | +10.5% | −3.33 | [−8.6, −2.2]% |

Positive pockets remain: **2021 OU** (still +), 2018 AH/1X2@5%, OU **8–12% edge** bucket ≈ flat/+2%.  
No overall market×threshold cell with positive ROI and n≥30.

Honest read: we got **closer on probability** (esp. OU), not yet to positive expectancy. OU ROI dipped because temperature shrinks overconfident edges; LL says those edges were miscalibrated.

## Promoted default after iteration 5

```
intensity_source: xg
calibration: temperature + ou_method: temperature
understat_advanced: true
λ adj: ppda=0.01, deep=0.03, rest_coef=-0.05, congestion_coef=0
referee: card_bias=0.5
coaching: bounce=0.05
motivation: stakes=0.10
```

Ablation: `configs/ablation_ou_rest_off.yaml`.

## Single best next action

**Open→close CLV audit** on football-data opening odds (`B365H`/`PSH`/etc. vs `close_*`, coverage ~100%). Measure whether the model is ahead of the open but behind the close. That tells us if the remaining gap is “late market info we lack” vs “fundamentally weak probabilities.” Still no GBM / FBref / Big 5.

## Experiment folders (iter 5)

- `experiments/*_iter5_*` + `ou_congestion_comparison_iter5.csv`
- `experiments/multi_market_iter5/{baseline,ou_temp,rest_m0p05,combo_*}/*.csv`
- `experiments/multi_market_iter5/combined_summary.csv`

---

# Iteration 6 — Open→close audit + residual hybrid / multi-task

Date: 2026-08-05  
Promoted stack (unchanged): iter5 combo — 1X2 LL **0.98339**, OU LL **0.68358**.

## Phase 1 — Open → close CLV audit

Artifacts: `experiments/open_close_audit_iter6/`  
Model: `20260805T173941Z_iter5_combo_ou_temp_rest_m0p05` predictions.

Coverage: open 1X2 / OU / AH ≈ 100% on the WF set (open ≠ close; mean |Δodds| home ≈ 0.21).

### Log-loss gaps (model − market; positive = model worse)

| Market | vs open | vs close | Close − open gap |
|--------|---------|----------|------------------|
| **1X2** | **+0.0314** | **+0.0348** | +0.0034 |
| **OU 2.5** | +0.0086 | +0.0091 | +0.0005 |

Market LL: open 1X2 **0.9520** → close **0.9486** (close is sharper).  
Model LL fixed at 0.9834 — already **behind the open** by ~3.1 nats; late move explains only ~10% of the close gap.

### Betting @ 3% edge

| Market | At open ROI | At close ROI |
|--------|-------------|--------------|
| 1X2 | −7.6% | −9.3% |
| OU 2.5 | −4.1% | −4.9% |
| AH | −5.4% | −5.4%* |

\*AH open/close books largely overlap in current attach logic for many rows.

Open-selected bets settled at close: slightly **worse** ROI than at open (edge decay tiny, ~0.0003–0.0007 in fair-prob). Line move mean |Δp| ≈ 1.7% (1X2), 1.5% (OU).

### Interpretation

**Mostly fundamental weakness**, with a smaller late-info component.  
The model is not “beating the open and losing to the close.” Stronger base probabilities are required; closing-only features will not close most of the 0.035 gap.

## Phase 2 — Residual hybrid + multi-task (v1)

Implementation: `src/origination/models/residual.py`  
Docs: `docs/RESIDUAL_HYBRID.md`  
Walk-forward: expanding OOS base preds inside train → LightGBM heads (1X2 multiclass + OU binary) → log-linear blend with α. YAML: `model.residual.*`.

Comparison: `experiments/residual_comparison_iter6.csv`

| Config | 1X2 LL | OU LL | OU ROI@3% | Verdict |
|--------|--------|-------|-----------|---------|
| **dc_only (baseline)** | **0.98339** | **0.68358** | −4.94% | keep |
| resid α=0.25 | 0.98739 | 0.68595 | **−2.26%** | LL worse; OU ROI better |
| resid α=0.35 | 0.99682 | 0.69072 | −2.21% | worse |
| resid 1X2-only α=0.35 | 0.99682 | 0.68358 | −4.94% | 1X2 worse |
| resid α=0.50 | 1.01820 | 0.70166 | −2.24% | worst LL |

**Do not promote.** Parallel GBM blend **hurts** primary 1X2 LL at all tested α. Soft note: multi-task OU head improves OU flat ROI (~−2.2%) while damaging OU LL — not a promote under ranking rules.

Architecture remains YAML-gated (`enabled: false`) for the next residual formulation.

### Multi-market @ 3% (dc_only vs mild residual)

| Model | 1X2 ROI | OU ROI | AH ROI | 1X2 LL |
|-------|---------|--------|--------|--------|
| dc_only | −9.3% | −4.9% | −5.4% | **0.98339** |
| resid α=0.25 | −7.4% | −2.3% | −5.4% | 0.98739 (worse) |

Still no overall positive expectancy.

## Promoted default after iteration 6

**Unchanged from iter5** (residual off):

```
calibration: temperature + ou_method: temperature
rest_coef: -0.05
motivation stakes: 0.10
referee card_bias: 0.5
coaching bounce: 0.05
residual.enabled: false
```

## Single best next action

**Reformulate residual as additive logit corrections** (train Δlogit = f(features) on OOS base errors; `logit_final = logit_base + α·Δ` with tiny α ∈ {0.05, 0.10, 0.15}), not a second probability model. The open→close audit says the core DC distribution is the bottleneck — a parallel GBM rewrite fights the base instead of nudging it. Keep multi-task OU head in that additive setup. Still no Big 5 / FBref progressive.

## Experiment folders (iter 6)

- `experiments/open_close_audit_iter6/`
- `experiments/*_iter6_*` + `residual_comparison_iter6.csv`
- `experiments/multi_market_iter6/`

---

# Iteration 7 — Additive logit residual + multi-league

Date: 2026-08-05  
Prior stack (iter6 / residual off): 1X2 LL **0.98339**, OU LL **0.68358**.  
Market 1X2 LL (EPL close): **0.9486**.

## Phase 1 — Additive logit residual (v2)

Formulation: `logit_final = logit_base + α · Δ̂`, with Δ̂ trained on OOS base **errors** (not a parallel probability model). Docs: `docs/RESIDUAL_HYBRID.md`.

Full grid: `experiments/additive_residual_comparison_iter7.csv`

| Config | α 1X2/OU | 1X2 LL | vs mkt | OU LL | OU ROI@3% | Verdict |
|--------|----------|--------|--------|-------|-----------|---------|
| **add α=0.10 multi** | **0.10 / 0.10** | **0.98239** | **−0.0338** | **0.68337** | **−3.89%** | **Promote** |
| add α=0.05 multi | 0.05 / 0.05 | 0.98244 | −0.0338 | 0.68333 | −4.40% | close 2nd |
| add 1X2-only α=0.10 | 0.10 / 0 | 0.98239 | −0.0338 | 0.68358 | −4.94% | same 1X2; OU worse |
| add α=0.15 multi | 0.15 / 0.15 | 0.98322 | −0.0346 | 0.68370 | −3.54% | mild vs DC; worse than 0.10 |
| dc_only | off | 0.98339 | −0.0348 | 0.68358 | −4.94% | baseline |
| add α=0.20 multi | 0.20 / 0.20 | 0.98489 | −0.0363 | 0.68431 | −3.06% | **Reject** — overfits |

**Decisions**
- **Promote** additive residual `mode: additive`, `alpha_1x2=0.10`, `alpha_ou=0.10` (multi-task).
- Gain is real but small (~0.001 nats on 1X2; OU LL + OU ROI both improve vs DC-only).
- α=0.20 confirms the “keep α small” rule — larger correction fights the base again.
- Ablation: `configs/ablation_residual_off.yaml`.

### Multi-market @ 3% (EPL)

| Model | 1X2 ROI | OU ROI | AH ROI | 1X2 LL |
|-------|---------|--------|--------|--------|
| dc_only | −9.3% | −4.9% | −5.4% | 0.98339 |
| **add α=0.10** | **−9.1%** | **−3.9%** | −5.4% | **0.98239** |

Still no positive EV at close. Residual helps probability quality and OU flat slightly; AH unchanged (no AH residual head).

## Phase 2 — Multi-league readiness + first expansion

Pipeline: league-specific YAML + `data.align.output` → per-league `matches_aligned_*.parquet` → `scripts/run_league_backtest.py`.  
Configs: `configs/league_E1_championship.yaml`, `configs/league_D1_bundesliga.yaml`.  
Artifacts: `experiments/multi_league_iter7/` (+ `combined_summary.csv`).

| League | Intensity | n_pred | 1X2 LL | Mkt LL | Gap | OU LL | OU mkt | OU gap | 1X2 ROI@3% | OU ROI@3% | AH ROI@3% |
|--------|-----------|--------|--------|--------|-----|-------|--------|--------|------------|-----------|-----------|
| **EPL** (promoted) | xG+λ | 4180* | **0.98239** | 0.9486 | **−0.0338** | 0.68337 | ~0.6745 | −0.0089 | −9.1% | −3.9% | −5.4% |
| **Championship (E1)** | goals | 6072 | 1.06693 | 1.03216 | −0.0348 | 0.69415 | 0.68577 | −0.0084 | −3.5% | −5.5% | −2.2% |
| **Bundesliga (D1)** | goals† | 3060 | 1.00246 | 0.97803 | **−0.0244** | 0.67576 | 0.65187 | −0.0239 | **+0.35%‡** | −10.9% | −3.2% |

\*EPL WF n varies by experiment folder; LL from residual run.  
†Understat xG join ~19% on D1 (team-name mismatch) → forced goals path. Referee column absent → card bias skipped.  
‡t-stat ≈ 0.10 — not significant; do not claim edge.

### Multi-league takeaways

- Ingest + walk-forward + multi-market reports work cleanly for football-data leagues with closing odds.
- **1X2 gap vs market** on Championship ≈ EPL (~0.035); Bundesliga closer (~0.024) on goals-only DC.
- **OU** is healthier on EPL/E1 (~0.008–0.009 behind market) than on D1 (~0.024) — D1 needs xG/PPDA parity, not more residual.
- Flat 1X2 ROI less negative outside EPL, but still no reliable positive expectancy.

## Promoted default after iteration 7

```
# prior iter5/6 stack retained, plus:
model.residual:
  enabled: true
  mode: additive
  alpha_1x2: 0.10
  alpha_ou: 0.10
```

## Single best next action

**Fix Understat team-name mapping for Big 5 (start with Bundesliga)** so non-EPL leagues can use the same xG + PPDA/deep λ path as EPL. Residual already delivered its cheap ~0.001 LL gain; the remaining ~0.034 EPL gap (and D1 OU gap) is still fundamental intensity quality. Then re-run D1 (and add Serie A or La Liga) under the promoted residual stack for a fair cross-league efficiency table.

Do **not** prioritize next: more residual α grids, player embeddings, or FBref progressive — those come after multi-league xG is honest.

## Experiment folders (iter 7)

- `experiments/*_iter7_*` + `additive_residual_comparison_iter7.csv`
- `experiments/multi_market_iter7/`
- `experiments/multi_league_iter7/` (+ `*_league_E1_champ`, `*_league_D1_bundesliga`)

---

# Iteration 8 — Big 5 Understat mapping + residual enrichment + elite interfaces

Date: 2026-08-05  
Prior promoted (iter7): additive residual α=0.10 → EPL 1X2 LL **0.98239**, gap **−0.0338**.

## Phase 1 — Understat team-name mapping (Big 5)

Root cause: `_TEAM_ALIASES` was EPL-only; join is exact on canonical names → D1 ~19%.

Expanded `src/origination/utils/team_names.py` with Bundesliga / Serie A / La Liga aliases (FD + Understat strings → one canonical). Tests: `tests/test_team_names.py`.

| League | Understat join (aligned) | PPDA advanced join |
|--------|--------------------------|--------------------|
| **Bundesliga (D1)** | **99.97%** (3671/3672) | 99.9% |
| **Serie A (I1)** | **99.71%** | 99.7% |
| **La Liga (SP1)** | **98.4%** | 98.4% |

Configs: `league_D1_bundesliga.yaml` (xG stack re-enabled), `league_I1_serie_a.yaml`, `league_SP1_la_liga.yaml`.

### Fair multi-league comparison (full stack: xG + PPDA/deep + residual α=0.10)*

`experiments/multi_league_iter8/comparison_table.csv`

| League | Intensity | 1X2 LL | vs mkt | OU gap | ROI@3% 1X2 | OU | AH |
|--------|-----------|--------|--------|--------|------------|-----|-----|
| **EPL**† | xg+PPDA | **0.98218** | **−0.0336** | −0.0086 | −9.3% | −4.0% | −5.4% |
| Championship | goals | 1.06693 | −0.0348 | −0.0084 | −3.5% | −5.5% | −2.2% |
| **Bundesliga** | xg+PPDA | 0.99731 | **−0.0193** | −0.0219 | −3.9% | −9.8% | −4.5% |
| **Serie A** | xg+PPDA | 0.97396 | −0.0286 | −0.0153 | −10.4% | −8.4% | −3.9% |
| **La Liga** | xg+PPDA | 0.98425 | **−0.0207** | −0.0157 | −4.0% | −6.9% | −0.6% |

\*D1/I1/SP1/E1 measured with α=0.10 residual (pre-interactions). League YAMLs updated to iter8 residual for future runs.  
†EPL row = promoted **interactions + deeper** residual (below).

**Bundesliga before→after mapping:** goals-only gap −0.0244 → xG+PPDA gap **−0.0193** (fairer, stronger). OU still the weak D1 channel.

## Phase 2 — Residual enrichment (EPL)

Grid: `experiments/residual_enrichment_iter8.csv`

| Config | 1X2 LL | OU LL | ROI | Verdict |
|--------|--------|-------|-----|---------|
| **ix + deeper** | **0.98218** | **0.68313** | −6.5% | **Promote** |
| interactions only | 0.98226 | 0.68324 | −5.9% | close 2nd |
| ix + α=0.08 | 0.98230 | 0.68322 | −6.4% | mild |
| baseline α=0.10 | 0.98263 | 0.68337 | −6.5% | prior |
| deeper only | 0.98279 | 0.68296 | −6.7% | reject (1X2 worse) |

**Promote:** `interactions: true` + deeper LGBM (`n_estimators=400`, `num_leaves=31`, `lr=0.04`, `reg_lambda=1.5`), keep α=0.10.  
Ablation: `configs/ablation_residual_shallow.yaml`. Deeper *without* interactions overfits.

## Phase 3 — Elite architecture readiness

Docs: `docs/ELITE_LAYERS.md`  
Code: `src/origination/features/elite.py`

| Layer | Interface | Wiring | Default |
|-------|-----------|--------|---------|
| Player / lineup strength | `PlayerStrengthProvider` | `LineupsAdjustment` | null (off) |
| Formation embeddings | `FormationEncoder` | `FormationAdjustment` | null (off) |
| Hierarchical DC shrink | `LeagueMeanShrinker` | `model.hierarchical` after MLE | enabled: false |

Same path for walk-forward and live. YAML-gated; no leakage until real as-of providers are registered.

## Promoted default after iteration 8

```
# prior stack + 
model.residual.interactions: true
model.residual.params: n_estimators=400, num_leaves=31, lr=0.04, reg_lambda=1.5
model.hierarchical.enabled: false
# Big 5 aliases in team_names.py
```

Best EPL 1X2 LL: **0.98218** (gap **−0.0336** vs market 0.9486).

## Single best next action

**Wire a real player-strength provider** (minutes-weighted / contribution embeddings from historical XIs — Understat/FBref player match data) behind `PlayerStrengthProvider`, enable `lineups` with a tiny `strength_coef` grid, and measure EPL 1X2 LL. The open→close audit and residual ceilings say the remaining ~0.034 gap is still fundamental team-strength resolution; player-level is the syndicate-grade lever. Parallel: optional mild `hierarchical.share_*` grid on Big 5 (now that xG joins are honest).

Do **not** prioritize: more residual α grids, FBref progressive without player wiring, or Ligue 1 until player layer is measured on EPL.

## Experiment folders (iter 8)

- `experiments/residual_enrichment_iter8.csv` + `*_iter8_resid_*`
- `experiments/multi_league_iter8/` (+ `comparison_table.csv`)
- `docs/ELITE_LAYERS.md`, `docs/RESIDUAL_HYBRID.md` (updated)

---

# Iteration 9 — Deep autopsy + squad quality + hierarchical

Date: 2026-08-05  
Prior (iter8): EPL 1X2 LL **0.98218**, gap **−0.0336**.

## Part 1 — Deep performance autopsy

Artifacts: `experiments/autopsy_iter9/` (per-league tables + `ROLLUP.md` + `SUMMARY.md`).  
Segments: odds buckets, fav/dog, home/away/draw, over/under, edge buckets, residual-size (EPL), n / hit-rate / ROI / avg odds / avg edge / t-stat.

### Overall @ 3% edge (promoted iter8 residual stack)

| League | Market | n | Hit% | ROI | Avg odds | t |
|--------|--------|---|------|-----|----------|---|
| EPL | 1X2 | 3530 | 28.6 | **−9.3%** | 4.38 | −3.18 |
| EPL | OU | 2599 | 45.1 | −4.0% | 2.21 | −1.86 |
| EPL | AH | 2989 | 48.4 | −5.4% | 1.96 | −3.33 |
| Championship | 1X2 | 6097 | 27.6 | −3.5% | 4.03 | −1.61 |
| Bundesliga | 1X2 | 2976 | 27.2 | −3.9% | 4.33 | −1.16 |
| Serie A | 1X2 | 3206 | 32.8 | **−10.4%** | 3.81 | −3.72 |
| La Liga | 1X2 | 3255 | 32.1 | −4.0% | 4.00 | −1.35 |
| La Liga | AH | 2813 | 50.8 | **−0.6%** | 1.96 | −0.39 |

### Biggest ROI drags (cross-league, n≥50)

1. **1X2 underdogs / long prices** — EPL mild dogs 2.70–4.00: n=882, hit 26.3%, ROI **−13.1%**; big dogs 4.00+: n=1670, hit 16.0%, ROI −10.1%. Betting *against* the market favorite: n=2505, ROI **−11.5%**.
2. **Serie A home 1X2 & underdogs** — home n=1328 ROI −18.3%; underdog n=1927 ROI −17.6%.
3. **Bundesliga OU underdogs** — OU underdog n=502 ROI **−23.2%**; mild/big dog OU prices similarly toxic.
4. **Large residual corrections (EPL)** — medium/large resid L1 1X2 ROI ~−10–11% vs small ~−6.7%; OU small residual near flat (+0.9%).

### Closest to breakeven / positive (n≥50, mostly NS)

- La Liga AH overall ≈ −0.6% (closest league-market to flat).
- La Liga 1X2 on market favorites / big favs / pick’ems: small positive ROI (t≈0.7–1.4).
- Bundesliga 1X2 high-edge (8–12%, 12%+): +9–11% ROI (t≈0.8–1.3) — **not significant**, investigate don’t bank.
- EPL OU mild dogs / OU underdogs: +4–5% (t≈0.5).
- EPL 1X2 big favorites (<1.50): ≈ flat (−0.4%).

### Failure-mode summary

The book is not losing because of one broken market — **1X2 underdog hunting** is the primary bleed (high claimed edge, long odds, low hit rate). Favorites are closer to fair. OU is healthier than 1X2 on EPL; AH is the least-bad on La Liga. Residual *size* correlating with worse ROI suggests large corrections still often point the wrong way on 1X2.

## Part 2 — Elite layers measured

### A. Squad quality (`UnderstatSquadQualityProvider`)

Prior-season top-15 minutes players (npxG+xA / xGChain / buildup), z-scored, λ via `strength_coef`. Docs: `docs/SQUAD_QUALITY.md`. YAML-gated; leakage-free (season S uses S−1 only).

| Config | 1X2 LL | OU LL | ROI |
|--------|--------|-------|-----|
| baseline | 0.98218 | 0.68313 | −6.51% |
| **squad coef=0.05** | **0.98212** | 0.68308 | −6.58% |
| squad 0.10 | 0.98261 | 0.68289 | −6.67% |
| squad 0.15 | 0.98251 | 0.68305 | −7.00% |

Mild LL help at 0.05; **ROI not improved**; higher coefs hurt. **Do not promote** as default (available, off).

### B. Hierarchical shrink (`share_attack=share_defence=0.05`)

| Config | 1X2 LL | OU LL | ROI | 1X2 ROI@3% |
|--------|--------|-------|-----|------------|
| baseline | 0.98218 | 0.68313 | −6.51% | −9.28% |
| **hier 0.05** | **0.98202** | **0.68304** | **−6.07%** | **−8.41%** |
| squad0.05+hier | 0.98230 | 0.68310 | −6.38% | — |
| squad0.10+hier | 0.98257 | 0.68323 | −6.77% | — |

**Promote hierarchical alone.** Stacking with squad quality fights the shrink. Ablation: `configs/ablation_hierarchical_off.yaml`.

## Promoted default after iteration 9

```
# iter8 residual retained, plus:
model.hierarchical:
  enabled: true
  share_attack: 0.05
  share_defence: 0.05
# lineups: understat_squad_quality wired but enabled: false
```

Best EPL 1X2 LL: **0.98202** (gap **−0.0334**).

## Multi-league table (unchanged vs iter8 xG stack; EPL row = hier promote)

| League | 1X2 gap | OU gap | Notes |
|--------|---------|--------|-------|
| EPL (hier) | −0.0334 | −0.0085 | promoted |
| Championship | −0.0348 | −0.0084 | goals-only |
| Bundesliga | −0.0193 | −0.0219 | fair xG |
| Serie A | −0.0286 | −0.0153 | fair xG; worst 1X2 ROI |
| La Liga | −0.0207 | −0.0157 | best AH health |

## Single best next action

**Underdog / long-shot filter + match-level player minutes (true XI).** Autopsy says 1X2 dogs are the cash drain — add a YAML bet filter (e.g. block 1X2 when close odds > 2.7 or side ≠ market fav) and re-measure ROI *and* LL on remaining bets. In parallel, ingest Understat match player data keyed by `understat_id` so `PlayerStrengthProvider` becomes real lineup strength, not prior-season squad quality. Formations after match-level players.

## Experiment folders (iter 9)

- `experiments/autopsy_iter9/` (+ `ROLLUP.md`)
- `experiments/player_strength_comparison_iter9.csv` + `multi_market_iter9/`
- `docs/SQUAD_QUALITY.md`

---

# Iteration 10 — Underdog filter + match-level player strength

Date: 2026-08-05  
Prior (iter9): EPL 1X2 LL **0.98202**, 1X2 ROI@3% **−8.41%**, gap **−0.0334**. Hierarchical on.

## Part 1 — Underdog / long-shot bet filter

YAML `backtest.bet_filters` wired into `evaluate_predictions` (`bet_filters.py`). Re-scored saved hier predictions (no re-fit).

### EPL @ 3% edge (source: `20260805T205243Z_iter9_hier_s0p05`)

| Filter | n_1X2 | Hit% | ROI | Avg odds | Port LL |
|--------|------:|-----:|----:|---------:|--------:|
| baseline (none) | 3532 | 28.9 | **−8.41%** | 4.39 | 0.994 |
| max_odds 2.70 | 988 | 52.2 | −3.70% | 1.95 | 1.014 |
| max_odds 2.00 | 525 | 62.5 | −2.22% | 1.60 | 0.921 |
| **max_odds 1.80** | **401** | **66.8** | **−0.43%** | **1.51** | **0.868** |
| market_fav_only | 1026 | 51.8 | −3.20% | 1.98 | 1.016 |
| fav + max 2.70 | 976 | 52.4 | −3.89% | 1.94 | 1.013 |

Full-model 1X2 LL unchanged (filter is book selection only). Portfolio LL on short prices is healthier (0.868). t-stat on max1.80 1X2 ≈ **−0.12** (statistically flat).

### Multi-league 1X2 filter (same edge 3%)

| League | raw ROI | max1.80 | max2.00 | fav-only |
|--------|--------:|--------:|--------:|---------:|
| EPL | −8.4% | **−0.4%** (n=401) | −2.2% | −3.2% |
| Championship | −3.5% | −12.4% (n=96, thin) | −9.6% | −3.0% |
| Bundesliga | −3.9% | **+0.2%** (n=153) | −2.3% | −9.7% |
| Serie A | −10.4% | −1.5% | **+1.6%** (n=712) | −1.2% |
| La Liga | −4.0% | **+3.1%** (n=387) | **+3.1%** | **+3.1%** |

**Promote EPL default:** `bet_filters.enabled: true`, `max_odds: 1.80`, `apply_markets: [1x2]`.  
La Liga / Serie A look **positive** under short-price filters — first clear near-+EV pockets. Championship needs a different rule (fav-only or leave raw; n too small at 1.80).

Autopsy: `experiments/autopsy_iter10_filters/` — after max1.80, remaining book drag is mostly **AH (−5.3%)** and **OU (−3.8%)**, not 1X2.

## Part 2 — Match-level player strength

Ingested Understat `getMatchData/{id}` for **4534/4534** EPL matches → **128,857** appearances  
(`data/interim/understat_match_rosters.parquet`). Provider: `understat_match_players`  
(docs: `docs/MATCH_PLAYER_STRENGTH.md`). Leakage-free expected XI = prior starters + expanding prior (xG+xA)/90.

| Config | 1X2 LL | OU LL | ROI all | 1X2 ROI@3% |
|--------|-------:|------:|--------:|-----------:|
| hier baseline | **0.98202** | 0.68304 | −6.07% | −8.41% |
| match coef=0.03 | 0.98241 | 0.68290 | −6.31% | −8.49% |
| match coef=0.05 | 0.98236 | 0.68313 | −6.31% | −8.72% |
| match coef=0.08 | 0.98260 | 0.68311 | −6.90% | −9.99% |
| match0.05 + fav/max2.7 | 0.98236 | 0.68313 | −4.68% | −4.16% |

**Do not promote** match-level λ adjustment — same pattern as prior-season squad quality: noise / double-counting with DC+residual. Layer stays wired (`enabled: false`). Confirmed pre-match XI still missing.

## Promoted default after iteration 10

```
model.hierarchical: enabled, share=0.05   # retained
backtest.bet_filters:
  enabled: true
  apply_markets: [1x2]
  max_odds: 1.80
# lineups.understat_match_players wired but enabled: false
```

Best EPL 1X2 LL (model): still **0.98202**.  
Best EPL 1X2 **realizable** ROI: **−0.43%** @ max_odds 1.80 (n=401) — essentially flat.

## How close to positive expectancy?

- **1X2 short book (EPL):** within ~0.5% ROI of breakeven; La Liga shorts already **+3%**. Expectancy on this slice is no longer the main fire.
- **Full multi-market book:** still **~−4%** ROI after the 1X2 filter — AH and OU are now the cash drains.
- **Probability gap:** unchanged at **−0.033** LL vs market; match-level players did not close it. Need confirmed XI / better DEF / formation or a different residual channel.

## Single best next action

**Attack AH + OU expectancy** on the filtered stack (autopsy segments after max1.80), and/or **league-specific 1X2 filters** (La Liga already +EV; Serie A prefers max≈2.0). Confirmed pre-match XI only if a true lineup feed appears — post-match expected-XI ratings are not helping LL.

## Experiment folders (iter 10)

- `experiments/bet_filter_grid_iter10/` + `bet_filter_multileague_iter10.csv`
- `experiments/autopsy_iter10_filters/`
- `experiments/match_player_strength_comparison_iter10.csv`
- `experiments/20260805T21*Z_iter10_*`
- `docs/MATCH_PLAYER_STRENGTH.md`

---

# Iteration 21 — Significance, wide hunt, ML paper packs, score preds

Date: 2026-08-12  
Prior (iter20): Bundesliga Unders paper; Champ/Serie ranking still weak; MLS OU blocked.

## Absolute rule

EPL Unders, EPL short Overs, Bundesliga Unders packs **untouched**.

## 1) Statistical significance (protected)

Full report: `experiments/iter21/significance/REPORT.md`

| System | n | ROI | t | p | Boot+ | Seasons+ | Max DD | Verdict |
|--------|--:|----:|--:|--:|------:|---------:|-------:|---------|
| EPL Unders | 495 | +9.4% | 1.61 | 0.108 | 95% | 7/10 | −6.40u | KEEP / MONITOR |
| EPL short Overs | 202 | +7.9% | 1.09 | 0.276 | 86% | 7/10 | −3.27u | KEEP / MONITOR |
| Bundesliga Unders | 161 | +8.6% | 0.97 | 0.335 | 84% | 8/10 | −1.72u | KEEP / MONITOR |

None reach classical p<0.05. Continue production/paper without upsizing on significance alone.

## 2) Wide hunt (1527 configs)

One-shot universe evaluate + pandas band filter + bootstrap shortlist (117).

- **New OU:** none promoted (existing EPL/D1 remain best)
- **AH:** flat / negative across leagues
- **1X2:** La Liga home shorts + Serie A away shorts clear bar

Curated paper packs (new YAML, separate from OU):

| Pack | Rules | Key metrics |
|------|-------|-------------|
| `LaLiga_home_ml_short_exp` | H @ e≥8% max 1.80 | n=124, +20.3%, t=3.54, 8/10, DD −2.0u |
| `SerieA_away_ml_exp` | A @ e≥3% max 2.00 | n=246, +12.4%, t=2.41, **10/10**, DD ~0 |

## 3) Other totals 1.5 / 3.5

Blocked historically (no closes). Live `p_over15`/`p_over35` only.

## 4) MLS + score predictions

Goals-only preds with confidence labels; no OU EV. Artifacts under `experiments/iter21/score_predictions/`.

Infra fixes: `predict_upcoming` match_id alignment; MLS `ftr`; MLS/Championship team aliases; gameday `flag_any` / `systems_flagged` / 1X2 pack flags.

## 5) Priority board

1. EPL Unders — Production  
2. EPL short Overs — Production  
3. Bundesliga Unders — Paper  
4. La Liga Home ML — Paper (new)  
5. Serie A Away ML — Paper (new)  
6. MLS / others — score preds only  

Master: `experiments/iter21/MASTER_REPORT.md`

---

# Iteration 20 — Bundesliga deep dive + multi-league expansion (EPL protected)

Date: 2026-08-11  
Prior (iter19): Bundesliga Unders only credible non-EPL; Champ/Serie ranking broken.

## Absolute rule

EPL_aggressive Unders and EPL_overs_short_exp **untouched**.

## 1) Bundesliga deep dive (complete)

Full report: `experiments/iter20_bundesliga_deep/REPORT.md`

**Primary paper pack `Bundesliga_unders_short_exp`:** Unders **1.70–2.50 @ e≥10%** on thresh05

| Metric | Value |
|--------|------:|
| n | 161 |
| Hit | 48.4% |
| ROI | **+8.6%** |
| Units | +13.89u |
| Avg odds | 2.265 |
| Avg edge | 18.0% |
| Seasons + | **8/10** |
| Max cum DD | **−1.72u** |

Losing seasons: 2021 (−14%, n=7), 2024 (−43%, n=4). Cum ROI stays positive throughout.

### Deep cuts (highlights)
- Odds **1.90–2.10** strongest (+22%); **2.30–2.50** still +4%.
- Edge **10–12%** alone −1.8%; profits in **≥12%**.
- **Home-fav** +14.7% vs away-fav +0.5%.
- **Early season** +19.6%; late −1.4%.
- Losses avg **4.45** goals (blow-ups).

### Variations
Decision-grade neighbors include thresh05 **1.80–2.50 @ e10%** (+9.6%, n=158, 8/10) and e12% bands with higher ROI / thinner n. Prefer e10% primary for sample.

Live: `python scripts/run_gameday_sheet.py --league Bundesliga --refresh-fixtures --refresh-odds`

## 2) Championship / Serie A ranking (complete)

Configs updated:
- Champ: shot-volume + allow via **shots proxy** (`shot_volume_shots_scale=10`), `min_abs_raw=0`
- Serie A: **vol06** enabled on xG stack
- Code: poisson intensity falls back to shots when xG missing

| League | Baseline corr(λ,goals) | New | Δ | Verdict |
|--------|----------------------:|----:|--:|---------|
| Championship | 0.034 | **0.044** | +0.010 | Still broken — stop filter grids |
| Serie A | 0.072 | **0.093** | +0.021 | Improved but still weak (&lt;0.10) |

Edge→outcome correlation remains negative. **Do not promote filters** until corr ≳0.12–0.15.  
Artifacts: `experiments/iter20_ranking/`, `20260811T193046Z_iter20_Championship_shots_vol`, `20260811T194347Z_iter20_SerieA_vol06`.

## 3) MLS feasibility

**Not viable for OU systems.** `football-data.co.uk/new/USA.csv` has 6085 MLS rows (2012–2026) with **1X2 closes only — no OU columns**. Understat MLS missing. Pinnacle live OU id **2663** exists for live only.  
Report: `experiments/iter20_mls/FEASIBILITY.md`

## 4) Multi-league fixtures + odds

- Registry: `src/origination/utils/league_registry.py`
- Pinnacle IDs: EPL 1980, Champ 1977, D1 **1842**, SerieA 2436, LaLiga 2196, MLS 2663
- Fixtures: `refresh_upcoming_fixtures_for_league` (FD / Understat / **Pinnacle matchup fallback**)
- Gameday: `--league` flag; pack flags per league; Bundesliga fixtures tested (9 upcoming, 8/9 OU matched)

## 5) Production vs paper

| Item | Status |
|------|--------|
| EPL Unders / short Overs | **Production** (unchanged) |
| `Bundesliga_unders_short_exp` | **Paper** (promoted from experimental) |
| Champ / Serie research packs | Research only until ranking improves |
| MLS OU | Blocked — no historical OU odds |

## Single best next action

Paper-trade Bundesliga Unders on upcoming D1 slate. Finish ranking WF; if corr(λ,goals) still &lt;0.10, stop filter grids and invest in features.

---

# Iteration 19 — Other-league totals hunt (EPL protected)


Date: 2026-08-11  
Prior (iter18): EPL two books + gameday/Pinnacle live path.

## Goal

Find profitable **non-EPL** OU systems. EPL_aggressive Unders and EPL_overs_short_exp remain untouched.

## EPL control (unchanged)

| Pack | n | ROI | Seasons + |
|------|--:|----:|----------:|
| Unders 2.00–4.00 @ e8% | 495 | +9.4% | 7/10 |
| Short Overs 1.60–2.50 @ e10% | 202 | +7.9% | 7/10 |

## Diagnosis

| League | Bias λ−goals | corr(λ,goals) | corr(edge,over) | Notes |
|--------|-------------:|--------------:|----------------:|-------|
| EPL vol06 | +0.04 | 0.16 | −0.06 | Baseline that works with filters |
| **Bundesliga** | **−0.02…−0.06** | **0.16–0.17** | −0.08 | Slightly under-projects; best non-EPL ranking |
| La Liga | +0.07…+0.09 | 0.21 | **−0.13…−0.14** | Over-projects; edge anti-correlated |
| Championship (stale) | −0.02 | **0.03** | −0.09 | Goals-only legacy — almost no ranking |
| Serie A (stale) | +0.09 | **0.06** | −0.16 | Pre-intercept — weak ranking |

**Why harder:** lower / wrong-signed edge→outcome correlation; Champ/Serie lacked modern intercept+allow stack in available preds; D1 historically over-damped by intercept but thresh05 restored filter edge.

## New experimental systems (NOT merged with EPL)

### 1) Bundesliga Unders (best find)

| Variant | Band | Edge | n | ROI | Hit | Seasons + |
|---------|------|-----:|--:|----:|----:|----------:|
| **thresh05 (prefer)** | 1.70–2.50 | **10%** | **161** | **+8.6%** | 48.4% | **8/10** |
| thresh05 | 1.70–2.50 | 12% | 99 | +15.1% | 51.5% | 7/10 |
| vol06 | 1.70–2.50 | 12% | 95 | +9.5% | 49.5% | 8/10 |

Season table (thresh05 @ e10%): only clear red seasons 2021 (−14%, n=7) and 2024 (−43%, n=4); cum ROI stays positive. Pack: `Bundesliga_unders_short_exp` in `league_rule_packs.yaml`.

### 2) La Liga Unders (weak)

vol06 Unders 2.20–3.50 @ e8%: +6.8% (n=180) but **only 5/10** seasons. Pack `LaLiga_unders_long_exp` = paper-only.

### 3) Championship / Serie A (fresh WF completed)

Artifacts:
- `20260811T164035Z_iter19_Championship_thresh_intercept`
- `20260811T165028Z_iter19_SerieA_signed_intercept`

| League | Bias | corr(λ,goals) | corr(edge,over) | Unfiltered ROI |
|--------|-----:|--------------:|----------------:|---------------:|
| Championship fresh | −0.03 | **0.03** | −0.09 | −3.8% |
| Serie A fresh | +0.05 | **0.07** | −0.15 | −7.7% |

**Diagnosis:** modern intercept/residual did **not** fix ranking. Champ intercept almost always skipped (|raw|<0.05). Stale Aug-5 +EV pockets were **false positives**.

Weak survivors only (research packs, not production):

| Pack | Band | n | ROI | Seasons + | Caveat |
|------|------|--:|----:|----------:|--------|
| `Championship_unders_mid_exp` | U 2.20–3.50 @ e12% | 139 | +2.8% | 8/10 | **2023 n=78 @ −19%** nearly wiped book |
| `SerieA_overs_short_exp` | O 1.50–2.20 @ e8% | 239 | +2.0% | 8/10 | 2025 −14% (n=65); ROI marginal |

## Other lines (1.5 / 3.5)

Still no FD closing books in aligned data — deferred until odds ingest.

## Production vs experimental

| Item | Status |
|------|--------|
| EPL Unders / short Overs | **Production** (unchanged) |
| `Bundesliga_unders_short_exp` | **Best non-EPL** — paper |
| `LaLiga_unders_long_exp` | Weak experimental |
| `Championship_unders_mid_exp` | Research only (fragile) |
| `SerieA_overs_short_exp` | Research only (marginal) |

## League priority (after iter19)

1. **Bundesliga** — only credible non-EPL +EV system
2. **La Liga** — Unders watchlist, unstable seasons
3. **Championship / Serie A** — model ranking still broken; need features (shots/xg) before more filter hunting
4. Smaller leagues — not yet modernized

## Single best next action

Paper-trade **Bundesliga Unders 1.70–2.50 @ e≥10%**. For Champ/Serie: invest in intensity ranking (shot volume / xG) before more filter grids. Do not port EPL bands blindly.

## Artifacts

- `experiments/iter19_other_leagues/` (REPORT, diagnosis, filter_grid, fresh_*, season CSVs)
- `scripts/run_iter19_other_leagues.py`
- `scripts/run_iter19_fresh_wf.py`
- `scripts/run_iter19_fresh_grid.py`

---

# Iteration 18 — Two-book post-mortem + gameday sheet + expansion

Date: 2026-08-11  
Prior (iter17): EPL_aggressive Unders + experimental short Overs on vol06; packs separate.

## Goal

1. Deep seasonal / conflict analysis of both EPL books on best model (vol06 + thresh).
2. Production-ready live gameday prediction sheet (same path as backtest).
3. Expansion search (other lines / leagues / AH) without damaging either book.

## Part 1 — Two-book post-mortem (mandatory)

Artifact: `experiments/iter18_epl_two_books/` (model `20260810T201358Z_iter17_EPL_vol06`).

| Book | n | ROI | Hit | Avg odds | Avg edge | Seasons + |
|------|--:|----:|----:|---------:|---------:|----------:|
| **EPL_aggressive Unders** | 495 | **+9.4%** | 42.8% | 2.59 | 12.3% | **7/10** |
| **EPL_overs_short_exp** | 202 | **+8.0%** | 53.0% | 2.06 | 13.2% | **7/10** |
| EPL_aggressive ALL | 1717 | +2.25% | — | — | — | — |

Seasonal tables in `REPORT.md` / `seasonal_roi.csv`. Unders: only clear red seasons 2016 (−6%) and 2023 (−9%); cum ROI stays positive after 2017. Short overs: volatile (2025 −24% on n=13; 2021 +60% on n=10); still +8% overall.

### Conflicts

**0 same-match opposite-side flags.** On OU 2.5, `edge_over + edge_under = 0` after shared vig removal — only one side can clear a positive threshold. Books are mutually exclusive per game; keep them separate for stake / risk accounting anyway.

### Diagnostics

- Unders odds buckets: 3.00–4.00 strongest (+36% n=85); 2.50–3.00 nearly flat.
- Short overs: both 1.60–2.00 and 2.00–2.50 ~+8.5%.
- Unders edge 0.12–0.15 soft (−1.3%); short overs soft in same band (−5.7%).
- Failure pattern: Unders losses average **4.24** goals; Over losses **1.31** — classic blow-up / blank.

## Part 2 — Gameday sheet (production path)

| Item | Path |
|------|------|
| CLI | `scripts/run_gameday_sheet.py` |
| Docs | `docs/GAMEDAY_SHEET.md` |
| Core | `src/origination/prediction/upcoming.py` (features + intercept + calib + residual) |

```bash
.venv\Scripts\python.exe scripts/run_gameday_sheet.py --fixtures fixtures.csv --odds-file odds.csv
# optional: --update-data --late-info late.csv --fast
```

**Edge (matches backtest):** `model_prob − power_devig_fair(side)` when both OU prices exist; else `model_prob − 1/odds`. Consensus = mean of all `*over25*` / `*under25*` columns. Pack flags independent: `flag_EPL_aggressive_under` vs `flag_EPL_overs_short`.

Smoke: `data/processed/gameday_sheet_smoke.csv` (fast mode ~90s on EPL history).

Also: American fair odds helpers in `utils/odds.py`; OU 1.5/3.5 probs in `markets_from_matrix`.

## Part 3 — Expansion (discovery only)

Artifact: `experiments/iter18_expansion/`.

| Finding | Detail |
|---------|--------|
| EPL Unders check | Still **+9.4%** (untouched) |
| EPL short overs | Still **+7.9%** |
| EPL mild OU | Flat/neg (−3.5% @ e5; ~0% @ e8) |
| EPL AH (aggressive) | **−0.9%** (n=825) — not a third book |
| Bundesliga short overs | **−8.2%** |
| La Liga short overs | **−12.6%** |
| D1 / SP1 mild | Still −EV |
| OU 1.5 / 3.5 | No FD closing books; hit-rate only (U3.5 @ e10 hit ~70% n=2990 — research, not priced EV) |

No new portable league pack. No merge of the two EPL systems.

## Production vs experimental

| Item | Status |
|------|--------|
| EPL vol06 + thresh intercept model | **Production** (sheet + paper) |
| `EPL_aggressive` Unders OU | **Production** |
| Gameday sheet CLI + docs | **Production** |
| `EPL_overs_short_exp` | **Experimental** (flagged separately) |
| AH / mild multi-league / OU 1.5–3.5 | **Research only** |

## Single best next action

Paper both EPL books with **separate** bankroll caps; use gameday sheet daily. Next research: price OU 3.5 (ingest books) or attack La Liga short gap — do not touch EPL vol06 stack.

## Experiment folders (iter 18)

- `experiments/iter18_epl_two_books/`
- `experiments/iter18_expansion/`
- `docs/GAMEDAY_SHEET.md`

---

# Iteration 17 — Overs hunt + hyperparam/combo explore

Date: 2026-08-10  
Prior (iter16): thresh intercept on EPL+D1; SP1 full signed; EPL_aggressive protected.

## Goal

Wide search for Overs paths and layer settings **without** damaging the EPL Under system. Multi-league WF + filter grids; Overs vs Unders reported everywhere.

## Part 1 — Overs filter search (fast)

Artifact: `experiments/iter17_overs_filter_grid/`.

| Finding | Detail |
|---------|--------|
| **EPL_aggressive intact** | thresh +1.54% ALL / +6.4% OU; vol06 **+1.88% / +9.7%** |
| **EPL short overs @ e10%** | vol06: **+7.6%** (n=191, **7/10** seasons); thresh +4.8% (n=195) |
| D1 overs | thin pockets only; not robust |
| La Liga overs | **all packs −EV** |
| Portable overs | **None** across 3 leagues |

Optional pack: `EPL_overs_short_exp` (1.60–2.50 overs @ edge 10%) — experimental, separate from Under pack.

## Part 2 — WF hyperparams / combos (12 × 3 leagues)

Artifact: `experiments/iter17_explore_multileague.csv`, `iter17_explore_SUMMARY.md`.

| Variant | mean Δ OU LL | mean Δ short | Mild overs | EPL pack ALL | Notes |
|---------|-------------:|-------------:|------------|-------------:|-------|
| **vol06** | −0.00040 | +0.00105 | still −EV | **+2.25%** | Best Under boost |
| vol08 | −0.00058 | +0.00137 | worse | +1.31% | Short regresses |
| allow08 | +0.00043 | +0.00026 | least-bad EPL mild | +1.08% | Hold |
| aou15 / overw* | flat/↓ | ↓/flat | no | weaker | No |
| intercept grids | no | no | no | weaker | Hold thresh |
| vol06×overw / aou | mixed | ↓ short | no | ok/+ | No short synergy |

**Bundesliga mild overs** briefly ≈flat under vol06 (+0.2%) — not a claim; oseek still −EV.

## Part 3 — Interaction matrix (summary)

| Combo | Synergy? |
|-------|----------|
| vol06 × EPL Under pack | **Yes** (selection ROI) |
| vol06 × short Overs @ e10% | **Yes on EPL** (filter) |
| vol06 × La Liga / short gap | **Anti** |
| over_fav residual weight × anything | No mild-overs unlock |
| Higher α_ou × vol | Helps OU LL EPL+D1; hurts short |

## Part 4 — Promote

```
# configs/default.yaml (EPL only)
shot_volume_coef: 0.06
# D1/SP1 remain 0.0
# EPL_aggressive rules UNCHANGED
# Optional: EPL_overs_short_exp in league_rule_packs.yaml
```

## Honest Overs verdict

Overs are **selectively possible on EPL** as a high-edge short-favorite book, especially with `vol06` probabilities. They are **not** a mild-band or multi-league product. Unders remain the robust EPL engine. Do not merge Overs into `EPL_aggressive`.

## Single best next action

Paper-trade **two separate EPL books**: (1) protected Under pack, (2) experimental short-Over pack — with correlated-risk limits. Model-side: attack **La Liga short gap** (still ~+0.011) without touching EPL vol06+thresh stack.

## Experiment folders (iter 17)

- `experiments/iter17_overs_filter_grid/`
- `experiments/iter17_explore_multileague.csv` / `iter17_explore_SUMMARY.md`
- Residual: `ou_over_fav_weight`

---

# Iteration 16 — D1-safe intercept + re-test layers on new base

Date: 2026-08-10  
Prior (iter15): totals intercept promoted EPL+La Liga; D1 off; short gaps narrowed but not closed; sidelined layers not portable on old base.

## Goal

Bundesliga-safe intercept; re-test vol / α_ou / suppress / short-weight **on** `xg_allow` + intercept base; protect `EPL_aggressive`.

## Part 1 — D1-safe intercept modes

Implemented: `mode` ∈ {signed, lift_only, dampen_only, asymmetric}, plus `min_abs_raw`, `dampen_shrink`.

| Mode | Effect on EPL/SP1 | Effect on D1 | Verdict |
|------|-------------------|--------------|---------|
| lift_only / asymmetric (dampen_shrink=1) | Reverts to no-intercept (blocks needed dampen) | ≈ base (already off) | **Not universal** |
| **signed + min_abs_raw=0.05** | Best EPL short + pack | Short ↑; OU LL slight ↓ | **Promote EPL+D1** |
| signed full (SP1) | — | — | **Keep SP1** |

`lift_only` cannot replace signed on leagues that over-predict.

## Part 2 — Layers on new base (EPL + D1 + SP1)

Artifact: `experiments/iter16_totals_multileague.csv`, `iter16_totals_SUMMARY.md`.

| Variant | mean Δ short | mean Δ OU LL | Portable? |
|---------|-------------:|-------------:|-----------|
| **int_thresh05** | **−0.00026** | +0.00025 | Short: EPL+D1 |
| vol06 | +0.00078 | **−0.00095** | OU LL: EPL+D1; short fails SP1 |
| aou15 | +0.00060 | −0.00011 | No |
| suppress04 | +0.00017 | +0.00035 | No |
| shortw2 | +0.00049 | +0.00045 | No |
| asym + vol/aou | worse short | mixed | Anti-synergy |

### Synergy table

| Layer | Short | OU LL | EPL pack | Notes |
|-------|-------|-------|----------|-------|
| thresh 0.05 | ↑ EPL+D1 | ↑ EPL | **+1.54%** | Promote EPL+D1 |
| vol06 on new base | SP1 short ↓ | ↑ EPL+D1, corr↑ | **+1.88% / OU +9.7%** | Pack juice; not short-portable |
| α_ou 0.15 | ↓ | tiny | +0.93% | No |
| suppress / shortw | ↓/flat | ↓ | weak/−EV | No |
| asym × anything | ↓ EPL/SP1 | — | −EV | Blocks dampen |

**New finding:** `shot_volume` looks better on the intercept+allow base for EPL pack / D1 OU LL than in iter14 — but still fails La Liga short-band. Do not promote globally.

## Part 3 — Promote

```
# default.yaml (EPL)
totals_intercept: {enabled: true, mode: signed, min_abs_raw: 0.05, ...}

# league_D1_bundesliga.yaml  (was off)
totals_intercept: {enabled: true, mode: signed, min_abs_raw: 0.05, ...}

# league_SP1_la_liga.yaml    (unchanged)
totals_intercept: {enabled: true, mode: signed, min_abs_raw: 0.0, ...}
```

`EPL_aggressive` pack YAML untouched; still +EV (stronger under thresh/vol experiments).

## Honest remaining gap

| League | Short LL gap (post-promote estimate) |
|--------|--------------------------------------:|
| EPL | ~+0.0051 |
| Bundesliga | ~+0.0067 |
| La Liga | ~+0.0114 |

Mild multi-league OU and overs still −EV. Corr ~0.17–0.18 (vol06 pushes ~0.18). Gap **not closed**.

## Single best next action

Attack **La Liga short-band** specifically (largest remaining gap): SP1-only residual/calibration or dampen-strength grid — while holding EPL/D1 thresh stack fixed. Parallel: paper `vol06` only as an EPL-pack sensitivity (not research default).

## Experiment folders (iter 16)

- `experiments/iter16_totals_multileague.csv` / `iter16_totals_rank.csv` / `iter16_totals_SUMMARY.md`
- Code: intercept `mode` / `min_abs_raw` / `dampen_shrink` in `poisson.py`

---

# Iteration 15 — League-aware totals intercept + short-OU residual grind

Date: 2026-08-10  
Prior (iter14): `xg_allow_coef=0.06` portable OU LL win; short gaps still +0.006–0.014; opposite league biases.

## Goal

Close the OU probability gap (esp. 1.60–2.50) across ≥3 leagues. Keep `EPL_aggressive` pack **intact**.

## Part 1 — Implementations

1. **League-aware totals intercept** (`DixonColesModel.calibrate_totals_intercept`): after fold fit, `offset = clip((1−shrink)·log(mean_goals / mean(λ+μ)))`, apply jointly to λ,μ. Wired in walk-forward, OOS residual collection, and live `predict_upcoming`.
2. **Short-price OU residual**: `residual.ou_short_weight` upweights training rows whose shorter OU closing price is in `[1.60, 2.50]`.
3. **Style layers**: `sum_suppress_resid_ewm`, `sum_pv_open_orth_ewm`, `tempo_ppda_coef`, `suppress_resid_coef`, `pv_open_orth_coef` (+ residual interactions).

## Part 2 — Multi-league results (EPL + D1 + SP1)

Artifact: `experiments/iter15_totals_multileague.csv`, `iter15_totals_SUMMARY.md`.

| Variant | mean Δ OU LL | short gap ↑ all 3? | Verdict |
|---------|-------------:|:------------------:|---------|
| **tot_int** | **−0.00004** | **Yes** | Promote EPL+La Liga |
| ou_shortw | +0.00029 | No (EPL only) | Hold |
| suppress04 | +0.00044 | No (D1 mild only) | Hold |
| tempo / pv_orth | worse | No | Reject |
| tot_int + short (±suppress) | worse mean | No | Anti-synergy cross-league |

### tot_int detail

| League | Δ OU LL | Δ short gap | goals_err base → tot_int |
|--------|--------:|------------:|--------------------------|
| EPL | −0.00038 | −0.00061 | +0.15 → −0.03 |
| La Liga | −0.00163 | −0.00045 | +0.20 → +0.07 |
| Bundesliga | **+0.00190** | −0.00048 | −0.02 → **−0.10** (over-damped) |

**Only `tot_int` improved short-band LL gap on all three leagues.** OU LL improves where the base over-predicts goals (EPL/SP1), not where already near-calibrated (D1).

### Correlation update

With iter14 `xg_allow` in the base stack, `corr(λ+μ, goals)` ≈ **0.14–0.21** (vs 0.02–0.07 on pre-allow diagnosis artifacts). Mean intensity is less broken than before; residual short-price information is now the bottleneck.

## Part 3 — EPL_aggressive pack (rules unchanged)

| Variant | ALL ROI | OU ROI |
|---------|--------:|-------:|
| base | +0.03% | +4.6% |
| tot_int | +0.53% | +3.8% |
| ou_shortw | +0.45% | +6.1% |
| tot_int_short | +1.04% | +5.2% |

Pack still +EV; YAML rules in `league_rule_packs.yaml` untouched.

## Part 4 — Promote / hold

**Promote**
```
# default.yaml (EPL) + league_SP1_la_liga.yaml
model.dixon_coles.totals_intercept: {enabled: true, shrink: 0.15, clip: 0.12}
```

**Hold off on Bundesliga** (`league_D1_bundesliga.yaml` totals_intercept.enabled: false).

**Do not promote:** ou_short_weight, suppress/tempo/PV orth intensity, short+intercept combos as global defaults.

## Honest assessment

- Closed a **slice** of the short-band gap (~8–10% on EPL; smaller absolute moves on D1/SP1).
- Remaining short gaps: EPL **+0.0056**, D1 **+0.0068**, SP1 **+0.0114**.
- Mild OU ROI still negative everywhere.
- Gap is **not closed**; mean-rate league fix was the right tool for opposite biases, but Bundesliga needs a different lever (not more global dampening).

## Single best next action

**Short-band OU residual with open-odds features only** (or market-free sharpness features: λ variance / DC underdispersion), trained with `ou_short_weight`, validated on ≥3 leagues — and a **Bundesliga-specific** totals path (e.g. only apply intercept when train goals_err magnitude exceeds a threshold, or a positive-only lift when under-predicting). Avoid stacking intercept + short-weight until D1 is fixed.

## Experiment folders (iter 15)

- `experiments/iter15_totals_multileague.csv` / `iter15_totals_rank.csv` / `iter15_totals_SUMMARY.md`
- `experiments/20260810T*_iter15_*`
- Code: `poisson.calibrate_totals_intercept`, `residual.ou_short_weight`, store suppress/PV-orth sums

---

# Iteration 14 — Fundamentals: OU probability quality (multi-league)

Date: 2026-08-07  
Prior (iter13): mild universal filters still −EV; EPL aggressive pack +EV but overfit; no layer fixed short OU.

## Goal

Improve **actual Over/Under probabilities** (esp. 1.60–2.50), validated on **≥3 leagues**. Keep `EPL_aggressive` as optional paper pack only.

## Part 1 — Deep OU diagnosis

Artifact: `experiments/iter14_ou_diagnosis/`.

| League | LL gap vs mkt | Short 1.60–2.50 gap | Goals err | Dominant failure |
|--------|--------------:|--------------------:|----------:|------------------|
| EPL | +0.0085 | +0.0063 | +0.07 | Mild; market sharper |
| Championship | +0.0084 | +0.0081 | −0.02 | Similar to EPL |
| Bundesliga | **+0.0219** | +0.0057 | **−0.09** | Under-rates high totals (1.40–1.60 bias **−0.09**) |
| Serie A | +0.0153 | +0.0118 | +0.09 | Under-confident on over favorites |
| La Liga | +0.0157 | +0.0138 | **+0.17** | Over-rates totals (1.40–1.60 bias **+0.06**) |

**Root causes:**
1. `corr(λ+μ, goals)` in short band only **0.02–0.07** — mean intensity barely tracks totals.
2. Biases are **opposite across leagues** → global “lift totals” knobs are dangerous.
3. Favorite-side: over-fav → model too low on p_over (D1/I1); under-fav → model too high (SP1/E1).

## Part 2 — Modeling changes tested

| Component | What | Alone | Combo notes |
|-----------|------|-------|-------------|
| `sum_*_ewm` + `sum_lambda` residual feats | Totals proxies in store/residual | In all runs (new base) | Keep |
| `shot_volume_coef=0.06` | Joint λ/μ from creation | Helps D1; hurts SP1 | Anti-portable |
| **`xg_allow_coef=0.06`** | Joint λ/μ from defensive allowance | **OU LL ↑ EPL+D1+SP1** | Best portable |
| OU platt calib | Replace temperature | Helps D1/SP1; **hurts EPL hard** | No |
| `alpha_ou=0.15` | Stronger OU residual | Flat | No synergy with volume |

## Part 3 — Multi-league results (EPL + Bundesliga + La Liga)

Artifact: `experiments/iter14_totals_multileague.csv`, `iter14_totals_SUMMARY.md`.

| Variant | mean Δ OU LL | All-3-league win? | mean short gap |
|---------|-------------:|:-----------------:|---------------:|
| vol_allow | −0.00156 | No (SP1 regresses) | 0.0091 |
| ou_platt | −0.00128 | No (EPL +0.003) | 0.0100 |
| vol06 | −0.00102 | No | 0.0092 |
| **allow06** | **−0.00059** | **Yes** | **0.0086** |
| aou15 | −0.00009 | ≈flat | 0.0091 |
| base | 0 | — | 0.0085 |

**allow06 per league Δ OU LL:** EPL −0.00032 | Bundesliga −0.00126 | La Liga −0.00020.  
1X2 LL stays acceptable (EPL/D1 slight ↑, SP1 slight ↓).

Mild-book OU ROI still negative everywhere; EPL mild improves base −4.4% → allow −2.8%.

## Part 4 — Promote / hold

**Promote**
```
model.dixon_coles.intensity_adjustments.xg_allow_coef: 0.06
# + sum_*_ewm features and residual sum_lambda / totals interactions (code)
```

**Hold / reject:** shot_volume, vol_allow, ou_platt, α_ou=0.15.

**EPL_aggressive pack** (optional): still +EV on new preds (base ~+1.1% ALL / +5% OU; allow06 ~+0.3% / +3.5% OU) — pack ROI softens as probabilities move; pack stays in `league_rule_packs.yaml`, not the research default.

## Honest assessment

We **narrowed** the OU probability gap with a **portable** defensive-allowance channel, but we have **not closed** it. Short-band LL gaps remain ~+0.006–0.014 vs market; mild multi-league OU books remain −EV. Opposite league biases mean further intensity coefs need league-aware or residual (not global multiplier) treatment.

## Architecture (1H / live)

Unchanged contract in `docs/LIVE_AND_1H_ARCHITECTURE.md`. New intensity knobs attach via the same `_intensity_multipliers_from_row` path usable later for live multipliers.

## Single best next action

**OU residual redesign for short prices:** train / weight the OU head on short-band errors (or isotonic residual on p_over − fair_open if open odds exist), and/or a **league-aware** totals offset (hierarchical totals intercept) so Bundesliga lift and La Liga dampen can coexist without a single global coef.

## Experiment folders (iter 14)

- `experiments/iter14_ou_diagnosis/`
- `experiments/iter14_totals_multileague.csv` / `iter14_totals_rank.csv` / `iter14_totals_SUMMARY.md`
- `experiments/20260807T13*Z_iter14_*`
- `docs/TOTALS_INTENSITY.md` (volume/allow channels)

---

# Iteration 13 — Robust multi-league totals (mild filters + layer study)

Date: 2026-08-06  
Prior (iter12): EPL aggressive pack **ALL +1.77%** (OU unders 2–4 +7.6%); PV v2 not promoted.

## Goal shift

Move from a strong **EPL selection** system toward a **robust, totals-focused, multi-league** model without heavy EPL-specific filters. Prefer odds inside **1.50–3.00**; make short OU **1.60–2.00** viable via modeling, not only long unders.

## Part 1 — Mild universal filters (5 leagues)

Artifact: `experiments/iter13_universal_filters/`. Same packs on EPL / Championship / Bundesliga / Serie A / La Liga.

| Pack | Mean OU ROI | Leagues +EV OU | Notes |
|------|------------:|---------------:|-------|
| raw edge 3% | −6.9% | 0/5 | |
| band OU 1.50–3.00 @ e5% | −6.6% | 0/5 | default research band |
| band OU 1.50–3.00 @ e8% | −6.3% | 0/5 | **EPL OU −0.43%** (near flat) |
| short OU 1.60–2.00 @ e8% | −3.8% | 1/5 | Serie A pocket only; EPL −6% |
| **iter12 EPL pack** | −5.3% | **1/5** | EPL +7.6%; others −6% to −13% |

**Honest:** no mild universal pack is +EV on OU across leagues. Raising edge only flattens EPL toward zero; it does not create a portable edge. The iter12 pack is **EPL-overfit** (1/5 leagues +EV).

## Part 2 — Totals modeling + elite layers (EPL WF)

Artifact: `experiments/iter13_layer_study.csv`, `experiments/iter13_layer_study_SUMMARY.md`.

| Variant | OU LL | Mild OU (1.5–3) | Short 1.6–2.0 | EPL-pack ALL |
|---------|------:|----------------:|--------------:|-------------:|
| **L_base** | **0.68304** | −4.51% | −8.0% | **+1.77%** |
| hier off | 0.68313 | −5.15% | −10.1% | +1.05% |
| PV intensity 0.05 | 0.68556 | −5.10% | −9.6% | **−2.02%** |
| OU α=0.15 | 0.68317 | **−4.09%** | −9.0% | +1.39% |
| OU-specialist | 0.68431 | −5.51% | −8.7% | +1.41% |
| PV + OU α | 0.68554 | −4.80% | −8.3% | −1.85% |
| hier off + OU α | 0.68341 | −4.36% | −9.2% | +1.36% |

### Layer interaction map

| Layer | Helps totals? | Helps 1X2 / EPL pack? | Combo notes |
|-------|---------------|----------------------|-------------|
| Hierarchical | Mild + pack vs off | Slight 1X2 LL help | **Keep** |
| PV intensity | No (worse OU LL) | **Kills** EPL pack | Anti-synergy with selection |
| Higher OU residual α | Mild ROI slight ↑; LL slight ↓ | Pack weaker | Marginal; **do not promote** |
| OU-specialist (α1x2=0) | No | Pack weaker | No |
| PV × OU α | No | Pack negative | No synergy |

**Short OU 1.60–2.00:** every layer still ≈ −8% to −10%. Core probability quality has not unlocked short prices.

## Part 3 — Default / packs after iter13

**`configs/default.yaml`** switched to **mild universal** (research default):

```
edge_threshold_by_market: {ou25: 0.05, ah: 0.05}
bet_filters.rules:
  - {markets: [1x2], max_odds: 2.00}
  - {markets: [ou25], min_odds: 1.50, max_odds: 3.00}
  - {markets: [ah], min_odds: 1.50, max_odds: 3.00}
```

**`configs/league_rule_packs.yaml`:** `mild_universal`, `mild_universal_e08`, and optional **`EPL_aggressive`** (iter12 +EV selection, not portable).

Model layers: **no promote** (keep hier on, PV intensity off, α 0.10/0.10).

## Part 4 — 1H / live readiness

`docs/LIVE_AND_1H_ARCHITECTURE.md` — reuse `evaluate_predictions` + intensity hooks; list HT goals / live feed / remaining-time model needs. No 1H/live backtest this iter.

## Robustness vs EPL-specific pack

| Criterion | Mild universal | EPL aggressive (iter12) |
|-----------|----------------|-------------------------|
| Cross-league OU | All negative (~−6%) | 1/5 +EV (EPL only) |
| Odds band | Inside 1.50–3.00 | Unders 2–4 (longer) |
| EPL realizable ROI | ≈ −3.5% ALL / −4.5% OU | **+1.8% ALL / +7.6% OU** |
| Overfit risk | Lower | High (pocket + side + league) |

**Verdict:** Selection still beats modeling for EPL paper P&L. Mild defaults are the honest research baseline; `EPL_aggressive` remains an optional paper-trade pack. Closing the gap requires **better OU probabilities at short prices**, not more EPL filters.

## How close to a robust multi-league totals system?

- **Not there yet.** Mild books lose ~4–7% OU everywhere.
- EPL near-flat at band @ e8% (−0.4%) is a hint that edge calibration helps one league, not a product.
- Layer knobs did not move short-OU viability.

## Single best next action

**Model-side OU gap work** aimed at short prices: market-implied OU calibration / residual features that target 1.60–2.00 mispricing, validated on ≥3 leagues before any filter tighten. Keep `EPL_aggressive` as a monitored paper book only.

## Experiment folders (iter 13)

- `experiments/iter13_universal_filters/`
- `experiments/iter13_layer_study.csv` + `iter13_layer_study_SUMMARY.md`
- `experiments/20260806T16*Z_iter13_*` (WF runs)
- `docs/LIVE_AND_1H_ARCHITECTURE.md`
- `configs/league_rule_packs.yaml` (rewritten)

---

# Iteration 12 — Lock OU pocket + cut AH + PV v2 + league packs

Date: 2026-08-06  
Prior (iter11): EPL OU **+4.1%** (min_odds 2.00 + edge≥8%, n=823); full book **−1.2%**.

## Part 1 — Deep autopsy of the +EV OU pocket

Source: iter11 promote book (OU min 2.00 @ edge≥8%). Artifacts: `experiments/iter12_totals_refine/autopsy_ou_pocket/`.

| Slice | n | ROI | Note |
|-------|--:|----:|------|
| **under / mild dog 2.70–4.00** | 190 | **+17.1%** | core juice |
| under / pickem 2.00–2.70 | 345 | +2.3% | |
| over / pickem | 269 | +0.7% | ≈flat |
| under overall | 545 | **+6.5%** | |
| over overall | 278 | −0.6% | drop |
| edge 8–12% | 469 | **+7.7%** | |
| edge 12%+ | 354 | −0.6% | soft cap helps |
| season 2023 | 99 | −23% | only bad year |
| seasons 2024–25 | 88 | +20–30% | recent strong |

**What makes it work:** betting **unders at longer prices** with solid edge — not overs, not short favorites, not mega-dogs 4.00+.

## Part 2 — OU refinement (promote)

| Config | n_OU | ROI | t | vs iter11 |
|--------|-----:|----:|--:|-----------|
| iter11 (min2 + e8%) | 823 | +4.1% | 0.95 | — |
| under only | 545 | +6.5% | 1.17 | ↑ |
| **under + max 4.00** | **535** | **+7.6%** | **1.35** | **↑ promote** |
| under dogs 2.7–4.0 | 190 | +17.1% | 1.56 | thin |
| under + edge≥10% | 374 | +7.4% | 1.09 | similar, less n |

**Promote `under + min_odds 2.00 + max_odds 4.00 + edge≥8%`** — best ROI/volume tradeoff.

## Part 3 — AH on the refined OU stack

| AH rule (with under_max4 OU) | n_AH | AH ROI | **ALL ROI** |
|------------------------------|-----:|-------:|------------:|
| edge≥5% (no odds cap) | 2401 | −3.2% | −1.1% |
| edge≥10% | 1257 | −1.9% | **+0.7%** |
| **max_odds 1.90** | **830** | **−0.9%** | **+1.8%** |
| AH off (1X2+OU only) | 0 | — | **+4.1%** |

**Promote AH `max_odds: 1.90`** alongside OU unders → **first positive full EPL multi-market book (+1.77%, n=1766, t=0.76)**.

## Part 4 — Possession Value v2

Open-play OBV + deep-orthogonal residual; intensity and OU-specialist residual (`α_1x2=0`).

On promoted book:

| Variant | OU LL | filt OU ROI | ALL ROI |
|---------|------:|------------:|--------:|
| **baseline (no PV)** | **0.68304** | **+7.56%** | **+1.77%** |
| intensity 0.08 | 0.68744 | −0.4% | −1.0% |
| intensity 0.12 | 0.68906 | −1.3% | −1.6% |
| OU-spec α_ou=0.15 | 0.68431 | +5.4% | +1.4% |
| OU-spec α_ou=0.20 | 0.68517 | +4.7% | +1.2% |

**Do not promote PV v2** — intensity hurts; OU-specialist underperforms filter-only baseline and loses 1X2 residual when `α_1x2=0`.

## Part 5 — League rule packs

File: `configs/league_rule_packs.yaml`

| League | Best pack | OU ROI | Notes |
|--------|-----------|-------:|-------|
| **EPL** | under 2–4 @ e8% + AH≤1.90 | **+7.6%** | **+EV ALL** |
| Championship | OU edge 8% | −5.0% | not +EV |
| Bundesliga | overs @ e8% | −2.4% | least-bad |
| Serie A | overs | −3.9% | least-bad |
| La Liga | unders | −3.2% | ALL ≈ −0.2% (AH healthy) |

## Promoted default after iteration 12 (EPL)

```
backtest.edge_threshold_by_market: {ou25: 0.08, ah: 0.05}
backtest.bet_filters.rules:
  - {markets: [1x2], max_odds: 1.80}
  - {markets: [ou25], min_odds: 2.00, max_odds: 4.00, allow_sides: [under]}
  - {markets: [ah], max_odds: 1.90}
```

Verified on hier preds: **1X2 −0.4% | OU +7.6% | AH −0.9% | ALL +1.77%**.

## How close to a consistently positive totals system?

- **EPL filtered book is now +EV overall (+1.8%)** with a **+7.6% OU core** (n=535, t=1.35 — still not huge t, one bad season 2023).
- Totals-only (drop AH) ≈ **+4%** ALL — cleaner if AH capacity is optional.
- Other leagues are **not** there yet; packs are “least harm,” not +EV claims.
- Model probability gap unchanged; wins remain **selection**. Path to consistency: season-stability checks, live paper trade, and only then model-side OU improvements that don’t fight the filter.

## Single best next action

**Paper / live the EPL pack** (unders 2–4 + AH≤1.90) with season-level monitoring (especially avoid repeating 2023-style drawdowns). Parallel: La Liga AH-led book (near flat already) as second market.

## Experiment folders (iter 12)

- `experiments/iter12_totals_refine/` (OU/AH grids, autopsies, league packs)
- `experiments/possession_value_v2_comparison_iter12.csv`
- `experiments/possession_value_v2_on_promoted_book_iter12.csv`
- `configs/league_rule_packs.yaml`
- `docs/POSSESSION_VALUE.md` (v2)

---

# Iteration 11 — OU/AH focus + Possession Value v1

Date: 2026-08-05  
Prior (iter10): 1X2 max_odds=1.80 → EPL 1X2 ROI **−0.43%**; full book still ~−4% on AH/OU drag.

## Part 1 — OU + AH autopsy (after 1X2 ≤ 1.80)

Artifacts: `experiments/autopsy_iter11_ou_ah/` (+ `ROLLUP.md`).

### Overall (1X2 filtered; OU/AH raw)

| League | OU n | OU ROI | AH n | AH ROI |
|--------|-----:|-------:|-----:|-------:|
| EPL | 2593 | **−3.8%** | 2962 | **−5.3%** |
| Championship | 3958 | −5.5% | 4802 | −2.2% |
| Bundesliga | 2163 | −9.8% | 2358 | −4.5% |
| Serie A | 2658 | −8.4% | 2825 | −3.9% |
| La Liga | 2811 | −6.9% | 2813 | **−0.6%** |

### EPL OU failure modes

| Segment | n | ROI | Note |
|---------|--:|----:|------|
| mild fav 1.50–2.00 | 790 | **−8.8%** | primary OU drag |
| OU favorites | 854 | −8.0% | |
| overs | 1227 | −4.8% | slightly worse than unders |
| unders | 1366 | −2.9% | |
| mild dogs 2.70–4.00 | 300 | **+5.2%** | NS but directionally +EV |
| edge 8–12% / 12%+ | 648 / 417 | **+1.5%** | high-edge OU healthier |

### EPL AH failure modes

| Segment | n | ROI |
|---------|--:|----:|
| edge 3–5% | 561 | **−14.3%** |
| pickem 2.00–2.70 | 1116 | −5.9% |
| edge 5–8% | 732 | **+0.5%** (≈flat) |

## Part 2 — Targeted OU/AH filters

Extended `bet_filters` with `rules` / `min_odds` / `allow_sides`, plus `edge_threshold_by_market`.

### EPL filter grid (highlights)

| Config | OU ROI | n_OU | ALL ROI |
|--------|-------:|-----:|--------:|
| 1X2≤1.80 only | −3.8% | 2593 | −4.3% |
| OU under only | −2.9% | 1366 | −4.2% |
| OU min_odds 2.00 | −1.7% | 1739 | −3.7% |
| OU min_odds 2.70 (dogs) | **+4.1%** | 316 | −3.9% |
| OU edge ≥ 8% | **+1.5%** | 1065 | −3.2% |
| **OU min2.00 + edge 8% + AH edge 5%** | **+4.1%** | **823** | **−1.2%** |

`max_odds` caps on OU **hurt** (they keep the toxic mild favorites). Correct direction is **min_odds** / higher edge.

### Multi-league caution

EPL-promoted OU filters are **not** portable:

| League | raw OU | promoted OU | promoted ALL |
|--------|-------:|------------:|-------------:|
| EPL | −3.8% | **+4.1%** | **−1.2%** |
| Championship | −5.5% | −5.0% | −3.5% |
| Bundesliga | −9.8% | −13.1% | −6.8% |
| Serie A | −8.4% | −7.0% | −4.0% |
| La Liga | −6.9% | −10.0% | −2.6% (AH ≈flat/+) |

**Promote for EPL default only.** Other leagues keep 1X2 shorts; tune OU per league later.

## Part 3 — Possession / On-Ball Value v1

From Understat match shots + roster buildup (`docs/POSSESSION_VALUE.md`):

- Built `understat_possession_value.parquet` (9068 team-match rows, mean `pv_obv`≈1.91)
- Lagged in feature store; `pv_coef` joint λ/μ totals channel

| pv_coef | 1X2 LL | OU LL | OU gap | OU ROI (1X2-filt book) | filt OU ROI* |
|--------:|-------:|------:|-------:|-----------------------:|-------------:|
| 0 (base) | **0.98202** | 0.68304 | −0.00854 | −3.79% | **+4.12%** |
| 0.05 | 0.98267 | 0.68290 | −0.00840 | −3.39% | **+4.54%** |
| 0.10 | 0.98270 | 0.68262 | −0.00812 | −3.92% | — |
| 0.15 | 0.98357 | **0.68230** | **−0.00780** | −3.80% | +4.00% |
| 0.20 | 0.98386 | 0.68234 | −0.00784 | −3.21% | +3.60% |

\*Promoted EPL filters (OU min 2.00 + edge 8%, AH edge 5%).

**Do not promote PV into default.** Mild OU LL help; 1X2 LL regresses; filtered OU ROI not clearly better than filter-only. Layer stays wired (`possession_value: false`, `pv_coef: 0`).

## Promoted default after iteration 11 (EPL)

```
backtest.bet_filters:
  enabled: true
  rules:
    - {markets: [1x2], max_odds: 1.80}
    - {markets: [ou25], min_odds: 2.00}
backtest.edge_threshold_by_market:
  ou25: 0.08
  ah: 0.05
# hierarchical retained; possession_value off
```

Realizable EPL book @ promote: **OU +4.1% (n=823)**, 1X2 −0.4%, AH −3.2%, **ALL −1.2%**.

## How close to a positive totals-focused system?

- **EPL OU filtered book is now positive (+4%)** — first clear +EV totals pocket with volume (n≈800).
- Still **selection**, not a closed probability gap: OU LL gap ≈ **−0.008** (PV can shave ~0.0007 at cost of 1X2).
- Full portfolio −1.2%: AH remains the residual bleed (~−3%).
- Non-EPL OU filters need separate tuning before claiming a multi-league totals system.

## Single best next action

**AH expectancy** (raise edge further / line-specific filters; La Liga AH already ~flat) and/or **league-specific OU rule packs**. Optional: fold PV only into an OU-specialist residual path that does not touch 1X2 logits.

## Experiment folders (iter 11)

- `experiments/autopsy_iter11_ou_ah/`
- `experiments/ou_ah_filter_grid_iter11/` (+ `promoted_multileague.csv`)
- `experiments/possession_value_comparison_iter11.csv`
- `experiments/20260805T22*Z_iter11_*`
- `docs/POSSESSION_VALUE.md`
