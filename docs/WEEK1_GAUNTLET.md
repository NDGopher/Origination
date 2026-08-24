# Week 1 gauntlet — Score Predictions, TT paper, live ledger

Updated: 2026-08-24T15:22:14.064356+00:00

Protected live pack **rules unchanged**. This is a post-week-1 performance review.

Window: **2026-08-15 → 2026-08-24**. Predictions snap: `SCORE_PREDICTIONS_20260821.csv`.

## 1. Inputs / freshness

| League | n current | latest | FD | extra | xG |
|--------|----------:|--------|---:|------:|---:|
| Belgium | 18 | 2026-08-16 | 18 | 0 | 18 |
| Bundesliga | 0 | — | 0 | 0 | 0 |
| Championship | 12 | 2026-08-17 | 12 | 0 | 12 |
| EPL | 9 | 2026-08-23 | 0 | 9 | 9 |
| Eredivisie | 18 | 2026-08-16 | 18 | 0 | 18 |
| LaLiga | 16 | 2026-08-23 | 7 | 9 | 16 |
| Ligue1 | 9 | 2026-08-23 | 0 | 9 | 9 |
| PrimeiraLiga | 17 | 2026-08-17 | 17 | 0 | 17 |
| Scotland | 11 | 2026-08-09 | 11 | 0 | 11 |
| SerieA | 8 | 2026-08-23 | 0 | 8 | 8 |
| Turkey | 9 | 2026-08-17 | 9 | 0 | 9 |

Aligned results in window: **98** matches written to settled_results + weekend actuals.

## 2. Score Predictions (match O/U lean)

- Joined **32** fixtures from the Aug 21 snap that have finals.
- Model O/U lean hit: **66%** (21/32)
- Pin O/U lean hit: **69%** (22/32)
- Total goals MAE: **1.26** · bias (actual−proj): **-0.65**

- Aligned (&lt;15pp): model hit 66% (21/32)
- HIGH profile: lean hit 65% (n=17, MAE=1.43)
- LOW profile: lean hit 100% (n=1, MAE=0.73)

| League | n | Model hit | Pin hit | MAE | Bias |
|--------|--:|----------:|--------:|----:|-----:|
| EPL | 9 | 78% | 78% | 0.75 | -0.52 |
| LaLiga | 7 | 57% | 71% | 1.32 | -0.38 |
| Ligue1 | 8 | 38% | 38% | 1.98 | -0.87 |
| SerieA | 8 | 88% | 88% | 1.07 | -0.81 |

### Largest total misses

- Ligue1 Nice vs Lorient: proj 3.42 → 0-0 (err -3.4) lean=OVER MISS
- Ligue1 Troyes vs Paris FC: proj 3.32 → 0-0 (err -3.3) lean=OVER MISS
- Ligue1 Lens vs Auxerre: proj 3.78 → 5-2 (err +3.2) lean=OVER HIT
- LaLiga Valencia vs Celta Vigo: proj 2.85 → 0-0 (err -2.9) lean=OVER MISS
- Ligue1 Le Havre vs Monaco: proj 3.58 → 0-1 (err -2.6) lean=OVER MISS
- EPL Nottingham Forest vs Leeds: proj 3.32 → 0-1 (err -2.3) lean=OVER MISS
- SerieA Frosinone vs Juventus: proj 3.27 → 0-1 (err -2.3) lean=OVER MISS
- LaLiga Real Betis vs Real Sociedad: proj 3.08 → 1-0 (err -2.1) lean=OVER MISS

## 3. Team totals (paper TRACK / CONFLICT)

- **All joined TT**: hit 48% (n=64) · paper units -8.22u @ Pin
- **Edge >=8pp**: hit 43% (n=30) · paper units -5.97u @ Pin
- **TRACK-style (>=8pp, not conflict)**: hit 33% (n=18) · paper units -6.84u @ Pin
- **CONFLICT >=15pp**: hit 58% (n=12) · paper units +0.86u @ Pin

### TRACK-style (>=8pp, not conflict)

- L **Lille** OVER 2.5 goals=2.0 | O+14.3pp | Angers vs Lille | -1.00u
- L **Monaco** OVER 1.5 goals=1.0 | O+14.2pp | Le Havre vs Monaco | -1.00u
- W **Brighton** OVER 1.5 goals=4.0 | O+14.0pp | Brighton vs Aston Villa | +1.02u
- L **Racing Santander** OVER 0.5 goals=0.0 | O+12.6pp | Getafe vs Racing Santander | -1.00u
- L **Venezia** OVER 1.5 goals=0.0 | O+12.5pp | Venezia vs Lecce | -1.00u
- L **Hull** UNDER 0.5 goals=2.0 | U+11.7pp | Hull vs Manchester United | -1.00u
- W **Cagliari** OVER 0.5 goals=1.0 | O+10.6pp | Parma vs Cagliari | +0.49u
- L **Valencia** OVER 1.5 goals=0.0 | O+10.5pp | Valencia vs Celta Vigo | -1.00u
- L **Manchester City** OVER 2.5 goals=2.0 | O+10.4pp | Manchester City vs Bournemouth | -1.00u
- L **Como** OVER 1.5 goals=1.0 | O+9.9pp | Udinese vs Como | -1.00u
- L **Parma** OVER 1.5 goals=0.0 | O+9.8pp | Parma vs Cagliari | -1.00u
- W **Espanyol** OVER 0.5 goals=1.0 | O+9.6pp | Espanyol vs Real Madrid | +0.71u
- W **Angers** UNDER 0.5 goals=0.0 | U+9.6pp | Angers vs Lille | +1.33u
- L **Atletico Madrid** OVER 2.5 goals=2.0 | O+8.9pp | Atletico Madrid vs Villarreal | -1.00u
- W **Torino** OVER 0.5 goals=1.0 | O+8.9pp | Torino vs AC Milan | +0.54u
- L **AC Milan** OVER 2.5 goals=2.0 | O+8.9pp | Torino vs AC Milan | -1.00u
- W **Barcelona** OVER 2.5 goals=5.0 | O+8.6pp | Elche vs Barcelona | +1.08u
- L **Juventus** OVER 1.5 goals=1.0 | O+8.3pp | Frosinone vs Juventus | -1.00u

### By lean side (all joined TT)

- OVER: hit 46% (n=52) · -7.85u
- UNDER: hit 58% (n=12) · -0.37u

## 4. Live packs (flagged PLAY ledger)

Open after settle: **0** · Settled: **5** · Total logged: **5**


### Play-by-play

- W 2026-08-14 **Primeira Liga AH e12%** Sporting CP vs Vitoria Guimaraes ah_away → 3.0-2.0 +0.88u
- W 2026-08-22 **EPL Unders** Hull vs Manchester United under → 2.0-0.0 +1.21u
- W 2026-08-22 **EPL short Overs** Brentford vs Tottenham over → 3.0-0.0 +0.70u
- L 2026-08-22 **EPL short Overs** Everton vs Crystal Palace over → 2.0-0.0 -1.00u
- L 2026-08-22 **EPL short Overs** Nottingham Forest vs Leeds over → 0.0-1.0 -1.00u

**Ledger total (all settled flags):** +0.79u

- **EPL Unders**: n=1 decided=1 W=1 units=1.21 ROI=+121.0% open=0
- **EPL short Overs**: n=3 decided=3 W=1 units=-1.3 ROI=-43.4% open=0
- **Primeira Liga AH e12%**: n=1 decided=1 W=1 units=0.88 ROI=+87.7% open=0

## 5. TT paper ledger summary

```json
{
  "updated_at": "2026-08-24T15:22:14.014715+00:00",
  "track": {
    "n": 20,
    "n_open": 14,
    "n_settled": 6,
    "n_decided": 6,
    "wins": 2,
    "hit": 0.3333,
    "units": -2.8,
    "roi": -0.4668
  },
  "conflict_watch": {
    "n": 10,
    "n_open": 3,
    "n_settled": 7,
    "n_decided": 7,
    "wins": 3,
    "hit": 0.4286,
    "units": -1.22,
    "roi": -0.1746
  },
  "n_total": 30
}
```

## 6. Recommendations (no live rule changes)

### Keep doing
- Score Predictions stay **information only** — week-1 lean hit vs Pin still does not clear a promotion bar.
- Keep CONFLICT ≥15pp flags on the Score tab.
- Keep TT paper TRACK logging; settle after each Full Model Refresh.
- Live pack rules: **unchanged**. Week-1 live n is tiny (4 new EPL flags) — do not retune edges.

### Inputs to watch
- **Bundesliga** still 0 current-season rows — season may not have started / Understat empty; refresh again when matchday 1 lands.
- **Primeira / Championship / Turkey / Eredivisie / Belgium** FD files lag (many still end ~16–17 Aug). Weekend form for those leagues is thin until FD catches up.
- **MLS refresh failed** this run — fix or skip; do not trust MLS Score/TT until aligned.
- EPL / Serie A / La Liga / Ligue 1: Understat extras are feeding form+xG — good; re-run Score Predictions before next slate.

### Optional process tweaks (not pack rules)
- Archive dated `SCORE_PREDICTIONS_YYYYMMDD.csv` automatically on each Score refresh (done manually for Aug 21).
- Expand Full Model Refresh default leagues to include Score slate leagues (Ligue1, etc.) so week-end retros are one click.
- Hull match Under PLAY won while Hull TT Under 0.5 lost (Hull scored 2) — treat match O/U and TT as separate signals.
