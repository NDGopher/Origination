# Origination — system status board

Last updated: 2026-08-24 (week-1 gauntlet; Full Model Refresh; live+TT settle)

## Live gameday (protected — do not change rules without a deliberate wire-up)

| # | System | Pack | Rules | Status |
|---|--------|------|-------|--------|
| 1 | EPL Unders | `EPL_aggressive` | Under 2.00–4.00 @ ≥8% | **Production** |
| 2 | EPL short Overs | `EPL_overs_short_exp` | Over 1.60–2.50 @ ≥10% | **Production** |
| 3 | Bundesliga Unders | `Bundesliga_unders_short_exp` | Under 1.70–2.50 @ ≥10% | **Paper** |
| 4 | La Liga Home ML | `LaLiga_home_ml_short_exp` | Home @ ≥8%, max 1.80 | **Paper** |
| 5 | Serie A Away ML | `SerieA_away_ml_exp` | Away @ ≥3%, max 2.00 | **Paper** |
| 6 | **Primeira AH e12%** | `PrimeiraLiga_ah_e12_exp` | AH @ ≥12%, max 1.90 | **Paper live** (primary) |

### Primeira AH siblings

| Pack | Role | n | ROI | t | Seasons+ | Max DD |
|------|------|--:|----:|--:|---------:|-------:|
| **e12 (live primary)** | Default live scan | 161 | +19.2% | 3.05 | 10/10 | −3.7u |
| e10 (wider sibling) | Optional only — not in default scan | 242 | +12.6% | 2.46 | 8/10 | −6.8u |
| e10-only slice (10–12%) | Dropped by e12 | 81 | −0.3% | −0.04 | 6/10 | −11.3u |

Promoted 2026-08-14 from iter25 evidence. **No pack rules changed 17 Aug.**

## Live performance tracking (new)

Append-only ledger: [`data/gameday/live_ledger.csv`](../data/gameday/live_ledger.csv)  
Human report: [`docs/LIVE_LEDGER.md`](LIVE_LEDGER.md)

- Daily scan now records every **PLAY** after writing `QUALIFIED_PLAYS.csv` (rules untouched).
- Settle: add finals to `experiments/weekend_retro/actuals.csv` or `data/gameday/settled_results.csv`, then `python scripts/update_live_ledger.py --settle-only`.
- Scan snapshots: `experiments/gameday_scan/history/`.
- **n=1 ROI is not a system evaluation.** Walk-forward history still governs promotion.

Current ledger: **1 settled** (Primeira AH e12 Sporting away +1.5, 3–2, **+0.877u**) · **4 open** (EPL Saturday 22 Aug: Hull Under; Brentford / Everton / Forest Overs).

## Daily workflow

1. Double-click **`START_HERE_LIVE.bat`**
2. **Daily Scan** tab: Data → Full Model Refresh (if stale) → Odds → Scan
3. **Score Predictions** tab: next **24h + through tomorrow** (real kickoffs) + later slate; Pin OU when available; ranked by strongest O/U lean

**Model stamp:** Full Model Refresh **2026-08-21 ~09:29 CDT** (`ok: true`, all 5 live leagues). Current-season results: La Liga **9** (latest 20 Aug), Primeira **17** (latest 17 Aug). EPL / Bundesliga / Serie A still **0** FD 2627 files (unpublished). Sources: [`DATA_SOURCES.md`](DATA_SOURCES.md)

**Today’s scan (21 Aug):** **1 PLAY** — Hull vs Man Utd Under 2.5 (EPL Unders, +10.2% vs Pin). **3 WATCH** — Arsenal–Coventry Under (6.2%); Forest–Leeds Over (9.5%); Sporting–Alverca AH home −1.75 (10.3%). Fresh odds dropped the Aug-17 short Overs (Brentford / Everton / Forest) below the live bar — ledger still has those as open until settled; treat **today’s scan** as the betting card.

Guide: [`DAILY_GUIDE.md`](DAILY_GUIDE.md)

## Weekend Score Predictions (14–16 Aug)

Friday snapshot vs 23 finals: model O/U lean **11/23 (48%)**, Pin **13/23 (57%)**. Worst misses: strong model Unders that fought Pin by ≥20pp (Sporting 3–2, Willem II 1–4). Report: [`experiments/weekend_retro/REPORT.md`](../experiments/weekend_retro/REPORT.md)

EPL / Bundesliga / Serie A had no weekend slate.

## Score Predictions coverage

Registered leagues with fixtures/odds support: EPL, Bundesliga, La Liga, Serie A, Championship, Ligue 1, Eredivisie, Primeira, Belgium, MLS, Scotland, Turkey (+ Austria config pending good FD source).

Turkey excluded from score slate (Pinnacle ID `1843` is German 2.Bundesliga). Scotland fixtures still fail Pin HTTP.

Rebuilt 21 Aug midday: **99** upcoming (**25** next 24h). **32 HIGH** / **7 LOW**. **4 Pin conflicts**. Leagues on slate: EPL, La Liga, Serie A, Ligue 1, Primeira, Championship, Eredivisie, Belgium, MLS, **Scotland**, **Turkey**.

**Team totals (new, info only):** Pin team-total main lines pulled; model vs Pin edges in [`SCORE_TEAM_TOTALS.csv`](../experiments/gameday_scan/SCORE_TEAM_TOTALS.csv). Historical team O/U lean (no Pin closes on FD): **63.3% @ 1.5**, **73.9% @ 0.5**, **82.5% @ 2.5** — better directional hit than match O/U 2.5, but **not** a betting pack (no historical Pin TT EV yet). Report: [`TEAM_TOTALS.md`](../experiments/score_predictions/TEAM_TOTALS.md)

Pin ID fixes: Turkey **2592** (was 1843 = German 2.Bundesliga); Scotland **2421**.

**Do not promote to a live pack.** Historical test (3 seasons, `--fast` path): Grade A model O/U **55.9%** vs Pin **59.8%**. CONFLICT ≥15pp: model **51.0%** vs Pin **65.4%**. Model Under vs Pin Over ≥15pp: **36.2% vs 63.8%**. HIGH totals are the most useful slice (62.6% hit) but run **+0.33 hot**.

Score-tab only (not live): Serie A / Belgium totals cooled by 50% of historical bias (−0.16 / −0.11). Weak O/U leagues (Serie A, Ligue 1, Belgium, Championship) never get HIGH O/U confidence.

Reports: [`HISTORICAL.md`](../experiments/score_predictions/HISTORICAL.md) · [`CALIBRATION.md`](../experiments/score_predictions/CALIBRATION.md)

## Research / hunt (iter27 + iter28)

Bar unchanged: n≥120, t≥2.0, seasons+≥70% with ≥8 seasons, last-3 ≥2 positive, boot CI lo>0, boot P(ROI>0)≥95%, max DD > −8u, ROI≥5%.

| Hunt | Result |
|------|--------|
| Unused markets on EPL / Bundesliga / La Liga / Serie A / Primeira (iter27) | **No new leagues.** La Liga Home *siblings* still clear — **do not wire**. Primeira e10 remains the known optional sibling. |
| MLS / Austria | SKIP — no predictions.parquet. FD 2627 still untrusted (wrong `Div`). |
| Closest new-league watch | Still iter26 **Scotland Over 1.4–2.0 @ e5** (t=2.26, CI lo +2.3%, **DD −8.9u fails**) |
| iter28 unused leagues (Eredivisie, Belgium, Scotland, Championship, Ligue 1) | **0 cells cleared** (DC + intercept + temperature, no residual). Closest: Scotland Over 1.4–2.0 @ e5 (+6.9%, t=1.29, **DD −9.3u**). Weaker than the iter26 residual WF watch. **Do not wire.** |

Report: [`experiments/iter27/REPORT.md`](../experiments/iter27/REPORT.md) · [`experiments/iter28/REPORT.md`](../experiments/iter28/REPORT.md)

## Pinnacle markets (live)

OU 2.5 · 1X2 · Asian Handicap main line.
