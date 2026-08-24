# Daily guide — LIVE betting

## What to double-click

**`START_HERE_LIVE.bat`**

That is the only file for daily live use.

(`Launch_Gameday.bat` is the same thing. Backtesting stays in Cursor / `scripts/` — do not use those for daily betting.)

---

## Two tabs

| Tab | Purpose |
|-----|---------|
| **Daily Scan (live betting)** | The 4 steps + **WHAT TO BET** |
| **Score Predictions** | Today’s (and upcoming) projected scores — **info only**, not a betting system |
| **System Performance** | Backtest vs live vs recent form for each of the 6 systems; every flagged PLAY |

---

## Daily Scan — buttons in order

| Step | Button | What it does |
|------|--------|----------------|
| **1** | **Update Data Sources** | Light / daily. Fixtures (+ optional recent results). Does **not** rebuild the full model. |
| **2** | **Full Model Refresh** | Heavy. Current-season xG / Understat / align / feature store / form / strengths / refs / coaching / context. |
| **3** | **Update Odds** | Pinnacle OU 2.5 + moneyline + Asian Handicap. |
| **4** | **Run Scan / Pipeline** | Evaluates all 6 protected systems. Shows **PLAY** / **WATCH**. |

### Typical day

- **Most days:** 1 → 3 → 4  
- **After matchdays / model >~3 days old:** also run **2** (add `--with-score-leagues` after big Score-slate weekends)
- **Close to kickoff:** re-run **3**
- **After a big weekend:** Settle finished games, or `python scripts/run_week1_gauntlet.py` — see [`WEEK1_GAUNTLET.md`](WEEK1_GAUNTLET.md)

---

## Score Predictions tab

Click **Refresh Score Predictions**.

- Window: **next 24 hours (rolling by kickoff)**, plus later slate (7 days)
- Ranked by **quality-weighted Over / Under lean** (NEXT 24H first; Pin conflicts ranked lower)
- Headline shows highest / lowest projected totals plus strongest Over / Under
- Columns: projected score, **H/L** (HIGH/LOW total), Over/Under 2.5 %, lean, Pin O/U, Model−Pin, **Why** (xG / form / Elo), confidence
- **Team totals (paper track):** Refresh Score Predictions writes:
  - `experiments/gameday_scan/TT_TODAY.md` — best TT leans today + **why**
  - `data/gameday/tt_ledger.csv` — paper log: **team OVER** on 0.5/1.5/2.5 (≥10pp vs Pin; Unders not TRACK)
  - Snapshots under `experiments/gameday_scan/tt_history/` for forward Pin-EV research
  - UI: **Open TT today** / **TT ledger** on the Score tab; **Settle finished games** also settles TT
  - Research (Pin-priced, not flat 2.00): `experiments/score_predictions/TT_OVER_PIN_VALUE.md`
- **HIGH / LOW** = projected total extremes (most useful informational slice historically)
- **CONFLICT** = model vs Pinnacle ≥15pp — historically the model loses these (especially Unders vs Pin Overs)
- Serie A and Belgium totals are slightly cooled on this tab only (historical bias). Live packs are unchanged.

This tab does **not** replace the scan. Do not treat a lean as a PLAY unless Daily Scan says so. Team totals are **paper / learning only** — not a 7th live pack.

### Backtesting team totals
- **Calibration (done):** model λ vs actual team goals @ 0.5 / 1.5 / 2.5 — see `experiments/score_predictions/TEAM_TOTALS.md` (no historical Pin TT closes on football-data).
- **True EV vs Pin:** starts **forward** from this ledger + daily `tt_history/` snapshots (no past Pin TT archive to replay).

---

## Reading PLAY / WATCH

| Word | Meaning |
|------|---------|
| **PLAY** | Bet this. |
| **WATCH** | Close but not quite — usually skip. |
| **NO PLAYS** | Nothing to bet today. |

---

## Protected systems (never change)

1. EPL Unders — Under 2.00–4.00 @ ≥8%  
2. EPL short Overs — Over 1.60–2.50 @ ≥10%  
3. Bundesliga Unders — Under 1.70–2.50 @ ≥10%  
4. La Liga Home ML — Home @ ≥8%, max 1.80  
5. Serie A Away ML — Away @ ≥3%, max 2.00  
6. Primeira AH e12% — AH @ ≥12%, max 1.90 (live primary; e10 optional sibling)  

Status: [`STATUS_BOARD.md`](STATUS_BOARD.md)

---

## CLI (optional / research)

```bash
.venv\Scripts\python.exe scripts/refresh_gameday_data.py --fixtures-only
.venv\Scripts\python.exe scripts/refresh_full_model.py
.venv\Scripts\python.exe scripts/refresh_gameday_odds.py
.venv\Scripts\python.exe scripts/run_daily_scan.py --no-refresh
.venv\Scripts\python.exe scripts/build_score_predictions.py
```

Backtests: `scripts/run_backtest.py`, `scripts/run_league_backtest.py` — not part of daily live use.
