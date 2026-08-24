# Multi-League Gameday UI

## Start here

Double-click **`START_HERE_LIVE.bat`**.

That is the live betting tool. Short guide: [`DAILY_GUIDE.md`](DAILY_GUIDE.md)

Backtesting = Cursor / `scripts/run_backtest.py` — not daily use.

---

## Tabs

| Tab | Purpose |
|-----|---------|
| **Daily Scan (live betting)** | 4 steps → PLAY / WATCH |
| **Score Predictions** | Projected scores + O/U 2.5% + data strength (info only) |
| **System Performance** | Backtest vs live vs recent form; every flagged PLAY |

---

## Daily Scan steps

| Step | Button | Does |
|------|--------|------|
| 1 | **Update Data Sources** | Light — fixtures (+ optional recent results) |
| 2 | **Full Model Refresh** | Heavy — xG, align, feature store, context layers |
| 3 | **Update Odds** | Pinnacle OU 2.5 · 1X2 · AH |
| 4 | **Run Scan / Pipeline** | All 6 systems → PLAY / WATCH |

---

## Protected systems (rules frozen)

| # | System | Rules | Status |
|---|--------|-------|--------|
| 1 | EPL Unders | Under 2.00–4.00 @ ≥8% | Production |
| 2 | EPL short Overs | Over 1.60–2.50 @ ≥10% | Production |
| 3 | Bundesliga Unders | Under 1.70–2.50 @ ≥10% | Paper |
| 4 | La Liga Home ML | Home @ ≥8%, max 1.80 | Paper |
| 5 | Serie A Away ML | Away @ ≥3%, max 2.00 | Paper |
| 6 | Primeira AH e12% | AH @ ≥12%, max 1.90 | Paper live (e10 optional sibling) |

Score Predictions: quality-weighted O/U leans; HIGH/LOW totals; **Why** = xG / form / Elo; **CONFLICT** = model vs Pin ≥15pp. Team totals paper card (`TT_TODAY.md`) + ledger. Info only — not a live pack. Data sources: [`DATA_SOURCES.md`](DATA_SOURCES.md)
