# Data sources — current-season freshness

Last updated: 2026-08-24

This is the source of truth for **what the Full Model Refresh actually uses**. Live pack **rules are unchanged**.

Daily flow stays: **Data → Full Model Refresh → Odds → Scan**.

After matchweeks, run **Full Model Refresh** (optionally `--with-score-leagues`) then `scripts/run_week1_gauntlet.py` / Settle to grade Score Predictions + TT paper + live ledger. See [`WEEK1_GAUNTLET.md`](WEEK1_GAUNTLET.md).

## What broke (and the fix)

football-data.co.uk **does not yet publish 2026/27 files** for most top leagues. `requests` followed redirects and **poisoned the local cache**:

| Requested | What the site actually served | Status |
|-----------|-------------------------------|--------|
| `E0.csv` (EPL) | **301 → `EC.csv`** (National League) | Rejected. Do not cache. |
| `SP1.csv` (La Liga) | **301 → `P1.csv`** (Primeira) | Rejected. Do not cache. |
| `D1.csv` / `I1.csv` / `F1.csv` | HTTP 300 (file missing) | Treated as unpublished. |
| `P1.csv` (Primeira) | Real file, `Div=P1`, includes `HxG`/`AxG` | **Used.** |

The ingest now:

1. Refuses redirects to a **different** `{code}.csv`
2. Treats HTTP 300 as missing (not fatal)
3. Requires `Div` == requested league code; deletes poisoned files
4. Treats a missing current-season FD file as **normal** early-season behaviour

Understat `getLeagueData/{league}/2026` **works**. Empty `dates` means the season has not started (EPL / Bundesliga / Serie A as of 17 Aug). That is saved as a valid empty season, not a fetch failure. **La Liga 2026 already has results** — those now enter the aligned table even when FD `SP1/2627` is missing.

## Sources in use

| Need | Source | Freshness | Notes |
|------|--------|-----------|-------|
| Historical results + closing odds | football-data.co.uk cached seasons through **2025/26** | Static until FD publishes 2627 | Training / CLV / live-system backtests |
| Current results + xG (Big 5) | **Understat** `isResult` rows | Re-fetched on Full Model Refresh | Appended when FD has no matching row (no closing odds on those rows) |
| Current Primeira results + xG | **FD `P1/2627`** (`HxG`/`AxG`) | Re-fetched on Full Model Refresh | No Understat for Primeira (by design) |
| Current EPL results (when they exist) | **Premier League Pulse** `statuses=C` | Re-fetched on Full Model Refresh | Goals only until Understat fills xG; 0 completed as of 17 Aug |
| Upcoming EPL fixtures | Pulse `statuses=U,L` | Light Data update + Full Refresh | Already in daily workflow |
| Other-league fixtures | football-data blank rows / Understat non-results / existing scrapers | Light Data update | FD fixture probe also rejects wrong-`Div` redirects |
| Team strength / form | Feature store on aligned matches (goals, xG EWM, Elo, context) | Rebuilt on Full Model Refresh | Extra current-season rows **do** feed form |
| Closing 1X2 / OU / AH for live scan | **Pinnacle** (Update Odds) | Separate step | Unchanged |
| FBref / Opta advanced | Scaffold only | Not used | Do not rely on it |

## How Full Model Refresh stays fresh

For each live league (EPL, Bundesliga, La Liga, Serie A, Primeira):

1. Force-download **current** FD season; skip if unpublished or wrong `Div`
2. Force-download **current** Understat season (empty = not started)
3. EPL only: pull Pulse completed results
4. Align: FD left-join Understat xG → fill xG from FD `HxG` when needed → **append Understat/Pulse-only results**
5. Rebuild feature matrix + upcoming fixtures
6. Stamp `data/gameday/last_data_update.json` with `current_season` counts (`n`, `n_football_data`, `n_extra`, `n_xg`, `latest`)

Historical seasons stay cached. Odds are **not** part of this step.

Extra current-season rows have **no closing odds**. They are for form / xG / Score Predictions inputs. They are **not** priced backtest bets and they do **not** change the 6 protected packs.

## Remaining gaps (honest)

- **Bundesliga 2026/27:** still 0 aligned current rows as of 24 Aug (season not in Understat/FD yet).
- **Primeira / Championship / Turkey / Eredivisie / Belgium:** FD current files often lag 5–7 days behind kickoff — form for Score/TT on those leagues stays stale until FD updates.
- **MLS:** refresh can fail; treat Score/TT as unreliable until aligned.
- **La Liga / EPL / Serie A / Ligue 1 closing odds:** often still missing on FD; results + xG come from Understat extras.
- **Score Predictions:** information only. Week-1 (Aug 21–23 snap): model O/U lean ~66% vs Pin ~69% on n=32; model totals biased low (actual−proj ≈ −0.65). TT paper TRACK rough (−6.8u on n=18) — keep tracking, do not promote.
- **FBref:** not implemented.

When football-data publishes real `E0`/`SP1`/`D1`/`I1` 2627 files, Full Model Refresh will pick them up automatically (redirect + `Div` guards stay in place).

Weekly after big slates: `python scripts/refresh_full_model.py --with-score-leagues` then `python scripts/run_week1_gauntlet.py`.