# Gameday prediction sheet

Production path for live / gameday multi-league sheets. Same inference stack as walk-forward backtests.

**Daily entry point:** double-click **`Launch_Gameday.bat`** → **Scan All 5 Systems**.  
Full UI guide: **[`docs/GAMEDAY_UI.md`](GAMEDAY_UI.md)**.

## Protected systems (rules frozen)

1. EPL Unders — Under 2.00–4.00 @ ≥8%  
2. EPL short Overs — Over 1.60–2.50 @ ≥10%  
3. Bundesliga Unders — Under 1.70–2.50 @ ≥10%  
4. La Liga Home ML — Home @ ≥8%, max 1.80  
5. Serie A Away ML — Away @ ≥3%, max 2.00  

Sheet columns `systems_flagged`, `odds_status`, and `flag_*` make it obvious which system(s) fired and when Pinnacle is missing.

## Fixtures + Pinnacle odds are automatic

Per-league refresh (UI buttons or `--refresh-fixtures` / `--refresh-odds`):

1. **Upcoming fixtures** (Pulse for EPL; football-data / league APIs for others)
2. **Pinnacle** sharp OU 2.5 + 1X2 when offered

Artifacts:

- `data/interim/fixtures_upcoming_{League}.csv` (+ `.meta.json`)
- `data/interim/pinnacle_ou25_{League}.csv` (+ `.meta.json`)
- `data/gameday/odds_pinnacle*.csv` (`ref_*` / `pin_*`)

Optional **your book**: `data/gameday/odds.csv` with `match_id,book_over25,book_under25`.

Pack flags use **Pinnacle**. Sheet shows fair / Pin / book / edge vs each.

## Preferred: desktop UI

Double-click **`Launch_Gameday.bat`**. Guide: **[`docs/GAMEDAY_UI.md`](GAMEDAY_UI.md)**.

## Quick start (CLI)

```bash
# refresh fixtures only
.venv\Scripts\python.exe scripts/update_data.py --fixtures-only

# run sheet (defaults to auto fixtures)
.venv\Scripts\python.exe scripts/run_gameday_sheet.py ^
  --odds-file data/gameday/odds.csv ^
  --out data/processed/gameday_sheet.csv ^
  --fast
```

| Flag | Effect |
|------|--------|
| `--update-data` | Full update (results + xG + align + fixtures) |
| `--refresh-fixtures` | Fixtures-only refresh before the sheet |
| `--fixtures path` | **Override** auto fixtures (not recommended for live) |
| `--late-info path.csv` | Confirmed XI / injuries |
| `--fast` | Skip residual OOS fit |

## Odds CSV (manual)

Keyed by `match_id` from the auto fixtures file.

| Column | Role |
|--------|------|
| `ref_over25`, `ref_under25` | Preferred sharp two-way reference |
| `book_over25`, `book_under25` | Fallback / extra book |
| Any other `*over25*` / `*under25*` | Averaged into **consensus** |

## Late info CSV (optional)

Confirmed XI / injuries / `lam_mult_*` overrides when available.

## Output / packs

| Column | Meaning |
|--------|---------|
| `proj_*_goals` | Model λ / μ / total |
| `fair_odds_*` / `fair_american_*` | Model fair prices |
| `edge_*_vs_ref` | Same edge formula as backtest |
| `flag_EPL_aggressive_under` | Production Under pack |
| `flag_EPL_overs_short` | Experimental short Over pack |

Packs stay **separate**. Edge = `model_prob − power_devig_fair(side)` when both OU prices exist.

## Architecture

```text
update_data → FD + Understat + align → Pulse upcoming fixtures
gameday → auto fixtures → features → model → sheet + pack flags
```
