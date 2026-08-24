# Origination

Production-grade **originator-style** predictive modeling for European football.

The model produces independent, well-calibrated probabilities from data and features, then compares them to bookmaker odds. We do **not** follow the market. Success is measured by walk-forward **Closing Line Value (CLV)** against real historical closing odds — not in-sample ROI or win rate.

## Core principles

1. **Originator first** — true probabilities from first principles; market is the benchmark, not the teacher.
2. **Walk-forward CLV is sacred** — expanding/rolling windows, no leakage, vig removed, closing odds only.
3. **Features must earn their keep** — if a feature group does not improve out-of-sample CLV, kill it.
4. **Same code path** for backtest and live prediction.

## Scope (MVP → expand)

| Layer | Status |
|-------|--------|
| EPL 2014/15+ via football-data.co.uk | Implemented |
| Understat xG / PPDA | Implemented |
| FBref advanced stats | Scaffold (incremental) |
| Feature store (lagged, leakage-free) | Implemented |
| Independent / Dixon–Coles Poisson | Implemented |
| Walk-forward CLV backtester | Implemented |
| LightGBM ensemble + calibration | Scaffold |
| Big 5 leagues | Config-ready |
| Live odds API | Manual/CSV input for now |

**Markets:** Match Result (1X2), Over/Under 2.5, Asian Handicap (main lines).

## Repository layout

```
data/
  raw/          # source dumps (football_data, understat, fbref)
  interim/      # cleaned / aligned
  processed/    # feature matrices, predictions
src/origination/
  data_ingestion/
  features/
  models/
  backtesting/
  prediction/
  utils/
configs/        # YAML experiment configs
experiments/    # logged run artifacts
scripts/        # CLI entry points
tests/
notebooks/      # exploration only
```

## Setup

```bash
# Python 3.10+
cd Origination
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -e ".[dev]"
```

## Daily betting (protected systems)

Double-click **`START_HERE_LIVE.bat`**, then:

1. **Update Data Sources** (daily light)
2. **Full Model Refresh** (when model stale)
3. **Update Odds**
4. **Run Scan / Pipeline**

Read **WHAT TO BET** in the UI. Guide: [`docs/DAILY_GUIDE.md`](docs/DAILY_GUIDE.md) · Status: [`docs/STATUS_BOARD.md`](docs/STATUS_BOARD.md)

## Data update

```bash
# Premier League, football-data + Understat (default config)
python scripts/update_data.py --config configs/default.yaml

# Specific seasons
python scripts/update_data.py --config configs/default.yaml --seasons 2014,2015,2016
```

Raw CSVs land in `data/raw/`; aligned Parquet in `data/interim/` and `data/processed/`.

## Walk-forward backtest

```bash
python scripts/run_backtest.py --config configs/default.yaml
```

This will:

1. Load aligned match + odds data
2. Build pre-match features (chronology enforced)
3. Train Poisson / Dixon–Coles on expanding history
4. Predict next fold, compare to **closing** odds (vig-removed)
5. Log CLV, ROI, Brier, log-loss, and breakdowns under `experiments/`

Primary metrics to watch:

- Average CLV (probability and odds space)
- Simulated ROI (fixed unit / fractional Kelly)
- Bet count, hit rate
- Brier / log-loss
- Season / edge-bucket / market breakdowns

## Feature ablation

```bash
python scripts/run_backtest.py --config configs/ablation_basic.yaml
python scripts/run_backtest.py --config configs/ablation_xg.yaml
```

Toggle feature groups in YAML under `features.groups`. Only keep groups that improve OOS CLV.

## Upcoming predictions

```bash
python scripts/predict_upcoming.py --config configs/default.yaml --odds-file path/to/odds.csv
```

Uses the **same** feature + model path as backtesting. Outputs fair probs, edge vs book, and suggested stake.

## Configuration

All knobs live in `configs/*.yaml`:

- leagues / seasons
- feature groups on/off
- model type and hyperparameters
- backtest window (expanding vs rolling)
- edge threshold, staking (flat / Kelly / caps)
- odds book preference (Pinnacle closing → B365 → Avg)

## Design decisions (MVP)

| Decision | Rationale |
|----------|-----------|
| football-data closing columns first | Real CLV requires real closes; AvgC / PSC / B365C preferred |
| Power method for vig removal | Stable for 1X2 and two-way markets |
| Dixon–Coles before GBMs | Correct baseline measurement before complexity |
| Parquet + DuckDB/SQLite optional | Fast columnar I/O; SQL when needed |
| Central team name map | Align football-data, Understat, FBref without silent mismatches |
| Explicit leakage tests | Chronology assertions fail hard if a feature peeks ahead |

## Iteration rule

> Does this improve out-of-sample CLV against real closing odds?

If not — revert or discard. Win rate alone is meaningless.

## Tests

```bash
pytest tests/ -q
```

## License

MIT — for research / personal use. Betting involves risk; this is not financial advice.
