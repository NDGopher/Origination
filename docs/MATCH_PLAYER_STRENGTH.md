# Match-level player strength (Understat rosters)

Leakage-free expected-XI strength from Understat `getMatchData/{id}` rosters.

## Pipeline

1. `scripts/ingest_understat_match_rosters.py` → `data/raw/understat/match_rosters/{id}.json`
   + `data/interim/understat_match_rosters.parquet`
2. Provider: `MatchLevelPlayerStrengthProvider` (`features/match_player_strength.py`)
3. YAML: `features.context_adjustments.lineups.provider: understat_match_players`

## Design

- **Expected XI** = previous fixture starters (minutes ≥ 45), top 11 by minutes.
- **Ratings** = expanding prior `(xG+xA)/90` and `xGBuildup/90` (strictly before match t).
- **Confirmed lineup** = always `False` (no pre-match XI feed yet).
- Same-match minutes/xG are **never** used for that match's λ.

## Status (iter10)

Measured on hierarchical stack; **not promoted** (hurts 1X2 LL and ROI at coef 0.03–0.08).
Wired and available for future work (confirmed XI / better DEF signal).
