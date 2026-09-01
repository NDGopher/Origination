# Line movement — PLAYS / WATCH radar → close

Updated: 2026-09-01T17:26:36.066911+00:00

Tracks Pin prices on **every daily scan** from first radar (PLAY or WATCH) 
through kickoff. **CLV%** = `(first_or_entry_odds / close_odds) - 1` on the bet side. 
Positive = line steamed **toward** our pick by close (early entry rewarded).

Backtests use **closing** Pin; live CLV here tells you whether to bet early or wait.

**Actions:** `BET_NOW` = steam toward us · `WAIT` = line moving against · 
`MONITOR` = flat/thin history · `INSUFFICIENT_DATA` = first observation only.

**Audit trail:** `data/gameday/line_scan_log.jsonl` (append-only JSONL per scan).

**Tracking:** 11 open · **Closed/settled:** 2

## By system

| System | Open | Closed | Avg CLV first | % toward us |
|--------|-----:|-------:|--------------:|------------:|
| EPL Unders | 2 | 0 | — | — |
| EPL short Overs | 4 | 0 | — | — |
| Bundesliga Unders | 1 | 0 | — | — |
| La Liga Home ML | 2 | 0 | — | — |
| Serie A Away ML | 2 | 1 | +0.0% | 0% |
| Primeira Liga AH e12% | 0 | 1 | +0.0% | 0% |

## Open — CLV vs last scan (bet now or wait?)

| Action | Match | System | First | Now | CLV vs now | Steam | Obs |
|--------|-------|--------|------:|----:|-----------:|-------|----:|
| **MONITOR** | Ipswich vs Liverpool | EPL short Overs | 1.535 | 1.535 | +0.0% | flat | 1 |
| **BET_NOW** | Manchester City vs Coventry | EPL Unders | 3.230 | 3.220 | +0.3% | flat | 2 |
| **MONITOR** | Paderborn vs Freiburg | Bundesliga Unders | 2.260 | 2.270 | -0.4% | flat | 2 |
| **BET_NOW** | Villarreal vs Deportivo La Coruna | La Liga Home ML | 1.524 | 1.495 | +1.9% | toward_us | 2 |
| **WAIT** | Arsenal vs Chelsea | EPL short Overs | 1.800 | 1.820 | -1.1% | against_us | 2 |
| **BET_NOW** | Crystal Palace vs Ipswich | EPL short Overs | 1.980 | 1.901 | +4.2% | toward_us | 2 |
| **BET_NOW** | Sassuolo vs Juventus | Serie A Away ML | 1.599 | 1.599 | +0.0% | flat | 2 |
| **MONITOR** | Bournemouth vs Brentford | EPL short Overs | 1.662 | 1.662 | +0.0% | flat | 2 |
| **BET_NOW** | Coventry vs Brighton | EPL Unders | 2.060 | 2.060 | +0.0% | flat | 2 |
| **MONITOR** | Celta Vigo vs Malaga | La Liga Home ML | 1.763 | 1.763 | +0.0% | flat | 2 |
| **MONITOR** | Torino vs Roma | Serie A Away ML | 1.690 | 1.690 | +0.0% | flat | 2 |

## Open — observation timelines

### Ipswich vs Liverpool (EPL short Overs)
  - 2026-08-31T14:41 · 1.535 · 75h to KO

### Manchester City vs Coventry (EPL Unders)
  - 2026-08-31T14:41 · 3.230 · 94h to KO
  - 2026-09-01T14:49 · 3.220 (-0.010) · 94h to KO

### Paderborn vs Freiburg (Bundesliga Unders)
  - 2026-08-31T14:41 · 2.260 · 94h to KO
  - 2026-09-01T14:49 · 2.270 (+0.010) · 94h to KO

### Villarreal vs Deportivo La Coruna (La Liga Home ML)
  - 2026-08-31T14:41 · 1.524 · 99h to KO
  - 2026-09-01T14:49 · 1.495 (-0.029) · 99h to KO

### Arsenal vs Chelsea (EPL short Overs)
  - 2026-08-31T14:41 · 1.800 · 120h to KO
  - 2026-09-01T14:49 · 1.820 (+0.020) · 120h to KO

### Crystal Palace vs Ipswich (EPL short Overs)
  - 2026-08-31T14:41 · 1.980 · 262h to KO
  - 2026-09-01T14:49 · 1.901 (-0.079) · 262h to KO

### Sassuolo vs Juventus (Serie A Away ML)
  - 2026-08-31T14:41 · 1.599 · 264h to KO
  - 2026-09-01T14:49 · 1.599 (+0.000) · 264h to KO

### Bournemouth vs Brentford (EPL short Overs)
  - 2026-08-31T14:41 · 1.662 · 262h to KO
  - 2026-09-01T14:49 · 1.662 (+0.000) · 262h to KO

### Coventry vs Brighton (EPL Unders)
  - 2026-08-31T14:41 · 2.060 · 285h to KO
  - 2026-09-01T14:49 · 2.060 (+0.000) · 285h to KO

### Celta Vigo vs Malaga (La Liga Home ML)
  - 2026-08-31T14:41 · 1.763 · 287h to KO
  - 2026-09-01T14:49 · 1.763 (+0.000) · 287h to KO

### Torino vs Roma (Serie A Away ML)
  - 2026-08-31T14:41 · 1.690 · 283h to KO
  - 2026-09-01T14:49 · 1.690 (+0.000) · 283h to KO

## Closed — CLV vs kickoff close

Average first→close CLV: **+0.0%** · toward us (>2%): **0** · against us (<-2%): **0**

| Match | First | Close | CLV first | CLV entry | Steam | Timing |
|-------|------:|------:|----------:|----------:|-------|--------|
| Lecce vs Roma | 1.476 | 1.476 | +0.0% | +11.4% | flat | ledger entry beat first radar — late add was fine |
| Sporting Braga vs Vitoria Guimaraes | 1.885 | 1.885 | +0.0% | — | flat | flat market — early vs close similar |
