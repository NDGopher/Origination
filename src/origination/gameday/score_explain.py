"""Explain Score Predictions from pre-match features (information only).

Score-only league totals offsets are display/calibration for this tab.
They do not change the 6 live packs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from origination.models.poisson import markets_from_matrix, score_matrix
from origination.utils.league_registry import get_league
from origination.utils.team_names import DEFAULT_MAPPER

ROOT = Path(__file__).resolve().parents[3]

# Mean (proj − actual) on last 3 complete seasons, --fast path (HISTORICAL.md).
# Score tab only: apply 50% shrink when |bias| >= 0.15 so we do not over-correct.
SCORE_TOTAL_BIAS = {
    "SerieA": 0.31,
    "Belgium": 0.21,
    "Ligue1": 0.12,
    "LaLiga": 0.05,
    "Scotland": 0.01,
    "EPL": -0.01,
    "Championship": -0.05,
    "Bundesliga": -0.05,
    "Eredivisie": -0.08,
    "MLS": -0.10,
    "PrimeiraLiga": -0.11,
}
SCORE_OFFSET_MIN_ABS = 0.15
SCORE_OFFSET_SHRINK = 0.5

# Historical O/U lean is near coin-flip vs a clearly better Pin — never HIGH confidence.
WEAK_OU_LEAGUES = frozenset({"SerieA", "Ligue1", "Belgium", "Championship"})


def _fmt_opt(x: float | None, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def _canon(name: str) -> str:
    key = str(name or "").strip().casefold()
    if not key:
        return ""
    return DEFAULT_MAPPER._map.get(key, str(name).strip())


def score_only_offset(league_key: str) -> float:
    """Additive total-goals shift for Score Predictions (negative cools a hot league)."""
    bias = float(SCORE_TOTAL_BIAS.get(league_key, 0.0))
    if abs(bias) < SCORE_OFFSET_MIN_ABS:
        return 0.0
    return -SCORE_OFFSET_SHRINK * bias


def apply_score_only_projection(
    proj_home: float,
    proj_away: float,
    league_key: str,
    p_over: float | None = None,
    p_under: float | None = None,
) -> tuple[float, float, float, float | None, float | None, float]:
    """Shift lambdas by the Score-only offset; nudge O/U by the Poisson delta.

    Live packs are unchanged. Returns (home, away, total, p_over, p_under, offset).
    """
    offset = score_only_offset(league_key)
    ph = float(proj_home)
    pa = float(proj_away)
    tot = ph + pa
    if offset == 0.0 or tot <= 0.2 or not np.isfinite(tot):
        return ph, pa, tot, p_over, p_under, 0.0
    mat0 = score_matrix(max(ph, 1e-6), max(pa, 1e-6), max_goals=10, dixon_coles=False)
    p0 = float(markets_from_matrix(mat0)["p_over25"])
    new_tot = max(0.40, tot + offset)
    scale = new_tot / tot
    ph2, pa2 = ph * scale, pa * scale
    mat1 = score_matrix(max(ph2, 1e-6), max(pa2, 1e-6), max_goals=10, dixon_coles=False)
    p1 = float(markets_from_matrix(mat1)["p_over25"])
    if p_over is not None and np.isfinite(p_over):
        po = float(np.clip(float(p_over) + (p1 - p0), 0.02, 0.98))
        pu = 1.0 - po
    else:
        po, pu = p1, 1.0 - p1
    return ph2, pa2, new_tot, po, pu, offset


def score_profile(proj_total: float | None, p_over: float | None) -> str:
    tot = proj_total if proj_total is not None and np.isfinite(proj_total) else None
    po = p_over if p_over is not None and np.isfinite(p_over) else None
    if tot is not None and tot >= 3.15:
        return "HIGH"
    if po is not None and po >= 0.62:
        return "HIGH"
    if tot is not None and tot <= 2.25:
        return "LOW"
    if po is not None and po <= 0.38:
        return "LOW"
    return "MID"


def load_feature_frame(league_key: str) -> pd.DataFrame:
    p = ROOT / "data" / "processed" / f"features_{league_key}.parquet"
    if not p.is_file():
        return pd.DataFrame()
    cols = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_xg_for_ewm",
        "away_xg_for_ewm",
        "home_xg_against_ewm",
        "away_xg_against_ewm",
        "home_goals_for_ewm",
        "away_goals_for_ewm",
        "home_goals_against_ewm",
        "away_goals_against_ewm",
        "home_points_roll5",
        "away_points_roll5",
        "elo_home",
        "elo_away",
        "elo_diff",
        "home_rest_days",
        "away_rest_days",
    ]
    df = pd.read_parquet(p)
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("date")
    if "home_team" in out.columns:
        out["_home_c"] = out["home_team"].map(_canon)
        out["_away_c"] = out["away_team"].map(_canon)
    return out


def load_aligned_recent(league_key: str, n_tail: int = 800) -> pd.DataFrame:
    try:
        info = get_league(league_key)
    except KeyError:
        return pd.DataFrame()
    p = ROOT / "data" / "interim" / info["aligned"]
    if not p.is_file():
        return pd.DataFrame()
    want = [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
    ]
    try:
        import pyarrow.parquet as pq

        schema_names = pq.ParquetFile(p).schema.names
        cols = [c for c in want if c in schema_names]
        df = pd.read_parquet(p, columns=cols)
    except Exception:  # noqa: BLE001
        df = pd.read_parquet(p)
        cols = [c for c in want if c in df.columns]
        df = df[cols]
    if "date" not in df.columns or "home_team" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"]).sort_values("date")
    if n_tail and len(df) > n_tail:
        df = df.tail(n_tail)
    df["_home_c"] = df["home_team"].map(_canon)
    df["_away_c"] = df["away_team"].map(_canon)
    return df


def _num(row: pd.Series, col: str) -> float | None:
    if col not in row.index:
        return None
    try:
        x = float(row.get(col))
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _side_snap(row: pd.Series, team: str) -> dict:
    team_c = _canon(team)
    home_c = str(row.get("_home_c") or _canon(row.get("home_team")))
    if home_c == team_c or str(row.get("home_team")) == str(team):
        p, elo_c = "home", "elo_home"
    else:
        p, elo_c = "away", "elo_away"
    return {
        "xg_for": _num(row, f"{p}_xg_for_ewm"),
        "xg_against": _num(row, f"{p}_xg_against_ewm"),
        "goals_for": _num(row, f"{p}_goals_for_ewm"),
        "goals_against": _num(row, f"{p}_goals_against_ewm"),
        "pts5": _num(row, f"{p}_points_roll5"),
        "elo": _num(row, elo_c),
        "rest": _num(row, f"{p}_rest_days"),
    }


def team_snapshot(feat: pd.DataFrame, team: str) -> dict:
    if feat is None or len(feat) == 0 or "home_team" not in feat.columns:
        return {}
    team_c = _canon(team)
    if "_home_c" in feat.columns:
        mask = (feat["_home_c"] == team_c) | (feat["_away_c"] == team_c)
    else:
        mask = (feat["home_team"].astype(str) == str(team)) | (feat["away_team"].astype(str) == str(team))
    sub = feat.loc[mask]
    if len(sub) == 0:
        return {}
    return _side_snap(sub.iloc[-1], team)


def team_form_aligned(aligned: pd.DataFrame, team: str, n: int = 5) -> dict:
    """Last-n completed matches from aligned results (fallback when features miss)."""
    if aligned is None or len(aligned) == 0 or "home_team" not in aligned.columns:
        return {}
    team_c = _canon(team)
    if "_home_c" in aligned.columns:
        mask = (aligned["_home_c"] == team_c) | (aligned["_away_c"] == team_c)
    else:
        mask = (aligned["home_team"].astype(str) == str(team)) | (
            aligned["away_team"].astype(str) == str(team)
        )
    sub = aligned.loc[mask].tail(n)
    if len(sub) == 0:
        return {}
    pts = gf = ga = xg = n_xg = 0.0
    games = 0
    for _, r in sub.iterrows():
        home_c = str(r.get("_home_c") or _canon(r.get("home_team")))
        is_home = home_c == team_c or str(r.get("home_team")) == str(team)
        try:
            hg = float(r.get("home_goals"))
            ag = float(r.get("away_goals"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(hg) or not np.isfinite(ag):
            continue
        games += 1
        if is_home:
            gf += hg
            ga += ag
            if hg > ag:
                pts += 3
            elif hg == ag:
                pts += 1
            xval = r.get("home_xg") if "home_xg" in r.index else None
        else:
            gf += ag
            ga += hg
            if ag > hg:
                pts += 3
            elif ag == hg:
                pts += 1
            xval = r.get("away_xg") if "away_xg" in r.index else None
        try:
            xv = float(xval)
            if np.isfinite(xv):
                xg += xv
                n_xg += 1
        except (TypeError, ValueError):
            pass
    if games <= 0:
        return {}
    out = {
        "pts5": pts / games,
        "goals_for": gf / games,
        "goals_against": ga / games,
        "n": games,
    }
    if n_xg > 0:
        out["xg_for"] = xg / n_xg
    return out


def explain_match(
    *,
    home: str,
    away: str,
    feat: pd.DataFrame,
    proj_total: float | None,
    lean: str,
    aligned: pd.DataFrame | None = None,
    league_key: str = "",
    offset: float = 0.0,
) -> str:
    hs = team_snapshot(feat, home)
    aws = team_snapshot(feat, away)
    if aligned is not None and len(aligned):
        if not hs:
            hs = team_form_aligned(aligned, home)
        if not aws:
            aws = team_form_aligned(aligned, away)

    bits: list[str] = []

    if proj_total is not None and np.isfinite(proj_total):
        if proj_total >= 3.15:
            bits.append("high total")
        elif proj_total <= 2.25:
            bits.append("low total")

    if abs(offset) >= 0.08:
        bits.append(f"score-adj {offset:+.2f}")

    xh, xa = hs.get("xg_for"), aws.get("xg_for")
    gh, ga = hs.get("goals_for"), aws.get("goals_for")
    if xh is not None and xa is not None:
        bits.append(f"xG {xh:.1f}+{xa:.1f}")
    elif gh is not None and ga is not None:
        bits.append(f"GF {gh:.1f}+{ga:.1f}")
    elif xh is not None or xa is not None:
        bits.append(
            f"xG {_fmt_opt(xh)}+{_fmt_opt(xa)}"
        )
    elif gh is not None or ga is not None:
        bits.append(f"GF {_fmt_opt(gh)}+{_fmt_opt(ga)}")

    eh, ea = hs.get("elo"), aws.get("elo")
    if eh is not None and ea is not None:
        diff = eh - ea
        if abs(diff) >= 40:
            who = "H" if diff > 0 else "A"
            bits.append(f"Elo {who}{diff:+.0f}")
    elif eh is not None or ea is not None:
        bits.append(f"Elo H{_fmt_opt(eh, 0)}/A{_fmt_opt(ea, 0)}")

    ph, pa = hs.get("pts5"), aws.get("pts5")
    if ph is not None and pa is not None:
        bits.append(f"form {ph:.1f}/{pa:.1f} ppg")
    elif ph is not None or pa is not None:
        bits.append(f"form {_fmt_opt(ph)}/{_fmt_opt(pa)} ppg")

    xca, xch = aws.get("xg_against"), hs.get("xg_against")
    gca, gch = aws.get("goals_against"), hs.get("goals_against")
    if xch is not None and xca is not None and (xch + xca) >= 2.7 and lean == "OVER":
        bits.append("leaky def")
    elif xch is not None and xca is not None and (xch + xca) <= 2.0 and lean == "UNDER":
        bits.append("tight def")
    elif gch is not None and gca is not None and (gch + gca) >= 3.0 and lean == "OVER":
        bits.append("leaky def")
    elif gch is not None and gca is not None and (gch + gca) <= 1.8 and lean == "UNDER":
        bits.append("tight def")

    rh, ra = hs.get("rest"), aws.get("rest")
    if rh is not None and ra is not None and min(rh, ra) <= 3.5:
        bits.append(f"short rest {min(rh, ra):.0f}d")

    if league_key in WEAK_OU_LEAGUES:
        bits.append("weak O/U hist")

    seen = set()
    out = []
    for b in bits:
        if b in seen:
            continue
        seen.add(b)
        out.append(b)
        if len(out) >= 6:
            break
    return " · ".join(out) if out else "limited pre-match snapshot"
