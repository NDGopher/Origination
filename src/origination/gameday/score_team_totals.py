"""Team-total helpers for Score Predictions (information / value scan only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from origination.models.poisson import team_total_over_prob
from origination.utils.odds import two_way_fair


def _implied_two_way(over: float | None, under: float | None) -> tuple[float | None, float | None]:
    try:
        oo = float(over)
        uu = float(under)
    except (TypeError, ValueError):
        return None, None
    if not np.isfinite(oo) or not np.isfinite(uu) or oo <= 1.0 or uu <= 1.0:
        return None, None
    fo, fu = two_way_fair(oo, uu, method="power")
    return float(fo), float(fu)


def team_total_edge(
    *,
    proj_goals: float | None,
    pin_line: float | None,
    pin_over: float | None,
    pin_under: float | None,
) -> dict:
    """Model P(over line) vs Pin fair; returns edge in probability points."""
    out = {
        "line": None,
        "proj": None,
        "p_over": None,
        "p_under": None,
        "pin_over": None,
        "pin_under": None,
        "pin_over_pct": None,
        "pin_under_pct": None,
        "edge_over_pp": None,
        "edge_under_pp": None,
        "lean": "",
        "lean_pp": 0.0,
        "vs_pin": "",
        "pin_conflict": False,
    }
    try:
        lam = float(proj_goals) if proj_goals is not None else float("nan")
        line = float(pin_line) if pin_line is not None else float("nan")
    except (TypeError, ValueError):
        return out
    if not np.isfinite(lam) or not np.isfinite(line):
        return out
    po = team_total_over_prob(lam, line)
    pu = 1.0 - po
    mkt_o, mkt_u = _implied_two_way(pin_over, pin_under)
    out.update(
        {
            "line": round(line, 3),
            "proj": round(lam, 2),
            "p_over": round(po, 4),
            "p_under": round(pu, 4),
            "pin_over": None if pin_over is None else round(float(pin_over), 3),
            "pin_under": None if pin_under is None else round(float(pin_under), 3),
            "pin_over_pct": None if mkt_o is None else round(100 * mkt_o, 1),
            "pin_under_pct": None if mkt_u is None else round(100 * mkt_u, 1),
        }
    )
    if mkt_o is not None and mkt_u is not None:
        eo = round(100 * (po - mkt_o), 1)
        eu = round(100 * (pu - mkt_u), 1)
        out["edge_over_pp"] = eo
        out["edge_under_pp"] = eu
        if eo >= eu:
            out["lean"] = "OVER"
            out["lean_pp"] = abs(eo)
            out["vs_pin"] = f"O{eo:+.1f}pp"
        else:
            out["lean"] = "UNDER"
            out["lean_pp"] = abs(eu)
            out["vs_pin"] = f"U{eu:+.1f}pp"
        out["pin_conflict"] = bool(out["lean_pp"] >= 15.0)
    else:
        # no pin — lean by model vs 50%
        if po >= 0.5:
            out["lean"] = "OVER"
            out["lean_pp"] = round(100 * (po - 0.5), 1)
        else:
            out["lean"] = "UNDER"
            out["lean_pp"] = round(100 * (0.5 - po), 1)
        out["pin_conflict"] = False
    return out


def build_team_total_rows(score_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team (home + away) with model vs Pin team-total value."""
    rows: list[dict] = []
    if score_df is None or len(score_df) == 0:
        return pd.DataFrame()
    for _, r in score_df.iterrows():
        base = {
            "when": r.get("when"),
            "in_next_24h": r.get("in_next_24h"),
            "in_focus": r.get("in_focus"),
            "kickoff_local": r.get("kickoff_local"),
            "kickoff_utc": r.get("kickoff_utc"),
            "date": r.get("date"),
            "league": r.get("league"),
            "match": r.get("match"),
            "match_id": r.get("match_id"),
            "data_grade": r.get("data_grade"),
            "confidence": r.get("confidence"),
            "pin_conflict_ou": r.get("pin_conflict"),
        }
        for side, team_col, proj_col, line_c, over_c, under_c in (
            (
                "home",
                "home_team",
                "proj_home",
                "pin_tt_home_line",
                "pin_tt_home_over",
                "pin_tt_home_under",
            ),
            (
                "away",
                "away_team",
                "proj_away",
                "pin_tt_away_line",
                "pin_tt_away_over",
                "pin_tt_away_under",
            ),
        ):
            edge = team_total_edge(
                proj_goals=r.get(proj_col),
                pin_line=r.get(line_c) if line_c in r.index else None,
                pin_over=r.get(over_c) if over_c in r.index else None,
                pin_under=r.get(under_c) if under_c in r.index else None,
            )
            if edge["line"] is None and edge["proj"] is None:
                continue
            if edge["line"] is None:
                # still useful: show projection without Pin
                continue
            rows.append(
                {
                    **base,
                    "side": side,
                    "team": r.get(team_col),
                    **{f"tt_{k}": v for k, v in edge.items()},
                    "has_pin_tt": edge["pin_over"] is not None,
                }
            )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    # value rank: largest |edge| with Pin, focus first
    df["rank_group"] = df["when"].map({"NEXT 24H": 0, "THROUGH TOM.": 1, "LATER": 2}).fillna(3)
    df["abs_edge"] = df["tt_lean_pp"].fillna(0.0)
    df = df.sort_values(
        ["rank_group", "abs_edge", "kickoff_utc"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df
