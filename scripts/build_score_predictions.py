#!/usr/bin/env python
"""
Score-prediction table for the UI (information only).

Primary focus: rolling next 24h + through end of tomorrow (busy-day slate).
Also writes a later slate. Ranked by strongest Over/Under lean.
Merges real fixture kickoffs (sheets often lack kickoff_utc) and Pinnacle OU odds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.gameday.score_explain import (
    WEAK_OU_LEAGUES,
    apply_score_only_projection,
    explain_match,
    load_aligned_recent,
    load_feature_frame,
    score_profile,
)
from origination.gameday.score_team_totals import build_team_total_rows
from origination.utils import load_config, resolve_data_dir, setup_logging
from origination.utils.league_registry import LEAGUES, get_league, list_league_keys

OUT = ROOT / "experiments" / "gameday_scan"
STAMP = ROOT / "data" / "gameday" / "last_data_update.json"

STRENGTH = {
    "EPL": ("A", "HIGH", "Understat xG + 10y history"),
    "Bundesliga": ("A", "HIGH", "Understat xG + 10y history"),
    "LaLiga": ("A", "HIGH", "Understat xG + 10y history"),
    "SerieA": ("A", "HIGH", "Understat xG + 10y history"),
    "Ligue1": ("A", "HIGH", "Understat xG (~76%)"),
    "PrimeiraLiga": ("B", "MED", "Goals-only (no Understat)"),
    "Eredivisie": ("B", "MED", "Goals-only (no Understat)"),
    "Belgium": ("B", "MED", "Goals-only (no Understat)"),
    "Scotland": ("B", "MED", "Goals-only (no Understat)"),
    "Turkey": ("B", "MED", "Goals-only (no Understat)"),
    "Austria": ("B", "MED", "Goals-only (no Understat)"),
    "Championship": ("C", "LOW", "No xG · weak ranking signal"),
    "MLS": ("C", "LOW", "Short history · no xG"),
}

SCORE_LEAGUE_PRIORITY = [
    "EPL",
    "Bundesliga",
    "LaLiga",
    "SerieA",
    "Ligue1",
    "PrimeiraLiga",
    "Eredivisie",
    "Belgium",
    "Championship",
    "Scotland",
    "Turkey",
    "MLS",
    "Austria",
]

STRONG_LEAN_PP = 8.0
PIN_CONFLICT_PP = 15.0
GRADE_WEIGHT = {"A": 1.0, "B": 0.85, "C": 0.65}


def _py() -> Path:
    v = ROOT / ".venv" / "Scripts" / "python.exe"
    return v if v.exists() else Path(sys.executable)


def _sheet_path(key: str) -> Path:
    if key == "EPL":
        return ROOT / "data" / "processed" / "gameday_sheet.csv"
    return ROOT / "data" / "processed" / f"gameday_sheet_{key}.csv"


def _fx_path(key: str) -> Path:
    return ROOT / "data" / "interim" / f"fixtures_upcoming_{key}.csv"


def _aligned_n(key: str) -> int:
    try:
        info = get_league(key)
    except KeyError:
        return 0
    p = ROOT / "data" / "interim" / info["aligned"]
    if not p.is_file():
        return 0
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(p).metadata.num_rows)
    except Exception:  # noqa: BLE001
        return 0


def _current_season_start() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


def _aligned_current_n(key: str) -> int:
    try:
        info = get_league(key)
    except KeyError:
        return 0
    p = ROOT / "data" / "interim" / info["aligned"]
    if not p.is_file():
        return 0
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(p, columns=["date"])
        dates = pd.to_datetime(table.column("date").to_pandas(), errors="coerce")
        start = pd.Timestamp(f"{_current_season_start()}-08-01")
        return int((dates >= start).sum())
    except Exception:  # noqa: BLE001
        return 0


def _score_league_keys() -> list[str]:
    known = list_league_keys()
    ordered = [k for k in SCORE_LEAGUE_PRIORITY if k in known]
    for k in known:
        if k not in ordered:
            ordered.append(k)
    usable = []
    for k in ordered:
        if _aligned_n(k) > 0 or _fx_path(k).is_file() or _sheet_path(k).is_file():
            usable.append(k)
    return usable or ordered


def _model_age_hours() -> float | None:
    if not STAMP.is_file():
        return None
    try:
        raw = json.loads(STAMP.read_text(encoding="utf-8"))
        s = (raw.get("model") or {}).get("updated_at") or raw.get("updated_at_model")
        if not s:
            return None
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600.0
    except Exception:  # noqa: BLE001
        return None


def _parse_kickoff(row: pd.Series, date_val) -> datetime | None:
    for col in ("kickoff_utc", "kickoff_utc_fx", "kickoff"):
        if col not in getattr(row, "index", []):
            continue
        raw = row.get(col)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw) in ("", "nan", "NaT"):
            continue
        try:
            dt = pd.to_datetime(raw, utc=True, errors="coerce")
            if pd.isna(dt):
                continue
            return dt.to_pydatetime()
        except Exception:  # noqa: BLE001
            continue
    if date_val is None or (isinstance(date_val, float) and np.isnan(date_val)):
        return None
    try:
        d = pd.to_datetime(date_val, errors="coerce")
        if pd.isna(d):
            return None
        ts = pd.Timestamp(d)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if ts.hour == 0 and ts.minute == 0:
            ts = ts + pd.Timedelta(hours=17)
        return ts.to_pydatetime()
    except Exception:  # noqa: BLE001
        return None


def _focus_horizon(now: datetime, focus_hours: int) -> datetime:
    by_hours = now + timedelta(hours=focus_hours)
    local = now.astimezone()
    end_tom = (local + timedelta(days=1)).replace(
        hour=23, minute=59, second=59, microsecond=0
    ).astimezone(timezone.utc)
    return max(by_hours, end_tom)


def _strength_row(
    key: str,
    has_proj: bool,
    n_hist: int,
    n_cur: int = 0,
    pin_conflict: bool = False,
) -> tuple[str, str, str]:
    grade, label, note = STRENGTH.get(key, ("C", "LOW", "Limited history"))
    mh = _model_age_hours()
    extras = []
    if n_hist:
        extras.append(f"n={n_hist}")
    extras.append(f"2026/27 n={n_cur}")
    if n_cur == 0:
        extras.append("no current-season results yet")
    if not has_proj:
        grade, label = "C", "LOW"
        extras.append("no model row yet")
    elif mh is not None and mh > 72:
        extras.append(f"model {mh/24:.1f}d old")
        if grade == "A":
            grade, label = "B", "MED"
        elif grade == "B":
            grade, label = "C", "LOW"
    if pin_conflict:
        extras.append(f"Pin conflict ≥{PIN_CONFLICT_PP:.0f}pp")
        if grade == "A":
            grade, label = "B", "MED"
    detail = note if not extras else f"{note} · {' · '.join(extras)}"
    return grade, label, detail


def _refresh_fixtures(keys: list[str]) -> None:
    from origination.data_ingestion.fixtures_upcoming import refresh_upcoming_fixtures_for_league

    setup_logging("WARNING")
    print("  Refreshing fixtures for score leagues...", flush=True)
    for key in keys:
        try:
            info = get_league(key)
            cfg = load_config(ROOT / info["config"])
            data_dir = resolve_data_dir(cfg)
            df, meta = refresh_upcoming_fixtures_for_league(data_dir, key, cfg)
            n = int(len(df)) if df is not None else 0
            print(f"    fixtures [{key}] = {n}  source={(meta or {}).get('source')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"    fixtures [{key}] SKIP: {exc}", flush=True)


def _refresh_odds(keys: list[str]) -> None:
    from origination.data_ingestion.fixtures_upcoming import load_upcoming_fixtures
    from origination.data_ingestion.pinnacle_odds import refresh_pinnacle_odds

    print("  Refreshing Pinnacle odds for score leagues...", flush=True)
    for key in keys:
        try:
            info = get_league(key)
            if not info.get("pinnacle_league_id"):
                print(f"    odds [{key}] SKIP — no pinnacle id", flush=True)
                continue
            cfg = load_config(ROOT / info["config"])
            data_dir = resolve_data_dir(cfg)
            fx = load_upcoming_fixtures(data_dir, league_key=key)
            if fx is None or len(fx) == 0:
                print(f"    odds [{key}] SKIP — no fixtures", flush=True)
                continue
            matched, meta = refresh_pinnacle_odds(
                data_dir, fixtures=fx, cfg=cfg, league_key=key
            )
            print(
                f"    odds [{key}] OU={meta.get('n_with_ou25')}  "
                f"1X2={meta.get('n_with_1x2')}  matched={len(matched) if matched is not None else 0}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    odds [{key}] SKIP: {exc}", flush=True)


def _rebuild_sheet(key: str) -> None:
    out = _sheet_path(key)
    cmd = [
        str(_py()),
        str(ROOT / "scripts" / "run_gameday_sheet.py"),
        "--league",
        key,
        "--fast",
        "--out",
        str(out),
        "--log-level",
        "WARNING",
    ]
    print(f"  Rebuild sheet [{key}]...", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=False)


def _norm_ou(p_o, p_u) -> tuple[float | None, float | None]:
    try:
        o = float(p_o)
        u = float(p_u)
    except (TypeError, ValueError):
        return None, None
    if not np.isfinite(o) or not np.isfinite(u) or (o + u) <= 0:
        return None, None
    s = o + u
    return o / s, u / s


def _implied_two_way(o, u) -> tuple[float | None, float | None]:
    try:
        oo = float(o)
        uu = float(u)
    except (TypeError, ValueError):
        return None, None
    if not np.isfinite(oo) or not np.isfinite(uu) or oo <= 1.0 or uu <= 1.0:
        return None, None
    inv_o, inv_u = 1.0 / oo, 1.0 / uu
    s = inv_o + inv_u
    if s <= 0:
        return None, None
    return inv_o / s, inv_u / s


def _safe_float(val) -> float | None:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _odds_path(key: str) -> Path:
    if key == "EPL":
        return ROOT / "data" / "gameday" / "odds_pinnacle.csv"
    return ROOT / "data" / "gameday" / f"odds_pinnacle_{key}.csv"


def _merge_league_frames(key: str) -> pd.DataFrame:
    fx_p, sh_p = _fx_path(key), _sheet_path(key)
    sheet = pd.read_csv(sh_p) if sh_p.is_file() else pd.DataFrame()
    fx = pd.read_csv(fx_p) if fx_p.is_file() else pd.DataFrame()
    if len(sheet) == 0 and len(fx) == 0:
        return pd.DataFrame()
    if len(sheet) == 0:
        merged = fx.copy()
    elif len(fx) == 0:
        merged = sheet.copy()
    else:
        fx_cols = [
            c
            for c in ("match_id", "kickoff_utc", "date", "home_team", "away_team")
            if c in fx.columns
        ]
        fx_slim = fx[fx_cols].drop_duplicates(subset=["match_id"], keep="first")
        merged = sheet.merge(fx_slim, on="match_id", how="outer", suffixes=("", "_fx"))
        if "kickoff_utc_fx" in merged.columns:
            if "kickoff_utc" not in merged.columns:
                merged["kickoff_utc"] = merged["kickoff_utc_fx"]
            else:
                blank = merged["kickoff_utc"].isna() | (
                    merged["kickoff_utc"].astype(str).isin(["", "nan", "NaT", "None"])
                )
                merged.loc[blank, "kickoff_utc"] = merged.loc[blank, "kickoff_utc_fx"]
        for col in ("date", "home_team", "away_team"):
            fx_c = f"{col}_fx"
            if fx_c in merged.columns:
                if col not in merged.columns:
                    merged[col] = merged[fx_c]
                else:
                    merged[col] = merged[col].where(merged[col].notna(), merged[fx_c])

    odds_p = _odds_path(key)
    if odds_p.is_file() and "match_id" in merged.columns:
        odds = pd.read_csv(odds_p)
        tt_cols = [
            c
            for c in odds.columns
            if c == "match_id"
            or c.startswith("pin_tt_")
            or c in ("pin_over25", "pin_under25", "pin_h", "pin_a", "pin_ah_line", "pin_ahh", "pin_aha")
        ]
        if len(tt_cols) > 1:
            odds_slim = odds[tt_cols].drop_duplicates(subset=["match_id"], keep="first")
            before = set(merged.columns)
            merged = merged.merge(odds_slim, on="match_id", how="left", suffixes=("", "_odds"))
            # Prefer sheet values; fill from odds when missing
            for c in odds_slim.columns:
                if c == "match_id":
                    continue
                alt = f"{c}_odds"
                if alt in merged.columns:
                    if c not in before:
                        merged[c] = merged[alt]
                    else:
                        merged[c] = merged[c].where(merged[c].notna(), merged[alt])
                    merged.drop(columns=[alt], inplace=True)
    return merged


def _row_payload(
    key: str,
    r,
    kick: datetime,
    now: datetime,
    horizon_24: datetime,
    horizon_focus: datetime,
    hist_n: dict,
    cur_n: dict,
    feat_cache: dict,
    aligned_cache: dict,
) -> dict | None:
    if kick < now - timedelta(hours=2):
        return None
    p_o, p_u = _norm_ou(
        r.get("p_over25") if "p_over25" in getattr(r, "index", []) else None,
        r.get("p_under25") if "p_under25" in getattr(r, "index", []) else None,
    )
    try:
        ph = (
            float(r.get("proj_home_goals"))
            if "proj_home_goals" in getattr(r, "index", [])
            else float("nan")
        )
        pa = (
            float(r.get("proj_away_goals"))
            if "proj_away_goals" in getattr(r, "index", [])
            else float("nan")
        )
        tot = (
            float(r.get("proj_total_goals"))
            if "proj_total_goals" in getattr(r, "index", [])
            else float("nan")
        )
    except (TypeError, ValueError):
        ph = pa = tot = float("nan")
    if not np.isfinite(tot) and np.isfinite(ph) and np.isfinite(pa):
        tot = ph + pa
    tot_raw = tot if np.isfinite(tot) else float("nan")
    offset = 0.0
    if np.isfinite(ph) and np.isfinite(pa):
        ph, pa, tot, p_adj_o, p_adj_u, offset = apply_score_only_projection(
            ph, pa, key, p_over=p_o, p_under=p_u
        )
        if offset != 0.0:
            p_o, p_u = p_adj_o, p_adj_u
    has_proj = bool(np.isfinite(tot))
    lean = ""
    lean_pp = 0.0
    if p_o is not None:
        lean_pp = abs(p_o - 0.5) * 100
        lean = "OVER" if p_o >= 0.5 else "UNDER"
    in_24 = now <= kick <= horizon_24
    in_focus = now <= kick <= horizon_focus
    if in_24:
        when, rank_group = "NEXT 24H", 0
    elif in_focus:
        when, rank_group = "THROUGH TOM.", 1
    else:
        when, rank_group = "LATER", 2
    strong = bool(lean and lean_pp >= STRONG_LEAN_PP)
    local = kick.astimezone()

    pin_o = _safe_float(r.get("pin_over25") if "pin_over25" in getattr(r, "index", []) else None)
    pin_u = _safe_float(r.get("pin_under25") if "pin_under25" in getattr(r, "index", []) else None)
    mkt_o, mkt_u = _implied_two_way(pin_o, pin_u)
    edge_o = None if (p_o is None or mkt_o is None) else round(100 * (p_o - mkt_o), 1)
    edge_u = None if (p_u is None or mkt_u is None) else round(100 * (p_u - mkt_u), 1)
    pin_conflict = False
    if lean == "OVER" and edge_o is not None:
        pin_conflict = abs(edge_o) >= PIN_CONFLICT_PP
    elif lean == "UNDER" and edge_u is not None:
        pin_conflict = abs(edge_u) >= PIN_CONFLICT_PP
    grade, label, detail = _strength_row(
        key,
        has_proj,
        hist_n.get(key, 0),
        n_cur=cur_n.get(key, 0),
        pin_conflict=pin_conflict,
    )
    if abs(offset) >= 0.08:
        detail = f"{detail} · score-adj {offset:+.2f} (Score tab only)"
    quality = round(
        lean_pp * GRADE_WEIGHT.get(grade, 0.6) * (0.45 if pin_conflict else 1.0),
        2,
    )
    if pin_conflict:
        confidence = "LOW — fights Pin"
    elif grade == "A" and strong:
        confidence = "HIGH"
    elif grade == "A":
        confidence = "MED-HIGH"
    elif grade == "B" and strong and not pin_conflict:
        confidence = "MED"
    else:
        confidence = "LOW"
    if key in WEAK_OU_LEAGUES and confidence in ("HIGH", "MED-HIGH"):
        confidence = "MED — weak O/U hist"
    sheet_edge_o = _safe_float(
        r.get("edge_over_vs_pinnacle") if "edge_over_vs_pinnacle" in getattr(r, "index", []) else None
    )
    sheet_edge_u = _safe_float(
        r.get("edge_under_vs_pinnacle") if "edge_under_vs_pinnacle" in getattr(r, "index", []) else None
    )

    return {
        "when": when,
        "in_next_24h": in_24,
        "in_focus": in_focus,
        "kickoff_utc": kick.isoformat(),
        "kickoff_local": local.strftime("%a %d %b %H:%M"),
        "date": kick.date().isoformat(),
        "league": key,
        "league_name": LEAGUES.get(key, {}).get("name", key),
        "home_team": r.get("home_team"),
        "away_team": r.get("away_team"),
        "match": f"{r.get('home_team')} vs {r.get('away_team')}",
        "proj_home": None if not np.isfinite(ph) else round(ph, 2),
        "proj_away": None if not np.isfinite(pa) else round(pa, 2),
        "proj_score": "—" if not (np.isfinite(ph) and np.isfinite(pa)) else f"{ph:.2f} - {pa:.2f}",
        "proj_total_raw": None if not np.isfinite(tot_raw) else round(float(tot_raw), 2),
        "proj_total": None if not np.isfinite(tot) else round(float(tot), 2),
        "score_totals_offset": round(float(offset), 3),
        "p_over25": None if p_o is None else round(p_o, 4),
        "p_under25": None if p_u is None else round(p_u, 4),
        "over_pct": None if p_o is None else round(100 * p_o, 1),
        "under_pct": None if p_u is None else round(100 * p_u, 1),
        "lean": lean,
        "lean_pp": round(lean_pp, 1),
        "strong": strong,
        "rank_group": rank_group,
        "data_grade": grade,
        "data_strength": label,
        "data_note": detail,
        "pin_conflict": pin_conflict,
        "confidence": confidence,
        "rank_quality": quality,
        "current_season_n": int(cur_n.get(key, 0)),
        "pin_over25": None if pin_o is None else round(pin_o, 3),
        "pin_under25": None if pin_u is None else round(pin_u, 3),
        "pin_over_pct": None if mkt_o is None else round(100 * mkt_o, 1),
        "pin_under_pct": None if mkt_u is None else round(100 * mkt_u, 1),
        "model_minus_pin_over_pp": edge_o,
        "model_minus_pin_under_pp": edge_u,
        "edge_over_vs_pinnacle": None if sheet_edge_o is None else round(sheet_edge_o, 4),
        "edge_under_vs_pinnacle": None if sheet_edge_u is None else round(sheet_edge_u, 4),
        "has_pin_ou": bool(pin_o is not None and pin_u is not None),
        "pin_lean": "" if mkt_o is None else ("OVER" if mkt_o >= 0.5 else "UNDER"),
        "pin_tt_home_line": _safe_float(r.get("pin_tt_home_line") if "pin_tt_home_line" in getattr(r, "index", []) else None),
        "pin_tt_home_over": _safe_float(r.get("pin_tt_home_over") if "pin_tt_home_over" in getattr(r, "index", []) else None),
        "pin_tt_home_under": _safe_float(r.get("pin_tt_home_under") if "pin_tt_home_under" in getattr(r, "index", []) else None),
        "pin_tt_away_line": _safe_float(r.get("pin_tt_away_line") if "pin_tt_away_line" in getattr(r, "index", []) else None),
        "pin_tt_away_over": _safe_float(r.get("pin_tt_away_over") if "pin_tt_away_over" in getattr(r, "index", []) else None),
        "pin_tt_away_under": _safe_float(r.get("pin_tt_away_under") if "pin_tt_away_under" in getattr(r, "index", []) else None),
        "score_profile": score_profile(None if not np.isfinite(tot) else float(tot), p_o),
        "why": explain_match(
            home=str(r.get("home_team") or ""),
            away=str(r.get("away_team") or ""),
            feat=feat_cache.get(key, pd.DataFrame()),
            proj_total=None if not np.isfinite(tot) else float(tot),
            lean=lean,
            aligned=aligned_cache.get(key, pd.DataFrame()),
            league_key=key,
            offset=offset,
        ),
        "match_id": str(r.get("match_id", "")),
    }


def build(
    *,
    hours: int,
    focus_hours: int,
    later_hours: int,
    rebuild: bool,
    refresh_fx: bool,
    refresh_odds: bool,
) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    horizon_24 = now + timedelta(hours=hours)
    horizon_focus = _focus_horizon(now, focus_hours)
    horizon_later = now + timedelta(hours=later_hours)
    keys = _score_league_keys()
    hist_n = {k: _aligned_n(k) for k in keys}
    cur_n = {k: _aligned_current_n(k) for k in keys}
    feat_cache = {k: load_feature_frame(k) for k in keys}
    aligned_cache = {k: load_aligned_recent(k) for k in keys}

    if refresh_fx or rebuild:
        _refresh_fixtures(keys)
    if refresh_odds or rebuild:
        _refresh_odds(keys)
    if rebuild:
        for key in keys:
            if _fx_path(key).is_file() and _aligned_n(key) > 0:
                _rebuild_sheet(key)

    rows: list[dict] = []
    seen: set[str] = set()
    for key in keys:
        frame = _merge_league_frames(key)
        if len(frame) == 0:
            continue
        for _, r in frame.iterrows():
            mid = str(r.get("match_id", ""))
            if not mid or mid in seen:
                continue
            kick = _parse_kickoff(r, r.get("date"))
            if kick is None or kick > horizon_later:
                continue
            payload = _row_payload(
                key,
                r,
                kick,
                now,
                horizon_24,
                horizon_focus,
                hist_n,
                cur_n,
                feat_cache,
                aligned_cache,
            )
            if payload is None:
                continue
            seen.add(mid)
            rows.append(payload)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df.sort_values(
        ["rank_group", "rank_quality", "lean_pp", "kickoff_utc"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def _print_leans(df: pd.DataFrame, side: str, n: int = 12) -> None:
    sub = df[(df["lean"] == side) & (df["lean_pp"] > 0)].copy()
    if side == "OVER":
        sub = sub.sort_values(["rank_group", "over_pct"], ascending=[True, False])
    else:
        sub = sub.sort_values(["rank_group", "under_pct"], ascending=[True, False])
    print(f"\n  Strongest {side} leans (focus first):", flush=True)
    for _, r in sub.head(n).iterrows():
        o = f"{r['over_pct']:.0f}%" if pd.notna(r.get("over_pct")) else "—"
        u = f"{r['under_pct']:.0f}%" if pd.notna(r.get("under_pct")) else "—"
        pin = (
            f"Pin {r['pin_over25']:.2f}/{r['pin_under25']:.2f} "
            f"({r['pin_over_pct']:.0f}/{r['pin_under_pct']:.0f}%)"
            if r.get("has_pin_ou")
            else "Pin —"
        )
        delta = ""
        if side == "OVER" and pd.notna(r.get("model_minus_pin_over_pp")):
            delta = f"  model-pin {r['model_minus_pin_over_pp']:+.1f}pp"
        elif side == "UNDER" and pd.notna(r.get("model_minus_pin_under_pp")):
            delta = f"  model-pin {r['model_minus_pin_under_pp']:+.1f}pp"
        flag = ""
        if r.get("strong"):
            flag += " **"
        if r.get("pin_conflict"):
            flag += " CONFLICT"
        prof = r.get("score_profile") or ""
        why = str(r.get("why") or "")[:72]
        print(
            f"  {r['when']:12} {r['kickoff_local']:16}  {r['league']:12}  "
            f"{str(r['match'])[:32]:32}  {r['proj_score']:>11}  tot={r['proj_total']!s:>4}  "
            f"{prof:4}  O {o:>4} / U {u:>4}  {r.get('confidence','')}  {pin}{delta}{flag}",
            flush=True,
        )
        if why:
            print(f"               why: {why}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Build rolling score-prediction table")
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--focus-hours", type=int, default=36)
    p.add_argument("--later-hours", type=int, default=168)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--refresh-fixtures", action="store_true")
    p.add_argument("--refresh-odds", action="store_true")
    args = p.parse_args()

    now = datetime.now(timezone.utc)
    focus_end = _focus_horizon(now, args.focus_hours)
    print("=== Score Predictions (rolling window) ===", flush=True)
    print(
        f"now={now.isoformat()}  next={args.hours}h  "
        f"focus_through={focus_end.isoformat()}  later={args.later_hours}h  "
        f"rebuild={args.rebuild}",
        flush=True,
    )
    df = build(
        hours=args.hours,
        focus_hours=args.focus_hours,
        later_hours=args.later_hours,
        rebuild=args.rebuild,
        refresh_fx=args.refresh_fixtures or args.rebuild,
        refresh_odds=args.refresh_odds or args.rebuild,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "SCORE_PREDICTIONS.csv"
    df.to_csv(out, index=False)
    # Dated archive for week-end retros (do not overwrite older dated snaps)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    dated = OUT / f"SCORE_PREDICTIONS_{stamp}.csv"
    if not dated.is_file():
        df.to_csv(dated, index=False)
        print(f"Archived {dated.name}", flush=True)
    tt = build_team_total_rows(df)
    tt_out = OUT / "SCORE_TEAM_TOTALS.csv"
    tt.to_csv(tt_out, index=False)
    try:
        from origination.gameday.tt_ledger import run_daily_from_score

        tt_info = run_daily_from_score(df, tt, record=True)
        print(
            f"TT paper card: candidates={tt_info['n_candidates']}  "
            f"newly_logged={tt_info['n_recorded']}  "
            f"line_obs={tt_info.get('n_line_obs', 0)}  → {tt_info['card']}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"TT ledger skipped: {exc}", flush=True)
    n24 = int(df["in_next_24h"].sum()) if len(df) else 0
    n_focus = int(df["in_focus"].sum()) if len(df) else 0
    n_strong = int(df["strong"].sum()) if len(df) else 0
    n_pin = int(df["has_pin_ou"].sum()) if len(df) and "has_pin_ou" in df.columns else 0
    n_conflict = int(df["pin_conflict"].sum()) if len(df) and "pin_conflict" in df.columns else 0
    n_high = int((df["score_profile"] == "HIGH").sum()) if len(df) and "score_profile" in df.columns else 0
    n_low = int((df["score_profile"] == "LOW").sum()) if len(df) and "score_profile" in df.columns else 0
    n_tt = int(len(tt))
    n_tt_pin = int(tt["has_pin_tt"].sum()) if len(tt) and "has_pin_tt" in tt.columns else 0
    print(
        f"NEXT24H={n24}  FOCUS={n_focus}  LATER={len(df)-n_focus}  "
        f"HIGH={n_high}  LOW={n_low}  STRONG_LEANS={n_strong}  "
        f"PIN_CONFLICT={n_conflict}  WITH_PIN_OU={n_pin}  "
        f"TEAM_TOTALS={n_tt} (with Pin={n_tt_pin})  total={len(df)}",
        flush=True,
    )
    print(f"Wrote {out}", flush=True)
    print(f"Wrote {tt_out}", flush=True)
    if len(df):
        focus = df[df["in_focus"]].copy() if "in_focus" in df.columns else df
        show = focus if len(focus) else df
        _print_leans(show, "OVER", n=8)
        _print_leans(show, "UNDER", n=8)
        highs = (
            show[show["score_profile"] == "HIGH"].sort_values("proj_total", ascending=False)
            if "score_profile" in show.columns
            else show.iloc[0:0]
        )
        lows = (
            show[show["score_profile"] == "LOW"].sort_values("proj_total", ascending=True)
            if "score_profile" in show.columns
            else show.iloc[0:0]
        )
        if len(highs):
            print("\n  Highest projected totals:", flush=True)
            for _, r in highs.head(6).iterrows():
                print(
                    f"  HIGH {r['league']:12} {str(r['match'])[:36]:36}  "
                    f"{r['proj_score']:>11}  tot={r['proj_total']}  {r.get('why','')[:50]}",
                    flush=True,
                )
        if len(lows):
            print("\n  Lowest projected totals:", flush=True)
            for _, r in lows.head(6).iterrows():
                print(
                    f"  LOW  {r['league']:12} {str(r['match'])[:36]:36}  "
                    f"{r['proj_score']:>11}  tot={r['proj_total']}  {r.get('why','')[:50]}",
                    flush=True,
                )
        if len(tt):
            show_tt = tt[tt["in_focus"] == True] if "in_focus" in tt.columns else tt
            if len(show_tt) == 0:
                show_tt = tt
            show_tt = show_tt[show_tt["has_pin_tt"] == True] if "has_pin_tt" in show_tt.columns else show_tt
            print("\n  Team totals vs Pin (largest |edge|, focus first):", flush=True)
            for _, r in show_tt.head(12).iterrows():
                print(
                    f"  TT {r['when']:12} {r['league']:12} {str(r['team'])[:18]:18}  "
                    f"proj={r['tt_proj']!s:>4}  line={r['tt_line']}  "
                    f"model O/U {100*float(r['tt_p_over']):.0f}/{100*float(r['tt_p_under']):.0f}%  "
                    f"Pin {r['tt_pin_over']}/{r['tt_pin_under']}  "
                    f"{r['tt_lean']} {r['tt_vs_pin']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
