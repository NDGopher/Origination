#!/usr/bin/env python
"""
Full Model Refresh — rebuild everything the protected systems need.

Unlike light "Update Data Sources" (fixtures / optional recent results), this:
  - Re-downloads the CURRENT season for football-data + Understat (old seasons stay cached)
  - Rebuilds aligned match tables
  - Loads Understat advanced (PPDA / deep / npxG) when configured
  - Rebuilds the feature matrix (form, Elo, residuals inputs, context layers)
  - Refreshes upcoming fixtures
  - Does NOT refresh Pinnacle odds (use Update Odds)

Stamps: data/gameday/last_data_update.json → model.updated_at
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.data_ingestion import (
    build_aligned_from_config,
    ingest_fbref_from_config,
)
from origination.data_ingestion.align import load_aligned
from origination.data_ingestion.football_data import FootballDataIngester
from origination.data_ingestion.fixtures_upcoming import (
    fetch_pulse_completed,
    refresh_upcoming_fixtures_for_league,
)
from origination.data_ingestion.understat import UnderstatIngester
from origination.data_ingestion.understat_advanced import (
    enrich_matches_with_understat_advanced,
    load_understat_team_history,
)
from origination.features.store import build_feature_matrix
from origination.utils import load_config, resolve_data_dir, setup_logging
from origination.utils.league_registry import get_league
from origination.utils.seeding import season_from_date

LIVE_LEAGUES = ["EPL", "Bundesliga", "LaLiga", "SerieA", "PrimeiraLiga"]
# Score Predictions slate (info only) — include with --with-score-leagues
SCORE_LEAGUES = [
    "Ligue1",
    "Eredivisie",
    "Belgium",
    "Championship",
    "MLS",
    "Turkey",
    "Scotland",
]
STAMP = ROOT / "data" / "gameday" / "last_data_update.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_stamp(section: str, payload: dict) -> None:
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    prior: dict = {}
    if STAMP.exists():
        try:
            prior = json.loads(STAMP.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prior = {}
    prior[section] = payload
    prior[f"updated_at_{section}"] = payload.get("updated_at")
    # Keep light "data" stamp in sync when full refresh also refreshed fixtures
    if section == "model" and payload.get("ok"):
        prior["data"] = {
            "kind": "full_model_refresh",
            "updated_at": payload["updated_at"],
            "note": "Updated as part of Full Model Refresh",
            "leagues": payload.get("leagues"),
        }
        prior["updated_at_data"] = payload["updated_at"]
    STAMP.write_text(json.dumps(prior, indent=2), encoding="utf-8")


def _current_season_start() -> int:
    """European season start year (Aug+ → this year, else previous)."""
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _current_season_summary(aligned: pd.DataFrame, season: int) -> dict:
    if aligned is None or len(aligned) == 0:
        return {"n": 0, "n_football_data": 0, "n_extra": 0, "n_xg": 0}
    df = aligned.copy()
    dates = pd.to_datetime(df["date"] if "date" in df.columns else df["Date"])
    start = pd.Timestamp(f"{season}-08-01")
    cur = df.loc[dates >= start]
    src = cur["result_source"].astype(str) if "result_source" in cur.columns else pd.Series(["football_data"] * len(cur))
    n_xg = int(cur["home_xg"].notna().sum()) if "home_xg" in cur.columns else 0
    return {
        "n": int(len(cur)),
        "n_football_data": int((src == "football_data").sum()),
        "n_extra": int((src != "football_data").sum()),
        "n_xg": n_xg,
        "latest": str(pd.to_datetime(cur["date"]).max().date()) if len(cur) else None,
    }


def _force_current_fd(cfg: dict, data_dir: Path, season: int) -> pd.DataFrame:
    """Re-download current FD season; load full history from cache via normal ingest."""
    from origination.data_ingestion import ingest_football_data_from_config

    fd_cfg = cfg.get("data", {}).get("football_data", {})
    if not fd_cfg.get("enabled", True):
        raise RuntimeError("football_data disabled")
    ingester = FootballDataIngester(
        raw_dir=data_dir / "raw" / "football_data",
        base_url=fd_cfg.get("base_url", "https://www.football-data.co.uk/mmz4281"),
    )
    for league in cfg.get("leagues", []):
        code = league["code"]
        print(f"    football-data [{code}] force season {season}...", flush=True)
        path = ingester.fetch_season(code, season, force=True)
        if path is None:
            print(
                f"      (no football-data {code} {season}/{season+1} file yet — "
                f"using history + Understat/Pulse current results)",
                flush=True,
            )
        else:
            print(f"      fetched {path.name}", flush=True)
    return ingest_football_data_from_config(cfg, data_dir)


def _force_current_understat(cfg: dict, data_dir: Path, season: int) -> pd.DataFrame | None:
    from origination.data_ingestion import ingest_understat_from_config

    us_cfg = cfg.get("data", {}).get("understat", {})
    if not us_cfg.get("enabled", False):
        print("    Understat disabled in config — skip", flush=True)
        return None
    ingester = UnderstatIngester(
        raw_dir=data_dir / "raw" / "understat",
        base_url=us_cfg.get("base_url", "https://understat.com"),
    )
    for league in cfg.get("leagues", []):
        us_league = league.get("understat")
        if not us_league:
            continue
        print(f"    Understat [{us_league}] force season {season}...", flush=True)
        path = ingester.fetch_season(us_league, season, force=True)
        if path is None:
            print(f"      WARN: could not fetch {us_league} {season}", flush=True)
        else:
            try:
                parsed = ingester.parse_season(us_league, season)
                n_cur = 0 if parsed is None else int(len(parsed))
                print(f"      {us_league} {season}: {n_cur} results", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"      WARN parse: {exc}", flush=True)
    return ingest_understat_from_config(cfg, data_dir)


def _rebuild_features(cfg: dict, data_dir: Path, league_key: str, aligned_name: str) -> dict:
    """Enrich + build feature matrix; write cache parquet. Context layers run inside store."""
    path = data_dir / "interim" / aligned_name
    if not path.is_file():
        return {"ok": False, "error": f"missing {aligned_name}"}
    matches = load_aligned(path)
    n0 = len(matches)
    print(f"    Feature rebuild on {n0} aligned matches...", flush=True)

    if cfg.get("features", {}).get("groups", {}).get("understat_advanced", False):
        hist_us = load_understat_team_history(data_dir / "raw" / "understat")
        matches = enrich_matches_with_understat_advanced(matches, hist_us)
        print("      + Understat advanced (PPDA/deep/npxG)", flush=True)

    if cfg.get("features", {}).get("groups", {}).get("possession_value", False):
        try:
            pv_path = data_dir / "interim" / "possession_value.parquet"
            if pv_path.is_file():
                from origination.features.possession_value import enrich_matches_with_possession_value

                matches = enrich_matches_with_possession_value(matches, pd.read_parquet(pv_path))
                print("      + Possession value", flush=True)
            else:
                print("      (possession_value enabled but table missing — skip)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"      WARN possession_value: {exc}", flush=True)

    # Ensure season for feature windows
    if "season" not in matches.columns or matches["season"].isna().any():
        matches = matches.copy()
        matches["season"] = pd.to_datetime(matches["date"]).map(season_from_date)

    feat_cfg = cfg.get("features", {})
    ctx = feat_cfg.get("context_adjustments") or {}
    if ctx.get("enabled"):
        enabled = [k for k, v in ctx.items() if isinstance(v, dict) and v.get("enabled")]
        print(f"      + Context layers: {', '.join(enabled) or '(none enabled)'}", flush=True)

    features = build_feature_matrix(matches, feat_cfg)
    out = data_dir / "processed" / f"features_{league_key}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    print(f"      -> {out.name}  rows={len(features)}  cols={len(features.columns)}", flush=True)
    return {
        "ok": True,
        "n_matches": int(n0),
        "n_feature_rows": int(len(features)),
        "n_feature_cols": int(len(features.columns)),
        "path": str(out),
    }


def refresh_league(league_key: str, *, season: int, skip_features: bool = False) -> dict:
    info = get_league(league_key)
    cfg = load_config(ROOT / info["config"])
    data_dir = resolve_data_dir(cfg)
    result: dict = {"league": league_key, "ok": False, "steps": []}

    print(f"\n=== Full Model Refresh [{league_key}] ===", flush=True)

    # 1) Results + xG
    print("  [1/4] Results / xG / advanced stats...", flush=True)
    try:
        fd = _force_current_fd(cfg, data_dir, season)
        result["steps"].append({"step": "football_data", "ok": True, "n": int(len(fd))})
        print(f"      FD rows={len(fd)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"      ERROR football-data: {exc}", flush=True)
        result["error"] = str(exc)
        return result

    understat = None
    try:
        understat = _force_current_understat(cfg, data_dir, season)
        n_us = int(len(understat)) if understat is not None else 0
        result["steps"].append({"step": "understat", "ok": True, "n": n_us})
        print(f"      Understat rows={n_us}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"      WARN Understat: {exc}", flush=True)
        result["steps"].append({"step": "understat", "ok": False, "error": str(exc)})

    fbref = None
    try:
        if cfg.get("data", {}).get("fbref", {}).get("enabled"):
            print("      FBref...", flush=True)
            fbref = ingest_fbref_from_config(cfg, data_dir)
            result["steps"].append(
                {"step": "fbref", "ok": True, "n": int(len(fbref)) if fbref is not None else 0}
            )
    except Exception as exc:  # noqa: BLE001
        print(f"      WARN FBref: {exc}", flush=True)

    extra_results = None
    if league_key == "EPL":
        try:
            pulse, pmeta = fetch_pulse_completed()
            n_pulse = int(pmeta.get("n") or 0)
            print(f"      Pulse completed EPL={n_pulse}", flush=True)
            result["steps"].append({"step": "pulse_completed", "ok": True, "n": n_pulse})
            if pulse is not None and len(pulse):
                extra_results = pulse
        except Exception as exc:  # noqa: BLE001
            print(f"      WARN Pulse completed: {exc}", flush=True)
            result["steps"].append({"step": "pulse_completed", "ok": False, "error": str(exc)})

    # 2) Align
    print("  [2/4] Align matches...", flush=True)
    try:
        aligned = build_aligned_from_config(
            cfg, data_dir, fd, understat, fbref, extra_results=extra_results
        )
        result["steps"].append({"step": "align", "ok": True, "n": int(len(aligned))})
        print(f"      Aligned={len(aligned)}", flush=True)
        cur_info = _current_season_summary(aligned, season)
        result["current_season"] = cur_info
        print(
            f"      Current {season}/{season+1}: n={cur_info['n']}  "
            f"fd={cur_info['n_football_data']}  extra={cur_info['n_extra']}  "
            f"with_xg={cur_info['n_xg']}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"      ERROR align: {exc}", flush=True)
        result["error"] = str(exc)
        return result

    # 3) Feature store + context / elite inputs
    if not skip_features:
        print("  [3/4] Feature store + context / elite layers...", flush=True)
        feat_info = _rebuild_features(cfg, data_dir, league_key, info["aligned"])
        result["steps"].append({"step": "features", **feat_info})
        if not feat_info.get("ok"):
            result["error"] = feat_info.get("error", "feature rebuild failed")
            return result
    else:
        print("  [3/4] Feature rebuild skipped (--skip-features)", flush=True)

    # 4) Fixtures (schedule changes)
    print("  [4/4] Upcoming fixtures...", flush=True)
    try:
        fx, meta = refresh_upcoming_fixtures_for_league(data_dir, league_key, cfg)
        n = int(len(fx)) if fx is not None else 0
        result["steps"].append(
            {"step": "fixtures", "ok": True, "n": n, "source": (meta or {}).get("source")}
        )
        print(f"      Fixtures={n} source={(meta or {}).get('source')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"      WARN fixtures: {exc}", flush=True)
        result["steps"].append({"step": "fixtures", "ok": False, "error": str(exc)})

    result["ok"] = True
    print(f"  OK {league_key} model data refreshed", flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="Full model data refresh for live leagues")
    p.add_argument("--leagues", default=None, help="Comma-separated league keys (default: live packs)")
    p.add_argument(
        "--with-score-leagues",
        action="store_true",
        help="Also refresh Score Predictions leagues (Ligue1, Championship, MLS, …)",
    )
    p.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature-matrix rebuild (ingest+align+fixtures only)",
    )
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    setup_logging(args.log_level)

    if args.leagues:
        leagues = [x.strip() for x in args.leagues.split(",") if x.strip()]
    else:
        leagues = list(LIVE_LEAGUES)
        if args.with_score_leagues:
            for k in SCORE_LEAGUES:
                if k not in leagues:
                    leagues.append(k)
    season = _current_season_start()

    print("=" * 64, flush=True)
    print("  FULL MODEL REFRESH", flush=True)
    print("=" * 64, flush=True)
    print(f"Leagues: {', '.join(leagues)}", flush=True)
    print(f"Force-refresh season: {season}/{season+1}", flush=True)
    print("Old seasons: kept from cache (not re-downloaded)", flush=True)
    print("Odds: NOT updated here — use Update Odds", flush=True)
    print(flush=True)

    results = []
    failed = []
    for key in leagues:
        try:
            r = refresh_league(key, season=season, skip_features=args.skip_features)
            results.append(r)
            if not r.get("ok"):
                failed.append(key)
        except Exception as exc:  # noqa: BLE001
            print(f"FATAL [{key}]: {exc}", flush=True)
            traceback.print_exc()
            results.append({"league": key, "ok": False, "error": str(exc)})
            failed.append(key)

    ok = len(failed) == 0
    # Merge current_season into prior stamp so partial refreshes don't wipe other leagues
    prior_cs: dict = {}
    if STAMP.exists():
        try:
            prior_cs = (json.loads(STAMP.read_text(encoding="utf-8")).get("model") or {}).get(
                "current_season"
            ) or {}
        except Exception:  # noqa: BLE001
            prior_cs = {}
    new_cs = {
        r["league"]: r.get("current_season")
        for r in results
        if r.get("current_season") is not None
    }
    merged_cs = {**prior_cs, **new_cs}

    payload = {
        "kind": "full_model_refresh",
        "updated_at": _utc_now(),
        "ok": ok,
        "force_season": season,
        "leagues": leagues,
        "failed": failed,
        "results": results,
        "current_season": merged_cs,
    }
    _merge_stamp("model", payload)

    print(flush=True)
    print("=" * 64, flush=True)
    if ok:
        print("  SUCCESS — Full Model Refresh complete", flush=True)
    else:
        print(f"  FAILED — issues in: {', '.join(failed)}", flush=True)
    print(f"  Stamp -> {STAMP}", flush=True)
    print("=" * 64, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
