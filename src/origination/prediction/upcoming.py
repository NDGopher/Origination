"""
Forward / live prediction path — same features + model + calibration + residual as backtest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
import yaml

from origination.backtesting.bet_filters import passes_bet_filters
from origination.features.store import assert_features_pre_match, build_feature_matrix
from origination.models.calibration import build_calibrator
from origination.models.poisson import apply_totals_intercept, build_model
from origination.models.residual import (
    build_residual_model,
    collect_oos_base_predictions,
    _numeric_feature_matrix,
)
from origination.utils.odds import (
    decimal_to_american,
    fair_decimal_odds,
    model_edge_vs_odds,
    model_edge_vs_two_way,
    two_way_fair,
)


def _load_rule_packs(path: Path | None = None) -> dict[str, Any]:
    p = path or Path(__file__).resolve().parents[3] / "configs" / "league_rule_packs.yaml"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return dict(raw.get("packs") or raw)


def _pack_flags_ou(
    *,
    side: str,
    close_odds: float | None,
    edge: float | None,
    pack: dict[str, Any],
) -> bool:
    """True if this OU side/odds/edge would qualify under a pack's filters."""
    if edge is None or close_odds is None:
        return False
    if not np.isfinite(edge) or not np.isfinite(close_odds):
        return False
    edge_by = dict(pack.get("edge_threshold_by_market") or {})
    thr = float(edge_by.get("ou25", pack.get("edge_threshold", 0.03)))
    if edge < thr:
        return False
    filt = pack.get("bet_filters") or {}
    # dummy match row — OU filters don't use match fields beyond odds/side
    match = pd.Series({})
    return passes_bet_filters(
        market="ou25",
        side=side,
        close_odds=float(close_odds),
        match=match,
        filt=filt,
    )


def _pack_flags_1x2(
    *,
    side: str,
    close_odds: float | None,
    edge: float | None,
    pack: dict[str, Any],
) -> bool:
    """True if this 1X2 side would qualify under a pack's filters."""
    if edge is None or close_odds is None:
        return False
    if not np.isfinite(edge) or not np.isfinite(close_odds):
        return False
    edge_by = dict(pack.get("edge_threshold_by_market") or {})
    thr = float(edge_by.get("1x2", pack.get("edge_threshold", 0.03)))
    if edge < thr:
        return False
    filt = pack.get("bet_filters") or {}
    return passes_bet_filters(
        market="1x2",
        side=side,
        close_odds=float(close_odds),
        match=pd.Series({}),
        filt=filt,
    )


def _pack_flags_ah(
    *,
    side: str,
    close_odds: float | None,
    edge: float | None,
    pack: dict[str, Any],
) -> bool:
    """True if this AH side would qualify under a pack's filters.

    ``side`` is ``ah_home`` or ``ah_away`` (bet_filters allow_sides may use these
    or omit allow_sides to accept either).
    """
    if edge is None or close_odds is None:
        return False
    if not np.isfinite(edge) or not np.isfinite(close_odds):
        return False
    edge_by = dict(pack.get("edge_threshold_by_market") or {})
    thr = float(edge_by.get("ah", pack.get("edge_threshold", 0.03)))
    if edge < thr:
        return False
    filt = pack.get("bet_filters") or {}
    return passes_bet_filters(
        market="ah",
        side=side,
        close_odds=float(close_odds),
        match=pd.Series({}),
        filt=filt,
    )


def predict_upcoming(
    history: pd.DataFrame,
    upcoming: pd.DataFrame,
    cfg: dict[str, Any],
    odds: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    *,
    apply_residual: bool = True,
) -> pd.DataFrame:
    """
    Fit model on completed history; optional fold-safe temperature/platt/isotonic;
    residual hybrid when enabled; predict upcoming fixtures.

    Same protocol as walk-forward test-fold inference (features → DC → totals
    intercept → calibration → residual).
    """
    hist = history.dropna(subset=["home_goals", "away_goals"]).copy()
    up = upcoming.copy()
    # Live AH: attach Pinnacle main line so predict_dataframe emits p_ah_*
    if odds is not None and len(odds) and "match_id" in odds.columns:
        ah_cols = [c for c in ("pin_ah_line", "ah_line") if c in odds.columns]
        if ah_cols:
            ah = odds[["match_id"] + ah_cols].drop_duplicates("match_id")
            up = up.merge(ah, on="match_id", how="left", suffixes=("", "_pin"))
            if "pin_ah_line" in up.columns:
                if "ah_line" in up.columns:
                    up["ah_line"] = up["ah_line"].where(up["ah_line"].notna(), up["pin_ah_line"])
                else:
                    up["ah_line"] = up["pin_ah_line"]
    as_of = pd.Timestamp(up["date"].min()) if len(up) else None

    # Build features on history ∪ upcoming so lagged rolls include as-of state
    if features is None:
        combo = pd.concat([hist, up], ignore_index=True, sort=False)
        # Stub scores for upcoming so panel builders don't crash; lag shift(1)
        # ensures current-match values are unused.
        for col in ("home_goals", "away_goals"):
            if col not in combo.columns:
                combo[col] = np.nan
            else:
                mask = combo["match_id"].isin(up["match_id"])
                combo.loc[mask, col] = np.nan
        if "season" not in combo.columns or combo["season"].isna().any():
            from origination.utils.seeding import season_from_date

            combo["season"] = combo["date"].map(season_from_date)
        features = build_feature_matrix(combo, cfg.get("features", {}))
        # Align feature rows to hist order (isin filter alone does not preserve hist order)
        feat_hist_aligned = (
            features.set_index("match_id").reindex(hist["match_id"]).reset_index()
        )
        assert_features_pre_match(hist.reset_index(drop=True), feat_hist_aligned)

    cal_cfg = cfg.get("model", {}).get("calibration", {})
    method = cal_cfg.get("method", "none")
    holdout = int(cal_cfg.get("holdout_seasons", 1))
    calibrator = build_calibrator(cfg)
    feat_hist = features[features["match_id"].isin(hist["match_id"])]
    feat_up = features[features["match_id"].isin(up["match_id"])]

    if method != "none" and holdout > 0:
        seasons = sorted(hist["season"].dropna().unique())
        if len(seasons) > holdout:
            calib_seasons = set(seasons[-holdout:])
            fit_df = hist[~hist["season"].isin(calib_seasons)]
            calib_df = hist[hist["season"].isin(calib_seasons)]
            model_c = build_model(cfg)
            model_c.fit(fit_df, as_of=as_of)
            feat_fit = features[features["match_id"].isin(fit_df["match_id"])]
            feat_calib = features[features["match_id"].isin(calib_df["match_id"])]
            apply_totals_intercept(model_c, fit_df, feat_fit, cfg)
            raw_c = model_c.predict_dataframe(calib_df, features=feat_calib)
            calibrator.fit(raw_c, calib_df)

    model = build_model(cfg)
    model.fit(hist, as_of=as_of)
    apply_totals_intercept(model, hist, feat_hist, cfg)
    intensity = None
    if "lam_mult_home" in feat_up.columns or "lam_mult_away" in feat_up.columns:
        intensity = feat_up[["match_id"]].copy()
        intensity["lam_mult_home"] = feat_up.get("lam_mult_home", 1.0)
        intensity["lam_mult_away"] = feat_up.get("lam_mult_away", 1.0)
    raw = model.predict_dataframe(up, features=feat_up, intensity_mults=intensity)
    preds = calibrator.transform(raw)

    if apply_residual:
        residual = build_residual_model(cfg)
        rcfg = cfg.get("model", {}).get("residual", {})
        if residual is not None:
            min_oos = int(rcfg.get("min_oos_rows", 400))
            bt_cfg = cfg.get("backtest", {})
            oos = collect_oos_base_predictions(
                hist,
                features,
                cfg,
                min_train=int(bt_cfg.get("min_train_matches", 500)),
                max_oos_seasons=int(rcfg.get("max_oos_seasons", 4)),
            )
            if len(oos) >= min_oos:
                oos_cal = calibrator.transform(oos)
                feat_oos = features[features["match_id"].isin(oos_cal["match_id"])]
                X_oos, cols = _numeric_feature_matrix(
                    feat_oos,
                    oos_cal,
                    interactions=bool(getattr(residual, "interactions", False)),
                    interaction_pairs=getattr(residual, "interaction_pairs", None),
                )
                matches_oos = hist[hist["match_id"].isin(X_oos["match_id"])]
                residual.fit(X_oos, matches_oos, cols)
                preds = residual.transform(preds, feat_up)
                logger.info("Residual applied on upcoming (OOS fit rows={})", len(X_oos))
            else:
                logger.warning(
                    "Residual skipped — OOS rows {} < {}", len(oos), min_oos
                )

    # Attach fixture meta
    meta_cols = [c for c in ("match_id", "date", "home_team", "away_team", "season") if c in up.columns]
    preds = preds.merge(up[meta_cols], on="match_id", how="left", suffixes=("", "_fix"))

    if odds is not None and len(odds):
        preds = preds.merge(odds, on="match_id", how="left", suffixes=("", "_odds"))

    return preds


def build_gameday_sheet(
    preds: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    packs_path: Path | None = None,
    active_packs: list[str] | None = None,
) -> pd.DataFrame:
    """
    Enrich predictions into a gameday comparison sheet.

    Edge calculation (documented):
      - fair_prob_over = p_over25, fair_odds = 1/p
      - vs sharp two-way (close_over25 / close_under25 or book_*):
            edge = model_prob - vig_removed_fair(side)   [power by default]
      - vs single reference line (ref_over25 or book_over25 alone):
            edge_single = model_prob - 1/odds
      - consensus: average of available book decimal prices, then same formulas

    Pack flags are independent — Under pack vs short-Over pack never merge.
    """
    vig = cfg.get("backtest", {}).get("odds", {}).get("remove_vig", "power")
    packs = _load_rule_packs(packs_path)
    # Multi-league: use packs listed in kwargs / cfg, else EPL defaults
    pack_names = list(active_packs) if active_packs else ["EPL_aggressive", "EPL_overs_short_exp"]
    loaded_packs = [(name, packs.get(name) or {}) for name in pack_names]

    rows = []
    for _, r in preds.iterrows():
        lam = float(r.get("lambda_home", np.nan))
        mu = float(r.get("lambda_away", np.nan))
        p_o = float(r.get("p_over25", np.nan))
        p_u = float(r.get("p_under25", np.nan))
        fair_o = fair_decimal_odds(p_o)
        fair_u = fair_decimal_odds(p_u)

        def _get(*names: str) -> float | None:
            for n in names:
                v = r.get(n)
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    continue
                if pd.isna(v):
                    continue
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    continue
                if x > 1.0:
                    return x
            return None

        # Sharp reference: Pinnacle preferred (pin_* / ref_*), then legacy aliases
        over_sharp = _get(
            "pin_over25",
            "ref_over25",
            "pinnacle_over25",
            "close_over25",
            "ps_over25",
        )
        under_sharp = _get(
            "pin_under25",
            "ref_under25",
            "pinnacle_under25",
            "close_under25",
            "ps_under25",
        )

        # User / soft book (explicit book_* only — not pin/ref)
        over_book = _get("book_over25", "user_over25")
        under_book = _get("book_under25", "user_under25")

        def _safe_odds_float(val: Any) -> float | None:
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                return None
            if pd.isna(val):
                return None
            try:
                x = float(val)
            except (TypeError, ValueError):
                return None
            return x if x > 1.0 else None

        # Consensus: mean of soft-book OU columns only (never pin/ref/fair/ids)
        skip_prefixes = (
            "pin_",
            "ref_",
            "fair_",
            "p_",
            "edge",
            "flag",
            "proj",
            "lambda",
            "match_id",
            "home_team",
            "away_team",
            "date",
            "season",
            "kickoff",
        )
        over_cand: list[float] = []
        under_cand: list[float] = []
        for c in r.index:
            cl = str(c).lower()
            if "over25" not in cl and "under25" not in cl:
                continue
            if any(cl.startswith(p) or cl == p.rstrip("_") for p in skip_prefixes):
                continue
            if "edge" in cl or "flag" in cl or "prob" in cl:
                continue
            x = _safe_odds_float(r[c])
            if x is None:
                continue
            if "over25" in cl:
                over_cand.append(x)
            if "under25" in cl:
                under_cand.append(x)
        over_cons = float(np.mean(over_cand)) if over_cand else None
        under_cons = float(np.mean(under_cand)) if under_cand else None

        edge_over_sharp = (
            model_edge_vs_two_way(p_o, over_sharp, under_sharp, side="over", method=vig)
            if over_sharp and under_sharp
            else model_edge_vs_odds(p_o, over_sharp) if over_sharp else None
        )
        edge_under_sharp = (
            model_edge_vs_two_way(p_u, over_sharp, under_sharp, side="under", method=vig)
            if over_sharp and under_sharp
            else model_edge_vs_odds(p_u, under_sharp) if under_sharp else None
        )
        edge_over_book = (
            model_edge_vs_two_way(p_o, over_book, under_book, side="over", method=vig)
            if over_book and under_book
            else model_edge_vs_odds(p_o, over_book) if over_book else None
        )
        edge_under_book = (
            model_edge_vs_two_way(p_u, over_book, under_book, side="under", method=vig)
            if over_book and under_book
            else model_edge_vs_odds(p_u, under_book) if under_book else None
        )
        edge_over_cons = (
            model_edge_vs_two_way(p_o, over_cons, under_cons, side="over", method=vig)
            if over_cons and under_cons
            else None
        )
        edge_under_cons = (
            model_edge_vs_two_way(p_u, over_cons, under_cons, side="under", method=vig)
            if over_cons and under_cons
            else None
        )

        # Pack flags use Pinnacle/sharp reference only (independent packs)
        flag_cols: dict[str, bool] = {}
        # 1X2 odds if present (book/pin/close)
        odds_h = _get("pin_h", "pin_home", "book_h", "book_home", "close_h", "PSCH")
        odds_d = _get("pin_d", "pin_draw", "book_d", "book_draw", "close_d", "PSCD")
        odds_a = _get("pin_a", "pin_away", "book_a", "book_away", "close_a", "PSCA")
        p_h = float(r.get("p_home", np.nan)) if pd.notna(r.get("p_home")) else np.nan
        p_d = float(r.get("p_draw", np.nan)) if pd.notna(r.get("p_draw")) else np.nan
        p_a = float(r.get("p_away", np.nan)) if pd.notna(r.get("p_away")) else np.nan
        edge_h = model_edge_vs_odds(p_h, odds_h) if odds_h and np.isfinite(p_h) else None
        edge_d = model_edge_vs_odds(p_d, odds_d) if odds_d and np.isfinite(p_d) else None
        edge_a = model_edge_vs_odds(p_a, odds_a) if odds_a and np.isfinite(p_a) else None

        # Asian Handicap (Pinnacle main line)
        ah_line_val = r.get("pin_ah_line", r.get("ah_line"))
        try:
            ah_line_f = float(ah_line_val) if pd.notna(ah_line_val) else None
        except (TypeError, ValueError):
            ah_line_f = None
        pin_ahh = _get("pin_ahh", "close_ahh")
        pin_aha = _get("pin_aha", "close_aha")
        book_ahh = _get("book_ahh", "user_ahh")
        book_aha = _get("book_aha", "user_aha")
        p_ahh = float(r.get("p_ah_home", np.nan)) if pd.notna(r.get("p_ah_home")) else np.nan
        p_aha = float(r.get("p_ah_away", np.nan)) if pd.notna(r.get("p_ah_away")) else np.nan
        edge_ahh = edge_aha = None
        if pin_ahh and pin_aha and np.isfinite(p_ahh) and np.isfinite(p_aha):
            fh, fa = two_way_fair(pin_ahh, pin_aha, method=vig)
            edge_ahh = float(p_ahh - fh)
            edge_aha = float(p_aha - fa)
        elif pin_ahh and np.isfinite(p_ahh):
            edge_ahh = model_edge_vs_odds(p_ahh, pin_ahh)
        elif pin_aha and np.isfinite(p_aha):
            edge_aha = model_edge_vs_odds(p_aha, pin_aha)
        if book_ahh and book_aha and np.isfinite(p_ahh) and np.isfinite(p_aha):
            fb_h, fb_a = two_way_fair(book_ahh, book_aha, method=vig)
            edge_ahh_book = float(p_ahh - fb_h)
            edge_aha_book = float(p_aha - fb_a)
        else:
            edge_ahh_book = model_edge_vs_odds(p_ahh, book_ahh) if book_ahh and np.isfinite(p_ahh) else None
            edge_aha_book = model_edge_vs_odds(p_aha, book_aha) if book_aha and np.isfinite(p_aha) else None
        fair_ahh = fair_decimal_odds(p_ahh) if np.isfinite(p_ahh) else None
        fair_aha = fair_decimal_odds(p_aha) if np.isfinite(p_aha) else None

        for pname, pack in loaded_packs:
            safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in pname)
            flag_cols[f"flag_{safe}_under"] = _pack_flags_ou(
                side="under", close_odds=under_sharp, edge=edge_under_sharp, pack=pack
            )
            flag_cols[f"flag_{safe}_over"] = _pack_flags_ou(
                side="over", close_odds=over_sharp, edge=edge_over_sharp, pack=pack
            )
            flag_cols[f"flag_{safe}_1x2_H"] = _pack_flags_1x2(
                side="H", close_odds=odds_h, edge=edge_h, pack=pack
            )
            flag_cols[f"flag_{safe}_1x2_D"] = _pack_flags_1x2(
                side="D", close_odds=odds_d, edge=edge_d, pack=pack
            )
            flag_cols[f"flag_{safe}_1x2_A"] = _pack_flags_1x2(
                side="A", close_odds=odds_a, edge=edge_a, pack=pack
            )
            flag_cols[f"flag_{safe}_ah_home"] = _pack_flags_ah(
                side="ah_home", close_odds=pin_ahh, edge=edge_ahh, pack=pack
            )
            flag_cols[f"flag_{safe}_ah_away"] = _pack_flags_ah(
                side="ah_away", close_odds=pin_aha, edge=edge_aha, pack=pack
            )
        # Backward-compatible EPL names
        flag_under = flag_cols.get("flag_EPL_aggressive_under", False)
        flag_over = flag_cols.get("flag_EPL_overs_short_exp_over", False)

        row = {
                "match_id": r.get("match_id"),
                "date": r.get("date"),
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
                "proj_home_goals": lam,
                "proj_away_goals": mu,
                "proj_total_goals": lam + mu if np.isfinite(lam) and np.isfinite(mu) else np.nan,
                "p_over25": p_o,
                "p_under25": p_u,
                "fair_odds_over25": fair_o,
                "fair_odds_under25": fair_u,
                "fair_american_over25": decimal_to_american(fair_o) if fair_o else None,
                "fair_american_under25": decimal_to_american(fair_u) if fair_u else None,
                "pin_over25": over_sharp,
                "pin_under25": under_sharp,
                "pin_american_over25": decimal_to_american(over_sharp) if over_sharp else None,
                "pin_american_under25": decimal_to_american(under_sharp) if under_sharp else None,
                "pin_matchup_id": r.get("pin_matchup_id"),
                "ref_over25": over_sharp,
                "ref_under25": under_sharp,
                "book_over25": over_book,
                "book_under25": under_book,
                "consensus_over25": over_cons,
                "consensus_under25": under_cons,
                "edge_over_vs_pinnacle": edge_over_sharp,
                "edge_under_vs_pinnacle": edge_under_sharp,
                "edge_over_vs_ref": edge_over_sharp,
                "edge_under_vs_ref": edge_under_sharp,
                "edge_over_vs_book": edge_over_book,
                "edge_under_vs_book": edge_under_book,
                "edge_over_vs_consensus": edge_over_cons,
                "edge_under_vs_consensus": edge_under_cons,
                "odds_1x2_h": odds_h,
                "odds_1x2_d": odds_d,
                "odds_1x2_a": odds_a,
                "edge_1x2_h": edge_h,
                "edge_1x2_d": edge_d,
                "edge_1x2_a": edge_a,
                "book_h": _get("book_h", "user_h"),
                "book_a": _get("book_a", "user_a"),
                "edge_1x2_h_vs_book": model_edge_vs_odds(p_h, _get("book_h", "user_h"))
                if _get("book_h", "user_h") and np.isfinite(p_h)
                else None,
                "edge_1x2_a_vs_book": model_edge_vs_odds(p_a, _get("book_a", "user_a"))
                if _get("book_a", "user_a") and np.isfinite(p_a)
                else None,
                "pin_h_american": r.get("pin_h_american")
                or (decimal_to_american(odds_h) if odds_h else None),
                "pin_a_american": r.get("pin_a_american")
                or (decimal_to_american(odds_a) if odds_a else None),
                "fair_odds_home": fair_decimal_odds(p_h) if np.isfinite(p_h) else None,
                "fair_odds_away": fair_decimal_odds(p_a) if np.isfinite(p_a) else None,
                "fair_american_home": decimal_to_american(fair_decimal_odds(p_h))
                if np.isfinite(p_h)
                else None,
                "fair_american_away": decimal_to_american(fair_decimal_odds(p_a))
                if np.isfinite(p_a)
                else None,
                "ah_line": ah_line_f,
                "pin_ahh": pin_ahh,
                "pin_aha": pin_aha,
                "pin_ahh_american": r.get("pin_ahh_american")
                or (decimal_to_american(pin_ahh) if pin_ahh else None),
                "pin_aha_american": r.get("pin_aha_american")
                or (decimal_to_american(pin_aha) if pin_aha else None),
                "book_ahh": book_ahh,
                "book_aha": book_aha,
                "p_ah_home": p_ahh if np.isfinite(p_ahh) else None,
                "p_ah_away": p_aha if np.isfinite(p_aha) else None,
                "fair_odds_ah_home": fair_ahh,
                "fair_odds_ah_away": fair_aha,
                "fair_american_ah_home": decimal_to_american(fair_ahh) if fair_ahh else None,
                "fair_american_ah_away": decimal_to_american(fair_aha) if fair_aha else None,
                "edge_ah_home": edge_ahh,
                "edge_ah_away": edge_aha,
                "edge_ah_home_vs_book": edge_ahh_book,
                "edge_ah_away_vs_book": edge_aha_book,
                "flag_EPL_aggressive_under": flag_under,
                "flag_EPL_overs_short": flag_over,
                **flag_cols,
                "p_home": r.get("p_home"),
                "p_draw": r.get("p_draw"),
                "p_away": r.get("p_away"),
                "p_over15": r.get("p_over15"),
                "p_under15": r.get("p_under15"),
                "p_over35": r.get("p_over35"),
                "p_under35": r.get("p_under35"),
            }
        rows.append(row)
    sheet = pd.DataFrame(rows)
    if len(sheet) and "date" in sheet.columns:
        sheet = sheet.sort_values(["date", "match_id"]).reset_index(drop=True)
    # Odds coverage (robust missing-odds handling for UI / scan)
    if len(sheet):
        def _has_price(val: Any) -> bool:
            try:
                x = float(val)
            except (TypeError, ValueError):
                return False
            return bool(np.isfinite(x) and x > 1.0)

        ou_o = sheet["pin_over25"] if "pin_over25" in sheet.columns else pd.Series([None] * len(sheet))
        ou_u = sheet["pin_under25"] if "pin_under25" in sheet.columns else pd.Series([None] * len(sheet))
        ml_h = sheet["odds_1x2_h"] if "odds_1x2_h" in sheet.columns else pd.Series([None] * len(sheet))
        ml_a = sheet["odds_1x2_a"] if "odds_1x2_a" in sheet.columns else pd.Series([None] * len(sheet))
        sheet["has_pin_ou"] = [_has_price(o) and _has_price(u) for o, u in zip(ou_o, ou_u)]
        sheet["has_pin_1x2"] = [_has_price(h) and _has_price(a) for h, a in zip(ml_h, ml_a)]
        ah_h = sheet["pin_ahh"] if "pin_ahh" in sheet.columns else pd.Series([None] * len(sheet))
        ah_a = sheet["pin_aha"] if "pin_aha" in sheet.columns else pd.Series([None] * len(sheet))
        sheet["has_pin_ah"] = [_has_price(h) and _has_price(a) for h, a in zip(ah_h, ah_a)]

        def _odds_status(row: pd.Series) -> str:
            bits = []
            if bool(row.get("has_pin_ou")):
                bits.append("OU")
            if bool(row.get("has_pin_1x2")):
                bits.append("1X2")
            if bool(row.get("has_pin_ah")):
                bits.append("AH")
            return "+".join(bits) if bits else "MISSING"

        sheet["odds_status"] = sheet.apply(_odds_status, axis=1)

    flag_cols_present = [
        c for c in sheet.columns if str(c).startswith("flag_") and c != "flag_any"
    ]
    # Friendly labels for the 5 protected systems (rules unchanged)
    SYSTEM_LABELS = {
        "flag_EPL_aggressive_under": "EPL Unders",
        "flag_EPL_overs_short_exp_over": "EPL short Overs",
        "flag_EPL_overs_short_over": "EPL short Overs",
        "flag_Bundesliga_unders_short_exp_under": "Bundesliga Unders",
        "flag_LaLiga_home_ml_short_exp_1x2_H": "La Liga Home ML",
        "flag_SerieA_away_ml_exp_1x2_A": "Serie A Away ML",
        "flag_PrimeiraLiga_ah_e12_exp_ah_home": "Primeira AH e12% (home)",
        "flag_PrimeiraLiga_ah_e12_exp_ah_away": "Primeira AH e12% (away)",
        "flag_PrimeiraLiga_ah_short_exp_ah_home": "Primeira AH e10% (home)",
        "flag_PrimeiraLiga_ah_short_exp_ah_away": "Primeira AH e10% (away)",
    }
    if flag_cols_present and len(sheet):
        sheet["flag_any"] = sheet[flag_cols_present].fillna(False).astype(bool).any(axis=1)

        def _systems_fired(row: pd.Series) -> str:
            names = []
            for c in flag_cols_present:
                if not bool(row.get(c)):
                    continue
                names.append(SYSTEM_LABELS.get(c, c.replace("flag_", "")))
            # de-dupe while preserving order
            seen: set[str] = set()
            out: list[str] = []
            for n in names:
                if n not in seen:
                    seen.add(n)
                    out.append(n)
            return " | ".join(out)

        sheet["systems_flagged"] = sheet.apply(_systems_fired, axis=1)
    flag_sum = {
        c: int(sheet[c].sum()) for c in sheet.columns if str(c).startswith("flag_") and len(sheet)
    }
    logger.info("Gameday sheet: {} fixtures | flags={}", len(sheet), flag_sum)
    return sheet


def load_odds_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "match_id" not in df.columns:
        raise ValueError("odds file must include match_id")
    return df


def load_late_info_csv(path: Path) -> pd.DataFrame:
    """
    Optional late information: confirmed lineups / injuries / squad deltas.

    Expected columns (any subset):
      match_id
      lineup_strength_home, lineup_strength_away, lineup_confirmed
      injury_attack_delta_home, injury_attack_delta_away,
      injury_defence_delta_home, injury_defence_delta_away
      lam_mult_home, lam_mult_away   # direct intensity overrides (optional)
    """
    df = pd.read_csv(path)
    if "match_id" not in df.columns:
        raise ValueError("late-info file must include match_id")
    return df


def apply_late_info_to_features(
    features: pd.DataFrame,
    late: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Merge late XI/injury signals into the feature matrix for upcoming matches."""
    if late is None or len(late) == 0:
        return features
    feat = features.copy()
    merged = feat.merge(late, on="match_id", how="left", suffixes=("", "_late"))
    # Prefer late columns when present
    for col in late.columns:
        if col == "match_id":
            continue
        late_col = f"{col}_late" if f"{col}_late" in merged.columns else col
        if late_col in merged.columns and col in merged.columns:
            merged[col] = merged[late_col].combine_first(merged[col])
        elif late_col in merged.columns:
            merged[col] = merged[late_col]
    # Drop *_late duplicates
    drop = [c for c in merged.columns if str(c).endswith("_late")]
    merged = merged.drop(columns=drop, errors="ignore")

    # Optional: map strength deltas into intensity multipliers used by predict
    coef = float(
        cfg.get("features", {})
        .get("context_adjustments", {})
        .get("lineups", {})
        .get("strength_coef", 0.05)
    )
    if "lineup_strength_home" in merged.columns and "lam_mult_home" not in late.columns:
        # Relative to 0 = neutral; positive home strength → higher home λ
        sh = merged["lineup_strength_home"].astype(float)
        sa = merged["lineup_strength_away"].astype(float)
        if "lam_mult_home" not in merged.columns:
            merged["lam_mult_home"] = 1.0
        if "lam_mult_away" not in merged.columns:
            merged["lam_mult_away"] = 1.0
        mask = sh.notna() | sa.notna()
        merged.loc[mask, "lam_mult_home"] = (
            merged.loc[mask, "lam_mult_home"].fillna(1.0)
            * np.exp(coef * sh.fillna(0.0).loc[mask])
        )
        merged.loc[mask, "lam_mult_away"] = (
            merged.loc[mask, "lam_mult_away"].fillna(1.0)
            * np.exp(coef * sa.fillna(0.0).loc[mask])
        )
    return merged
