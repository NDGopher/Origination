"""
Walk-forward backtester with Closing Line Value as the primary metric.

Train only on past data → predict next fold → compare to vig-removed closing odds.
Calibration is fit on a holdout from the training window only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from origination.features.store import assert_features_pre_match, build_feature_matrix
from origination.models.calibration import ProbabilityCalibrator, build_calibrator
from origination.models.poisson import apply_totals_intercept, build_model
from origination.models.residual import (
    build_residual_model,
    collect_oos_base_predictions,
    _numeric_feature_matrix,
)
from origination.utils.odds import (
    apply_stake,
    clv_odds,
    clv_probability,
    fair_probs,
    two_way_fair,
)


@dataclass
class BacktestResult:
    experiment_id: str
    config: dict[str, Any]
    bets: pd.DataFrame
    predictions: pd.DataFrame
    summary: dict[str, Any]
    by_season: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_edge_bucket: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_threshold: pd.DataFrame = field(default_factory=pd.DataFrame)


def _folds_by_season(
    matches: pd.DataFrame, min_train: int
) -> list[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    seasons = sorted(matches["season"].dropna().unique())
    folds = []
    for season in seasons:
        test = matches[matches["season"] == season]
        train = matches[matches["season"] < season]
        if len(train) < min_train:
            logger.info("Skip season {} — train size {} < {}", season, len(train), min_train)
            continue
        if len(test) == 0:
            continue
        folds.append((pd.Timestamp(f"{int(season)}-08-01"), train, test))
    return folds


def _split_fit_calib(
    train: pd.DataFrame, holdout_seasons: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last ``holdout_seasons`` seasons of train for calibration."""
    seasons = sorted(train["season"].dropna().unique())
    if holdout_seasons <= 0 or len(seasons) < holdout_seasons + 1:
        return train, train
    calib_seasons = set(seasons[-holdout_seasons:])
    fit_df = train[~train["season"].isin(calib_seasons)]
    calib_df = train[train["season"].isin(calib_seasons)]
    if len(fit_df) < 200:
        return train, train
    return fit_df, calib_df


def _edge_bucket(edge: float) -> str:
    if edge < 0.03:
        return "0-3%"
    if edge < 0.05:
        return "3-5%"
    if edge < 0.08:
        return "5-8%"
    if edge < 0.12:
        return "8-12%"
    return "12%+"


def _settle_1x2(ftr: str, side: str) -> float:
    return 1.0 if ftr == side else 0.0


def _settle_ou25(total_goals: float, side: str) -> float:
    if side == "over":
        return 1.0 if total_goals > 2.5 else 0.0
    return 1.0 if total_goals < 2.5 else 0.0


def evaluate_predictions(
    preds: pd.DataFrame,
    matches: pd.DataFrame,
    bt_cfg: dict[str, Any],
    *,
    edge_threshold: float | None = None,
) -> pd.DataFrame:
    """Compare model probs to closing odds; emit bet-level rows for selected edges."""
    from origination.backtesting.bet_filters import passes_bet_filters, passes_edge_rule

    vig = bt_cfg.get("odds", {}).get("remove_vig", "power")
    edge_thr_default = float(
        edge_threshold if edge_threshold is not None else bt_cfg.get("edge_threshold", 0.03)
    )
    edge_by_mkt = dict(bt_cfg.get("edge_threshold_by_market") or {})

    def _edge_thr(market: str) -> float:
        if market in edge_by_mkt:
            return float(edge_by_mkt[market])
        return edge_thr_default

    markets = bt_cfg.get("markets", ["1x2"])
    stake_cfg = bt_cfg.get("stake", {})
    min_odds = float(bt_cfg.get("odds", {}).get("min_odds", 1.2))
    max_odds = float(bt_cfg.get("odds", {}).get("max_odds", 15.0))
    bet_filt = bt_cfg.get("bet_filters") or {}

    m = matches.set_index("match_id")
    bets: list[dict[str, Any]] = []

    for _, row in preds.iterrows():
        mid = row["match_id"]
        if mid not in m.index:
            continue
        match = m.loc[mid]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]

        if "1x2" in markets:
            edge_thr = _edge_thr("1x2")
            close_odds = np.array(
                [match["close_h"], match["close_d"], match["close_a"]], dtype=float
            )
            if np.all(np.isfinite(close_odds)):
                fair = fair_probs(close_odds, method=vig)
                sides = [
                    ("H", float(row["p_home"]), float(close_odds[0]), float(fair[0])),
                    ("D", float(row["p_draw"]), float(close_odds[1]), float(fair[1])),
                    ("A", float(row["p_away"]), float(close_odds[2]), float(fair[2])),
                ]
                for side, mp, odds, fp in sides:
                    edge = mp - fp
                    if edge < edge_thr:
                        continue
                    if not (min_odds <= odds <= max_odds):
                        continue
                    if not passes_bet_filters(
                        market="1x2",
                        side=side,
                        close_odds=odds,
                        match=match,
                        filt=bet_filt,
                    ):
                        continue
                    if not passes_edge_rule(edge, "1x2", bet_filt):
                        continue
                    stake = apply_stake(
                        mp,
                        odds,
                        method=stake_cfg.get("method", "flat"),
                        unit=float(stake_cfg.get("unit", 1.0)),
                        kelly_fraction_mult=float(stake_cfg.get("kelly_fraction", 0.25)),
                        max_stake=float(stake_cfg.get("max_stake", 5.0)),
                    )
                    won = _settle_1x2(str(match["ftr"]), side)
                    profit = stake * ((odds - 1.0) if won else -1.0)
                    bets.append(
                        {
                            "match_id": mid,
                            "date": match["date"],
                            "season": match["season"],
                            "market": "1x2",
                            "side": side,
                            "model_prob": mp,
                            "close_fair_prob": fp,
                            "close_odds": odds,
                            "edge": edge,
                            "clv_prob": clv_probability(mp, fp),
                            "clv_odds": clv_odds(mp, odds),
                            "stake": stake,
                            "won": won,
                            "profit": profit,
                            "edge_bucket": _edge_bucket(edge),
                            "edge_threshold": edge_thr,
                            "home_team": match["home_team"],
                            "away_team": match["away_team"],
                        }
                    )

        if (
            "ou25" in markets
            and pd.notna(match.get("close_over25"))
            and pd.notna(match.get("close_under25"))
        ):
            edge_thr = _edge_thr("ou25")
            o_odds = float(match["close_over25"])
            u_odds = float(match["close_under25"])
            if o_odds > 1.0 and u_odds > 1.0:
                fo, fu = two_way_fair(o_odds, u_odds, method=vig)
                for side, mp, odds, fp in [
                    ("over", float(row["p_over25"]), o_odds, fo),
                    ("under", float(row["p_under25"]), u_odds, fu),
                ]:
                    edge = mp - fp
                    if edge < edge_thr:
                        continue
                    if not (min_odds <= odds <= max_odds):
                        continue
                    if not passes_bet_filters(
                        market="ou25",
                        side=side,
                        close_odds=odds,
                        match=match,
                        filt=bet_filt,
                    ):
                        continue
                    if not passes_edge_rule(edge, "ou25", bet_filt):
                        continue
                    stake = apply_stake(
                        mp,
                        odds,
                        method=stake_cfg.get("method", "flat"),
                        unit=float(stake_cfg.get("unit", 1.0)),
                        kelly_fraction_mult=float(stake_cfg.get("kelly_fraction", 0.25)),
                        max_stake=float(stake_cfg.get("max_stake", 5.0)),
                    )
                    won = _settle_ou25(float(match["total_goals"]), side)
                    profit = stake * ((odds - 1.0) if won else -1.0)
                    bets.append(
                        {
                            "match_id": mid,
                            "date": match["date"],
                            "season": match["season"],
                            "market": "ou25",
                            "side": side,
                            "model_prob": mp,
                            "close_fair_prob": fp,
                            "close_odds": odds,
                            "edge": edge,
                            "clv_prob": clv_probability(mp, fp),
                            "clv_odds": clv_odds(mp, odds),
                            "stake": stake,
                            "won": won,
                            "profit": profit,
                            "edge_bucket": _edge_bucket(edge),
                            "edge_threshold": edge_thr,
                            "home_team": match["home_team"],
                            "away_team": match["away_team"],
                        }
                    )

        if "ah" in markets and pd.notna(match.get("ah_line")):
            edge_thr = _edge_thr("ah")
            line = float(match["ah_line"])
            ahh = match.get("close_ahh")
            aha = match.get("close_aha")
            if pd.notna(ahh) and pd.notna(aha) and float(ahh) > 1.0 and float(aha) > 1.0:
                fh, fa = two_way_fair(float(ahh), float(aha), method=vig)
                mh = float(row.get("p_ah_home", np.nan))
                ma = float(row.get("p_ah_away", np.nan))
                # Fallbacks for legacy prediction files without p_ah_*
                if not np.isfinite(mh) or not np.isfinite(ma):
                    if abs(line) < 0.01:
                        mh = float(row.get("p_ah0_home", np.nan))
                        ma = float(row.get("p_ah0_away", np.nan))
                    elif abs(line + 0.5) < 0.01:
                        mh = float(row.get("p_home", np.nan))
                        ma = 1.0 - mh if np.isfinite(mh) else np.nan
                if np.isfinite(mh) and np.isfinite(ma):
                    from origination.models.poisson import ah_settle_fraction

                    gd = float(match["home_goals"] - match["away_goals"])
                    for side, mp, odds, fp in [
                        ("ah_home", mh, float(ahh), fh),
                        ("ah_away", ma, float(aha), fa),
                    ]:
                        edge = mp - fp
                        if edge < edge_thr:
                            continue
                        if not (min_odds <= odds <= max_odds):
                            continue
                        if not passes_bet_filters(
                            market="ah",
                            side=side,
                            close_odds=odds,
                            match=match,
                            filt=bet_filt,
                        ):
                            continue
                        if not passes_edge_rule(edge, "ah", bet_filt):
                            continue
                        stake = apply_stake(
                            mp,
                            odds,
                            method=stake_cfg.get("method", "flat"),
                            unit=float(stake_cfg.get("unit", 1.0)),
                            kelly_fraction_mult=float(stake_cfg.get("kelly_fraction", 0.25)),
                            max_stake=float(stake_cfg.get("max_stake", 5.0)),
                        )
                        frac = ah_settle_fraction(gd, line, side)
                        # frac: 1=win, 0.5=push, 0=lose; quarter averages in between
                        if abs(frac - 0.5) < 1e-9:
                            won = 0.5
                            profit = 0.0
                        elif frac > 0.5:
                            # full or half win: profit = stake * (odds-1) * 2*(frac-0.5)
                            # half-win frac=0.75 → half stake at odds; full frac=1 → full
                            win_portion = 2.0 * (frac - 0.5)
                            won = frac
                            profit = stake * (odds - 1.0) * win_portion
                        else:
                            # full or half lose
                            lose_portion = 2.0 * (0.5 - frac)
                            won = frac
                            profit = -stake * lose_portion
                        bets.append(
                            {
                                "match_id": mid,
                                "date": match["date"],
                                "season": match["season"],
                                "market": "ah",
                                "side": side,
                                "model_prob": mp,
                                "close_fair_prob": fp,
                                "close_odds": odds,
                                "edge": edge,
                                "clv_prob": clv_probability(mp, fp),
                                "clv_odds": clv_odds(mp, odds),
                                "stake": stake,
                                "won": won,
                                "profit": profit,
                                "edge_bucket": _edge_bucket(edge),
                                "edge_threshold": edge_thr,
                                "home_team": match["home_team"],
                                "away_team": match["away_team"],
                                "ah_line": line,
                            }
                        )

    return pd.DataFrame(bets)


def _summarize(
    bets: pd.DataFrame, preds: pd.DataFrame, matches: pd.DataFrame
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_predictions": int(len(preds)),
        "n_bets": int(len(bets)),
    }
    if len(bets) == 0:
        summary["avg_clv_prob"] = None
        summary["avg_clv_odds"] = None
        summary["roi"] = None
    else:
        settled = bets.dropna(subset=["profit"])
        summary["avg_clv_prob"] = float(bets["clv_prob"].mean())
        summary["avg_clv_odds"] = float(bets["clv_odds"].mean())
        summary["median_clv_prob"] = float(bets["clv_prob"].median())
        summary["hit_rate"] = float(settled["won"].mean()) if len(settled) else None
        stake_sum = float(settled["stake"].sum()) if len(settled) else 0.0
        profit_sum = float(settled["profit"].sum()) if len(settled) else 0.0
        summary["units_profit"] = profit_sum
        summary["stake_total"] = stake_sum
        summary["roi"] = profit_sum / stake_sum if stake_sum > 0 else None

    m = matches.set_index("match_id")
    ll_model: list[float] = []
    ll_market: list[float] = []
    brier: list[float] = []
    for _, row in preds.iterrows():
        if row["match_id"] not in m.index:
            continue
        match = m.loc[row["match_id"]]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        probs = np.array([row["p_home"], row["p_draw"], row["p_away"]], dtype=float)
        probs = probs / probs.sum()
        probs = np.clip(probs, 1e-6, 1.0)
        outcome = {"H": 0, "D": 1, "A": 2}[str(match["ftr"])]
        y = np.zeros(3)
        y[outcome] = 1.0
        ll_model.append(float(-np.log(probs[outcome])))
        brier.append(float(np.sum((probs - y) ** 2)))
        close = np.array([match["close_h"], match["close_d"], match["close_a"]], dtype=float)
        if np.all(np.isfinite(close)) and np.all(close > 1.0):
            mkt = np.clip(fair_probs(close, method="power"), 1e-6, 1.0)
            ll_market.append(float(-np.log(mkt[outcome])))

    if ll_model:
        summary["log_loss_1x2"] = float(np.mean(ll_model))
        summary["brier_1x2"] = float(np.mean(brier))
    if ll_market:
        summary["log_loss_market_1x2"] = float(np.mean(ll_market))
        summary["log_loss_edge_vs_market"] = float(np.mean(ll_market) - np.mean(ll_model))

    # O/U 2.5 log-loss vs market (secondary tracking)
    ll_ou_model: list[float] = []
    ll_ou_market: list[float] = []
    for _, row in preds.iterrows():
        if row["match_id"] not in m.index:
            continue
        match = m.loc[row["match_id"]]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        if "p_over25" not in row or not np.isfinite(float(row.get("p_over25", np.nan))):
            continue
        if pd.isna(match.get("total_goals")):
            continue
        y_over = 1.0 if float(match["total_goals"]) > 2.5 else 0.0
        po = float(np.clip(row["p_over25"], 1e-6, 1.0 - 1e-6))
        ll_ou_model.append(float(-(y_over * np.log(po) + (1 - y_over) * np.log(1 - po))))
        o_odds = match.get("close_over25")
        u_odds = match.get("close_under25")
        if pd.notna(o_odds) and pd.notna(u_odds) and float(o_odds) > 1.0 and float(u_odds) > 1.0:
            fo, fu = two_way_fair(float(o_odds), float(u_odds), method="power")
            pm = float(np.clip(fo, 1e-6, 1.0 - 1e-6))
            ll_ou_market.append(float(-(y_over * np.log(pm) + (1 - y_over) * np.log(1 - pm))))
    if ll_ou_model:
        summary["log_loss_ou25"] = float(np.mean(ll_ou_model))
    if ll_ou_market:
        summary["log_loss_market_ou25"] = float(np.mean(ll_ou_market))
        if ll_ou_model:
            summary["log_loss_edge_vs_market_ou25"] = float(
                np.mean(ll_ou_market) - np.mean(ll_ou_model)
            )

    return summary


def _threshold_sweep(
    preds: pd.DataFrame,
    matches: pd.DataFrame,
    bt_cfg: dict[str, Any],
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        bets = evaluate_predictions(preds, matches, bt_cfg, edge_threshold=thr)
        if len(bets) == 0:
            rows.append(
                {
                    "edge_threshold": thr,
                    "n_bets": 0,
                    "avg_clv_prob": None,
                    "roi": None,
                    "units_profit": 0.0,
                    "hit_rate": None,
                }
            )
            continue
        settled = bets.dropna(subset=["profit"])
        stake = float(settled["stake"].sum())
        profit = float(settled["profit"].sum())
        rows.append(
            {
                "edge_threshold": thr,
                "n_bets": int(len(bets)),
                "avg_clv_prob": float(bets["clv_prob"].mean()),
                "avg_clv_odds": float(bets["clv_odds"].mean()),
                "roi": profit / stake if stake > 0 else None,
                "units_profit": profit,
                "hit_rate": float(settled["won"].mean()) if len(settled) else None,
            }
        )
    return pd.DataFrame(rows)


def run_walk_forward(
    matches: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    experiments_dir: Path | None = None,
    features: pd.DataFrame | None = None,
) -> BacktestResult:
    """
    Full walk-forward loop with optional fold-safe calibration.
    """
    bt_cfg = cfg.get("backtest", {})
    min_train = int(bt_cfg.get("min_train_matches", 500))
    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)

    if features is None:
        features = build_feature_matrix(matches, cfg.get("features", {}))
    assert_features_pre_match(matches, features)

    folds = _folds_by_season(matches, min_train=min_train)
    if not folds:
        raise RuntimeError("No valid walk-forward folds — need more historical seasons")

    cal_cfg = cfg.get("model", {}).get("calibration", {})
    holdout_seasons = int(cal_cfg.get("holdout_seasons", 1))

    pred_frames: list[pd.DataFrame] = []
    for _fold_date, train, test in folds:
        logger.info(
            "Fold season≥{} | train={} test={}",
            test["season"].iloc[0],
            len(train),
            len(test),
        )
        fit_df, calib_df = _split_fit_calib(train, holdout_seasons)
        first_test = pd.Timestamp(test["date"].min())
        feat_test = features[features["match_id"].isin(test["match_id"])]
        feat_calib = features[features["match_id"].isin(calib_df["match_id"])]
        feat_train = features[features["match_id"].isin(train["match_id"])]

        calibrator = build_calibrator(cfg)
        feat_fit = features[features["match_id"].isin(fit_df["match_id"])]
        if cal_cfg.get("method", "none") != "none" and len(calib_df) and len(fit_df) < len(train):
            # Fit strengths on pre-holdout only → true OOS calib predictions
            model_calib = build_model(cfg)
            model_calib.fit(fit_df, as_of=first_test)
            apply_totals_intercept(model_calib, fit_df, feat_fit, cfg)
            raw_calib = model_calib.predict_dataframe(calib_df, features=feat_calib)
            calibrator.fit(raw_calib, calib_df)
            # Refit on FULL train for test predictions (calibrator stays frozen)
            model = build_model(cfg)
            model.fit(train, as_of=first_test)
            apply_totals_intercept(model, train, feat_train, cfg)
        else:
            model = build_model(cfg)
            model.fit(train, as_of=first_test)
            apply_totals_intercept(model, train, feat_train, cfg)
            if cal_cfg.get("method", "none") != "none":
                # Fallback: in-sample calibration on train (still no test leakage)
                raw_calib = model.predict_dataframe(train, features=feat_train)
                calibrator.fit(raw_calib, train)

        raw_test = model.predict_dataframe(test, features=feat_test)
        fold_preds = calibrator.transform(raw_test)

        # Residual hybrid (optional): fit on expanding OOS base preds within train
        residual = build_residual_model(cfg)
        rcfg = cfg.get("model", {}).get("residual", {})
        if residual is not None:
            min_oos = int(rcfg.get("min_oos_rows", 400))
            oos = collect_oos_base_predictions(
                train,
                features,
                cfg,
                min_train=int(bt_cfg.get("min_train_matches", 500)),
                max_oos_seasons=int(rcfg.get("max_oos_seasons", 4)),
            )
            if len(oos) >= min_oos:
                # Apply same calibrator to OOS base before residual fit for consistency
                oos_cal = calibrator.transform(oos)
                feat_oos = features[features["match_id"].isin(oos_cal["match_id"])]
                X_oos, cols = _numeric_feature_matrix(
                    feat_oos,
                    oos_cal,
                    interactions=bool(getattr(residual, "interactions", False)),
                    interaction_pairs=getattr(residual, "interaction_pairs", None),
                )
                matches_oos = train[train["match_id"].isin(X_oos["match_id"])]
                residual.fit(X_oos, matches_oos, cols)
                fold_preds = residual.transform(fold_preds, feat_test)
            else:
                logger.warning(
                    "Residual skipped for fold {} — OOS rows {} < {}",
                    test["season"].iloc[0],
                    len(oos),
                    min_oos,
                )

        fold_preds["fold_season"] = test["season"].iloc[0]
        pred_frames.append(fold_preds)

    preds = pd.concat(pred_frames, ignore_index=True)
    test_matches = matches[matches["match_id"].isin(preds["match_id"])]
    bets = evaluate_predictions(preds, test_matches, bt_cfg)
    summary = _summarize(bets, preds, test_matches)

    thresholds = bt_cfg.get("edge_thresholds", [0.02, 0.03, 0.04, 0.05])
    by_threshold = _threshold_sweep(preds, test_matches, bt_cfg, list(thresholds))
    summary["by_threshold"] = by_threshold.to_dict(orient="records")

    by_season = (
        bets.groupby("season")
        .agg(
            n_bets=("match_id", "count"),
            avg_clv_prob=("clv_prob", "mean"),
            avg_clv_odds=("clv_odds", "mean"),
            roi_proxy=("profit", "sum"),
            stake=("stake", "sum"),
            hit_rate=("won", "mean"),
        )
        .reset_index()
        if len(bets)
        else pd.DataFrame()
    )
    if len(by_season):
        by_season["roi"] = by_season["roi_proxy"] / by_season["stake"].replace(0, np.nan)

    by_edge = (
        bets.groupby("edge_bucket")
        .agg(
            n_bets=("match_id", "count"),
            avg_clv_prob=("clv_prob", "mean"),
            units_profit=("profit", "sum"),
            hit_rate=("won", "mean"),
        )
        .reset_index()
        if len(bets)
        else pd.DataFrame()
    )

    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Tag experiment id with a short label if provided
    label = cfg.get("project", {}).get("experiment_label")
    if label:
        experiment_id = f"{experiment_id}_{label}"

    result = BacktestResult(
        experiment_id=experiment_id,
        config=cfg,
        bets=bets,
        predictions=preds,
        summary=summary,
        by_season=by_season,
        by_edge_bucket=by_edge,
        by_threshold=by_threshold,
    )

    if experiments_dir is not None:
        save_experiment(result, experiments_dir)

    logger.info(
        "Backtest done | bets={} ll_model={} ll_mkt={} roi={}",
        summary.get("n_bets"),
        summary.get("log_loss_1x2"),
        summary.get("log_loss_market_1x2"),
        summary.get("roi"),
    )
    return result


def save_experiment(result: BacktestResult, experiments_dir: Path) -> Path:
    out = Path(experiments_dir) / result.experiment_id
    out.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(out / "predictions.parquet", index=False)
    if len(result.bets):
        result.bets.to_parquet(out / "bets.parquet", index=False)
        result.by_season.to_csv(out / "by_season.csv", index=False)
        result.by_edge_bucket.to_csv(out / "by_edge_bucket.csv", index=False)
    if len(result.by_threshold):
        result.by_threshold.to_csv(out / "by_threshold.csv", index=False)
    # summary without nested duplication issues
    summary_out = {k: v for k, v in result.summary.items()}
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, default=str)
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(result.config, f, indent=2, default=str)
    try:
        import subprocess

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[3]),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        (out / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    logger.info("Experiment saved -> {}", out)
    return out
