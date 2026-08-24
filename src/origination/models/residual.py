"""
Additive logit residual + multi-task heads on Dixon–Coles base probabilities.

v2 (iter7, promoted)
----------
Train corrections on OOS base *errors* in logit space:

  logit_final = logit_base + α · Δ̂

where Δ̂ is predicted by LightGBM/HistGB regressors from features + base probs.

1X2: 3 centered logit deltas (multi-output regression)
OU:  1 Bernoulli logit delta

YAML::

    model.residual:
      enabled: true
      mode: additive          # additive | blend (v1, deprecated)
      backend: lightgbm
      alpha_1x2: 0.10
      alpha_ou: 0.10
      interactions: false     # iter8: pairwise λ/p products (measure before promote)
      label_smoothing: 0.02
      delta_clip: 2.0
      min_oos_rows: 300
      max_oos_seasons: 4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger

ResidualMode = Literal["additive", "blend"]

BASE_PROB_COLS = [
    "p_home",
    "p_draw",
    "p_away",
    "p_over25",
    "p_under25",
    "lambda_home",
    "lambda_away",
]

# Default pairwise interactions for residual heads (only created when both parents exist).
DEFAULT_INTERACTION_PAIRS: list[tuple[str, str]] = [
    ("lambda_home", "lambda_away"),
    ("lambda_home", "p_home"),
    ("lambda_away", "p_away"),
    ("p_home", "p_away"),
    ("p_over25", "lambda_home"),
    ("p_over25", "lambda_away"),
    # iter14: totals-oriented interactions
    ("sum_lambda", "p_over25"),
    ("sum_xg_for_ewm", "sum_xg_against_ewm"),
    ("sum_shots_for_ewm", "sum_shots_against_ewm"),
    ("sum_deep_ewm", "sum_deep_allowed_ewm"),
    ("sum_suppress_resid_ewm", "p_over25"),
    ("sum_ppda_ewm", "sum_lambda"),
    ("sum_pv_open_orth_ewm", "p_over25"),
]


def _add_interaction_features(
    X: pd.DataFrame,
    cols: list[str],
    *,
    enabled: bool = False,
    pairs: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Append product interactions; returns updated frame + column list."""
    if not enabled:
        return X, cols
    out = X.copy()
    new_cols = list(cols)
    for a, b in pairs or DEFAULT_INTERACTION_PAIRS:
        if a not in out.columns or b not in out.columns:
            continue
        name = f"ix_{a}__{b}"
        if name in out.columns:
            continue
        out[name] = out[a].astype(float) * out[b].astype(float)
        new_cols.append(name)
    return out, new_cols


def _numeric_feature_matrix(
    features: pd.DataFrame,
    preds: pd.DataFrame,
    *,
    interactions: bool = False,
    interaction_pairs: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = features.merge(preds, on="match_id", how="inner")
    skip = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "fold_season",
        "score_matrix",
        "oos_season",
    }
    cols = [
        c
        for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[["match_id"] + cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[cols] = X[cols].replace([np.inf, -np.inf], np.nan)
    med = X[cols].median(numeric_only=True)
    X[cols] = X[cols].fillna(med).fillna(0.0)
    # Derived totals signals for OU residual (always available when λ present)
    if "lambda_home" in X.columns and "lambda_away" in X.columns:
        derived = pd.DataFrame(
            {
                "sum_lambda": X["lambda_home"].astype(float) + X["lambda_away"].astype(float),
                "abs_lambda_diff": (
                    X["lambda_home"].astype(float) - X["lambda_away"].astype(float)
                ).abs(),
            },
            index=X.index,
        )
        X = pd.concat([X, derived], axis=1)
        for c in ("sum_lambda", "abs_lambda_diff"):
            if c not in cols:
                cols.append(c)
    X, cols = _add_interaction_features(
        X, cols, enabled=interactions, pairs=interaction_pairs
    )
    return X, cols

def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _additive_apply_1x2(p_base: np.ndarray, delta: np.ndarray, alpha: float) -> np.ndarray:
    logits = np.log(np.clip(p_base, 1e-8, 1.0))
    # Center predicted deltas for simplex stability
    d = delta - delta.mean(axis=1, keepdims=True)
    return _softmax(logits + float(alpha) * d)


def _additive_apply_ou(p_base: np.ndarray, delta: np.ndarray, alpha: float) -> np.ndarray:
    return _sigmoid(_logit(p_base) + float(alpha) * delta)


def _target_deltas_1x2(
    p_base: np.ndarray,
    y: np.ndarray,
    *,
    label_smoothing: float = 0.02,
    delta_clip: float = 2.0,
) -> np.ndarray:
    """Δ* = logit(y_soft) - logit(p_base), row-centered."""
    n, k = p_base.shape
    y_oh = np.zeros((n, k), dtype=float)
    y_oh[np.arange(n), y] = 1.0
    eps = float(label_smoothing)
    y_soft = (1.0 - eps) * y_oh + eps / k
    logit_t = np.log(np.clip(y_soft, 1e-8, 1.0))
    logit_b = np.log(np.clip(p_base, 1e-8, 1.0))
    d = logit_t - logit_b
    d = np.clip(d, -delta_clip, delta_clip)
    d = d - d.mean(axis=1, keepdims=True)
    return d


def _target_deltas_ou(
    p_base: np.ndarray,
    y: np.ndarray,
    *,
    label_smoothing: float = 0.02,
    delta_clip: float = 2.0,
) -> np.ndarray:
    eps = float(label_smoothing)
    y_soft = (1.0 - eps) * y.astype(float) + eps * 0.5
    d = _logit(y_soft) - _logit(p_base)
    return np.clip(d, -delta_clip, delta_clip)


@dataclass
class ResidualMultiTaskModel:
    """Additive (or legacy blend) residual heads for 1X2 + OU."""

    backend: str = "lightgbm"
    mode: ResidualMode = "additive"
    alpha_1x2: float = 0.10
    alpha_ou: float = 0.10
    label_smoothing: float = 0.02
    delta_clip: float = 2.0
    interactions: bool = False
    interaction_pairs: list[tuple[str, str]] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=list)
    ou_short_weight: float = 1.0
    ou_short_lo: float = 1.60
    ou_short_hi: float = 2.50
    ou_over_fav_weight: float = 1.0  # iter17: upweight when over is the shorter price
    _models_1x2: list[Any] = field(default_factory=list)  # 3 regressors (additive) or 1 clf
    _model_ou: Any = None
    fitted: bool = False

    def _lgb_regressor(
        self, X: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None
    ) -> Any:
        import lightgbm as lgb

        p = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": self.params.get("learning_rate", 0.05),
            "num_leaves": int(self.params.get("num_leaves", 15)),
            "min_child_samples": int(self.params.get("min_child_samples", 50)),
            "feature_fraction": float(self.params.get("feature_fraction", 0.7)),
            "bagging_fraction": float(self.params.get("bagging_fraction", 0.7)),
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": int(self.params.get("seed", 42)),
            "reg_lambda": float(self.params.get("reg_lambda", 1.0)),
        }
        dtrain = lgb.Dataset(X, label=y, weight=weight, free_raw_data=False)
        return lgb.train(p, dtrain, num_boost_round=int(self.params.get("n_estimators", 200)))

    def _hist_regressor(
        self, X: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None
    ) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor

        clf = HistGradientBoostingRegressor(
            max_depth=self.params.get("max_depth", 4),
            learning_rate=self.params.get("learning_rate", 0.05),
            max_iter=int(self.params.get("n_estimators", 200)),
            min_samples_leaf=int(self.params.get("min_child_samples", 50)),
            l2_regularization=float(self.params.get("reg_lambda", 1.0)),
            random_state=int(self.params.get("seed", 42)),
        )
        clf.fit(X, y, sample_weight=weight)
        return clf

    def _fit_reg(
        self, X: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None
    ) -> Any:
        if self.backend == "lightgbm":
            return self._lgb_regressor(X, y, weight=weight)
        return self._hist_regressor(X, y, weight=weight)

    def _predict_reg(self, model: Any, X: np.ndarray) -> np.ndarray:
        if self.backend == "lightgbm":
            return np.asarray(model.predict(X), dtype=float)
        return np.asarray(model.predict(X), dtype=float)

    # ---- legacy blend helpers (kept for ablation) ----
    def _fit_lightgbm_multiclass(self, X: np.ndarray, y: np.ndarray) -> Any:
        import lightgbm as lgb

        p = {
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": self.params.get("learning_rate", 0.05),
            "num_leaves": int(self.params.get("num_leaves", 31)),
            "min_child_samples": int(self.params.get("min_child_samples", 40)),
            "feature_fraction": float(self.params.get("feature_fraction", 0.8)),
            "bagging_fraction": float(self.params.get("bagging_fraction", 0.8)),
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": int(self.params.get("seed", 42)),
        }
        return lgb.train(
            p, lgb.Dataset(X, label=y, free_raw_data=False), num_boost_round=int(self.params.get("n_estimators", 300))
        )

    def _fit_lightgbm_binary(self, X: np.ndarray, y: np.ndarray) -> Any:
        import lightgbm as lgb

        p = {
            "objective": "binary",
            "learning_rate": self.params.get("learning_rate", 0.05),
            "num_leaves": int(self.params.get("num_leaves", 31)),
            "min_child_samples": int(self.params.get("min_child_samples", 40)),
            "feature_fraction": float(self.params.get("feature_fraction", 0.8)),
            "bagging_fraction": float(self.params.get("bagging_fraction", 0.8)),
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": int(self.params.get("seed", 42)),
        }
        return lgb.train(
            p, lgb.Dataset(X, label=y, free_raw_data=False), num_boost_round=int(self.params.get("n_estimators", 300))
        )

    def fit(
        self,
        X: pd.DataFrame,
        matches: pd.DataFrame,
        feature_cols: list[str],
        *,
        base_preds: pd.DataFrame | None = None,
    ) -> "ResidualMultiTaskModel":
        """
        Fit residual heads.

        For additive mode, ``X`` must contain base probability columns (from
        merging OOS base preds into the feature matrix) OR pass ``base_preds``.
        """
        m = matches.set_index("match_id")
        if base_preds is not None:
            X = X.merge(
                base_preds[
                    [c for c in ["match_id", "p_home", "p_draw", "p_away", "p_over25"] if c in base_preds.columns]
                ],
                on="match_id",
                how="inner",
                suffixes=("", "_base"),
            )

        rows, y_1x2, y_ou = [], [], []
        p_home, p_draw, p_away, p_over = [], [], [], []
        for mid in X["match_id"]:
            if mid not in m.index:
                continue
            row = m.loc[mid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            xr = X.loc[X["match_id"] == mid].iloc[0]
            if not all(np.isfinite(float(xr.get(c, np.nan))) for c in ("p_home", "p_draw", "p_away")):
                continue
            y_1x2.append({"H": 0, "D": 1, "A": 2}[str(row["ftr"])])
            y_ou.append(1 if float(row["total_goals"]) > 2.5 else 0)
            p_home.append(float(xr["p_home"]))
            p_draw.append(float(xr["p_draw"]))
            p_away.append(float(xr["p_away"]))
            p_over.append(float(xr.get("p_over25", 0.5)))
            rows.append(mid)

        if len(rows) < 50:
            logger.warning("Residual fit skipped — only {} rows", len(rows))
            self.fitted = False
            return self

        Xin = X.set_index("match_id").loc[rows, feature_cols].astype(float).values
        self.feature_cols = list(feature_cols)
        y1 = np.asarray(y_1x2)
        yo = np.asarray(y_ou, dtype=float)
        p_base = np.column_stack([p_home, p_draw, p_away])
        p_base = p_base / p_base.sum(axis=1, keepdims=True)
        po = np.asarray(p_over, dtype=float)

        if self.mode == "additive":
            d1 = _target_deltas_1x2(
                p_base, y1, label_smoothing=self.label_smoothing, delta_clip=self.delta_clip
            )
            do = _target_deltas_ou(
                po, yo, label_smoothing=self.label_smoothing, delta_clip=self.delta_clip
            )
            w_ou = np.ones(len(rows), dtype=float)
            need_odds = (
                float(self.ou_short_weight) > 1.0 + 1e-9
                or float(self.ou_over_fav_weight) > 1.0 + 1e-9
            )
            if need_odds:
                mm = matches.set_index("match_id")
                for i, mid in enumerate(rows):
                    if mid not in mm.index:
                        continue
                    row = mm.loc[mid]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    o, u = row.get("close_over25"), row.get("close_under25")
                    if not (pd.notna(o) and pd.notna(u)):
                        continue
                    o, u = float(o), float(u)
                    short = min(o, u)
                    w = 1.0
                    if (
                        float(self.ou_short_weight) > 1.0 + 1e-9
                        and self.ou_short_lo <= short <= self.ou_short_hi
                    ):
                        w *= float(self.ou_short_weight)
                    if float(self.ou_over_fav_weight) > 1.0 + 1e-9 and o <= u:
                        w *= float(self.ou_over_fav_weight)
                    w_ou[i] = w
            self._models_1x2 = [self._fit_reg(Xin, d1[:, k]) for k in range(3)]
            self._model_ou = self._fit_reg(Xin, do, weight=w_ou)
        else:
            # legacy blend classifiers
            if self.backend == "lightgbm":
                self._models_1x2 = [self._fit_lightgbm_multiclass(Xin, y1)]
                self._model_ou = self._fit_lightgbm_binary(Xin, yo.astype(int))
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier

                clf = HistGradientBoostingClassifier(
                    max_depth=self.params.get("max_depth", 6),
                    learning_rate=self.params.get("learning_rate", 0.05),
                    max_iter=int(self.params.get("n_estimators", 200)),
                    min_samples_leaf=int(self.params.get("min_child_samples", 40)),
                    random_state=int(self.params.get("seed", 42)),
                )
                clf.fit(Xin, y1)
                self._models_1x2 = [clf]
                clf_o = HistGradientBoostingClassifier(
                    max_depth=self.params.get("max_depth", 6),
                    learning_rate=self.params.get("learning_rate", 0.05),
                    max_iter=int(self.params.get("n_estimators", 200)),
                    min_samples_leaf=int(self.params.get("min_child_samples", 40)),
                    random_state=int(self.params.get("seed", 42)),
                )
                clf_o.fit(Xin, yo.astype(int))
                self._model_ou = clf_o

        self.fitted = True
        logger.info(
            "Residual fitted | mode={} backend={} n={} α1x2={} αou={}",
            self.mode,
            self.backend,
            len(rows),
            self.alpha_1x2,
            self.alpha_ou,
        )
        return self

    def transform(self, base_preds: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        out = base_preds.copy()
        if not self.fitted or not self._models_1x2:
            return out
        X, _ = _numeric_feature_matrix(
            features,
            base_preds,
            interactions=self.interactions,
            interaction_pairs=self.interaction_pairs,
        )
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            X = pd.concat([X, pd.DataFrame(0.0, index=X.index, columns=missing)], axis=1)
        Xin = X.set_index("match_id").loc[out["match_id"], self.feature_cols].astype(float).values
        p_base = out[["p_home", "p_draw", "p_away"]].astype(float).values
        p_base = p_base / np.clip(p_base.sum(axis=1, keepdims=True), 1e-12, None)

        if self.mode == "additive":
            delta = np.column_stack([self._predict_reg(m, Xin) for m in self._models_1x2])
            delta = np.clip(delta, -self.delta_clip, self.delta_clip)
            blended = _additive_apply_1x2(p_base, delta, self.alpha_1x2)
            out["resid_delta_l1_1x2"] = np.abs(delta).sum(axis=1) * float(self.alpha_1x2)
        else:
            m0 = self._models_1x2[0]
            if self.backend == "lightgbm":
                p_res = np.asarray(m0.predict(Xin), dtype=float)
            else:
                p_res = m0.predict_proba(Xin)
            # legacy log-linear blend
            alpha = float(np.clip(self.alpha_1x2, 0.0, 1.0))
            log_b = np.log(np.clip(p_base, 1e-8, 1.0))
            log_r = np.log(np.clip(p_res, 1e-8, 1.0))
            mixed = (1.0 - alpha) * log_b + alpha * log_r
            blended = _softmax(mixed)
            out["resid_delta_l1_1x2"] = np.abs(blended - p_base).sum(axis=1)

        out["p_home"] = blended[:, 0]
        out["p_draw"] = blended[:, 1]
        out["p_away"] = blended[:, 2]
        denom = out["p_home"] + out["p_away"]
        out["p_ah0_home"] = np.where(denom > 0, out["p_home"] / denom, np.nan)
        out["p_ah0_away"] = np.where(denom > 0, out["p_away"] / denom, np.nan)

        if self._model_ou is not None and "p_over25" in out.columns:
            po_base = out["p_over25"].astype(float).values
            if self.mode == "additive":
                d = self._predict_reg(self._model_ou, Xin)
                d = np.clip(d, -self.delta_clip, self.delta_clip)
                po = _additive_apply_ou(po_base, d, self.alpha_ou)
            else:
                if self.backend == "lightgbm":
                    po_res = np.asarray(self._model_ou.predict(Xin), dtype=float)
                else:
                    po_res = self._model_ou.predict_proba(Xin)[:, 1]
                alpha = float(np.clip(self.alpha_ou, 0.0, 1.0))
                po = _sigmoid((1 - alpha) * _logit(po_base) + alpha * _logit(po_res))
            po = np.clip(po, 1e-6, 1 - 1e-6)
            out["p_over25"] = po
            out["p_under25"] = 1.0 - po
        return out


def build_residual_model(cfg: dict[str, Any]) -> ResidualMultiTaskModel | None:
    rcfg = cfg.get("model", {}).get("residual", {})
    if not rcfg.get("enabled", False):
        return None
    mode = str(rcfg.get("mode", "additive"))
    if mode not in ("additive", "blend"):
        logger.warning("Unknown residual mode {}; using additive", mode)
        mode = "additive"
    pairs_raw = rcfg.get("interaction_pairs")
    pairs: list[tuple[str, str]] | None = None
    if pairs_raw:
        pairs = [(str(a), str(b)) for a, b in pairs_raw]
    return ResidualMultiTaskModel(
        backend=str(rcfg.get("backend", "lightgbm")),
        mode=mode,  # type: ignore[arg-type]
        alpha_1x2=float(rcfg.get("alpha_1x2", 0.10)),
        alpha_ou=float(rcfg.get("alpha_ou", 0.10)),
        label_smoothing=float(rcfg.get("label_smoothing", 0.02)),
        delta_clip=float(rcfg.get("delta_clip", 2.0)),
        interactions=bool(rcfg.get("interactions", False)),
        interaction_pairs=pairs,
        params=dict(rcfg.get("params", {})),
        ou_short_weight=float(rcfg.get("ou_short_weight", 1.0)),
        ou_short_lo=float(rcfg.get("ou_short_lo", 1.60)),
        ou_short_hi=float(rcfg.get("ou_short_hi", 2.50)),
        ou_over_fav_weight=float(rcfg.get("ou_over_fav_weight", 1.0)),
    )


def collect_oos_base_predictions(
    matches: pd.DataFrame,
    features: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    min_train: int = 500,
    max_oos_seasons: int | None = None,
) -> pd.DataFrame:
    """Expanding-season OOS base predictions inside the train window."""
    from origination.models.poisson import build_model

    seasons = sorted(matches["season"].dropna().unique())
    if max_oos_seasons is not None and max_oos_seasons > 0:
        seasons_for_oos = seasons[1:][-max_oos_seasons:]
    else:
        seasons_for_oos = seasons[1:]

    frames: list[pd.DataFrame] = []
    for season in seasons_for_oos:
        test = matches[matches["season"] == season]
        train = matches[matches["season"] < season]
        if len(train) < min_train or len(test) == 0:
            continue
        model = build_model(cfg)
        first_test = pd.Timestamp(test["date"].min())
        model.fit(train, as_of=first_test)
        from origination.models.poisson import apply_totals_intercept

        feat_tr = features[features["match_id"].isin(train["match_id"])]
        apply_totals_intercept(model, train, feat_tr, cfg)
        feat_test = features[features["match_id"].isin(test["match_id"])]
        pred = model.predict_dataframe(test, features=feat_test)
        pred["oos_season"] = season
        frames.append(pred)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
