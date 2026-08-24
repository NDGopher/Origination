"""
Independent Poisson and Dixon–Coles scoreline models.

Attack/defence strengths estimated from goals, xG, or a blend (config-driven).
Home advantage as a free parameter. Time decay optional (Dixon–Coles xi).
Optional intensity multipliers from lagged advanced metrics / context layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from scipy.stats import poisson

IntensitySource = Literal["goals", "xg", "blend"]


@dataclass
class TeamStrengths:
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.25
    rho: float = -0.05
    avg_goals: float = 1.3
    intensity_source: str = "goals"
    totals_log_offset: float = 0.0  # iter15: league-aware joint λ/μ intercept


def _tau(hg: int, ag: int, lam: float, mu: float, rho: float) -> float:
    """Dixon–Coles dependence correction for low scores."""
    if hg == 0 and ag == 0:
        return 1.0 - lam * mu * rho
    if hg == 0 and ag == 1:
        return 1.0 + lam * rho
    if hg == 1 and ag == 0:
        return 1.0 + mu * rho
    if hg == 1 and ag == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(
    lam: float,
    mu: float,
    *,
    max_goals: int = 10,
    rho: float = 0.0,
    dixon_coles: bool = False,
) -> np.ndarray:
    """P(home_goals=i, away_goals=j) matrix."""
    ys = np.arange(0, max_goals + 1)
    home_p = poisson.pmf(ys, lam)
    away_p = poisson.pmf(ys, mu)
    mat = np.outer(home_p, away_p)
    if dixon_coles and rho != 0.0:
        for i in range(min(2, max_goals + 1)):
            for j in range(min(2, max_goals + 1)):
                mat[i, j] *= _tau(i, j, lam, mu, rho)
        s = mat.sum()
        if s > 0:
            mat /= s
    return mat


def ah_settle_fraction(goal_diff: float, line: float, side: str) -> float:
    """
    Asian handicap settlement fraction in {0, 0.25, 0.5, 0.75, 1}.

    ``line`` is the home handicap (football-data AHh). Positive line means
    home receives goals. ``side`` is ``ah_home`` or ``ah_away``.
    Push → 0.5 (stake returned). Quarter lines average the two half-lines.
    """
    line = float(line)
    # Split quarter lines into two half-unit lines
    if abs(line * 4 - round(line * 4)) < 1e-9 and abs(round(line * 4)) % 2 == 1:
        lo = np.floor(line * 2) / 2.0
        hi = lo + 0.5
        return 0.5 * (
            ah_settle_fraction(goal_diff, lo, side) + ah_settle_fraction(goal_diff, hi, side)
        )

    margin = (goal_diff + line) if side == "ah_home" else (-goal_diff - line)
    if abs(margin) < 1e-9:
        return 0.5  # push
    if margin > 0:
        return 1.0
    return 0.0


def ah_probs_from_matrix(mat: np.ndarray, line: float) -> tuple[float, float]:
    """
    Model cover probabilities for home/away AH at ``line``.

    Uses expected settlement (push=0.5, half-win=0.75, etc.) as the probability
    mass for edge comparison against two-way fair odds.
    """
    n = mat.shape[0]
    p_h = 0.0
    p_a = 0.0
    for i in range(n):
        for j in range(n):
            p = float(mat[i, j])
            if p <= 0:
                continue
            gd = float(i - j)
            p_h += p * ah_settle_fraction(gd, line, "ah_home")
            p_a += p * ah_settle_fraction(gd, line, "ah_away")
    return p_h, p_a


def team_total_over_prob(lam: float, line: float, *, max_goals: int = 15) -> float:
    """P(team goals > line) under independent Poisson with mean ``lam``.

    For half-lines (0.5, 1.5, 2.5, …) this is 1 − F(floor(line)).
    """
    lam = float(max(lam, 1e-9))
    line = float(line)
    # Integer push lines: mass on exactly line is neither over nor under in Asian
    # settlement; for two-way over/under quotes we still treat over as > line.
    thresh = int(np.floor(line + 1e-12))
    # P(X <= thresh) when line is *.5 equals P(X <= floor(line))
    # Over wins when X > line ⇒ X >= thresh+1
    cdf = float(poisson.cdf(thresh, lam))
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def markets_from_matrix(mat: np.ndarray, *, ht_share: float = 0.45) -> dict[str, Any]:
    """Derive 1X2, O/U 2.5, AH helpers, and first-half markets from scoreline matrix."""
    p_home = float(np.sum(np.tril(mat, -1)))
    p_draw = float(np.trace(mat))
    p_away = float(np.sum(np.triu(mat, 1)))

    goals = np.arange(mat.shape[0])
    total = goals[:, None] + goals[None, :]
    p_over25 = float(mat[total > 2.5].sum())
    p_under25 = float(mat[total < 2.5].sum())
    p_over15 = float(mat[total > 1.5].sum())
    p_under15 = float(mat[total < 1.5].sum())
    p_over35 = float(mat[total > 3.5].sum())
    p_under35 = float(mat[total < 3.5].sum())

    denom = p_home + p_away
    p_ah0_home = p_home / denom if denom > 0 else np.nan
    p_ah0_away = p_away / denom if denom > 0 else np.nan
    p_ah_m05_home = p_home
    p_ah_m05_away = p_draw + p_away

    # First-half approximation: independent Poisson with λ_ht = ht_share * λ_ft
    # Recover lam/mu from marginal means of the FT matrix.
    i_idx = np.arange(mat.shape[0], dtype=float)
    j_idx = np.arange(mat.shape[1], dtype=float)
    lam_ft = float((mat.sum(axis=1) * i_idx).sum())
    mu_ft = float((mat.sum(axis=0) * j_idx).sum())
    ht_mat = score_matrix(
        max(lam_ft * ht_share, 1e-6),
        max(mu_ft * ht_share, 1e-6),
        max_goals=min(mat.shape[0] - 1, 6),
        rho=0.0,
        dixon_coles=False,
    )
    p_ht_home = float(np.sum(np.tril(ht_mat, -1)))
    p_ht_draw = float(np.trace(ht_mat))
    p_ht_away = float(np.sum(np.triu(ht_mat, 1)))
    ht_goals = np.arange(ht_mat.shape[0])
    ht_total = ht_goals[:, None] + ht_goals[None, :]
    p_ht_over15 = float(ht_mat[ht_total > 1.5].sum())
    p_ht_under15 = float(ht_mat[ht_total < 1.5].sum())
    p_ht_over05 = float(ht_mat[ht_total > 0.5].sum())
    p_ht_under05 = float(ht_mat[ht_total < 0.5].sum())

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over25": p_over25,
        "p_under25": p_under25,
        "p_over15": p_over15,
        "p_under15": p_under15,
        "p_over35": p_over35,
        "p_under35": p_under35,
        "p_ah0_home": p_ah0_home,
        "p_ah0_away": p_ah0_away,
        "p_ah_m05_home": p_ah_m05_home,
        "p_ah_m05_away": p_ah_m05_away,
        "p_ht_home": p_ht_home,
        "p_ht_draw": p_ht_draw,
        "p_ht_away": p_ht_away,
        "p_ht_over15": p_ht_over15,
        "p_ht_under15": p_ht_under15,
        "p_ht_over05": p_ht_over05,
        "p_ht_under05": p_ht_under05,
        "score_matrix": mat,
    }


class DixonColesModel:
    """
    Team attack/defence via (pseudo-)likelihood with optional time decay.

    intensity_source:
      - goals: classic Poisson on integer goals + DC tau on low scores
      - xG: continuous Poisson kernel  xG*log(lam) - lam  (no factorial);
            DC tau still applied using observed integer goals for dependence
      - blend: weighted mix of goals and xG kernels (blend_xg_weight in [0,1])
    """

    def __init__(
        self,
        *,
        max_goals: int = 10,
        rho_init: float = -0.05,
        xi: float = 0.0018,
        use_dc: bool = True,
        intensity_source: IntensitySource = "goals",
        blend_xg_weight: float = 0.7,
        intensity_adj_cfg: dict[str, Any] | None = None,
        hierarchical: Any | None = None,
        hierarchical_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.max_goals = max_goals
        self.rho_init = rho_init
        self.xi = xi
        self.use_dc = use_dc
        self.intensity_source: IntensitySource = intensity_source
        self.blend_xg_weight = float(blend_xg_weight)
        self.intensity_adj_cfg = intensity_adj_cfg or {}
        self._hierarchical = hierarchical
        self._hierarchical_cfg = hierarchical_cfg or {}
        self.strengths: TeamStrengths | None = None
        self.teams: list[str] = []

    def _observation_arrays(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return goals_h, goals_a, xg_h, xg_a (xg may be NaN)."""
        gh = df["home_goals"].astype(float).values
        ga = df["away_goals"].astype(float).values
        if "home_xg" in df.columns:
            xh = df["home_xg"].astype(float).values
            xa = df["away_xg"].astype(float).values
        else:
            xh = np.full_like(gh, np.nan)
            xa = np.full_like(ga, np.nan)
        return gh, ga, xh, xa

    def fit(self, train: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "DixonColesModel":
        df = train.dropna(subset=["home_goals", "away_goals"]).copy()
        if as_of is not None:
            df = df[df["date"] < as_of]

        src = self.intensity_source
        if src in ("xg", "blend"):
            if "home_xg" not in df.columns or df["home_xg"].notna().sum() < 50:
                logger.warning(
                    "intensity_source={} but insufficient xG — falling back to goals",
                    src,
                )
                src = "goals"
            else:
                df = df.dropna(subset=["home_xg", "away_xg"])

        if len(df) < 50:
            raise ValueError(f"Insufficient training matches: {len(df)}")

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams = teams
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        gh, ga, xh, xa = self._observation_arrays(df)
        home_i = df["home_team"].map(idx).values
        away_i = df["away_team"].map(idx).values
        dates = pd.to_datetime(df["date"]).values
        as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(dates.max())
        days = (as_of_ts - pd.to_datetime(dates)).days.values.astype(float)
        weights = np.exp(-self.xi * np.maximum(days, 0))

        w_xg = self.blend_xg_weight if src == "blend" else (1.0 if src == "xg" else 0.0)
        w_g = 1.0 - w_xg

        def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
            attack = x[:n].copy()
            defence = x[n : 2 * n].copy()
            attack -= attack.mean()
            defence -= defence.mean()
            home_adv = float(x[2 * n])
            rho = float(x[2 * n + 1]) if self.use_dc else 0.0
            return attack, defence, home_adv, rho

        def nll(x: np.ndarray) -> float:
            attack, defence, home_adv, rho = unpack(x)
            lam = np.clip(np.exp(attack[home_i] - defence[away_i] + home_adv), 0.05, 8.0)
            mu = np.clip(np.exp(attack[away_i] - defence[home_i]), 0.05, 8.0)

            # Continuous Poisson kernel for xG; classic PMF for goals
            ll = np.zeros(len(df))
            if w_g > 0:
                ll += w_g * (poisson.logpmf(gh, lam) + poisson.logpmf(ga, mu))
            if w_xg > 0:
                # xG * log(lam) - lam  (omit log-factorial; constant w.r.t params)
                ll += w_xg * (xh * np.log(lam) - lam + xa * np.log(mu) - mu)

            if self.use_dc:
                # Dependence correction always uses observed integer goals
                for k in range(len(gh)):
                    hgi, agi = int(gh[k]), int(ga[k])
                    if hgi <= 1 and agi <= 1:
                        t = _tau(hgi, agi, float(lam[k]), float(mu[k]), rho)
                        if t <= 0:
                            return 1e12
                        ll[k] += np.log(t)
            return float(-np.sum(weights * ll))

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.25
        x0[2 * n + 1] = self.rho_init
        res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 250})
        attack, defence, home_adv, rho = unpack(res.x)

        if src == "xg":
            avg = float((np.nanmean(xh) + np.nanmean(xa)) / 2)
        elif src == "blend":
            avg = float(
                w_xg * (np.nanmean(xh) + np.nanmean(xa)) / 2
                + w_g * (gh.mean() + ga.mean()) / 2
            )
        else:
            avg = float((gh.mean() + ga.mean()) / 2)

        self.strengths = TeamStrengths(
            attack={t: float(attack[idx[t]]) for t in teams},
            defence={t: float(defence[idx[t]]) for t in teams},
            home_adv=home_adv,
            rho=rho,
            avg_goals=avg,
            intensity_source=src,
        )
        if self._hierarchical is not None:
            atk, dfn = self._hierarchical.shrink(
                self.strengths.attack,
                self.strengths.defence,
                league=None,
                config=self._hierarchical_cfg,
            )
            self.strengths.attack = atk
            self.strengths.defence = dfn
        logger.debug(
            "Fitted DC ({}) on {} matches | home_adv={:.3f} rho={:.3f} nll={:.1f}",
            src,
            len(df),
            home_adv,
            rho,
            res.fun,
        )
        return self

    def _lambda_mu(
        self,
        home: str,
        away: str,
        *,
        lam_mult: float = 1.0,
        mu_mult: float = 1.0,
    ) -> tuple[float, float]:
        assert self.strengths is not None
        s = self.strengths
        ah = s.attack.get(home, 0.0)
        aa = s.attack.get(away, 0.0)
        dh = s.defence.get(home, 0.0)
        da = s.defence.get(away, 0.0)
        lam = float(np.exp(ah - da + s.home_adv)) * lam_mult
        mu = float(np.exp(aa - dh)) * mu_mult
        return np.clip(lam, 0.05, 8.0), np.clip(mu, 0.05, 8.0)

    def _intensity_multipliers_from_row(self, row: pd.Series | None) -> tuple[float, float]:
        """Optional multiplicative adjustments from lagged advanced / context features."""
        if row is None:
            return 1.0, 1.0

        def _get(name: str, default: float = 0.0) -> float:
            v = row.get(name, default) if hasattr(row, "get") else default
            try:
                v = float(v)
            except (TypeError, ValueError):
                return default
            return default if not np.isfinite(v) else v

        # Context-layer multipliers (e.g. referee tempo) — default 1.0
        ctx_l = _get("ctx_lam_mult_home", 1.0)
        ctx_m = _get("ctx_lam_mult_away", 1.0)
        if ctx_l == 0.0:
            ctx_l = 1.0
        if ctx_m == 0.0:
            ctx_m = 1.0

        cfg = self.intensity_adj_cfg
        if not cfg.get("enabled", False):
            return float(np.clip(ctx_l, 0.7, 1.4)), float(np.clip(ctx_m, 0.7, 1.4))

        coef_ppda = float(cfg.get("ppda_coef", 0.0))
        coef_deep = float(cfg.get("deep_coef", 0.02))
        deep_diff = _get("diff_deep_ewm", _get("home_deep_ewm", 0.0) - _get("away_deep_ewm", 0.0))
        ppda_diff = _get("diff_ppda_ewm", _get("home_ppda_ewm", 0.0) - _get("away_ppda_ewm", 0.0))
        lam_mult = float(np.exp(coef_deep * deep_diff - coef_ppda * ppda_diff)) * ctx_l
        mu_mult = float(np.exp(-coef_deep * deep_diff + coef_ppda * ppda_diff)) * ctx_m

        # Totals channel: schedule congestion / rest jointly scales λ and μ
        # (moves O/U without strongly tilting 1X2).
        cong_coef = float(cfg.get("congestion_coef", 0.0))
        rest_coef = float(cfg.get("rest_coef", 0.0))
        if cong_coef != 0.0 or rest_coef != 0.0:
            g7_h = _get("home_games_last_7", 0.0)
            g7_a = _get("away_games_last_7", 0.0)
            # Center near ~1 league game in prior 7d; higher → more congested
            cong = 0.5 * (g7_h + g7_a) - 1.0
            rest_h = _get("home_rest_days", 7.0)
            rest_a = _get("away_rest_days", 7.0)
            # Centered shortfall vs ~weekly rest (7d); positive when short-rested
            rest_short = 7.0 - 0.5 * (rest_h + rest_a)
            # Default hypothesis: congestion / short rest → fewer goals (negative coef)
            # or more chaos (positive). Sign decided by grid.
            totals_delta = cong_coef * cong + rest_coef * (rest_short / 7.0)
            joint = float(np.exp(totals_delta))
            joint = float(np.clip(joint, 0.85, 1.15))
            lam_mult *= joint
            mu_mult *= joint

        # Possession / on-ball value totals channel (lagged OBV-lite)
        pv_coef = float(cfg.get("pv_coef", 0.0))
        if pv_coef != 0.0:
            # Prefer orthogonalized resid; fall back to raw OBV
            h_pv = _get(
                "home_pv_resid_ewm",
                _get("home_pv_obv_ewm", _get("home_pv_depth_w_ewm", np.nan)),
            )
            a_pv = _get(
                "away_pv_resid_ewm",
                _get("away_pv_obv_ewm", _get("away_pv_depth_w_ewm", np.nan)),
            )
            if np.isfinite(h_pv) and np.isfinite(a_pv):
                combined = 0.5 * (h_pv + a_pv)
                # Typical resid scale smaller than raw OBV; center near 0
                center = float(cfg.get("pv_center", 0.0))
                totals_pv = pv_coef * (combined - center)
                joint_pv = float(np.clip(np.exp(totals_pv), 0.85, 1.15))
                lam_mult *= joint_pv
                mu_mult *= joint_pv

        # Shot / xG volume totals channel (both sides' attacking volume)
        # Positive coef: high combined creation → more goals (joint λ,μ lift).
        # Falls back to shots_for EWM (scaled) when xG is unavailable (e.g. Championship).
        vol_coef = float(cfg.get("shot_volume_coef", 0.0))
        if vol_coef != 0.0:
            sum_xg = _get(
                "sum_xg_for_ewm",
                _get("home_xg_for_ewm", np.nan) + _get("away_xg_for_ewm", np.nan),
            )
            if not np.isfinite(sum_xg):
                sum_shots = _get(
                    "sum_shots_for_ewm",
                    _get("home_shots_for_ewm", np.nan) + _get("away_shots_for_ewm", np.nan),
                )
                if np.isfinite(sum_shots):
                    # ~24 combined shots / scale ≈ 2.4 xG-equivalent center
                    scale = float(cfg.get("shot_volume_shots_scale", 10.0))
                    if scale > 0:
                        sum_xg = float(sum_shots) / scale
            if np.isfinite(sum_xg):
                # Center near ~2.4 combined xG-for EWM (league-ish)
                center = float(cfg.get("shot_volume_center", 2.4))
                joint_vol = float(
                    np.clip(np.exp(vol_coef * (sum_xg - center)), 0.85, 1.15)
                )
                lam_mult *= joint_vol
                mu_mult *= joint_vol

        # Defensive allowance totals channel (both sides leak xG / shots)
        # Positive coef: soft defenses → higher totals.
        allow_coef = float(cfg.get("xg_allow_coef", 0.0))
        if allow_coef != 0.0:
            sum_xa = _get(
                "sum_xg_against_ewm",
                _get("home_xg_against_ewm", np.nan) + _get("away_xg_against_ewm", np.nan),
            )
            if not np.isfinite(sum_xa):
                sum_xa = _get(
                    "sum_deep_allowed_ewm",
                    _get("home_deep_allowed_ewm", np.nan)
                    + _get("away_deep_allowed_ewm", np.nan),
                )
            if not np.isfinite(sum_xa):
                sum_sa = _get(
                    "sum_shots_against_ewm",
                    _get("home_shots_against_ewm", np.nan)
                    + _get("away_shots_against_ewm", np.nan),
                )
                if np.isfinite(sum_sa):
                    scale = float(cfg.get("xg_allow_shots_scale", 10.0))
                    if scale > 0:
                        sum_xa = float(sum_sa) / scale
            if np.isfinite(sum_xa):
                center = float(cfg.get("xg_allow_center", 2.4))
                joint_allow = float(
                    np.clip(np.exp(allow_coef * (sum_xa - center)), 0.85, 1.15)
                )
                lam_mult *= joint_allow
                mu_mult *= joint_allow

        # Tempo: combined PPDA (higher = less intense press → typically more space)
        tempo_coef = float(cfg.get("tempo_ppda_coef", 0.0))
        if tempo_coef != 0.0:
            sum_ppda = _get(
                "sum_ppda_ewm",
                _get("home_ppda_ewm", np.nan) + _get("away_ppda_ewm", np.nan),
            )
            if np.isfinite(sum_ppda):
                center = float(cfg.get("tempo_ppda_center", 24.0))
                joint_t = float(
                    np.clip(
                        np.exp(tempo_coef * (sum_ppda - center) / 10.0),
                        0.85,
                        1.15,
                    )
                )
                lam_mult *= joint_t
                mu_mult *= joint_t

        # Shot-suppression residual vs xG allowed (orthogonal to xg_allow channel)
        sup_coef = float(cfg.get("suppress_resid_coef", 0.0))
        if sup_coef != 0.0:
            resid = _get("sum_suppress_resid_ewm", np.nan)
            if np.isfinite(resid):
                joint_s = float(
                    np.clip(np.exp(sup_coef * resid / 5.0), 0.85, 1.15)
                )
                lam_mult *= joint_s
                mu_mult *= joint_s

        # Open-play PV orthogonal to xG-for (cleaner than raw OBV intensity)
        pv_o_coef = float(cfg.get("pv_open_orth_coef", 0.0))
        if pv_o_coef != 0.0:
            val = _get("sum_pv_open_orth_ewm", np.nan)
            if np.isfinite(val):
                joint_p = float(
                    np.clip(np.exp(pv_o_coef * val), 0.85, 1.15)
                )
                lam_mult *= joint_p
                mu_mult *= joint_p

        return float(np.clip(lam_mult, 0.7, 1.4)), float(np.clip(mu_mult, 0.7, 1.4))

    def calibrate_totals_intercept(
        self,
        train: pd.DataFrame,
        features: pd.DataFrame | None = None,
        *,
        shrink: float = 0.15,
        clip: float = 0.12,
        enabled: bool = True,
        mode: str = "signed",
        dampen_shrink: float = 1.0,
        min_abs_raw: float = 0.0,
    ) -> float:
        """
        Fold-safe league scoring-rate intercept.

        After DC (+ intensity) fit, compare train mean(λ+μ) to mean(goals) and
        store a joint log-offset.

        Modes (iter16):
          signed      — apply ±offset (iter15)
          lift_only   — only when goals > pred (Bundesliga-safe)
          dampen_only — only when goals < pred
          asymmetric  — full lift; dampen uses dampen_shrink (1.0 = block dampen)
        """
        if self.strengths is None:
            return 0.0
        self.strengths.totals_log_offset = 0.0
        if not enabled or train is None or len(train) < 50:
            return 0.0
        preds = self.predict_dataframe(train, features=features)
        m = train.set_index("match_id")
        pred_sum: list[float] = []
        actual: list[float] = []
        for _, r in preds.iterrows():
            mid = r["match_id"]
            if mid not in m.index:
                continue
            row = m.loc[mid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            tg = row.get("total_goals")
            if pd.isna(tg):
                try:
                    tg = float(row["home_goals"]) + float(row["away_goals"])
                except (TypeError, ValueError, KeyError):
                    continue
            pred_sum.append(float(r["lambda_home"]) + float(r["lambda_away"]))
            actual.append(float(tg))
        if len(actual) < 50:
            return 0.0
        mp = float(np.mean(pred_sum))
        ma = float(np.mean(actual))
        if mp < 0.5 or ma < 0.5:
            return 0.0
        raw = float(np.log(ma / mp))
        if abs(raw) < float(min_abs_raw):
            logger.info(
                "Totals intercept | skipped |raw|={:.4f} < min_abs_raw={:.4f} predμ={:.3f} goalsμ={:.3f}",
                abs(raw),
                min_abs_raw,
                mp,
                ma,
            )
            return 0.0

        mode_l = str(mode or "signed").lower()
        shrink = float(np.clip(shrink, 0.0, 1.0))
        d_shrink = float(np.clip(dampen_shrink, 0.0, 1.0))
        if mode_l == "lift_only":
            if raw <= 0.0:
                return 0.0
            off = (1.0 - shrink) * raw
        elif mode_l == "dampen_only":
            if raw >= 0.0:
                return 0.0
            off = (1.0 - shrink) * raw
        elif mode_l == "asymmetric":
            if raw > 0.0:
                off = (1.0 - shrink) * raw
            else:
                off = (1.0 - d_shrink) * raw
        else:
            off = (1.0 - shrink) * raw

        off = float(np.clip(off, -abs(clip), abs(clip)))
        self.strengths.totals_log_offset = off
        logger.info(
            "Totals intercept | mode={} log_offset={:.4f} raw={:.4f} predμ={:.3f} goalsμ={:.3f} n={}",
            mode_l,
            off,
            raw,
            mp,
            ma,
            len(actual),
        )
        return off

    def predict_match(
        self,
        home: str,
        away: str,
        *,
        feature_row: pd.Series | None = None,
        lam_mult: float = 1.0,
        mu_mult: float = 1.0,
    ) -> dict[str, Any]:
        adj_l, adj_m = self._intensity_multipliers_from_row(feature_row)
        lam, mu = self._lambda_mu(home, away, lam_mult=lam_mult * adj_l, mu_mult=mu_mult * adj_m)
        off = float(getattr(self.strengths, "totals_log_offset", 0.0) or 0.0) if self.strengths else 0.0
        if off != 0.0:
            scale = float(np.exp(off))
            lam = float(np.clip(lam * scale, 0.05, 8.0))
            mu = float(np.clip(mu * scale, 0.05, 8.0))
        mat = score_matrix(
            lam,
            mu,
            max_goals=self.max_goals,
            rho=self.strengths.rho if self.strengths else 0.0,
            dixon_coles=self.use_dc,
        )
        out = markets_from_matrix(mat)
        out["lambda_home"] = lam
        out["lambda_away"] = mu
        return out

    def predict_dataframe(
        self,
        matches: pd.DataFrame,
        features: pd.DataFrame | None = None,
        intensity_mults: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        feat_by_id = None
        if features is not None and len(features):
            feat_by_id = features.set_index("match_id")
        mult_by_id = None
        if intensity_mults is not None and len(intensity_mults):
            mult_by_id = intensity_mults.set_index("match_id")

        rows = []
        for _, m in matches.iterrows():
            fr = feat_by_id.loc[m["match_id"]] if feat_by_id is not None and m["match_id"] in feat_by_id.index else None
            if isinstance(fr, pd.DataFrame):
                fr = fr.iloc[0]
            lm = mm = 1.0
            if mult_by_id is not None and m["match_id"] in mult_by_id.index:
                mr = mult_by_id.loc[m["match_id"]]
                if isinstance(mr, pd.DataFrame):
                    mr = mr.iloc[0]
                lm = float(mr.get("lam_mult_home", 1.0) or 1.0)
                mm = float(mr.get("lam_mult_away", 1.0) or 1.0)
            pred = self.predict_match(
                m["home_team"], m["away_team"], feature_row=fr, lam_mult=lm, mu_mult=mm
            )
            # AH probs at match line when available
            ah_line = m.get("ah_line", np.nan)
            p_ahh = p_aha = np.nan
            if pd.notna(ah_line):
                p_ahh, p_aha = ah_probs_from_matrix(pred["score_matrix"], float(ah_line))
            rows.append(
                {
                    "match_id": m["match_id"],
                    "p_home": pred["p_home"],
                    "p_draw": pred["p_draw"],
                    "p_away": pred["p_away"],
                    "p_over25": pred["p_over25"],
                    "p_under25": pred["p_under25"],
                    "p_over15": pred["p_over15"],
                    "p_under15": pred["p_under15"],
                    "p_over35": pred["p_over35"],
                    "p_under35": pred["p_under35"],
                    "p_ah0_home": pred["p_ah0_home"],
                    "p_ah0_away": pred["p_ah0_away"],
                    "p_ah_home": p_ahh,
                    "p_ah_away": p_aha,
                    "p_ht_home": pred["p_ht_home"],
                    "p_ht_draw": pred["p_ht_draw"],
                    "p_ht_away": pred["p_ht_away"],
                    "p_ht_over15": pred["p_ht_over15"],
                    "p_ht_under15": pred["p_ht_under15"],
                    "p_ht_over05": pred["p_ht_over05"],
                    "p_ht_under05": pred["p_ht_under05"],
                    "lambda_home": pred["lambda_home"],
                    "lambda_away": pred["lambda_away"],
                }
            )
        return pd.DataFrame(rows)


class IndependentPoissonModel(DixonColesModel):
    def __init__(self, **kwargs: Any) -> None:
        kwargs["use_dc"] = False
        kwargs["rho_init"] = 0.0
        super().__init__(**kwargs)


def build_model(cfg: dict[str, Any]) -> DixonColesModel:
    from origination.features.elite import build_hierarchical_shrinker

    mcfg = cfg.get("model", {})
    mtype = mcfg.get("type", "dixon_coles")
    dc = mcfg.get("dixon_coles", {})
    intensity = dc.get("intensity_source", mcfg.get("intensity_source", "goals"))
    hier = build_hierarchical_shrinker(cfg)
    hier_cfg = dict(mcfg.get("hierarchical", {}))
    common = dict(
        max_goals=int(mcfg.get("max_goals", 10)),
        rho_init=float(dc.get("rho_init", -0.05)),
        xi=float(dc.get("xi", 0.0018)),
        intensity_source=intensity,  # type: ignore[arg-type]
        blend_xg_weight=float(dc.get("blend_xg_weight", 0.7)),
        intensity_adj_cfg=dc.get("intensity_adjustments", {}),
        hierarchical=hier,
        hierarchical_cfg=hier_cfg,
    )
    if mtype == "poisson":
        return IndependentPoissonModel(**common)
    if mtype in ("dixon_coles", "dc"):
        return DixonColesModel(use_dc=True, **common)
    logger.warning("Model type {} not fully implemented; using Dixon–Coles", mtype)
    return DixonColesModel(use_dc=True, **common)


def totals_intercept_params(
    cfg: dict[str, Any] | None,
) -> tuple[bool, float, float, str, float, float]:
    """Read totals intercept from hierarchical and/or dixon_coles YAML."""
    cfg = cfg or {}
    h = (cfg.get("model") or {}).get("hierarchical") or {}
    d = ((cfg.get("model") or {}).get("dixon_coles") or {}).get("totals_intercept") or {}
    enabled = bool(d.get("enabled", h.get("totals_intercept", False)))
    shrink = float(d.get("shrink", h.get("totals_shrink", 0.15)))
    clip = float(d.get("clip", h.get("totals_clip", 0.12)))
    mode = str(d.get("mode", h.get("totals_mode", "signed")))
    dampen_shrink = float(d.get("dampen_shrink", h.get("totals_dampen_shrink", 1.0)))
    min_abs_raw = float(d.get("min_abs_raw", h.get("totals_min_abs_raw", 0.0)))
    return enabled, shrink, clip, mode, dampen_shrink, min_abs_raw


def apply_totals_intercept(
    model: DixonColesModel,
    train: pd.DataFrame,
    features: pd.DataFrame | None,
    cfg: dict[str, Any] | None,
) -> float:
    enabled, shrink, clip, mode, dampen_shrink, min_abs_raw = totals_intercept_params(cfg)
    return model.calibrate_totals_intercept(
        train,
        features,
        shrink=shrink,
        clip=clip,
        enabled=enabled,
        mode=mode,
        dampen_shrink=dampen_shrink,
        min_abs_raw=min_abs_raw,
    )
