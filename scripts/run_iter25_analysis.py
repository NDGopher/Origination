#!/usr/bin/env python
"""
Iteration 25 — Primeira e12 vs e10, safe overlays on live packs, new-league hunt.

Does NOT modify protected pack rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.backtesting.walk_forward import evaluate_predictions
from origination.data_ingestion.align import load_aligned
from origination.utils import setup_logging
from origination.utils.league_registry import get_league

OUT = ROOT / "experiments" / "iter25"
OUT.mkdir(parents=True, exist_ok=True)

KNOWN = {
    "PrimeiraLiga": "20260812T185040Z_iter22_PrimeiraLiga_base",
    "Ligue1": "20260812T182646Z_iter22_Ligue1_base",
    "Eredivisie": "20260812T184347Z_iter22_Eredivisie_base",
    "Belgium": "20260812T185650Z_iter22_Belgium_base",
    "Championship": "20260811T193046Z_iter20_Championship_shots_vol",
}


def _bt(edge: float, markets: list[str]) -> dict:
    return {
        "markets": markets,
        "edge_threshold": edge,
        "edge_threshold_by_market": {m: edge for m in markets},
        "bet_filters": {"enabled": False, "rules": []},
        "stake": {"method": "flat", "unit": 1.0, "kelly_fraction": 0.25, "max_stake": 5.0},
        "odds": {"remove_vig": "power", "min_odds": 1.2, "max_odds": 15.0},
    }


def find_preds(key: str) -> Path | None:
    named = KNOWN.get(key)
    if named:
        p = ROOT / "experiments" / named / "predictions.parquet"
        if p.is_file():
            return p
    cands = sorted(
        (ROOT / "experiments").glob(f"*{key}*/predictions.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Prefer vol06 / base / iter22
    prefer = [c for c in cands if any(x in str(c) for x in ("vol06", "iter22", "iter25", "_base"))]
    pool = prefer or cands
    return pool[0] if pool else None


def season_pos(bets: pd.DataFrame) -> tuple[int, int]:
    if bets is None or len(bets) == 0 or "season" not in bets.columns:
        return 0, 0
    pos = n = 0
    for _, g in bets.groupby("season"):
        st = float(g["stake"].sum())
        if st <= 0:
            continue
        n += 1
        if float(g["profit"].sum()) / st > 0:
            pos += 1
    return pos, n


def last3_pos(bets: pd.DataFrame) -> tuple[int, int]:
    if bets is None or len(bets) == 0 or "season" not in bets.columns:
        return 0, 0
    seasons = sorted(bets["season"].dropna().unique())[-3:]
    pos = n = 0
    for s in seasons:
        g = bets[bets["season"] == s]
        st = float(g["stake"].sum())
        if st <= 0:
            continue
        n += 1
        if float(g["profit"].sum()) / st > 0:
            pos += 1
    return pos, n


def max_dd(bets: pd.DataFrame) -> float | None:
    if bets is None or len(bets) == 0:
        return None
    b = bets.sort_values("date") if "date" in bets.columns else bets
    eq = b["profit"].astype(float).cumsum()
    return float((eq - eq.cummax()).min()) if len(eq) else None


def sig(bets: pd.DataFrame) -> dict:
    if bets is None or len(bets) < 20:
        return {"n": int(len(bets) if bets is not None else 0)}
    r = (bets["profit"] / bets["stake"].replace(0, np.nan)).astype(float).dropna()
    n = len(r)
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else np.nan
    t = mean / se if n > 1 and se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if n > 2 and np.isfinite(t) else np.nan
    spos, sn = season_pos(bets)
    l3p, l3n = last3_pos(bets)
    stake = float(bets["stake"].sum())
    return {
        "n": n,
        "roi": float(bets["profit"].sum() / stake) if stake else None,
        "hit": float((bets["won"].astype(float) >= 0.5).mean()),
        "t_stat": float(t) if np.isfinite(t) else None,
        "p_value": p if np.isfinite(p) else None,
        "units": float(bets["profit"].sum()),
        "seasons_pos": spos,
        "seasons_n": sn,
        "last3_pos": l3p,
        "last3_n": l3n,
        "max_dd_u": max_dd(bets),
        "bets_per_season": (n / sn) if sn else None,
    }


def bootstrap_roi(bets: pd.DataFrame, n_boot: int = 3000) -> dict:
    profits = bets["profit"].astype(float).values
    stakes = bets["stake"].astype(float).values
    n = len(profits)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        st = stakes[idx].sum()
        if st > 0:
            boots.append(profits[idx].sum() / st)
    boots = np.array(boots)
    return {
        "boot_ci95_lo": float(np.quantile(boots, 0.025)),
        "boot_ci95_hi": float(np.quantile(boots, 0.975)),
        "pct_boot_pos": float((boots > 0).mean()),
    }


def load_pair(key: str):
    info = get_league(key)
    aligned = ROOT / "data" / "interim" / info["aligned"]
    pred = find_preds(key)
    if pred is None or not aligned.is_file():
        return None, None, None
    return pd.read_parquet(pred), load_aligned(aligned), pred


def filter_pack(uni: pd.DataFrame, market: str, side, min_o, max_o, edge) -> pd.DataFrame:
    b = uni[(uni["market"] == market) & (uni["edge"] >= edge)]
    if side not in (None, "all", "best"):
        sides = [side] if isinstance(side, str) else list(side)
        # AH uses ah_home/ah_away or H/A depending on evaluator
        if market == "ou25":
            b = b[b["side"] == side]
        elif market == "1x2":
            b = b[b["side"].isin(sides)]
    if min_o is not None:
        b = b[b["close_odds"] >= min_o]
    if max_o is not None:
        b = b[b["close_odds"] <= max_o]
    return b.copy()


def annotate_prior(bets: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values("date")
    # prior games for home/away this season
    home_n = {}
    away_n = {}
    prior_h = []
    prior_a = []
    for _, r in m.iterrows():
        key_h = (r.get("season"), r.get("home_team"))
        key_a = (r.get("season"), r.get("away_team"))
        prior_h.append(home_n.get(key_h, 0) + away_n.get((r.get("season"), r.get("home_team")), 0))
        prior_a.append(home_n.get((r.get("season"), r.get("away_team")), 0) + away_n.get(key_a, 0))
        home_n[key_h] = home_n.get(key_h, 0) + 1
        away_n[key_a] = away_n.get(key_a, 0) + 1
    m = m.assign(_prior_home=prior_h, _prior_away=prior_a)
    keep = m[["match_id", "_prior_home", "_prior_away"]]
    out = bets.merge(keep, on="match_id", how="left")
    out["min_prior"] = out[["_prior_home", "_prior_away"]].min(axis=1)
    return out


def fmt_sig(s: dict, extra: dict | None = None) -> str:
    d = {**s, **(extra or {})}
    t = d.get("t_stat")
    t_s = f"{t:.2f}" if t is not None else "—"
    roi = d.get("roi")
    roi_s = f"{100*roi:+.1f}%" if roi is not None else "—"
    dd = d.get("max_dd_u")
    dd_s = f"{dd:+.1f}u" if dd is not None else "—"
    lo = d.get("boot_ci95_lo")
    lo_s = f"{100*lo:+.1f}%" if lo is not None else "—"
    return (
        f"n={d.get('n')} ROI={roi_s} t={t_s} seasons+={d.get('seasons_pos')}/{d.get('seasons_n')} "
        f"last3={d.get('last3_pos')}/{d.get('last3_n')} DD={dd_s} CIlo={lo_s}"
    )


def primeira_compare() -> tuple[str, pd.DataFrame]:
    preds, matches, path = load_pair("PrimeiraLiga")
    lines = ["# Primeira AH e10 vs e12", ""]
    if preds is None:
        return "Missing Primeira predictions\n", pd.DataFrame()
    lines.append(f"Preds: `{path.parent.name}`")
    uni = evaluate_predictions(preds, matches, _bt(0.03, ["ah"]), edge_threshold=0.03)
    e10 = filter_pack(uni, "ah", "all", None, 1.90, 0.10)
    e12 = filter_pack(uni, "ah", "all", None, 1.90, 0.12)
    only10 = e10[~e10["match_id"].isin(set(e12["match_id"]))].copy() if "match_id" in e10.columns else e10.iloc[0:0]

    rows = []
    for name, bets in [("e10_live", e10), ("e12_sibling", e12), ("e10_only_dropped_by_e12", only10)]:
        s = sig(bets)
        b = bootstrap_roi(bets) if len(bets) >= 40 else {}
        rows.append({"pack": name, **s, **b})
        lines.append(f"- **{name}**: {fmt_sig(s, b)}")

    # season table
    lines += ["", "## Season ROI", "", "| Season | e10 n | e10 ROI | e12 n | e12 ROI |", "|--------|------:|--------:|------:|--------:|"]
    seasons = sorted(set(list(e10.get("season", pd.Series(dtype=float)).dropna().unique()) + list(e12.get("season", pd.Series(dtype=float)).dropna().unique())))
    for ssn in seasons:
        a = e10[e10["season"] == ssn] if "season" in e10.columns else e10.iloc[0:0]
        c = e12[e12["season"] == ssn] if "season" in e12.columns else e12.iloc[0:0]
        def _roi(df):
            if len(df) == 0 or float(df["stake"].sum()) == 0:
                return "—"
            return f"{100*float(df['profit'].sum()/df['stake'].sum()):+.1f}%"
        lines.append(f"| {ssn} | {len(a)} | {_roi(a)} | {len(c)} | {_roi(c)} |")

    s10, s12 = sig(e10), sig(e12)
    lines += ["", "## Recommendation", ""]
    # Decision logic
    vol_drop = 1 - (s12.get("n", 0) / s10.get("n", 1) if s10.get("n") else 1)
    lines.append(
        f"e12 cuts volume by ~{100*vol_drop:.0f}% ({s10.get('n')} → {s12.get('n')} bets). "
        f"ROI {100*(s10.get('roi') or 0):+.1f}% → {100*(s12.get('roi') or 0):+.1f}%, "
        f"t {s10.get('t_stat'):.2f} → {s12.get('t_stat'):.2f}, "
        f"DD {s10.get('max_dd_u')} → {s12.get('max_dd_u')}, "
        f"seasons {s10.get('seasons_pos')}/{s10.get('seasons_n')} → {s12.get('seasons_pos')}/{s12.get('seasons_n')}."
    )
    lines.append("")
    so = sig(only10) if only10 is not None and len(only10) else {}
    so_roi = so.get("roi")
    if so_roi is not None and so_roi <= 0.02:
        lines.append(
            "**Promote e12 to main paper-live.** The e10-only dropped slice is flat/negative "
            f"({fmt_sig(so)}), so e10 primary mostly adds noise. Keep e10 as optional wider sibling. "
            "Do not silently change the scan pack list — wire deliberately."
        )
    else:
        lines.append(
            "**Keep e10 as live paper pack** and keep e12 as separate higher-threshold sibling "
            "(dropped slice still looks +EV)."
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "primeira_e10_vs_e12.csv", index=False)
    return "\n".join(lines) + "\n", df


PROTECTED = [
    {"id": "EPL_unders", "league": "EPL", "market": "ou25", "side": "under", "min_o": 2.00, "max_o": 4.00, "edge": 0.08},
    {"id": "EPL_overs", "league": "EPL", "market": "ou25", "side": "over", "min_o": 1.60, "max_o": 2.50, "edge": 0.10},
    {"id": "Bundesliga_unders", "league": "Bundesliga", "market": "ou25", "side": "under", "min_o": 1.70, "max_o": 2.50, "edge": 0.10},
    {"id": "LaLiga_home", "league": "LaLiga", "market": "1x2", "side": "H", "min_o": 1.01, "max_o": 1.80, "edge": 0.08},
    {"id": "SerieA_away", "league": "SerieA", "market": "1x2", "side": "A", "min_o": 1.01, "max_o": 2.00, "edge": 0.03},
    {"id": "Primeira_ah", "league": "PrimeiraLiga", "market": "ah", "side": "all", "min_o": 1.01, "max_o": 1.90, "edge": 0.10},
]


def overlays() -> str:
    lines = ["# Safe overlays on live systems (subtractive only)", ""]
    lines.append("Core rules unchanged. Overlays only *drop* bets (early-season / thin history).")
    lines.append("")
    rows = []
    for sys_ in PROTECTED:
        preds, matches, path = load_pair(sys_["league"])
        if preds is None:
            lines.append(f"- **{sys_['id']}**: missing preds — skipped")
            continue
        uni = evaluate_predictions(
            preds, matches, _bt(0.03, ["1x2", "ou25", "ah"]), edge_threshold=0.03
        )
        # evaluate_predictions uses ou25/ah/1x2
        mkt = "ou25" if sys_["market"] == "ou25" else sys_["market"]
        base = filter_pack(uni, mkt, sys_["side"], sys_["min_o"], sys_["max_o"], sys_["edge"])
        base = annotate_prior(base, matches)
        skip_early = base[base["min_prior"].fillna(0) >= 4]
        skip_thin = base[base["min_prior"].fillna(0) >= 6]
        for name, bets in [("baseline", base), ("skip_first_4_prior", skip_early), ("min_6_prior_games", skip_thin)]:
            s = sig(bets)
            b = bootstrap_roi(bets) if len(bets) >= 40 else {}
            rows.append({"system": sys_["id"], "overlay": name, **s, **b})
        sb, se = sig(base), sig(skip_early)
        improve = (se.get("t_stat") or 0) >= (sb.get("t_stat") or 0) and (se.get("roi") or 0) >= (sb.get("roi") or -9)
        flag = "improves t/ROI" if improve and se.get("n", 0) >= 80 else "does not clearly help"
        lines.append(
            f"- **{sys_['id']}** (`{path.parent.name if path else '?'}`): "
            f"base {fmt_sig(sb)} → skip-early {fmt_sig(se)} — **{flag}**"
        )
    pd.DataFrame(rows).to_csv(OUT / "overlays.csv", index=False)
    lines.append("")
    lines.append(
        "Do **not** change live packs based on these overlays unless an overlay improves "
        "t-stat and ROI without collapsing n, and is pre-registered. Default: leave live rules frozen."
    )
    return "\n".join(lines) + "\n"


def hunt_league(key: str) -> pd.DataFrame:
    preds, matches, path = load_pair(key)
    if preds is None:
        print(f"HUNT skip {key} (no preds)", flush=True)
        return pd.DataFrame()
    print(f"HUNT {key} {path.parent.name}", flush=True)
    uni = evaluate_predictions(preds, matches, _bt(0.03, ["ou25", "ah", "1x2"]), edge_threshold=0.03)
    rows = []
    ou, ah, ml = uni[uni["market"] == "ou25"], uni[uni["market"] == "ah"], uni[uni["market"] == "1x2"]
    for side, bands in [("under", [(1.7, 2.5), (1.8, 2.5), (2.0, 4.0)]), ("over", [(1.6, 2.5), (1.7, 2.8)])]:
        base = ou[ou["side"] == side]
        for lo, hi in bands:
            band = base[(base["close_odds"] >= lo) & (base["close_odds"] <= hi)]
            for e in (0.08, 0.10, 0.12):
                bets = band[band["edge"] >= e]
                s = sig(bets)
                if s.get("n", 0) < 80:
                    continue
                rows.append({"league": key, "market": "ou25", "side": side, "min_odds": lo, "max_odds": hi, "edge": e, **s, "tag": f"{key}_ou25_{side}_{lo}-{hi}_e{int(e*100)}"})
    for e in (0.08, 0.10, 0.12):
        for mx in (1.80, 1.90, 2.00):
            bets = ah[(ah["edge"] >= e) & (ah["close_odds"] <= mx)]
            s = sig(bets)
            if s.get("n", 0) < 80:
                continue
            rows.append({"league": key, "market": "ah", "side": "all", "min_odds": None, "max_odds": mx, "edge": e, **s, "tag": f"{key}_ah_e{int(e*100)}_max{mx}"})
    for e in (0.05, 0.08, 0.10):
        for mx in (1.80, 2.00, 2.20):
            for sides, stag in [(["H"], "H"), (["A"], "A")]:
                bets = ml[(ml["edge"] >= e) & (ml["close_odds"] <= mx) & (ml["side"].isin(sides))]
                s = sig(bets)
                if s.get("n", 0) < 80:
                    continue
                rows.append({"league": key, "market": "1x2", "side": stag, "min_odds": None, "max_odds": mx, "edge": e, **s, "tag": f"{key}_1x2_{stag}_e{int(e*100)}_max{mx}"})
    return pd.DataFrame(rows)


def main() -> None:
    setup_logging("ERROR")
    p1_md, _ = primeira_compare()
    ov_md = overlays()

    hunt_keys = ["Scotland", "Turkey", "Austria", "Ligue1", "Eredivisie", "Belgium", "Championship"]
    grids = [hunt_league(k) for k in hunt_keys]
    grid = pd.concat([g for g in grids if len(g)], ignore_index=True) if any(len(g) for g in grids) else pd.DataFrame()
    if len(grid):
        grid.to_csv(OUT / "full_grid.csv", index=False)
        short = grid[
            (grid["n"] >= 120)
            & (grid["roi"] >= 0.05)
            & (grid["t_stat"].fillna(0) >= 2.0)
            & (grid["seasons_n"] >= 8)
            & (grid["seasons_pos"] >= grid["seasons_n"] * 0.70)
            & (grid["last3_pos"].fillna(0) >= 2)
            & (grid["max_dd_u"].fillna(-99) > -8)
        ]
        short.to_csv(OUT / "shortlist.csv", index=False)
    else:
        short = pd.DataFrame()

    lines = [
        "# Iteration 25",
        "",
        "Protected live rules **unchanged**.",
        "",
        p1_md,
        ov_md,
        "# Expanded league hunt",
        "",
        f"Leagues scanned: {', '.join(hunt_keys)}",
        "",
        f"Cells clearing the strict bar: **{len(short)}**",
        "",
    ]
    if len(short) == 0:
        lines.append("_None. No new pack._")
    else:
        lines += ["| Tag | n | ROI | t | Seasons+ | Last3 | DD |", "|-----|--:|----:|--:|---------:|------:|---:|"]
        for _, r in short.sort_values("t_stat", ascending=False).head(15).iterrows():
            lines.append(
                f"| `{r['tag']}` | {int(r['n'])} | {100*r['roi']:+.1f}% | {r['t_stat']:.2f} | "
                f"{int(r['seasons_pos'])}/{int(r['seasons_n'])} | {int(r['last3_pos'])}/{int(r['last3_n'])} | "
                f"{r['max_dd_u']:+.1f}u |"
            )
        lines.append("")
        lines.append("New cells stay **research** until a separate paper decision. Do not auto-promote.")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT / "REPORT.md", flush=True)
    print(f"SHORTLIST={len(short)}", flush=True)


if __name__ == "__main__":
    main()
