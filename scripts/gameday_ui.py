#!/usr/bin/env python
"""
Origination — Simple Daily Gameday UI

Clear steps:
  1) Update Data Sources (light — fixtures / optional recent results)
  2) Full Model Refresh (xG, features, strengths, context layers)
  3) Update Odds (Pinnacle)
  4) Run Scan / Pipeline (PLAY / WATCH)
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox, ttk
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
DATA_SCRIPT = ROOT / "scripts" / "refresh_gameday_data.py"
MODEL_SCRIPT = ROOT / "scripts" / "refresh_full_model.py"
ODDS_SCRIPT = ROOT / "scripts" / "refresh_gameday_odds.py"
SCAN_SCRIPT = ROOT / "scripts" / "run_daily_scan.py"
SCORE_SCRIPT = ROOT / "scripts" / "build_score_predictions.py"
STAMP = ROOT / "data" / "gameday" / "last_data_update.json"
PLAYS_SIMPLE = ROOT / "experiments" / "gameday_scan" / "PLAYS_SIMPLE.txt"
SCORE_CSV = ROOT / "experiments" / "gameday_scan" / "SCORE_PREDICTIONS.csv"
TT_TODAY = ROOT / "experiments" / "gameday_scan" / "TT_TODAY.md"
TT_LEDGER_MD = ROOT / "docs" / "TT_LEDGER.md"
PLAYS_CSV = ROOT / "experiments" / "gameday_scan" / "PLAYS_DECISION.csv"
DECISION = ROOT / "experiments" / "gameday_scan" / "DECISION_CARD.md"
DOCS = ROOT / "docs" / "DAILY_GUIDE.md"
USER_ODDS = ROOT / "data" / "gameday" / "odds.csv"
LEDGER_CSV = ROOT / "data" / "gameday" / "live_ledger.csv"
LEDGER_MD = ROOT / "docs" / "LIVE_LEDGER.md"
SETTLE_SCRIPT = ROOT / "scripts" / "update_live_ledger.py"

STALE_HOURS = 24
MODEL_STALE_HOURS = 72  # full model: warn after ~3 days

C = {
    "bg": "#0f1419",
    "surface": "#1a222c",
    "surface2": "#243040",
    "border": "#2e3d4f",
    "text": "#e8eef4",
    "muted": "#8b9aab",
    "accent": "#2dd4bf",
    "accent_dim": "#14b8a6",
    "warn": "#f59e0b",
    "danger": "#f87171",
    "ok": "#34d399",
    "run": "#38bdf8",
}


def _python() -> Path:
    return VENV_PY if VENV_PY.exists() else Path(sys.executable)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def _age_hours(dt: datetime | None) -> float | None:
    if dt is None:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


def _fmt_age(hours: float | None, when: datetime | None) -> str:
    if when is None or hours is None:
        return "never"
    local = when.astimezone().strftime("%Y-%m-%d %H:%M")
    if hours < 1:
        return f"{local}  ({int(hours * 60)}m ago)"
    if hours < 48:
        return f"{local}  ({hours:.1f}h ago)"
    return f"{local}  ({hours / 24:.1f}d ago)"


class GamedayApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Origination · LIVE")
        self.geometry("1240x1020")
        self.minsize(1040, 860)
        self.configure(bg=C["bg"])
        self._log_q: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._busy = False
        self._job_name = ""
        self._pulse_on = False

        self.var_status = tk.StringVar(value="Ready — pick a step below")
        self.var_banner = tk.StringVar(value="Idle")
        self.var_live = tk.StringVar(value="")
        self.var_data_fresh = tk.StringVar(value="Fixtures / results: —")
        self.var_odds_fresh = tk.StringVar(value="Odds: —")
        self.var_model_fresh = tk.StringVar(value="Full Model Refresh: —")
        self.var_fresh_warn = tk.StringVar(value="")
        self.var_plays_summary = tk.StringVar(value="No scan yet — run Step 4")
        self.var_scores_summary = tk.StringVar(value="No score table yet — click Refresh")
        self.var_perf_summary = tk.StringVar(value="No live plays logged yet")

        self._build_style()
        self._build()
        self.after(100, self._drain_log)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_freshness()
        self._load_plays()
        self._load_perf()
        self._load_scores()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=C["bg"], foreground=C["text"], font=("Segoe UI", 10))
        style.configure("TFrame", background=C["bg"])
        style.configure("TLabel", background=C["bg"], foreground=C["text"])
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 18))
        style.configure("Sub.TLabel", foreground=C["muted"], font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=C["surface"], foreground=C["text"])
        style.configure(
            "Primary.TButton",
            background=C["accent_dim"],
            foreground="#042f2e",
            font=("Segoe UI Semibold", 11),
            padding=(16, 11),
            borderwidth=0,
        )
        style.map(
            "Primary.TButton",
            background=[("active", C["accent"]), ("disabled", C["border"])],
            foreground=[("disabled", C["muted"])],
        )
        style.configure(
            "Step.TButton",
            background=C["surface2"],
            foreground=C["text"],
            font=("Segoe UI Semibold", 10),
            padding=(12, 9),
            borderwidth=0,
        )
        style.map(
            "Step.TButton",
            background=[("active", C["border"]), ("disabled", C["border"])],
            foreground=[("disabled", C["muted"])],
        )
        style.configure(
            "Model.TButton",
            background="#0e7490",
            foreground="#ecfeff",
            font=("Segoe UI Semibold", 10),
            padding=(12, 9),
            borderwidth=0,
        )
        style.map(
            "Model.TButton",
            background=[("active", "#0891b2"), ("disabled", C["border"])],
            foreground=[("disabled", C["muted"])],
        )
        style.configure(
            "Ghost.TButton",
            background=C["surface"],
            foreground=C["muted"],
            font=("Segoe UI", 9),
            padding=(8, 4),
            borderwidth=0,
        )
        style.map("Ghost.TButton", background=[("active", C["surface2"])])
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=C["surface2"],
            background=C["accent"],
            bordercolor=C["border"],
            thickness=14,
        )
        style.configure("TNotebook", background=C["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=C["surface2"],
            foreground=C["text"],
            padding=(16, 8),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", C["accent_dim"])],
            foreground=[("selected", "#042f2e")],
        )
        style.configure(
            "Scores.Treeview",
            background="#0c1218",
            foreground=C["text"],
            fieldbackground="#0c1218",
            rowheight=26,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Scores.Treeview.Heading",
            background=C["surface2"],
            foreground=C["text"],
            font=("Segoe UI Semibold", 9),
        )
        style.map("Scores.Treeview", background=[("selected", "#134e4a")])

    def _card(self, parent: tk.Widget) -> tuple[tk.Frame, tk.Frame]:
        outer = tk.Frame(parent, bg=C["border"])
        inner = tk.Frame(outer, bg=C["surface"])
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        return outer, inner

    def _build(self) -> None:
        head = ttk.Frame(self)
        head.pack(fill="x", padx=22, pady=(14, 4))
        ttk.Label(head, text="Origination", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            head,
            text="LIVE betting  ·  double-click START_HERE_LIVE.bat  ·  6 protected systems",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        # Live status banner (impossible to miss while running)
        ban = tk.Frame(self, bg=C["surface2"], highlightthickness=1, highlightbackground=C["border"])
        ban.pack(fill="x", padx=22, pady=(8, 4))
        self.banner_frame = ban
        self.lbl_banner = tk.Label(
            ban,
            textvariable=self.var_banner,
            bg=C["surface2"],
            fg=C["muted"],
            font=("Segoe UI Semibold", 13),
            anchor="w",
            padx=14,
            pady=8,
        )
        self.lbl_banner.pack(fill="x")
        self.lbl_live = tk.Label(
            ban,
            textvariable=self.var_live,
            bg=C["surface2"],
            fg=C["text"],
            font=("Cascadia Mono", 9),
            anchor="w",
            padx=14,
            pady=4,
            wraplength=1000,
            justify="left",
        )
        self.lbl_live.pack(fill="x", pady=(0, 8))

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=22, pady=(6, 4))
        live = tk.Frame(self.nb, bg=C["bg"])
        scores = tk.Frame(self.nb, bg=C["bg"])
        perf = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(live, text="  Daily Scan (live betting)  ")
        self.nb.add(scores, text="  Score Predictions  ")
        self.nb.add(perf, text="  System Performance  ")

        # Systems (compact)
        sys_o, sys_c = self._card(live)
        sys_o.pack(fill="x", padx=4, pady=(4, 4))
        ttk.Label(
            sys_c,
            text="Protected systems (rules never change)",
            style="Card.TLabel",
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", padx=12, pady=(8, 2))
        sys_txt = "   ·   ".join(
            [
                "EPL Under",
                "EPL Over",
                "Bundesliga Under",
                "La Liga Home",
                "Serie A Away",
                "Primeira AH e12% (live)",
            ]
        )
        tk.Label(
            sys_c, text=sys_txt, bg=C["surface"], fg=C["muted"], font=("Segoe UI", 8), anchor="w"
        ).pack(anchor="w", padx=12, pady=(0, 8))

        # Freshness timestamps
        fr_o, fr_c = self._card(live)
        fr_o.pack(fill="x", padx=4, pady=(4, 4))
        ttk.Label(
            fr_c, text="Last updated", style="Card.TLabel", font=("Segoe UI Semibold", 10)
        ).pack(anchor="w", padx=14, pady=(10, 4))
        self.lbl_data = tk.Label(
            fr_c,
            textvariable=self.var_data_fresh,
            bg=C["surface"],
            fg=C["text"],
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.lbl_data.pack(fill="x", padx=14)
        self.lbl_model = tk.Label(
            fr_c,
            textvariable=self.var_model_fresh,
            bg=C["surface"],
            fg=C["text"],
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.lbl_model.pack(fill="x", padx=14, pady=(2, 0))
        self.lbl_odds = tk.Label(
            fr_c,
            textvariable=self.var_odds_fresh,
            bg=C["surface"],
            fg=C["text"],
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.lbl_odds.pack(fill="x", padx=14, pady=(2, 0))
        self.lbl_warn = tk.Label(
            fr_c,
            textvariable=self.var_fresh_warn,
            bg=C["surface"],
            fg=C["warn"],
            font=("Segoe UI Semibold", 10),
            anchor="w",
            wraplength=1000,
            justify="left",
        )
        self.lbl_warn.pack(fill="x", padx=14, pady=(6, 10))

        # Steps
        st_o, st_c = self._card(live)
        st_o.pack(fill="x", padx=4, pady=(4, 4))
        ttk.Label(
            st_c,
            text="Daily steps (do in order)",
            style="Card.TLabel",
            font=("Segoe UI Semibold", 11),
        ).pack(anchor="w", padx=14, pady=(12, 6))

        self._action_buttons: list[ttk.Button] = []

        def _step_row(parent, btn, hint: str) -> None:
            row = tk.Frame(parent, bg=C["surface"])
            row.pack(fill="x", padx=14, pady=3)
            btn.pack(side="left")
            self._action_buttons.append(btn)
            tk.Label(
                row,
                text=hint,
                bg=C["surface"],
                fg=C["muted"],
                font=("Segoe UI", 9),
                anchor="w",
                wraplength=720,
                justify="left",
            ).pack(side="left", padx=(12, 0), fill="x", expand=True)

        self.btn_data = ttk.Button(
            st_c, text="1. Update Data Sources", style="Step.TButton", command=self._run_data
        )
        _step_row(
            st_c,
            self.btn_data,
            "Light / daily — fixtures (+ optional recent results). Does NOT rebuild the full model.",
        )

        self.btn_model = ttk.Button(
            st_c, text="2. Full Model Refresh", style="Model.TButton", command=self._run_model
        )
        _step_row(
            st_c,
            self.btn_model,
            "Heavy — xG / Understat / features / form / strengths / refs / coaching / context. Run when stale.",
        )

        self.btn_odds = ttk.Button(
            st_c, text="3. Update Odds", style="Step.TButton", command=self._run_odds
        )
        _step_row(
            st_c,
            self.btn_odds,
            "Pinnacle OU 2.5 · Moneyline · Asian Handicap. Do this close to kickoff.",
        )

        self.btn_scan = ttk.Button(
            st_c, text="4. Run Scan / Pipeline", style="Primary.TButton", command=self._run_scan
        )
        _step_row(
            st_c,
            self.btn_scan,
            "Evaluate all 6 systems → clear PLAY / WATCH list. Uses data already on disk.",
        )
        tk.Frame(st_c, bg=C["surface"], height=10).pack()

        # WHAT TO BET — large
        pl_o, pl_c = self._card(live)
        pl_o.pack(fill="both", expand=True, padx=4, pady=(4, 4))
        pt = tk.Frame(pl_c, bg=C["surface"])
        pt.pack(fill="x", padx=14, pady=(12, 4))
        ttk.Label(
            pt, text="WHAT TO BET", style="Card.TLabel", font=("Segoe UI Semibold", 14)
        ).pack(side="left")
        ttk.Button(pt, text="Open Decision Card", style="Ghost.TButton", command=self._open_decision).pack(
            side="right"
        )
        ttk.Button(pt, text="Open My Book Odds", style="Ghost.TButton", command=self._open_user_odds).pack(
            side="right", padx=(0, 8)
        )
        self.lbl_plays_sum = tk.Label(
            pl_c,
            textvariable=self.var_plays_summary,
            bg=C["surface"],
            fg=C["ok"],
            font=("Segoe UI Semibold", 11),
            anchor="w",
        )
        self.lbl_plays_sum.pack(fill="x", padx=14, pady=(0, 4))
        plays_wrap = tk.Frame(pl_c, bg=C["surface"])
        plays_wrap.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.plays_box = tk.Text(
            plays_wrap,
            height=16,
            wrap="none",
            bg="#0c1218",
            fg=C["text"],
            insertbackground=C["text"],
            relief="flat",
            font=("Cascadia Mono", 11),
            padx=14,
            pady=12,
            highlightthickness=0,
        )
        psy = ttk.Scrollbar(plays_wrap, orient="vertical", command=self.plays_box.yview)
        self.plays_box.configure(yscrollcommand=psy.set)
        self.plays_box.pack(side="left", fill="both", expand=True)
        psy.pack(side="right", fill="y")
        self.plays_box.configure(state="disabled")
        self.plays_box.tag_configure("play", foreground=C["ok"], font=("Cascadia Mono", 11, "bold"))
        self.plays_box.tag_configure("watch", foreground=C["warn"])
        self.plays_box.tag_configure("muted", foreground=C["muted"])
        self.plays_box.tag_configure("action", foreground=C["accent"], font=("Cascadia Mono", 11, "bold"))

        # Activity log (inside live tab)
        log_o, log_c = self._card(live)
        log_o.pack(fill="x", padx=0, pady=(6, 8))
        lt = tk.Frame(log_c, bg=C["surface"])
        lt.pack(fill="x", padx=14, pady=(8, 4))
        ttk.Label(lt, text="Live activity log", style="Card.TLabel", font=("Segoe UI Semibold", 10)).pack(
            side="left"
        )
        ttk.Button(lt, text="Clear log", style="Ghost.TButton", command=self._clear_log).pack(side="right")
        log_wrap = tk.Frame(log_c, bg=C["surface"])
        log_wrap.pack(fill="x", padx=12, pady=(0, 10))
        self.log = tk.Text(
            log_wrap,
            height=9,
            wrap="word",
            bg=C["surface2"],
            fg=C["text"],
            relief="flat",
            font=("Cascadia Mono", 9),
            padx=10,
            pady=8,
            highlightthickness=0,
        )
        lsy = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=lsy.set)
        self.log.pack(side="left", fill="x", expand=True)
        lsy.pack(side="right", fill="y")
        self.log.configure(state="disabled")
        self.log.tag_configure("ok", foreground=C["ok"])
        self.log.tag_configure("err", foreground=C["danger"])
        self.log.tag_configure("info", foreground=C["accent"])
        self.log.tag_configure("warn", foreground=C["warn"])
        self.log.tag_configure("muted", foreground=C["muted"])
        self.log.tag_configure("run", foreground=C["run"])

        self._build_scores_tab(scores)
        self._build_perf_tab(perf)

        # Progress + status (always visible)
        status = tk.Frame(self, bg=C["surface2"])
        status.pack(fill="x", padx=22, pady=(0, 12))
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=220)
        self.progress.pack(side="left", padx=12, pady=10)
        self.lbl_status = ttk.Label(status, textvariable=self.var_status, foreground=C["muted"])
        self.lbl_status.pack(side="left", padx=(4, 12))
        ttk.Button(status, text="Help", style="Ghost.TButton", command=self._open_docs).pack(
            side="right", padx=12
        )

    def _build_scores_tab(self, parent: tk.Widget) -> None:
        hint = tk.Label(
            parent,
            text="Information only — not the protected-system scan.  "
            "HIGH/LOW = projected total extremes  ·  why = form / xG / Elo  ·  "
            "CONFLICT = model vs Pin ≥15pp.",
            bg=C["bg"],
            fg=C["muted"],
            font=("Segoe UI", 9),
            anchor="w",
        )
        hint.pack(fill="x", pady=(8, 2))
        self.var_scores_headline = tk.StringVar(value="")
        self.lbl_scores_headline = tk.Label(
            parent,
            textvariable=self.var_scores_headline,
            bg=C["bg"],
            fg=C["accent"],
            font=("Segoe UI Semibold", 10),
            anchor="w",
            justify="left",
            wraplength=1100,
        )
        self.lbl_scores_headline.pack(fill="x", pady=(0, 6))

        bar = tk.Frame(parent, bg=C["bg"])
        bar.pack(fill="x", pady=(0, 6))
        self.btn_scores = ttk.Button(
            bar,
            text="Refresh Score Predictions",
            style="Primary.TButton",
            command=self._run_scores,
        )
        self.btn_scores.pack(side="left")
        self._action_buttons.append(self.btn_scores)
        ttk.Button(bar, text="Open TT today", command=self._open_tt_today).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="TT ledger", command=self._open_tt_ledger).pack(side="left", padx=(6, 0))
        self.lbl_scores_sum = tk.Label(
            bar,
            textvariable=self.var_scores_summary,
            bg=C["bg"],
            fg=C["ok"],
            font=("Segoe UI Semibold", 10),
            anchor="w",
        )
        self.lbl_scores_sum.pack(side="left", padx=(14, 0))

        cols = (
            "rank",
            "when",
            "kickoff",
            "league",
            "match",
            "score",
            "total",
            "profile",
            "over",
            "under",
            "lean",
            "pin",
            "vs_pin",
            "why",
            "strength",
        )
        wrap = tk.Frame(parent, bg=C["bg"])
        wrap.pack(fill="both", expand=True)
        self.score_tree = ttk.Treeview(
            wrap, columns=cols, show="headings", style="Scores.Treeview", selectmode="browse"
        )
        headings = {
            "rank": ("#", 32),
            "when": ("When", 90),
            "kickoff": ("Kickoff", 120),
            "league": ("League", 80),
            "match": ("Match", 190),
            "score": ("Proj. score", 90),
            "total": ("Total", 48),
            "profile": ("H/L", 48),
            "over": ("Over 2.5", 68),
            "under": ("Under 2.5", 72),
            "lean": ("Lean", 72),
            "pin": ("Pin O/U", 88),
            "vs_pin": ("Model−Pin", 88),
            "why": ("Why", 220),
            "strength": ("Confidence", 140),
        }
        for c, (title, w) in headings.items():
            self.score_tree.heading(c, text=title, command=lambda col=c: self._sort_scores(col))
            anchor = "w" if c in ("match", "strength", "league", "why") else "center"
            self.score_tree.column(c, width=w, minwidth=50, anchor=anchor)
        sy = ttk.Scrollbar(wrap, orient="vertical", command=self.score_tree.yview)
        self.score_tree.configure(yscrollcommand=sy.set)
        self.score_tree.pack(side="left", fill="both", expand=True)
        sy.pack(side="right", fill="y")
        self.score_tree.tag_configure("today", foreground=C["accent"])
        self.score_tree.tag_configure("next24", foreground=C["accent"])
        self.score_tree.tag_configure("over", foreground=C["ok"])
        self.score_tree.tag_configure("under", foreground=C["warn"])
        self.score_tree.tag_configure("strong", foreground=C["ok"], font=("Segoe UI Semibold", 10))
        self.score_tree.tag_configure("low", foreground=C["muted"])
        self.score_tree.tag_configure("conflict", foreground=C["warn"])
        self.score_tree.tag_configure("high", foreground=C["ok"])
        self.score_tree.tag_configure("lowtot", foreground=C["accent"])

        foot = tk.Label(
            parent,
            text="Ranked by data-quality-weighted Over/Under lean (NEXT 24H first).  "
            "HIGH / LOW = projected total extremes.  why = xG, form, Elo, rest.  "
            "CONFLICT = model vs Pin ≥15pp — those Unders missed last weekend.  "
            "Info only — not a betting system.",
            bg=C["bg"],
            fg=C["muted"],
            font=("Segoe UI", 8),
            anchor="w",
            wraplength=1100,
            justify="left",
        )
        foot.pack(fill="x", pady=(6, 4))

    def _build_perf_tab(self, parent: tk.Widget) -> None:
        hint = tk.Label(
            parent,
            text="Every PLAY the scan flags is logged here, whether or not it was bet.  "
            "Rules are frozen.  Backtest = signed walk-forward.  Live = ledger since 14 Aug 2026.",
            bg=C["bg"],
            fg=C["muted"],
            font=("Segoe UI", 9),
            anchor="w",
            wraplength=1100,
            justify="left",
        )
        hint.pack(fill="x", pady=(8, 2))
        self.lbl_perf_sum = tk.Label(
            parent,
            textvariable=self.var_perf_summary,
            bg=C["bg"],
            fg=C["accent"],
            font=("Segoe UI Semibold", 10),
            anchor="w",
        )
        self.lbl_perf_sum.pack(fill="x", pady=(0, 6))

        bar = tk.Frame(parent, bg=C["bg"])
        bar.pack(fill="x", pady=(0, 6))
        self.btn_perf = ttk.Button(
            bar, text="Refresh metrics", style="Primary.TButton", command=self._load_perf
        )
        self.btn_perf.pack(side="left")
        self.btn_settle = ttk.Button(
            bar, text="Settle finished games", style="Step.TButton", command=self._settle_ledger
        )
        self.btn_settle.pack(side="left", padx=(8, 0))
        self._action_buttons.append(self.btn_settle)
        ttk.Button(bar, text="Open ledger CSV", style="Ghost.TButton", command=self._open_ledger).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(bar, text="Open report", style="Ghost.TButton", command=self._open_ledger_md).pack(
            side="left", padx=(8, 0)
        )

        ttk.Label(
            parent,
            text="By system",
            style="Card.TLabel",
            font=("Segoe UI Semibold", 11),
        ).pack(anchor="w", pady=(4, 2))

        sys_cols = (
            "system",
            "bt_n",
            "bt_roi",
            "bt_dd",
            "live_n",
            "live_wl",
            "live_u",
            "live_roi",
            "live_dd",
            "recent",
            "open",
        )
        sys_wrap = tk.Frame(parent, bg=C["bg"])
        sys_wrap.pack(fill="x", pady=(0, 8))
        self.perf_sys = ttk.Treeview(
            sys_wrap, columns=sys_cols, show="headings", style="Scores.Treeview", height=7, selectmode="browse"
        )
        sys_head = {
            "system": ("System", 170),
            "bt_n": ("BT n", 55),
            "bt_roi": ("BT ROI", 70),
            "bt_dd": ("BT DD", 70),
            "live_n": ("Live n", 60),
            "live_wl": ("Live W-L", 75),
            "live_u": ("Live u", 70),
            "live_roi": ("Live ROI", 75),
            "live_dd": ("Live DD", 70),
            "recent": ("Recent form", 140),
            "open": ("Open", 50),
        }
        for c, (title, w) in sys_head.items():
            self.perf_sys.heading(c, text=title)
            self.perf_sys.column(c, width=w, minwidth=40, anchor="w" if c in ("system", "recent") else "center")
        psy = ttk.Scrollbar(sys_wrap, orient="vertical", command=self.perf_sys.yview)
        self.perf_sys.configure(yscrollcommand=psy.set)
        self.perf_sys.pack(side="left", fill="x", expand=True)
        psy.pack(side="right", fill="y")
        self.perf_sys.tag_configure("pos", foreground=C["ok"])
        self.perf_sys.tag_configure("neg", foreground=C["danger"])
        self.perf_sys.tag_configure("muted", foreground=C["muted"])

        ttk.Label(
            parent,
            text="Flagged plays (all systems)",
            style="Card.TLabel",
            font=("Segoe UI Semibold", 11),
        ).pack(anchor="w", pady=(4, 2))

        play_cols = ("date", "system", "match", "side", "odds", "edge", "result", "score", "pl", "status")
        play_wrap = tk.Frame(parent, bg=C["bg"])
        play_wrap.pack(fill="both", expand=True)
        self.perf_plays = ttk.Treeview(
            play_wrap, columns=play_cols, show="headings", style="Scores.Treeview", selectmode="browse"
        )
        play_head = {
            "date": ("Date", 90),
            "system": ("System", 160),
            "match": ("Match", 220),
            "side": ("Side", 80),
            "odds": ("Pin odds", 80),
            "edge": ("Edge", 70),
            "result": ("Result", 60),
            "score": ("Score", 70),
            "pl": ("P/L u", 70),
            "status": ("Status", 80),
        }
        for c, (title, w) in play_head.items():
            self.perf_plays.heading(c, text=title)
            self.perf_plays.column(c, width=w, minwidth=50, anchor="w" if c in ("system", "match") else "center")
        ppy = ttk.Scrollbar(play_wrap, orient="vertical", command=self.perf_plays.yview)
        self.perf_plays.configure(yscrollcommand=ppy.set)
        self.perf_plays.pack(side="left", fill="both", expand=True)
        ppy.pack(side="right", fill="y")
        self.perf_plays.tag_configure("W", foreground=C["ok"])
        self.perf_plays.tag_configure("L", foreground=C["danger"])
        self.perf_plays.tag_configure("open", foreground=C["accent"])
        self.perf_plays.tag_configure("push", foreground=C["warn"])

        foot = tk.Label(
            parent,
            text="BT = full walk-forward backtest (promotion evidence).  "
            "Live = paper/production flags since go-live.  "
            "Recent form = last 10 settled (W/L/P).  Small live n is not a reason to change rules.",
            bg=C["bg"],
            fg=C["muted"],
            font=("Segoe UI", 8),
            anchor="w",
            wraplength=1100,
            justify="left",
        )
        foot.pack(fill="x", pady=(6, 4))

    # ----- freshness -----
    def _refresh_freshness(self) -> None:
        data_when = odds_when = model_when = None
        model_ok = True
        if STAMP.is_file():
            try:
                raw = json.loads(STAMP.read_text(encoding="utf-8"))
                data_when = _parse_iso(
                    (raw.get("data") or {}).get("updated_at") or raw.get("updated_at_data")
                )
                odds_when = _parse_iso(
                    (raw.get("odds") or {}).get("updated_at") or raw.get("updated_at_odds")
                )
                model_when = _parse_iso(
                    (raw.get("model") or {}).get("updated_at") or raw.get("updated_at_model")
                )
                if raw.get("model") and raw["model"].get("ok") is False:
                    model_ok = False
            except Exception:  # noqa: BLE001
                pass
        if odds_when is None:
            meta = ROOT / "data" / "interim" / "pinnacle_ou25_EPL.meta.json"
            if meta.is_file():
                try:
                    m = json.loads(meta.read_text(encoding="utf-8"))
                    odds_when = _parse_iso(m.get("fetched_at"))
                except Exception:  # noqa: BLE001
                    pass
        if data_when is None:
            fx = ROOT / "data" / "interim" / "fixtures_upcoming_EPL.meta.json"
            if fx.is_file():
                try:
                    m = json.loads(fx.read_text(encoding="utf-8"))
                    data_when = _parse_iso(m.get("fetched_at"))
                except Exception:  # noqa: BLE001
                    pass

        dh = _age_hours(data_when)
        oh = _age_hours(odds_when)
        mh = _age_hours(model_when)
        self.var_data_fresh.set(f"Fixtures / results:     {_fmt_age(dh, data_when)}")
        self.var_model_fresh.set(f"Full Model Refresh:     {_fmt_age(mh, model_when)}")
        self.var_odds_fresh.set(f"Odds (Pinnacle):        {_fmt_age(oh, odds_when)}")

        warns: list[str] = []
        if dh is None:
            warns.append("Fixtures never updated — click Step 1 (or Step 2).")
        elif dh > STALE_HOURS:
            warns.append(f"Fixtures/results are {dh:.0f}h old — click Step 1.")
        if mh is None:
            warns.append("Full Model Refresh never run — click Step 2 before trusting scans.")
        elif not model_ok:
            warns.append("Last Full Model Refresh FAILED — re-run Step 2.")
        elif mh > MODEL_STALE_HOURS:
            warns.append(
                f"Full model is {mh / 24:.1f}d old — click Step 2 (xG/features/context may be stale)."
            )
        if oh is None:
            warns.append("Odds never updated — click Step 3.")
        elif oh > STALE_HOURS:
            warns.append(f"Odds are {oh:.0f}h old — click Step 3 before betting.")
        elif oh > 6:
            warns.append(f"Odds are {oh:.1f}h old — consider Step 3 again near kickoff.")

        if warns:
            self.var_fresh_warn.set("⚠ " + "  |  ".join(warns))
            self.lbl_warn.configure(fg=C["warn"])
        else:
            self.var_fresh_warn.set("✓ Fixtures, model, and odds look fresh enough to scan.")
            self.lbl_warn.configure(fg=C["ok"])

    # ----- logging / process -----
    def _append_log(self, text: str, tag: str | None = None) -> None:
        self.log.configure(state="normal")
        if tag:
            self.log.insert("end", text, tag)
        else:
            self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _drain_log(self) -> None:
        try:
            while True:
                text, tag = self._log_q.get_nowait()
                self._append_log(text, tag)
                # Update live line from meaningful progress lines
                line = text.strip()
                if line and not line.startswith("="):
                    if len(line) > 140:
                        line = line[:137] + "…"
                    self.var_live.set(line)
        except queue.Empty:
            pass
        self.after(80, self._drain_log)

    def _set_banner(self, mode: str, title: str, detail: str = "") -> None:
        self.var_banner.set(title)
        self.var_live.set(detail)
        colors = {
            "idle": (C["surface2"], C["muted"]),
            "run": ("#0c4a6e", C["run"]),
            "ok": ("#064e3b", C["ok"]),
            "err": ("#7f1d1d", C["danger"]),
        }
        bg, fg = colors.get(mode, colors["idle"])
        self.banner_frame.configure(bg=bg, highlightbackground=bg)
        self.lbl_banner.configure(bg=bg, fg=fg)
        self.lbl_live.configure(bg=bg, fg=C["text"] if mode != "idle" else C["muted"])

    def _pulse(self) -> None:
        if not self._busy:
            self._pulse_on = False
            return
        # Keep banner attention while running
        self._pulse_on = not self._pulse_on
        dots = "." * (1 + (int(datetime.now().timestamp()) % 3))
        self.var_banner.set(f"▶  RUNNING: {self._job_name}{dots}   (please wait — do not click again)")
        self.after(500, self._pulse)

    def _set_busy(self, busy: bool, msg: str = "Ready — pick a step below") -> None:
        self._busy = busy
        self.var_status.set(msg)
        state = "disabled" if busy else "normal"
        for b in self._action_buttons:
            b.configure(state=state)
        if busy:
            self.progress.start(10)
            self.after(0, self._pulse)
        else:
            self.progress.stop()

    def _exec(self, cmd: list[str], job_name: str, on_done=None) -> None:
        code = -1
        try:
            self._log_q.put((f"\n── {job_name} started ──\n", "run"))
            self._log_q.put((f"$ {' '.join(cmd)}\n", "muted"))
            env = os.environ.copy()
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("PYTHONUNBUFFERED", "1")
            creation = 0
            if sys.platform == "win32":
                creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creation,
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                tag = "muted"
                low = line.lower()
                if "error" in low or "failed" in low or "traceback" in low:
                    tag = "err"
                elif "success" in low or "✓" in line or "complete" in low:
                    tag = "ok"
                elif line.startswith("===") or line.startswith("──") or "[1/" in line or "[2/" in line:
                    tag = "info"
                elif "warn" in low:
                    tag = "warn"
                self._log_q.put((line, tag))
            code = self._proc.wait()
            if code == 0:
                self._log_q.put((f"\n✓ SUCCESS — {job_name} finished OK\n", "ok"))
            else:
                self._log_q.put((f"\n✗ FAILED — {job_name} exited with code {code}\n", "err"))
        except Exception as exc:  # noqa: BLE001
            self._log_q.put((f"\n✗ FAILED — {exc}\n", "err"))
            code = -1
        finally:
            self._proc = None

            def _finish() -> None:
                self._set_busy(False)
                self._refresh_freshness()
                if code == 0:
                    self._set_banner("ok", f"✓ SUCCESS — {job_name}", "Finished cleanly. See activity log for details.")
                    self.var_status.set(f"Done: {job_name}")
                else:
                    self._set_banner(
                        "err",
                        f"✗ FAILED — {job_name}",
                        "See the activity log below for the error. Fix and try again.",
                    )
                    self.var_status.set(f"Failed: {job_name}")
                    messagebox.showerror(
                        "Job failed",
                        f"{job_name} failed (exit {code}).\n\nScroll the Live activity log for details.",
                    )
                if on_done:
                    on_done(code == 0)

            self.after(0, _finish)

    def _start_job(self, job_name: str, cmd: list[str], on_done=None) -> None:
        if self._busy:
            messagebox.showinfo(
                "Already running",
                f"Still running: {self._job_name}\n\nWait for it to finish.",
            )
            return
        self._job_name = job_name
        self._set_banner("run", f"▶  RUNNING: {job_name}", "Starting… live log below will update.")
        self._set_busy(True, f"Running: {job_name}…")
        threading.Thread(target=self._exec, args=(cmd, job_name, on_done), daemon=True).start()

    def _run_data(self) -> None:
        if self._busy:
            messagebox.showinfo("Already running", f"Still running: {self._job_name}")
            return
        if not DATA_SCRIPT.exists():
            messagebox.showerror("Missing", str(DATA_SCRIPT))
            return
        fixtures_only = messagebox.askyesno(
            "Update Data Sources",
            "Light update — fixtures for all live leagues?\n\n"
            "YES = fixtures only (fast, recommended most days)\n"
            "NO  = also refresh recent results/xG from cache\n"
            "      (still skips re-downloading old seasons)\n\n"
            "For a full xG + feature rebuild, use Step 2 instead.",
        )
        cmd = [str(_python()), str(DATA_SCRIPT)]
        if fixtures_only:
            cmd.append("--fixtures-only")
            name = "Update Data Sources (fixtures)"
        else:
            cmd.append("--with-results")
            name = "Update Data Sources (fixtures + results)"
        self._start_job(name, cmd)

    def _run_model(self) -> None:
        if self._busy:
            messagebox.showinfo("Already running", f"Still running: {self._job_name}")
            return
        if not MODEL_SCRIPT.exists():
            messagebox.showerror("Missing", str(MODEL_SCRIPT))
            return
        if not messagebox.askokcancel(
            "Full Model Refresh",
            "This rebuilds EVERYTHING the model needs:\n\n"
            "• Current-season results + Understat xG / advanced stats\n"
            "• Aligned match tables\n"
            "• Feature store (form, Elo, strengths)\n"
            "• Context layers (refs, coaching, motivation, …)\n"
            "• Upcoming fixtures\n\n"
            "Old seasons stay cached (not re-downloaded).\n"
            "Odds are NOT updated — use Step 3 after this.\n\n"
            "This can take several minutes. Continue?",
        ):
            return
        self._start_job(
            "Full Model Refresh",
            [str(_python()), str(MODEL_SCRIPT), "--log-level", "INFO"],
        )

    def _run_odds(self) -> None:
        if self._busy:
            messagebox.showinfo("Already running", f"Still running: {self._job_name}")
            return
        if not ODDS_SCRIPT.exists():
            messagebox.showerror("Missing", str(ODDS_SCRIPT))
            return
        self._start_job("Update Odds (Pinnacle)", [str(_python()), str(ODDS_SCRIPT)])

    def _run_scan(self) -> None:
        if self._busy:
            messagebox.showinfo("Already running", f"Still running: {self._job_name}")
            return
        if not SCAN_SCRIPT.exists():
            messagebox.showerror("Missing", str(SCAN_SCRIPT))
            return
        warn = self.var_fresh_warn.get()
        if "never" in self.var_data_fresh.get() or "never" in self.var_odds_fresh.get():
            if not messagebox.askokcancel(
                "Missing updates",
                "Fixtures or odds look missing.\n\n"
                "Recommended: Step 1 → Step 3 (and Step 2 if model is stale),\n"
                "then Scan.\n\n"
                "Continue with whatever is on disk anyway?",
            ):
                return
        elif "⚠" in warn:
            if not messagebox.askokcancel("Stale data", warn + "\n\nContinue scan anyway?"):
                return

        def _done(ok: bool) -> None:
            self._load_plays()
            self._load_scores()
            self._load_perf()
            if ok:
                # Strengthen success banner with play count
                summary = self.var_plays_summary.get()
                self._set_banner("ok", "✓ SUCCESS — Scan complete", summary)

        self._start_job(
            "Run Scan / Pipeline",
            [str(_python()), str(SCAN_SCRIPT), "--no-refresh"],
            on_done=_done,
        )

    def _run_scores(self) -> None:
        if self._busy:
            messagebox.showinfo("Already running", f"Still running: {self._job_name}")
            return
        if not SCORE_SCRIPT.exists():
            messagebox.showerror("Missing", str(SCORE_SCRIPT))
            return
        rebuild = messagebox.askyesno(
            "Score Predictions",
            "Refresh the score table (next 24h + through tomorrow + later slate)?\n\n"
            "YES = refresh fixtures + Pinnacle odds + rebuild model sheets (slower, recommended)\n"
            "NO  = use files already on disk (fast)\n\n"
            "This does NOT place bets. Use Daily Scan for PLAY / WATCH.",
        )

        def _done(ok: bool) -> None:
            self._load_scores()
            if ok:
                self._set_banner("ok", "✓ SUCCESS — Score Predictions", self.var_scores_summary.get())

        cmd = [str(_python()), str(SCORE_SCRIPT), "--hours", "24", "--later-hours", "168"]
        if rebuild:
            cmd.append("--rebuild")
        else:
            cmd.append("--refresh-fixtures")
        self._start_job("Score Predictions", cmd, on_done=_done)

    def _load_scores(self) -> None:
        tree = getattr(self, "score_tree", None)
        if tree is None:
            return
        for iid in tree.get_children():
            tree.delete(iid)
        if not SCORE_CSV.is_file():
            self.var_scores_summary.set("No score table yet — click Refresh Score Predictions.")
            self.lbl_scores_sum.configure(fg=C["muted"])
            return
        try:
            import csv

            with SCORE_CSV.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception as exc:  # noqa: BLE001
            self.var_scores_summary.set(f"Could not read scores: {exc}")
            return
        n24 = sum(1 for r in rows if r.get("when") == "NEXT 24H" or str(r.get("in_next_24h")).lower() == "true")
        n_focus = sum(
            1
            for r in rows
            if r.get("when") in ("NEXT 24H", "THROUGH TOM.")
            or str(r.get("in_focus")).lower() == "true"
        )
        n_strong = sum(1 for r in rows if str(r.get("strong")).lower() in ("true", "1"))
        n_proj = sum(1 for r in rows if r.get("proj_score") and r.get("proj_score") != "—")
        n_pin = sum(1 for r in rows if str(r.get("has_pin_ou")).lower() in ("true", "1"))
        n_conflict = sum(1 for r in rows if str(r.get("pin_conflict")).lower() in ("true", "1"))
        n_high = sum(1 for r in rows if (r.get("score_profile") or "").upper() == "HIGH")
        n_low = sum(1 for r in rows if (r.get("score_profile") or "").upper() == "LOW")
        self.var_scores_summary.set(
            f"{n24} in next 24h  ·  {n_focus} through tomorrow  ·  {len(rows)} listed  ·  "
            f"{n_high} HIGH tot  ·  {n_low} LOW tot  ·  {n_conflict} Pin conflicts  ·  "
            f"{n_pin} with Pin OU"
        )
        self.lbl_scores_sum.configure(fg=C["ok"] if n_proj else C["muted"])

        # Headline: strongest Over / Under within focus window when possible
        focus_rows = [
            r
            for r in rows
            if r.get("when") in ("NEXT 24H", "THROUGH TOM.")
            or str(r.get("in_focus")).lower() == "true"
        ] or rows
        overs = [
            r
            for r in focus_rows
            if (r.get("lean") or "").upper() == "OVER" and r.get("over_pct") not in ("", None, "None")
        ]
        unders = [
            r
            for r in focus_rows
            if (r.get("lean") or "").upper() == "UNDER" and r.get("under_pct") not in ("", None, "None")
        ]

        def _pct(r, k):
            try:
                return float(r.get(k))
            except (TypeError, ValueError):
                return -1.0

        overs.sort(key=lambda r: _pct(r, "over_pct"), reverse=True)
        unders.sort(key=lambda r: _pct(r, "under_pct"), reverse=True)
        by_tot = [r for r in focus_rows if r.get("proj_total") not in ("", None, "None")]
        by_tot.sort(key=lambda r: _pct(r, "proj_total"), reverse=True)
        bits = []
        if by_tot:
            hi, lo = by_tot[0], by_tot[-1]
            bits.append(f"HIGHEST TOT  {hi.get('proj_total')}  {hi.get('match')} ({hi.get('league')})")
            bits.append(f"LOWEST TOT  {lo.get('proj_total')}  {lo.get('match')} ({lo.get('league')})")
        if overs:
            o = overs[0]
            bits.append(
                f"STRONGEST OVER  {o.get('over_pct')}%  {o.get('match')} ({o.get('league')})"
            )
        if unders:
            u = unders[0]
            bits.append(
                f"STRONGEST UNDER  {u.get('under_pct')}%  {u.get('match')} ({u.get('league')})"
            )
        if not bits:
            bits.append("No projected leans yet — refresh with rebuild after fixtures exist.")
        self.var_scores_headline.set("   ·   ".join(bits))

        for r in rows:
            lean = (r.get("lean") or "").upper()
            strength = r.get("data_strength") or ""
            note = r.get("data_note") or ""
            strong = str(r.get("strong")).lower() in ("true", "1")
            tags = []
            if r.get("when") in ("NEXT 24H", "THROUGH TOM."):
                tags.append("next24")
            if strong:
                tags.append("strong")
            elif lean == "OVER":
                tags.append("over")
            elif lean == "UNDER":
                tags.append("under")
            if strength == "LOW":
                tags.append("low")
            if str(r.get("pin_conflict")).lower() in ("true", "1"):
                tags.append("conflict")
            if (r.get("score_profile") or "").upper() == "HIGH":
                tags.append("high")
            elif (r.get("score_profile") or "").upper() == "LOW":
                tags.append("lowtot")
            over = r.get("over_pct") or ""
            under = r.get("under_pct") or ""
            if over not in ("", "None"):
                try:
                    over = f"{float(over):.0f}%"
                except ValueError:
                    pass
            if under not in ("", "None"):
                try:
                    under = f"{float(under):.0f}%"
                except ValueError:
                    pass
            lean_s = lean
            if strong and r.get("lean_pp"):
                try:
                    lean_s = f"{lean} {float(r.get('lean_pp')):.0f}pp"
                except (TypeError, ValueError):
                    pass
            pin_s = "—"
            try:
                po, pu = r.get("pin_over25"), r.get("pin_under25")
                if po not in ("", None, "None") and pu not in ("", None, "None"):
                    pin_s = f"{float(po):.2f}/{float(pu):.2f}"
            except (TypeError, ValueError):
                pin_s = "—"
            vs_pin = "—"
            try:
                if lean == "OVER" and r.get("model_minus_pin_over_pp") not in ("", None, "None"):
                    vs_pin = f"{float(r.get('model_minus_pin_over_pp')):+.1f}pp"
                elif lean == "UNDER" and r.get("model_minus_pin_under_pp") not in ("", None, "None"):
                    vs_pin = f"{float(r.get('model_minus_pin_under_pp')):+.1f}pp"
            except (TypeError, ValueError):
                vs_pin = "—"
            if str(r.get("pin_conflict")).lower() in ("true", "1") and vs_pin != "—":
                vs_pin = f"CONFLICT {vs_pin}"
            conf = r.get("confidence") or ""
            strength_s = conf or f"{strength}  {note}".strip()
            why_s = r.get("why") or note or "—"
            pin_lean = r.get("pin_lean") or ""
            if pin_lean and vs_pin != "—" and "CONFLICT" not in vs_pin:
                vs_pin = f"{vs_pin} (Pin {pin_lean})"
            tree.insert(
                "",
                "end",
                values=(
                    r.get("rank", ""),
                    r.get("when", ""),
                    r.get("kickoff_local") or r.get("date", ""),
                    r.get("league", ""),
                    r.get("match", ""),
                    r.get("proj_score", "—"),
                    r.get("proj_total") or "—",
                    r.get("score_profile") or "—",
                    over or "—",
                    under or "—",
                    lean_s,
                    pin_s,
                    vs_pin,
                    why_s,
                    strength_s,
                ),
                tags=tuple(tags),
            )

    def _sort_scores(self, col: str) -> None:
        tree = self.score_tree
        data = [(tree.set(k, col), k) for k in tree.get_children("")]
        numeric_cols = {"total", "over", "under", "rank"}
        if col in numeric_cols:

            def _key(item):
                s = str(item[0]).replace("%", "").replace("—", "").strip()
                try:
                    return float(s)
                except ValueError:
                    return -1.0

            data.sort(key=_key, reverse=True)
        else:
            data.sort(key=lambda x: str(x[0]).lower(), reverse=(col in ("lean", "when")))
        for i, (_, k) in enumerate(data):
            tree.move(k, "", i)

    def _load_plays(self) -> None:
        self.plays_box.configure(state="normal")
        self.plays_box.delete("1.0", "end")
        if PLAYS_SIMPLE.is_file():
            text = PLAYS_SIMPLE.read_text(encoding="utf-8")
            n_play = text.count(">>> PLAY")
            n_watch = len(re.findall(r"WATCH ·", text))
            if n_play:
                self.var_plays_summary.set(
                    f"{n_play} PLAY(s) — place these bets.    {n_watch} WATCH (usually skip)."
                )
                self.lbl_plays_sum.configure(fg=C["ok"])
            else:
                self.var_plays_summary.set("0 PLAYs — nothing to bet right now.")
                self.lbl_plays_sum.configure(fg=C["muted"])
            for line in text.splitlines(True):
                tag = "muted"
                if ">>> PLAY" in line or "BET:" in line:
                    tag = "play"
                elif "ACTION:" in line:
                    tag = "action"
                elif "WATCH" in line:
                    tag = "watch"
                self.plays_box.insert("end", line, tag)
        elif PLAYS_CSV.is_file():
            self.var_plays_summary.set("Scan file found — open Decision Card for details.")
            self.plays_box.insert("end", f"See {DECISION}\n", "muted")
        else:
            self.var_plays_summary.set("No scan yet — finish Steps 1–3, then click Step 4.")
            self.lbl_plays_sum.configure(fg=C["muted"])
            self.plays_box.insert(
                "end",
                "No scan output yet.\n\n"
                "1) Update Data Sources (light)\n"
                "2) Full Model Refresh (when model is stale)\n"
                "3) Update Odds\n"
                "4) Run Scan / Pipeline\n",
                "muted",
            )
        self.plays_box.configure(state="disabled")

    def _pct(self, x) -> str:
        if x is None:
            return "—"
        try:
            return f"{100 * float(x):+.1f}%"
        except (TypeError, ValueError):
            return "—"

    def _load_perf(self) -> None:
        try:
            from origination.gameday.live_ledger import performance_snapshot, write_report

            snap = performance_snapshot()
            write_report()
        except Exception as exc:  # noqa: BLE001
            self.var_perf_summary.set(f"Could not load ledger: {exc}")
            return
        self.var_perf_summary.set(
            f"Logged {snap['n_total']} flag(s)  ·  {snap['n_settled']} settled  ·  "
            f"{snap['n_open']} open  ·  since {snap['ledger_start']}"
        )
        for tree in (self.perf_sys, self.perf_plays):
            for iid in tree.get_children(""):
                tree.delete(iid)
        for s in snap["systems"]:
            lv = s["live"]
            bt = s["backtest"]
            rec = s["recent"]
            wl = f"{lv['wins']}-{lv['n_decided'] - lv['wins']}" if lv["n_decided"] else "—"
            dd = "—" if lv.get("max_dd_u") is None else f"{lv['max_dd_u']:+.1f}u"
            bt_dd = "—" if bt.get("max_dd_u") is None else f"{bt['max_dd_u']:+.1f}u"
            tag = "muted"
            if lv["n_decided"]:
                tag = "pos" if (lv.get("units") or 0) >= 0 else "neg"
            self.perf_sys.insert(
                "",
                "end",
                values=(
                    s["system"],
                    bt.get("n") or 0,
                    self._pct(bt.get("roi")),
                    bt_dd,
                    lv["n"],
                    wl,
                    f"{lv['units']:+.2f}u",
                    self._pct(lv.get("roi")),
                    dd,
                    rec.get("form") or "—",
                    lv["n_open"],
                ),
                tags=(tag,),
            )
        for p in reversed(snap.get("plays") or []):
            edge = p.get("edge_vs_pin")
            edge_s = "—" if edge is None else f"{100 * float(edge):+.1f}%"
            odds = p.get("pin_odds")
            odds_s = "—" if odds is None else f"{float(odds):.3f}"
            pl = p.get("profit_u")
            pl_s = "—" if pl is None else f"{float(pl):+.3f}"
            result = p.get("result") or ""
            tag = result if result in ("W", "L", "open", "push") else ""
            self.perf_plays.insert(
                "",
                "end",
                values=(
                    p.get("date") or "",
                    p.get("system") or "",
                    p.get("match") or "",
                    p.get("side") or "",
                    odds_s,
                    edge_s,
                    result,
                    p.get("actual") or "—",
                    pl_s,
                    p.get("status") or "",
                ),
                tags=(tag,) if tag else (),
            )

    def _settle_ledger(self) -> None:
        if self._busy:
            messagebox.showinfo("Already running", f"Still running: {self._job_name}")
            return
        if not SETTLE_SCRIPT.exists():
            messagebox.showerror("Missing", str(SETTLE_SCRIPT))
            return

        def _done(_ok: bool) -> None:
            self._load_perf()

        self._start_job(
            "Settle live ledger",
            [str(_python()), str(SETTLE_SCRIPT), "--settle-only"],
            on_done=_done,
        )

    def _open_tt_today(self) -> None:
        path = TT_TODAY if TT_TODAY.exists() else ROOT / "experiments" / "gameday_scan" / "SCORE_TEAM_TOTALS.csv"
        if not path.exists():
            messagebox.showinfo(
                "No TT card yet",
                "Click Refresh Score Predictions first — it builds today’s team-total paper card.",
            )
            return
        os.startfile(str(path))  # noqa: S606

    def _open_tt_ledger(self) -> None:
        path = TT_LEDGER_MD if TT_LEDGER_MD.exists() else ROOT / "experiments" / "gameday_scan" / "TT_LEDGER.md"
        if not path.exists():
            messagebox.showinfo("No TT ledger", "Refresh Score Predictions once to start the paper log.")
            return
        os.startfile(str(path))  # noqa: S606

    def _open_ledger(self) -> None:
        if not LEDGER_CSV.exists():
            messagebox.showinfo("No ledger", "Run a scan first — PLAYS are logged automatically.")
            return
        os.startfile(str(LEDGER_CSV))  # noqa: S606

    def _open_ledger_md(self) -> None:
        path = LEDGER_MD if LEDGER_MD.exists() else ROOT / "docs" / "LIVE_LEDGER.md"
        if not path.exists():
            messagebox.showinfo("No report", "Run a scan first.")
            return
        os.startfile(str(path))  # noqa: S606

    def _open_decision(self) -> None:
        path = DECISION if DECISION.exists() else PLAYS_SIMPLE
        if not path.exists():
            messagebox.showinfo("No scan", "Run Step 4 first.")
            return
        os.startfile(str(path))  # noqa: S606

    def _open_user_odds(self) -> None:
        USER_ODDS.parent.mkdir(parents=True, exist_ok=True)
        if not USER_ODDS.exists():
            example = ROOT / "data" / "gameday" / "odds.example.csv"
            if example.exists():
                USER_ODDS.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                USER_ODDS.write_text(
                    "match_id,book_over25,book_under25,book_h,book_a,book_ahh,book_aha\n",
                    encoding="utf-8",
                )
        os.startfile(str(USER_ODDS))  # noqa: S606

    def _open_docs(self) -> None:
        path = DOCS if DOCS.exists() else ROOT / "docs" / "GAMEDAY_UI.md"
        if path.exists():
            os.startfile(str(path))  # noqa: S606
        else:
            messagebox.showinfo("Help", "See docs/DAILY_GUIDE.md")

    def _on_close(self) -> None:
        if self._proc and self._proc.poll() is None:
            if not messagebox.askokcancel("Quit", "A job is still running. Quit anyway?"):
                return
            try:
                self._proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        self.destroy()


def main() -> None:
    app = GamedayApp()
    app.mainloop()


if __name__ == "__main__":
    main()
