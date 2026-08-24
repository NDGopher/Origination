#!/usr/bin/env python
"""Probe football-data.co.uk and Understat current-season endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def peek_csv(url: str) -> str:
    r = requests.get(url, timeout=30, headers=UA)
    text = r.content[:400].decode("utf-8", "replace")
    first = text.splitlines()[0] if text else ""
    second = text.splitlines()[1] if text.count("\n") else ""
    div = ""
    if "Div" in first and second:
        cols = first.split(",")
        vals = second.split(",")
        if "Div" in cols:
            div = vals[cols.index("Div")] if len(vals) > cols.index("Div") else ""
    return f"status={r.status_code} bytes={len(r.content)} div={div!r} row1={second[:80]!r}"


def peek_us(league: str, year: int) -> str:
    url = f"https://understat.com/getLeagueData/{league}/{year}"
    h = {
        **UA,
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": f"https://understat.com/league/{league}/{year}",
    }
    r = requests.get(url, timeout=60, headers=h)
    n_dates = n_res = None
    err = ""
    try:
        data = r.json()
        dates = data.get("dates") or []
        n_dates = len(dates)
        n_res = sum(1 for m in dates if m.get("isResult"))
    except Exception as exc:  # noqa: BLE001
        err = f" json_err={exc} start={r.text[:80]!r}"
    return f"status={r.status_code} bytes={len(r.content)} dates={n_dates} results={n_res}{err}"


def main() -> int:
    print("=== football-data mmz4281/2627 ===")
    for code in ["E0", "D1", "I1", "SP1", "P1", "F1", "N1", "B1", "EC", "E1"]:
        url = f"https://www.football-data.co.uk/mmz4281/2627/{code}.csv"
        print(f"  {code}: {peek_csv(url)}")

    print("\n=== football-data new/ ===")
    for name in ["E0.csv", "SP1.csv", "D1.csv", "I1.csv", "P1.csv"]:
        url = f"https://www.football-data.co.uk/new/{name}"
        print(f"  new/{name}: {peek_csv(url)}")

    print("\n=== Understat ===")
    for lg in ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]:
        for y in (2025, 2026):
            print(f"  {lg}/{y}: {peek_us(lg, y)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
