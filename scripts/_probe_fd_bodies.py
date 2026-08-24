#!/usr/bin/env python
"""Inspect FD 2627 bodies and redirects."""

from __future__ import annotations

import requests

UA = {"User-Agent": "Mozilla/5.0"}


def show(code: str) -> None:
    url = f"https://www.football-data.co.uk/mmz4281/2627/{code}.csv"
    r = requests.get(url, timeout=30, headers=UA, allow_redirects=False)
    loc = r.headers.get("Location") or r.headers.get("location")
    print(f"\n{code} status={r.status_code} loc={loc!r} ctype={r.headers.get('Content-Type')!r}")
    print(repr(r.content[:250]))
    if r.status_code in (301, 302, 303, 307, 308, 300) and loc:
        r2 = requests.get(url, timeout=30, headers=UA, allow_redirects=True)
        print(f"  followed status={r2.status_code} final={r2.url} bytes={len(r2.content)}")
        print("  ", repr(r2.content[:200]))


def pages() -> None:
    for path in ["englandm.php", "spainm.php", "germanym.php", "italym.php", "portugalm.php"]:
        url = f"https://www.football-data.co.uk/{path}"
        r = requests.get(url, timeout=30, headers=UA)
        text = r.text
        hits = [line.strip() for line in text.splitlines() if "2627" in line or "mmz4281" in line][:8]
        print(f"\n=== {path} status={r.status_code} 2627/mmz hits={len(hits)} ===")
        for h in hits[:8]:
            print(" ", h[:160])


if __name__ == "__main__":
    for c in ["E0", "EC", "SP1", "P1", "D1", "I1", "N1"]:
        show(c)
    pages()
