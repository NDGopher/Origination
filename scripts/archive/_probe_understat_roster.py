"""Probe Understat match page for roster JSON."""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

mid = "16642"
url = f"https://understat.com/match/{mid}"
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
print("status", r.status_code, "len", len(r.text))
for name in ["rostersData", "shotsData", "match_info"]:
    m = re.search(name + r"\s*=\s*JSON\.parse\('(.+?)'\)", r.text)
    print(name, "found" if m else "no")
    if not m:
        continue
    raw = m.group(1).encode("utf-8").decode("unicode_escape")
    data = json.loads(raw)
    Path(f"_probe_{name}.json").write_text(json.dumps(data)[:2000], encoding="utf-8")
    if isinstance(data, dict):
        print(" keys", list(data.keys())[:8])
        for side in ("h", "a", "home", "away"):
            if side in data:
                block = data[side]
                if isinstance(block, dict) and block:
                    p = next(iter(block.values()))
                    print(" sample", side, p)
                    break
                if isinstance(block, list) and block:
                    print(" sample", side, block[0])
                    break
