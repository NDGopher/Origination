"""Parse getMatchData response for roster fields."""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

mid = "16642"
url = f"https://understat.com/getMatchData/{mid}"
r = requests.get(
    url,
    headers={
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://understat.com/match/{mid}",
    },
    timeout=60,
)
text = r.text
Path("_probe_getMatchData.txt").write_text(text[:100000], encoding="utf-8")
print("status", r.status_code, "len", len(text), "ctype", r.headers.get("content-type"))
print("starts", text[:200])

# try raw JSON
try:
    data = r.json()
    print("json type", type(data), list(data.keys())[:20] if isinstance(data, dict) else len(data))
except Exception as e:
    print("not json", e)
    # HTML with embeds?
    parses = re.findall(r"(?:var|let|const)\s+(\w+)\s*=\s*JSON\.parse", text)
    print("parses", parses)
    for name in ["rostersData", "shotsData", "match_info", "teamsData"]:
        m = re.search(name + r"\s*=\s*JSON\.parse\('(.+?)'\)", text)
        if m:
            raw = m.group(1).encode("utf-8").decode("unicode_escape")
            data = json.loads(raw)
            print(name, type(data).__name__)
            if isinstance(data, dict):
                print(" keys", list(data.keys())[:15])
                for side in ("h", "a"):
                    if side in data and isinstance(data[side], dict) and data[side]:
                        p = next(iter(data[side].values()))
                        print(" sample", side, p)
                        break
            elif isinstance(data, list) and data:
                print(" sample", data[0] if isinstance(data[0], dict) else data[0])
