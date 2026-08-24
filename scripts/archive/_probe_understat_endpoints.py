"""Find Understat endpoints that expose match roster / player match data."""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

UA = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest", "Referer": "https://understat.com/"}

mid = "16642"
candidates = [
    f"https://understat.com/match/{mid}",
    f"https://understat.com/match/{mid}/1",
    f"https://understat.com/api/match/{mid}",
    f"https://understat.com/getMatchData/{mid}",
    f"https://understat.com/matchData/{mid}",
    f"https://understat.com/player/1697",  # sample player
]

session = requests.Session()
session.headers.update(UA)

for url in candidates:
    try:
        r = session.get(url, timeout=30)
        text = r.text
        has_roster = "rostersData" in text or "roster" in text.lower()[:5000]
        parses = re.findall(r"(?:var|let|const)\s+(\w+)\s*=\s*JSON\.parse", text)
        print(f"{r.status_code} {len(text):6d} parses={parses[:8]} rosterish={has_roster} {url}")
    except Exception as e:
        print("ERR", url, e)

# Try player page for matchesData
r = session.get("https://understat.com/player/1697", timeout=60)
text = r.text
Path("_probe_player.html").write_text(text[:50000], encoding="utf-8")
parses = re.findall(r"(?:var|let|const)\s+(\w+)\s*=\s*JSON\.parse", text)
print("player parses", parses)
for name in parses:
    m = re.search(name + r"\s*=\s*JSON\.parse\('(.+?)'\)", text)
    if not m:
        continue
    raw = m.group(1).encode("utf-8").decode("unicode_escape")
    data = json.loads(raw)
    print(name, type(data).__name__, end=" ")
    if isinstance(data, list) and data:
        print("len", len(data), "keys", list(data[0].keys())[:12] if isinstance(data[0], dict) else data[0])
    elif isinstance(data, dict):
        print("keys", list(data.keys())[:10])
    else:
        print()
