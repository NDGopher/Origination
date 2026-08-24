"""Probe Understat match page for embedded JSON."""
from __future__ import annotations

import re
from pathlib import Path

import requests

mid = "16642"
url = f"https://understat.com/match/{mid}"
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
text = r.text
Path("_probe_match.html").write_text(text, encoding="utf-8")
print("status", r.status_code, "len", len(text))

for m in re.finditer(r"(?:var|let|const)\s+(\w+)\s*=\s*JSON\.parse", text):
    print("JSON.parse var:", m.group(1), "at", m.start())

for kw in ["roster", "player", "lineup", "shots", "teamsData", "datesData", "rostersData"]:
    print(f"count {kw}:", text.lower().count(kw.lower()))

# dump script tags with understat content
scripts = re.findall(r"<script[^>]*>(.*?)</script>", text, flags=re.S | re.I)
print("n_scripts", len(scripts))
for i, s in enumerate(scripts):
    if "JSON.parse" in s or "roster" in s.lower() or "match_info" in s:
        print("--- script", i, "len", len(s))
        print(s[:800])
        print("...")
