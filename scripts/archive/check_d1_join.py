"""Quick D1 join-rate check after team-name alias expansion."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.data_ingestion.understat import UnderstatIngester
from origination.utils.team_names import DEFAULT_MAPPER as m

fd = pd.read_parquet(ROOT / "data/interim/matches_aligned_D1.parquet")
ing = UnderstatIngester(ROOT / "data/raw/understat", mapper=m)
frames = []
for y in range(2014, 2026):
    if ing.season_json_path("Bundesliga", y).exists():
        frames.append(ing.parse_season("Bundesliga", y))
us = pd.concat(frames, ignore_index=True)
us["date"] = pd.to_datetime(us["date"]).dt.normalize()
fd2 = fd.copy()
fd2["date"] = pd.to_datetime(fd2["date"]).dt.normalize()
# Re-canonicalize in case interim was written with old mapper
fd2["home_team"] = m.map_series(fd["home_team"])
fd2["away_team"] = m.map_series(fd["away_team"])
# Drop any prior xG cols so join metric is clean
drop_cols = [c for c in fd2.columns if c.endswith("_xg") or c in {"home_xg", "away_xg"}]
fd2 = fd2.drop(columns=drop_cols, errors="ignore")
merged = fd2.merge(
    us[["date", "home_team", "away_team", "home_xg"]],
    on=["date", "home_team", "away_team"],
    how="left",
)
rate = float(merged["home_xg"].notna().mean())
print(f"D1 join {int(merged['home_xg'].notna().sum())}/{len(merged)} = {rate:.1%}")
if rate < 0.9:
    miss = merged[merged["home_xg"].isna()][["date", "home_team", "away_team"]].head(15)
    print("sample misses:\n", miss.to_string(index=False))
print("FD unmapped", m.unmapped(set(fd["home_team"]).union(fd["away_team"])))
raw: set[str] = set()
for y in range(2014, 2026):
    p = ROOT / f"data/raw/understat/Bundesliga/{y}_league.json"
    if not p.exists():
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    for match in data.get("dates", []):
        raw.add(match["h"]["title"])
        raw.add(match["a"]["title"])
print("US unmapped", m.unmapped(raw))
