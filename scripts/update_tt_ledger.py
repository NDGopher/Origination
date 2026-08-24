#!/usr/bin/env python
"""Record / settle Score Predictions team-total paper ledger. Not a live pack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.gameday.tt_ledger import (  # noqa: E402
    archive_tt_snapshot,
    record_candidates,
    select_candidates,
    settle_open,
    write_report,
    write_today_card,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Update team-totals paper ledger")
    p.add_argument("--record-only", action="store_true")
    p.add_argument("--settle-only", action="store_true")
    p.add_argument("--card-only", action="store_true", help="Rebuild TT_TODAY from SCORE_TEAM_TOTALS")
    args = p.parse_args()

    scan = ROOT / "experiments" / "gameday_scan"
    tt_path = scan / "SCORE_TEAM_TOTALS.csv"
    score_path = scan / "SCORE_PREDICTIONS.csv"

    if args.card_only or not args.settle_only:
        import pandas as pd

        tt = pd.read_csv(tt_path) if tt_path.is_file() else pd.DataFrame()
        score = pd.read_csv(score_path) if score_path.is_file() else None
        archive_tt_snapshot(tt if len(tt) else None)
        cands = select_candidates(tt, score_df=score)
        card = write_today_card(cands)
        print(f"Candidates: {len(cands)}  card={card}", flush=True)
        if not args.card_only and not args.settle_only:
            n_add = record_candidates(cands)
            print(f"Recorded {n_add} new TT row(s)", flush=True)

    if not args.record_only and not args.card_only:
        n_set = settle_open()
        print(f"Settled {n_set} TT row(s)", flush=True)

    path = write_report()
    print(f"Wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
