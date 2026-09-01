#!/usr/bin/env python
"""Backfill line history from scan archives and print CLV / steam report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.gameday.play_line_tracker import (  # noqa: E402
    backfill_from_scan_history,
    write_report,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Line-move CLV report for flagged plays")
    p.add_argument(
        "--backfill",
        action="store_true",
        help="Import experiments/gameday_scan/history/* snapshots into line history",
    )
    p.add_argument("--dry-run", action="store_true", help="With --backfill, count only")
    args = p.parse_args()
    if args.backfill:
        n = backfill_from_scan_history(dry_run=args.dry_run)
        print(f"Backfill: {n} observation rows {'would be ' if args.dry_run else ''}added", flush=True)
    from origination.gameday.play_line_tracker import close_past_fixtures

    close_past_fixtures()
    path = write_report()
    print(f"Wrote {path}", flush=True)
    try:
        from origination.gameday.tt_line_tracker import write_report as write_tt

        tt_path = write_tt()
        print(f"Wrote {tt_path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"TT line report skipped: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
