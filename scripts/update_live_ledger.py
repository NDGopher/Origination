#!/usr/bin/env python
"""Record scan PLAYS and settle the live-system ledger. Does not change pack rules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from origination.gameday.live_ledger import record_from_scan, settle_open, summary, write_report


def main() -> int:
    p = argparse.ArgumentParser(description="Update live performance ledger")
    p.add_argument("--record-only", action="store_true")
    p.add_argument("--settle-only", action="store_true")
    args = p.parse_args()
    n_add = n_set = 0
    if not args.settle_only:
        n_add = record_from_scan()
        print(f"Recorded {n_add} new PLAY(s)", flush=True)
    if not args.record_only:
        n_set = settle_open()
        print(f"Settled {n_set} play(s)", flush=True)
    path = write_report()
    s = summary()
    print(
        f"Ledger: open={s['n_open']} settled={s['n_settled']} total={s['n_total']}",
        flush=True,
    )
    print(f"Wrote {path}", flush=True)
    # Also settle / refresh team-totals paper ledger (does not touch live packs)
    try:
        from origination.gameday.tt_ledger import settle_open as tt_settle
        from origination.gameday.tt_ledger import write_report as tt_report

        if not args.record_only:
            n_tt = tt_settle()
            print(f"Settled {n_tt} TT paper row(s)", flush=True)
        tt_path = tt_report()
        print(f"Wrote {tt_path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"TT ledger note: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
