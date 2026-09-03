from __future__ import annotations

import screen_big_money as core
from screen_entry import fetch_jpx_issues_fixed

# backfill_history.py は fetch_jpx_issues を import 時にローカル参照へ束縛するため、
# import 前に core を直し、import 後にも明示的に差し替える。
core.fetch_jpx_issues = fetch_jpx_issues_fixed

import backfill_history as backfill  # noqa: E402

backfill.fetch_jpx_issues = fetch_jpx_issues_fixed


if __name__ == "__main__":
    raise SystemExit(backfill.main())
