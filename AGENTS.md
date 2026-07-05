# Project Guidance

## Secrets

- Do not commit Discord webhook URLs or other credentials. Use the `DISCORD_WEBHOOK_URL` environment variable locally and the GitHub Actions secret with the same name in CI.

## Build and test

- Before posting to Discord, verify changes with `py -3 -m py_compile screen_big_money.py` and `py -3 screen_big_money.py --dry-run --no-wait --allow-stale-data`.
- For historical notification tests, pass `--run-at-jst YYYY-MM-DDTHH:MM:SS`; this also screens only OHLCV bars on or before that date unless `--as-of-date` is supplied.
- Generated files under `reports/` are local run artifacts and should stay untracked.
