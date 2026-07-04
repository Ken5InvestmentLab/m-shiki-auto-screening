# Project Guidance

## Secrets

- Do not commit Discord webhook URLs or other credentials. Use the `DISCORD_WEBHOOK_URL` environment variable locally and the GitHub Actions secret with the same name in CI.

## Build and test

- Before posting to Discord, verify changes with `py -3 -m py_compile screen_big_money.py` and `py -3 screen_big_money.py --dry-run --no-wait --allow-stale-data`.
- Generated files under `reports/` are local run artifacts and should stay untracked.
