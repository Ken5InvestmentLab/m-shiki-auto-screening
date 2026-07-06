# Project Guidance

## Secrets

- Do not commit Discord webhook URLs or other credentials. Use the `DISCORD_WEBHOOK_URL` environment variable locally and the GitHub Actions secret with the same name in CI.

## Build and test

- Before posting to Discord, verify changes with `py -3 -m py_compile screen_big_money.py` and `py -3 screen_big_money.py --dry-run --no-wait --allow-stale-data`.
- For historical notification tests, pass `--run-at-jst YYYY-MM-DDTHH:MM:SS`; this also screens only OHLCV bars on or before that date unless `--as-of-date` is supplied.
- Generated files under `reports/` are local run artifacts and should stay untracked.

## GAS automation

- The dedicated Apps Script project is linked by `.clasp.json` with `gas/` as the root directory. After editing `gas/Code.js` or `gas/appsscript.json`, sync it with `npx @google/clasp push -f`.
- Do not commit GitHub tokens or GAS Script Properties. The GAS runtime should read the workflow dispatch token from the `GITHUB_TOKEN` Script Property.
