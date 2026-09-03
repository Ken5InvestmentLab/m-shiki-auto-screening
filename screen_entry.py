from __future__ import annotations

import re
from io import BytesIO

import pandas as pd
import requests

import screen_big_money as core


MIN_JPX_ISSUES = 3000
MIN_YAHOO_OK_RATIO = 0.50


def _normalize_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def fetch_jpx_issues_fixed() -> tuple[str | None, list[core.Issue]]:
    url = core.resolve_jpx_list_url()
    print(f"JPX listed issues URL: {url}", flush=True)

    response = requests.get(
        url,
        headers={"User-Agent": core.USER_AGENT},
        timeout=45,
    )
    response.raise_for_status()

    df = pd.read_excel(BytesIO(response.content))
    print(
        f"JPX workbook rows={len(df)} cols={len(df.columns)} "
        f"headers={list(map(str, df.columns[:8]))}",
        flush=True,
    )

    if len(df.columns) < 4:
        raise RuntimeError(
            f"JPX workbook schema is invalid: expected >=4 columns, got {len(df.columns)}"
        )

    date_col, code_col, name_col, market_col = (
        df.columns[0],
        df.columns[1],
        df.columns[2],
        df.columns[3],
    )

    market_text = df[market_col].astype(str)

    # JPXの正式表記は「内国株式」。
    # 2026-09-03の修正時に誤って「国内株式」に変わり、対象0件になったため固定する。
    domestic = df[
        market_text.str.contains("内国株式", na=False, regex=False)
    ]

    if domestic.empty:
        samples = market_text.value_counts().head(12).to_dict()
        raise RuntimeError(
            "JPX domestic issue filter returned 0 rows. "
            f"market column={market_col!r}, samples={samples}"
        )

    issues: list[core.Issue] = []
    for _, row in domestic.iterrows():
        code = _normalize_code(row[code_col])
        name = str(row[name_col]).strip()
        market = str(row[market_col]).replace("（内国株式）", "").strip()

        if not code or code.lower() == "nan" or not name or name.lower() == "nan":
            continue

        issues.append(core.Issue(code=code, name=name, market=market))

    list_date = str(domestic[date_col].iloc[0]) if not domestic.empty else None

    print(
        f"JPX parsed issues={len(issues)} list_date={list_date}",
        flush=True,
    )

    if len(issues) < MIN_JPX_ISSUES:
        raise RuntimeError(
            f"JPX issue count is unexpectedly low: {len(issues)} < {MIN_JPX_ISSUES}"
        )

    return list_date, issues


_original_retry_until_fresh = core.retry_until_fresh


def retry_until_fresh_strict(args, config):
    result = _original_retry_until_fresh(args, config)

    if result.universe_count < MIN_JPX_ISSUES:
        raise RuntimeError(
            f"Invalid screening result: universe_count={result.universe_count}"
        )

    if not result.target_latest_date:
        raise RuntimeError(
            "Invalid screening result: target_latest_date is None. "
            "Refusing to write reports or send a false '条件通過なし' notification."
        )

    minimum_yahoo_ok = max(100, int(result.universe_count * MIN_YAHOO_OK_RATIO))
    if result.yahoo_ok < minimum_yahoo_ok:
        raise RuntimeError(
            "Invalid screening result: Yahoo data success count is too low: "
            f"yahoo_ok={result.yahoo_ok}, universe_count={result.universe_count}"
        )

    print(
        "Screening validation OK: "
        f"universe={result.universe_count}, yahoo_ok={result.yahoo_ok}, "
        f"target_latest_date={result.target_latest_date}",
        flush=True,
    )
    return result


# 元コード本体は維持し、壊れたJPXパーサーと異常結果チェックだけ差し替える。
core.fetch_jpx_issues = fetch_jpx_issues_fixed
core.retry_until_fresh = retry_until_fresh_strict


if __name__ == "__main__":
    raise SystemExit(core.main())
