from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo


JPX_LISTED_ISSUES_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.T"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
JST = ZoneInfo("Asia/Tokyo")


@dataclass(frozen=True)
class ScreeningConfig:
    max_results: int = 20
    max_workers: int = 48
    as_of_date: str | None = None
    run_at_jst: datetime | None = None
    lookback_days: int = 7
    min_turnover_yen: float = 20_000_000
    min_turnover_ratio: float = 5.0
    strong_reaction_pct: float = 0.985
    strong_raw_to_max: float = 0.70
    strong_reaction_to_max: float = 0.70
    quiet_reaction_pct: float = 0.960
    quiet_reaction_to_max: float = 0.55
    min_day_ret: float = -0.06
    max_day_ret: float = 0.06
    min_5d_ret: float = -0.10
    max_5d_ret: float = 0.10
    min_20d_ret: float = -0.18
    max_20d_ret: float = 0.18
    max_price_pos_252: float = 0.55
    max_dd120: float = -0.05
    min_strong_close_loc: float = 0.55
    max_strong_upper_wick: float = 0.35
    min_quiet_close_loc: float = 0.40
    max_quiet_upper_wick: float = 0.40
    max_quiet_abs_5d_ret: float = 0.055
    max_quiet_abs_20d_ret: float = 0.08
    max_quiet_pos20: float = 0.85
    yahoo_range: str = "2y"
    yahoo_interval: str = "1d"


@dataclass
class Issue:
    code: str
    name: str
    market: str


@dataclass
class Candidate:
    code: str
    name: str
    market: str
    lane: str
    date: str
    score: float
    close: float
    day_ret: float
    ret5: float
    ret20: float
    turnover: float
    volume: float
    turnover_ratio: float
    volume_ratio: float
    raw_pct: float
    raw_to_max: float
    qual_pct: float
    qual_to_max: float
    close_loc: float
    upper_wick: float
    pos20: float
    pos252: float
    dd120: float
    range_pct: float

    @property
    def tradingview_url(self) -> str:
        return f"https://jp.tradingview.com/chart/?symbol={quote(f'TSE:{self.code}')}"


@dataclass
class RunResult:
    generated_at_jst: str
    screening_started_at_jst: str
    target_latest_date: str | None
    lookback_start_date: str | None
    jpx_list_date: str | None
    universe_count: int
    yahoo_ok: int
    yahoo_errors: int
    filtered_count: int
    candidate_count: int
    posted_count: int
    latest_date_distribution: dict[str, int]
    lane_counts: dict[str, int]
    error_counts: dict[str, int]
    candidates: list[Candidate]
    notes: list[str]


def now_jst() -> datetime:
    return datetime.now(JST)


def runtime_now(config: ScreeningConfig) -> datetime:
    return config.run_at_jst or now_jst()


def wait_until_jst(target_hhmm: str, no_wait: bool) -> int:
    if no_wait:
        return 0
    hour, minute = parse_hhmm(target_hhmm)
    now = now_jst()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        return int((now - target).total_seconds())
    wait_seconds = int((target - now).total_seconds())
    print(f"Waiting {wait_seconds}s until {target.isoformat()}...", flush=True)
    time.sleep(wait_seconds)
    return 0


def parse_hhmm(value: str) -> tuple[int, int]:
    try:
        hour_s, minute_s = value.split(":", 1)
        hour = int(hour_s)
        minute = int(minute_s)
    except Exception as exc:
        raise argparse.ArgumentTypeError("time must be HH:MM") from exc
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise argparse.ArgumentTypeError("time must be HH:MM")
    return hour, minute


def parse_iso_date(value: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def parse_jst_datetime(value: str) -> datetime:
    normalized = value.strip().replace(" ", "T")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("datetime must be YYYY-MM-DDTHH:MM[:SS]") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=JST)
    return parsed.astimezone(JST)


def safe_median(values: np.ndarray) -> float:
    valid = values[np.isfinite(values) & (values > 0)]
    return 0.0 if valid.size == 0 else float(np.median(valid))


def percentile_rank(values: np.ndarray, value: float) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return 0.0
    return float((np.sum(valid <= value) + 1.0) / (valid.size + 1.0))


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt_signed_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:+.{digits}f}%"


def fmt_oku(value_yen: float) -> str:
    return f"{value_yen / 100_000_000:.2f}億円"


def fmt_volume(value: float) -> str:
    if value >= 100_000_000:
        return f"{value / 100_000_000:.2f}億株"
    if value >= 10_000:
        return f"{value / 10_000:.1f}万株"
    return f"{value:,.0f}株"


def lane_label(lane: str) -> str:
    return {"strong": "strong 強反応", "quiet": "quiet 静かな反応"}.get(lane, lane)


def fmt_lane_counts(lane_counts: dict[str, int]) -> str:
    if not lane_counts:
        return "-"
    return " / ".join(f"{lane_label(lane)} {count}" for lane, count in lane_counts.items())


def lookback_start_date(target_latest_date: str | None, lookback_days: int) -> str | None:
    if not target_latest_date:
        return None
    return (datetime.strptime(target_latest_date, "%Y-%m-%d").date() - timedelta(days=lookback_days)).isoformat()


def fmt_jst_datetime(value: str | None) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=JST)
    return parsed.astimezone(JST).strftime("%Y-%m-%d %H:%M JST")


def embed_timestamp(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(timezone.utc).isoformat()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=JST)
    return parsed.astimezone(timezone.utc).isoformat()


def fetch_jpx_issues(url: str = JPX_LISTED_ISSUES_URL) -> tuple[str | None, list[Issue]]:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content))
    date_col, code_col, name_col, market_col = df.columns[0], df.columns[1], df.columns[2], df.columns[3]
    domestic = df[df[market_col].astype(str).str.contains("\u5185\u56fd\u682a\u5f0f", na=False, regex=False)]
    issues: list[Issue] = []
    for _, row in domestic.iterrows():
        issues.append(
            Issue(
                code=str(row[code_col]).strip(),
                name=str(row[name_col]).strip(),
                market=str(row[market_col]).replace("（内国株式）", "").strip(),
            )
        )
    list_date = str(domestic[date_col].iloc[0]) if not domestic.empty else None
    return list_date, issues


def fetch_chart(issue: Issue, config: ScreeningConfig) -> tuple[dict[str, Any] | None, str | None]:
    url = YAHOO_CHART_URL.format(symbol=issue.code)
    params = {"range": config.yahoo_range, "interval": config.yahoo_interval}
    last_error: str | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/plain,*/*"},
                timeout=25,
            )
            if response.status_code in {429, 502, 503, 504}:
                last_error = f"http_{response.status_code}"
                time.sleep(0.8 * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json()
            err = payload.get("chart", {}).get("error")
            if err:
                return None, err.get("description") or "chart_error"
            result = payload["chart"]["result"][0]
            timestamps = result.get("timestamp") or []
            quote_data = result.get("indicators", {}).get("quote", [{}])[0]
            if not timestamps or not quote_data:
                return None, "empty_chart"
            return {
                "timestamp": timestamps,
                "open": quote_data.get("open") or [],
                "high": quote_data.get("high") or [],
                "low": quote_data.get("low") or [],
                "close": quote_data.get("close") or [],
                "volume": quote_data.get("volume") or [],
            }, None
        except Exception as exc:  # noqa: BLE001 - network errors need a compact reason
            last_error = type(exc).__name__
            time.sleep(0.5 * (attempt + 1))
    return None, last_error or "fetch_failed"


def chart_to_arrays(chart: dict[str, Any], as_of_date: str | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    n = min(
        len(chart["timestamp"]),
        len(chart["open"]),
        len(chart["high"]),
        len(chart["low"]),
        len(chart["close"]),
        len(chart["volume"]),
    )
    rows: list[tuple[str, float, float, float, float, float]] = []
    for i in range(n):
        values = [chart["open"][i], chart["high"][i], chart["low"][i], chart["close"][i], chart["volume"][i]]
        if any(value is None for value in values):
            continue
        open_, high, low, close, volume = map(float, values)
        if close <= 0 or volume <= 0 or high <= 0 or low <= 0 or high < low:
            continue
        date = datetime.fromtimestamp(chart["timestamp"][i], tz=timezone.utc).date().isoformat()
        if as_of_date and date > as_of_date:
            continue
        rows.append((date, open_, high, low, close, volume))
    if len(rows) < 160:
        return None
    dates = np.array([row[0] for row in rows])
    arrays = {
        key: np.array([row[index] for row in rows], dtype=float)
        for key, index in (("open", 1), ("high", 2), ("low", 3), ("close", 4), ("volume", 5))
    }
    return dates, arrays


def bar_features(open_: float, high: float, low: float, close: float) -> tuple[float, float, float, float]:
    candle_range = high - low
    if candle_range <= 0:
        return 0.5, 0.0, 0.0, 0.0
    close_loc = (close - low) / candle_range
    upper = (high - max(open_, close)) / candle_range
    lower = (min(open_, close) - low) / candle_range
    body = (close - open_) / candle_range
    return float(close_loc), float(upper), float(lower), float(body)


def calc_reactions(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    turnover = close * volume
    n = len(close)
    raw = np.full(n, np.nan, dtype=float)
    quality = np.full(n, np.nan, dtype=float)
    turnover_ratios = np.full(n, np.nan, dtype=float)
    volume_ratios = np.full(n, np.nan, dtype=float)
    close_locs = np.full(n, np.nan, dtype=float)
    upper_wicks = np.full(n, np.nan, dtype=float)

    for i in range(120, n):
        median_turnover = safe_median(turnover[max(0, i - 120) : i])
        median_volume = safe_median(volume[max(0, i - 120) : i])
        if median_turnover <= 0 or median_volume <= 0:
            continue
        turnover_ratio = turnover[i] / median_turnover
        volume_ratio = volume[i] / median_volume
        close_loc, upper_wick, _, _ = bar_features(open_[i], high[i], low[i], close[i])

        base = math.log1p(turnover_ratio) * 34.0 + math.log1p(volume_ratio) * 11.0
        range_pct = (high[i] - low[i]) / close[i - 1] if i > 0 and close[i - 1] > 0 else 0.0
        range_bonus = min(1.20, 0.85 + min(range_pct, 0.12) * 2.0)
        candle_quality = max(0.0, min(1.25, 0.55 + 0.70 * (close_loc - 0.5) - 0.45 * max(0.0, upper_wick - 0.25)))

        raw[i] = base
        quality[i] = base * candle_quality * range_bonus
        turnover_ratios[i] = turnover_ratio
        volume_ratios[i] = volume_ratio
        close_locs[i] = close_loc
        upper_wicks[i] = upper_wick

    return raw, quality, turnover_ratios, volume_ratios, close_locs, upper_wicks


def make_candidate(
    issue: Issue,
    dates: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    raw: np.ndarray,
    quality: np.ndarray,
    turnover_ratios: np.ndarray,
    volume_ratios: np.ndarray,
    i: int,
    config: ScreeningConfig,
) -> tuple[Candidate | None, str | None]:
    turnover = close * volume
    if not np.isfinite(raw[i]) or not np.isfinite(quality[i]):
        return None, "no_reaction"

    raw_valid = raw[120:i][np.isfinite(raw[120:i]) & (raw[120:i] > 0)]
    quality_valid = quality[120:i][np.isfinite(quality[120:i]) & (quality[120:i] > 0)]
    if raw_valid.size < 80 or quality_valid.size < 80:
        return None, "too_few_reactions"

    raw_pct = percentile_rank(raw_valid, raw[i])
    raw_to_max = raw[i] / float(np.nanmax(raw[120 : i + 1]))
    quality_pct = percentile_rank(quality_valid, quality[i])
    quality_to_max = quality[i] / float(np.nanmax(quality[120 : i + 1]))

    day_ret = close[i] / close[i - 1] - 1.0 if i > 0 and close[i - 1] > 0 else 0.0
    ret5 = close[i] / close[i - 5] - 1.0 if i >= 5 and close[i - 5] > 0 else 0.0
    ret20 = close[i] / close[i - 20] - 1.0 if i >= 20 and close[i - 20] > 0 else 0.0
    high20 = float(np.max(high[max(0, i - 20) : i + 1]))
    low20 = float(np.min(low[max(0, i - 20) : i + 1]))
    pos20 = (close[i] - low20) / (high20 - low20) if high20 > low20 else 0.5
    high252 = float(np.max(high[max(0, i - 252) : i + 1]))
    low252 = float(np.min(low[max(0, i - 252) : i + 1]))
    pos252 = (close[i] - low252) / (high252 - low252) if high252 > low252 else 0.5
    high120 = float(np.max(high[max(0, i - 120) : i + 1]))
    dd120 = close[i] / high120 - 1.0 if high120 > 0 else 0.0
    close_loc, upper_wick, _, _ = bar_features(open_[i], high[i], low[i], close[i])
    range_pct = (high[i] - low[i]) / close[i - 1] if i > 0 and close[i - 1] > 0 else 0.0

    common = (
        turnover[i] >= config.min_turnover_yen
        and config.min_day_ret <= day_ret <= config.max_day_ret
        and config.min_5d_ret <= ret5 <= config.max_5d_ret
        and config.min_20d_ret <= ret20 <= config.max_20d_ret
        and pos252 <= config.max_price_pos_252
        and dd120 <= config.max_dd120
        and turnover_ratios[i] >= config.min_turnover_ratio
    )
    strong_lane = (
        common
        and quality_pct >= config.strong_reaction_pct
        and raw_to_max >= config.strong_raw_to_max
        and quality_to_max >= config.strong_reaction_to_max
        and close_loc >= config.min_strong_close_loc
        and upper_wick <= config.max_strong_upper_wick
    )
    quiet_lane = (
        common
        and raw_pct >= config.quiet_reaction_pct
        and raw_to_max >= config.quiet_reaction_to_max
        and close_loc >= config.min_quiet_close_loc
        and upper_wick <= config.max_quiet_upper_wick
        and abs(ret5) <= config.max_quiet_abs_5d_ret
        and abs(ret20) <= config.max_quiet_abs_20d_ret
        and pos20 <= config.max_quiet_pos20
        and turnover_ratios[i] >= 5.0
    )
    if not strong_lane and not quiet_lane:
        return None, "no_lane"

    lane = "strong" if strong_lane else "quiet"
    score = 0.0
    if lane == "strong":
        score += min(30.0, (quality_pct - config.strong_reaction_pct) / (1.0 - config.strong_reaction_pct) * 30.0)
        score += 20.0 * min(1.0, quality_to_max)
    else:
        score += min(24.0, (raw_pct - config.quiet_reaction_pct) / (1.0 - config.quiet_reaction_pct) * 24.0)
        score += 15.0 * min(1.0, raw_to_max)
    score += min(16.0, math.log1p(turnover_ratios[i]) * 4.8)
    score += min(8.0, math.log1p(volume_ratios[i]) * 2.2)
    score += 8.0 if pos252 <= 0.25 else 6.0 if pos252 <= 0.45 else 3.0 if pos252 <= 0.60 else 0.0
    score += 7.0 if dd120 <= -0.30 else 5.0 if dd120 <= -0.15 else 2.0
    score += 5.0 if abs(ret5) <= config.max_quiet_abs_5d_ret and abs(ret20) <= config.max_quiet_abs_20d_ret else 0.0
    score += 3.0 if range_pct <= 0.08 else 0.0
    score += 2.0 if lane == "strong" else 0.0

    return (
        Candidate(
            code=issue.code,
            name=issue.name,
            market=issue.market,
            lane=lane,
            date=str(dates[i]),
            score=float(score),
            close=float(close[i]),
            day_ret=float(day_ret),
            ret5=float(ret5),
            ret20=float(ret20),
            turnover=float(turnover[i]),
            volume=float(volume[i]),
            turnover_ratio=float(turnover_ratios[i]),
            volume_ratio=float(volume_ratios[i]),
            raw_pct=float(raw_pct),
            raw_to_max=float(raw_to_max),
            qual_pct=float(quality_pct),
            qual_to_max=float(quality_to_max),
            close_loc=float(close_loc),
            upper_wick=float(upper_wick),
            pos20=float(pos20),
            pos252=float(pos252),
            dd120=float(dd120),
            range_pct=float(range_pct),
        ),
        None,
    )


def score_issue(issue: Issue, config: ScreeningConfig) -> tuple[list[Candidate], str | None, str | None]:
    chart, error = fetch_chart(issue, config)
    if error:
        return [], None, error
    parsed = chart_to_arrays(chart or {}, config.as_of_date)
    if parsed is None:
        return [], None, "too_few_rows"

    dates, arrays = parsed
    open_ = arrays["open"]
    high = arrays["high"]
    low = arrays["low"]
    close = arrays["close"]
    volume = arrays["volume"]

    raw, quality, turnover_ratios, volume_ratios, _, _ = calc_reactions(open_, high, low, close, volume)
    target_date = str(dates[-1])
    window_start = (datetime.strptime(target_date, "%Y-%m-%d").date() - timedelta(days=config.lookback_days)).isoformat()
    candidates: list[Candidate] = []
    latest_error: str | None = "no_lane"
    for i, date_value in enumerate(dates):
        if i < 120 or str(date_value) < window_start:
            continue
        candidate, candidate_error = make_candidate(
            issue,
            dates,
            open_,
            high,
            low,
            close,
            volume,
            raw,
            quality,
            turnover_ratios,
            volume_ratios,
            i,
            config,
        )
        if candidate:
            candidates.append(candidate)
        elif candidate_error:
            latest_error = candidate_error
    if not candidates:
        return [], target_date, latest_error
    candidates.sort(key=lambda candidate: (candidate.date, candidate.score), reverse=True)
    return candidates[:1], target_date, None


def run_screening(config: ScreeningConfig) -> RunResult:
    run_now = runtime_now(config)
    started_at = run_now.isoformat(timespec="seconds")
    list_date, issues = fetch_jpx_issues()
    candidates: list[Candidate] = []
    latest_dates: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    filtered_count = 0
    yahoo_ok = 0

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {executor.submit(score_issue, issue, config): issue for issue in issues}
        for index, future in enumerate(as_completed(futures), 1):
            issue_candidates, latest_date, error = future.result()
            if latest_date:
                latest_dates[latest_date] = latest_dates.get(latest_date, 0) + 1
                yahoo_ok += 1
            if issue_candidates:
                candidates.extend(issue_candidates)
            elif error:
                if latest_date:
                    filtered_count += 1
                else:
                    error_counts[error] = error_counts.get(error, 0) + 1
            if index % 500 == 0:
                print(f"progress {index}/{len(issues)}", flush=True)

    target_latest_date = max(latest_dates.items(), key=lambda item: item[1])[0] if latest_dates else None
    start_date = lookback_start_date(target_latest_date, config.lookback_days)
    eligible = [
        candidate
        for candidate in candidates
        if target_latest_date and start_date and start_date <= candidate.date <= target_latest_date
    ]
    eligible.sort(key=lambda candidate: (candidate.date, candidate.score), reverse=True)
    top = eligible[: config.max_results]

    lane_counts: dict[str, int] = {}
    for candidate in top:
        lane_counts[candidate.lane] = lane_counts.get(candidate.lane, 0) + 1

    notes: list[str] = []
    if target_latest_date:
        today = run_now.date().isoformat()
        if run_now.weekday() < 5 and target_latest_date < today:
            notes.append(f"Yahoo latest market date is {target_latest_date}; today is {today}. Data may not be updated or today may be a market holiday.")

    return RunResult(
        generated_at_jst=run_now.isoformat(timespec="seconds"),
        screening_started_at_jst=started_at,
        target_latest_date=target_latest_date,
        lookback_start_date=start_date,
        jpx_list_date=list_date,
        universe_count=len(issues),
        yahoo_ok=yahoo_ok,
        yahoo_errors=sum(error_counts.values()),
        filtered_count=filtered_count,
        candidate_count=len(eligible),
        posted_count=len(top),
        latest_date_distribution=dict(sorted(latest_dates.items(), key=lambda item: item[1], reverse=True)[:5]),
        lane_counts=lane_counts,
        error_counts=dict(sorted(error_counts.items(), key=lambda item: item[1], reverse=True)[:10]),
        candidates=top,
        notes=notes,
    )


def write_outputs(result: RunResult, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latest_result.json"
    csv_path = output_dir / "latest_candidates.csv"
    html_path = output_dir / "latest_candidates.html"

    serializable = asdict(result)
    json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "code",
                "name",
                "market",
                "lane",
                "date",
                "score",
                "close",
                "day_ret_pct",
                "ret5_pct",
                "ret20_pct",
                "turnover_oku",
                "volume",
                "turnover_ratio",
                "volume_ratio",
                "raw_pct",
                "raw_to_max",
                "quality_pct",
                "quality_to_max",
                "close_loc",
                "upper_wick",
                "pos20",
                "pos252",
                "dd120_pct",
                "tradingview_url",
            ]
        )
        for rank, candidate in enumerate(result.candidates, 1):
            writer.writerow(
                [
                    rank,
                    candidate.code,
                    candidate.name,
                    candidate.market,
                    candidate.lane,
                    candidate.date,
                    f"{candidate.score:.2f}",
                    f"{candidate.close:.2f}",
                    f"{candidate.day_ret * 100:.2f}",
                    f"{candidate.ret5 * 100:.2f}",
                    f"{candidate.ret20 * 100:.2f}",
                    f"{candidate.turnover / 100_000_000:.4f}",
                    f"{candidate.volume:.0f}",
                    f"{candidate.turnover_ratio:.4f}",
                    f"{candidate.volume_ratio:.4f}",
                    f"{candidate.raw_pct:.4f}",
                    f"{candidate.raw_to_max:.4f}",
                    f"{candidate.qual_pct:.4f}",
                    f"{candidate.qual_to_max:.4f}",
                    f"{candidate.close_loc:.4f}",
                    f"{candidate.upper_wick:.4f}",
                    f"{candidate.pos20:.4f}",
                    f"{candidate.pos252:.4f}",
                    f"{candidate.dd120 * 100:.2f}",
                    candidate.tradingview_url,
                ]
            )

    html_path.write_text(render_html(result), encoding="utf-8")
    return json_path, csv_path, html_path


def render_html(result: RunResult) -> str:
    rows = []
    for rank, candidate in enumerate(result.candidates, 1):
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td><a href=\"{escape(candidate.tradingview_url)}\">{escape(candidate.code)}</a></td>"
            f"<td>{escape(candidate.name)}</td>"
            f"<td>{escape(candidate.lane)}</td>"
            f"<td>{escape(candidate.market)}</td>"
            f"<td>{escape(candidate.date)}</td>"
            f"<td>{candidate.score:.1f}</td>"
            f"<td>{candidate.close:.1f}</td>"
            f"<td>{fmt_pct(candidate.day_ret)}</td>"
            f"<td>{fmt_pct(candidate.ret5)}</td>"
            f"<td>{fmt_pct(candidate.ret20)}</td>"
            f"<td>{fmt_oku(candidate.turnover)}</td>"
            f"<td>{fmt_volume(candidate.volume)}</td>"
            f"<td>{candidate.turnover_ratio:.1f}x</td>"
            f"<td>{candidate.raw_pct * 100:.1f}%</td>"
            f"<td>{candidate.raw_to_max:.2f}</td>"
            f"<td>{candidate.close_loc:.2f}</td>"
            f"<td>{candidate.upper_wick:.2f}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>Big Money Screening</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    th:nth-child(2), th:nth-child(3), th:nth-child(4), th:nth-child(5),
    td:nth-child(2), td:nth-child(3), td:nth-child(4), td:nth-child(5) {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    .meta {{ color: #555; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <h1>Big Money Screening</h1>
  <div class="meta">
    generated: {escape(result.generated_at_jst)} / target date: {escape(str(result.target_latest_date))} /
    candidates: {result.posted_count}
  </div>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Code</th><th>Name</th><th>Lane</th><th>Market</th><th>Date</th>
        <th>Score</th><th>Close</th><th>1D</th><th>5D</th><th>20D</th><th>Turnover</th>
        <th>Volume</th><th>Ratio</th><th>Raw pct</th><th>Raw max</th><th>Close loc</th><th>Upper</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</body>
</html>
"""


def build_summary_embed(result: RunResult, delay_seconds: int, data_stale: bool = False) -> dict[str, Any]:
    title = "大口仕込みスクリーニング"
    description = "条件通過なし" if not result.candidates else f"{result.posted_count}銘柄を検出"
    if data_stale:
        description = "Yahoo日足データが本日分に更新されていない可能性があります"
    fields = [
        {"name": "判定日時", "value": fmt_jst_datetime(result.generated_at_jst), "inline": True},
        {"name": "通知", "value": f"{result.posted_count}銘柄", "inline": True},
        {"name": "対象期間", "value": f"{result.lookback_start_date or '-'} - {result.target_latest_date or '-'}", "inline": False},
        {"name": "内訳", "value": fmt_lane_counts(result.lane_counts), "inline": False},
    ]
    if result.notes:
        fields.append({"name": "注意", "value": "\n".join(result.notes)[:1000], "inline": False})
    return {
        "title": title,
        "description": description,
        "color": 0xF59E0B if data_stale else 0x2DD4BF,
        "timestamp": embed_timestamp(result.generated_at_jst),
        "fields": fields,
        "footer": {"text": "Yahoo Finance OHLCV / JPX listed issues"},
    }


def build_candidate_embeds(candidates: list[Candidate]) -> list[dict[str, Any]]:
    embeds: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, 1):
        color = 0x22C55E if candidate.lane == "strong" else 0xEAB308
        embeds.append(
            {
                "title": f"#{rank} {candidate.code} {candidate.name}",
                "url": candidate.tradingview_url,
                "description": f"反応日 {candidate.date} / {lane_label(candidate.lane)} / {candidate.market} / スコア {candidate.score:.1f}",
                "color": color,
                "fields": [
                    {
                        "name": "株価",
                        "value": f"終値 {candidate.close:.1f}\n1D {fmt_signed_pct(candidate.day_ret)}",
                        "inline": True,
                    },
                    {
                        "name": "出来高・代金",
                        "value": f"{fmt_volume(candidate.volume)}\n{fmt_oku(candidate.turnover)}",
                        "inline": True,
                    },
                    {"name": "通常比", "value": f"{candidate.turnover_ratio:.1f}x", "inline": True},
                    {
                        "name": "短期推移",
                        "value": f"5D {fmt_signed_pct(candidate.ret5)} / 20D {fmt_signed_pct(candidate.ret20)}",
                        "inline": False,
                    },
                ],
                "footer": {"text": "タイトルからTradingViewを開けます"},
            }
        )
    return embeds


def post_webhook(webhook_url: str, payload: dict[str, Any], files: list[tuple[str, bytes, str]] | None = None) -> None:
    data = {"payload_json": json.dumps(payload, ensure_ascii=False)}
    request_files = None
    if files:
        request_files = {}
        for index, (filename, content, mime_type) in enumerate(files):
            request_files[f"files[{index}]"] = (filename, content, mime_type)
    for attempt in range(4):
        response = requests.post(webhook_url, data=data, files=request_files, timeout=30)
        if response.status_code == 429:
            retry_after = response.json().get("retry_after", 1.0)
            time.sleep(float(retry_after) + 0.25)
            continue
        response.raise_for_status()
        return
    response.raise_for_status()


def post_to_discord(result: RunResult, webhook_url: str, output_paths: tuple[Path, Path, Path], delay_seconds: int, dry_run: bool) -> None:
    data_stale = bool(result.notes and result.target_latest_date and result.target_latest_date < now_jst().date().isoformat())
    summary_payload = {"embeds": [build_summary_embed(result, delay_seconds, data_stale=data_stale)]}

    if dry_run:
        print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
        for embed in build_candidate_embeds(result.candidates):
            print(json.dumps(embed, ensure_ascii=False, indent=2))
        return

    post_webhook(webhook_url, summary_payload)
    embeds = build_candidate_embeds(result.candidates)
    for start in range(0, len(embeds), 10):
        post_webhook(webhook_url, {"embeds": embeds[start : start + 10]})


def retry_until_fresh(args: argparse.Namespace, config: ScreeningConfig) -> RunResult:
    deadline = time.time() + args.retry_until_updated_minutes * 60
    attempt = 0
    while True:
        attempt += 1
        print(f"screening attempt {attempt}", flush=True)
        result = run_screening(config)
        run_now = runtime_now(config)
        today = run_now.date().isoformat()
        if args.allow_stale_data or run_now.weekday() >= 5 or result.target_latest_date == today:
            return result
        if time.time() >= deadline:
            result.notes.append(f"Retry deadline reached; latest date remains {result.target_latest_date}.")
            return result
        print(
            f"latest date {result.target_latest_date} is not {today}; retrying in {args.retry_interval_seconds}s",
            flush=True,
        )
        time.sleep(args.retry_interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen TSE names for fresh pre-rise big-money reactions.")
    parser.add_argument("--dry-run", action="store_true", help="Do not post to Discord; print generated payloads.")
    parser.add_argument("--no-wait", action="store_true", help="Do not wait until the configured JST time.")
    parser.add_argument("--wait-until", default="15:52", help="JST HH:MM to start screening after the workflow boots.")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=48)
    parser.add_argument("--min-turnover-yen", type=float, default=20_000_000)
    parser.add_argument("--lookback-days", type=int, default=7, help="Notify symbols with reactions within this many calendar days.")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--as-of-date", type=parse_iso_date, help="Backtest using OHLCV bars on or before YYYY-MM-DD.")
    parser.add_argument("--run-at-jst", type=parse_jst_datetime, help="Override run timestamp, e.g. 2026-07-02T15:52:00.")
    parser.add_argument("--retry-until-updated-minutes", type=int, default=20)
    parser.add_argument("--retry-interval-seconds", type=int, default=120)
    parser.add_argument("--allow-stale-data", action="store_true", help="Do not retry when latest Yahoo date is older than today.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    delay_seconds = wait_until_jst(args.wait_until, args.no_wait)
    as_of_date = args.as_of_date or (args.run_at_jst.date().isoformat() if args.run_at_jst else None)
    config = ScreeningConfig(
        max_results=args.max_results,
        max_workers=args.max_workers,
        as_of_date=as_of_date,
        run_at_jst=args.run_at_jst,
        lookback_days=args.lookback_days,
        min_turnover_yen=args.min_turnover_yen,
    )
    result = retry_until_fresh(args, config)
    output_paths = write_outputs(result, Path(args.output_dir))

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url and not args.dry_run:
        print("DISCORD_WEBHOOK_URL is required unless --dry-run is used.", file=sys.stderr)
        return 2

    post_to_discord(result, webhook_url, output_paths, delay_seconds, args.dry_run)
    print(
        json.dumps(
            {
                "target_latest_date": result.target_latest_date,
                "candidate_count": result.candidate_count,
                "posted_count": result.posted_count,
                "output_files": [str(path) for path in output_paths],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
