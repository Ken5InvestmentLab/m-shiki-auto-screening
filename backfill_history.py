from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
from pathlib import Path

from screen_big_money import (
    Candidate,
    ScreeningConfig,
    calc_reactions,
    chart_to_arrays,
    fetch_chart,
    fetch_jpx_issues,
    make_candidate,
    now_jst,
)


def parse_iso_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def score_issue_history(
    issue,
    config: ScreeningConfig,
    start_date: str,
    end_date: str,
) -> tuple[list[Candidate], str | None, str | None]:
    chart, error = fetch_chart(issue, config)
    if error:
        return [], None, error

    parsed = chart_to_arrays(chart or {}, end_date)
    if parsed is None:
        return [], None, "too_few_rows"

    dates, arrays = parsed
    open_ = arrays["open"]
    high = arrays["high"]
    low = arrays["low"]
    close = arrays["close"]
    volume = arrays["volume"]

    raw, quality, turnover_ratios, volume_ratios, _, _ = calc_reactions(
        open_, high, low, close, volume
    )

    latest_date = str(dates[-1])
    candidates: list[Candidate] = []
    latest_error: str | None = "no_lane"

    for i, date_value in enumerate(dates):
        date_key = str(date_value)
        if i < 120 or date_key < start_date or date_key > end_date:
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

    return candidates, latest_date, None if candidates else latest_error


def run_backfill(
    start_date: str,
    end_date: str,
    max_results_per_day: int,
    max_workers: int,
) -> dict:
    config = ScreeningConfig(
        max_results=max_results_per_day,
        max_workers=max_workers,
        as_of_date=end_date,
    )

    jpx_list_date, issues = fetch_jpx_issues()
    all_candidates: list[Candidate] = []
    latest_dates: dict[str, int] = {}
    errors: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(score_issue_history, issue, config, start_date, end_date): issue
            for issue in issues
        }
        for index, future in enumerate(as_completed(futures), 1):
            try:
                candidates, latest_date, error = future.result()
            except Exception as exc:  # noqa: BLE001
                candidates, latest_date, error = [], None, type(exc).__name__

            if latest_date:
                latest_dates[latest_date] = latest_dates.get(latest_date, 0) + 1
            if candidates:
                all_candidates.extend(candidates)
            elif error:
                errors[error] = errors.get(error, 0) + 1

            if index % 500 == 0:
                print(f"progress {index}/{len(issues)}", flush=True)

    by_date: dict[str, list[Candidate]] = {}
    for candidate in all_candidates:
        by_date.setdefault(candidate.date, []).append(candidate)

    selected: list[Candidate] = []
    daily_counts: dict[str, int] = {}
    for date_key in sorted(by_date):
        day_candidates = sorted(by_date[date_key], key=lambda c: c.score, reverse=True)
        top = day_candidates[:max_results_per_day]
        selected.extend(top)
        daily_counts[date_key] = len(top)

    # 保存先では古い順に並ぶ方が確認しやすい。
    selected.sort(key=lambda c: (c.date, -c.score, c.code))

    candidate_dicts = []
    for candidate in selected:
        row = asdict(candidate)
        row["tradingview_url"] = candidate.tradingview_url
        candidate_dicts.append(row)

    return {
        "generated_at_jst": now_jst().isoformat(timespec="seconds"),
        "mode": "historical_backfill_current_rules",
        "start_date": start_date,
        "end_date": end_date,
        "jpx_list_date": jpx_list_date,
        "universe_count": len(issues),
        "raw_qualifying_count": len(all_candidates),
        "candidate_count": len(selected),
        "max_results_per_day": max_results_per_day,
        "daily_counts": daily_counts,
        "latest_date_distribution": dict(
            sorted(latest_dates.items(), key=lambda item: item[1], reverse=True)[:5]
        ),
        "error_counts": dict(sorted(errors.items(), key=lambda item: item[1], reverse=True)[:10]),
        "candidates": candidate_dicts,
        "notes": [
            "Historical results were recalculated with the current screening rules.",
            "The current JPX listed-issue universe is used, so delisted names may be absent.",
        ],
    }


def write_outputs(result: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "backfill_result.json"
    csv_path = output_dir / "backfill_candidates.csv"

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "date", "code", "name", "market", "lane", "score", "close",
            "day_ret_pct", "ret5_pct", "ret20_pct", "turnover_oku",
            "volume", "turnover_ratio", "volume_ratio", "tradingview_url",
        ])
        for candidate in result["candidates"]:
            writer.writerow([
                candidate["date"],
                candidate["code"],
                candidate["name"],
                candidate["market"],
                candidate["lane"],
                f'{candidate["score"]:.2f}',
                f'{candidate["close"]:.2f}',
                f'{candidate["day_ret"] * 100:.2f}',
                f'{candidate["ret5"] * 100:.2f}',
                f'{candidate["ret20"] * 100:.2f}',
                f'{candidate["turnover"] / 100_000_000:.4f}',
                f'{candidate["volume"]:.0f}',
                f'{candidate["turnover_ratio"]:.4f}',
                f'{candidate["volume_ratio"]:.4f}',
                candidate["tradingview_url"],
            ])

    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recalculate M-shiki detections for a historical date range."
    )
    parser.add_argument("--start-date", type=parse_iso_date, default="2026-01-01")
    parser.add_argument("--end-date", type=parse_iso_date)
    parser.add_argument("--max-results-per-day", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=48)
    parser.add_argument("--output-dir", default="reports_backfill")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    end_date = args.end_date or now_jst().date().isoformat()
    if args.start_date > end_date:
        raise SystemExit("start-date must be on or before end-date")

    result = run_backfill(
        start_date=args.start_date,
        end_date=end_date,
        max_results_per_day=args.max_results_per_day,
        max_workers=args.max_workers,
    )
    output_paths = write_outputs(result, Path(args.output_dir))

    print(json.dumps({
        "start_date": result["start_date"],
        "end_date": result["end_date"],
        "candidate_count": result["candidate_count"],
        "raw_qualifying_count": result["raw_qualifying_count"],
        "output_files": [str(path) for path in output_paths],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
