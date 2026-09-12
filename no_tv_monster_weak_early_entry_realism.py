from __future__ import annotations

import numpy as np
import pandas as pd

import no_tv_daily_v3 as v3
import screen_big_money as core

# Executable-entry repricing copied from the already-audited V4 entry-realism audit,
# but kept self-contained on the Monster research branch.
# This changes no features, thresholds, policy selection, Top/day, or 2026 usage.

_orig_build_symbol_frame = v3.build_symbol_frame
_orig_stats = v3.stats


def _augment_entry_prices(issue, chart, history_start, end_date):
    out = _orig_build_symbol_frame(issue, chart, history_start, end_date)
    if out is None or out.empty:
        return out

    parsed = core.chart_to_arrays(chart, None)
    if parsed is None:
        return out
    dates, a = parsed
    px = pd.DataFrame({
        "bar_index": np.arange(len(dates)),
        "open_raw": a["open"],
        "close_raw": a["close"],
    })
    o = pd.to_numeric(px["open_raw"], errors="coerce")
    c = pd.to_numeric(px["close_raw"], errors="coerce")

    px["perf_next_open_to_signal_t5"] = c.shift(-5) / o.shift(-1) - 1.0
    px["perf_next_open_5bd"] = c.shift(-6) / o.shift(-1) - 1.0
    px["perf_next_close_5bd"] = c.shift(-6) / c.shift(-1) - 1.0

    return out.merge(
        px[["bar_index", "perf_next_open_to_signal_t5", "perf_next_open_5bd", "perf_next_close_5bd"]],
        on="bar_index",
        how="left",
    )


def _metric(events: pd.DataFrame, col: str):
    if col not in events.columns:
        return None
    v = pd.to_numeric(events[col], errors="coerce").dropna().to_numpy(float)
    if not len(v):
        return {"n": 0}
    dec = v[v != 0]
    wr = float(np.mean(dec > 0)) if len(dec) else 0.0
    if len(v) >= 10:
        lo, hi = np.quantile(v, [.05, .95])
        robust = float(np.mean(np.clip(v, lo, hi)))
    else:
        robust = float(np.mean(v))
    sv = np.sort(v)[::-1]
    return {
        "n": int(len(v)),
        "avg": float(np.mean(v)),
        "robust_avg": robust,
        "median": float(np.median(v)),
        "wr": wr,
        "hit10": float(np.mean(v >= .10)),
        "hit20": float(np.mean(v >= .20)),
        "hit50": float(np.mean(v >= .50)),
        "loss10": float(np.mean(v <= -.10)),
        "loss20": float(np.mean(v <= -.20)),
        "top1_excluded_avg": float(np.mean(sv[1:])) if len(sv) > 1 else None,
        "top3_excluded_avg": float(np.mean(sv[3:])) if len(sv) > 3 else None,
        "max": float(np.max(v)),
        "min": float(np.min(v)),
    }


def _calendar_dependence(events: pd.DataFrame, col: str):
    if col not in events.columns or "date" not in events.columns:
        return None
    x = events[["date", col]].copy()
    x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna()
    if x.empty:
        return None
    d = pd.to_datetime(x["date"], errors="coerce")
    x = x[d.notna()].copy()
    d = pd.to_datetime(x["date"])
    x["month"] = d.dt.strftime("%Y-%m")
    iso = d.dt.isocalendar()
    x["iso_week"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)

    def summarize(group_col):
        rows = []
        for key, g in x.groupby(group_col, sort=True):
            v = g[col].to_numpy(float)
            rows.append({
                group_col: str(key),
                "n": int(len(v)),
                "avg": float(np.mean(v)),
                "median": float(np.median(v)),
                "wr": float(np.mean(v > 0)),
            })
        return rows

    def leave_one_out(group_col):
        rows = []
        for key in sorted(x[group_col].unique()):
            v = x.loc[x[group_col] != key, col].to_numpy(float)
            rows.append({group_col: str(key), "n": int(len(v)), "avg": float(np.mean(v)) if len(v) else None})
        return rows

    full = x[col].to_numpy(float)
    monthly = summarize("month")
    weekly = summarize("iso_week")
    month_means = sorted((r["avg"], r["month"]) for r in monthly)
    week_means = sorted((r["avg"], r["iso_week"]) for r in weekly)
    loo_month = leave_one_out("month")
    loo_week = leave_one_out("iso_week")
    loo_month_valid = [r for r in loo_month if r["avg"] is not None]
    loo_week_valid = [r for r in loo_week if r["avg"] is not None]
    months_ge5 = [r for r in monthly if r["n"] >= 5]
    weeks_ge3 = [r for r in weekly if r["n"] >= 3]
    return {
        "n": int(len(full)),
        "monthly": monthly,
        "weekly": weekly,
        "positive_month_share": float(np.mean([r["avg"] > 0 for r in monthly])) if monthly else None,
        "positive_month_share_n_ge_5": float(np.mean([r["avg"] > 0 for r in months_ge5])) if months_ge5 else None,
        "positive_week_share_n_ge_3": float(np.mean([r["avg"] > 0 for r in weeks_ge3])) if weeks_ge3 else None,
        "leave_one_month_out": loo_month,
        "leave_one_week_out": loo_week,
        "leave_one_month_out_min_avg": min((r["avg"] for r in loo_month_valid), default=None),
        "leave_one_month_out_max_avg": max((r["avg"] for r in loo_month_valid), default=None),
        "leave_one_week_out_min_avg": min((r["avg"] for r in loo_week_valid), default=None),
        "worst_month": {"month": month_means[0][1], "avg": month_means[0][0]} if month_means else None,
        "best_month": {"month": month_means[-1][1], "avg": month_means[-1][0]} if month_means else None,
        "worst_week": {"iso_week": week_means[0][1], "avg": week_means[0][0]} if week_means else None,
        "best_week": {"iso_week": week_means[-1][1], "avg": week_means[-1][0]} if week_means else None,
    }


def _stats_with_entry_realism(events):
    base = _orig_stats(events)
    extra = {}
    for col in (
        "perf_next_open_to_signal_t5",
        "perf_next_open_5bd",
        "perf_next_close_5bd",
    ):
        m = _metric(events, col)
        if m is not None:
            extra[col] = m
    if extra:
        base["entry_realism"] = extra
    dep = _calendar_dependence(events, "perf_next_open_5bd")
    if dep is not None:
        base["next_open_5bd_calendar_dependence"] = dep
    return base


v3.build_symbol_frame = _augment_entry_prices
v3.stats = _stats_with_entry_realism

import no_tv_monster_weak_early_pre2026 as monster  # noqa: E402


def main():
    monster.main()


if __name__ == "__main__":
    main()
