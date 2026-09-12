from __future__ import annotations

import numpy as np
import pandas as pd

import no_tv_daily_v3 as v3
import no_tv_daily_v4_regime as v4
import no_tv_v4_lowdof_dense as dense  # patches v4.choose to the fixed 4-policy Low-DOF selector
import screen_big_money as core

# IMPORTANT:
# This test does NOT change features, model, policy search, q, Top/day, or 2026 usage.
# It only re-prices the exact selected events with executable next-session entries.

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

    # Three explicit repricings so horizon semantics are unambiguous:
    # 1) next open -> original signal t+5 close (same calendar exit as legacy research metric)
    # 2) next open -> 5BD after entry (signal t+6 close)
    # 3) next close -> 5BD after entry (signal t+6 close)
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
    return base


v3.build_symbol_frame = _augment_entry_prices
v3.stats = _stats_with_entry_realism


if __name__ == "__main__":
    v4.main()
