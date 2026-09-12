from __future__ import annotations

import numpy as np
import pandas as pd

import no_tv_daily_v3 as v3
import screen_big_money as core

# New Core research direction:
# train/evaluate the model on executable next-session-open -> +5BD close returns
# from the start, instead of training on signal-close -> +5BD and auditing later.
# This is independent from legacy Stable/Sniper/Mega signal matching.

_orig_build_symbol_frame = v3.build_symbol_frame


def _build_symbol_frame_nextopen(issue, chart, history_start, end_date):
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

    # Signal on t close -> executable entry at t+1 open -> exit at t+6 close.
    px["perf_nextopen_5bd"] = c.shift(-6) / o.shift(-1) - 1.0
    out = out.merge(px[["bar_index", "perf_nextopen_5bd"]], on="bar_index", how="left")

    # Replace only the research target used by V4 model/stat code.
    out["perf_signalclose_5bd_audit"] = out["perf_5bd"]
    out["perf_5bd"] = out["perf_nextopen_5bd"]
    return out


v3.build_symbol_frame = _build_symbol_frame_nextopen

import no_tv_daily_v4_regime as v4  # noqa: E402


# Deliberately keep the first test low-DOF and directly comparable with the
# prior V4 fixed architecture. No holdout policy search.
def choose_fixed(_packs):
    return {
        "blend_name": "balanced",
        "blend": v3.BLENDS["balanced"],
        "q": 0.97,
        "top_per_day": 3,
        "fold_stats": [],
        "fold_thresholds": [],
        "rank": None,
        "robust_policy_found": True,
        "policy_space_size": 1,
        "selection_mode": "nextopen_native_fixed_balanced_q097_top3",
    }


v4.choose = choose_fixed


def main():
    v4.main()


if __name__ == "__main__":
    main()
