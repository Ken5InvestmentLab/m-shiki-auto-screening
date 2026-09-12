from __future__ import annotations

import json
import numpy as np

import no_tv_daily_v3 as v3
import no_tv_daily_v31 as v31

# Fixed transfer test for the pre-existing Cloud Monster weak+early structure.
# IMPORTANT: these two thresholds are frozen from prior pre-2026 research and must not be tuned here.
PREV_MARKET_MEDIAN_RET5_MAX = 0.0
CANDIDATE_RET10_MAX = 0.5735294117647058

# Architecture transfer target is deliberately fixed as well. This is NOT a V29 reconstruction.
# It asks whether the exact weak+early gate adds value on a reproducible V31-style Three-head base.
FIXED_BLEND_NAME = "balanced"
FIXED_Q = 0.98
FIXED_TOP_PER_DAY = 3

_orig_apply_policy = v3.apply_policy
_orig_run = v31.run


def _with_prev_market_median_ret5(scored):
    if scored.empty:
        out = scored.copy()
        out["prev_market_median_ret5"] = np.nan
        return out
    out = scored.copy()
    daily = out.groupby("date", sort=True)["ret5"].median().sort_index()
    prev = daily.shift(1)
    out["prev_market_median_ret5"] = out["date"].map(prev)
    return out


def apply_weak_early_policy(scored, threshold, top_per_day):
    # Gate is applied before Top/day selection, preserving the exact fixed conditions.
    x = _with_prev_market_median_ret5(scored)
    x = x[
        (x["prev_market_median_ret5"] <= PREV_MARKET_MEDIAN_RET5_MAX)
        & (x["ret10"] <= CANDIDATE_RET10_MAX)
    ].copy()
    return _orig_apply_policy(x, threshold, top_per_day)


def choose_fixed(packs):
    blend = v3.BLENDS[FIXED_BLEND_NAME]
    fold_stats = []
    fold_thresholds = []
    for pack in packs:
        s, t = v31.policy_stats_for_fold(pack, blend, FIXED_Q, FIXED_TOP_PER_DAY)
        fold_stats.append(s)
        fold_thresholds.append(t)
    return {
        "blend_name": FIXED_BLEND_NAME,
        "blend": blend,
        "q": FIXED_Q,
        "top_per_day": FIXED_TOP_PER_DAY,
        "fold_stats": fold_stats,
        "fold_thresholds": fold_thresholds,
        "rank": None,
        "robust_policy_found": True,
        "selection_mode": "fixed_v31_q098_top3_plus_frozen_weak_early_gate",
    }


def run(args):
    # v31.policy_stats_for_fold and final holdout both route through v3.apply_policy.
    # Patch only for this process; production/main are untouched.
    v31.choose_walkforward = choose_fixed
    v3.apply_policy = apply_weak_early_policy
    r = _orig_run(args)
    r["research_variant"] = {
        "name": "Monster weak+early exact frozen gate on fixed V31 transfer architecture",
        "v29_reconstruction": False,
        "thresholds_frozen": True,
        "prev_market_median_ret5_max": PREV_MARKET_MEDIAN_RET5_MAX,
        "candidate_ret10_max": CANDIDATE_RET10_MAX,
        "base_blend": FIXED_BLEND_NAME,
        "base_q": FIXED_Q,
        "base_top_per_day": FIXED_TOP_PER_DAY,
        "tuning_uses_2026": False,
        "note": "Cross-architecture transfer test only; do not infer V29 reproduction from this run.",
    }
    return r


def main():
    # Reuse v31 CLI while swapping run() in-module.
    v31.run = run
    v31.main()


if __name__ == "__main__":
    main()
