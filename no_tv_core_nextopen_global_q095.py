from __future__ import annotations

# Adjacent sensitivity check for the low-DOF next-open Core.
# Purpose: test whether the global-only model's fold weakness is merely an
# overly-thin q=.97 tail artifact. This is NOT holdout tuning and must not be
# selected by 2025 holdout performance alone.
# Fixed architecture: global-only / balanced / q=.95 / Top3 / no 2026.

import no_tv_core_nextopen_global  # noqa: F401
import no_tv_core_nextopen_native as native
import no_tv_daily_v3 as v3
import no_tv_daily_v4_regime as v4


def choose_fixed_q095(packs):
    blend = v3.BLENDS["balanced"]
    fold_stats = []
    fold_thresholds = []
    for pack in packs:
        s, t = v4.policy_eval(pack, blend, 0.95, 3)
        fold_stats.append(s)
        fold_thresholds.append(t)
    return {
        "blend_name": "balanced",
        "blend": blend,
        "q": 0.95,
        "top_per_day": 3,
        "fold_stats": fold_stats,
        "fold_thresholds": fold_thresholds,
        "rank": None,
        "robust_policy_found": True,
        "policy_space_size": 1,
        "selection_mode": "core_nextopen_global_sensitivity_q095_top3",
    }


v4.choose = choose_fixed_q095


if __name__ == "__main__":
    native.main()
