from __future__ import annotations

import no_tv_daily_v3 as v3
import no_tv_daily_v4_regime as v4
import no_tv_v4_entry_realism as entry  # applies executable next-session repricing/stat patches

# Strict fixed-policy full-universe validation.
# No policy search and no 2026 tuning.
# The selected architecture is frozen from the pre-2026 400-issue Dense Low-DOF result:
# balanced blend / q=0.97 / Top3 per day.


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
        "selection_mode": "strict_fixed_pre2026_balanced_q097_top3",
        "policy_diagnostics": [],
    }


v4.choose = choose_fixed


if __name__ == "__main__":
    v4.main()
