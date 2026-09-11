from __future__ import annotations

import no_tv_daily_v4_regime as v4
import no_tv_daily_v3 as v3

# Deliberately tiny policy space: 2 blends x 2 fixed upper-tail quantiles x Top1/2.
# No 2026 result is used to choose among these values.
POLICIES = (
    ("balanced", v3.BLENDS["balanced"], 0.98, 1),
    ("balanced", v3.BLENDS["balanced"], 0.98, 2),
    ("balanced", v3.BLENDS["balanced"], 0.99, 1),
    ("balanced", v3.BLENDS["balanced"], 0.99, 2),
    ("hit10", v3.BLENDS["hit10"], 0.98, 1),
    ("hit10", v3.BLENDS["hit10"], 0.98, 2),
    ("hit10", v3.BLENDS["hit10"], 0.99, 1),
    ("hit10", v3.BLENDS["hit10"], 0.99, 2),
)


def choose_lowdof(packs):
    best = None
    for name, blend, q, k in POLICIES:
        stats = []
        thresholds = []
        for pack in packs:
            st, threshold = v4.policy_eval(pack, blend, q, k)
            stats.append(st)
            thresholds.append(threshold)
        rank = v4.robust_rank(stats)
        if rank is None:
            continue
        candidate = {
            "blend_name": name,
            "blend": blend,
            "q": q,
            "top_per_day": k,
            "fold_stats": stats,
            "fold_thresholds": thresholds,
            "rank": float(rank),
            "robust_policy_found": True,
            "policy_space_size": len(POLICIES),
            "selection_mode": "fixed_low_dof_pre2026",
        }
        if best is None or candidate["rank"] > best["rank"]:
            best = candidate
    if best is not None:
        return best
    return {
        "blend_name": "balanced",
        "blend": v3.BLENDS["balanced"],
        "q": 0.99,
        "top_per_day": 1,
        "fold_stats": [],
        "fold_thresholds": [],
        "rank": None,
        "robust_policy_found": False,
        "policy_space_size": len(POLICIES),
        "selection_mode": "fixed_low_dof_pre2026_fallback",
    }


# Keep all V4 data/model/regime logic unchanged; replace policy selection only.
v4.choose = choose_lowdof


if __name__ == "__main__":
    v4.main()
