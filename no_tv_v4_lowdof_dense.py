from __future__ import annotations

import no_tv_daily_v4_regime as v4
import no_tv_daily_v3 as v3

# Dense Low-DOF probe.
# Motivation is fixed before seeing this test's result: the previous 8-policy probe
# was too sparse (q .98/.99, Top1/2) and no policy survived n>=8 in every fold.
# Keep blend fixed to balanced and only relax density, not model/features/regime logic.
POLICIES = (
    ("balanced", v3.BLENDS["balanced"], 0.96, 2),
    ("balanced", v3.BLENDS["balanced"], 0.96, 3),
    ("balanced", v3.BLENDS["balanced"], 0.97, 2),
    ("balanced", v3.BLENDS["balanced"], 0.97, 3),
)


def choose_dense_lowdof(packs):
    best = None
    diagnostics = []
    for name, blend, q, k in POLICIES:
        stats = []
        thresholds = []
        for pack in packs:
            st, threshold = v4.policy_eval(pack, blend, q, k)
            stats.append(st)
            thresholds.append(threshold)
        rank = v4.robust_rank(stats)
        diagnostics.append({
            "blend_name": name,
            "q": q,
            "top_per_day": k,
            "fold_stats": stats,
            "fold_thresholds": thresholds,
            "robust_pass": rank is not None,
            "rank": None if rank is None else float(rank),
        })
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
            "selection_mode": "dense_fixed_low_dof_pre2026",
            "policy_diagnostics": diagnostics.copy(),
        }
        if best is None or candidate["rank"] > best["rank"]:
            best = candidate
    if best is not None:
        best["policy_diagnostics"] = diagnostics
        return best
    return {
        "blend_name": "balanced",
        "blend": v3.BLENDS["balanced"],
        "q": 0.97,
        "top_per_day": 2,
        "fold_stats": [],
        "fold_thresholds": [],
        "rank": None,
        "robust_policy_found": False,
        "policy_space_size": len(POLICIES),
        "selection_mode": "dense_fixed_low_dof_pre2026_fallback",
        "policy_diagnostics": diagnostics,
    }


v4.choose = choose_dense_lowdof


if __name__ == "__main__":
    v4.main()
