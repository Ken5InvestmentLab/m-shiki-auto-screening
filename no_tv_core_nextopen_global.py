from __future__ import annotations

# Low-DOF Core variant: keep the executable next-open target and the exact
# fixed balanced/q=.97/Top3 policy from no_tv_core_nextopen_native, but remove
# regime-specialist models. This tests whether the large fold instability seen
# in the native smoke is caused by specialist fragmentation/overfit.
# No 2026 tuning, no holdout policy search, no production changes.

import no_tv_core_nextopen_native as native
import no_tv_daily_v4_regime as v4


def fit_global_only(train):
    return v4.fit_model(train), {}


v4.fit_regime_pack = fit_global_only


if __name__ == "__main__":
    native.main()
