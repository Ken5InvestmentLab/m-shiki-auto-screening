from __future__ import annotations

# Reuse the already-audited executable-entry repricing patch.
# Importing this module patches v3.build_symbol_frame and v3.stats only for this process.
import no_tv_v4_entry_realism as entry  # noqa: F401
import no_tv_monster_weak_early_pre2026 as monster

# IMPORTANT:
# - Monster weak+early thresholds remain frozen.
# - V31 base policy remains balanced / q=0.98 / Top3.
# - No 2026 tuning.
# - This wrapper only adds next-session executable-entry metrics.


def main():
    monster.main()


if __name__ == "__main__":
    main()
