# TV-Free Scoring Bot Research State

## Goal
Build an independent TradingView-free scoring system that can compete with or exceed legacy Stable★6 performance. Legacy Stable★6/Sniper/Mega are benchmark-only and must not be components of the new system. 2026 data is reporting/robustness-only and must not be used for threshold/rule tuning.

## Current Monster candidates
- V29 `fixed_min98_both`: saved headline n=35, 5BD avg +4.86%, median +2.90%, +10% rate 31.4%. Raw picks/artifact not yet recovered, so not independently auditable.
- V31: reproducible Three-head Consensus baseline, but strongly week-dependent; de-prioritized as final candidate.
- Cloud Monster weak+early cross-check: fixed conditions from pre-2026 research were previous-day market `median_ret5 <= 0` and candidate `ret10 <= 0.5735294117647058`. 2026 descriptive result was n=43, avg +14.56%, median +5.86%, win 65.1%, +20% 39.5%, loss10 18.6%, Top3-excluded +8.81%. This condition is architecture-dependent and must be revalidated on purge-safe pre-2026 data without retuning.

## Current Core candidates
- 4H SAFE/Core remains defensive reference: 2026 descriptive n=114, avg +1.28%, median +0.54%, win 51.75%, loss10 1.75%.
- V4 Market Regime / Dense Low-DOF is now a live Core research candidate after a pre-2026 4-policy probe survived all three folds. It is NOT promoted to final Core yet because the test was only a 400-issue smoke universe and 2025 events are month-concentrated.
- V8 Adaptive: structurally leakage-aware but original policy search has 84 candidates and is vulnerable to selection overfit; nested pre-2026 validation required.
- V7 Rule Challenger de-prioritized because search path considers 49,392 rule combinations despite simple final rules.
- V31 remains a reproducible research base, not a final Core candidate.

## 2026-09-12 Sparse Low-DOF V4 test
Branch: `experiment/no-tv-research-v3`
Implementation: `no_tv_v4_lowdof.py`
Workflow: `.github/workflows/no-tv-v4-lowdof-smoke.yml`
Policy space reduced from 160 to 8: 2 blends (`balanced`,`hit10`) x q {0.98,0.99} x Top {1,2}. No 2026 data used for policy values.

Pre-2026 smoke result (`end-date=2025-12-30`, 400-issue smoke universe):
- Universe requested: 400; Yahoo usable: 250
- Candidate rows: 50,141
- No robust policy survived all pre-holdout folds (`robust_policy_found=false`)
- Fallback policy: balanced, q=0.99, Top1
- Holdout 2024-12-30 to 2025-12-30: n=4, 5BD avg +3.664%, median +1.502%, win 75%, +10% 25%
- Sample is far too small and fallback was used. Do NOT promote this sparse setup.

Interpretation: q .98/.99 + Top1/2 was too sparse to assess V4 fairly while keeping search freedom low.

## 2026-09-12 Dense Low-DOF V4 test
Implementation: `no_tv_v4_lowdof_dense.py`
Workflow: `.github/workflows/no-tv-v4-lowdof-dense-smoke.yml`
Commits: implementation `ea0e801e3d5a914e42a43dbaca41cbbc399a0837`; workflow `e0d22767c93e14b1b52291019baa551cad11647d`.
Run: `34655530366`.

Pre-registered policy space was only 4 candidates and did not use 2026 results:
- blend fixed to `balanced` only
- q in {0.96, 0.97}
- Top/day in {2, 3}
The reason for lowering density was fixed before the run: the prior q .98/.99 test failed mainly on sparse fold sample size.

Pre-2026 smoke result (`end-date=2025-12-30`, same 400-issue smoke universe, Yahoo usable 250, candidate rows 50,141):
- A robust policy DID survive all three pre-holdout folds.
- Selected policy: balanced, q=0.97, Top3/day.
- Fold1: n=9, avg +2.89%, median +0.71%, win 62.5%, +10% 11.1%.
- Fold2: n=38, avg +0.98%, median +1.33%, win 52.6%, +10% 13.2%.
- Fold3: n=58, avg +3.82%, robust avg +0.93%, median +0.18%, win 51.8%, +10% 20.7%.
- Holdout 2024-12-30 to 2025-12-30: n=70, avg +2.633%, robust avg +2.312%, median +1.174%, win 55.7%.
- Holdout +10% 15.7%, +20% 5.7%, +50% 1.4%, loss10 4.3%, loss20 1.4%.
- Top1 winner excluded avg +1.81%; Top3 winners excluded avg +1.08%.
- Excluding best-return week (2025-08-12 to 2025-08-18) leaves avg about +1.74%, so the result does not collapse to zero on one week.
- Month concentration is the main concern: April has 31/70 events (~44%). Monthly means: Jan -1.47%, Feb -3.10%, Mar -3.40%, Apr +2.46%, Jun +7.52%, Jul +1.89%, Aug +22.23%, Sep +0.21%, Oct +2.04%, Nov +2.02%. No May/Dec picks in this smoke sample.
- There is one severe holdout loser around -38.8%, so loss-tail control still needs work despite low loss10 frequency.

Interpretation:
- This is materially stronger evidence than the sparse Low-DOF test because a policy survived all pre-holdout folds with only 4 candidates and the holdout has n=70.
- Do NOT call it final Core yet. It is still a 400-issue smoke universe, month concentration is high, and early-2025 months are negative.
- Next priority for Core: run the exact fixed `balanced/q=.97/Top3` architecture on a much larger/full JPX universe with no policy search, then audit month/week concentration, Top1/Top3 exclusion, +10/+20/+50, loss10/loss20. If it survives, promote V4 Dense Low-DOF to serious Core candidate.
- Do not alter q=.97 or Top3 using 2026.

## Monster next step
V31 code confirms the model itself can remain fixed as the Three-head base; most search freedom is in blend/q/Top/day. For V29 reconstruction, prefer a fixed-policy probe around the historical `min98` idea rather than reopening the full V31 176-policy search. Do not infer V29 success from 2026; validate on pre-2026/purge-safe periods first.

## Safety / production boundary
Do not modify main, production Discord, Spreadsheet, legacy Stable★6/Sniper/Mega, TradingView, or watchlist production logic from research runs.
