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

## 2026-09-12 V4 executable-entry audit
Implementation: `no_tv_v4_entry_realism.py`
Workflow: `.github/workflows/no-tv-v4-entry-realism-smoke.yml`
Run: `34665843274` at commit `18f957bd582436029aa51228da3133a554ddf199`.

The already-selected fixed Core policy (`balanced`, q=0.97, Top3/day) was re-evaluated using executable next-session entries without changing thresholds or using 2026 for tuning. Same pre-2026 400-issue smoke universe: Yahoo usable 250, candidate rows 50,141, holdout n=70.

Holdout results:
- Original signal-close research return: avg +2.633%, robust avg +2.312%, median +1.174%, win 55.7%.
- Production-baseline Next Open -> 5BD close: avg +2.566%, robust avg +2.434%, median +2.762%, win 60.0%.
- Next Open +10% 17.1%, +20% 5.7%, +50% 0.0%, loss10 10.0%, loss20 1.4%.
- Next Open Top1 winner excluded avg +2.087%; Top3 winners excluded avg +1.148%.
- Alternative Next Open -> original signal t+5 close: avg +2.312%, robust avg +2.034%, median +1.084%, win 54.3%, Top3-excluded avg +0.801%.
- Next Close -> 5BD close: avg +2.986%, robust avg +2.770%, median +2.224%, win 60.0%, Top3-excluded avg +1.704%.

Interpretation:
- The Core headline did NOT disappear when same-close execution bias was removed. Under the preferred Next Open baseline, average changed only from +2.633% to +2.566%, while median and win rate improved.
- Top3-excluded Next Open avg remains positive at +1.148%, so the holdout is not solely one to three winners.
- Tail risk remains meaningful: Next Open loss10 rises to 10.0% and the worst trade is about -41.6%, so Core still needs tail-risk/regime work before production promotion.
- This materially clears the execution-realism gate for the 400-issue smoke sample, but it does NOT clear the universe-size/month-concentration gate. Next step remains a fixed-policy larger/full-JPX validation with Next Open as the primary metric; no retuning of q=.97/Top3.

## 2026-09-12 score-tail diagnostic (diagnostic only, not a new cutoff)
Source: artifact from run `34665843274`, all 70 holdout events. This diagnostic uses the existing signal-close `perf_5bd` because the artifact did not contain per-event Next-Open returns; therefore it must NOT be treated as a production-entry rule test.

Observed score quartiles:
- Q1 score ~70.42-71.23: n=18, avg +0.33%, median +0.32%, win 50.0%, loss10 11.1%, loss20 0%.
- Q2 score ~71.23-72.73: n=17, avg +4.33%, median +2.75%, win 58.8%, loss10 5.9%, loss20 5.9%.
- Q3 score ~72.73-77.29: n=17, avg +0.96%, median -0.58%, win 47.1%, loss10 0%, loss20 0%.
- Q4 score ~77.29-89.58: n=18, avg +4.92%, median +4.28%, win 66.7%, loss10 0%, loss20 0%.

Additional observations:
- Highest score quartile retained strong average and had no <=-10% loss in this smoke holdout.
- Lowest score quartile was weak (+0.33% avg) and had the highest loss10 rate (11.1%).
- However score/return is not monotonic because Q3 underperformed Q2, so this does NOT justify inventing a new score cutoff from holdout data.
- Regime diagnostic: Bear n=37 avg +3.10%, median +1.95%, win 59.5%; Bull n=30 avg +2.77%, median +1.32%, win 56.7%; Neutral n=3 avg -4.48% and is too small for inference.
- Worst signal-close trade was 2025-03-24, code 2459, -38.84%. Next-worst losses were much smaller (~-13.3%, -11.3%). This supports investigating a catastrophic-tail guard, but only with pre-registered/simple features and outer validation.

Conclusion: ranking information appears potentially useful for risk stratification, but do not tune a holdout-derived score floor. Full-JPX fixed-policy validation remains the higher-priority gate.

## 2026-09-12 strict fixed full-JPX validation
Branch: `experiment/no-tv-v4-fulljpx-fixed`
Fixed policy: `balanced / q=.97 / Top3`; no policy search and no 2026 tuning.
Workflow: `.github/workflows/no-tv-v4-fulljpx-fixed.yml`
Run: `34673663238` at commit `a2b425d969075de5af94d027a3705ef15e8cc6b4`.
Status at last check: in progress in `Run strict fixed full-JPX pre-2026 validation` step. Compile/setup already passed.

This run is the current Core gate. Do not alter q/Top/day after seeing its result. Primary production-realistic metric is Next Open -> 5BD close, followed by median, win rate, +10/+20/+50, -10/-20, Top1/Top3 exclusion, and month/week concentration.

## Monster next step
V31 code confirms the model itself can remain fixed as the Three-head base; most search freedom is in blend/q/Top/day. For V29 reconstruction, prefer a fixed-policy probe around the historical `min98` idea rather than reopening the full V31 176-policy search. Do not infer V29 success from 2026; validate on pre-2026/purge-safe periods first.

## Safety / production boundary
Do not modify main, production Discord, Spreadsheet, legacy Stable★6/Sniper/Mega, TradingView, or watchlist production logic from research runs.
