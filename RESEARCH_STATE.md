# TV-Free Scoring Bot Research State

## Goal
Build an independent TradingView-free scoring system that can compete with or exceed legacy Stable★6 performance. Legacy Stable★6/Sniper/Mega are benchmark-only and must not be components of the new system. 2026 data is reporting/robustness-only and must not be used for threshold/rule tuning.

## Current Monster candidates
- V29 `fixed_min98_both`: saved headline n=35, 5BD avg +4.86%, median +2.90%, +10% rate 31.4%. Raw picks/artifact not yet recovered, so not independently auditable.
- V31: reproducible Three-head Consensus baseline, but strongly week-dependent; de-prioritized as final candidate.
- Cloud Monster weak+early cross-check: fixed conditions from pre-2026 research were previous-day market `median_ret5 <= 0` and candidate `ret10 <= 0.5735294117647058`. 2026 descriptive result was n=43, avg +14.56%, median +5.86%, win 65.1%, +20% 39.5%, loss10 18.6%, Top3-excluded +8.81%. This condition is architecture-dependent and must be revalidated on purge-safe pre-2026 data without retuning.

## Current Core candidates
- 4H SAFE/Core remains defensive reference: 2026 descriptive n=114, avg +1.28%, median +0.54%, win 51.75%, loss10 1.75%.
- V8 Adaptive: structurally leakage-aware but original policy search has 84 candidates and is vulnerable to selection overfit; nested pre-2026 validation required.
- V7 Rule Challenger de-prioritized because search path considers 49,392 rule combinations despite simple final rules.
- V4 Market Regime remains conceptually interesting, but original policy search has 160 candidates.

## 2026-09-12 Low-DOF V4 test
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
- Sample is far too small and fallback was used. Do NOT promote Low-DOF V4 to Core candidate based on this result.

Interpretation: reducing selection freedom exposed that V4's apparent strength is not yet robust enough under this strict low-DOF setup. Next Core work should test whether a less sparse but still tightly controlled fixed policy can achieve adequate sample size without reopening a large search space, ideally via nested pre-2026 walk-forward. Do not tune from 2026.

## Safety / production boundary
Do not modify main, production Discord, Spreadsheet, legacy Stable★6/Sniper/Mega, TradingView, or watchlist production logic from research runs.
