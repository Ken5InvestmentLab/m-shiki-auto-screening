from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import screen_big_money as core
from screen_entry import fetch_jpx_issues_fixed
import no_tv_daily_v3 as v3


QS=(.88,.90,.92,.94,.95,.96,.97,.98,.985,.99,.995)
TOPS=(1,2,3,5)


def parse_date(v):
    try: return date.fromisoformat(v).isoformat()
    except ValueError as e: raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from e


def build_folds(df,holdout_start):
    dates=sorted(df.loc[df.date<holdout_start,"date"].unique())
    if len(dates)<200: raise RuntimeError("not enough pre-holdout dates")
    p40=int(len(dates)*.40); p60=int(len(dates)*.60); p80=int(len(dates)*.80)
    bounds=[(0,p40,p40,p60),(0,p60,p60,p80),(0,p80,p80,len(dates))]
    folds=[]
    for a,b,c,d in bounds:
        folds.append({
            "train_start":dates[a],"train_end":dates[b-1],
            "valid_start":dates[c],"valid_end":dates[d-1],
        })
    return folds


def fold_data(df,f):
    tr=v3.period(df,f["train_start"],f["train_end"])
    va=v3.period(df,f["valid_start"],f["valid_end"])
    return tr,va


def policy_stats_for_fold(pack,blend,q,k):
    tr,va,trc,vac=pack
    tr_score=v3.score_frame(tr,trc,trc[2],blend)
    va_score=v3.score_frame(va,vac,trc[2],blend)
    threshold=float(np.quantile(tr_score.score100,q))
    ev=v3.apply_policy(va_score,threshold,k)
    return v3.stats(ev),threshold


def robust_rank(fold_stats):
    if not fold_stats: return -1e18
    ns=[s["n"] for s in fold_stats]
    if min(ns)<8: return -1e18
    av=[s["robust_avg"] for s in fold_stats]
    wr=[s["wr"] for s in fold_stats]
    hit=[s["target_rate"] for s in fold_stats]
    # どこか1期間だけ勝つ案を排除。
    if min(av)<=0 or min(wr)<.46:
        return -1e17+min(av)
    return (
        3.4*min(av)
        +0.45*min(wr)
        +0.35*min(hit)
        +1.3*float(np.median(av))
        +0.20*float(np.median(wr))
        +0.10*float(np.median(hit))
        +0.01*math.log1p(sum(ns))
    )


def choose_walkforward(packs):
    best=None
    for name,blend in v3.BLENDS.items():
        for q in QS:
            for k in TOPS:
                fs=[]; th=[]
                for pack in packs:
                    s,t=policy_stats_for_fold(pack,blend,q,k)
                    fs.append(s); th.append(t)
                rank=robust_rank(fs)
                if best is None or rank>best["rank"]:
                    best={"blend_name":name,"blend":blend,"q":q,"top_per_day":k,
                          "fold_stats":fs,"fold_thresholds":th,"rank":rank}
    if best is None: raise RuntimeError("no walk-forward policy")
    return best


def report(r):
    cur=r["current_reference"]; ex=r["holdout"]["pine_exact_current"]; vv=r["holdout"]["v2"]; m=r["holdout"]["v31"]
    lines=[
        "# No-TV Stable V3.1 — Walk-Forward 0〜100点",
        "",
        f"- 生成: {r['generated_at_jst']}",
        f"- Data: {r['history_start']} ～ {r['end_date']}",
        f"- 完全未使用Holdout: {r['holdout_start']} ～ {r['end_date']}",
        f"- JPX: {r['universe_count']} / Yahoo成功: {r['yahoo_ok']}",
        "",
        "|方式|n|5BD平均|Robust平均|勝率|+10%Hit|Hit率|",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"|現行4H Stable★6|{cur['n']}|{cur['avg']*100:+.1f}%|-|{cur['wr']*100:.1f}%|{cur['hits']}|{cur['target_rate']*100:.1f}%|",
        f"|日足 Pine Exact+現行|{ex['n']}|{ex['avg']*100:+.1f}%|{ex['robust_avg']*100:+.1f}%|{ex['wr']*100:.1f}%|{ex['hits']}|{ex['target_rate']*100:.1f}%|",
        f"|日足 V2|{vv['n']}|{vv['avg']*100:+.1f}%|{vv['robust_avg']*100:+.1f}%|{vv['wr']*100:.1f}%|{vv['hits']}|{vv['target_rate']*100:.1f}%|",
        f"|**日足 V3.1 WalkForward**|**{m['n']}**|**{m['avg']*100:+.1f}%**|**{m['robust_avg']*100:+.1f}%**|**{m['wr']*100:.1f}%**|**{m['hits']}**|**{m['target_rate']*100:.1f}%**|",
        "",
        "## Policy",
        f"- blend={r['policy']['blend_name']} {r['policy']['blend']}",
        f"- quantile={r['policy']['q']}",
        f"- top/day={r['policy']['top_per_day']}",
        f"- final threshold={r['policy']['final_threshold']:.2f}",
        "",
        "## Walk-forward validation",
    ]
    for i,s in enumerate(r["policy"]["fold_stats"],1):
        lines.append(f"- Fold{i}: n={s['n']} avg={s['avg']*100:+.1f}% robust={s['robust_avg']*100:+.1f}% wr={s['wr']*100:.1f}% hit={s['target_rate']*100:.1f}%")
    lines += [
        "",
        "- 各Foldでモデルを再学習し、未来側のValidationを評価。",
        "- blend / quantile / top/day は3Foldの最悪値を重視して決定。",
        "- 最終Holdoutは探索・モデル選択・閾値選択に一切不使用。",
    ]
    return "\n".join(lines)


def run(args):
    end_d=date.fromisoformat(args.end_date)
    history_start=(end_d-timedelta(days=args.history_days)).isoformat()
    holdout_start=(end_d-timedelta(days=args.holdout_days)).isoformat()

    list_date,issues=fetch_jpx_issues_fixed()
    issues=[i for i in issues if "優先" not in i.name]
    if args.max_issues: issues=issues[:args.max_issues]
    frames=[]; errors={}; ok=0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs={ex.submit(v3.fetch_one,i,history_start,args.end_date):i for i in issues}
        for n,f in enumerate(as_completed(futs),1):
            try: fr,err=f.result()
            except Exception as e: fr,err=None,type(e).__name__
            if fr is not None and not fr.empty: frames.append(fr); ok+=1
            elif err: errors[err]=errors.get(err,0)+1
            if n%500==0: print(f"progress {n}/{len(issues)} frames={len(frames)}",flush=True)
    if not frames: raise RuntimeError("no candidate data")
    df=pd.concat(frames,ignore_index=True)
    df=v3.add_cross_sectional_ranks(df).sort_values(["date","symbol","bar_index"]).reset_index(drop=True)

    folds=build_folds(df,holdout_start)
    packs=[]
    for i,f in enumerate(folds,1):
        tr,va=fold_data(df,f)
        print(f"fold{i} train={len(tr)} valid={len(va)}",flush=True)
        ms=v3.fit_models(tr)
        packs.append((tr,va,v3.predict_components(ms,tr),v3.predict_components(ms,va)))
    policy=choose_walkforward(packs)

    pre=v3.period(df,df.loc[df.date<holdout_start,"date"].min(),df.loc[df.date<holdout_start,"date"].max())
    hold=v3.period(df,holdout_start,args.end_date)
    final_models=v3.fit_models(pre)
    pre_comp=v3.predict_components(final_models,pre)
    hold_comp=v3.predict_components(final_models,hold)
    blend=tuple(policy["blend"])
    pre_score=v3.score_frame(pre,pre_comp,pre_comp[2],blend)
    hold_score=v3.score_frame(hold,hold_comp,pre_comp[2],blend)
    threshold=float(np.quantile(pre_score.score100,policy["q"]))
    ev=v3.apply_policy(hold_score,threshold,policy["top_per_day"])

    exact=v3.baseline_events(hold,"exact_current")
    vv=v3.baseline_events(hold,"v2")

    out={
        "generated_at_jst":core.now_jst().isoformat(timespec="seconds"),
        "history_start":history_start,"holdout_start":holdout_start,"end_date":args.end_date,
        "jpx_list_date":list_date,"universe_count":len(issues),"yahoo_ok":ok,
        "errors":errors,"candidate_rows":len(df),"folds":folds,
        "current_reference":v3.CURRENT_REFERENCE["stable_s6"],
        "policy":{
            "blend_name":policy["blend_name"],"blend":policy["blend"],"q":policy["q"],
            "top_per_day":policy["top_per_day"],"fold_stats":policy["fold_stats"],
            "fold_thresholds":policy["fold_thresholds"],"final_threshold":threshold,
        },
        "holdout":{
            "pine_exact_current":v3.stats(exact),
            "v2":v3.stats(vv),
            "v31":v3.stats(ev),
        },
        "top_events":ev.sort_values("score100",ascending=False).head(100)[
            ["date","symbol","name","score100","p_pos","p_hit10","pred_ret","perf_5bd"]
        ].to_dict("records"),
    }
    return out


def main():
    today=core.now_jst().date()
    p=argparse.ArgumentParser()
    p.add_argument("--end-date",type=parse_date,default=today.isoformat())
    p.add_argument("--history-days",type=int,default=1825)
    p.add_argument("--holdout-days",type=int,default=365)
    p.add_argument("--max-workers",type=int,default=32)
    p.add_argument("--max-issues",type=int)
    p.add_argument("--output-dir",default="reports_no_tv_v31")
    a=p.parse_args()
    r=run(a); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"comparison_v31.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"comparison_v31.md").write_text(report(r),encoding="utf-8")
    print(json.dumps({"holdout":r["holdout"],"policy":r["policy"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
