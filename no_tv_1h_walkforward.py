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
import no_tv_1h_lab as h1


QS=(.88,.90,.92,.94,.95,.96,.97,.98,.985,.99,.995)
TOPS=(1,2,3,5)


def parse_date(v):
    try: return date.fromisoformat(v).isoformat()
    except ValueError as e: raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from e


def folds(df,holdout_start):
    ds=sorted(df.loc[df.date<holdout_start,"date"].unique())
    if len(ds)<120: raise RuntimeError("not enough pre-holdout sessions")
    p40=int(len(ds)*.40); p60=int(len(ds)*.60); p80=int(len(ds)*.80)
    out=[]
    for a,b,c,d in [(0,p40,p40,p60),(0,p60,p60,p80),(0,p80,p80,len(ds))]:
        out.append({"train_start":ds[a],"train_end":ds[b-1],"valid_start":ds[c],"valid_end":ds[d-1]})
    return out


def pack_fold(df,f):
    tr=h1.period(df,f["train_start"],f["train_end"])
    va=h1.period(df,f["valid_start"],f["valid_end"])
    ms=h1.fit(tr)
    return tr,va,h1.comps(ms,tr),h1.comps(ms,va)


def fold_eval(pack,blend,q,k):
    tr,va,trc,vac=pack
    ts=h1.score(tr,trc,trc[2],blend)
    vs=h1.score(va,vac,trc[2],blend)
    thr=float(np.quantile(ts.score100,q))
    ev=h1.apply_policy(vs,thr,k)
    return h1.stats(ev),thr


def rank_fold_stats(ss):
    if not ss: return None
    if min(s["n"] for s in ss)<5: return None
    av=[s["robust_avg"] for s in ss]; wr=[s["wr"] for s in ss]; hit=[s["target_rate"] for s in ss]
    if min(av)<=0 or min(wr)<.45:
        return None
    return (
        3.2*min(av)+.45*min(wr)+.35*min(hit)
        +1.2*float(np.median(av))+.2*float(np.median(wr))+.1*float(np.median(hit))
        +.008*math.log1p(sum(s["n"] for s in ss))
    )


def choose(packs):
    best=None
    for name,blend in h1.BLENDS.items():
        for q in QS:
            for k in TOPS:
                ss=[]; th=[]
                for p in packs:
                    s,t=fold_eval(p,blend,q,k); ss.append(s); th.append(t)
                r=rank_fold_stats(ss)
                if r is None:
                    continue
                if best is None or r>best["rank"]:
                    best={"blend_name":name,"blend":blend,"q":q,"top_per_day":k,
                          "fold_stats":ss,"fold_thresholds":th,"rank":r,
                          "robust_policy_found":True}
    if best is not None:
        return best
    name="balanced"; blend=h1.BLENDS[name]; q=.99; k=1
    ss=[]; th=[]
    for p in packs:
        s,t=fold_eval(p,blend,q,k); ss.append(s); th.append(t)
    return {"blend_name":name,"blend":blend,"q":q,"top_per_day":k,
            "fold_stats":ss,"fold_thresholds":th,"rank":None,
            "robust_policy_found":False}


def report(r):
    cur=r["current_reference"]; ex=r["holdout"]["h1_pine_current_prevday"]; ml=r["holdout"]["h1_walkforward"]
    lines=[
        "# No-TV 1H Walk-Forward",
        "",
        f"- Data: {r['history_start']} ～ {r['end_date']}",
        f"- Holdout: {r['holdout_start']} ～ {r['end_date']}",
        f"- Universe: {r['universe_count']} / Yahoo成功: {r['yahoo_ok']}",
        "",
        "|方式|n|5BD平均|Robust平均|勝率|+10%Hit|Hit率|",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"|現行4H Stable★6|{cur['n']}|{cur['avg']*100:+.1f}%|-|{cur['wr']*100:.1f}%|{cur['hits']}|{cur['target_rate']*100:.1f}%|",
        f"|1H Pine Exact+前日Stable|{ex['n']}|{ex['avg']*100:+.1f}%|{ex['robust_avg']*100:+.1f}%|{ex['wr']*100:.1f}%|{ex['hits']}|{ex['target_rate']*100:.1f}%|",
        f"|**1H WalkForward ML**|**{ml['n']}**|**{ml['avg']*100:+.1f}%**|**{ml['robust_avg']*100:+.1f}%**|**{ml['wr']*100:.1f}%**|**{ml['hits']}**|**{ml['target_rate']*100:.1f}%**|",
        "",
        f"- blend={r['policy']['blend_name']} / q={r['policy']['q']} / top/day={r['policy']['top_per_day']} / final threshold={r['policy']['final_threshold']:.2f}",
        "",
        "## Fold results",
    ]
    for i,s in enumerate(r["policy"]["fold_stats"],1):
        lines.append(f"- Fold{i}: n={s['n']} avg={s['avg']*100:+.1f}% robust={s['robust_avg']*100:+.1f}% wr={s['wr']*100:.1f}% hit={s['target_rate']*100:.1f}%")
    return "\n".join(lines)


def run(args):
    end_d=date.fromisoformat(args.end_date)
    history_start=(end_d-timedelta(days=args.history_days)).isoformat()
    holdout_start=(end_d-timedelta(days=args.holdout_days)).isoformat()

    list_date,issues=fetch_jpx_issues_fixed(); issues=[i for i in issues if "優先" not in i.name]
    if args.max_issues: issues=issues[:args.max_issues]
    frames=[]; errors={}; ok=0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs={ex.submit(h1.fetch_one,i,history_start,args.end_date):i for i in issues}
        for n,f in enumerate(as_completed(futs),1):
            try: fr,err=f.result()
            except Exception as e: fr,err=None,type(e).__name__
            if fr is not None and not fr.empty: frames.append(fr); ok+=1
            elif err: errors[err]=errors.get(err,0)+1
            if n%250==0: print(f"progress {n}/{len(issues)} frames={len(frames)}",flush=True)
    if not frames: raise RuntimeError("no 1h data")
    df=pd.concat(frames,ignore_index=True)
    df=h1.add_xranks(df).sort_values(["ts","symbol"]).reset_index(drop=True)

    fs=folds(df,holdout_start)
    packs=[]
    for i,f in enumerate(fs,1):
        p=pack_fold(df,f); packs.append(p)
        print(f"fold{i} train={len(p[0])} valid={len(p[1])}",flush=True)
    pol=choose(packs)

    pre_start=df.loc[df.date<holdout_start,"date"].min()
    pre_end=df.loc[df.date<holdout_start,"date"].max()
    pre=h1.period(df,pre_start,pre_end)
    hold=h1.period(df,holdout_start,args.end_date)
    ms=h1.fit(pre)
    pc=h1.comps(ms,pre); hc=h1.comps(ms,hold)
    bl=tuple(pol["blend"])
    ps=h1.score(pre,pc,pc[2],bl); hs=h1.score(hold,hc,pc[2],bl)
    thr=float(np.quantile(ps.score100,pol["q"]))
    ev=h1.apply_policy(hs,thr,pol["top_per_day"])
    exact=h1.current_score_events(hold)

    return {
        "generated_at_jst":core.now_jst().isoformat(timespec="seconds"),
        "history_start":history_start,"holdout_start":holdout_start,"end_date":args.end_date,
        "jpx_list_date":list_date,"universe_count":len(issues),"yahoo_ok":ok,"errors":errors,
        "candidate_rows":len(df),"folds":fs,"current_reference":h1.CURRENT_REFERENCE,
        "policy":{"robust_policy_found":pol.get("robust_policy_found",False),
                  "blend_name":pol["blend_name"],"blend":pol["blend"],"q":pol["q"],
                  "top_per_day":pol["top_per_day"],"fold_stats":pol["fold_stats"],
                  "fold_thresholds":pol["fold_thresholds"],"final_threshold":thr},
        "holdout":{"h1_pine_current_prevday":h1.stats(exact),"h1_walkforward":h1.stats(ev)},
        "top_events":ev.sort_values("score100",ascending=False).head(100)[
            ["ts","symbol","name","score100","p_pos","p_hit10","pred_ret","perf_5bd"]
        ].astype({"ts":str}).to_dict("records"),
    }


def main():
    today=core.now_jst().date()
    p=argparse.ArgumentParser()
    p.add_argument("--end-date",type=parse_date,default=today.isoformat())
    p.add_argument("--history-days",type=int,default=720)
    p.add_argument("--holdout-days",type=int,default=365)
    p.add_argument("--max-workers",type=int,default=24)
    p.add_argument("--max-issues",type=int)
    p.add_argument("--output-dir",default="reports_no_tv_1h_wf")
    a=p.parse_args()
    r=run(a); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"comparison_1h_wf.json").write_text(json.dumps(r,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    (out/"comparison_1h_wf.md").write_text(report(r),encoding="utf-8")
    print(json.dumps({"holdout":r["holdout"],"policy":r["policy"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
