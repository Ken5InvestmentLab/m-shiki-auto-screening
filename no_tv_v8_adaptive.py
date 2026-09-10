from __future__ import annotations

import argparse, calendar, json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import screen_big_money as core
from screen_entry import fetch_jpx_issues_fixed
import no_tv_daily_v3 as v3

CURRENT={"n":55,"avg":.06587272727272729,"wr":.5636363636363636,"hits":10,"target_rate":10/55}
BLENDS=v3.BLENDS
QS=(.94,.96,.97,.98,.985,.99,.995)
TOPS=(1,2,3)

def parse_date(v):
    try:return date.fromisoformat(v).isoformat()
    except ValueError as e:raise argparse.ArgumentTypeError("YYYY-MM-DD") from e

def month_bounds(ts):
    ts=pd.Timestamp(ts).normalize().replace(day=1)
    nxt=ts+pd.offsets.MonthBegin(1)
    return ts,nxt-pd.Timedelta(days=1)

def previous_months(anchor,n):
    a=pd.Timestamp(anchor).replace(day=1)
    return [month_bounds(a-pd.DateOffset(months=i))[0] for i in range(n,0,-1)]

def known_training(df,target_start,lookback_days=900):
    start=(pd.Timestamp(target_start)-pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    t=pd.Timestamp(target_start).strftime("%Y-%m-%d")
    return df[(df.date>=start)&(df.date<t)&df.exit_date_5bd.notna()&(df.exit_date_5bd<t)].copy()

def target_frame(df,start,end,overall_end):
    a=pd.Timestamp(start).strftime("%Y-%m-%d");b=min(pd.Timestamp(end),pd.Timestamp(overall_end)).strftime("%Y-%m-%d")
    return df[(df.date>=a)&(df.date<=b)&df.exit_date_5bd.notna()&(df.exit_date_5bd<=overall_end)].copy()

def sanitize_stats(e):
    return v3.stats(e)

def score_pair(models,train,target,blend):
    trc=v3.predict_components(models,train);tgc=v3.predict_components(models,target)
    ref=trc[2]
    return v3.score_frame(train,trc,ref,blend),v3.score_frame(target,tgc,ref,blend)

def select_scored(scored,threshold,k):
    return v3.apply_policy(scored,threshold,k)

def prepare_window(df,start,end,overall_end):
    tr=known_training(df,start)
    tg=target_frame(df,start,end,overall_end)
    if len(tr)<500 or len(tg)==0:return None
    models=v3.fit_models(tr)
    trc=v3.predict_components(models,tr);tgc=v3.predict_components(models,tg)
    return tr,tg,trc,tgc

def choose_policy(df,holdout_start,overall_end):
    # 3 pseudo-future months, all strictly before final holdout.
    hs=pd.Timestamp(holdout_start).replace(day=1)
    starts=[hs-pd.DateOffset(months=9),hs-pd.DateOffset(months=6),hs-pd.DateOffset(months=3)]
    packs=[]
    for s in starts:
        a,b=month_bounds(s)
        pack=prepare_window(df,a,b,overall_end)
        if pack is not None:
            packs.append((a,b,pack))
    if len(packs)<2:
        raise RuntimeError("too few pre-holdout rolling calibration windows")

    best=None
    for name,blend in BLENDS.items():
      for q in QS:
       for k in TOPS:
        ss=[];valid=True
        for a,b,(tr,tg,trc,tgc) in packs:
            trs=v3.score_frame(tr,trc,trc[2],blend);tgs=v3.score_frame(tg,tgc,trc[2],blend)
            thr=float(np.quantile(trs.score100,q))
            st=sanitize_stats(select_scored(tgs,thr,k));ss.append(st)
            if st["n"]<4 or st["robust_avg"]<=0 or st["wr"]<.45:
                valid=False;break
        if not valid:continue
        minavg=min(s["robust_avg"] for s in ss);minwr=min(s["wr"] for s in ss)
        medavg=float(np.median([s["robust_avg"] for s in ss]));medhit=float(np.median([s["target_rate"] for s in ss]))
        rank=4*minavg+.5*minwr+1.4*medavg+.4*medhit+.008*math.log1p(sum(s["n"] for s in ss))
        if best is None or rank>best["rank"]:
            best={"blend_name":name,"blend":blend,"q":q,"top_per_day":k,"calibration_stats":ss,"rank":rank,
                  "calibration_months":[a.strftime("%Y-%m") for a,_,_ in packs]}
    return best

def rolling_holdout(df,policy,holdout_start,overall_end):
    start=pd.Timestamp(holdout_start).replace(day=1)
    end=pd.Timestamp(overall_end)
    months=[]
    cur=start
    while cur<=end:
        a,b=month_bounds(cur);b=min(b,end)
        tr=known_training(df,a)
        tg=target_frame(df,a,b,overall_end)
        if len(tr)>=500 and len(tg):
            models=v3.fit_models(tr)
            trc=v3.predict_components(models,tr);tgc=v3.predict_components(models,tg)
            bl=tuple(policy["blend"])
            trs=v3.score_frame(tr,trc,trc[2],bl);tgs=v3.score_frame(tg,tgc,trc[2],bl)
            thr=float(np.quantile(trs.score100,policy["q"]))
            ev=select_scored(tgs,thr,policy["top_per_day"]).copy()
            ev["model_month"]=a.strftime("%Y-%m")
            months.append((a,ev,sanitize_stats(ev),thr,len(tr),len(tg)))
        cur=cur+pd.offsets.MonthBegin(1)
    if not months:return pd.DataFrame(),[]
    combined=pd.concat([x[1] for x in months],ignore_index=True)
    # monthly chunks can repeat same symbol within 5BD across boundary; enforce global cooldown.
    combined=v3.cooldown(combined,5) if hasattr(v3,"cooldown") else combined
    meta=[{"month":a.strftime("%Y-%m"),"stats":st,"threshold":thr,"train_n":tn,"candidate_n":cn} for a,_,st,thr,tn,cn in months]
    return combined,meta

def global_cooldown(events,days=5):
    if events.empty:return events
    keep=[]
    for _,g in events.sort_values(["symbol","date","score100"],ascending=[True,True,False]).groupby("symbol",sort=False):
        last_date=None
        for idx,r in g.iterrows():
            d=pd.Timestamp(r.date)
            if last_date is None or (d-last_date).days>=days:
                keep.append(idx);last_date=d
    return events.loc[keep].sort_values(["date","score100"],ascending=[True,False]).reset_index(drop=True)

def common_stats(events):
    c=events[(events.date>="2026-03-05")&(events.date<="2026-09-09")]
    return v3.stats(c)

def report(r):
    h=r["holdout"];c=r["common_period"];cur=r["current_reference"]
    lines=["# No-TV V8 — Monthly Adaptive Stable","",
      f"- Full holdout: {r['holdout_start']} ～ {r['end_date']}",
      "- Current-comparable window: 2026-03-05 ～ 2026-09-09","",
      "|方式|期間|n|5BD平均|Robust平均|勝率|+10%Hit率|","|---|---|---:|---:|---:|---:|---:|",
      f"|現行4H Stable★6|3/5〜9/9|{cur['n']}|{cur['avg']*100:+.1f}%|-|{cur['wr']*100:.1f}%|{cur['target_rate']*100:.1f}%|",
      f"|V8 Adaptive|1年|{h['n']}|{h['avg']*100:+.1f}%|{h['robust_avg']*100:+.1f}%|{h['wr']*100:.1f}%|{h['target_rate']*100:.1f}%|",
      f"|**V8 Adaptive**|**3/5〜9/9**|**{c['n']}**|**{c['avg']*100:+.1f}%**|**{c['robust_avg']*100:+.1f}%**|**{c['wr']*100:.1f}%**|**{c['target_rate']*100:.1f}%**|","",
      f"- policy fixed from pre-holdout pseudo-future months: {json.dumps(r['policy'],ensure_ascii=False)}","",
      "## Monthly walk-forward"]
    for m in r["monthly"]:
        s=m["stats"];lines.append(f"- {m['month']}: n={s['n']} avg={s['avg']*100:+.1f}% wr={s['wr']*100:.1f}% train={m['train_n']}")
    lines+=["","- 各月開始時に、その時点で5BD結果が確定済みの過去データだけで再学習。","- policy(blend/q/top/day)は最終Holdoutより前の3 pseudo-future月だけで固定。","- TradingView不使用。"]
    return "\n".join(lines)

def run(args):
    endd=date.fromisoformat(args.end_date);start=(endd-timedelta(days=args.history_days)).isoformat();hold=(endd-timedelta(days=args.holdout_days)).isoformat()
    _,issues=fetch_jpx_issues_fixed();issues=[i for i in issues if "優先" not in i.name]
    if args.max_issues:issues=issues[:args.max_issues]
    frames=[];errs={};ok=0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
      fut={ex.submit(v3.fetch_one,i,start,args.end_date):i for i in issues}
      for n,f in enumerate(as_completed(fut),1):
        try:fr,er=f.result()
        except Exception as e:fr,er=None,type(e).__name__
        if fr is not None and not fr.empty:frames.append(fr);ok+=1
        elif er:errs[er]=errs.get(er,0)+1
        if n%500==0:print(f"progress {n}/{len(issues)} frames={len(frames)}",flush=True)
    if not frames:raise RuntimeError("no V8 candidates")
    df=v3.add_cross_sectional_ranks(pd.concat(frames,ignore_index=True)).sort_values(["date","symbol","bar_index"]).reset_index(drop=True)
    policy=choose_policy(df,hold,args.end_date)
    if policy is None:raise RuntimeError("no robust rolling policy")
    ev,monthly=rolling_holdout(df,policy,hold,args.end_date)
    ev=global_cooldown(ev,5)
    result={"generated_at_jst":core.now_jst().isoformat(timespec="seconds"),"history_start":start,"holdout_start":hold,"end_date":args.end_date,
      "universe_count":len(issues),"yahoo_ok":ok,"errors":errs,"candidate_rows":len(df),"current_reference":CURRENT,
      "policy":{k:v for k,v in policy.items() if k!="rank"},"holdout":v3.stats(ev),"common_period":common_stats(ev),"monthly":monthly,
      "top_events":ev.sort_values("score100",ascending=False).head(100)[["date","symbol","name","model_month","score100","perf_5bd"]].to_dict("records")}
    return result

def main():
    today=core.now_jst().date();p=argparse.ArgumentParser()
    p.add_argument("--end-date",type=parse_date,default=today.isoformat());p.add_argument("--history-days",type=int,default=1825);p.add_argument("--holdout-days",type=int,default=365)
    p.add_argument("--max-workers",type=int,default=32);p.add_argument("--max-issues",type=int);p.add_argument("--output-dir",default="reports_no_tv_v8")
    a=p.parse_args();r=run(a);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"comparison_v8.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"comparison_v8.md").write_text(report(r),encoding="utf-8")
    print(json.dumps({"policy":r["policy"],"holdout":r["holdout"],"common_period":r["common_period"]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
