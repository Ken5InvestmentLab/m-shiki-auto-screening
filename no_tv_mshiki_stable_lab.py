from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import screen_big_money as core
from screen_entry import fetch_jpx_issues_fixed
import no_tv_daily_v3 as v3


CURRENT={"n":55,"avg":.066,"wr":.564,"hits":10,"target_rate":10/55}


def parse_date(v):
    try: return date.fromisoformat(v).isoformat()
    except ValueError as e: raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from e


def build(issue,chart,start,end):
    parsed=core.chart_to_arrays(chart,None)
    if parsed is None: return None
    dates,a=parsed
    o,h,l,c,v=a["open"],a["high"],a["low"],a["close"],a["volume"]
    if len(c)<220: return None

    raw,quality,tr,vr,_,_=core.calc_reactions(o,h,l,c,v)
    cfg=core.ScreeningConfig(yahoo_range="5y",yahoo_interval="1d")
    rows=[]
    for i in range(120,len(c)-5):
        d=str(dates[i])
        if d<start or d>end: continue
        # 現行候補の基本3条件も維持
        if not (c[i-1]<=1000 and v[i-1]>=10000 and v[i]>=5000):
            continue
        cand,err=core.make_candidate(issue,dates,o,h,l,c,v,raw,quality,tr,vr,i,cfg)
        if cand is None: continue
        rows.append({
            "date":d,"symbol":issue.code,"name":issue.name,"market":issue.market,
            "bar_index":i,"lane":cand.lane,"m_score":cand.score,
            "m_raw_pct":cand.raw_pct,"m_raw_to_max":cand.raw_to_max,
            "m_qual_pct":cand.qual_pct,"m_qual_to_max":cand.qual_to_max,
            "turnover_ratio":cand.turnover_ratio,"volume_ratio":cand.volume_ratio,
            "close_loc":cand.close_loc,"upper_wick":cand.upper_wick,
            "close":float(c[i]),"perf_5bd":float(c[i+5]/c[i]-1),
            "exit_date_5bd":str(dates[i+5]),
        })
    return pd.DataFrame(rows)


def fetch_one(issue,start,end):
    cfg=core.ScreeningConfig(yahoo_range="5y",yahoo_interval="1d")
    chart,err=core.fetch_chart(issue,cfg)
    if err: return None,err
    fr=build(issue,chart or {},start,end)
    return fr,None if fr is not None else "too_few_rows"


def stats(e):
    if e.empty: return {"n":0,"avg":0.0,"robust_avg":0.0,"wr":0.0,"hits":0,"target_rate":0.0}
    x=e.perf_5bd.dropna().to_numpy(float)
    if not len(x): return {"n":0,"avg":0.0,"robust_avg":0.0,"wr":0.0,"hits":0,"target_rate":0.0}
    dec=x[x!=0]; wr=float(np.mean(dec>0)) if len(dec) else 0.0
    if len(x)>=10:
        lo,hi=np.quantile(x,[.05,.95]); ra=float(np.mean(np.clip(x,lo,hi)))
    else: ra=float(np.mean(x))
    hit=int(np.sum(x>=.10))
    return {"n":len(x),"avg":float(np.mean(x)),"robust_avg":ra,"wr":wr,"hits":hit,"target_rate":float(hit/len(x))}


def cooldown(e,days=5):
    if e.empty: return e
    keep=[]
    for _,g in e.sort_values(["symbol","date","m_score"],ascending=[True,True,False]).groupby("symbol",sort=False):
        g=g.sort_values(["date","m_score"],ascending=[True,False])
        last=-10**9
        for idx,row in g.iterrows():
            bi=int(row.bar_index)
            if bi-last>=days:
                keep.append(idx); last=bi
    return e.loc[keep]


def select(e,topday,lane_mode,min_score=None):
    x=e.copy()
    if lane_mode!="all": x=x[x.lane==lane_mode]
    if min_score is not None: x=x[x.m_score>=min_score]
    x=x.sort_values(["date","m_score"],ascending=[True,False]).groupby("date",sort=False).head(topday)
    return cooldown(x,5)


def split(df,hold_start):
    pre=sorted(df.loc[df.date<hold_start,"date"].unique())
    cut=int(len(pre)*.75)
    return pre[0],pre[cut-1],pre[cut],pre[-1]


def choose(valid):
    best=None
    for lane in ("all","strong","quiet","watch"):
        base=valid if lane=="all" else valid[valid.lane==lane]
        if base.empty: continue
        thresholds=[None]+[float(q) for q in np.unique(np.quantile(base.m_score,[.50,.70,.80,.90,.95]))]
        for t in thresholds:
            for k in (1,2,3,5,10):
                s=stats(select(valid,k,lane,t))
                if s["n"]<20 or s["n"]>220 or s["robust_avg"]<=0: continue
                rank=.5*s["wr"]+3*s["robust_avg"]+.5*s["target_rate"]+.01*np.log1p(s["n"])
                if best is None or rank>best["rank"]:
                    best={"lane":lane,"threshold":t,"top_per_day":k,"valid":s,"rank":float(rank)}
    return best


def report(r):
    b=r["holdout"]["mshiki"]; cur=r["current_reference"]
    return "\n".join([
        "# No-TV M式 Candidate → Stable benchmark",
        "",
        f"- Holdout: {r['holdout_start']} ～ {r['end_date']}",
        f"- Universe: {r['universe_count']} / Yahoo成功: {r['yahoo_ok']}",
        "",
        "|方式|n|5BD平均|Robust平均|勝率|+10%Hit|Hit率|",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"|現行4H Stable★6|{cur['n']}|{cur['avg']*100:+.1f}%|-|{cur['wr']*100:.1f}%|{cur['hits']}|{cur['target_rate']*100:.1f}%|",
        f"|M式大口反応 Candidate|{b['n']}|{b['avg']*100:+.1f}%|{b['robust_avg']*100:+.1f}%|{b['wr']*100:.1f}%|{b['hits']}|{b['target_rate']*100:.1f}%|",
        "",
        f"- policy: {json.dumps(r['policy'],ensure_ascii=False)}",
        "- TradingView不使用。Yahoo日足＋現行M式の大口反応ロジックのみ。",
    ])


def run(args):
    endd=date.fromisoformat(args.end_date)
    start=(endd-timedelta(days=args.history_days)).isoformat()
    hold=(endd-timedelta(days=args.holdout_days)).isoformat()

    _,issues=fetch_jpx_issues_fixed(); issues=[i for i in issues if "優先" not in i.name]
    if args.max_issues: issues=issues[:args.max_issues]
    frames=[]; errs={}; ok=0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        fut={ex.submit(fetch_one,i,start,args.end_date):i for i in issues}
        for n,f in enumerate(as_completed(fut),1):
            try: fr,er=f.result()
            except Exception as e: fr,er=None,type(e).__name__
            if fr is not None:
                ok+=1
                if not fr.empty: frames.append(fr)
            elif er: errs[er]=errs.get(er,0)+1
            if n%500==0: print(f"progress {n}/{len(issues)} rows={sum(len(x) for x in frames)}",flush=True)
    if not frames: raise RuntimeError("no M-shiki candidates")
    df=pd.concat(frames,ignore_index=True).sort_values(["date","symbol"]).reset_index(drop=True)

    a,b,c,d=split(df,hold)
    valid=df[(df.date>=c)&(df.date<=d)&(df.exit_date_5bd<=d)]
    holdf=df[(df.date>=hold)&(df.date<=args.end_date)&(df.exit_date_5bd<=args.end_date)]
    pol=choose(valid)
    if pol is None:
        pol={"lane":"all","threshold":None,"top_per_day":3,"valid":stats(select(valid,3,"all",None)),"rank":None}
    ev=select(holdf,pol["top_per_day"],pol["lane"],pol["threshold"])
    return {
        "generated_at_jst":core.now_jst().isoformat(timespec="seconds"),
        "history_start":start,"holdout_start":hold,"end_date":args.end_date,
        "universe_count":len(issues),"yahoo_ok":ok,"errors":errs,"candidate_rows":len(df),
        "current_reference":CURRENT,"policy":{k:v for k,v in pol.items() if k!="rank"},
        "holdout":{"mshiki":stats(ev)},
        "top_events":ev.sort_values("m_score",ascending=False).head(100).to_dict("records"),
    }


def main():
    today=core.now_jst().date()
    p=argparse.ArgumentParser()
    p.add_argument("--end-date",type=parse_date,default=today.isoformat())
    p.add_argument("--history-days",type=int,default=1825)
    p.add_argument("--holdout-days",type=int,default=365)
    p.add_argument("--max-workers",type=int,default=32)
    p.add_argument("--max-issues",type=int)
    p.add_argument("--output-dir",default="reports_no_tv_mshiki")
    a=p.parse_args(); r=run(a); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"comparison_mshiki.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"comparison_mshiki.md").write_text(report(r),encoding="utf-8")
    print(json.dumps({"holdout":r["holdout"],"policy":r["policy"]},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
