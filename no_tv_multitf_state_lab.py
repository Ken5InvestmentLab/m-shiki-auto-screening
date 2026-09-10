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
import independent_daily_lab_v2 as v2
import no_tv_1h_lab as h1

CURRENT={"n":55,"avg":.066,"wr":.564,"hits":10,"target_rate":10/55}
EXACT=next(p for p in v2.PINE_PROFILES if p["id"]=="exact")

def parse_date(v):
    try:return date.fromisoformat(v).isoformat()
    except ValueError as e:raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from e

def daily_ctx(hourly):
    d=hourly.groupby("date",sort=True).agg(open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum")).reset_index()
    c=d.close.astype(float);h=d.high.astype(float);l=d.low.astype(float);o=d.open.astype(float)
    ema12=c.ewm(span=12,adjust=False,min_periods=12).mean()
    ema25=c.ewm(span=25,adjust=False,min_periods=25).mean()
    ema26=c.ewm(span=26,adjust=False,min_periods=26).mean()
    macd=ema12-ema26;sig=macd.ewm(span=9,adjust=False,min_periods=9).mean()
    lo14=l.rolling(14,min_periods=14).min();hi14=h.rolling(14,min_periods=14).max()
    stoch=(c-lo14)/(hi14-lo14).replace(0,np.nan)
    bm=c.rolling(20,min_periods=20).mean();bs=c.rolling(20,min_periods=20).std(ddof=0)
    bb=(c-(bm-2*bs))/(4*bs.replace(0,np.nan))
    out=pd.DataFrame({"date":d.date})
    out["prev_close"]=c.shift(1);out["prev_volume"]=d.volume.shift(1)
    out["s_ema25"]=(c>ema25).shift(1)
    out["s_macdpos"]=((macd-sig)>0).shift(1)
    out["s_stoch75"]=(stoch>=.75).shift(1)
    out["s_bb80"]=(bb>=.80).shift(1)
    out["s_pre_down3"]=((c.shift(1)<c.shift(2))&(c.shift(2)<c.shift(3))).shift(1)
    out["s_gapup"]=(o>c.shift(1)).shift(1)
    out["day_index"]=np.arange(len(out))
    for n in (5,10,20,40):
        out[f"future_close_{n}"]=c.shift(-n);out[f"exit_date_{n}bd"]=d.date.shift(-n)
    return out

def aggregate_tf(hourly,hours):
    x=hourly.copy()
    # TSE intraday session begins around 09:00 JST. Build groups within each market date,
    # using chronological bar order rather than assuming TradingView boundaries.
    x["seq"]=x.groupby("date",sort=False).cumcount()
    x["grp"]=(x.seq//hours).astype(int)
    g=x.groupby(["date","grp"],sort=False).agg(
        ts=("ts","last"),open=("open","first"),high=("high","max"),low=("low","min"),
        close=("close","last"),volume=("volume","sum"),seq_end=("seq","last")
    ).reset_index()
    return g

def frame_for_tf(issue,hourly,hours,start,end):
    ctx=daily_ctx(hourly)
    bars=aggregate_tf(hourly,hours).merge(ctx,on="date",how="left")
    c=bars.close.astype(float);h=bars.high.astype(float);l=bars.low.astype(float)
    bottom,_,reason=v2.pine_state_signals(c.to_numpy(),h.to_numpy(),l.to_numpy(),EXACT)
    bars["bottom"]=bottom;bars["reason"]=reason;bars["bar_index"]=np.arange(len(bars))
    bars["stable_score"]=bars[["s_ema25","s_macdpos","s_stoch75","s_bb80","s_pre_down3","s_gapup"]].fillna(False).sum(axis=1)
    bars["base"]=(bars.prev_close<=1000)&(bars.prev_volume>=10000)&(bars.volume>=5000)
    for n in (5,10,20,40):
        bars[f"perf_{n}bd"]=bars[f"future_close_{n}"]/c-1
    use=(bars.date>=start)&(bars.date<=end)&bars.base&bars.bottom
    cols=["ts","date","day_index","bar_index","close","volume","stable_score","reason",
          "perf_5bd","perf_10bd","perf_20bd","perf_40bd",
          "exit_date_5bd","exit_date_10bd","exit_date_20bd","exit_date_40bd"]
    out=bars.loc[use,cols].copy();out["symbol"]=issue.code;out["name"]=issue.name;out["tf_hours"]=hours
    return out

def fetch_one(issue,start,end):
    cfg=core.ScreeningConfig(yahoo_range="730d",yahoo_interval="1h")
    chart,err=core.fetch_chart(issue,cfg)
    if err:return [],err
    hourly=h1.parse_1h_chart(chart or {})
    if hourly is None:return [],"too_few_rows"
    return [frame_for_tf(issue,hourly,h,start,end) for h in (1,2,3,4)],None

def stats(e):
    if e.empty:return {"n":0,"avg":0.0,"robust_avg":0.0,"wr":0.0,"hits":0,"target_rate":0.0}
    a=e.perf_5bd.dropna().to_numpy(float)
    if not len(a):return {"n":0,"avg":0.0,"robust_avg":0.0,"wr":0.0,"hits":0,"target_rate":0.0}
    dec=a[a!=0];wr=float(np.mean(dec>0)) if len(dec) else 0.0
    if len(a)>=10:
        lo,hi=np.quantile(a,[.05,.95]);rob=float(np.mean(np.clip(a,lo,hi)))
    else:rob=float(np.mean(a))
    hits=int(np.sum(a>=.10))
    return {"n":len(a),"avg":float(np.mean(a)),"robust_avg":rob,"wr":wr,"hits":hits,"target_rate":float(hits/len(a))}

def cooldown(e,days=5):
    if e.empty:return e
    keep=[]
    for _,g in e.sort_values(["symbol","day_index","stable_score","bar_index"],ascending=[True,True,False,False]).groupby("symbol",sort=False):
        g=g.groupby("day_index",sort=False).head(1)
        last=-10**9
        for idx,row in g.iterrows():
            di=int(row.day_index)
            if di-last>=days:keep.append(idx);last=di
    return e.loc[keep]

def select(e,min_score=6,topday=5):
    x=e[e.stable_score>=min_score].copy()
    x=x.sort_values(["date","stable_score","bar_index"],ascending=[True,False,False]).groupby("date",sort=False).head(topday)
    return cooldown(x)

def choose(valid):
    best=None
    for tf in (1,2,3,4):
        z=valid[valid.tf_hours==tf]
        for smin in (4,5,6):
            for k in (1,2,3,5):
                st=stats(select(z,smin,k))
                if st["n"]<15 or st["n"]>220 or st["robust_avg"]<=0:continue
                rank=.5*st["wr"]+3*st["robust_avg"]+.5*st["target_rate"]+.01*np.log1p(st["n"])
                if best is None or rank>best["rank"]:
                    best={"tf_hours":tf,"min_score":smin,"top_per_day":k,"valid":st,"rank":float(rank)}
    return best

def report(r):
    lines=["# No-TV Multi-TF Stable Lab","",f"- Holdout: {r['holdout_start']} ～ {r['end_date']}",f"- Universe: {r['universe_count']} / Yahoo成功: {r['yahoo_ok']}","",
           "|方式|n|5BD平均|Robust平均|勝率|+10%Hit|Hit率|","|---|---:|---:|---:|---:|---:|---:|"]
    cur=r["current_reference"]
    lines.append(f"|現行4H Stable★6|{cur['n']}|{cur['avg']*100:+.1f}%|-|{cur['wr']*100:.1f}%|{cur['hits']}|{cur['target_rate']*100:.1f}%|")
    for tf,s in r["holdout_by_tf"].items():
        lines.append(f"|Yahoo {tf}H + Stable★6|{s['n']}|{s['avg']*100:+.1f}%|{s['robust_avg']*100:+.1f}%|{s['wr']*100:.1f}%|{s['hits']}|{s['target_rate']*100:.1f}%|")
    if r["policy"]:
        p=r["policy"];s=r["selected_holdout"]
        lines+=["",f"Validation選択: {p['tf_hours']}H / score>={p['min_score']} / top/day={p['top_per_day']}",
                f"Holdout: n={s['n']} avg={s['avg']*100:+.1f}% wr={s['wr']*100:.1f}%"]
    lines+=["","- 1HはYahoo原足、2/3/4HはYahoo 1Hから独自集約。TradingView境界は使用しない。","- 日足Stable条件は前日確定値。"]
    return "\n".join(lines)

def run(args):
    endd=date.fromisoformat(args.end_date);start=(endd-timedelta(days=args.history_days)).isoformat();hold=(endd-timedelta(days=args.holdout_days)).isoformat()
    _,issues=fetch_jpx_issues_fixed();issues=[i for i in issues if "優先" not in i.name]
    if args.max_issues:issues=issues[:args.max_issues]
    frames=[];errs={};ok=0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        fut={ex.submit(fetch_one,i,start,args.end_date):i for i in issues}
        for n,f in enumerate(as_completed(fut),1):
            try:arr,er=f.result()
            except Exception as e:arr,er=[],type(e).__name__
            if arr:
                ok+=1;frames.extend([x for x in arr if not x.empty])
            elif er:errs[er]=errs.get(er,0)+1
            if n%250==0:print(f"progress {n}/{len(issues)}",flush=True)
    if not frames:raise RuntimeError("no multitf signals")
    df=pd.concat(frames,ignore_index=True)
    pre=sorted(df.loc[df.date<hold,"date"].unique());cut=int(len(pre)*.70);vs,ve=pre[cut],pre[-1]
    valid=df[(df.date>=vs)&(df.date<=ve)&df.exit_date_5bd.notna()&(df.exit_date_5bd<=ve)]
    ho=df[(df.date>=hold)&(df.date<=args.end_date)&df.exit_date_5bd.notna()&(df.exit_date_5bd<=args.end_date)]
    by={}
    for tf in (1,2,3,4):by[str(tf)]=stats(select(ho[ho.tf_hours==tf],6,5))
    pol=choose(valid)
    sel=stats(select(ho[ho.tf_hours==pol["tf_hours"]],pol["min_score"],pol["top_per_day"])) if pol else stats(ho.iloc[0:0])
    return {"generated_at_jst":core.now_jst().isoformat(timespec="seconds"),"history_start":start,"holdout_start":hold,"end_date":args.end_date,
            "universe_count":len(issues),"yahoo_ok":ok,"errors":errs,"signal_rows":len(df),"current_reference":CURRENT,
            "holdout_by_tf":by,"policy":({k:v for k,v in pol.items() if k!="rank"} if pol else None),"selected_holdout":sel}

def main():
    today=core.now_jst().date();p=argparse.ArgumentParser()
    p.add_argument("--end-date",type=parse_date,default=today.isoformat());p.add_argument("--history-days",type=int,default=720);p.add_argument("--holdout-days",type=int,default=365)
    p.add_argument("--max-workers",type=int,default=20);p.add_argument("--max-issues",type=int);p.add_argument("--output-dir",default="reports_no_tv_multitf")
    a=p.parse_args();r=run(a);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"comparison_multitf.json").write_text(json.dumps(r,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    (out/"comparison_multitf.md").write_text(report(r),encoding="utf-8")
    print(json.dumps({"holdout_by_tf":r["holdout_by_tf"],"policy":r["policy"],"selected_holdout":r["selected_holdout"]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
