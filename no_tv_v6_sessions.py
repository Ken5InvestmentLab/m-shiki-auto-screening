from __future__ import annotations

import argparse, json, math
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
    except ValueError as e:raise argparse.ArgumentTypeError("YYYY-MM-DD") from e

def ema_arr(vals,p):
    if len(vals)<p:return [None]*len(vals)
    k=2/(p+1);res=[None]*(p-1);e=sum(vals[:p])/p;res.append(e)
    for v in vals[p:]:
        e=v*k+e*(1-k);res.append(e)
    return res

def exact_stable_features(daily):
    if len(daily)<30:return None
    C=[float(x["close"]) for x in daily];H=[float(x["high"]) for x in daily]
    L=[float(x["low"]) for x in daily];O=[float(x["open"]) for x in daily]
    last=len(C)-1;lc=C[-1];lo=O[-1]
    e25=ema_arr(C,25)[-1]
    if e25 is None:return None
    m12=ema_arr(C,12);m26=ema_arr(C,26)
    ml=[a-b for a,b in zip(m12,m26) if a is not None and b is not None]
    sig=ema_arr(ml,9)
    macdpos=bool(ml and sig and sig[-1] is not None and ml[-1]>sig[-1])
    lo14=min(L[max(0,last-13):last+1]);hi14=max(H[max(0,last-13):last+1])
    stoch=(lc-lo14)/(hi14-lo14)*100 if hi14>lo14 else 50
    bb=C[max(0,last-19):last+1];bm=sum(bb)/len(bb);bs=(sum((x-bm)**2 for x in bb)/len(bb))**.5
    bbpct=max(0,min(1,(lc-(bm-2*bs))/(4*bs) if bs>0 else .5))
    pre_down3=last>=3 and C[last-1]<C[last-2] and C[last-2]<C[last-3]
    gap_up=last>0 and lo>C[last-1]
    return {
        "ema25":lc>e25,"macdpos":macdpos,"stoch75":stoch>=75,
        "bb80":bbpct>=.80,"pre_down3":pre_down3,"gap_up":gap_up,
        "_stoch":stoch,"_bbpct":bbpct,
    }

def synthetic_sessions(hourly):
    x=hourly.copy()
    # TV/productionは09:00・13:00の2本/日を前提。
    # Yahoo 1hのJST時刻で13:00未満を第1セッション、13:00以降を第2セッションへ集約。
    mins=x.ts.dt.hour*60+x.ts.dt.minute
    x["session"]=np.where(mins<13*60,9,13)
    g=x.groupby(["date","session"],sort=True).agg(
        ts=("ts","last"),open=("open","first"),high=("high","max"),low=("low","min"),
        close=("close","last"),volume=("volume","sum")
    ).reset_index()
    return g

def daily_completed(hourly):
    return hourly.groupby("date",sort=True).agg(
        open=("open","first"),high=("high","max"),low=("low","min"),
        close=("close","last"),volume=("volume","sum")
    ).reset_index()

def build_asof_daily(completed, sessions, upto_idx):
    cur=sessions.iloc[upto_idx]
    cur_date=cur.date
    past=completed[completed.date<cur_date]
    rows=past[["date","open","high","low","close","volume"]].to_dict("records")
    today=sessions.iloc[:upto_idx+1]
    today=today[today.date==cur_date]
    rows.append({
        "date":cur_date,
        "open":float(today.iloc[0].open),
        "high":float(today.high.max()),
        "low":float(today.low.min()),
        "close":float(today.iloc[-1].close),
        "volume":float(today.volume.sum()),
    })
    return rows

def build(issue,chart,start,end):
    hourly=h1.parse_1h_chart(chart)
    if hourly is None:return None
    sessions=synthetic_sessions(hourly)
    completed=daily_completed(hourly)
    if len(sessions)<100:return None

    c=sessions.close.to_numpy(float);h=sessions.high.to_numpy(float);l=sessions.low.to_numpy(float)
    bottom,_,reason=v2.pine_state_signals(c,h,l,EXACT)
    sessions["bottom"]=bottom;sessions["reason"]=reason
    sessions["bar_index"]=np.arange(len(sessions))

    # completed day lookup / future closes
    d=completed.copy();d["day_index"]=np.arange(len(d))
    for n in (5,10,20,40):
        d[f"future_close_{n}"]=d.close.shift(-n);d[f"exit_date_{n}bd"]=d.date.shift(-n)
    daymap=d.set_index("date")

    rows=[]
    for i,row in sessions.iterrows():
        if not bool(row.bottom):continue
        dt=str(row.date)
        if dt<start or dt>end or dt not in daymap.index:continue
        di=int(daymap.loc[dt,"day_index"])
        if di<1:continue
        prev=d.iloc[di-1]
        # strict base filters
        if not (float(prev.close)<=1000 and float(prev.volume)>=10000 and float(row.volume)>=5000):
            continue
        feats=exact_stable_features(build_asof_daily(completed,sessions,i))
        if feats is None:continue
        rec={
            "date":dt,"ts":row.ts,"session":int(row.session),"bar_index":int(row.bar_index),
            "day_index":di,"symbol":issue.code,"name":issue.name,
            "entry":float(row.close),"session_volume":float(row.volume),
            "stable_score":sum(int(feats[k]) for k in ("ema25","macdpos","stoch75","bb80","pre_down3","gap_up")),
            "reason":int(row.reason),**feats,
        }
        for n in (5,10,20,40):
            fc=daymap.loc[dt,f"future_close_{n}"]; ex=daymap.loc[dt,f"exit_date_{n}bd"]
            rec[f"perf_{n}bd"]=float(fc/row.close-1) if pd.notna(fc) else np.nan
            rec[f"exit_date_{n}bd"]=ex if pd.notna(ex) else ""
        rows.append(rec)
    return pd.DataFrame(rows)

def fetch_one(issue,start,end):
    cfg=core.ScreeningConfig(yahoo_range="730d",yahoo_interval="1h")
    chart,err=core.fetch_chart(issue,cfg)
    if err:return None,err
    fr=build(issue,chart or {},start,end)
    return fr,None if fr is not None else "too_few_rows"

def stats(e):
    if e.empty:return {"n":0,"avg":0.,"robust_avg":0.,"wr":0.,"hits":0,"target_rate":0.}
    x=pd.to_numeric(e.perf_5bd,errors="coerce").dropna().to_numpy(float)
    if not len(x):return {"n":0,"avg":0.,"robust_avg":0.,"wr":0.,"hits":0,"target_rate":0.}
    dec=x[x!=0];wr=float(np.mean(dec>0)) if len(dec) else 0.
    if len(x)>=10:
        lo,hi=np.quantile(x,[.05,.95]);rob=float(np.mean(np.clip(x,lo,hi)))
    else:rob=float(np.mean(x))
    hit=int(np.sum(x>=.10))
    return {"n":int(len(x)),"avg":float(np.mean(x)),"robust_avg":rob,"wr":wr,"hits":hit,"target_rate":float(hit/len(x))}

def cooldown(e,days=5):
    if e.empty:return e
    keep=[]
    for _,g in e.sort_values(["symbol","day_index","stable_score","bar_index"],ascending=[True,True,False,False]).groupby("symbol",sort=False):
        g=g.groupby("day_index",sort=False).head(1);last=-10**9
        for idx,r in g.iterrows():
            di=int(r.day_index)
            if di-last>=days:keep.append(idx);last=di
    return e.loc[keep]

def select(e,min_score,topday,session_mode="all"):
    x=e.copy()
    if session_mode!="all":x=x[x.session==int(session_mode)]
    x=x[x.stable_score>=min_score]
    if x.empty:return x
    x=x.sort_values(["date","stable_score","bar_index"],ascending=[True,False,False]).groupby("date",sort=False).head(topday)
    return cooldown(x)

def choose(valid):
    best=None
    for sess in ("all",9,13):
      for s in (4,5,6):
       for k in (1,2,3,5):
        st=stats(select(valid,s,k,sess))
        if st["n"]<18 or st["n"]>220 or st["robust_avg"]<=0:continue
        rank=.55*st["wr"]+3.2*st["robust_avg"]+.55*st["target_rate"]+.01*math.log1p(st["n"])
        if best is None or rank>best["rank"]:
            best={"session":sess,"min_score":s,"top_per_day":k,"valid":st,"rank":rank}
    return best

def report(r):
    cur=r["current_reference"];f=r["holdout"]["stable6_fixed"];s=r["holdout"]["selected"]
    return "\n".join([
      "# No-TV V6 — Yahoo 1H → 09/13 Synthetic Sessions",
      "",f"- Holdout: {r['holdout_start']} ～ {r['end_date']}",
      f"- Universe: {r['universe_count']} / Yahoo成功: {r['yahoo_ok']}","",
      "|方式|n|5BD平均|Robust平均|勝率|+10%Hit|Hit率|","|---|---:|---:|---:|---:|---:|---:|",
      f"|現行4H Stable★6|{cur['n']}|{cur['avg']*100:+.1f}%|-|{cur['wr']*100:.1f}%|{cur['hits']}|{cur['target_rate']*100:.1f}%|",
      f"|V6 09/13 Session + Stable★6|{f['n']}|{f['avg']*100:+.1f}%|{f['robust_avg']*100:+.1f}%|{f['wr']*100:.1f}%|{f['hits']}|{f['target_rate']*100:.1f}%|",
      f"|V6 Validation-selected|{s['n']}|{s['avg']*100:+.1f}%|{s['robust_avg']*100:+.1f}%|{s['wr']*100:.1f}%|{s['hits']}|{s['target_rate']*100:.1f}%|",
      "",f"- policy: {json.dumps(r['policy'],ensure_ascii=False)}",
      "- Yahoo 1hをJST 13:00未満/以降の2セッションへ集約。",
      "- Stable6はproduction optimizerと同じSMA-seed EMA / MACD / Stoch / BB / pre_down3 / gap_up式。",
      "- シグナル時点までの当日セッションだけで日足を再構成。未来参照なし。",
      "- TradingView不使用。"
    ])

def run(args):
    endd=date.fromisoformat(args.end_date);start=(endd-timedelta(days=args.history_days)).isoformat();hold=(endd-timedelta(days=args.holdout_days)).isoformat()
    _,issues=fetch_jpx_issues_fixed();issues=[i for i in issues if "優先" not in i.name]
    if args.max_issues:issues=issues[:args.max_issues]
    frames=[];errs={};ok=0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
      fut={ex.submit(fetch_one,i,start,args.end_date):i for i in issues}
      for n,f in enumerate(as_completed(fut),1):
        try:fr,er=f.result()
        except Exception as e:fr,er=None,type(e).__name__
        if fr is not None:
            ok+=1
            if not fr.empty:frames.append(fr)
        elif er:errs[er]=errs.get(er,0)+1
        if n%250==0:print(f"progress {n}/{len(issues)} frames={len(frames)}",flush=True)
    if not frames:raise RuntimeError("no V6 signals")
    df=pd.concat(frames,ignore_index=True).sort_values(["date","symbol","bar_index"]).reset_index(drop=True)
    pre=sorted(df.loc[df.date<hold,"date"].unique())
    if len(pre)<80:raise RuntimeError("not enough pre-holdout dates")
    cut=int(len(pre)*.70);vs,ve=pre[cut],pre[-1]
    valid=df[(df.date>=vs)&(df.date<=ve)&df.exit_date_5bd.notna()&(df.exit_date_5bd<=ve)]
    ho=df[(df.date>=hold)&(df.date<=args.end_date)&df.exit_date_5bd.notna()&(df.exit_date_5bd<=args.end_date)]
    pol=choose(valid)
    fixed=select(ho,6,5,"all")
    selected=select(ho,pol["min_score"],pol["top_per_day"],pol["session"]) if pol else ho.iloc[0:0]
    return {"generated_at_jst":core.now_jst().isoformat(timespec="seconds"),"history_start":start,"holdout_start":hold,"end_date":args.end_date,
            "universe_count":len(issues),"yahoo_ok":ok,"errors":errs,"signal_rows":len(df),"current_reference":CURRENT,
            "policy":({k:v for k,v in pol.items() if k!="rank"} if pol else None),
            "holdout":{"stable6_fixed":stats(fixed),"selected":stats(selected)},
            "by_session":{"9":stats(select(ho,6,5,9)),"13":stats(select(ho,6,5,13))}}

def main():
    today=core.now_jst().date();p=argparse.ArgumentParser()
    p.add_argument("--end-date",type=parse_date,default=today.isoformat());p.add_argument("--history-days",type=int,default=720);p.add_argument("--holdout-days",type=int,default=365)
    p.add_argument("--max-workers",type=int,default=20);p.add_argument("--max-issues",type=int);p.add_argument("--output-dir",default="reports_no_tv_v6")
    a=p.parse_args();r=run(a);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"comparison_v6.json").write_text(json.dumps(r,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    (out/"comparison_v6.md").write_text(report(r),encoding="utf-8")
    print(json.dumps({"holdout":r["holdout"],"by_session":r["by_session"],"policy":r["policy"]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
