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


def daily_state(hourly):
    d=hourly.groupby("date",sort=True).agg(open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum")).reset_index()
    c=d.close.astype(float);h=d.high.astype(float);l=d.low.astype(float)
    ema12=c.ewm(span=12,adjust=False,min_periods=12).mean()
    ema25=c.ewm(span=25,adjust=False,min_periods=25).mean()
    ema26=c.ewm(span=26,adjust=False,min_periods=26).mean()
    macd=ema12-ema26
    sig=macd.ewm(span=9,adjust=False,min_periods=9).mean()

    # current partial daily barに適用するため「前日終了時点」の状態を持つ
    st=pd.DataFrame({"date":d.date})
    st["prev_close"]=c.shift(1)
    st["prev_volume"]=d.volume.shift(1)
    st["prev_ema12"]=ema12.shift(1)
    st["prev_ema25"]=ema25.shift(1)
    st["prev_ema26"]=ema26.shift(1)
    st["prev_macd_sig"]=sig.shift(1)
    st["prev13_high"]=h.shift(1).rolling(13,min_periods=13).max()
    st["prev13_low"]=l.shift(1).rolling(13,min_periods=13).min()
    st["prev19_sum"]=c.shift(1).rolling(19,min_periods=19).sum()
    st["prev19_sqsum"]=(c.shift(1)**2).rolling(19,min_periods=19).sum()
    st["pre_down3"]=(c.shift(1)<c.shift(2))&(c.shift(2)<c.shift(3))
    st["day_index"]=np.arange(len(st))
    for n in (5,10,20,40):
        st[f"future_close_{n}"]=c.shift(-n)
        st[f"exit_date_{n}"]=d.date.shift(-n)
    return st


def build(issue,chart,start,end):
    df=h1.parse_1h_chart(chart)
    if df is None:return None

    # 当日partial OHLC（各1H barまで）
    df["day_open"]=df.groupby("date",sort=False).open.transform("first")
    df["day_high_sofar"]=df.groupby("date",sort=False).high.cummax()
    df["day_low_sofar"]=df.groupby("date",sort=False).low.cummin()

    st=daily_state(df)
    df=df.merge(st,on="date",how="left")
    c=df.close.astype(float)

    # 日足EMAを「この1H barで日足が終わった」と仮定して更新
    def ema_asof(prev,n):
        a=2/(n+1)
        return a*c+(1-a)*prev
    df["d_ema12_asof"]=ema_asof(df.prev_ema12,12)
    df["d_ema25_asof"]=ema_asof(df.prev_ema25,25)
    df["d_ema26_asof"]=ema_asof(df.prev_ema26,26)
    df["d_macd_asof"]=df.d_ema12_asof-df.d_ema26_asof
    a9=2/(9+1)
    df["d_macd_sig_asof"]=a9*df.d_macd_asof+(1-a9)*df.prev_macd_sig
    df["d_macd_hist_asof"]=df.d_macd_asof-df.d_macd_sig_asof

    # Stoch14: current partial day + previous13 completed days
    hh=np.maximum(df.day_high_sofar,df.prev13_high)
    ll=np.minimum(df.day_low_sofar,df.prev13_low)
    df["d_stoch_asof"]=(c-ll)/(hh-ll).replace(0,np.nan)

    # BB20: current partial close + previous19 completed closes
    sm=df.prev19_sum+c
    sq=df.prev19_sqsum+c*c
    mean=sm/20
    var=(sq/20-mean*mean).clip(lower=0)
    sd=np.sqrt(var)
    lower=mean-2*sd;upper=mean+2*sd
    df["d_bbpos_asof"]=(c-lower)/(upper-lower).replace(0,np.nan)

    df["s_ema25"]=(c>df.d_ema25_asof)
    df["s_macdpos"]=df.d_macd_hist_asof>0
    df["s_stoch75"]=df.d_stoch_asof>=.75
    df["s_bb80"]=df.d_bbpos_asof>=.80
    df["s_pre_down3"]=df.pre_down3.fillna(False)
    df["s_gapup"]=df.day_open>df.prev_close
    score_cols=["s_ema25","s_macdpos","s_stoch75","s_bb80","s_pre_down3","s_gapup"]
    df["stable_score"]=df[score_cols].sum(axis=1)

    # 1H天底型state trigger
    bottom,_,reason=v2.pine_state_signals(df.close.to_numpy(),df.high.to_numpy(),df.low.to_numpy(),EXACT)
    df["pine_bottom"]=bottom
    df["pine_reason"]=reason
    df["bar_index"]=np.arange(len(df))
    df["base"]=(df.prev_close<=1000)&(df.prev_volume>=10000)&(df.volume>=5000)

    for n in (5,10,20,40):
        df[f"perf_{n}bd"]=df[f"future_close_{n}"]/c-1
        df.rename(columns={f"exit_date_{n}":f"exit_date_{n}bd"},inplace=True)

    use=(df.date>=start)&(df.date<=end)&df.base&df.pine_bottom
    cols=["ts","date","day_index","bar_index","close","volume","stable_score","pine_reason",
          "perf_5bd","perf_10bd","perf_20bd","perf_40bd","exit_date_5bd","exit_date_10bd","exit_date_20bd","exit_date_40bd"]+score_cols+[
          "d_ema25_asof","d_macd_hist_asof","d_stoch_asof","d_bbpos_asof"
    ]
    out=df.loc[use,cols].copy()
    out["symbol"]=issue.code;out["name"]=issue.name
    return out


def fetch_one(issue,start,end):
    cfg=core.ScreeningConfig(yahoo_range="730d",yahoo_interval="1h")
    chart,err=core.fetch_chart(issue,cfg)
    if err:return None,err
    fr=build(issue,chart or {},start,end)
    return fr,None if fr is not None else "too_few_rows"


def stats(e):
    if e.empty:return {"n":0,"avg":0.0,"robust_avg":0.0,"wr":0.0,"hits":0,"target_rate":0.0}
    x=e.perf_5bd.dropna().to_numpy(float)
    if not len(x):return {"n":0,"avg":0.0,"robust_avg":0.0,"wr":0.0,"hits":0,"target_rate":0.0}
    dec=x[x!=0];wr=float(np.mean(dec>0)) if len(dec) else 0
    if len(x)>=10:
        lo,hi=np.quantile(x,[.05,.95]);ra=float(np.mean(np.clip(x,lo,hi)))
    else:ra=float(np.mean(x))
    hits=int(np.sum(x>=.10))
    return {"n":len(x),"avg":float(np.mean(x)),"robust_avg":ra,"wr":wr,"hits":hits,"target_rate":float(hits/len(x))}


def cooldown(e,days=5):
    if e.empty:return e
    keep=[]
    for _,g in e.sort_values(["symbol","day_index","stable_score"],ascending=[True,True,False]).groupby("symbol",sort=False):
        # 同一日の複数BOTTOMはStable score最大、同点なら遅いbarを採用
        g=g.sort_values(["day_index","stable_score","bar_index"],ascending=[True,False,False]).groupby("day_index",sort=False).head(1)
        last=-10**9
        for idx,row in g.iterrows():
            di=int(row.day_index)
            if di-last>=days:keep.append(idx);last=di
    return e.loc[keep]


def select(e,min_score,topday):
    x=e[e.stable_score>=min_score].copy()
    x=x.sort_values(["date","stable_score","bar_index"],ascending=[True,False,False]).groupby("date",sort=False).head(topday)
    return cooldown(x,5)


def choose(valid):
    best=None
    # ★6を中心に、5/6もValidationのみで比較
    for smin in (4,5,6):
        for k in (1,2,3,5):
            st=stats(select(valid,smin,k))
            if st["n"]<15 or st["n"]>220 or st["robust_avg"]<=0:continue
            rank=.5*st["wr"]+3*st["robust_avg"]+.5*st["target_rate"]+.01*np.log1p(st["n"])
            if best is None or rank>best["rank"]:
                best={"min_stable_score":smin,"top_per_day":k,"valid":st,"rank":float(rank)}
    return best or {"min_stable_score":6,"top_per_day":3,"valid":stats(select(valid,6,3)),"rank":None}


def report(r):
    cur=r["current_reference"];fixed=r["holdout"]["asof_stable6"];tuned=r["holdout"]["asof_tuned"]
    return "\n".join([
        "# No-TV 1H + As-Of Daily Stable",
        "",
        f"- Holdout: {r['holdout_start']} ～ {r['end_date']}",
        f"- Universe: {r['universe_count']} / Yahoo成功: {r['yahoo_ok']}",
        "",
        "|方式|n|5BD平均|Robust平均|勝率|+10%Hit|Hit率|",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"|現行4H Stable★6|{cur['n']}|{cur['avg']*100:+.1f}%|-|{cur['wr']*100:.1f}%|{cur['hits']}|{cur['target_rate']*100:.1f}%|",
        f"|1H Pine + **As-of Stable★6固定**|{fixed['n']}|{fixed['avg']*100:+.1f}%|{fixed['robust_avg']*100:+.1f}%|{fixed['wr']*100:.1f}%|{fixed['hits']}|{fixed['target_rate']*100:.1f}%|",
        f"|1H Pine + As-of Stable Validation選択|{tuned['n']}|{tuned['avg']*100:+.1f}%|{tuned['robust_avg']*100:+.1f}%|{tuned['wr']*100:.1f}%|{tuned['hits']}|{tuned['target_rate']*100:.1f}%|",
        "",
        f"- policy: {json.dumps(r['policy'],ensure_ascii=False)}",
        "- 各1H bar時点までの当日OHLCだけで日足EMA/MACD/Stoch/BBを再計算。後続1H barは使用しない。",
        "- TradingView不使用。",
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
            if fr is not None and not fr.empty:frames.append(fr);ok+=1
            elif er:errs[er]=errs.get(er,0)+1
            if n%250==0:print(f"progress {n}/{len(issues)}",flush=True)
    if not frames:raise RuntimeError("no asof signals")
    df=pd.concat(frames,ignore_index=True).sort_values(["date","symbol","bar_index"]).reset_index(drop=True)

    pre=sorted(df.loc[df.date<hold,"date"].unique());cut=int(len(pre)*.70)
    vs,ve=pre[cut],pre[-1]
    valid=df[(df.date>=vs)&(df.date<=ve)&df.exit_date_5bd.notna()&(df.exit_date_5bd<=ve)]
    ho=df[(df.date>=hold)&(df.date<=args.end_date)&df.exit_date_5bd.notna()&(df.exit_date_5bd<=args.end_date)]
    pol=choose(valid)
    fixed=select(ho,6,5)
    tuned=select(ho,pol["min_stable_score"],pol["top_per_day"])
    return {
        "generated_at_jst":core.now_jst().isoformat(timespec="seconds"),
        "history_start":start,"holdout_start":hold,"end_date":args.end_date,
        "universe_count":len(issues),"yahoo_ok":ok,"errors":errs,"signal_rows":len(df),
        "current_reference":CURRENT,"policy":{k:v for k,v in pol.items() if k!="rank"},
        "holdout":{"asof_stable6":stats(fixed),"asof_tuned":stats(tuned)},
        "top_events":tuned.sort_values(["stable_score","date"],ascending=[False,False]).head(100).to_dict("records"),
    }


def main():
    today=core.now_jst().date();p=argparse.ArgumentParser()
    p.add_argument("--end-date",type=parse_date,default=today.isoformat());p.add_argument("--history-days",type=int,default=720);p.add_argument("--holdout-days",type=int,default=365)
    p.add_argument("--max-workers",type=int,default=20);p.add_argument("--max-issues",type=int);p.add_argument("--output-dir",default="reports_no_tv_1h_asof")
    a=p.parse_args();r=run(a);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"comparison_1h_asof.json").write_text(json.dumps(r,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    (out/"comparison_1h_asof.md").write_text(report(r),encoding="utf-8")
    print(json.dumps({"holdout":r["holdout"],"policy":r["policy"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
