from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import screen_big_money as core
from screen_entry import fetch_jpx_issues_fixed
import independent_daily_lab_v2 as v2


CURRENT_REFERENCE={"n":55,"avg":.066,"wr":.564,"hits":10,"target_rate":10/55}
EXACT_PROFILE=next(p for p in v2.PINE_PROFILES if p["id"]=="exact")

FEATURES=[
    "ret1b","ret2b","ret3b","ret6b","ret12b","ret24b",
    "body_pct","range_pct","close_loc","upper_wick","lower_wick",
    "atr_pct","rsi12","rsi14","bb_pos","bb_width","bb_width_rel60",
    "dist_ema25","dist_ema75","ema25_slope5","macd_hist_pct",
    "break20","break60","vol_ratio5","vol_ratio20","vol_ratio60","turnover_ratio20",
    "rv6","rv20","money_flow20","sin_time","cos_time","day_progress",
    "d_ret1","d_ret5","d_ret20","d_atr_pct","d_rsi14","d_bb_pos","d_bb_width",
    "d_dist_ema25","d_ema25_slope5","d_dd20","d_dd60","d_vol_ratio20",
    "tr_pine_exact","tr_breakout","tr_reversal","tr_volume_reversal","tr_squeeze",
    "tr_pullback","tr_trend_resume","tr_accumulation","tr_momentum",
    "xrank_ret6","xrank_vol20","xrank_atr","xrank_rsi12","xrank_break20","xrank_close_loc",
]

BLENDS={
    "balanced":(.45,.35,.20),
    "hit10":(.30,.50,.20),
    "positive":(.65,.15,.20),
    "return":(.35,.20,.45),
}


def parse_date(v):
    try: return date.fromisoformat(v).isoformat()
    except ValueError as e: raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from e


def parse_1h_chart(chart):
    n=min(len(chart.get(k,[])) for k in ("timestamp","open","high","low","close","volume"))
    rows=[]
    for i in range(n):
        vals=[chart["open"][i],chart["high"][i],chart["low"][i],chart["close"][i],chart["volume"][i]]
        if any(x is None for x in vals): continue
        o,h,l,c,vv=map(float,vals)
        if c<=0 or h<=0 or l<=0 or h<l or vv<0: continue
        ts=pd.to_datetime(int(chart["timestamp"][i]),unit="s",utc=True).tz_convert("Asia/Tokyo")
        rows.append((ts,o,h,l,c,vv))
    if len(rows)<200: return None
    df=pd.DataFrame(rows,columns=["ts","open","high","low","close","volume"])
    df["date"]=df.ts.dt.date.astype(str)
    return df


def daily_context(hourly):
    d=hourly.groupby("date",sort=True).agg(
        open=("open","first"),high=("high","max"),low=("low","min"),
        close=("close","last"),volume=("volume","sum")
    ).reset_index()
    o,h,l,c,v=d.open,d.high,d.low,d.close,d.volume
    ret=c.pct_change()
    d["d_ret1"]=c/c.shift(1)-1
    d["d_ret5"]=c/c.shift(5)-1
    d["d_ret20"]=c/c.shift(20)-1
    atr=pd.Series(v2.pine_atr(h.to_numpy(),l.to_numpy(),c.to_numpy(),14),index=d.index)
    d["d_atr_pct"]=atr/c
    d["d_rsi14"]=v2.pine_rsi(c.to_numpy(),14)
    bm=c.rolling(20,min_periods=20).mean(); bs=c.rolling(20,min_periods=20).std(ddof=0)
    up=bm+2*bs; lo=bm-2*bs
    d["d_bb_pos"]=(c-lo)/(up-lo).replace(0,np.nan)
    d["d_bb_width"]=(up-lo)/bm.replace(0,np.nan)
    ema12=c.ewm(span=12,adjust=False,min_periods=12).mean()
    ema25=c.ewm(span=25,adjust=False,min_periods=25).mean()
    ema26=c.ewm(span=26,adjust=False,min_periods=26).mean()
    sig=(ema12-ema26).ewm(span=9,adjust=False,min_periods=9).mean()
    d["d_dist_ema25"]=c/ema25-1
    d["d_ema25_slope5"]=ema25/ema25.shift(5)-1
    hh20=h.shift(1).rolling(20,min_periods=20).max()
    hh60=h.shift(1).rolling(60,min_periods=40).max()
    d["d_dd20"]=c/hh20-1
    d["d_dd60"]=c/hh60-1
    d["d_vol_ratio20"]=v/v.shift(1).rolling(20,min_periods=20).mean()

    # 現行Stable条件を「前日確定情報」で構成（1H途中で未来の日足を見ない）
    lo14=l.rolling(14,min_periods=14).min(); hi14=h.rolling(14,min_periods=14).max()
    stoch=(c-lo14)/(hi14-lo14).replace(0,np.nan)
    d["s_ema25"]=c>ema25
    d["s_macdpos"]=(ema12-ema26-sig)>0
    d["s_stoch75"]=stoch>=.75
    d["s_bb80"]=d.d_bb_pos>=.80
    d["s_pre_down3"]=(c.shift(1)<c.shift(2))&(c.shift(2)<c.shift(3))
    d["s_gapup"]=o>c.shift(1)

    # 各hourly barには前日の確定daily contextだけを付与
    context_cols=[x for x in d.columns if x.startswith("d_") or x.startswith("s_")]
    prev=d[["date"]+context_cols].copy()
    prev[context_cols]=prev[context_cols].shift(1)

    # 前日終値・前日総出来高（ユーザー指定基本条件）
    prev["prev_day_close"]=c.shift(1)
    prev["prev_day_volume"]=v.shift(1)

    # 5/10/20/40取引日後終値
    for n in (5,10,20,40):
        prev[f"future_close_{n}"]=c.shift(-n)
        prev[f"exit_date_{n}"]=d.date.shift(-n)

    # 銘柄内の取引日index
    prev["day_index"]=np.arange(len(prev))
    return prev


def build_frame(issue,chart,history_start,end_date):
    df=parse_1h_chart(chart)
    if df is None: return None
    ctx=daily_context(df)
    df=df.merge(ctx,on="date",how="left")
    o,h,l,c,v=df.open.astype(float),df.high.astype(float),df.low.astype(float),df.close.astype(float),df.volume.astype(float)
    ret=c.pct_change()

    df["bar_index"]=np.arange(len(df))
    for n in (1,2,3,6,12,24):
        df[f"ret{n}b"]=c/c.shift(n)-1
    prev=c.shift(1)
    df["body_pct"]=(c-o)/prev
    df["range_pct"]=(h-l)/prev
    cr=(h-l).replace(0,np.nan)
    df["close_loc"]=((c-l)/cr).clip(0,1).fillna(.5)
    df["upper_wick"]=((h-np.maximum(o,c))/cr).clip(0,1).fillna(0)
    df["lower_wick"]=((np.minimum(o,c)-l)/cr).clip(0,1).fillna(0)

    atr=pd.Series(v2.pine_atr(h.to_numpy(),l.to_numpy(),c.to_numpy(),14),index=df.index)
    df["atr_pct"]=atr/c
    df["rsi12"]=v2.pine_rsi(c.to_numpy(),12)
    df["rsi14"]=v2.pine_rsi(c.to_numpy(),14)

    bm=c.rolling(20,min_periods=20).mean(); bs=c.rolling(20,min_periods=20).std(ddof=0)
    up=bm+2*bs; lo=bm-2*bs
    df["bb_pos"]=(c-lo)/(up-lo).replace(0,np.nan)
    df["bb_width"]=(up-lo)/bm.replace(0,np.nan)
    df["bb_width_rel60"]=df.bb_width/df.bb_width.shift(1).rolling(60,min_periods=30).median()

    ema12=c.ewm(span=12,adjust=False,min_periods=12).mean()
    ema25=c.ewm(span=25,adjust=False,min_periods=25).mean()
    ema26=c.ewm(span=26,adjust=False,min_periods=26).mean()
    ema75=c.ewm(span=75,adjust=False,min_periods=75).mean()
    macd=ema12-ema26; sig=macd.ewm(span=9,adjust=False,min_periods=9).mean()
    df["dist_ema25"]=c/ema25-1
    df["dist_ema75"]=c/ema75-1
    df["ema25_slope5"]=ema25/ema25.shift(5)-1
    df["macd_hist_pct"]=(macd-sig)/c

    hh20=h.shift(1).rolling(20,min_periods=20).max()
    hh60=h.shift(1).rolling(60,min_periods=40).max()
    df["break20"]=c/hh20-1
    df["break60"]=c/hh60-1

    for n in (5,20,60):
        df[f"vol_ratio{n}"]=v/v.shift(1).rolling(n,min_periods=n).mean()
    turnover=c*v
    df["turnover_ratio20"]=turnover/turnover.shift(1).rolling(20,min_periods=20).mean()
    df["rv6"]=ret.rolling(6,min_periods=6).std(ddof=0)
    df["rv20"]=ret.rolling(20,min_periods=20).std(ddof=0)
    signed=np.sign(ret.fillna(0))*v
    df["money_flow20"]=signed.rolling(20,min_periods=20).sum()/v.rolling(20,min_periods=20).sum()

    mins=df.ts.dt.hour*60+df.ts.dt.minute
    phase=2*np.pi*mins/(24*60)
    df["sin_time"]=np.sin(phase); df["cos_time"]=np.cos(phase)
    # TSE日中の位置。Yahooのbar時刻が多少特殊でも単調特徴として使う。
    df["day_progress"]=((mins-540)/(390)).clip(0,1)

    bottom,_,_=v2.pine_state_signals(c.to_numpy(),h.to_numpy(),l.to_numpy(),EXACT_PROFILE)
    df["tr_pine_exact"]=bottom.astype(float)
    rsi12=pd.Series(df.rsi12,index=df.index)
    df["tr_breakout"]=((df.break20>0)&(df.vol_ratio20>=1.2)&(df.close_loc>=.55)).astype(float)
    df["tr_reversal"]=((rsi12.shift(1)<42)&(rsi12>rsi12.shift(1))&(df.close_loc>=.55)).astype(float)
    df["tr_volume_reversal"]=((df.vol_ratio20>=2)&(df.ret1b>0)&(df.close_loc>=.65)).astype(float)
    df["tr_squeeze"]=((df.bb_width_rel60<=.8)&(c>bm)&(df.ret1b>0)&(df.close_loc>=.6)).astype(float)
    df["tr_pullback"]=((df.d_dd60<=-.08)&(df.d_dd60>=-.35)&(df.ret1b>=.01)&(df.close_loc>=.6)).astype(float)
    df["tr_trend_resume"]=((c>ema25)&(df.ema25_slope5>0)&(df.ret6b>=-.05)&(df.ret6b<=.05)&(df.ret1b>0)).astype(float)
    df["tr_accumulation"]=((df.turnover_ratio20>=1.5)&(df.ret1b.abs()<=.02)&(df.close_loc>=.55)&(df.upper_wick<=.35)).astype(float)
    df["tr_momentum"]=((df.ret12b>=0)&(df.ret12b<=.15)&(df.ret3b>0)&(df.vol_ratio20>=1.2)&(df.bb_pos>=.6)).astype(float)

    trcols=[x for x in df.columns if x.startswith("tr_")]
    df["candidate"]=df[trcols].sum(axis=1)>0
    df["base"]=(df.prev_day_close<=1000)&(df.prev_day_volume>=10000)&(v>=5000)

    for n in (5,10,20,40):
        df[f"perf_{n}bd"]=df[f"future_close_{n}"]/c-1

    use=(df.date>=history_start)&(df.date<=end_date)&df.base&df.candidate
    local_features=[x for x in FEATURES if not x.startswith("xrank_")]
    cols=["ts","date","day_index","bar_index","close","volume","perf_5bd","perf_10bd","perf_20bd","perf_40bd",
          "exit_date_5","exit_date_10","exit_date_20","exit_date_40"]+local_features+[
          "s_ema25","s_macdpos","s_stoch75","s_bb80","s_pre_down3","s_gapup"
    ]
    out=df.loc[use,cols].copy()
    out.rename(columns={f"exit_date_{n}":f"exit_date_{n}bd" for n in (5,10,20,40)},inplace=True)
    out["symbol"]=issue.code; out["name"]=issue.name; out["market"]=issue.market
    return out


def fetch_one(issue,history_start,end_date):
    cfg=core.ScreeningConfig(yahoo_range="730d",yahoo_interval="1h")
    chart,err=core.fetch_chart(issue,cfg)
    if err: return None,err
    f=build_frame(issue,chart or {},history_start,end_date)
    if f is None: return None,"too_few_rows"
    return f,None


def add_xranks(df):
    m={"ret6b":"xrank_ret6","vol_ratio20":"xrank_vol20","atr_pct":"xrank_atr",
       "rsi12":"xrank_rsi12","break20":"xrank_break20","close_loc":"xrank_close_loc"}
    for src,dst in m.items():
        df[dst]=df.groupby("ts",sort=False)[src].rank(pct=True,method="average")
    return df


def stats(e):
    if e.empty: return {"n":0,"avg":0.0,"robust_avg":0.0,"wr":0.0,"hits":0,"target_rate":0.0}
    v=pd.to_numeric(e.perf_5bd,errors="coerce").dropna().to_numpy(float)
    if not len(v): return {"n":0,"avg":0.0,"robust_avg":0.0,"wr":0.0,"hits":0,"target_rate":0.0}
    dec=v[v!=0]; wr=float(np.mean(dec>0)) if len(dec) else 0
    if len(v)>=10:
        lo,hi=np.quantile(v,[.05,.95]); robust=float(np.mean(np.clip(v,lo,hi)))
    else: robust=float(np.mean(v))
    hits=int(np.sum(v>=.10))
    return {"n":int(len(v)),"avg":float(np.mean(v)),"robust_avg":robust,"wr":wr,"hits":hits,"target_rate":float(hits/len(v))}


def split_periods(df,holdout_start):
    pre=sorted(df.loc[df.date<holdout_start,"date"].unique())
    if len(pre)<80: raise RuntimeError("not enough pre-holdout sessions")
    cut=int(len(pre)*.70)
    return {"train_start":pre[0],"train_end":pre[cut-1],"valid_start":pre[cut],"valid_end":pre[-1]}


def period(df,start,end):
    return df[(df.date>=start)&(df.date<=end)&df.exit_date_5bd.notna()&(df.exit_date_5bd<=end)].copy()


def weights(y):
    y=np.asarray(y,int); n=len(y); p=max(1,int(y.sum())); q=max(1,n-p)
    return np.where(y==1,n/(2*p),n/(2*q))


def models():
    lin=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                  ("m",LogisticRegression(max_iter=500,class_weight="balanced",C=.5))])
    pos=Pipeline([("imp",SimpleImputer(strategy="median")),
                  ("m",HistGradientBoostingClassifier(learning_rate=.06,max_iter=220,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.0,random_state=51))])
    hit=Pipeline([("imp",SimpleImputer(strategy="median")),
                  ("m",HistGradientBoostingClassifier(learning_rate=.05,max_iter=240,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.5,random_state=52))])
    reg=Pipeline([("imp",SimpleImputer(strategy="median")),
                  ("m",HistGradientBoostingRegressor(learning_rate=.05,max_iter=230,max_leaf_nodes=15,min_samples_leaf=35,l2_regularization=1.0,random_state=53))])
    return lin,pos,hit,reg


def fit(train):
    X=train[FEATURES]; yp=(train.perf_5bd>0).astype(int); yh=(train.perf_5bd>=.10).astype(int)
    lin,pos,hit,reg=models()
    lin.fit(X,yp); pos.fit(X,yp,m__sample_weight=weights(yp)); hit.fit(X,yh,m__sample_weight=weights(yh))
    reg.fit(X,train.perf_5bd.clip(-.20,.35),m__sample_weight=1+2*yh.to_numpy())
    return lin,pos,hit,reg


def comps(ms,f):
    lin,pos,hit,reg=ms; X=f[FEATURES]
    pp=.35*lin.predict_proba(X)[:,1]+.65*pos.predict_proba(X)[:,1]
    ph=hit.predict_proba(X)[:,1]; pr=reg.predict(X)
    return pp,ph,pr


def ecdf(v,ref):
    ref=np.sort(np.asarray(ref,float)); v=np.asarray(v,float)
    return np.searchsorted(ref,v,side="right")/max(1,len(ref))


def score(f,cp,ref,blend):
    pp,ph,pr=cp; rr=ecdf(pr,ref); a,b,c=blend
    o=f.copy(); o["p_pos"]=pp; o["p_hit10"]=ph; o["pred_ret"]=pr
    o["score100"]=np.clip(100*(a*pp+b*ph+c*rr),0,100)
    return o


def cooldown(e,days=5):
    if e.empty: return e
    keep=[]
    for _,g in e.sort_values(["symbol","day_index","score100"],ascending=[True,True,False]).groupby("symbol",sort=False):
        last=-10**9
        # 同日複数は最高scoreだけ
        g=g.sort_values(["day_index","score100"],ascending=[True,False]).groupby("day_index",sort=False).head(1)
        for idx,row in g.iterrows():
            di=int(row.day_index)
            if di-last>=days:
                keep.append(idx); last=di
    return e.loc[keep].sort_values(["date","score100"],ascending=[True,False]).reset_index(drop=True)


def apply_policy(s,thr,k):
    e=s[s.score100>=thr].copy()
    if e.empty: return e
    e=e.sort_values(["date","score100"],ascending=[True,False]).groupby("date",sort=False).head(k)
    return cooldown(e,5)


def obj(s):
    if s["n"]<25 or s["n"]>220: return -1e9
    if s["robust_avg"]<=0: return -1e8+s["robust_avg"]
    return .45*s["wr"]+2.8*s["robust_avg"]+.5*s["target_rate"]+.012*math.log1p(s["n"])


def choose(valid,cp,retref):
    best=None
    for name,bl in BLENDS.items():
        sc=score(valid,cp,retref,bl)
        qs=np.unique(np.quantile(sc.score100,[.70,.75,.80,.85,.88,.90,.92,.94,.95,.96,.97,.98,.99]))
        for t in qs:
            for k in (1,2,3,5,8):
                st=stats(apply_policy(sc,float(t),k)); r=obj(st)
                if best is None or r>best["rank"]:
                    best={"blend_name":name,"blend":bl,"threshold":float(t),"top_per_day":k,"valid_stats":st,"rank":r}
    return best


def current_score_events(hold):
    m=(hold.tr_pine_exact>0)&hold.s_ema25.fillna(False)&hold.s_macdpos.fillna(False)&hold.s_stoch75.fillna(False)&hold.s_bb80.fillna(False)&hold.s_pre_down3.fillna(False)&hold.s_gapup.fillna(False)
    e=hold[m].copy(); e["score100"]=100.0
    return cooldown(e,5)


def report(result):
    cur=result["current_reference"]; ex=result["holdout"]["h1_pine_current_prevday"]; ml=result["holdout"]["h1_ml"]
    return "\n".join([
        "# No-TV 1H Lab",
        "",
        f"- 生成: {result['generated_at_jst']}",
        f"- 期間: {result['history_start']} ～ {result['end_date']}",
        f"- Holdout: {result['holdout_start']} ～ {result['end_date']}",
        f"- Universe: {result['universe_count']} / Yahoo成功: {result['yahoo_ok']}",
        "",
        "|方式|n|5BD平均|Robust平均|勝率|+10% Hit|Hit率|",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"|現行4H Stable★6|{cur['n']}|{cur['avg']*100:+.1f}%|-|{cur['wr']*100:.1f}%|{cur['hits']}|{cur['target_rate']*100:.1f}%|",
        f"|1H Pine Exact + 前日確定Stable条件|{ex['n']}|{ex['avg']*100:+.1f}%|{ex['robust_avg']*100:+.1f}%|{ex['wr']*100:.1f}%|{ex['hits']}|{ex['target_rate']*100:.1f}%|",
        f"|**1H ML 0-100 Score**|**{ml['n']}**|**{ml['avg']*100:+.1f}%**|**{ml['robust_avg']*100:+.1f}%**|**{ml['wr']*100:.1f}%**|**{ml['hits']}**|**{ml['target_rate']*100:.1f}%**|",
        "",
        "## 注意",
        "- Yahoo 1hだけを使用。TradingViewは使用しない。",
        "- 1H途中で当日の日足終値を使うと未来参照になるため、daily contextは前日確定値だけ。",
        "- 基本条件: 前日終値<=1000円 / 前日総出来高>=10000株 / シグナル1H出来高>=5000株。",
        "- 5BD: シグナル1H終値→5取引日後の終値。",
        "- Holdoutはモデル・blend・threshold・top/day選択に不使用。",
    ])


def run(args):
    end_d=date.fromisoformat(args.end_date)
    history_start=(end_d-timedelta(days=args.history_days)).isoformat()
    holdout_start=(end_d-timedelta(days=args.holdout_days)).isoformat()

    list_date,issues=fetch_jpx_issues_fixed(); issues=[i for i in issues if "優先" not in i.name]
    if args.max_issues: issues=issues[:args.max_issues]
    frames=[]; errors={}; ok=0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs={ex.submit(fetch_one,i,history_start,args.end_date):i for i in issues}
        for n,f in enumerate(as_completed(futs),1):
            try: fr,err=f.result()
            except Exception as e: fr,err=None,type(e).__name__
            if fr is not None and not fr.empty: frames.append(fr); ok+=1
            elif err: errors[err]=errors.get(err,0)+1
            if n%250==0: print(f"progress {n}/{len(issues)} frames={len(frames)}",flush=True)
    if not frames: raise RuntimeError("no 1h candidate data")
    df=pd.concat(frames,ignore_index=True)
    df=add_xranks(df).sort_values(["ts","symbol"]).reset_index(drop=True)

    sp=split_periods(df,holdout_start)
    train=period(df,sp["train_start"],sp["train_end"])
    valid=period(df,sp["valid_start"],sp["valid_end"])
    hold=period(df,holdout_start,args.end_date)
    print(f"rows train={len(train)} valid={len(valid)} hold={len(hold)}",flush=True)

    ms=fit(train); trc=comps(ms,train); vac=comps(ms,valid); hoc=comps(ms,hold)
    pol=choose(valid,vac,trc[2])
    hscore=score(hold,hoc,trc[2],tuple(pol["blend"]))
    hev=apply_policy(hscore,pol["threshold"],pol["top_per_day"])
    exact=current_score_events(hold)

    return {
        "generated_at_jst":core.now_jst().isoformat(timespec="seconds"),
        "history_start":history_start,"holdout_start":holdout_start,"end_date":args.end_date,
        "jpx_list_date":list_date,"universe_count":len(issues),"yahoo_ok":ok,"errors":errors,
        "candidate_rows":len(df),"split":sp,"current_reference":CURRENT_REFERENCE,
        "policy":{k:v for k,v in pol.items() if k!="rank"},
        "holdout":{"h1_pine_current_prevday":stats(exact),"h1_ml":stats(hev)},
        "top_events":hev.sort_values("score100",ascending=False).head(100)[
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
    p.add_argument("--output-dir",default="reports_no_tv_1h")
    a=p.parse_args()
    if a.history_days<=a.holdout_days: raise SystemExit("history-days must > holdout-days")
    r=run(a); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"comparison_1h.json").write_text(json.dumps(r,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    (out/"comparison_1h.md").write_text(report(r),encoding="utf-8")
    print(json.dumps({"holdout":r["holdout"],"policy":r["policy"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
