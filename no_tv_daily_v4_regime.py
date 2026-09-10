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


MKT_FEATURES=[
    "mkt_ret1","mkt_ret5","mkt_ret20","mkt_ret60","mkt_atr_pct","mkt_rsi14",
    "mkt_dist_ema25","mkt_ema25_slope5","mkt_bb_pos","mkt_bb_width",
    "mkt_bull","mkt_bear"
]
FEATURES=v3.FEATURES+MKT_FEATURES
QS=(.90,.92,.94,.95,.96,.97,.98,.985,.99,.995)
TOPS=(1,2,3,5)


def parse_date(v):
    try:return date.fromisoformat(v).isoformat()
    except ValueError as e:raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from e


def market_frame():
    issue=core.Issue(code="1306",name="TOPIX ETF proxy",market="")
    cfg=core.ScreeningConfig(yahoo_range="5y",yahoo_interval="1d")
    chart,err=core.fetch_chart(issue,cfg)
    if err: raise RuntimeError(f"market proxy fetch failed: {err}")
    parsed=core.chart_to_arrays(chart or {},None)
    if parsed is None: raise RuntimeError("market proxy too few rows")
    dates,a=parsed
    d=pd.DataFrame({"date":dates.astype(str),"high":a["high"],"low":a["low"],"close":a["close"]})
    c=d.close.astype(float); h=d.high.astype(float); l=d.low.astype(float)
    for n in (1,5,20,60): d[f"mkt_ret{n}"]=c/c.shift(n)-1
    atr=pd.Series(v3.v2.pine_atr(h.to_numpy(),l.to_numpy(),c.to_numpy(),14),index=d.index)
    d["mkt_atr_pct"]=atr/c
    d["mkt_rsi14"]=v3.v2.pine_rsi(c.to_numpy(),14)/100.0
    ema25=c.ewm(span=25,adjust=False,min_periods=25).mean()
    d["mkt_dist_ema25"]=c/ema25-1
    d["mkt_ema25_slope5"]=ema25/ema25.shift(5)-1
    bm=c.rolling(20,min_periods=20).mean(); bs=c.rolling(20,min_periods=20).std(ddof=0)
    up=bm+2*bs; lo=bm-2*bs
    d["mkt_bb_pos"]=(c-lo)/(up-lo).replace(0,np.nan)
    d["mkt_bb_width"]=(up-lo)/bm.replace(0,np.nan)
    d["mkt_bull"]=((c>ema25)&(d.mkt_ema25_slope5>0)).astype(float)
    d["mkt_bear"]=((c<ema25)&(d.mkt_ema25_slope5<0)).astype(float)
    d["regime"]=np.where(d.mkt_bull>0,1,np.where(d.mkt_bear>0,-1,0))
    return d[["date","regime"]+MKT_FEATURES]


def fit_model(frame):
    X=frame[FEATURES]
    yp=(frame.perf_5bd>0).astype(int); yh=(frame.perf_5bd>=.10).astype(int)
    linear,pos,hit,reg=v3.make_models()
    linear.fit(X,yp)
    pos.fit(X,yp,clf__sample_weight=v3.balanced_weights(yp))
    hit.fit(X,yh,clf__sample_weight=v3.balanced_weights(yh))
    reg.fit(X,frame.perf_5bd.clip(-.20,.35),reg__sample_weight=1+2*yh.to_numpy())
    return linear,pos,hit,reg


def pred(ms,frame):
    linear,pos,hit,reg=ms; X=frame[FEATURES]
    pp=.35*linear.predict_proba(X)[:,1]+.65*pos.predict_proba(X)[:,1]
    ph=hit.predict_proba(X)[:,1]; pr=reg.predict(X)
    return pp,ph,pr


def fit_regime_pack(train):
    global_model=fit_model(train)
    specialists={}
    for regime in (-1,0,1):
        sub=train[train.regime==regime]
        yp=(sub.perf_5bd>0).astype(int); yh=(sub.perf_5bd>=.10).astype(int)
        if len(sub)>=700 and yp.nunique()==2 and yh.sum()>=15 and (len(yh)-yh.sum())>=15:
            specialists[regime]=fit_model(sub)
    return global_model,specialists


def predict_regime(pack,frame):
    gm,spec=pack
    pp,ph,pr=pred(gm,frame)
    pp=pp.copy(); ph=ph.copy(); pr=pr.copy()
    for regime,ms in spec.items():
        mask=(frame.regime.to_numpy()==regime)
        if not np.any(mask): continue
        a,b,c=pred(ms,frame.loc[mask])
        pp[mask]=a;ph[mask]=b;pr[mask]=c
    return pp,ph,pr


def folds(df,hold):
    ds=sorted(df.loc[df.date<hold,"date"].unique())
    p40=int(len(ds)*.40);p60=int(len(ds)*.60);p80=int(len(ds)*.80)
    return [
        {"train_start":ds[0],"train_end":ds[p40-1],"valid_start":ds[p40],"valid_end":ds[p60-1]},
        {"train_start":ds[0],"train_end":ds[p60-1],"valid_start":ds[p60],"valid_end":ds[p80-1]},
        {"train_start":ds[0],"train_end":ds[p80-1],"valid_start":ds[p80],"valid_end":ds[-1]},
    ]


def score_frame(frame,cp,retref,blend):
    pp,ph,pr=cp
    rr=v3.ecdf_map(pr,retref);a,b,c=blend
    o=frame.copy();o["p_pos"]=pp;o["p_hit10"]=ph;o["pred_ret"]=pr
    o["score100"]=np.clip(100*(a*pp+b*ph+c*rr),0,100)
    return o


def policy_eval(pack,blend,q,k):
    tr,va,trc,vac=pack
    ts=score_frame(tr,trc,trc[2],blend);vs=score_frame(va,vac,trc[2],blend)
    t=float(np.quantile(ts.score100,q))
    return v3.stats(v3.apply_policy(vs,t,k)),t


def robust_rank(ss):
    if min(s["n"] for s in ss)<8:return None
    av=[s["robust_avg"] for s in ss];wr=[s["wr"] for s in ss];hit=[s["target_rate"] for s in ss]
    if min(av)<=0 or min(wr)<.47:return None
    return 3.6*min(av)+.5*min(wr)+.4*min(hit)+1.2*np.median(av)+.2*np.median(wr)+.01*np.log1p(sum(s["n"] for s in ss))


def choose(packs):
    best=None
    for name,bl in v3.BLENDS.items():
        for q in QS:
            for k in TOPS:
                ss=[];th=[]
                for p in packs:
                    s,t=policy_eval(p,bl,q,k);ss.append(s);th.append(t)
                r=robust_rank(ss)
                if r is None:continue
                if best is None or r>best["rank"]:
                    best={"blend_name":name,"blend":bl,"q":q,"top_per_day":k,"fold_stats":ss,"fold_thresholds":th,"rank":float(r),"robust_policy_found":True}
    if best:return best
    return {"blend_name":"balanced","blend":v3.BLENDS["balanced"],"q":.99,"top_per_day":1,"fold_stats":[],"fold_thresholds":[],"rank":None,"robust_policy_found":False}


def report(r):
    cur=r["current_reference"];m=r["holdout"]["v4_regime"]
    return "\n".join([
        "# No-TV Stable V4 — Market Regime",
        "",
        f"- Holdout: {r['holdout_start']} ～ {r['end_date']}",
        f"- Universe: {r['universe_count']} / Yahoo成功: {r['yahoo_ok']}",
        f"- robust policy found: {r['policy']['robust_policy_found']}",
        "",
        "|方式|n|5BD平均|Robust平均|勝率|+10%Hit|Hit率|",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"|現行4H Stable★6|{cur['n']}|{cur['avg']*100:+.1f}%|-|{cur['wr']*100:.1f}%|{cur['hits']}|{cur['target_rate']*100:.1f}%|",
        f"|**V4 地合い別ML**|**{m['n']}**|**{m['avg']*100:+.1f}%**|**{m['robust_avg']*100:+.1f}%**|**{m['wr']*100:.1f}%**|**{m['hits']}**|**{m['target_rate']*100:.1f}%**|",
        "",
        f"- policy: {json.dumps(r['policy'],ensure_ascii=False)}",
        "- 1306.TをTOPIX地合いproxyとして使用。TradingView不使用。",
    ])


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
            if n%500==0:print(f"progress {n}/{len(issues)}",flush=True)
    if not frames:raise RuntimeError("no candidate data")
    df=pd.concat(frames,ignore_index=True)
    df=v3.add_cross_sectional_ranks(df).merge(market_frame(),on="date",how="left")
    df=df.sort_values(["date","symbol","bar_index"]).reset_index(drop=True)

    fs=folds(df,hold);packs=[]
    for i,f in enumerate(fs,1):
        tr=v3.period(df,f["train_start"],f["train_end"]);va=v3.period(df,f["valid_start"],f["valid_end"])
        mp=fit_regime_pack(tr);packs.append((tr,va,predict_regime(mp,tr),predict_regime(mp,va)))
        print(f"fold{i} train={len(tr)} valid={len(va)} specialists={list(mp[1])}",flush=True)
    pol=choose(packs)

    pre=v3.period(df,df.loc[df.date<hold,"date"].min(),df.loc[df.date<hold,"date"].max())
    ho=v3.period(df,hold,args.end_date)
    mp=fit_regime_pack(pre);pc=predict_regime(mp,pre);hc=predict_regime(mp,ho)
    bl=tuple(pol["blend"]);ps=score_frame(pre,pc,pc[2],bl);hs=score_frame(ho,hc,pc[2],bl)
    t=float(np.quantile(ps.score100,pol["q"]))
    ev=v3.apply_policy(hs,t,pol["top_per_day"])
    return {
        "generated_at_jst":core.now_jst().isoformat(timespec="seconds"),
        "history_start":start,"holdout_start":hold,"end_date":args.end_date,
        "universe_count":len(issues),"yahoo_ok":ok,"errors":errs,"candidate_rows":len(df),
        "current_reference":v3.CURRENT_REFERENCE["stable_s6"],
        "policy":{**{k:v for k,v in pol.items() if k!="rank"},"final_threshold":t,"final_specialists":list(mp[1])},
        "holdout":{"v4_regime":v3.stats(ev)},
        "top_events":ev.sort_values("score100",ascending=False).head(100)[["date","symbol","name","regime","score100","perf_5bd"]].to_dict("records"),
    }


def main():
    today=core.now_jst().date();p=argparse.ArgumentParser()
    p.add_argument("--end-date",type=parse_date,default=today.isoformat());p.add_argument("--history-days",type=int,default=1825);p.add_argument("--holdout-days",type=int,default=365)
    p.add_argument("--max-workers",type=int,default=24);p.add_argument("--max-issues",type=int);p.add_argument("--output-dir",default="reports_no_tv_v4")
    a=p.parse_args();r=run(a);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"comparison_v4.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"comparison_v4.md").write_text(report(r),encoding="utf-8")
    print(json.dumps({"holdout":r["holdout"],"policy":r["policy"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
