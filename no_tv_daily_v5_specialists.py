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
import no_tv_daily_v4_regime as v4


TRIGGERS=[
    "tr_pine_exact","tr_breakout","tr_reversal","tr_volume_reversal",
    "tr_squeeze","tr_pullback","tr_trend_resume","tr_accumulation","tr_momentum"
]
BM_FEATURES=[
    "bm_turnover_strength","bm_volume_strength","bm_candle_quality",
    "bm_pressure","bm_quiet_accum","bm_shock_quality"
]
FEATURES=v4.FEATURES+BM_FEATURES

QS=(.90,.92,.94,.95,.96,.97,.98,.985,.99,.995)
TOPS=(1,2,3,5)

def parse_date(v):
    try:return date.fromisoformat(v).isoformat()
    except ValueError as e:raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from e

def add_big_money_features(df):
    x=df.copy()
    tv=pd.to_numeric(x.turnover_ratio20,errors="coerce").clip(lower=0)
    vv=pd.to_numeric(x.vol_ratio20,errors="coerce").clip(lower=0)
    cl=pd.to_numeric(x.close_loc,errors="coerce").clip(0,1)
    uw=pd.to_numeric(x.upper_wick,errors="coerce").clip(0,1)
    lw=pd.to_numeric(x.lower_wick,errors="coerce").clip(0,1)
    mf=pd.to_numeric(x.money_flow20,errors="coerce").clip(-1,1)
    r1=pd.to_numeric(x.ret1,errors="coerce").abs()
    x["bm_turnover_strength"]=np.log1p(tv)
    x["bm_volume_strength"]=np.log1p(vv)
    x["bm_candle_quality"]=(.7*cl-.5*uw+.2*lw)
    x["bm_pressure"]=mf*np.log1p(vv)
    x["bm_quiet_accum"]=np.log1p(tv)*(1-(r1/.05).clip(0,1))*cl
    x["bm_shock_quality"]=np.log1p(tv)*np.log1p(vv)*cl*(1-uw)
    return x

def fit_model(frame):
    X=frame[FEATURES]
    yp=(frame.perf_5bd>0).astype(int)
    yh=(frame.perf_5bd>=.10).astype(int)
    if yp.nunique()<2 or yh.nunique()<2:
        return None
    linear,pos,hit,reg=v3.make_models()
    linear.fit(X,yp)
    pos.fit(X,yp,clf__sample_weight=v3.balanced_weights(yp))
    hit.fit(X,yh,clf__sample_weight=v3.balanced_weights(yh))
    reg.fit(X,frame.perf_5bd.clip(-.20,.35),reg__sample_weight=1+2*yh.to_numpy())
    return linear,pos,hit,reg

def predict_model(ms,frame):
    linear,pos,hit,reg=ms;X=frame[FEATURES]
    pp=.35*linear.predict_proba(X)[:,1]+.65*pos.predict_proba(X)[:,1]
    ph=hit.predict_proba(X)[:,1];pr=reg.predict(X)
    return pp,ph,pr

def fit_specialists(train):
    global_model=fit_model(train)
    if global_model is None:raise RuntimeError("global model invalid")
    trig={}
    for t in TRIGGERS:
        sub=train[train[t]>0]
        if len(sub)>=500 and (sub.perf_5bd>0).nunique()==2 and int((sub.perf_5bd>=.10).sum())>=20:
            m=fit_model(sub)
            if m is not None:trig[t]=m
    regimes={}
    for r in (-1,0,1):
        sub=train[train.regime==r]
        if len(sub)>=700 and int((sub.perf_5bd>=.10).sum())>=20:
            m=fit_model(sub)
            if m is not None:regimes[r]=m
    return global_model,trig,regimes

def predict_specialists(pack,frame):
    gm,trig,regs=pack
    gp,gh,gr=predict_model(gm,frame)
    sum_p=gp.copy();sum_h=gh.copy();sum_r=gr.copy();count=np.ones(len(frame),float)
    for t,m in trig.items():
        mask=frame[t].to_numpy()>0
        if not np.any(mask):continue
        p,h,r=predict_model(m,frame.loc[mask])
        # trigger specialist receives equal weight to global model.
        sum_p[mask]+=p;sum_h[mask]+=h;sum_r[mask]+=r;count[mask]+=1
    rv=frame.regime.to_numpy()
    for reg,m in regs.items():
        mask=rv==reg
        if not np.any(mask):continue
        p,h,r=predict_model(m,frame.loc[mask])
        # regime specialist also gets one vote.
        sum_p[mask]+=p;sum_h[mask]+=h;sum_r[mask]+=r;count[mask]+=1
    return sum_p/count,sum_h/count,sum_r/count

def score(frame,cp,retref,blend):
    p,h,r=cp;rr=v3.ecdf_map(r,retref);a,b,c=blend
    out=frame.copy();out["p_pos"]=p;out["p_hit10"]=h;out["pred_ret"]=r
    out["score100"]=np.clip(100*(a*p+b*h+c*rr),0,100)
    return out

def build_folds(df,hold):
    ds=sorted(df.loc[df.date<hold,"date"].unique())
    if len(ds)<200:raise RuntimeError("not enough pre-holdout dates")
    p40=int(len(ds)*.40);p60=int(len(ds)*.60);p80=int(len(ds)*.80)
    return [
      {"train_start":ds[0],"train_end":ds[p40-1],"valid_start":ds[p40],"valid_end":ds[p60-1]},
      {"train_start":ds[0],"train_end":ds[p60-1],"valid_start":ds[p60],"valid_end":ds[p80-1]},
      {"train_start":ds[0],"train_end":ds[p80-1],"valid_start":ds[p80],"valid_end":ds[-1]},
    ]

def fold_pack(df,f):
    tr=v3.period(df,f["train_start"],f["train_end"])
    va=v3.period(df,f["valid_start"],f["valid_end"])
    ms=fit_specialists(tr)
    return tr,va,predict_specialists(ms,tr),predict_specialists(ms,va),list(ms[1]),list(ms[2])

def eval_policy(pack,blend,q,k):
    tr,va,trc,vac,_,_=pack
    ts=score(tr,trc,trc[2],blend);vs=score(va,vac,trc[2],blend)
    thr=float(np.quantile(ts.score100,q))
    ev=v3.apply_policy(vs,thr,k)
    return v3.stats(ev),thr

def robust_rank(ss):
    if min(s["n"] for s in ss)<8:return None
    av=[s["robust_avg"] for s in ss];wr=[s["wr"] for s in ss];hit=[s["target_rate"] for s in ss]
    if min(av)<=0 or min(wr)<.48:return None
    return 3.8*min(av)+.5*min(wr)+.45*min(hit)+1.3*np.median(av)+.2*np.median(wr)+.01*np.log1p(sum(s["n"] for s in ss))

def choose(packs):
    best=None
    for name,bl in v3.BLENDS.items():
        for q in QS:
            for k in TOPS:
                ss=[];ths=[]
                for p in packs:
                    s,t=eval_policy(p,bl,q,k);ss.append(s);ths.append(t)
                rank=robust_rank(ss)
                if rank is None:continue
                if best is None or rank>best["rank"]:
                    best={"blend_name":name,"blend":bl,"q":q,"top_per_day":k,
                          "fold_stats":ss,"fold_thresholds":ths,"rank":float(rank),
                          "robust_policy_found":True}
    if best:return best
    return {"blend_name":"balanced","blend":v3.BLENDS["balanced"],"q":.99,"top_per_day":1,
            "fold_stats":[],"fold_thresholds":[],"rank":None,"robust_policy_found":False}

def report(r):
    cur=r["current_reference"];s=r["holdout"]["v5_specialist"]
    lines=["# No-TV Stable V5 — Trigger + Regime Specialists","",
           f"- Holdout: {r['holdout_start']} ～ {r['end_date']}",f"- Universe: {r['universe_count']} / Yahoo成功: {r['yahoo_ok']}",
           f"- robust policy found: {r['policy']['robust_policy_found']}","",
           "|方式|n|5BD平均|Robust平均|勝率|+10%Hit|Hit率|","|---|---:|---:|---:|---:|---:|---:|",
           f"|現行4H Stable★6|{cur['n']}|{cur['avg']*100:+.1f}%|-|{cur['wr']*100:.1f}%|{cur['hits']}|{cur['target_rate']*100:.1f}%|",
           f"|**V5 Trigger Specialist**|**{s['n']}**|**{s['avg']*100:+.1f}%**|**{s['robust_avg']*100:+.1f}%**|**{s['wr']*100:.1f}%**|**{s['hits']}**|**{s['target_rate']*100:.1f}%**|","",
           f"- policy: {json.dumps(r['policy'],ensure_ascii=False)}",
           "- Candidate種別ごとの専門モデル + 市場レジーム専門モデル + globalモデルを平均。",
           "- M式由来の売買代金/出来高/ローソク品質を連続特徴量として追加。",
           "- TradingView不使用。"]
    return "\\n".join(lines)

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
    if not frames:raise RuntimeError("no V5 candidate data")
    df=pd.concat(frames,ignore_index=True)
    df=v3.add_cross_sectional_ranks(df).merge(v4.market_frame(),on="date",how="left")
    df=add_big_money_features(df).sort_values(["date","symbol","bar_index"]).reset_index(drop=True)

    folds=build_folds(df,hold);packs=[]
    for i,f in enumerate(folds,1):
        p=fold_pack(df,f);packs.append(p)
        print(f"fold{i} train={len(p[0])} valid={len(p[1])} triggers={p[4]} regimes={p[5]}",flush=True)
    pol=choose(packs)

    pre=v3.period(df,df.loc[df.date<hold,"date"].min(),df.loc[df.date<hold,"date"].max())
    ho=v3.period(df,hold,args.end_date)
    ms=fit_specialists(pre);pc=predict_specialists(ms,pre);hc=predict_specialists(ms,ho)
    bl=tuple(pol["blend"]);ps=score(pre,pc,pc[2],bl);hs=score(ho,hc,pc[2],bl)
    thr=float(np.quantile(ps.score100,pol["q"]))
    ev=v3.apply_policy(hs,thr,pol["top_per_day"])
    return {"generated_at_jst":core.now_jst().isoformat(timespec="seconds"),"history_start":start,"holdout_start":hold,"end_date":args.end_date,
            "universe_count":len(issues),"yahoo_ok":ok,"errors":errs,"candidate_rows":len(df),"current_reference":v3.CURRENT_REFERENCE["stable_s6"],
            "policy":{**{k:v for k,v in pol.items() if k!="rank"},"final_threshold":thr,"final_trigger_specialists":list(ms[1]),"final_regime_specialists":list(ms[2])},
            "holdout":{"v5_specialist":v3.stats(ev)},
            "top_events":ev.sort_values("score100",ascending=False).head(100)[["date","symbol","name","score100","p_pos","p_hit10","pred_ret","perf_5bd"]].to_dict("records")}

def main():
    today=core.now_jst().date();p=argparse.ArgumentParser()
    p.add_argument("--end-date",type=parse_date,default=today.isoformat());p.add_argument("--history-days",type=int,default=1825);p.add_argument("--holdout-days",type=int,default=365)
    p.add_argument("--max-workers",type=int,default=24);p.add_argument("--max-issues",type=int);p.add_argument("--output-dir",default="reports_no_tv_v5")
    a=p.parse_args();r=run(a);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"comparison_v5.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"comparison_v5.md").write_text(report(r),encoding="utf-8")
    print(json.dumps({"holdout":r["holdout"],"policy":r["policy"]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
