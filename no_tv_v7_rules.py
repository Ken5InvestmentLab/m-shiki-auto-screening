from __future__ import annotations

import argparse, itertools, json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import screen_big_money as core
from screen_entry import fetch_jpx_issues_fixed
import no_tv_daily_v3 as v3

TRIGGERS=[
 "tr_pine_exact","tr_breakout","tr_reversal","tr_volume_reversal","tr_squeeze",
 "tr_pullback","tr_trend_resume","tr_accumulation","tr_momentum"
]
RULES=[
 "ema25","ema75","macdpos","ret1_pos","ret3_pos","ret5_pos","ret20_pos",
 "ret20_under20","dd60_5_30","dd120_10_45","rsi_45_70","rsi_50_75",
 "stoch60","stoch75","bb60","bb80","bbwidth_low","atr_under5","atr_under8",
 "vol12","vol15","vol20","turn15","turn20","close60","close75",
 "lowerwick30","upperwick35","gap_pos","pre_down2","pre_down3","fromlow5_30"
]
CURRENT={"n":55,"avg":.066,"wr":.564,"hits":10,"target_rate":10/55}

def parse_date(v):
 try:return date.fromisoformat(v).isoformat()
 except ValueError as e:raise argparse.ArgumentTypeError("YYYY-MM-DD") from e

def add_rules(df):
 x=df.copy()
 x["ema25"]=x.s_ema25.astype(bool)
 x["ema75"]=x.dist_ema75>0
 x["macdpos"]=x.s_macdpos.astype(bool)
 x["ret1_pos"]=x.ret1>0;x["ret3_pos"]=x.ret3>0;x["ret5_pos"]=x.ret5>0;x["ret20_pos"]=x.ret20>0
 x["ret20_under20"]=(x.ret20>-.05)&(x.ret20<.20)
 x["dd60_5_30"]=(x.dd60<=-.05)&(x.dd60>=-.30)
 x["dd120_10_45"]=(x.dd120<=-.10)&(x.dd120>=-.45)
 x["rsi_45_70"]=(x.rsi12>=45)&(x.rsi12<70);x["rsi_50_75"]=(x.rsi12>=50)&(x.rsi12<75)
 x["stoch60"]=x.stoch14>=.60;x["stoch75"]=x.stoch14>=.75
 x["bb60"]=x.bb_pos>=.60;x["bb80"]=x.bb_pos>=.80
 x["bbwidth_low"]=x.bb_width_rel60<=.90
 x["atr_under5"]=x.atr_pct<.05;x["atr_under8"]=x.atr_pct<.08
 x["vol12"]=x.vol_ratio20>=1.2;x["vol15"]=x.vol_ratio20>=1.5;x["vol20"]=x.vol_ratio20>=2
 x["turn15"]=x.turnover_ratio20>=1.5;x["turn20"]=x.turnover_ratio20>=2
 x["close60"]=x.close_loc>=.60;x["close75"]=x.close_loc>=.75
 x["lowerwick30"]=x.lower_wick>=.30;x["upperwick35"]=x.upper_wick<=.35
 x["gap_pos"]=x.gap>0
 x["pre_down2"]=x.down_count3>=2;x["pre_down3"]=x.down_count3>=3
 x["fromlow5_30"]=(x.from_low20>=.05)&(x.from_low20<=.30)
 return x

def stats_mask(frame,mask):
 vals=pd.to_numeric(frame.loc[mask,"perf_5bd"],errors="coerce").dropna().to_numpy(float)
 if not len(vals):return {"n":0,"avg":0.,"robust":0.,"wr":0.,"hit":0.}
 dec=vals[vals!=0];wr=float(np.mean(dec>0)) if len(dec) else 0.
 if len(vals)>=10:
  lo,hi=np.quantile(vals,[.05,.95]);rob=float(np.mean(np.clip(vals,lo,hi)))
 else:rob=float(np.mean(vals))
 return {"n":int(len(vals)),"avg":float(np.mean(vals)),"robust":rob,"wr":wr,"hit":float(np.mean(vals>=.10))}

def folds(df,hold):
 ds=sorted(df.loc[df.date<hold,"date"].unique())
 p40=int(len(ds)*.40);p60=int(len(ds)*.60);p80=int(len(ds)*.80)
 return [
  (ds[0],ds[p40-1],ds[p40],ds[p60-1]),
  (ds[0],ds[p60-1],ds[p60],ds[p80-1]),
  (ds[0],ds[p80-1],ds[p80],ds[-1]),
 ]

def period(df,a,b):
 return df[(df.date>=a)&(df.date<=b)&df.exit_date_5bd.notna()&(df.exit_date_5bd<=b)]

def search(df,hold):
 fs=folds(df,hold)
 val_frames=[period(df,c,d) for _,_,c,d in fs]
 # boolean matrices once
 rule_arrays=[{r:f[r].fillna(False).to_numpy(bool) for r in RULES} for f in val_frames]
 trig_arrays=[{t:(f[t].fillna(0).to_numpy(float)>0) for t in TRIGGERS} for f in val_frames]
 best=None
 for t in TRIGGERS:
  for size in (1,2,3):
   for combo in itertools.combinations(RULES,size):
    foldstats=[];ok=True
    for fi,f in enumerate(val_frames):
     m=trig_arrays[fi][t].copy()
     for r in combo:m &= rule_arrays[fi][r]
     s=stats_mask(f,m);foldstats.append(s)
     if s["n"]<8 or s["robust"]<=0 or s["wr"]<.48:
      ok=False;break
    if not ok:continue
    minrob=min(s["robust"] for s in foldstats);minwr=min(s["wr"] for s in foldstats)
    medrob=float(np.median([s["robust"] for s in foldstats]))
    medhit=float(np.median([s["hit"] for s in foldstats]))
    rank=4*minrob+.55*minwr+1.2*medrob+.35*medhit+.008*math.log1p(sum(s["n"] for s in foldstats))
    if best is None or rank>best["rank"]:
     best={"trigger":t,"rules":list(combo),"fold_stats":foldstats,"rank":rank}
 return best,fs

def cooldown(e,days=5):
 if e.empty:return e
 keep=[]
 for _,g in e.sort_values(["symbol","bar_index"]).groupby("symbol",sort=False):
  last=-10**9
  for idx,r in g.iterrows():
   bi=int(r.bar_index)
   if bi-last>=days:keep.append(idx);last=bi
 return e.loc[keep]

def apply_rule(frame,p):
 if not p:return frame.iloc[0:0]
 m=frame[p["trigger"]].fillna(0).to_numpy(float)>0
 for r in p["rules"]:m &= frame[r].fillna(False).to_numpy(bool)
 e=frame.loc[m].copy()
 # strongest candidates per day by simple quality rank: close location + turnover + 5/20 momentum bounded
 e["rule_rank"]=(
   e.close_loc.fillna(.5)+np.log1p(e.turnover_ratio20.clip(lower=0).fillna(0))
   +.5*e.vol_ratio20.clip(0,5).fillna(0)
 )
 e=e.sort_values(["date","rule_rank"],ascending=[True,False]).groupby("date",sort=False).head(3)
 return cooldown(e,5)

def report(r):
 s=r["holdout"];cur=r["current_reference"]
 return "\n".join([
  "# No-TV V7 — Low-complexity Rule Challenger","",
  f"- Holdout: {r['holdout_start']} ～ {r['end_date']}",f"- Universe: {r['universe_count']} / Yahoo成功: {r['yahoo_ok']}",
  f"- robust policy found: {bool(r['policy'])}","",
  "|方式|n|5BD平均|Robust平均|勝率|+10%Hit率|","|---|---:|---:|---:|---:|---:|",
  f"|現行4H Stable★6|{cur['n']}|{cur['avg']*100:+.1f}%|-|{cur['wr']*100:.1f}%|{cur['target_rate']*100:.1f}%|",
  f"|V7 Rule|{s['n']}|{s['avg']*100:+.1f}%|{s['robust']*100:+.1f}%|{s['wr']*100:.1f}%|{s['hit']*100:.1f}%|","",
  f"- policy: {json.dumps(r['policy'],ensure_ascii=False)}",
  "- Triggerごとに追加条件1〜3個だけ。3つの未来Validation Foldすべてでrobust平均>0・勝率>=48%を要求。",
  "- 最終1年Holdoutは探索に不使用。TradingView不使用。"
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
   if n%500==0:print(f"progress {n}/{len(issues)} frames={len(frames)}",flush=True)
 if not frames:raise RuntimeError("no candidates")
 df=add_rules(pd.concat(frames,ignore_index=True)).sort_values(["date","symbol","bar_index"]).reset_index(drop=True)
 pol,fs=search(df,hold)
 ho=period(df,hold,args.end_date)
 ev=apply_rule(ho,pol)
 s=stats_mask(ev,np.ones(len(ev),bool))
 return {"generated_at_jst":core.now_jst().isoformat(timespec="seconds"),"history_start":start,"holdout_start":hold,"end_date":args.end_date,
         "universe_count":len(issues),"yahoo_ok":ok,"errors":errs,"candidate_rows":len(df),"current_reference":CURRENT,
         "folds":fs,"policy":({k:v for k,v in pol.items() if k!="rank"} if pol else None),"holdout":s}

def main():
 today=core.now_jst().date();p=argparse.ArgumentParser()
 p.add_argument("--end-date",type=parse_date,default=today.isoformat());p.add_argument("--history-days",type=int,default=1825);p.add_argument("--holdout-days",type=int,default=365)
 p.add_argument("--max-workers",type=int,default=32);p.add_argument("--max-issues",type=int);p.add_argument("--output-dir",default="reports_no_tv_v7")
 a=p.parse_args();r=run(a);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 (out/"comparison_v7.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")
 (out/"comparison_v7.md").write_text(report(r),encoding="utf-8")
 print(json.dumps({"policy":r["policy"],"holdout":r["holdout"]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
