from __future__ import annotations
import argparse,json
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import date,timedelta
from pathlib import Path
import numpy as np,pandas as pd
import screen_big_money as core
from screen_entry import fetch_jpx_issues_fixed
import no_tv_daily_v3 as v3

CURRENT={"n":55,"avg":.06587272727272729,"wr":.5636363636363636,"hits":10,"target_rate":10/55}

def parse_date(v):
 try:return date.fromisoformat(v).isoformat()
 except ValueError as e:raise argparse.ArgumentTypeError("YYYY-MM-DD") from e

def enrich(df):
 x=df.copy()
 x["stable_score"]=(
   x.s_ema25.astype(int)+x.s_macdpos.astype(int)+x.s_stoch75.astype(int)+
   x.s_bb80.astype(int)+x.s_pre_down3.astype(int)+x.s_gapup.astype(int)
 )
 x["rci9_os"]=x.rci9<=-.50
 x["vol15"]=x.vol_ratio20>=1.5
 x["body2"]=x.body_pct>=.02
 x["guard_vol"]=(~x.vol15)|x.rci9_os
 x["guard_body"]=(~x.body2)|x.rci9_os
 x["guard_both"]=x.guard_vol&x.guard_body
 x["guard_either"]=x.guard_vol|x.guard_body
 return x

def cooldown(e,days=5):
 if e.empty:return e
 keep=[]
 for _,g in e.sort_values(["symbol","bar_index"]).groupby("symbol",sort=False):
  last=-10**9
  for idx,r in g.iterrows():
   bi=int(r.bar_index)
   if bi-last>=days:keep.append(idx);last=bi
 return e.loc[keep]

def select(df,min_score,guard):
 m=df.stable_score>=min_score
 if guard!="none":m &= df[guard].fillna(False)
 e=df[m].copy()
 e["rank"]=e.stable_score+0.25*e.close_loc+0.10*np.log1p(e.turnover_ratio20.clip(lower=0))
 e=e.sort_values(["date","rank"],ascending=[True,False]).groupby("date",sort=False).head(3)
 return cooldown(e,5)

def stats(e):
 return v3.stats(e)

def common(e):
 return stats(e[(e.date>="2026-03-05")&(e.date<="2026-09-09")])

def report(r):
 lines=["# No-TV V9 — Teacher-informed Overheat Guard","",
  "※ このV9のガードは現行2026年Stable55件を観察して作ったため、同期間成績は厳密なOOSではなく探索評価。","",
  "|Variant|1年 n|1年平均|1年勝率|3/5〜9/9 n|同平均|同勝率|",
  "|---|---:|---:|---:|---:|---:|---:|"]
 for k,v in r["variants"].items():
  a=v["holdout"];c=v["common"]
  lines.append(f"|{k}|{a['n']}|{a['avg']*100:+.1f}%|{a['wr']*100:.1f}%|{c['n']}|{c['avg']*100:+.1f}%|{c['wr']*100:.1f}%|")
 lines+=["","Teacher guard:",
  "- guard_vol: 出来高20日比 <1.5 または RCI9<=-50",
  "- guard_body: 実体<2% または RCI9<=-50",
  "- guard_both: 上記2つを両方満たす",
  "- Yahooのみ。TradingViewは候補生成/運用には不使用。Teacher知見の由来だけが過去TV実績。"]
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
   if n%500==0:print(f"progress {n}/{len(issues)}",flush=True)
 if not frames:raise RuntimeError("no candidates")
 df=enrich(pd.concat(frames,ignore_index=True))
 ho=df[(df.date>=hold)&(df.date<=args.end_date)&df.exit_date_5bd.notna()&(df.exit_date_5bd<=args.end_date)]
 variants={}
 for score in (4,5,6):
  for guard in ("none","guard_vol","guard_body","guard_both","guard_either"):
   name=f"S{score}_{guard}"
   ev=select(ho,score,guard)
   variants[name]={"holdout":stats(ev),"common":common(ev)}
 return {"generated_at_jst":core.now_jst().isoformat(timespec="seconds"),"history_start":start,"holdout_start":hold,"end_date":args.end_date,
         "universe_count":len(issues),"yahoo_ok":ok,"errors":errs,"current_reference":CURRENT,"variants":variants}

def main():
 today=core.now_jst().date();p=argparse.ArgumentParser()
 p.add_argument("--end-date",type=parse_date,default=today.isoformat());p.add_argument("--history-days",type=int,default=1825);p.add_argument("--holdout-days",type=int,default=365)
 p.add_argument("--max-workers",type=int,default=32);p.add_argument("--max-issues",type=int);p.add_argument("--output-dir",default="reports_no_tv_v9")
 a=p.parse_args();r=run(a);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 (out/"comparison_v9.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")
 (out/"comparison_v9.md").write_text(report(r),encoding="utf-8")
 print(json.dumps(r["variants"],ensure_ascii=False,indent=2))
if __name__=="__main__":main()
