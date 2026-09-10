from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import screen_big_money as core
from screen_entry import fetch_jpx_issues_fixed
from no_tv_v10_features import FEATURES, fetch_one

TEACHER_DEFAULT = "experiments/v10_teacher_full_bottom.csv.gz"
WATCHLIST_CSV = "output/tradingview_tse_price_le_1000.csv"
CHAMPION = {"name":"Production TradingView Stable★6","n":55,"avg":0.066,"wr":0.564,"hits":10,"target_rate":10/55}
TRAIN_START="2026-03-05"; TRAIN_END="2026-06-30"
VALID_START="2026-07-01"; VALID_END="2026-07-31"
TEST_START="2026-08-01"; TEST_END="2026-08-31"
FETCH_END="2026-09-10"


def run_git(repo, *args):
    p=subprocess.run(["git","-C",str(repo),*args],capture_output=True,text=True,encoding="utf-8",errors="replace")
    if p.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr[-1000:]}")
    return p.stdout


def parse_watchlist_csv(text):
    text=text.lstrip("\ufeff")
    out=set()
    for row in csv.DictReader(io.StringIO(text)):
        tv=str(row.get("tv_symbol") or "").strip()
        if not tv: continue
        code=tv.split(":",1)[-1].strip()
        if code: out.add(code)
    return out


def load_exact_watchlists(repo, start_date, end_date):
    # The builder commits output/ after every successful run. A no-change day keeps the prior commit/list valid.
    log=run_git(repo,"log","--format=%H\t%cI","--",WATCHLIST_CSV)
    commits=[]
    for line in log.splitlines():
        if not line.strip(): continue
        sha,ts=line.split("\t",1)
        d=datetime.fromisoformat(ts.replace("Z","+00:00")).astimezone(timezone(timedelta(hours=9)))
        # Normal runs are around 09:17 JST. If a repair build happened after noon, do not let it define that morning's list.
        effective=d.date() if d.hour < 12 else d.date()+timedelta(days=1)
        commits.append((effective,d,sha))
    commits.sort(key=lambda x:x[1])
    if not commits: raise RuntimeError("no historical watchlist commits")

    by_effective={}
    commit_hours=[]
    for effective,local,sha in commits:
        if effective > date.fromisoformat(end_date): continue
        if effective < date.fromisoformat(start_date)-timedelta(days=14): continue
        try:
            text=run_git(repo,"show",f"{sha}:{WATCHLIST_CSV}")
            symbols=parse_watchlist_csv(text)
        except Exception:
            continue
        if symbols:
            by_effective[effective]=(local,sha,symbols)
            commit_hours.append(local.hour+local.minute/60)

    if not by_effective: raise RuntimeError("no usable historical watchlist snapshots")
    effective_dates=sorted(by_effective)
    mapping={}; meta={}
    latest=None; ei=0
    cur=date.fromisoformat(start_date); end=date.fromisoformat(end_date)
    while cur<=end:
        while ei<len(effective_dates) and effective_dates[ei]<=cur:
            latest=effective_dates[ei]; ei+=1
        if latest is not None:
            local,sha,symbols=by_effective[latest]
            mapping[cur.isoformat()]=symbols
            meta[cur.isoformat()]={"snapshot_effective":latest.isoformat(),"commit":sha,"commit_jst":local.isoformat(),"symbols":len(symbols)}
        cur+=timedelta(days=1)
    return mapping,meta,{
        "snapshot_commits":len(by_effective),
        "earliest_effective":min(effective_dates).isoformat(),
        "latest_effective":max(effective_dates).isoformat(),
        "commit_hour_min":float(min(commit_hours)) if commit_hours else None,
        "commit_hour_max":float(max(commit_hours)) if commit_hours else None,
    }


def load_teacher(path):
    t=pd.read_csv(path,dtype={"symbol_code":str})
    t["signal_date"]=pd.to_datetime(t.signal_date).dt.strftime("%Y-%m-%d")
    t["symbol_code"]=t.symbol_code.astype(str).str.replace(r"\.0$","",regex=True).str.strip()
    t["session"]=pd.to_numeric(t.session,errors="coerce").astype("Int64")
    t=t[t.session.isin([9,13])].copy()
    t["key"]=t.symbol_code+"|"+t.signal_date+"|"+t.session.astype(str)
    return t.drop_duplicates("key",keep="last")


def balanced_weights(y):
    y=np.asarray(y,dtype=int); n=len(y); pos=max(1,int(y.sum())); neg=max(1,n-pos)
    return np.where(y==1,n/(2.0*pos),n/(2.0*neg))


def choose_threshold(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    best=None
    qs=np.linspace(.40,.999,180) if len(p) else []
    cand=set(np.linspace(.001,.999,250).tolist())
    if len(p): cand.update(np.quantile(p,qs).tolist())
    for th in sorted(cand):
        pred=(p>=th).astype(int); n=int(pred.sum())
        if n<5: continue
        pr=float(precision_score(y,pred,zero_division=0)); rc=float(recall_score(y,pred,zero_division=0)); f=float(f1_score(y,pred,zero_division=0))
        # F1 is primary. Slight precision tie-break avoids flooding downstream Discord/Codex.
        rank=f+.02*pr
        if best is None or rank>best["rank"]:
            best={"threshold":float(th),"precision":pr,"recall":rc,"f1":f,"predicted":n,"rank":rank}
    return best or {"threshold":.5,"precision":0.,"recall":0.,"f1":0.,"predicted":0,"rank":0.}


def class_stats(df, probs, th):
    y=df.label.to_numpy(int); p=np.asarray(probs,float); pred=(p>=th).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    days=max(1,df.date.nunique())
    auc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan
    ap=average_precision_score(y,p) if y.sum()>0 else np.nan
    return {"n":int(len(y)),"positive":int(y.sum()),"positive_rate":float(y.mean()),
            "roc_auc":float(auc) if np.isfinite(auc) else None,"pr_auc":float(ap) if np.isfinite(ap) else None,
            "precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),
            "f1":float(f1_score(y,pred,zero_division=0)),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
            "predicted":int(pred.sum()),"predicted_per_day":float(pred.sum()/days),"false_positives_per_day":float(fp/days)}


def trade_stats(df):
    x=pd.to_numeric(df.perf_5bd,errors="coerce").dropna().to_numpy(float) if not df.empty else np.array([])
    if not len(x): return {"n":0,"avg":0.,"median":0.,"robust_avg":0.,"wr":0.,"hits":0,"target_rate":0.}
    dec=x[x!=0]; wr=float(np.mean(dec>0)) if len(dec) else 0.
    if len(x)>=10:
        lo,hi=np.quantile(x,[.05,.95]); robust=float(np.mean(np.clip(x,lo,hi)))
    else: robust=float(np.mean(x))
    hits=int(np.sum(x>=.10))
    return {"n":int(len(x)),"avg":float(np.mean(x)),"median":float(np.median(x)),"robust_avg":robust,
            "wr":wr,"hits":hits,"target_rate":float(hits/len(x))}


def top_per_day(df,n):
    if df.empty:return df
    return df.sort_values(["date","prob"],ascending=[True,False]).groupby("date",sort=False).head(n)


def source_kind(alert_id):
    s=str(alert_id).lower()
    if s.startswith("tv_"):return "tv"
    if s.startswith("screenshot"):return "screenshot"
    if s.startswith("ss_"):return "ss"
    if s.startswith("manual"):return "manual"
    return "other"


def pct(x): return f"{x*100:.1f}%"


def make_report(r):
    c=r["classification"]["test"]; t=r["trading_test"]; ch=r["current_champion"]
    lines=["# No-TV V10 — Full BOTTOM History + Exact Historical Watchlists","",
      f"- Teacher BOTTOM: {r['teacher']['rows']}件 ({r['teacher']['date_min']} ～ {r['teacher']['date_max']})",
      f"- Teacher unique symbols: {r['teacher']['symbols']}",
      f"- Exact watchlist snapshots: {r['watchlists']['snapshot_commits']} commits",
      f"- Watchlist commit JST hour range: {r['watchlists']['commit_hour_min']:.2f} ～ {r['watchlists']['commit_hour_max']:.2f}",
      f"- Monitored candidate rows: {r['candidate_rows']}",
      f"- Teacher matched to candidates: {r['teacher']['matched_candidate_keys']} / monitored teacher keys: {r['teacher']['monitored_keys']}",
      f"- Train: {TRAIN_START} ～ {TRAIN_END}",f"- Validation: {VALID_START} ～ {VALID_END}",f"- Untouched test: {TEST_START} ～ {TEST_END}",
      f"- Validation-selected threshold: {r['threshold']['threshold']:.5f}","",
      "## Untouched August classification","","|Metric|Result|","|---|---:|",
      f"|Positive rate|{pct(c['positive_rate'])}|",f"|ROC-AUC|{c['roc_auc']:.3f}|" if c['roc_auc'] is not None else "|ROC-AUC|-|",
      f"|PR-AUC|{c['pr_auc']:.3f}|" if c['pr_auc'] is not None else "|PR-AUC|-|",
      f"|Precision|{pct(c['precision'])}|",f"|Recall|{pct(c['recall'])}|",f"|F1|{pct(c['f1'])}|",
      f"|Predicted/day|{c['predicted_per_day']:.2f}|",f"|False positive/day|{c['false_positives_per_day']:.2f}|",
      f"|Confusion|TN={c['tn']} / FP={c['fp']} / FN={c['fn']} / TP={c['tp']}|","",
      "## Untouched August 5BD","","|Variant|n|Avg|Median|Robust|Win|+10% hit|","|---|---:|---:|---:|---:|---:|---:|",
      f"|Production Stable★6 reference (1y)|{ch['n']}|{pct(ch['avg'])}|-|-|{pct(ch['wr'])}|{ch['hits']} ({pct(ch['target_rate'])})|",
    ]
    labels=[("teacher_all","Actual TV BOTTOM all"),("teacher_stable6","Actual TV BOTTOM ∩ current Stable★6"),
            ("model_all","V10 threshold"),("model_top30","V10 top30/day"),("model_stable6","V10 ∩ current Stable★6"),
            ("monitored_stable6","Monitored sessions ∩ current Stable★6")]
    for k,label in labels:
        s=t[k];lines.append(f"|{label}|{s['n']}|{pct(s['avg'])}|{pct(s['median'])}|{pct(s['robust_avg'])}|{pct(s['wr'])}|{s['hits']} ({pct(s['target_rate'])})|")
    lines += ["","## Guardrails","",
      "- Positive = actual saved BOTTOM history from weekly-report `alerts_raw + signals_archive`.",
      "- Negative = session that was actually inside the historical TradingView watchlist and did not become a saved BOTTOM.",
      "- Historical watchlists are read from the exact committed `tv-watchlist-builder/output` snapshots; no full-JPX false negatives.",
      "- 09/13 sessions also require the production signal-bar volume filter (>=5,000) because the feature builder applies it.",
      "- Threshold is chosen on July only. August is untouched until final evaluation.",
      "- September is excluded from model selection/test because 5BD outcomes are incomplete.",
      "- No production Sheets/Discord/main writes."]
    return "\n".join(lines)


def run(args):
    teacher=load_teacher(args.teacher)
    wl,wl_meta,wl_stats=load_exact_watchlists(args.watchlist_repo,TRAIN_START,TEST_END)
    monitor_keys=set()
    union=set()
    for d,symbols in wl.items():
        if TRAIN_START<=d<=TEST_END:
            union.update(symbols)
            monitor_keys.update(f"{d}|{s}" for s in symbols)

    teacher["monitor_key"]=teacher.signal_date+"|"+teacher.symbol_code
    teacher_mon=teacher[teacher.monitor_key.isin(monitor_keys)&(teacher.signal_date>=TRAIN_START)&(teacher.signal_date<=TEST_END)].copy()
    positive_keys=set(teacher_mon.key)

    _,jpx=fetch_jpx_issues_fixed(); by_code={str(i.code):i for i in jpx}
    issues=[]; missing=[]
    for code in sorted(union):
        if code in by_code: issues.append(by_code[code])
        else:
            # A recently delisted/foreign issue can still exist in a historical watchlist; core.Issue is enough for Yahoo fetch.
            issues.append(core.Issue(code=code,name=code,market="historical-watchlist")); missing.append(code)
    if args.max_symbols:
        # Deterministic smoke sample: highest number of monitored dates, but always include test positives when possible.
        freq={s:0 for s in union}
        for symbols in wl.values():
            for s in symbols:
                if s in freq:freq[s]+=1
        test_pos=set(teacher_mon.loc[(teacher_mon.signal_date>=TEST_START)&(teacher_mon.signal_date<=TEST_END),"symbol_code"])
        ordered=sorted(union,key=lambda s:(s not in test_pos,-freq[s],s))[:args.max_symbols]
        keep=set(ordered);issues=[i for i in issues if str(i.code) in keep]; union=keep
        monitor_keys={k for k in monitor_keys if k.split("|",1)[1] in keep}
        positive_keys={k for k in positive_keys if k.split("|",1)[0] in keep}

    frames=[];errors={};ok=0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        fut={ex.submit(fetch_one,i,TRAIN_START,FETCH_END):i for i in issues}
        for n,f in enumerate(as_completed(fut),1):
            try:fr,err=f.result()
            except Exception as e:fr,err=None,type(e).__name__
            if fr is not None:
                ok+=1
                if not fr.empty:frames.append(fr)
            elif err:errors[err]=errors.get(err,0)+1
            if n%200==0:print(f"progress {n}/{len(issues)} frames={len(frames)}",flush=True)
    if not frames:raise RuntimeError("no candidate rows")
    data=pd.concat(frames,ignore_index=True)
    data["symbol"]=data.symbol.astype(str).str.replace(r"\.0$","",regex=True)
    data["monitor_key"]=data.date.astype(str)+"|"+data.symbol
    data=data[data.monitor_key.isin(monitor_keys)&(data.date>=TRAIN_START)&(data.date<=TEST_END)].copy()
    data["key"]=data.symbol+"|"+data.date.astype(str)+"|"+data.session.astype(int).astype(str)
    data["label"]=data.key.isin(positive_keys).astype(int)

    train=data[(data.date>=TRAIN_START)&(data.date<=TRAIN_END)].copy()
    valid=data[(data.date>=VALID_START)&(data.date<=VALID_END)].copy()
    test=data[(data.date>=TEST_START)&(data.date<=TEST_END)].copy()
    if min(train.label.sum(),valid.label.sum(),test.label.sum())<10:
        raise RuntimeError(f"too few positives train={train.label.sum()} valid={valid.label.sum()} test={test.label.sum()}")

    model=HistGradientBoostingClassifier(learning_rate=.05,max_iter=260,max_leaf_nodes=31,min_samples_leaf=35,l2_regularization=1.5,random_state=42)
    ytr=train.label.to_numpy(int);model.fit(train[FEATURES].astype(float),ytr,sample_weight=balanced_weights(ytr))
    pv=model.predict_proba(valid[FEATURES].astype(float))[:,1];th=choose_threshold(valid.label.to_numpy(int),pv)
    pt=model.predict_proba(test[FEATURES].astype(float))[:,1];test["prob"]=pt
    pred=test[test.prob>=th["threshold"]].copy()

    matched_keys=set(data.loc[data.label==1,"key"])
    test_teacher=test[test.label==1]
    src_counts=teacher.alert_id.map(source_kind).value_counts().to_dict()
    result={
      "generated_at_jst":core.now_jst().isoformat(timespec="seconds"),
      "teacher":{"path":args.teacher,"rows":int(len(teacher)),"date_min":teacher.signal_date.min(),"date_max":teacher.signal_date.max(),
                 "symbols":int(teacher.symbol_code.nunique()),"source_counts":src_counts,"monitored_keys":int(len(positive_keys)),
                 "matched_candidate_keys":int(len(matched_keys)),"unmatched_monitored_keys":int(len(positive_keys-matched_keys))},
      "watchlists":{**wl_stats,"mapped_calendar_days":len(wl),"union_symbols":len(union)},
      "issues":{"requested":len(issues),"yahoo_ok":ok,"current_jpx_missing":missing,"errors":errors},
      "candidate_rows":int(len(data)),
      "split":{"train":{"n":int(len(train)),"pos":int(train.label.sum())},"valid":{"n":int(len(valid)),"pos":int(valid.label.sum())},"test":{"n":int(len(test)),"pos":int(test.label.sum())}},
      "features":FEATURES,"model":"HistGradientBoostingClassifier","threshold":th,
      "classification":{"validation":class_stats(valid,pv,th["threshold"]),"test":class_stats(test,pt,th["threshold"])},
      "current_champion":CHAMPION,
      "trading_test":{"teacher_all":trade_stats(test_teacher),"teacher_stable6":trade_stats(test_teacher[test_teacher.stable_score==6]),
          "model_all":trade_stats(pred),"model_top30":trade_stats(top_per_day(test.sort_values("prob",ascending=False),30)),
          "model_stable6":trade_stats(pred[pred.stable_score==6]),"monitored_stable6":trade_stats(test[test.stable_score==6])},
      "test_top_predictions":test.sort_values("prob",ascending=False)[["date","session","symbol","name","prob","label","stable_score","perf_5bd"]].head(200).to_dict("records"),
    }
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--teacher",default=TEACHER_DEFAULT);p.add_argument("--watchlist-repo",default="tv-watchlist-builder-history")
    p.add_argument("--max-workers",type=int,default=24);p.add_argument("--max-symbols",type=int);p.add_argument("--output-dir",default="reports_no_tv_v10_full")
    a=p.parse_args();r=run(a);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"comparison_v10_full.json").write_text(json.dumps(r,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    (out/"comparison_v10_full.md").write_text(make_report(r),encoding="utf-8")
    print(json.dumps({"teacher":r["teacher"],"watchlists":r["watchlists"],"split":r["split"],"threshold":r["threshold"],"classification":r["classification"],"trading_test":r["trading_test"]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
