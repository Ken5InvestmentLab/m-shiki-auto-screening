from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

import screen_big_money as core
from screen_entry import fetch_jpx_issues_fixed
from no_tv_v10_features import FEATURES, fetch_one

TEACHER_DEFAULT = "experiments/v10_teacher_legacy_score6_bottom.csv"
CHAMPION = {"name":"Production TradingView Stable★6","n":55,"avg":0.066,"wr":0.564,"hits":10,"target_rate":10/55}


def load_teacher(path):
    t = pd.read_csv(path, dtype={"symbol_code": str})
    t["signal_date"] = pd.to_datetime(t.signal_date).dt.strftime("%Y-%m-%d")
    t["bar_time"] = pd.to_datetime(t.bar_time)
    t["session"] = t.bar_time.dt.hour.astype(int)
    t["symbol_code"] = t.symbol_code.astype(str).str.strip()
    t["key"] = t.symbol_code + "|" + t.signal_date + "|" + t.session.astype(str)
    return t


def balanced_weights(y):
    y = np.asarray(y, dtype=int); n = len(y)
    pos = max(1, int(y.sum())); neg = max(1, n - pos)
    return np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg))


def choose_threshold(y, p):
    best = None
    cand = set(np.linspace(0.02, 0.98, 193).tolist())
    if len(p):
        cand.update(np.quantile(p, np.linspace(0.50, 0.999, 120)).tolist())
    for th in sorted(cand):
        pred = (p >= th).astype(int); n_pred = int(pred.sum())
        if n_pred < 3:
            continue
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        rank = f1 + 0.05 * prec
        if best is None or rank > best["rank"]:
            best = {"threshold":float(th),"precision":float(prec),"recall":float(rec),"f1":float(f1),"predicted":n_pred,"rank":float(rank)}
    return best or {"threshold":0.5,"precision":0.0,"recall":0.0,"f1":0.0,"predicted":0,"rank":0.0}


def class_stats(y, p, th, dates):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); pred = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    ap = average_precision_score(y, p) if y.sum() > 0 else np.nan
    ndays = max(1, pd.Series(dates).nunique())
    return {
        "n":int(len(y)),"positive":int(y.sum()),"positive_rate":float(y.mean()) if len(y) else 0.0,
        "roc_auc":float(auc) if np.isfinite(auc) else None,"pr_auc":float(ap) if np.isfinite(ap) else None,
        "accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"predicted":int(pred.sum()),
        "false_positives_per_day":float(fp/ndays),"predicted_per_day":float(pred.sum()/ndays),
    }


def trade_stats(e):
    x = pd.to_numeric(e.perf_5bd, errors="coerce").dropna().to_numpy(float) if not e.empty else np.array([])
    if not len(x):
        return {"n":0,"avg":0.0,"median":0.0,"robust_avg":0.0,"wr":0.0,"hits":0,"target_rate":0.0}
    d = x[x != 0]; wr = float(np.mean(d > 0)) if len(d) else 0.0
    if len(x) >= 10:
        lo, hi = np.quantile(x,[0.05,0.95]); robust = float(np.mean(np.clip(x,lo,hi)))
    else:
        robust = float(np.mean(x))
    hits = int(np.sum(x >= 0.10))
    return {"n":int(len(x)),"avg":float(np.mean(x)),"median":float(np.median(x)),"robust_avg":robust,
            "wr":wr,"hits":hits,"target_rate":float(hits/len(x))}


def top_per_day(e, n=3):
    if e.empty: return e
    return e.sort_values(["date","prob"],ascending=[True,False]).groupby("date",sort=False).head(n)


def pct(v):
    return "-" if v is None else f"{v*100:.1f}%"


def markdown(r):
    c = r["classification"]["test"]; t = r["trading_test"]; ch = r["current_champion"]
    lines = [
        "# No-TV V10 — Full-market Distillation OOS", "",
        "**Teacher caveat:** positive labels are the saved legacy `6点満点銘柄` BOTTOM records, not the current Stable★6 definition. Ordinary eligible sessions are negatives, so unrecorded lower-score BOTTOMs can create label noise.", "",
        f"- Teacher rows: {r['teacher']['rows']} / matched: {r['teacher']['matched']} / unmatched: {r['teacher']['unmatched']}",
        f"- Universe: {r['universe_count']} / Yahoo success: {r['yahoo_ok']}",
        f"- Candidate rows: {r['candidate_rows']}",
        "- Train: 2026-03-05 ～ 2026-03-31",
        "- Validation: 2026-04-01 ～ 2026-04-15",
        "- Untouched test: 2026-04-16 ～ 2026-04-30",
        f"- Validation-selected threshold: {r['threshold']['threshold']:.4f}", "",
        "## Untouched test — teacher classification", "",
        "|Metric|Result|","|---|---:|",
        f"|ROC-AUC|{c['roc_auc']:.3f}|" if c['roc_auc'] is not None else "|ROC-AUC|-|",
        f"|PR-AUC|{c['pr_auc']:.3f}|" if c['pr_auc'] is not None else "|PR-AUC|-|",
        f"|Precision|{pct(c['precision'])}|",f"|Recall|{pct(c['recall'])}|",f"|F1|{pct(c['f1'])}|",
        f"|FP/day|{c['false_positives_per_day']:.2f}|",f"|Predicted/day|{c['predicted_per_day']:.2f}|",
        f"|Confusion|TN={c['tn']} / FP={c['fp']} / FN={c['fn']} / TP={c['tp']}|", "",
        "## Untouched test — 5BD trading performance", "",
        "|Variant|n|Avg|Median|Robust avg|Win rate|+10% hit|","|---|---:|---:|---:|---:|---:|---:|",
        f"|Production Stable★6 champion|{ch['n']}|{pct(ch['avg'])}|-|-|{pct(ch['wr'])}|{ch['hits']} ({pct(ch['target_rate'])})|",
    ]
    for key,label in [
        ("model_all","V10 model threshold"),("model_top3","V10 model top3/day"),
        ("model_and_current_stable6","V10 model ∩ current Stable★6"),
        ("current_stable6_only","Yahoo current Stable★6 only"),("teacher_positive","Teacher positive (test matched)"),
    ]:
        s=t[key]
        lines.append(f"|{label}|{s['n']}|{pct(s['avg'])}|{pct(s['median'])}|{pct(s['robust_avg'])}|{pct(s['wr'])}|{s['hits']} ({pct(s['target_rate'])})|")
    lines += ["","## Guardrails","",
        "- Champion is a one-year/current Stable★6 sample; V10 test is only 2026-04-16..2026-04-30. No promotion from this short window alone.",
        "- ROC-AUC alone is insufficient under class imbalance; PR-AUC and false positives/day are primary checks.",
        "- Threshold is selected on validation only; test is untouched until final evaluation.",
        "- TradingView is not used for V10 candidate generation. Historical TV-derived labels are teacher data only.",
        "- This workflow does not write to production Sheets/Discord or main."]
    return "\n".join(lines)


def run(args):
    teacher = load_teacher(args.teacher)
    start_date, end_date = teacher.signal_date.min(), teacher.signal_date.max()
    teacher_symbols = set(teacher.symbol_code.astype(str))
    _, issues = fetch_jpx_issues_fixed(); issues = [i for i in issues if "優先" not in i.name]
    if args.max_issues:
        base = issues[:args.max_issues]; by_code = {str(i.code):i for i in issues}
        include = [by_code[s] for s in teacher_symbols if s in by_code]
        seen=set(); merged=[]
        for i in base+include:
            if str(i.code) not in seen: merged.append(i); seen.add(str(i.code))
        issues=merged

    frames=[]; errors={}; ok=0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        fut={ex.submit(fetch_one,i,start_date,end_date):i for i in issues}
        for n,f in enumerate(as_completed(fut),1):
            try: fr,err=f.result()
            except Exception as e: fr,err=None,type(e).__name__
            if fr is not None:
                ok+=1
                if not fr.empty: frames.append(fr)
            elif err: errors[err]=errors.get(err,0)+1
            if n%250==0: print(f"progress {n}/{len(issues)} frames={len(frames)}",flush=True)
    if not frames: raise RuntimeError("no candidate rows")
    data=pd.concat(frames,ignore_index=True)
    data["key"]=data.symbol.astype(str)+"|"+data.date.astype(str)+"|"+data.session.astype(str)
    poskeys=set(teacher.key); data["label"]=data.key.isin(poskeys).astype(int)
    matched=set(data.loc[data.label==1,"key"]); unmatched=sorted(poskeys-matched)

    train=data[(data.date>="2026-03-05")&(data.date<="2026-03-31")].copy()
    valid=data[(data.date>="2026-04-01")&(data.date<="2026-04-15")].copy()
    test=data[(data.date>="2026-04-16")&(data.date<="2026-04-30")].copy()
    if min(train.label.sum(),valid.label.sum(),test.label.sum())<3:
        raise RuntimeError(f"too few matched positives train={train.label.sum()} valid={valid.label.sum()} test={test.label.sum()}")

    model=HistGradientBoostingClassifier(learning_rate=.05,max_iter=220,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.0,random_state=42)
    ytr=train.label.to_numpy(int); model.fit(train[FEATURES].astype(float),ytr,sample_weight=balanced_weights(ytr))
    pv=model.predict_proba(valid[FEATURES].astype(float))[:,1]; th=choose_threshold(valid.label.to_numpy(int),pv)
    pt=model.predict_proba(test[FEATURES].astype(float))[:,1]; test["prob"]=pt
    pred=test[test.prob>=th["threshold"]].copy()
    result={
        "generated_at_jst":core.now_jst().isoformat(timespec="seconds"),
        "teacher":{"path":args.teacher,"definition":"legacy saved 6-point BOTTOM; NOT current Stable★6","rows":int(len(teacher)),"matched":int(len(matched)),"unmatched":int(len(unmatched)),"unmatched_keys":unmatched},
        "universe_count":len(issues),"yahoo_ok":ok,"errors":errors,"candidate_rows":int(len(data)),"candidate_positive_rate":float(data.label.mean()),
        "split":{"train":["2026-03-05","2026-03-31"],"valid":["2026-04-01","2026-04-15"],"test":["2026-04-16","2026-04-30"],
                 "counts":{"train":{"n":int(len(train)),"pos":int(train.label.sum())},"valid":{"n":int(len(valid)),"pos":int(valid.label.sum())},"test":{"n":int(len(test)),"pos":int(test.label.sum())}}},
        "features":FEATURES,"model":"HistGradientBoostingClassifier","threshold":th,
        "classification":{"validation":class_stats(valid.label.to_numpy(int),pv,th["threshold"],valid.date),"test":class_stats(test.label.to_numpy(int),pt,th["threshold"],test.date)},
        "current_champion":CHAMPION,
        "trading_test":{"model_all":trade_stats(pred),"model_top3":trade_stats(top_per_day(pred,3)),
            "model_and_current_stable6":trade_stats(pred[pred.stable_score==6]),"current_stable6_only":trade_stats(test[test.stable_score==6]),
            "teacher_positive":trade_stats(test[test.label==1])},
        "test_top_predictions":test.sort_values("prob",ascending=False)[["date","session","symbol","name","prob","label","stable_score","perf_5bd"]].head(100).to_dict("records"),
    }
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--teacher",default=TEACHER_DEFAULT); p.add_argument("--max-workers",type=int,default=24)
    p.add_argument("--max-issues",type=int); p.add_argument("--output-dir",default="reports_no_tv_v10"); args=p.parse_args()
    r=run(args); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"comparison_v10.json").write_text(json.dumps(r,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    (out/"comparison_v10.md").write_text(markdown(r),encoding="utf-8")
    print(json.dumps({"teacher":r["teacher"],"split":r["split"],"threshold":r["threshold"],"classification":r["classification"],"trading_test":r["trading_test"]},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
