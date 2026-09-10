from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import screen_big_money as core
from screen_entry import fetch_jpx_issues_fixed
import independent_daily_lab_v2 as v2


CURRENT_REFERENCE = {
    "generated_at": "2026-09-09 19:41:50 JST",
    "stable_s6": {"n": 55, "avg": .066, "wr": .564, "hits": 10, "target_rate": 10 / 55},
}

EXACT_PROFILE = next(p for p in v2.PINE_PROFILES if p["id"] == "exact")

FEATURES = [
    "ret1","ret3","ret5","ret10","ret20","ret60",
    "gap","body_pct","range_pct","close_loc","upper_wick","lower_wick",
    "atr_pct","rv5","rv20","rsi12","rsi14","stoch14","rci9","cci14",
    "bb_pos","bb_width","bb_width_rel60",
    "dist_ema10","dist_ema25","dist_ema50","dist_ema75","dist_ema200",
    "ema25_slope5","ema75_slope10","macd_hist_pct",
    "pos20","pos60","pos252","dd20","dd60","dd120",
    "break20","break60","from_low20",
    "vol_ratio5","vol_ratio20","vol_ratio60","vol_z20",
    "turnover_ratio20","money_flow20","range_mean5","range_mean20",
    "down_count3","up_count3",
    "tr_pine_exact","tr_breakout","tr_reversal","tr_volume_reversal",
    "tr_squeeze","tr_pullback","tr_trend_resume","tr_accumulation","tr_momentum",
    "xrank_ret5","xrank_ret20","xrank_vol20","xrank_atr","xrank_rsi12",
    "xrank_dd60","xrank_bbwidth","xrank_close_loc",
]

BLENDS = {
    "balanced": (0.45, 0.35, 0.20),
    "hit10": (0.30, 0.50, 0.20),
    "positive": (0.65, 0.15, 0.20),
    "return": (0.35, 0.20, 0.45),
}


def parse_date(v):
    try:
        return date.fromisoformat(v).isoformat()
    except ValueError as e:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from e


def rolling_rci9(close: pd.Series) -> pd.Series:
    return close.rolling(9, min_periods=9).apply(v2.rci9, raw=True)


def build_symbol_frame(issue, chart, history_start, end_date):
    parsed = core.chart_to_arrays(chart, None)
    if parsed is None:
        return None
    dates, a = parsed
    df = pd.DataFrame({
        "date": dates.astype(str),
        "open": a["open"], "high": a["high"], "low": a["low"],
        "close": a["close"], "volume": a["volume"],
    })
    if len(df) < 220:
        return None

    o = df.open.astype(float)
    h = df.high.astype(float)
    l = df.low.astype(float)
    c = df.close.astype(float)
    vol = df.volume.astype(float)
    prev = c.shift(1)
    ret = c.pct_change()

    df["bar_index"] = np.arange(len(df))
    df["base"] = (prev <= 1000) & (vol.shift(1) >= 10000) & (vol >= 5000)

    for n in (1,3,5,10,20,60):
        df[f"ret{n}"] = c / c.shift(n) - 1.0

    df["gap"] = o / prev - 1.0
    df["body_pct"] = (c - o) / prev
    df["range_pct"] = (h - l) / prev

    candle_range = (h-l).replace(0,np.nan)
    df["close_loc"] = ((c-l)/candle_range).clip(0,1).fillna(.5)
    df["upper_wick"] = ((h-np.maximum(o,c))/candle_range).clip(0,1).fillna(0)
    df["lower_wick"] = ((np.minimum(o,c)-l)/candle_range).clip(0,1).fillna(0)

    atr_abs = pd.Series(v2.pine_atr(h.to_numpy(), l.to_numpy(), c.to_numpy(), 14), index=df.index)
    df["atr_pct"] = atr_abs / c
    df["rv5"] = ret.rolling(5,min_periods=5).std(ddof=0)
    df["rv20"] = ret.rolling(20,min_periods=20).std(ddof=0)

    df["rsi12"] = v2.pine_rsi(c.to_numpy(),12)
    df["rsi14"] = v2.pine_rsi(c.to_numpy(),14)

    lo14 = l.rolling(14,min_periods=14).min()
    hi14 = h.rolling(14,min_periods=14).max()
    df["stoch14"] = ((c-lo14)/(hi14-lo14).replace(0,np.nan)).clip(0,1)
    df["rci9"] = rolling_rci9(c) / 100.0

    tp=(h+l+c)/3
    tm=tp.rolling(14,min_periods=14).mean()
    md=tp.rolling(14,min_periods=14).apply(lambda x: float(np.mean(np.abs(x-np.mean(x)))),raw=True)
    df["cci14"]=(tp-tm)/(0.015*md.replace(0,np.nan))

    bm=c.rolling(20,min_periods=20).mean()
    bs=c.rolling(20,min_periods=20).std(ddof=0)
    upper=bm+2*bs
    lower=bm-2*bs
    df["bb_pos"]=((c-lower)/(upper-lower).replace(0,np.nan))
    df["bb_width"]=(upper-lower)/bm.replace(0,np.nan)
    df["bb_width_rel60"]=df.bb_width/df.bb_width.shift(1).rolling(60,min_periods=30).median()

    ema={}
    for n in (10,12,25,26,50,75,200):
        ema[n]=c.ewm(span=n,adjust=False,min_periods=n).mean()
    for n in (10,25,50,75,200):
        df[f"dist_ema{n}"]=c/ema[n]-1
    df["ema25_slope5"]=ema[25]/ema[25].shift(5)-1
    df["ema75_slope10"]=ema[75]/ema[75].shift(10)-1
    macd=ema[12]-ema[26]
    macd_sig=macd.ewm(span=9,adjust=False,min_periods=9).mean()
    df["macd_hist_pct"]=(macd-macd_sig)/c

    for n in (20,60,252):
        hh=h.shift(1).rolling(n,min_periods=max(10,min(n,60))).max()
        ll=l.shift(1).rolling(n,min_periods=max(10,min(n,60))).min()
        df[f"pos{n}"]=((c-ll)/(hh-ll).replace(0,np.nan))
        if n in (20,60):
            df[f"dd{n}"]=c/hh-1
    hh120=h.shift(1).rolling(120,min_periods=60).max()
    df["dd120"]=c/hh120-1

    hh20=h.shift(1).rolling(20,min_periods=20).max()
    hh60=h.shift(1).rolling(60,min_periods=40).max()
    ll20=l.shift(1).rolling(20,min_periods=20).min()
    df["break20"]=c/hh20-1
    df["break60"]=c/hh60-1
    df["from_low20"]=c/ll20-1

    for n in (5,20,60):
        vm=vol.shift(1).rolling(n,min_periods=n).mean()
        df[f"vol_ratio{n}"]=vol/vm
    vm20=vol.shift(1).rolling(20,min_periods=20).mean()
    vs20=vol.shift(1).rolling(20,min_periods=20).std(ddof=0)
    df["vol_z20"]=(vol-vm20)/vs20.replace(0,np.nan)

    turnover=c*vol
    tm20=turnover.shift(1).rolling(20,min_periods=20).mean()
    df["turnover_ratio20"]=turnover/tm20

    signed=np.sign(ret.fillna(0))*vol
    df["money_flow20"]=signed.rolling(20,min_periods=20).sum()/vol.rolling(20,min_periods=20).sum()
    df["range_mean5"]=df.range_pct.rolling(5,min_periods=5).mean()
    df["range_mean20"]=df.range_pct.rolling(20,min_periods=20).mean()

    down1=(c.shift(1)<c.shift(2)).astype(float)
    down2=(c.shift(2)<c.shift(3)).astype(float)
    down3=(c.shift(3)<c.shift(4)).astype(float)
    up1=(c.shift(1)>c.shift(2)).astype(float)
    up2=(c.shift(2)>c.shift(3)).astype(float)
    up3=(c.shift(3)>c.shift(4)).astype(float)
    df["down_count3"]=down1+down2+down3
    df["up_count3"]=up1+up2+up3

    bottom, _, _ = v2.pine_state_signals(c.to_numpy(),h.to_numpy(),l.to_numpy(),EXACT_PROFILE)
    df["tr_pine_exact"]=bottom.astype(float)
    df["tr_breakout"]=((df.break20>0)&(df.vol_ratio20>=1.2)&(df.close_loc>=.55)).astype(float)
    df["tr_reversal"]=((pd.Series(df.rsi12,index=df.index).shift(1)<42)&(df.rsi12>pd.Series(df.rsi12,index=df.index).shift(1))&(df.close_loc>=.55)).astype(float)
    df["tr_volume_reversal"]=((df.vol_ratio20>=2)&(df.ret1>0)&(df.close_loc>=.65)).astype(float)
    df["tr_squeeze"]=((df.bb_width_rel60<=.8)&(c>bm)&(df.ret1>0)&(df.close_loc>=.6)).astype(float)
    df["tr_pullback"]=((df.dd60<=-.08)&(df.dd60>=-.35)&(df.ret1>=.015)&(df.close_loc>=.6)).astype(float)
    df["tr_trend_resume"]=((c>ema[25])&(df.ema25_slope5>0)&(df.ret5>=-.08)&(df.ret5<=.06)&(df.ret1>0)).astype(float)
    df["tr_accumulation"]=((df.turnover_ratio20>=1.5)&(df.ret1.abs()<=.03)&(df.close_loc>=.55)&(df.upper_wick<=.35)).astype(float)
    df["tr_momentum"]=((df.ret20>=0)&(df.ret20<=.25)&(df.ret5>0)&(df.vol_ratio20>=1.2)&(df.bb_pos>=.6)).astype(float)

    trigger_cols=[x for x in df.columns if x.startswith("tr_")]
    df["candidate"]=(df[trigger_cols].sum(axis=1)>0)

    # 現行Stable★6 / V2 Stableの比較用boolean
    df["s_ema25"]=c>ema[25]
    df["s_macdpos"]=(macd-macd_sig)>0
    df["s_stoch75"]=df.stoch14>=.75
    df["s_bb80"]=df.bb_pos>=.80
    df["s_pre_down3"]=(c.shift(1)<c.shift(2))&(c.shift(2)<c.shift(3))
    df["s_gapup"]=df.gap>0
    df["s_atr5"]=df.atr_pct<.05
    df["s_hb20"]=df.break20>0

    for n in (5,10,20,40):
        df[f"perf_{n}bd"]=c.shift(-n)/c-1
        df[f"exit_date_{n}bd"]=df.date.shift(-n)

    use=(df.date>=history_start)&(df.date<=end_date)&df.base&df.candidate
    cols=["date","bar_index","close","volume","perf_5bd","perf_10bd","perf_20bd","perf_40bd",
          "exit_date_5bd","exit_date_10bd","exit_date_20bd","exit_date_40bd"]+FEATURES+[
          "s_ema25","s_macdpos","s_stoch75","s_bb80","s_pre_down3","s_gapup","s_atr5","s_hb20"
    ]
    out=df.loc[use,cols].copy()
    out["symbol"]=issue.code
    out["name"]=issue.name
    out["market"]=issue.market
    return out


def fetch_one(issue, history_start, end_date):
    cfg=core.ScreeningConfig(yahoo_range="5y",yahoo_interval="1d")
    chart,err=core.fetch_chart(issue,cfg)
    if err:
        return None,err
    frame=build_symbol_frame(issue,chart or {},history_start,end_date)
    if frame is None:
        return None,"too_few_rows"
    return frame,None


def add_cross_sectional_ranks(df):
    rank_map={
        "ret5":"xrank_ret5","ret20":"xrank_ret20","vol_ratio20":"xrank_vol20",
        "atr_pct":"xrank_atr","rsi12":"xrank_rsi12","dd60":"xrank_dd60",
        "bb_width":"xrank_bbwidth","close_loc":"xrank_close_loc",
    }
    for src,dst in rank_map.items():
        df[dst]=df.groupby("date",sort=False)[src].rank(pct=True,method="average")
    return df


def stats(events):
    if events.empty:
        return {"n":0,"decisive_n":0,"avg":0.0,"robust_avg":0.0,"median":0.0,"wr":0.0,"hits":0,"target_rate":0.0}
    v=pd.to_numeric(events.perf_5bd,errors="coerce").dropna().to_numpy(float)
    if not len(v):
        return {"n":0,"decisive_n":0,"avg":0.0,"robust_avg":0.0,"median":0.0,"wr":0.0,"hits":0,"target_rate":0.0}
    dec=v[v!=0]
    wr=float(np.mean(dec>0)) if len(dec) else 0.0
    if len(v)>=10:
        lo,hi=np.quantile(v,[.05,.95]); robust=float(np.mean(np.clip(v,lo,hi)))
    else:
        robust=float(np.mean(v))
    hits=int(np.sum(v>=.10))
    return {"n":int(len(v)),"decisive_n":int(len(dec)),"avg":float(np.mean(v)),
            "robust_avg":robust,"median":float(np.median(v)),"wr":wr,
            "hits":hits,"target_rate":float(hits/len(v))}


def split_periods(df, holdout_start):
    pre=sorted(df.loc[df.date<holdout_start,"date"].unique())
    if len(pre)<100:
        raise RuntimeError("not enough pre-holdout dates")
    cut=int(len(pre)*.75)
    return {
        "train_start":pre[0],"train_end":pre[cut-1],
        "valid_start":pre[cut],"valid_end":pre[-1],
        "holdout_start":holdout_start,
    }


def period(df,start,end):
    return df[(df.date>=start)&(df.date<=end)&df.exit_date_5bd.notna()&(df.exit_date_5bd<=end)].copy()


def balanced_weights(y):
    y=np.asarray(y,int)
    n=len(y); pos=max(1,int(y.sum())); neg=max(1,n-pos)
    return np.where(y==1,n/(2*pos),n/(2*neg))


def make_models():
    linear=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("clf",LogisticRegression(max_iter=500,class_weight="balanced",C=.5,n_jobs=1)),
    ])
    hgb_pos=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("clf",HistGradientBoostingClassifier(
            learning_rate=.06,max_iter=220,max_leaf_nodes=15,min_samples_leaf=35,
            l2_regularization=1.0,random_state=42
        )),
    ])
    hgb_hit=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("clf",HistGradientBoostingClassifier(
            learning_rate=.05,max_iter=250,max_leaf_nodes=15,min_samples_leaf=25,
            l2_regularization=1.5,random_state=43
        )),
    ])
    reg=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("reg",HistGradientBoostingRegressor(
            learning_rate=.05,max_iter=240,max_leaf_nodes=15,min_samples_leaf=30,
            l2_regularization=1.0,loss="squared_error",random_state=44
        )),
    ])
    return linear,hgb_pos,hgb_hit,reg


def fit_models(train):
    X=train[FEATURES]
    y_pos=(train.perf_5bd>0).astype(int)
    y_hit=(train.perf_5bd>=.10).astype(int)
    y_ret=train.perf_5bd.clip(-.20,.35)

    linear,hgb_pos,hgb_hit,reg=make_models()
    linear.fit(X,y_pos)
    hgb_pos.fit(X,y_pos,clf__sample_weight=balanced_weights(y_pos))
    hgb_hit.fit(X,y_hit,clf__sample_weight=balanced_weights(y_hit))
    reg.fit(X,y_ret,reg__sample_weight=(1+2*y_hit.to_numpy()))
    return linear,hgb_pos,hgb_hit,reg


def predict_components(models, frame):
    linear,hgb_pos,hgb_hit,reg=models
    X=frame[FEATURES]
    p_lin=linear.predict_proba(X)[:,1]
    p_hgb=hgb_pos.predict_proba(X)[:,1]
    p_pos=.35*p_lin+.65*p_hgb
    p_hit=hgb_hit.predict_proba(X)[:,1]
    pred_ret=reg.predict(X)
    return p_pos,p_hit,pred_ret


def ecdf_map(values, reference):
    ref=np.sort(np.asarray(reference,float))
    v=np.asarray(values,float)
    if not len(ref):
        return np.full(len(v),.5)
    return np.searchsorted(ref,v,side="right")/len(ref)


def score_frame(frame, comps, ret_reference, blend):
    p_pos,p_hit,pred_ret=comps
    ret_rank=ecdf_map(pred_ret,ret_reference)
    a,b,c=blend
    out=frame.copy()
    out["p_pos"]=p_pos
    out["p_hit10"]=p_hit
    out["pred_ret"]=pred_ret
    out["score100"]=np.clip(100*(a*p_pos+b*p_hit+c*ret_rank),0,100)
    return out


def cooldown(events,bars=5):
    if events.empty:
        return events
    keep=[]
    for _,g in events.sort_values(["symbol","bar_index"]).groupby("symbol",sort=False):
        last=-10**9
        for idx,row in g.iterrows():
            bi=int(row.bar_index)
            if bi-last>=bars:
                keep.append(idx); last=bi
    return events.loc[keep].sort_values(["date","score100"],ascending=[True,False]).reset_index(drop=True)


def apply_policy(scored,threshold,top_per_day):
    e=scored[scored.score100>=threshold].copy()
    if e.empty:
        return e
    e=e.sort_values(["date","score100"],ascending=[True,False]).groupby("date",sort=False).head(top_per_day)
    return cooldown(e,5)


def objective(s):
    if s["n"]<35 or s["n"]>220:
        return -1e9
    if s["robust_avg"]<=0:
        return -1e8+s["robust_avg"]
    return (
        0.45*s["wr"]
        +2.80*s["robust_avg"]
        +0.50*s["target_rate"]
        +0.012*math.log1p(s["n"])
    )


def choose_policy(valid_base,valid_components,ret_reference):
    best=None
    for name,blend in BLENDS.items():
        scored=score_frame(valid_base,valid_components,ret_reference,blend)
        qs=np.unique(np.quantile(scored.score100,[.70,.75,.80,.85,.88,.90,.92,.94,.95,.96,.97,.98,.985,.99]))
        for threshold in qs:
            for k in (1,2,3,5,8):
                ev=apply_policy(scored,float(threshold),k)
                s=stats(ev)
                rank=objective(s)
                if best is None or rank>best["rank"]:
                    best={"blend_name":name,"blend":blend,"threshold":float(threshold),
                          "top_per_day":k,"valid_stats":s,"rank":rank}
    if best is None:
        raise RuntimeError("no V3 policy")
    return best


def baseline_events(frame,kind):
    if kind=="exact_current":
        m=(frame.tr_pine_exact>0)&frame.s_ema25&frame.s_macdpos&frame.s_stoch75&frame.s_bb80&frame.s_pre_down3&frame.s_gapup
    elif kind=="v2":
        m=(frame.tr_pine_exact>0)&frame.s_ema25&frame.s_macdpos&frame.s_atr5&frame.s_hb20&frame.s_pre_down3&frame.s_gapup
    else:
        raise ValueError(kind)
    e=frame[m].copy()
    e["score100"]=100.0
    return cooldown(e,5)


def report_md(result):
    cur=result["current_reference"]
    ex=result["holdout"]["pine_exact_current"]
    vv=result["holdout"]["v2"]
    m=result["holdout"]["v3"]
    out=[
        "# No-TV Stable V3 — 0〜100点スコア",
        "",
        f"- 生成: {result['generated_at_jst']}",
        f"- データ期間: {result['history_start']} ～ {result['end_date']}",
        f"- 完全未使用Holdout: {result['holdout_start']} ～ {result['end_date']}",
        f"- JPX対象: {result['universe_count']} / Yahoo成功: {result['yahoo_ok']}",
        f"- V3候補row: {result['candidate_rows']}",
        "",
        "|方式|n|5BD平均|Robust平均|勝率|+10% Hit|Hit率|",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"|現行4H Stable★6|{cur['n']}|{cur['avg']*100:+.1f}%|-|{cur['wr']*100:.1f}%|{cur['hits']}|{cur['target_rate']*100:.1f}%|",
        f"|日足 Pine Exact + 現行Score|{ex['n']}|{ex['avg']*100:+.1f}%|{ex['robust_avg']*100:+.1f}%|{ex['wr']*100:.1f}%|{ex['hits']}|{ex['target_rate']*100:.1f}%|",
        f"|日足 V2|{vv['n']}|{vv['avg']*100:+.1f}%|{vv['robust_avg']*100:+.1f}%|{vv['wr']*100:.1f}%|{vv['hits']}|{vv['target_rate']*100:.1f}%|",
        f"|**日足 V3 ML Score**|**{m['n']}**|**{m['avg']*100:+.1f}%**|**{m['robust_avg']*100:+.1f}%**|**{m['wr']*100:.1f}%**|**{m['hits']}**|**{m['target_rate']*100:.1f}%**|",
        "",
        "## V3 policy（Holdout未使用）",
        f"- blend: {result['policy']['blend_name']} {result['policy']['blend']}",
        f"- score threshold: {result['policy']['threshold']:.2f}",
        f"- top/day: {result['policy']['top_per_day']}",
        f"- Validation: {json.dumps(result['policy']['valid_stats'],ensure_ascii=False)}",
        "",
        "## V3 Score構成",
        "- 上昇確率: Logistic + HistGradientBoosting",
        "- +10%到達確率: HistGradientBoosting",
        "- 5BD期待リターン: HistGradientBoosting Regressor",
        "- 上記をValidationだけで選んだblendで0〜100点化。",
        "- Holdoutはモデル・重み・閾値・top/day選択に一切使用しない。",
        "- 同一銘柄は5営業日クールダウン。",
        "",
        "## Candidate Trigger",
        "- Pine状態遷移 / 20日高値突破 / RSI反転 / 出来高反転 / BB収縮→拡大 / 深押し反発 / トレンド再開 / 静かな蓄積 / モメンタム。",
        "- TradingViewデータ・TradingViewアラートは使用しない。",
    ]
    return "\n".join(out)


def run(args):
    end_d=date.fromisoformat(args.end_date)
    history_start=(end_d-timedelta(days=args.history_days)).isoformat()
    holdout_start=(end_d-timedelta(days=args.holdout_days)).isoformat()

    list_date,issues=fetch_jpx_issues_fixed()
    issues=[i for i in issues if "優先" not in i.name]
    if args.max_issues:
        issues=issues[:args.max_issues]

    frames=[]; errors={}; ok=0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs={ex.submit(fetch_one,i,history_start,args.end_date):i for i in issues}
        for n,f in enumerate(as_completed(futs),1):
            try:
                frame,err=f.result()
            except Exception as e:
                frame,err=None,type(e).__name__
            if frame is not None and not frame.empty:
                frames.append(frame); ok+=1
            elif err:
                errors[err]=errors.get(err,0)+1
            if n%500==0:
                print(f"progress {n}/{len(issues)} frames={len(frames)}",flush=True)

    if not frames:
        raise RuntimeError("no candidate data")
    df=pd.concat(frames,ignore_index=True)
    df=add_cross_sectional_ranks(df)
    df=df.sort_values(["date","symbol","bar_index"]).reset_index(drop=True)

    split=split_periods(df,holdout_start)
    train=period(df,split["train_start"],split["train_end"])
    valid=period(df,split["valid_start"],split["valid_end"])
    hold=period(df,holdout_start,args.end_date)
    if min(len(train),len(valid),len(hold))==0:
        raise RuntimeError("empty split")

    print(f"rows train={len(train)} valid={len(valid)} holdout={len(hold)}",flush=True)
    models=fit_models(train)
    train_components=predict_components(models,train)
    valid_components=predict_components(models,valid)
    hold_components=predict_components(models,hold)
    # 期待リターン順位の基準もHoldoutを見ずTrain predictionだけで固定
    ret_reference=train_components[2]

    policy=choose_policy(valid,valid_components,ret_reference)
    hold_scored=score_frame(hold,hold_components,ret_reference,tuple(policy["blend"]))
    v3_events=apply_policy(hold_scored,policy["threshold"],policy["top_per_day"])

    exact_events=baseline_events(hold,"exact_current")
    v2_events=baseline_events(hold,"v2")

    result={
        "generated_at_jst":core.now_jst().isoformat(timespec="seconds"),
        "history_start":history_start,"holdout_start":holdout_start,"end_date":args.end_date,
        "jpx_list_date":list_date,"universe_count":len(issues),"yahoo_ok":ok,
        "errors":errors,"candidate_rows":len(df),"split":split,
        "current_reference":CURRENT_REFERENCE["stable_s6"],
        "policy":{k:v for k,v in policy.items() if k!="rank"},
        "holdout":{
            "pine_exact_current":stats(exact_events),
            "v2":stats(v2_events),
            "v3":stats(v3_events),
        },
        "top_v3_events":v3_events.sort_values("score100",ascending=False).head(100)[
            ["date","symbol","name","score100","p_pos","p_hit10","pred_ret","perf_5bd"]
        ].to_dict("records"),
    }
    return result


def main():
    today=core.now_jst().date()
    p=argparse.ArgumentParser()
    p.add_argument("--end-date",type=parse_date,default=today.isoformat())
    p.add_argument("--history-days",type=int,default=1825)
    p.add_argument("--holdout-days",type=int,default=365)
    p.add_argument("--max-workers",type=int,default=32)
    p.add_argument("--max-issues",type=int)
    p.add_argument("--output-dir",default="reports_no_tv_v3")
    a=p.parse_args()
    if a.history_days<=a.holdout_days:
        raise SystemExit("history-days must be > holdout-days")
    result=run(a)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"comparison_v3.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"comparison_v3.md").write_text(report_md(result),encoding="utf-8")
    print(json.dumps({"outputs":[str(out/"comparison_v3.json"),str(out/"comparison_v3.md")],
                      "holdout":result["holdout"],"policy":result["policy"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
