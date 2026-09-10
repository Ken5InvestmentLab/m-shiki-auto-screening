from __future__ import annotations

import argparse
import itertools
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import screen_big_money as core
from screen_entry import fetch_jpx_issues_fixed


MODE_CONFIG = {
    "stable_s6": {
        "label": "Stable ★6",
        "eval_days": 5,
        "target": 0.10,
        "size": 6,
        "current": ["ema25", "macdpos", "stoch75", "bb80", "pre_down3", "gap_up"],
        "pool": ["ema25", "ema75", "macdpos", "vol12", "vol20", "sbull", "atr5",
                 "hb20", "stoch75", "rsi5070", "bb80", "pre_down3", "gap_up", "ich_chikou"],
        "required_any": [],
        "min_train": 20,
        "min_valid": 8,
        "objective": "stable",
    },
    "sniper": {
        "label": "Sniper 勝率重視",
        "eval_days": 5,
        "target": 0.0,
        "size": 6,
        "current": ["vol12", "atr3", "hb20", "rsi5070", "ich_chikou", "rci9_os"],
        "pool": ["ema25", "ema75", "vol12", "vol20", "sbull", "atr3", "atr5",
                 "hb20", "stoch75", "rsi5070", "rsi4060", "bb80", "ich_chikou", "rci9_os"],
        "required_any": [],
        "min_train": 15,
        "min_valid": 6,
        "objective": "sniper",
    },
    "mega5_rebound": {
        "label": "Mega5 短期リバウンド",
        "eval_days": 5,
        "target": 0.20,
        "size": 3,
        "current": ["vol20", "rci9_os", "pre_down3"],
        "pool": ["vol12", "vol20", "sbull", "body2", "atr5", "rsi4060", "bb_lower",
                 "pre_down3", "pre_decline15", "lower_wick50", "rci9_os", "cci_os"],
        "required_any": ["rci9_os", "bb_lower", "pre_down3", "pre_decline15"],
        "min_train": 8,
        "min_valid": 3,
        "objective": "mega",
    },
    "mega40_deep_reversal": {
        "label": "Mega40 深押し反転",
        "eval_days": 40,
        "target": 0.30,
        "size": 4,
        "current": ["pre_decline15", "pre_down3", "bb_lower", "body2"],
        "pool": ["vol12", "vol20", "sbull", "body1", "body2", "atr5", "rsi4060",
                 "bb_lower", "pre_down3", "pre_decline15", "lower_wick50", "rci9_os", "cci_os"],
        "required_any": ["pre_decline15", "bb_lower"],
        "min_train": 8,
        "min_valid": 3,
        "objective": "mega",
    },
    "mega40_wick_recovery": {
        "label": "Mega40 下ヒゲ回復",
        "eval_days": 40,
        "target": 0.50,
        "size": 4,
        "current": ["pre_decline15", "cci_os", "lower_wick50", "ich_chikou"],
        "pool": ["vol12", "sbull", "body1", "atr5", "rsi4060", "bb_lower",
                 "pre_down3", "pre_decline15", "lower_wick50", "ich_chikou", "rci9_os", "cci_os"],
        "required_any": ["lower_wick50"],
        "min_train": 5,
        "min_valid": 2,
        "objective": "mega",
    },
}

# 現行HTML（2026-09-09 19:41:50 JST）の過去1年確定値。
CURRENT_REFERENCE = {
    "generated_at": "2026-09-09 19:41:50 JST",
    "stable_s6": {"n": 55, "avg": .066, "wr": .564, "hits": 10},
    "sniper": {"n": 40, "avg": .024, "wr": .658, "hits": 27},
    "mega5_rebound": {"n": 10, "avg": .145, "wr": .500, "hits": 4},
    "mega40_deep_reversal": {"n": 26, "avg": .176, "wr": .538, "hits": 5},
    "mega40_wick_recovery": {"n": 8, "avg": .106, "wr": .625, "hits": 2},
}

ALL_CONDITIONS = sorted(set(
    itertools.chain.from_iterable(cfg["pool"] + cfg["current"] for cfg in MODE_CONFIG.values())
))
COND_BIT = {name: i for i, name in enumerate(ALL_CONDITIONS)}


def parse_date(v):
    try:
        return date.fromisoformat(v).isoformat()
    except ValueError as e:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from e


def rci9(values):
    a = np.asarray(values, dtype=float)
    if len(a) != 9 or not np.all(np.isfinite(a)):
        return np.nan
    order = np.argsort(-a, kind="mergesort")
    rank = np.empty(9, dtype=float)
    rank[order] = np.arange(1, 10)
    t = np.arange(1, 10, dtype=float)
    return float((1 - 6 * np.sum((t-rank)**2) / (9*(9*9-1))) * 100)


def build_frame(dates, arrays, start_date, end_date):
    df = pd.DataFrame({
        "date": dates.astype(str),
        "open": arrays["open"], "high": arrays["high"], "low": arrays["low"],
        "close": arrays["close"], "volume": arrays["volume"],
    })
    c, h, l, o, v = df.close, df.high, df.low, df.open, df.volume

    df["base"] = (c.shift(1) <= 1000) & (v.shift(1) >= 10000) & (v >= 5000)

    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema25 = c.ewm(span=25, adjust=False, min_periods=25).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    ema75 = c.ewm(span=75, adjust=False, min_periods=75).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()

    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean() / c * 100
    vs = v / v.shift(1).rolling(20, min_periods=20).mean()
    body = (c-o) / c * 100

    d = c.diff()
    ag = d.clip(lower=0).rolling(14, min_periods=14).mean()
    al = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
    rsi = (100 - 100/(1 + ag/al.replace(0, np.nan))).fillna(100)

    lo14 = l.rolling(14, min_periods=14).min()
    hi14 = h.rolling(14, min_periods=14).max()
    stoch = ((c-lo14)/(hi14-lo14).replace(0, np.nan)*100).fillna(50)

    bm = c.rolling(20, min_periods=20).mean()
    bs = c.rolling(20, min_periods=20).std(ddof=0)
    bb = ((c-(bm-2*bs))/(4*bs.replace(0, np.nan))).clip(0,1).fillna(.5)

    hi20 = h.shift(1).rolling(20, min_periods=20).max()
    low_wick = np.minimum(c,o)-l
    rci = c.rolling(9, min_periods=9).apply(rci9, raw=True)

    tp = (h+l+c)/3
    tm = tp.rolling(14, min_periods=14).mean()
    md = tp.rolling(14, min_periods=14).apply(lambda x: float(np.mean(np.abs(x-np.mean(x)))), raw=True)
    cci = (tp-tm)/(0.015*md.replace(0,np.nan))

    cond = {
        "ema25": c > ema25,
        "ema75": c > ema75,
        "macdpos": (macd-macd_sig) > 0,
        "vol12": vs >= 1.2,
        "vol20": vs >= 2.0,
        "sbull": body >= .5,
        "body1": body >= 1.0,
        "body2": body >= 2.0,
        "atr3": atr < 3.0,
        "atr5": atr < 5.0,
        "hb20": c > hi20,
        "stoch75": stoch >= 75,
        "rsi5070": (rsi >= 50) & (rsi < 70),
        "rsi4060": (rsi >= 40) & (rsi < 60),
        "bb80": bb >= .8,
        "bb_lower": bb <= .2,
        "pre_down3": (c.shift(1)<c.shift(2)) & (c.shift(2)<c.shift(3)),
        "gap_up": o > c.shift(1),
        "pre_decline15": (c/hi20-1) <= -.15,
        "lower_wick50": (low_wick >= (c-o).abs()) & (low_wick > 0),
        "rci9_os": rci <= -50,
        "cci_os": cci <= -100,
        "ich_chikou": c > c.shift(26),
    }

    bits = np.zeros(len(df), dtype=np.uint64)
    for name, series in cond.items():
        if name in COND_BIT:
            bits |= series.fillna(False).to_numpy(bool).astype(np.uint64) << np.uint64(COND_BIT[name])
    df["bits"] = bits
    df["prev_bits"] = pd.Series(bits).shift(1, fill_value=0).astype("uint64")
    df["bar_index"] = np.arange(len(df))
    for n in (5,10,20,40):
        df[f"perf_{n}bd"] = c.shift(-n)/c-1

    use = (df.date >= start_date) & (df.date <= end_date) & df.base
    return df.loc[use, ["date","close","volume","bits","prev_bits","bar_index",
                        "perf_5bd","perf_10bd","perf_20bd","perf_40bd"]].copy()


def fetch_one(issue, start_date, end_date, yahoo_range):
    cfg = core.ScreeningConfig(yahoo_range=yahoo_range, yahoo_interval="1d")
    chart, err = core.fetch_chart(issue, cfg)
    if err:
        return [], err
    parsed = core.chart_to_arrays(chart or {}, None)
    if parsed is None:
        return [], "too_few_rows"
    dates, arrays = parsed
    f = build_frame(dates, arrays, start_date, end_date)
    rows = f.to_dict("records")
    for r in rows:
        r.update(symbol=issue.code, name=issue.name, market=issue.market)
        r["bits"], r["prev_bits"] = int(r["bits"]), int(r["prev_bits"])
    return rows, None


def mask_for(conditions):
    m = 0
    for c in conditions:
        m |= 1 << COND_BIT[c]
    return np.uint64(m)


def rising(df, conditions):
    m = mask_for(conditions)
    b = df.bits.to_numpy(np.uint64)
    p = df.prev_bits.to_numpy(np.uint64)
    return ((b&m)==m) & ((p&m)!=m)


def cooldown(events, bars=5):
    if events.empty:
        return events
    keep = []
    for _, g in events.sort_values(["symbol","bar_index"]).groupby("symbol", sort=False):
        last = -10**9
        for idx, row in g.iterrows():
            bi = int(row.bar_index)
            if bi-last >= bars:
                keep.append(idx)
                last = bi
    return events.loc[keep].sort_values(["date","symbol"]).reset_index(drop=True)


def events_for(df, conditions):
    return cooldown(df.loc[rising(df, conditions)].copy(), 5)


def stats(events, days, target):
    col = f"perf_{days}bd"
    if events.empty:
        return {"n":0,"decisive_n":0,"avg":0.0,"median":0.0,"wr":0.0,"target_hits":0,"target_rate":0.0}
    v = pd.to_numeric(events[col], errors="coerce").dropna()
    if v.empty:
        return {"n":0,"decisive_n":0,"avg":0.0,"median":0.0,"wr":0.0,"target_hits":0,"target_rate":0.0}
    dec = v[v != 0]
    wr = float((dec>0).mean()) if len(dec) else 0.0
    hit = int((v >= target).sum()) if target > 0 else int((v>0).sum())
    return {"n":int(len(v)),"decisive_n":int(len(dec)),"avg":float(v.mean()),
            "median":float(v.median()),"wr":wr,"target_hits":hit,"target_rate":float(hit/len(v))}


def split_602020(df):
    ds = sorted(df.date.unique())
    if len(ds) < 5:
        return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy()
    i = max(1, int(len(ds)*.6))
    j = max(i+1, int(len(ds)*.8))
    j = min(j, len(ds)-1)
    return df[df.date<ds[i]].copy(), df[(df.date>=ds[i])&(df.date<ds[j])].copy(), df[df.date>=ds[j]].copy()


def rank_tuple(a, b, objective):
    if objective == "sniper":
        return (b["wr"], min(a["wr"],b["wr"]), b["avg"], math.log1p(b["n"]))
    if objective == "mega":
        return (b["target_rate"], b["avg"], min(a["target_rate"],b["target_rate"]), b["wr"], math.log1p(b["n"]))
    return (min(a["wr"],b["wr"]), b["wr"], b["avg"], min(a["avg"],b["avg"]), math.log1p(b["n"]))


def optimize(df, mode_id):
    cfg = MODE_CONFIG[mode_id]
    col = f"perf_{cfg['eval_days']}bd"
    ev = df.dropna(subset=[col]).copy()
    train, valid, lockbox = split_602020(ev)
    best = None
    req = set(cfg["required_any"])
    for combo in itertools.combinations(cfg["pool"], cfg["size"]):
        if req and not (set(combo)&req):
            continue
        sa = stats(train.loc[rising(train,combo)], cfg["eval_days"], cfg["target"])
        sb = stats(valid.loc[rising(valid,combo)], cfg["eval_days"], cfg["target"])
        if sa["n"] < cfg["min_train"] or sb["n"] < cfg["min_valid"]:
            continue
        rank = rank_tuple(sa,sb,cfg["objective"])
        if best is None or rank > best["rank"]:
            best = {"conditions":list(combo),"train":sa,"valid":sb,"rank":rank}
    if best is None:
        best = {"conditions":list(cfg["current"]),"train":{},"valid":{},"fallback":True}
    else:
        best["fallback"] = False
    best["lockbox"] = stats(lockbox.loc[rising(lockbox,best["conditions"])], cfg["eval_days"], cfg["target"])
    return best


def compare(mode_id, s):
    r = CURRENT_REFERENCE[mode_id]
    return {"reference_generated_at":CURRENT_REFERENCE["generated_at"],"reference":r,
            "delta_n":s["n"]-r["n"],"delta_avg":s["avg"]-r["avg"],"delta_wr":s["wr"]-r["wr"],
            "delta_hits":s["target_hits"]-r["hits"]}


def fmt_pct(v):
    return f"{v*100:+.1f}%"


def report_md(result):
    out = [
        "# 独立日足スクリーナー比較",
        "",
        f"- 期間: {result['start_date']} ～ {result['end_date']}",
        f"- ユニバース: {result['universe_count']}銘柄",
        f"- 基本条件通過row: {result['base_row_count']}",
        f"- 現行基準: {CURRENT_REFERENCE['generated_at']}生成HTML",
        "",
        "|モード|系統|条件|n|平均|勝率|Hit|平均差|勝率差|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode_id, m in result["modes"].items():
        for k in ("current_rules_on_independent_universe","optimized_independent"):
            x = m[k]; s=x["stats"]; c=x["comparison"]
            out.append(f"|{m['label']}|{k}|{' + '.join(x['conditions'])}|{s['n']}|{fmt_pct(s['avg'])}|"
                       f"{s['wr']*100:.1f}%|{s['target_hits']}|{fmt_pct(c['delta_avg'])}|{fmt_pct(c['delta_wr'])}|")
    out += [
        "",
        "## ルール",
        "- Stable ★6を最重要KPIとして扱う。",
        "- Yahoo Finance 1dのみ。TradingView 4H / 天底極致は使わない。",
        "- 基本条件は前日終値<=1000円、前日出来高>=10000株、当日出来高>=5000株。",
        "- 条件が未成立→成立になった日だけをシグナル化し、同一銘柄・同一モードは5営業日クールダウン。",
        "- optimizedは60/20/20(train/validation/lockbox)。lockboxは条件選択に使わない。",
        "- このブランチから本番スプレッドシート・本番Discordへは書き込まない。",
    ]
    return "\n".join(out)


def run(args):
    list_date, issues = fetch_jpx_issues_fixed()
    if args.max_issues:
        issues = issues[:args.max_issues]
    rows, errors, ok = [], {}, 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(fetch_one,i,args.start_date,args.end_date,args.yahoo_range):i for i in issues}
        for n, f in enumerate(as_completed(futs),1):
            try:
                part, err = f.result()
            except Exception as e:
                part, err = [], type(e).__name__
            if err:
                errors[err] = errors.get(err,0)+1
            else:
                ok += 1
                rows.extend(part)
            if n % 500 == 0:
                print(f"progress {n}/{len(issues)} rows={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("基本条件通過データが0件")
    df = df.sort_values(["date","symbol","bar_index"]).reset_index(drop=True)

    modes = {}
    for mode_id, cfg in MODE_CONFIG.items():
        cur = cfg["current"]
        s_cur = stats(events_for(df,cur),cfg["eval_days"],cfg["target"])
        opt = optimize(df,mode_id)
        s_opt = stats(events_for(df,opt["conditions"]),cfg["eval_days"],cfg["target"])
        modes[mode_id] = {
            "label":cfg["label"],"eval_days":cfg["eval_days"],"target":cfg["target"],
            "current_rules_on_independent_universe":{"conditions":cur,"stats":s_cur,"comparison":compare(mode_id,s_cur)},
            "optimized_independent":{"conditions":opt["conditions"],"search_train":opt.get("train",{}),
                "search_valid":opt.get("valid",{}),"search_lockbox":opt.get("lockbox",{}),
                "fallback":opt.get("fallback",False),"stats":s_opt,"comparison":compare(mode_id,s_opt)},
        }
    return {"generated_at_jst":core.now_jst().isoformat(timespec="seconds"),"start_date":args.start_date,
            "end_date":args.end_date,"jpx_list_date":list_date,"universe_count":len(issues),
            "yahoo_ok":ok,"yahoo_errors":sum(errors.values()),"error_counts":errors,
            "base_row_count":len(df),"modes":modes}


def main():
    today = core.now_jst().date()
    p = argparse.ArgumentParser()
    p.add_argument("--start-date",type=parse_date,default=(today-timedelta(days=365)).isoformat())
    p.add_argument("--end-date",type=parse_date,default=today.isoformat())
    p.add_argument("--max-workers",type=int,default=32)
    p.add_argument("--yahoo-range",default="2y")
    p.add_argument("--max-issues",type=int)
    p.add_argument("--output-dir",default="reports_independent_lab")
    a = p.parse_args()
    if a.start_date > a.end_date:
        raise SystemExit("start-date must be <= end-date")
    result = run(a)
    out = Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"comparison.md").write_text(report_md(result),encoding="utf-8")
    print(json.dumps({"outputs":[str(out/"comparison.json"),str(out/"comparison.md")],
                      "stable":result["modes"]["stable_s6"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
