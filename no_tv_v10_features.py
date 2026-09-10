from __future__ import annotations

import math
import numpy as np
import pandas as pd

import screen_big_money as core
import no_tv_1h_lab as h1

FEATURES = [
    "session13", "log_price", "session_ret", "session_range_pct",
    "session_body_pct", "session_close_loc", "session_vol_ratio20",
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "gap_pct", "day_range_pct", "day_body_pct", "day_close_loc",
    "drawdown_high20", "recovery_low20", "rsi14", "atr14_pct",
    "bb_pct", "bb_width_pct", "ema25_gap_pct", "ema75_gap_pct",
    "macd_hist_pct", "stoch14", "day_vol_ratio5", "day_vol_ratio20",
    "pre_down3", "gap_up",
]


def ema_seed(values, period):
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < period:
        return out
    e = float(np.mean(values[:period]))
    out[period - 1] = e
    k = 2.0 / (period + 1.0)
    for i in range(period, len(values)):
        e = values[i] * k + e * (1.0 - k)
        out[i] = e
    return out


def wilder_rsi(values, period=14):
    values = np.asarray(values, dtype=float)
    if len(values) <= period:
        return np.nan
    d = np.diff(values)
    gains = np.maximum(d, 0.0)
    losses = np.maximum(-d, 0.0)
    ag = float(np.mean(gains[:period]))
    al = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)


def synthetic_sessions(hourly):
    x = hourly.copy()
    mins = x.ts.dt.hour * 60 + x.ts.dt.minute
    x["session"] = np.where(mins < 13 * 60, 9, 13)
    return x.groupby(["date", "session"], sort=True).agg(
        ts=("ts", "last"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum")
    ).reset_index()


def daily_completed(hourly):
    return hourly.groupby("date", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")
    ).reset_index()


def build_asof_daily(completed, sessions, upto_idx):
    cur = sessions.iloc[upto_idx]
    past = completed[completed.date < cur.date][["date", "open", "high", "low", "close", "volume"]].copy()
    today = sessions.iloc[:upto_idx + 1]
    today = today[today.date == cur.date]
    current = pd.DataFrame([{
        "date": cur.date,
        "open": float(today.iloc[0].open),
        "high": float(today.high.max()),
        "low": float(today.low.min()),
        "close": float(today.iloc[-1].close),
        "volume": float(today.volume.sum()),
    }])
    return pd.concat([past, current], ignore_index=True)


def technical_features(asof):
    if len(asof) < 80:
        return None
    o = asof.open.to_numpy(float); h = asof.high.to_numpy(float)
    l = asof.low.to_numpy(float); c = asof.close.to_numpy(float)
    v = asof.volume.to_numpy(float)
    last = len(c) - 1; close = c[-1]
    if not np.isfinite(close) or close <= 0:
        return None

    e12 = ema_seed(c, 12); e25 = ema_seed(c, 25)
    e26 = ema_seed(c, 26); e75 = ema_seed(c, 75)
    valid = np.where(np.isfinite(e12) & np.isfinite(e26))[0]
    macd_hist = np.nan; macd_pos = False
    if len(valid) >= 9:
        ml = (e12[valid] - e26[valid]).astype(float)
        sig = ema_seed(ml, 9)
        if np.isfinite(sig[-1]):
            macd_hist = float(ml[-1] - sig[-1]); macd_pos = macd_hist > 0

    tr = []
    for i in range(max(1, last - 13), last + 1):
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    atr14_pct = float(np.mean(tr)) / close * 100.0 if tr else np.nan

    bb = c[max(0, last - 19):last + 1]
    basis = float(np.mean(bb)); dev = float(np.std(bb, ddof=0))
    if dev > 0:
        bb_pct = float(np.clip((close - (basis - 2 * dev)) / (4 * dev), 0.0, 1.0))
        bb_width_pct = float(4 * dev / basis * 100.0) if basis else np.nan
    else:
        bb_pct, bb_width_pct = 0.5, 0.0

    lo14 = float(np.min(l[max(0, last - 13):last + 1])); hi14 = float(np.max(h[max(0, last - 13):last + 1]))
    stoch14 = (close - lo14) / (hi14 - lo14) * 100.0 if hi14 > lo14 else 50.0
    prev_close = c[-2]
    gap_pct = (o[-1] / prev_close - 1.0) * 100.0 if prev_close else np.nan
    day_range_pct = (h[-1] - l[-1]) / prev_close * 100.0 if prev_close else np.nan
    day_body_pct = (c[-1] - o[-1]) / o[-1] * 100.0 if o[-1] else np.nan
    day_close_loc = (c[-1] - l[-1]) / (h[-1] - l[-1]) if h[-1] > l[-1] else 0.5

    def ret(n):
        return float(c[-1] / c[-1 - n] - 1.0) if len(c) > n and c[-1 - n] else np.nan

    hi20 = float(np.max(h[max(0, last - 20):last])); lo20 = float(np.min(l[max(0, last - 20):last]))
    prior_v = v[:-1]
    vr5 = v[-1] / float(np.mean(prior_v[-5:])) if len(prior_v) >= 5 and np.mean(prior_v[-5:]) > 0 else np.nan
    vr20 = v[-1] / float(np.mean(prior_v[-20:])) if len(prior_v) >= 20 and np.mean(prior_v[-20:]) > 0 else np.nan
    pre_down3 = bool(last >= 3 and c[last - 1] < c[last - 2] and c[last - 2] < c[last - 3])
    gap_up = bool(last > 0 and o[-1] > c[-2])
    bits = {
        "ema25": bool(np.isfinite(e25[-1]) and close > e25[-1]),
        "macdpos": macd_pos,
        "stoch75": bool(stoch14 >= 75.0),
        "bb80": bool(bb_pct >= 0.80),
        "pre_down3": pre_down3,
        "gap_up": gap_up,
    }
    return {
        "ret_1d": ret(1), "ret_2d": ret(2), "ret_3d": ret(3), "ret_5d": ret(5),
        "ret_10d": ret(10), "ret_20d": ret(20), "gap_pct": gap_pct,
        "day_range_pct": day_range_pct, "day_body_pct": day_body_pct,
        "day_close_loc": day_close_loc, "drawdown_high20": close / hi20 - 1.0 if hi20 else np.nan,
        "recovery_low20": close / lo20 - 1.0 if lo20 else np.nan,
        "rsi14": wilder_rsi(c, 14), "atr14_pct": atr14_pct,
        "bb_pct": bb_pct, "bb_width_pct": bb_width_pct,
        "ema25_gap_pct": (close / e25[-1] - 1.0) * 100.0 if np.isfinite(e25[-1]) else np.nan,
        "ema75_gap_pct": (close / e75[-1] - 1.0) * 100.0 if np.isfinite(e75[-1]) else np.nan,
        "macd_hist_pct": macd_hist / close * 100.0 if np.isfinite(macd_hist) else np.nan,
        "stoch14": stoch14, "day_vol_ratio5": vr5, "day_vol_ratio20": vr20,
        "pre_down3": int(pre_down3), "gap_up": int(gap_up),
        "stable_score": sum(int(x) for x in bits.values()),
        **{f"stable_{k}": int(val) for k, val in bits.items()},
    }


def build_issue_candidates(issue, chart, start_date, end_date):
    hourly = h1.parse_1h_chart(chart)
    if hourly is None or hourly.empty:
        return None
    sessions = synthetic_sessions(hourly).reset_index(drop=True)
    completed = daily_completed(hourly)
    if len(completed) < 90 or len(sessions) < 120:
        return None
    d = completed.copy(); d["day_index"] = np.arange(len(d))
    d["future_close_5"] = d.close.shift(-5); d["exit_date_5bd"] = d.date.shift(-5)
    daymap = d.set_index("date")
    svolume = sessions.volume.to_numpy(float)
    rows = []
    for i, row in sessions.iterrows():
        dt = str(row.date)
        if dt < start_date or dt > end_date or row.date not in daymap.index:
            continue
        di = int(daymap.loc[row.date, "day_index"])
        if di < 1:
            continue
        prev = d.iloc[di - 1]
        if not (float(prev.close) <= 1000.0 and float(prev.volume) >= 10000.0 and float(row.volume) >= 5000.0):
            continue
        tf = technical_features(build_asof_daily(completed, sessions, i))
        if tf is None:
            continue
        prev20 = svolume[max(0, i - 20):i]
        svr = float(row.volume / np.mean(prev20)) if len(prev20) >= 5 and np.mean(prev20) > 0 else np.nan
        rng = float(row.high - row.low)
        rec = {
            "date": dt, "session": int(row.session), "symbol": str(issue.code), "name": issue.name,
            "entry": float(row.close), "session_volume": float(row.volume), "session13": int(row.session == 13),
            "log_price": math.log(max(float(row.close), 1e-9)),
            "session_ret": float(row.close / row.open - 1.0) if row.open else np.nan,
            "session_range_pct": float(rng / row.open) if row.open else np.nan,
            "session_body_pct": float((row.close - row.open) / row.open) if row.open else np.nan,
            "session_close_loc": float((row.close - row.low) / rng) if rng > 0 else 0.5,
            "session_vol_ratio20": svr, **tf,
        }
        fc = daymap.loc[row.date, "future_close_5"]; ex = daymap.loc[row.date, "exit_date_5bd"]
        rec["perf_5bd"] = float(fc / row.close - 1.0) if pd.notna(fc) and row.close else np.nan
        rec["exit_date_5bd"] = str(ex) if pd.notna(ex) else ""
        rows.append(rec)
    return pd.DataFrame(rows)


def fetch_one(issue, start_date, end_date):
    cfg = core.ScreeningConfig(yahoo_range="730d", yahoo_interval="1h")
    chart, err = core.fetch_chart(issue, cfg)
    if err:
        return None, err
    fr = build_issue_candidates(issue, chart or {}, start_date, end_date)
    return fr, None if fr is not None else "no_data"
