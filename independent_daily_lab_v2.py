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


# ─────────────────────────────────────────────────────────────
# V2の目的
#   TradingView / 4H / 天底極致そのものは使わず、
#   ユーザー提供Pineの「状態遷移」という考え方を日足で再構築する。
#
#   Candidate Trigger（BOTTOM相当）
#       ↓
#   Stable ★6 / Sniper / Mega3種のスコア条件
#       ↓
#   5BD / 40BD 成績
#
# V1のように「スコア条件が成立した瞬間=シグナル」にはしない。
# ─────────────────────────────────────────────────────────────

PINE_PROFILES = [
    {
        "id": "exact",
        "label": "Pine Daily Exact",
        "rsi_len": 12, "ob": 75, "os": 35,
        "bb_len": 20, "bb_mult": 2.0,
        "min_bars": 5, "use_emergency": True, "emerg_atr_mult": 2.5,
        "use_squeeze": False, "squeeze_thresh": 0.18,
    },
    {
        "id": "os30",
        "label": "OS30",
        "rsi_len": 12, "ob": 75, "os": 30,
        "bb_len": 20, "bb_mult": 2.0,
        "min_bars": 5, "use_emergency": True, "emerg_atr_mult": 2.5,
        "use_squeeze": False, "squeeze_thresh": 0.18,
    },
    {
        "id": "os40",
        "label": "OS40",
        "rsi_len": 12, "ob": 75, "os": 40,
        "bb_len": 20, "bb_mult": 2.0,
        "min_bars": 5, "use_emergency": True, "emerg_atr_mult": 2.5,
        "use_squeeze": False, "squeeze_thresh": 0.18,
    },
    {
        "id": "rsi9",
        "label": "RSI9",
        "rsi_len": 9, "ob": 75, "os": 35,
        "bb_len": 20, "bb_mult": 2.0,
        "min_bars": 5, "use_emergency": True, "emerg_atr_mult": 2.5,
        "use_squeeze": False, "squeeze_thresh": 0.18,
    },
    {
        "id": "rsi14",
        "label": "RSI14",
        "rsi_len": 14, "ob": 75, "os": 35,
        "bb_len": 20, "bb_mult": 2.0,
        "min_bars": 5, "use_emergency": True, "emerg_atr_mult": 2.5,
        "use_squeeze": False, "squeeze_thresh": 0.18,
    },
    {
        "id": "cool3",
        "label": "Cooldown3",
        "rsi_len": 12, "ob": 75, "os": 35,
        "bb_len": 20, "bb_mult": 2.0,
        "min_bars": 3, "use_emergency": True, "emerg_atr_mult": 2.5,
        "use_squeeze": False, "squeeze_thresh": 0.18,
    },
    {
        "id": "cool8",
        "label": "Cooldown8",
        "rsi_len": 12, "ob": 75, "os": 35,
        "bb_len": 20, "bb_mult": 2.0,
        "min_bars": 8, "use_emergency": True, "emerg_atr_mult": 2.5,
        "use_squeeze": False, "squeeze_thresh": 0.18,
    },
    {
        "id": "emerg2",
        "label": "Emergency2.0ATR",
        "rsi_len": 12, "ob": 75, "os": 35,
        "bb_len": 20, "bb_mult": 2.0,
        "min_bars": 5, "use_emergency": True, "emerg_atr_mult": 2.0,
        "use_squeeze": False, "squeeze_thresh": 0.18,
    },
    {
        "id": "no_emergency",
        "label": "No Emergency",
        "rsi_len": 12, "ob": 75, "os": 35,
        "bb_len": 20, "bb_mult": 2.0,
        "min_bars": 5, "use_emergency": False, "emerg_atr_mult": 2.5,
        "use_squeeze": False, "squeeze_thresh": 0.18,
    },
    {
        "id": "squeeze10",
        "label": "Squeeze>10%",
        "rsi_len": 12, "ob": 75, "os": 35,
        "bb_len": 20, "bb_mult": 2.0,
        "min_bars": 5, "use_emergency": True, "emerg_atr_mult": 2.5,
        "use_squeeze": True, "squeeze_thresh": 0.10,
    },
]
PROFILE_BIT = {p["id"]: i for i, p in enumerate(PINE_PROFILES)}


MODE_CONFIG = {
    "stable_s6": {
        "label": "Stable ★6",
        "eval_days": 5,
        "target": 0.10,
        "size": 6,
        "current": ["ema25", "macdpos", "stoch75", "bb80", "pre_down3", "gap_up"],
        # 6条件は維持。探索対象を広げすぎず、過学習を抑える。
        "pool": ["ema25", "ema75", "macdpos", "vol20", "sbull", "atr5",
                 "hb20", "stoch75", "rsi5070", "bb80", "pre_down3", "gap_up"],
        "min_train": 40,
        "min_valid": 15,
        "objective": "stable",
    },
    "sniper": {
        "label": "Sniper 勝率重視",
        "eval_days": 5,
        "target": 0.0,
        "size": 6,
        "current": ["vol12", "atr3", "hb20", "rsi5070", "ich_chikou", "rci9_os"],
        "pool": ["ema25", "ema75", "vol12", "vol20", "atr3", "atr5",
                 "hb20", "stoch75", "rsi5070", "rsi4060", "ich_chikou", "rci9_os"],
        "min_train": 30,
        "min_valid": 10,
        "objective": "sniper",
    },
    "mega5_rebound": {
        "label": "Mega5 短期リバウンド",
        "eval_days": 5,
        "target": 0.20,
        "size": 3,
        "current": ["vol20", "rci9_os", "pre_down3"],
        "pool": ["vol12", "vol20", "sbull", "body2", "atr5", "rsi4060",
                 "bb_lower", "pre_down3", "pre_decline15", "rci9_os"],
        "min_train": 12,
        "min_valid": 5,
        "objective": "mega",
    },
    "mega40_deep_reversal": {
        "label": "Mega40 深押し反転",
        "eval_days": 40,
        "target": 0.30,
        "size": 4,
        "current": ["pre_decline15", "pre_down3", "bb_lower", "body2"],
        "pool": ["vol12", "vol20", "sbull", "body1", "body2", "atr5",
                 "rsi4060", "bb_lower", "pre_down3", "pre_decline15"],
        "min_train": 10,
        "min_valid": 4,
        "objective": "mega",
    },
    "mega40_wick_recovery": {
        "label": "Mega40 下ヒゲ回復",
        "eval_days": 40,
        "target": 0.50,
        "size": 4,
        "current": ["pre_decline15", "cci_os", "lower_wick50", "ich_chikou"],
        "pool": ["vol12", "sbull", "body1", "atr5", "rsi4060",
                 "bb_lower", "pre_decline15", "lower_wick50", "ich_chikou", "cci_os"],
        "min_train": 8,
        "min_valid": 3,
        "objective": "mega",
    },
}

# 2026-09-09 19:41:50 JST の公開レポート基準。
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


def wilder_rma(values, length):
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    if len(x) < length:
        return out
    seed_i = None
    for i in range(length - 1, len(x)):
        w = x[i - length + 1:i + 1]
        if np.all(np.isfinite(w)):
            out[i] = float(np.mean(w))
            seed_i = i
            break
    if seed_i is None:
        return out
    alpha = 1.0 / length
    for i in range(seed_i + 1, len(x)):
        if np.isfinite(x[i]):
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
        else:
            out[i] = out[i - 1]
    return out


def pine_rsi(close, length):
    c = np.asarray(close, dtype=float)
    d = np.full(len(c), np.nan)
    d[1:] = c[1:] - c[:-1]
    up = np.where(np.isnan(d), np.nan, np.maximum(d, 0.0))
    dn = np.where(np.isnan(d), np.nan, np.maximum(-d, 0.0))
    au = wilder_rma(up, length)
    ad = wilder_rma(dn, length)
    out = np.full(len(c), np.nan)
    both0 = (au == 0) & (ad == 0)
    out[both0] = 50.0
    only_dn0 = (ad == 0) & (au > 0)
    out[only_dn0] = 100.0
    normal = (ad > 0) & np.isfinite(au)
    rs = np.zeros(len(c))
    rs[normal] = au[normal] / ad[normal]
    out[normal] = 100.0 - 100.0 / (1.0 + rs[normal])
    return out


def true_range(high, low, close):
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    tr = np.full(len(c), np.nan)
    if len(c):
        tr[0] = h[0] - l[0]
    if len(c) > 1:
        pc = c[:-1]
        tr[1:] = np.maximum.reduce([
            h[1:] - l[1:],
            np.abs(h[1:] - pc),
            np.abs(l[1:] - pc),
        ])
    return tr


def pine_atr(high, low, close, length=14):
    return wilder_rma(true_range(high, low, close), length)


def rolling_mean(x, n):
    return pd.Series(np.asarray(x, dtype=float)).rolling(n, min_periods=n).mean().to_numpy()


def rolling_std0(x, n):
    return pd.Series(np.asarray(x, dtype=float)).rolling(n, min_periods=n).std(ddof=0).to_numpy()


def pine_state_signals(close, high, low, profile):
    """
    ユーザー提供Pineの f_normal_signal() を日足確定値で移植。
    日足なので useAutoAdjust の低時間足補正は発動しない。
    """
    c = np.asarray(close, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    n = len(c)

    mid = rolling_mean(c, profile["bb_len"])
    sd = rolling_std0(c, profile["bb_len"])
    upper = mid + profile["bb_mult"] * sd
    lower = mid - profile["bb_mult"] * sd
    rsi = pine_rsi(c, profile["rsi_len"])
    atr = pine_atr(h, l, c, 14)

    slope = np.full(n, np.nan)
    slope[1:] = mid[1:] - mid[:-1]
    width = np.zeros(n)
    good_mid = np.isfinite(mid) & (mid != 0)
    width[good_mid] = (upper[good_mid] - lower[good_mid]) / mid[good_mid]

    bottom = np.zeros(n, dtype=bool)
    top = np.zeros(n, dtype=bool)
    reason = np.zeros(n, dtype=np.uint8)

    last_signal = 0
    extreme_price = 0.0
    last_signal_index = 0
    wait_pullback = False

    for i in range(1, n):
        ri = rsi[i]
        rp = rsi[i - 1]
        si = slope[i]
        sp = slope[i - 1]

        cross_buy = (
            np.isfinite(ri) and np.isfinite(rp)
            and ri > profile["os"] and rp <= profile["os"]
        )
        cross_sell = (
            np.isfinite(ri) and np.isfinite(rp)
            and ri < profile["ob"] and rp >= profile["ob"]
        )
        trend_flip_down = (
            np.isfinite(si) and np.isfinite(sp) and si < 0 and sp >= 0
            and np.isfinite(ri) and np.isfinite(rp) and ri < 50 and rp >= 50
        )
        trend_flip_up = (
            np.isfinite(si) and np.isfinite(sp) and si > 0 and sp <= 0
            and np.isfinite(ri) and np.isfinite(rp) and ri > 50 and rp <= 50
        )

        ai = atr[i]
        atr_reverse_dist = ai * profile["emerg_atr_mult"] if np.isfinite(ai) else np.nan
        # Pine原文は atrExitDist = atr * 2.5 固定。
        atr_exit_dist = ai * 2.5 if np.isfinite(ai) else np.nan

        is_cooldown_ok = (i - last_signal_index) > profile["min_bars"]
        trig_exit_long = (
            last_signal == 1 and np.isfinite(atr_exit_dist)
            and c[i] < (extreme_price - atr_exit_dist)
        )
        trig_exit_short = (
            last_signal == -1 and np.isfinite(atr_exit_dist)
            and c[i] > (extreme_price + atr_exit_dist)
        )
        force_reverse_buy = (
            profile["use_emergency"] and last_signal == -1
            and np.isfinite(atr_reverse_dist)
            and c[i] > extreme_price + atr_reverse_dist
        )
        force_reverse_sell = (
            profile["use_emergency"] and last_signal == 1
            and np.isfinite(atr_reverse_dist)
            and c[i] < extreme_price - atr_reverse_dist
        )

        wp_check = wait_pullback
        if last_signal == -1 and np.isfinite(si) and si > 0 and not force_reverse_buy:
            wp_check = True
        if last_signal == 1 and np.isfinite(si) and si < 0 and not force_reverse_sell:
            wp_check = True
        if trend_flip_up or trend_flip_down:
            wp_check = False

        pass_squeeze = (
            (not profile["use_squeeze"])
            or (good_mid[i] and width[i] > profile["squeeze_thresh"])
        )
        common_filter = (not wp_check) and pass_squeeze and is_cooldown_ok

        raw_bottom = (
            (common_filter and (cross_buy or trig_exit_short or trend_flip_up) and last_signal != 1)
            or force_reverse_buy
        )
        raw_top = (
            (common_filter and (cross_sell or trig_exit_long or trend_flip_down) and last_signal != -1)
            or force_reverse_sell
        )

        # Historical daily bars are all confirmed bars.
        show_bottom = bool(raw_bottom)
        show_top = bool(raw_top)
        bottom[i] = show_bottom
        top[i] = show_top

        r = 0
        if show_bottom:
            if cross_buy:
                r |= 1
            if trig_exit_short:
                r |= 2
            if trend_flip_up:
                r |= 4
            if force_reverse_buy:
                r |= 8
            reason[i] = r

        wait_pullback = wp_check
        # Pineの if showBottom → if showTop の順を維持。
        if show_bottom:
            last_signal = 1
            last_signal_index = i
            extreme_price = c[i]
        if show_top:
            last_signal = -1
            last_signal_index = i
            extreme_price = c[i]

    return bottom, top, reason


def rci9(values):
    a = np.asarray(values, dtype=float)
    if len(a) != 9 or not np.all(np.isfinite(a)):
        return np.nan
    # 直近ほど時間順位1、価格が高いほど価格順位1という一般的RCI定義。
    price_order = np.argsort(-a, kind="mergesort")
    price_rank = np.empty(9, dtype=float)
    price_rank[price_order] = np.arange(1, 10)
    time_rank = np.arange(9, 0, -1, dtype=float)
    d = time_rank - price_rank
    return float((1.0 - 6.0 * np.sum(d * d) / (9 * (9 * 9 - 1))) * 100.0)


def build_frame(dates, arrays, history_start, end_date):
    df = pd.DataFrame({
        "date": dates.astype(str),
        "open": arrays["open"], "high": arrays["high"], "low": arrays["low"],
        "close": arrays["close"], "volume": arrays["volume"],
    })
    c = df.close.astype(float)
    h = df.high.astype(float)
    l = df.low.astype(float)
    o = df.open.astype(float)
    v = df.volume.astype(float)

    # ユーザー指定の基本3条件。
    df["base"] = (c.shift(1) <= 1000) & (v.shift(1) >= 10000) & (v >= 5000)

    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema25 = c.ewm(span=25, adjust=False, min_periods=25).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    ema75 = c.ewm(span=75, adjust=False, min_periods=75).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()

    atr_abs = pine_atr(h.to_numpy(), l.to_numpy(), c.to_numpy(), 14)
    atr_pct = pd.Series(atr_abs, index=df.index) / c * 100.0

    prev_v20 = v.shift(1).rolling(20, min_periods=20).mean()
    vs = v / prev_v20
    body = (c - o) / c * 100.0

    rsi14 = pd.Series(pine_rsi(c.to_numpy(), 14), index=df.index)

    lo14 = l.rolling(14, min_periods=14).min()
    hi14 = h.rolling(14, min_periods=14).max()
    stoch = ((c - lo14) / (hi14 - lo14).replace(0, np.nan) * 100.0).fillna(50.0)

    bm = c.rolling(20, min_periods=20).mean()
    bs = c.rolling(20, min_periods=20).std(ddof=0)
    bb = ((c - (bm - 2 * bs)) / (4 * bs.replace(0, np.nan))).fillna(0.5)

    hi20 = h.shift(1).rolling(20, min_periods=20).max()
    low_wick = np.minimum(c, o) - l

    rci = c.rolling(9, min_periods=9).apply(rci9, raw=True)

    tp = (h + l + c) / 3.0
    tm = tp.rolling(14, min_periods=14).mean()
    md = tp.rolling(14, min_periods=14).apply(
        lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True
    )
    cci = (tp - tm) / (0.015 * md.replace(0, np.nan))

    cond = {
        "ema25": c > ema25,
        "ema75": c > ema75,
        "macdpos": (macd - macd_sig) > 0,
        "vol12": vs >= 1.2,
        "vol20": vs >= 2.0,
        "sbull": body >= 0.5,
        "body1": body >= 1.0,
        "body2": body >= 2.0,
        "atr3": atr_pct < 3.0,
        "atr5": atr_pct < 5.0,
        "hb20": c > hi20,
        "stoch75": stoch >= 75,
        "rsi5070": (rsi14 >= 50) & (rsi14 < 70),
        "rsi4060": (rsi14 >= 40) & (rsi14 < 60),
        "bb80": bb >= 0.80,
        "bb_lower": bb <= 0.20,
        "pre_down3": (c.shift(1) < c.shift(2)) & (c.shift(2) < c.shift(3)),
        "gap_up": o > c.shift(1),
        "pre_decline15": (c / hi20 - 1.0) <= -0.15,
        "lower_wick50": (low_wick >= (c - o).abs()) & (low_wick > 0),
        "rci9_os": rci <= -50,
        "cci_os": cci <= -100,
        "ich_chikou": c > c.shift(26),
    }

    score_bits = np.zeros(len(df), dtype=np.uint64)
    for name, series in cond.items():
        if name in COND_BIT:
            score_bits |= (
                series.fillna(False).to_numpy(bool).astype(np.uint64)
                << np.uint64(COND_BIT[name])
            )
    df["score_bits"] = score_bits

    trigger_bits = np.zeros(len(df), dtype=np.uint16)
    exact_reason = np.zeros(len(df), dtype=np.uint8)
    close_np = c.to_numpy()
    high_np = h.to_numpy()
    low_np = l.to_numpy()
    for p in PINE_PROFILES:
        bottom, _, reason = pine_state_signals(close_np, high_np, low_np, p)
        trigger_bits |= bottom.astype(np.uint16) << np.uint16(PROFILE_BIT[p["id"]])
        if p["id"] == "exact":
            exact_reason = reason
    df["trigger_bits"] = trigger_bits
    df["exact_reason"] = exact_reason
    df["bar_index"] = np.arange(len(df))

    dstr = df["date"]
    for n in (5, 10, 20, 40):
        df[f"perf_{n}bd"] = c.shift(-n) / c - 1.0
        df[f"exit_date_{n}bd"] = dstr.shift(-n)

    use = (df.date >= history_start) & (df.date <= end_date) & df.base
    cols = [
        "date", "close", "volume", "score_bits", "trigger_bits", "exact_reason", "bar_index",
        "perf_5bd", "perf_10bd", "perf_20bd", "perf_40bd",
        "exit_date_5bd", "exit_date_10bd", "exit_date_20bd", "exit_date_40bd",
    ]
    return df.loc[use, cols].copy()


def fetch_one(issue, history_start, end_date, yahoo_range):
    cfg = core.ScreeningConfig(yahoo_range=yahoo_range, yahoo_interval="1d")
    chart, err = core.fetch_chart(issue, cfg)
    if err:
        return [], err
    parsed = core.chart_to_arrays(chart or {}, None)
    if parsed is None:
        return [], "too_few_rows"
    dates, arrays = parsed
    f = build_frame(dates, arrays, history_start, end_date)
    rows = f.to_dict("records")
    for r in rows:
        r.update(symbol=issue.code, name=issue.name, market=issue.market)
        r["score_bits"] = int(r["score_bits"])
        r["trigger_bits"] = int(r["trigger_bits"])
        r["exact_reason"] = int(r["exact_reason"])
    return rows, None


def condition_mask(conditions):
    m = 0
    for c in conditions:
        m |= 1 << COND_BIT[c]
    return np.uint64(m)


def profile_mask(profile_id):
    return np.uint16(1 << PROFILE_BIT[profile_id])


def trigger_period(df, profile_id, start_date, end_date, eval_days):
    """Trigger/date/confirmedだけを絞る。Score条件は後段で適用する。"""
    if df.empty:
        return df
    tb = df.trigger_bits.to_numpy(np.uint16)
    mask = (tb & profile_mask(profile_id)) != 0
    dates = df.date.to_numpy(str)
    mask &= (dates >= start_date) & (dates <= end_date)

    exit_col = f"exit_date_{eval_days}bd"
    exits = df[exit_col].fillna("").to_numpy(str)
    mask &= (exits != "") & (exits <= end_date)
    return df.loc[mask].copy()


def select_conditions(frame, conditions):
    if frame.empty or not conditions:
        return frame
    cm = condition_mask(conditions)
    sb = frame.score_bits.to_numpy(np.uint64)
    return frame.loc[(sb & cm) == cm]


def build_trigger_cache(df, split, holdout_start, end_date):
    """
    最適化中に全base rowを何千回も走査しないよう、
    profile × 評価日数 × period のTrigger候補を一度だけキャッシュする。
    """
    cache = {}
    for p in PINE_PROFILES:
        pid = p["id"]
        cache[pid] = {}
        for days in (5, 40):
            cache[pid][days] = {
                "train": trigger_period(
                    df, pid, split["train_start"], split["train_end"], days
                ),
                "valid": trigger_period(
                    df, pid, split["valid_start"], split["valid_end"], days
                ),
                "holdout": trigger_period(
                    df, pid, holdout_start, end_date, days
                ),
            }
    return cache


def stats(events, days, target):
    col = f"perf_{days}bd"
    if events.empty:
        return {
            "n": 0, "decisive_n": 0, "avg": 0.0, "robust_avg": 0.0,
            "median": 0.0, "wr": 0.0, "target_hits": 0, "target_rate": 0.0,
        }
    v = pd.to_numeric(events[col], errors="coerce").dropna().to_numpy(float)
    if len(v) == 0:
        return {
            "n": 0, "decisive_n": 0, "avg": 0.0, "robust_avg": 0.0,
            "median": 0.0, "wr": 0.0, "target_hits": 0, "target_rate": 0.0,
        }
    dec = v[v != 0]
    wr = float(np.mean(dec > 0)) if len(dec) else 0.0
    hits = int(np.sum(v >= target)) if target > 0 else int(np.sum(v > 0))
    if len(v) >= 10:
        lo, hi = np.quantile(v, [0.05, 0.95])
        robust = float(np.mean(np.clip(v, lo, hi)))
    else:
        robust = float(np.mean(v))
    return {
        "n": int(len(v)),
        "decisive_n": int(len(dec)),
        "avg": float(np.mean(v)),
        "robust_avg": robust,
        "median": float(np.median(v)),
        "wr": wr,
        "target_hits": hits,
        "target_rate": float(hits / len(v)),
    }


def split_dates(df, holdout_start):
    history_dates = sorted(df.loc[df.date < holdout_start, "date"].unique())
    if len(history_dates) < 20:
        raise RuntimeError("pre-holdout history is too short")
    idx = max(1, int(len(history_dates) * 0.75))
    idx = min(idx, len(history_dates) - 1)
    valid_start = history_dates[idx]
    train_end = history_dates[idx - 1]
    history_end = history_dates[-1]
    return {
        "train_start": history_dates[0],
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": history_end,
    }


def quality_rank(train, valid, cfg):
    if train["n"] < cfg["min_train"] or valid["n"] < cfg["min_valid"]:
        return None

    min_wr = min(train["wr"], valid["wr"])
    min_avg = min(train["robust_avg"], valid["robust_avg"])
    min_target = min(train["target_rate"], valid["target_rate"])

    if cfg["objective"] == "sniper":
        if valid["robust_avg"] < -0.02:
            return None
        return (
            min_wr,
            valid["wr"],
            min_avg,
            valid["robust_avg"],
            math.log1p(valid["n"]),
        )

    if cfg["objective"] == "mega":
        if train["target_hits"] < 1 or valid["target_hits"] < 1:
            return None
        return (
            min_target,
            min_avg,
            min_wr,
            valid["target_rate"],
            valid["robust_avg"],
            math.log1p(valid["n"]),
        )

    # Stable ★6: 平均・勝率・+10%Hitを同時に要求。
    if min_wr < 0.48 or min_avg <= 0:
        return None
    score = (
        0.45 * min_wr
        + 2.50 * min_avg
        + 0.45 * min_target
        + 0.015 * math.log1p(valid["n"])
    )
    return (
        score,
        min_avg,
        min_wr,
        min_target,
        valid["robust_avg"],
        valid["wr"],
        math.log1p(valid["n"]),
    )


def eval_train_valid(cache, profile_id, conditions, cfg):
    frames = cache[profile_id][cfg["eval_days"]]
    tr = select_conditions(frames["train"], conditions)
    va = select_conditions(frames["valid"], conditions)
    return (
        stats(tr, cfg["eval_days"], cfg["target"]),
        stats(va, cfg["eval_days"], cfg["target"]),
    )


def tune_profile_fixed_score(cache, cfg):
    best = None
    for p in PINE_PROFILES:
        tr, va = eval_train_valid(cache, p["id"], cfg["current"], cfg)
        rank = quality_rank(tr, va, cfg)
        if rank is None:
            continue
        item = {
            "profile": p["id"],
            "conditions": list(cfg["current"]),
            "train": tr,
            "valid": va,
            "rank": rank,
        }
        if best is None or rank > best["rank"]:
            best = item
    return best


def tune_profile_and_score(cache, cfg):
    best = None
    combos = list(itertools.combinations(cfg["pool"], cfg["size"]))
    for p in PINE_PROFILES:
        pid = p["id"]
        frames = cache[pid][cfg["eval_days"]]
        train_frame = frames["train"]
        valid_frame = frames["valid"]

        # profileごとの候補行は既に絞られているため、ここではScore bitだけを評価。
        train_bits = train_frame.score_bits.to_numpy(np.uint64)
        valid_bits = valid_frame.score_bits.to_numpy(np.uint64)

        for combo in combos:
            cm = condition_mask(combo)
            tr_sel = train_frame.loc[(train_bits & cm) == cm]
            va_sel = valid_frame.loc[(valid_bits & cm) == cm]
            tr = stats(tr_sel, cfg["eval_days"], cfg["target"])
            va = stats(va_sel, cfg["eval_days"], cfg["target"])
            rank = quality_rank(tr, va, cfg)
            if rank is None:
                continue
            item = {
                "profile": pid,
                "conditions": list(combo),
                "train": tr,
                "valid": va,
                "rank": rank,
            }
            if best is None or rank > best["rank"]:
                best = item
    return best


def compare(mode_id, s):
    r = CURRENT_REFERENCE[mode_id]
    return {
        "reference_generated_at": CURRENT_REFERENCE["generated_at"],
        "reference": r,
        "delta_n": s["n"] - r["n"],
        "delta_avg": s["avg"] - r["avg"],
        "delta_wr": s["wr"] - r["wr"],
        "delta_hits": s["target_hits"] - r["hits"],
    }


def fmt_pct(v):
    return f"{v * 100:+.1f}%"


def profile_by_id(pid):
    return next(p for p in PINE_PROFILES if p["id"] == pid)


def profile_text(pid):
    p = profile_by_id(pid)
    return (
        f"{pid}(RSI{p['rsi_len']} OS{p['os']} OB{p['ob']} "
        f"CD{p['min_bars']} Emergency={p['use_emergency']} "
        f"ATRx{p['emerg_atr_mult']} Squeeze={p['use_squeeze']})"
    )


def holdout_result(cache, profile_id, conditions, cfg):
    frame = cache[profile_id][cfg["eval_days"]]["holdout"]
    ev = select_conditions(frame, conditions)
    return stats(ev, cfg["eval_days"], cfg["target"])


def exact_reason_summary(cache):
    ev = cache["exact"][5]["holdout"]
    counts = {"crossBuy": 0, "trigExitShort": 0, "trendFlipUp": 0, "forceReverseBuy": 0}
    for r in ev.exact_reason.to_numpy(np.uint8):
        if r & 1:
            counts["crossBuy"] += 1
        if r & 2:
            counts["trigExitShort"] += 1
        if r & 4:
            counts["trendFlipUp"] += 1
        if r & 8:
            counts["forceReverseBuy"] += 1
    return {"n": int(len(ev)), "reason_counts": counts}


def report_md(result):
    out = [
        "# 独立日足スクリーナー V2 — Pine状態遷移型",
        "",
        f"- 生成: {result['generated_at_jst']}",
        f"- データ期間: {result['history_start']} ～ {result['end_date']}",
        f"- 完全未使用Holdout: {result['holdout_start']} ～ {result['end_date']}",
        f"- JPX対象: {result['universe_count']}銘柄",
        f"- Yahoo成功: {result['yahoo_ok']} / {result['universe_count']}",
        f"- 現行比較基準: {CURRENT_REFERENCE['generated_at']}",
        "",
        "## Stable ★6 を最優先したHoldout比較",
        "",
        "|モード|方式|Trigger|Score条件|n|平均|Robust平均|勝率|Hit|現行比 平均|現行比 勝率|",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode_id, m in result["modes"].items():
        for key in ("pine_exact_current_score", "tuned_trigger_current_score", "v2_optimized"):
            x = m[key]
            s = x["holdout"]
            c = x["comparison"]
            out.append(
                f"|{m['label']}|{key}|{x['profile']}|{' + '.join(x['conditions']) or 'なし'}|"
                f"{s['n']}|{fmt_pct(s['avg'])}|{fmt_pct(s['robust_avg'])}|{s['wr']*100:.1f}%|"
                f"{s['target_hits']}|{fmt_pct(c['delta_avg'])}|{fmt_pct(c['delta_wr'])}|"
            )
    out += [
        "",
        "## Exact Triggerの内訳",
        f"- Holdout確定BOTTOM件数: {result['exact_trigger']['n']}",
        f"- 原因: {json.dumps(result['exact_trigger']['reason_counts'], ensure_ascii=False)}",
        "",
        "## 選定ルール",
        "- PineのlastSignal / extremePrice / lastSignalIndex / waitPullbackを銘柄ごとに時系列再現。",
        "- 日足なのでPineの低時間足AutoAdjustは発動させない。",
        "- Exactは lengthBB=20 / RSI=12 / OB=75 / OS=35 / minBars=5 / Emergency=2.5ATR。",
        "- V2のパラメータ・Score条件選定にはHoldoutを一切使わない。",
        "- Train/Validationでは各評価期間の末尾をpurgeし、評価日が次区間へ跨ぐシグナルを除外。",
        "- 本番Discord / 本番Spreadsheet / mainブランチには書き込まない。",
    ]
    return "\n".join(out)


def run(args):
    end_d = date.fromisoformat(args.end_date)
    history_start = (end_d - timedelta(days=args.history_days)).isoformat()
    holdout_start = (end_d - timedelta(days=args.holdout_days)).isoformat()

    list_date, issues = fetch_jpx_issues_fixed()
    # 優先株を明示除外。ETF/REIT等はJPX「内国株式」フィルタで既に除外。
    issues = [i for i in issues if "優先" not in i.name]
    if args.max_issues:
        issues = issues[:args.max_issues]

    rows, errors, ok = [], {}, 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(fetch_one, i, history_start, args.end_date, args.yahoo_range): i
            for i in issues
        }
        for n, f in enumerate(as_completed(futs), 1):
            try:
                part, err = f.result()
            except Exception as e:
                part, err = [], type(e).__name__
            if err:
                errors[err] = errors.get(err, 0) + 1
            else:
                ok += 1
                rows.extend(part)
            if n % 500 == 0:
                print(f"progress {n}/{len(issues)} rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("基本条件通過データが0件")
    df = df.sort_values(["date", "symbol", "bar_index"]).reset_index(drop=True)

    split = split_dates(df, holdout_start)
    print("building trigger cache ...", flush=True)
    cache = build_trigger_cache(df, split, holdout_start, args.end_date)
    modes = {}

    for mode_id, cfg in MODE_CONFIG.items():
        print(f"tuning {mode_id} ...", flush=True)

        # 1) Pine原文を日足移植 + 現行Score
        exact_hold = holdout_result(cache, "exact", cfg["current"], cfg)

        # 2) Triggerパラメータだけ過去期間で調整、Score条件は現行固定
        tuned_fixed = tune_profile_fixed_score(cache, cfg)
        if tuned_fixed is None:
            tuned_fixed = {
                "profile": "exact", "conditions": list(cfg["current"]),
                "train": {}, "valid": {}, "fallback": True,
            }
        fixed_hold = holdout_result(
            cache, tuned_fixed["profile"], tuned_fixed["conditions"], cfg
        )

        # 3) Trigger + Score条件を過去期間だけで同時最適化
        optimized = tune_profile_and_score(cache, cfg)
        if optimized is None:
            optimized = {
                "profile": tuned_fixed["profile"],
                "conditions": list(cfg["current"]),
                "train": tuned_fixed.get("train", {}),
                "valid": tuned_fixed.get("valid", {}),
                "fallback": True,
            }
        opt_hold = holdout_result(
            cache, optimized["profile"], optimized["conditions"], cfg
        )

        modes[mode_id] = {
            "label": cfg["label"],
            "eval_days": cfg["eval_days"],
            "target": cfg["target"],
            "pine_exact_current_score": {
                "profile": "exact",
                "profile_detail": profile_by_id("exact"),
                "conditions": list(cfg["current"]),
                "holdout": exact_hold,
                "comparison": compare(mode_id, exact_hold),
            },
            "tuned_trigger_current_score": {
                "profile": tuned_fixed["profile"],
                "profile_detail": profile_by_id(tuned_fixed["profile"]),
                "conditions": list(cfg["current"]),
                "train": tuned_fixed.get("train", {}),
                "valid": tuned_fixed.get("valid", {}),
                "holdout": fixed_hold,
                "comparison": compare(mode_id, fixed_hold),
            },
            "v2_optimized": {
                "profile": optimized["profile"],
                "profile_detail": profile_by_id(optimized["profile"]),
                "conditions": optimized["conditions"],
                "train": optimized.get("train", {}),
                "valid": optimized.get("valid", {}),
                "fallback": optimized.get("fallback", False),
                "holdout": opt_hold,
                "comparison": compare(mode_id, opt_hold),
            },
        }

    result = {
        "generated_at_jst": core.now_jst().isoformat(timespec="seconds"),
        "history_start": history_start,
        "holdout_start": holdout_start,
        "end_date": args.end_date,
        "split": split,
        "jpx_list_date": list_date,
        "universe_count": len(issues),
        "yahoo_ok": ok,
        "yahoo_errors": sum(errors.values()),
        "error_counts": errors,
        "base_row_count": len(df),
        "profiles": PINE_PROFILES,
        "exact_trigger": exact_reason_summary(cache),
        "modes": modes,
    }
    return result


def main():
    today = core.now_jst().date()
    p = argparse.ArgumentParser()
    p.add_argument("--end-date", type=parse_date, default=today.isoformat())
    p.add_argument("--history-days", type=int, default=1825)
    p.add_argument("--holdout-days", type=int, default=365)
    p.add_argument("--max-workers", type=int, default=32)
    p.add_argument("--yahoo-range", default="5y")
    p.add_argument("--max-issues", type=int)
    p.add_argument("--output-dir", default="reports_independent_lab_v2")
    a = p.parse_args()
    if a.history_days <= a.holdout_days:
        raise SystemExit("history-days must be > holdout-days")

    result = run(a)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison_v2.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "comparison_v2.md").write_text(report_md(result), encoding="utf-8")
    print(json.dumps({
        "outputs": [str(out / "comparison_v2.json"), str(out / "comparison_v2.md")],
        "stable": result["modes"]["stable_s6"],
        "exact_trigger": result["exact_trigger"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
