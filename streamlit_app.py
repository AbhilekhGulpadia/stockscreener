import math
from io import BytesIO
from pathlib import Path
import os
import pandas as pd
import numpy as np
import yaml
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

############################################
# Streamlit App: Nifty-250 Analysis Suite  #
# Data Download + Steps 1..6 + Dashboard   #
############################################

# -------- Defaults (overridden by UI / YAML) --------
DEFAULT_CFG = {
    "general": {
        "data_dir": "nifty250_csv",
        "out_dir": "analysis_out",
        "top_n": 25
    },
    "benchmark": {
        "file": None,
        "autodetect": True,
        "detect_keywords": ["benchmark_", "nsei", "nifty50", "nifty"]
    },
    "weights": {
        "overall": {
            "sma": 0.25,
            "market": 0.15,
            "breakout": 0.25,
            "rs": 0.25,
            "risk": 0.10,
            "volatility": 0.00
        },
        "sma": {"position": 0.60, "trend": 0.40},
        "market": {"position": 0.65, "trend": 0.35},
        "breakout": {"volume": 0.50, "near_high": 0.35, "upper_range": 0.15},
        "rs": {"growth": 0.60, "near_high": 0.30, "consistency": 0.10},
        "risk": {"liquidity": 0.55, "stop": 0.45}
    },
    "thresholds": {
        "sma": {
            "trend_lookback_weeks": 3,
            "target_margin_above_sma": 0.05,
            "target_sma_growth": 0.01
        },
        "market": {
            "target_margin_above_sma": 0.03,
            "target_sma_growth": 0.005
        },
        "breakout": {
            "high_window_weeks": 52,
            "near_high_zero_at": -0.06,
            "volume_avg_weeks": 4,
            "target_vol_ratio": 2.0
        },
        "rs": {
            "lookback_weeks": 6,
            "high_window_weeks": 26,
            "target_rs_growth": 0.02,
            "near_high_zero_at": -0.01
        },
        "risk": {
            "turnover_target_min": 50_000_000,
            "turnover_target_full": 200_000_000,
            "support_lookback_weeks": 4,
            "stop_pct_best": 0.08,
            "stop_pct_worst": 0.15
        },
        "volatility": {
            "atr_lookback_weeks": 14,
            "atrp_best": 0.03,
            "atrp_worst": 0.10
        }
    }
}
CFG_FILE = "analysis_config.yaml"

# ---------------- Utility helpers ----------------
def deep_merge(base: dict, updates: dict) -> dict:
    out = base.copy()
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

@st.cache_data(show_spinner=False)
def load_yaml(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

# ---------------- Data IO ----------------
@st.cache_data(show_spinner=False)
def list_csvs(data_dir: str) -> list:
    p = Path(data_dir)
    if not p.exists():
        return []
    return sorted([str(x) for x in p.glob("*.csv")])

@st.cache_data(show_spinner=False)
def read_csv_cached(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    if "Price" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Price": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    return df[keep]

@st.cache_data(show_spinner=False)
def to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open":"first","High":"max","Low":"min","Close":"last"}
    if "Adj Close" in df_daily.columns:
        agg["Adj Close"] = "last"
    if "Volume" in df_daily.columns:
        agg["Volume"] = "sum"
    dfw = df_daily.resample("W-FRI").agg(agg).dropna(subset=["Close"])
    return dfw

# -------------- Feature builders --------------
def add_30w_sma(dfw: pd.DataFrame) -> pd.DataFrame:
    d = dfw.copy()
    d["SMA30"] = d["Close"].rolling(30, min_periods=30).mean()
    return d

# ---- Step 1: SMA scores ----
def sma_scores(dfw: pd.DataFrame, th: dict, wts: dict) -> dict:
    TREND_LOOKBACK_WEEKS = th["sma"]["trend_lookback_weeks"]
    TARGET_MARGIN_ABOVE_SMA = th["sma"]["target_margin_above_sma"]
    TARGET_SMA_GROWTH = th["sma"]["target_sma_growth"]

    out = {
        "close": None, "sma30": None, "pct_above_sma": None,
        "sma_change_pct": None, "sma_rising_fraction": None,
        "position_score": 0.0, "trend_score": 0.0, "sma_confidence": 0.0,
        "has_sma30": False, "weeks_available": len(dfw),
    }
    if len(dfw) < 30 or pd.isna(dfw["SMA30"].iloc[-1]):
        return out

    last = dfw.iloc[-1]
    out["close"] = float(last["Close"])
    out["sma30"] = float(last["SMA30"])
    out["has_sma30"] = True

    pct_above = (last["Close"]/last["SMA30"] - 1.0) if last["SMA30"] > 0 else np.nan
    out["pct_above_sma"] = pct_above
    if pd.notna(pct_above):
        out["position_score"] = round(100.0 * clamp01(pct_above / TARGET_MARGIN_ABOVE_SMA), 1)

    s = dfw["SMA30"].dropna()
    if len(s) > TREND_LOOKBACK_WEEKS:
        prev, cur = s.iloc[-(TREND_LOOKBACK_WEEKS+1)], s.iloc[-1]
        chg_pct = (cur/prev - 1.0) if prev > 0 else 0.0
        out["sma_change_pct"] = chg_pct
        diffs = s.diff().dropna().iloc[-TREND_LOOKBACK_WEEKS:]
        rising_frac = (diffs > 0).mean() if len(diffs) else 0.0
        out["sma_rising_fraction"] = round(float(rising_frac), 3)
        growth_component = clamp01(chg_pct / th["sma"]["target_sma_growth"])
        trend_component  = 0.7*growth_component + 0.3*rising_frac
        out["trend_score"] = round(100.0 * clamp01(trend_component), 1)

    comp = wts["sma"]["position"]*(out["position_score"]/100.0) + \
           wts["sma"]["trend"]*(out["trend_score"]/100.0)
    out["sma_confidence"] = round(100.0 * clamp01(comp), 1)
    return out

# ---- Step 2: Market / Benchmark ----
def find_benchmark_file(data_dir: Path, cfg_bench: dict) -> Path | None:
    if cfg_bench.get("file"):
        return Path(cfg_bench["file"])
    if not cfg_bench.get("autodetect", True):
        return None
    kws = [k.lower() for k in cfg_bench.get("detect_keywords", [])]
    cands = []
    for p in data_dir.glob("*.csv"):
        n = p.name.lower()
        if any(kw in n for kw in kws):
            cands.append(p)
    if not cands:
        return None
    def key(p: Path):
        n = p.name.lower()
        return (0 if n.startswith("benchmark_") else 1, 0 if "nsei" in n else 1, n)
    return sorted(cands, key=key)[0]

def market_scores(dfw_bench: pd.DataFrame, th: dict, wts: dict) -> dict:
    out = {
        "benchmark_name": None,
        "mkt_close": None, "mkt_sma30": None,
        "mkt_pct_above_sma": None, "mkt_sma_change_pct": None, "mkt_rising_fraction": None,
        "mkt_position_score": 0.0, "mkt_trend_score": 0.0, "market_confidence": 0.0,
        "mkt_has_sma30": False, "mkt_weeks_available": len(dfw_bench),
    }
    if len(dfw_bench) < 30 or pd.isna(dfw_bench["SMA30"].iloc[-1]):
        return out
    last = dfw_bench.iloc[-1]
    out["mkt_close"] = float(last["Close"])
    out["mkt_sma30"] = float(last["SMA30"])
    out["mkt_has_sma30"] = True

    pct_above = (last["Close"]/last["SMA30"] - 1.0) if last["SMA30"] > 0 else np.nan
    out["mkt_pct_above_sma"] = pct_above
    if pd.notna(pct_above):
        out["mkt_position_score"] = round(100.0 * clamp01(pct_above / th["market"]["target_margin_above_sma"]), 1)

    s = dfw_bench["SMA30"].dropna()
    L = th["sma"]["trend_lookback_weeks"]
    if len(s) > L:
        prev, cur = s.iloc[-(L+1)], s.iloc[-1]
        chg_pct = (cur/prev - 1.0) if prev > 0 else 0.0
        out["mkt_sma_change_pct"] = chg_pct
        diffs = s.diff().dropna().iloc[-L:]
        rising_frac = (diffs > 0).mean() if len(diffs) else 0.0
        out["mkt_rising_fraction"] = round(float(rising_frac), 3)
        growth_component = clamp01(chg_pct / th["market"]["target_sma_growth"])
        trend_component  = 0.7*growth_component + 0.3*rising_frac
        out["mkt_trend_score"] = round(100.0 * clamp01(trend_component), 1)

    comp = wts["market"]["position"]*(out["mkt_position_score"]/100.0) + \
           wts["market"]["trend"]*(out["mkt_trend_score"]/100.0)
    out["market_confidence"] = round(100.0 * clamp01(comp), 1)
    return out

# ---- Step 3: Breakout ----
def breakout_scores(dfw: pd.DataFrame, th: dict, wts: dict) -> dict:
    HIGH_WINDOW_WEEKS = th["breakout"]["high_window_weeks"]
    NEAR_HIGH_ZERO_AT = th["breakout"]["near_high_zero_at"]
    VOLUME_AVG_WEEKS  = th["breakout"]["volume_avg_weeks"]
    TARGET_VOL_RATIO  = th["breakout"]["target_vol_ratio"]

    out = {
        "near_high_pct": None, "vol_ratio": None, "upper_range_frac": None,
        "is_new_52w_high": False,
        "near_high_score": 0.0, "vol_surge_score": 0.0, "upper_range_score": 0.0,
        "breakout_confidence": 0.0,
    }
    if len(dfw) < max(HIGH_WINDOW_WEEKS, VOLUME_AVG_WEEKS) + 1 or "High" not in dfw or "Low" not in dfw:
        return out
    d = dfw.copy()
    d["prior_52w_high"] = d["High"].rolling(HIGH_WINDOW_WEEKS, min_periods=10).max().shift(1)
    if "Volume" in d.columns:
        d["vol_avg_4w"] = d["Volume"].rolling(VOLUME_AVG_WEEKS, min_periods=1).mean()
    last = d.iloc[-1]
    prior_high = last.get("prior_52w_high")
    vol_avg4   = last.get("vol_avg_4w")

    if pd.notna(prior_high) and prior_high > 0:
        nh_pct = last["Close"]/prior_high - 1.0
        out["near_high_pct"] = nh_pct
        out["is_new_52w_high"] = bool(nh_pct >= 0)
        scaled = (nh_pct - NEAR_HIGH_ZERO_AT) / (0.0 - NEAR_HIGH_ZERO_AT)
        out["near_high_score"] = round(100.0 * clamp01(scaled), 1)

    if "Volume" in d.columns and pd.notna(vol_avg4) and vol_avg4 and vol_avg4 > 0:
        vr = float(last["Volume"])/float(vol_avg4)
        out["vol_ratio"] = vr
        out["vol_surge_score"] = round(100.0 * clamp01(vr / TARGET_VOL_RATIO), 1)

    rng = last["High"] - last["Low"]
    if pd.notna(rng) and rng > 0:
        upper_frac = (last["Close"] - last["Low"]) / rng
        out["upper_range_frac"] = round(float(upper_frac), 3)
        out["upper_range_score"] = round(100.0 * clamp01(upper_frac), 1)

    comp = wts["breakout"]["volume"]*(out["vol_surge_score"]/100.0) + \
           wts["breakout"]["near_high"]*(out["near_high_score"]/100.0) + \
           wts["breakout"]["upper_range"]*(out["upper_range_score"]/100.0)
    out["breakout_confidence"] = round(100.0 * clamp01(comp), 1)
    return out

# ---- Step 4: RS ----
def rs_scores(dfw_stock: pd.DataFrame, dfw_bench: pd.DataFrame, th: dict, wts: dict) -> dict:
    RS_LOOKBACK_WEEKS    = th["rs"]["lookback_weeks"]
    RS_HIGH_WINDOW_WEEKS = th["rs"]["high_window_weeks"]
    TARGET_RS_GROWTH     = th["rs"]["target_rs_growth"]
    RS_NEAR_HIGH_ZERO_AT = th["rs"]["near_high_zero_at"]

    out = {
        "rs_last": None, "rs_change_pct": None, "rs_rising_fraction": None, "rs_near_high_pct": None,
        "rs_growth_score": 0.0, "rs_consistency_score": 0.0, "rs_high_score": 0.0, "rs_confidence": 0.0,
    }
    s = dfw_stock[["Close"]].rename(columns={"Close": "S"})
    b = dfw_bench[["Close"]].rename(columns={"Close": "B"})
    merged = s.join(b, how="inner").dropna()
    if len(merged) < max(RS_LOOKBACK_WEEKS + 1, RS_HIGH_WINDOW_WEEKS + 1):
        return out
    merged["RS"] = merged["S"] / merged["B"]
    rs = merged["RS"].dropna()
    rs_prev, rs_last = rs.iloc[-(RS_LOOKBACK_WEEKS+1)], rs.iloc[-1]
    out["rs_last"] = float(rs_last)
    change_pct = (rs_last/rs_prev - 1.0) if rs_prev > 0 else 0.0
    out["rs_change_pct"] = change_pct
    diffs = rs.diff().dropna().iloc[-RS_LOOKBACK_WEEKS:]
    rising_frac = (diffs > 0).mean() if len(diffs) else 0.0
    out["rs_rising_fraction"] = round(float(rising_frac), 3)
    prior = rs.rolling(RS_HIGH_WINDOW_WEEKS, min_periods=10).max().shift(1).iloc[-1]
    if pd.notna(prior) and prior > 0:
        nh = rs_last/prior - 1.0
        out["rs_near_high_pct"] = nh
        out["rs_high_score"] = round(100.0 * clamp01((nh - RS_NEAR_HIGH_ZERO_AT)/(0.0 - RS_NEAR_HIGH_ZERO_AT)), 1)
    out["rs_growth_score"] = round(100.0 * clamp01(change_pct / TARGET_RS_GROWTH), 1)
    out["rs_consistency_score"] = round(100.0 * clamp01(rising_frac), 1)
    comp = wts["rs"]["growth"]*(out["rs_growth_score"]/100.0) + \
           wts["rs"]["near_high"]*(out["rs_high_score"]/100.0) + \
           wts["rs"]["consistency"]*(out["rs_consistency_score"]/100.0)
    out["rs_confidence"] = round(100.0 * clamp01(comp), 1)
    return out

# ---- Step 5: Risk & Liquidity ----
def liquidity_scores(df_daily: pd.DataFrame, th: dict) -> dict:
    MINV = th["risk"]["turnover_target_min"]
    FULL = th["risk"]["turnover_target_full"]
    out = {"adv_20d": None, "adv_60d": None, "liquidity_score": 0.0}
    if "Volume" not in df_daily.columns or df_daily.empty:
        return out
    dv = (df_daily["Close"] * df_daily["Volume"]).astype("float64")
    adv20 = dv.rolling(20, min_periods=10).mean().iloc[-1] if len(dv) >= 10 else None
    adv60 = dv.rolling(60, min_periods=20).median().iloc[-1] if len(dv) >= 20 else None
    out["adv_20d"] = float(adv20) if pd.notna(adv20) else None
    out["adv_60d"] = float(adv60) if pd.notna(adv60) else None
    if adv20 is not None and pd.notna(adv20):
        scaled = (adv20 - MINV) / (FULL - MINV)
        out["liquidity_score"] = round(100.0 * clamp01(scaled), 1)
    return out

def stop_distance_scores(dfw: pd.DataFrame, th: dict) -> dict:
    LOOK = th["risk"]["support_lookback_weeks"]
    BEST = th["risk"]["stop_pct_best"]
    WORST= th["risk"]["stop_pct_worst"]
    out = {"support_ref": None, "stop_distance_pct": None, "stop_score": 0.0}
    if len(dfw) < max(LOOK + 1, 5):
        return out
    d = dfw.copy()
    d["swing_low_prior"] = d["Low"].rolling(LOOK, min_periods=1).min().shift(1)
    last = d.iloc[-1]
    cands = [last.get("swing_low_prior")]
    if pd.notna(last.get("SMA30", np.nan)):
        cands.append(last["SMA30"])
    support = max([s for s in cands if pd.notna(s)], default=np.nan)
    out["support_ref"] = float(support) if pd.notna(support) else None
    if pd.notna(support) and last["Close"] > 0:
        stop_pct = (last["Close"] - support) / last["Close"]
        if stop_pct < 0:
            stop_pct = WORST + 0.05
        out["stop_distance_pct"] = float(stop_pct)
        if stop_pct <= BEST:
            score = 1.0
        elif stop_pct >= WORST:
            score = 0.0
        else:
            score = 1.0 - (stop_pct - BEST) / (WORST - BEST)
        out["stop_score"] = round(100.0 * clamp01(score), 1)
    return out

def risk_confidence(liq_score: float, stop_score: float, wts: dict) -> float:
    comp = (wts["risk"]["liquidity"] * (liq_score/100.0)) + (wts["risk"]["stop"] * (stop_score/100.0))
    return round(100.0 * clamp01(comp), 1)

# ---- Step: Volatility (ATR%) ----
def atr_volatility_scores(dfw: pd.DataFrame, th: dict) -> dict:
    N = th["volatility"]["atr_lookback_weeks"]
    BEST = th["volatility"]["atrp_best"]
    WORST= th["volatility"]["atrp_worst"]
    out = {"atr": None, "atr_pct": None, "volatility_confidence": 0.0}
    if len(dfw) < N + 2 or not set(["High","Low","Close"]).issubset(dfw.columns):
        return out
    d = dfw.copy()
    d["prev_close"] = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["prev_close"]).abs(),
        (d["Low"]  - d["prev_close"]).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(N, min_periods=N).mean()
    last_atr = atr.iloc[-1] if pd.notna(atr.iloc[-1]) else None
    out["atr"] = float(last_atr) if last_atr is not None else None
    if last_atr is not None and d["Close"].iloc[-1] > 0:
        atr_pct = last_atr / d["Close"].iloc[-1]
        out["atr_pct"] = float(atr_pct)
        if atr_pct <= BEST:
            sc = 1.0
        elif atr_pct >= WORST:
            sc = 0.0
        else:
            sc = 1.0 - (atr_pct - BEST) / (WORST - BEST)
        out["volatility_confidence"] = round(100.0 * clamp01(sc), 1)
    return out

# ---- Overall blend ----
def normalize_overall_weights(w_overall: dict) -> dict:
    s = sum(max(0.0, float(v)) for v in w_overall.values())
    if s == 0:
        return w_overall
    return {k: float(v)/s for k, v in w_overall.items()}

def overall_score(row: pd.Series, w_overall_norm: dict) -> float:
    parts = {
        "sma": row.get("sma_confidence", 0.0),
        "market": row.get("market_confidence", 0.0),
        "breakout": row.get("breakout_confidence", 0.0),
        "rs": row.get("rs_confidence", 0.0),
        "risk": row.get("risk_confidence", 0.0),
        "volatility": row.get("volatility_confidence", 0.0),
    }
    s = 0.0
    for k, v in parts.items():
        w = w_overall_norm.get(k, 0.0)
        if pd.notna(v):
            s += w * (float(v)/100.0)
    return round(100.0 * clamp01(s), 1)

# ---- Per-ticker evaluation ----
def evaluate_ticker(sym: str, df_daily: pd.DataFrame, df_bench_w: pd.DataFrame | None, th: dict, wts: dict) -> dict:
    try:
        dfw = add_30w_sma(to_weekly(df_daily))
        sma = sma_scores(dfw, th, wts)
        brk = breakout_scores(dfw, th, wts)
        row = {
            "ticker": sym,
            **sma,
            "above_sma": bool(sma["pct_above_sma"] is not None and (pd.notna(sma["pct_above_sma"]) and sma["pct_above_sma"] > 0)),
            **brk,
        }
        if df_bench_w is not None and not df_bench_w.empty:
            row.update(rs_scores(dfw, df_bench_w, th, wts))
        else:
            row.update({
                "rs_confidence": 0.0, "rs_growth_score": 0.0,
                "rs_consistency_score": 0.0, "rs_high_score": 0.0,
                "rs_last": None, "rs_change_pct": None, "rs_rising_fraction": None, "rs_near_high_pct": None
            })
        row.update(liquidity_scores(df_daily, th))
        row.update(stop_distance_scores(dfw, th))
        row["risk_confidence"] = risk_confidence(row.get("liquidity_score", 0.0), row.get("stop_score", 0.0), wts)
        row.update(atr_volatility_scores(dfw, th))
        return row
    except Exception as e:
        return {"ticker": sym, "error": str(e)}

# --------------- Plotting helpers ---------------
def plot_price_sma(dfw: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dfw.index, open=dfw['Open'], high=dfw['High'], low=dfw['Low'], close=dfw['Close'], name='Price'))
    if 'SMA30' in dfw.columns:
        fig.add_trace(go.Scatter(x=dfw.index, y=dfw['SMA30'], mode='lines', name='SMA30'))
    fig.update_layout(title=f"{ticker} — Weekly Price & SMA30", xaxis_title="Week", yaxis_title="Price")
    return fig

def plot_rs(dfw_stock: pd.DataFrame, dfw_bench: pd.DataFrame, ticker: str):
    s = dfw_stock[["Close"]].rename(columns={"Close": "S"})
    b = dfw_bench[["Close"]].rename(columns={"Close": "B"})
    merged = s.join(b, how="inner").dropna()
    if merged.empty:
        return go.Figure()
    merged["RS"] = merged["S"]/merged["B"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged.index, y=merged['RS'], mode='lines', name='RS (Stock / Benchmark)'))
    fig.update_layout(title=f"{ticker} — Relative Strength vs Benchmark", xaxis_title="Week", yaxis_title="RS")
    return fig

def scatter_dashboard(df: pd.DataFrame):
    if df.empty:
        return go.Figure()
    # Default scatter: RS vs Breakout, size by Liquidity, color by SMA
    size = df.get("liquidity_score", pd.Series([50]*len(df)))
    color = df.get("sma_confidence", pd.Series([50]*len(df)))
    fig = go.Figure(data=go.Scatter(
        x=df.get("rs_confidence", pd.Series([0]*len(df))),
        y=df.get("breakout_confidence", pd.Series([0]*len(df))),
        mode='markers',
        marker=dict(size=10 + 0.2*size.fillna(0), color=color, showscale=True),
        text=df.get("ticker")
    ))
    fig.update_layout(title="RS vs Breakout (size: Liquidity, color: SMA)", xaxis_title="RS confidence", yaxis_title="Breakout confidence")
    return fig

# --------------- Data Download utilities ---------------
def parse_tickers_from_excel(file_bytes: bytes | None, path: str | None) -> list:
    try:
        if file_bytes:
            df = pd.read_excel(BytesIO(file_bytes))
        else:
            df = pd.read_excel(path)
        candidate_cols = [c for c in ["Ticker","ticker","Symbol","symbol"] if c in df.columns]
        tickers = (df[candidate_cols[0]] if candidate_cols else df.iloc[:,0]).dropna().astype(str).str.strip()
        return tickers.tolist()
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        return []

def save_clean_csv(ticker: str, out_dir: Path, period: str = "1y", interval: str = "1d", force: bool=False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.csv"
    if out_path.exists() and not force:
        return {"ticker": ticker, "status": "skipped_exists"}
    try:
        ddata = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        if ddata.empty:
            return {"ticker": ticker, "status": "empty"}
        ddata.index.name = "Date"
        keep_cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in ddata.columns]
        ddata = ddata[keep_cols]
        ddata.to_csv(out_path)
        # Safety fix for stray rows (Price header)
        df_raw = pd.read_csv(out_path)
        if df_raw.columns[0].strip().lower() == "price":
            df_fixed = df_raw.drop(index=[0,1]).rename(columns={"Price":"Date"}).reset_index(drop=True)
            df_fixed.to_csv(out_path, index=False)
        return {"ticker": ticker, "status": "ok", "rows": len(ddata)}
    except Exception as e:
        return {"ticker": ticker, "status": f"error: {e}"}

def download_batch(tickers: list, out_dir: Path, period: str, interval: str, force: bool, include_benchmark: bool) -> pd.DataFrame:
    logs = []
    prog = st.progress(0, text="Starting downloads...")
    total = len(tickers) + (1 if include_benchmark else 0)
    for i, t in enumerate(tickers, 1):
        res = save_clean_csv(t, out_dir, period, interval, force)
        logs.append(res)
        prog.progress(i/total, text=f"{t}: {res['status']}")
    if include_benchmark:
        bench_res = save_clean_csv("^NSEI", out_dir, period="5y", interval="1d", force=force)
        # rename benchmark file for clarity
        try:
            src = out_dir / "^NSEI.csv"
            if src.exists():
                dst = out_dir / "benchmark_nifty50.csv"
                src.replace(dst)
        except Exception:
            pass
        logs.append({"ticker": "^NSEI", "status": bench_res.get("status", "ok")})
        prog.progress(1.0, text="Benchmark done")
    return pd.DataFrame(logs)

# --------------- App UI ---------------
st.set_page_config(page_title="Nifty-250 Stock Analyzer", layout="wide")
st.title("📈 Nifty-250 Stock Analyzer — Weinstein-style Scoring")

# Tabs
tab_dl, tab_scores, tab_dash = st.tabs(["📥 Data Download", "🧮 Scores", "📊 Dashboard"])

# Sidebar: config
with st.sidebar:
    st.header("⚙️ Configuration")
    data_dir = st.text_input("Data folder", value=DEFAULT_CFG["general"]["data_dir"])
    out_dir = st.text_input("Output folder", value=DEFAULT_CFG["general"]["out_dir"])
    top_n = st.slider("Top N", min_value=5, max_value=100, value=DEFAULT_CFG["general"]["top_n"], step=5)

    cfg_path = Path(CFG_FILE)
    user_cfg = load_yaml(cfg_path)
    cfg = deep_merge(DEFAULT_CFG, user_cfg)

    st.subheader("Overall Weights")
    w_overall = cfg["weights"]["overall"].copy()
    w_overall["sma"] = st.slider("SMA weight", 0.0, 1.0, float(w_overall["sma"]))
    w_overall["market"] = st.slider("Market weight", 0.0, 1.0, float(w_overall["market"]))
    w_overall["breakout"] = st.slider("Breakout weight", 0.0, 1.0, float(w_overall["breakout"]))
    w_overall["rs"] = st.slider("RS weight", 0.0, 1.0, float(w_overall["rs"]))
    w_overall["risk"] = st.slider("Risk weight", 0.0, 1.0, float(w_overall["risk"]))
    w_overall["volatility"] = st.slider("Volatility weight", 0.0, 1.0, float(w_overall["volatility"]))

# ---------- TAB 1: Data Download ----------
with tab_dl:
    st.subheader("Download or Update Price Data")
    colA, colB, colC = st.columns([2,1,1])
    with colA:
        uploaded = st.file_uploader("Upload symbols Excel (one column with tickers like SBIN.NS)", type=["xlsx","xls"])    
        excel_path_input = st.text_input("OR path to symbols.xlsx", value="symbols.xlsx")
        include_benchmark = st.checkbox("Also download ^NSEI as benchmark", value=True)
    with colB:
        period = st.selectbox("Period", ["6mo","1y","2y","5y","max"], index=1)
        interval = st.selectbox("Interval", ["1d","1wk"], index=0)
    with colC:
        force = st.checkbox("Force overwrite existing CSVs", value=False)
        run_dl = st.button("📥 Download / Update", type="primary")

    if run_dl:
        # Determine tickers
        if uploaded is not None:
            tickers = parse_tickers_from_excel(uploaded.getvalue(), None)
        else:
            tickers = parse_tickers_from_excel(None, excel_path_input)
        if not tickers:
            st.error("No tickers found in the provided Excel.")
        else:
            st.info(f"Found {len(tickers)} tickers. Saving to {data_dir}")
            logs = download_batch(tickers, Path(data_dir), period, interval, force, include_benchmark)
            # Clear caches so new files get picked up
            list_csvs.clear()
            read_csv_cached.clear()
            to_weekly.clear()
            st.success("Download complete.")
            st.dataframe(logs, use_container_width=True, height=320)

# Discover files (post-download safe)
csv_files = list_csvs(data_dir)
bench_file = find_benchmark_file(Path(data_dir), cfg["benchmark"]) if csv_files else None

# ---------- Common: analysis run button ----------
run_btn = tab_scores.button("🚀 Run Analysis", type="primary")

# ---------- TAB 2: Scores ----------
with tab_scores:
    if not csv_files:
        st.warning(f"No CSVs found in '{data_dir}'. Use the Data Download tab first.")
    else:
        if bench_file is None:
            st.info("No benchmark detected. Add ^NSEI CSV or check the download option. RS & Market scores will be zero.")
        if run_btn:
            with st.spinner("Running analysis..."):
                # Load benchmark
                df_bench_w = None
                mkt_broadcast = {"benchmark_name": None}
                if bench_file is not None:
                    try:
                        df_bench = read_csv_cached(str(bench_file))
                        df_bench_w = add_30w_sma(to_weekly(df_bench))
                        mkt_broadcast = market_scores(df_bench_w, cfg["thresholds"], cfg["weights"])
                        mkt_broadcast["benchmark_name"] = Path(bench_file).name
                    except Exception as e:
                        st.error(f"Benchmark error: {e}")
                        df_bench_w = None

                rows = []
                prog = st.progress(0, text="Evaluating tickers...")
                filtered_files = [fp for fp in csv_files if Path(fp).name.lower() not in {"download_log.csv"} and not Path(fp).name.lower().startswith("benchmark_")]
                for i, fp in enumerate(filtered_files, 1):
                    sym = Path(fp).stem
                    try:
                        df_daily = read_csv_cached(fp)
                        row = evaluate_ticker(sym, df_daily, df_bench_w, cfg["thresholds"], cfg["weights"])
                        rows.append(row)
                    except Exception as e:
                        rows.append({"ticker": sym, "error": str(e)})
                    prog.progress(i/len(filtered_files), text=f"Processed {i}/{len(filtered_files)}")

                df = pd.DataFrame(rows)
                # Broadcast market columns
                for k, v in mkt_broadcast.items():
                    df[k] = v

                # Final score with live overall weights
                w_overall_norm = normalize_overall_weights(w_overall)
                df["final_confidence"] = df.apply(lambda r: overall_score(r, w_overall_norm), axis=1)

                # Save outputs
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                full_path = Path(out_dir)/"app_full_scores.csv"
                df.sort_values("final_confidence", ascending=False).to_csv(full_path, index=False)

                # Top-N view
                top_cols = [c for c in [
                    "ticker", "final_confidence",
                    "sma_confidence", "market_confidence", "breakout_confidence",
                    "rs_confidence", "risk_confidence", "volatility_confidence",
                    "near_high_score", "vol_surge_score", "rs_growth_score", "rs_high_score",
                    "position_score", "trend_score", "mkt_position_score", "mkt_trend_score",
                    "liquidity_score", "stop_score", "atr_pct"
                ] if c in df.columns]
                df_top = df.sort_values("final_confidence", ascending=False).head(top_n)[top_cols]
                top_path = Path(out_dir)/"app_topN_overall.csv"
                df_top.to_csv(top_path, index=False)

                st.success("Analysis complete.")
                c1, c2, c3 = st.columns([2,1,1])
                with c1:
                    st.subheader("Top Ranked (live weights)")
                    st.dataframe(df_top, use_container_width=True, height=480)
                with c2:
                    st.metric("Universe size", len(df))
                    st.metric("Benchmark", mkt_broadcast.get("benchmark_name", "None"))
                with c3:
                    st.download_button("⬇️ Download Full CSV", data=open(full_path,'rb').read(), file_name=full_path.name)
                    st.download_button("⬇️ Download Top-N CSV", data=open(top_path,'rb').read(), file_name=top_path.name)

                # Persist DataFrame for dashboard tab
                st.session_state["last_df_scores"] = df
                st.session_state["bench_w"] = df_bench_w

# ---------- TAB 3: Dashboard ----------
with tab_dash:
    st.subheader("Interactive Dashboard")
    df = st.session_state.get("last_df_scores", pd.DataFrame())
    df = df.copy()
    if df.empty:
        st.info("Run the analysis in the Scores tab to populate the dashboard.")
    else:
        # Filters
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            min_final = st.slider("Min Final", 0, 100, 0)
            min_rs = st.slider("Min RS", 0, 100, 0)
        with fcol2:
            min_breakout = st.slider("Min Breakout", 0, 100, 0)
            min_sma = st.slider("Min SMA", 0, 100, 0)
        with fcol3:
            min_market = st.slider("Min Market", 0, 100, 0)
            min_risk = st.slider("Min Risk", 0, 100, 0)
        with fcol4:
            min_vol = st.slider("Min Volatility", 0, 100, 0)
            search = st.text_input("Search ticker contains")

        mask = (
            (df.get("final_confidence",0) >= min_final) &
            (df.get("rs_confidence",0) >= min_rs) &
            (df.get("breakout_confidence",0) >= min_breakout) &
            (df.get("sma_confidence",0) >= min_sma) &
            (df.get("market_confidence",0) >= min_market) &
            (df.get("risk_confidence",0) >= min_risk) &
            (df.get("volatility_confidence",0) >= min_vol)
        )
        if search:
            mask &= df["ticker"].str.contains(search, case=False, na=False)

        df_f = df.loc[mask].sort_values("final_confidence", ascending=False)

        cA, cB = st.columns([2,1])
        with cA:
            st.plotly_chart(scatter_dashboard(df_f), use_container_width=True)
            st.dataframe(df_f[[c for c in [
                "ticker","final_confidence","rs_confidence","breakout_confidence","sma_confidence",
                "market_confidence","risk_confidence","volatility_confidence","adv_20d","atr_pct"
            ] if c in df_f.columns]], use_container_width=True, height=500)
        with cB:
            # Simple histograms
            for metric in ["final_confidence","rs_confidence","breakout_confidence","sma_confidence"]:
                if metric in df_f.columns:
                    hist = go.Figure(data=[go.Histogram(x=df_f[metric])])
                    hist.update_layout(title=f"{metric} distribution")
                    st.plotly_chart(hist, use_container_width=True)

        # Stock detail quick look
        st.divider()
        sel = st.selectbox("Inspect ticker", options=sorted(df_f["ticker"].unique().tolist()))
        if sel:
            fp = str(Path(data_dir)/f"{sel}.csv")
            df_daily = read_csv_cached(fp)
            dfw = add_30w_sma(to_weekly(df_daily))
            c1, c2 = st.columns([2,1])
            with c1:
                st.plotly_chart(plot_price_sma(dfw, sel), use_container_width=True)
                bench_w = st.session_state.get("bench_w")
                if bench_w is not None:
                    st.plotly_chart(plot_rs(dfw, bench_w, sel), use_container_width=True)
            with c2:
                row = df[df["ticker"]==sel].iloc[0].fillna(0)
                st.write("**Scores**")
                for key in ["final_confidence","sma_confidence","breakout_confidence","rs_confidence","market_confidence","risk_confidence","volatility_confidence"]:
                    st.progress(float(row.get(key,0))/100.0, text=f"{key} {row.get(key,0)}")
