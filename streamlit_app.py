import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from arch import arch_model
from scipy import stats
from groq import Groq
from datetime import date
import re
import json
import base64
import warnings
warnings.filterwarnings("ignore")

FF_URL = "https://raw.githubusercontent.com/lakshyaworkch-cell/Lakshy/main/F-F_Research_Data_5_Factors_2x3.csv"

# AQR public data URLs (monthly, Global except US — using US proxy via their Excel files)
# AQR hosts Excel files; we use a CSV mirror for QMJ and BAB
AQR_QMJ_URL = "https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx"
AQR_BAB_URL = "https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx"

st.set_page_config(page_title="Factor Regression", page_icon="📈", layout="wide")

def get_bg_base64():
    try:
        with open("ranger-4df6c1b6.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

_BG = get_bg_base64()

_BG_CSS = (
    f"""background-image:
        linear-gradient(rgba(4,52,44,0.88), rgba(4,52,44,0.88)),
        url("data:image/png;base64,{_BG}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;"""
    if _BG else
    """background: #1a0933;
    background-image:
        radial-gradient(ellipse at 15% 0%, rgba(29,158,117,0.18) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 100%, rgba(15,110,86,0.14) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 50%, rgba(4,52,44,0.95) 0%, transparent 100%);"""
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{
    {_BG_CSS}
    color: #e2e8f0;
}}
h1, h2, h3 {{ font-family: 'JetBrains Mono', monospace !important; letter-spacing: -0.5px; }}

/* === METRIC CARDS === */
.metric-card {{
    background: rgba(29,158,117,0.08); border: 1px solid rgba(29,158,117,0.18);
    border-left: 2px solid #34d399; border-radius: 12px; padding: 14px 16px;
    margin-bottom: 10px;
}}
.metric-card.red  {{ border-left-color: #f87171; }}
.metric-card.blue {{ border-left-color: #60a5fa; }}
.metric-card.gold {{ border-left-color: #fbbf24; }}
.metric-card.gray {{ border-left-color: #6b7280; }}
.metric-label {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 1px; color: #5DCAA5; text-transform: uppercase; margin-bottom: 4px; }}
.metric-value {{ font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 600; color: #e2e8f0; }}
.metric-sub {{ font-size: 12px; color: #9FE1CB; margin-top: 4px; font-family: 'JetBrains Mono', monospace; word-break: break-all; }}

/* === FACTOR ROWS — stack on mobile, grid on desktop === */
.factor-row {{
    display: flex; flex-wrap: wrap; gap: 6px 12px;
    align-items: center; padding: 10px 12px; border-radius: 8px;
    margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; font-size: 13px;
    background: rgba(29,158,117,0.05); border: 1px solid rgba(29,158,117,0.12);
}}
.factor-row > div:first-child {{ width: 100%; font-size: 13px; color: #e2e8f0; font-weight: 500; }}
.factor-row > div:not(:first-child) {{ flex: 1 1 60px; min-width: 55px; }}
.factor-row:hover {{ background: rgba(29,158,117,0.09); }}
.factor-row.header {{ background: transparent; border-color: transparent; font-size: 12px; letter-spacing: 1px; color: #5DCAA5; text-transform: uppercase; }}
.factor-row.header > div:first-child {{ width: 100%; }}
.factor-row.sig   {{ border-left: 2px solid #34d399; }}
.factor-row.marg  {{ border-left: 2px solid #fbbf24; }}
.factor-row.insig {{ border-left: 2px solid rgba(29,158,117,0.2); }}
.factor-row.alpha {{ border-left: 2px solid #a78bfa; }}
@media (min-width: 600px) {{
    .factor-row {{
        display: grid;
        grid-template-columns: 140px 90px 90px 90px 80px 1fr;
    }}
    .factor-row > div:first-child {{ width: auto; }}
}}

/* === SIG BADGES === */
.sig-badge {{ display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 12px; font-weight: 500; letter-spacing: 1px; }}
.badge-001 {{ background: rgba(52,211,153,0.15); color: #34d399; }}
.badge-01  {{ background: rgba(52,211,153,0.10); color: #6ee7b7; }}
.badge-05  {{ background: rgba(251,191,36,0.12); color: #fbbf24; }}
.badge-10  {{ background: rgba(255,255,255,0.05); color: #6b7280; }}
.badge-ns  {{ background: rgba(255,255,255,0.03); color: #4a7c6f; }}

/* === SECTION TITLES === */
.section-title {{
    font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 2px;
    text-transform: uppercase; color: #5DCAA5; border-bottom: 1px solid rgba(29,158,117,0.2);
    padding-bottom: 8px; margin: 24px 0 14px 0;
}}

/* === DIAGNOSTICS GRID — 1 col mobile, 2 col desktop === */
.diag-grid {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
@media (min-width: 500px) {{ .diag-grid {{ grid-template-columns: 1fr 1fr; }} }}
.diag-item {{ background: rgba(29,158,117,0.06); border: 1px solid rgba(29,158,117,0.15); border-radius: 10px; padding: 14px 16px; }}
.diag-name {{ font-size: 12px; color: #5DCAA5; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; }}
.diag-val  {{ font-size: 16px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }}
.diag-pass {{ color: #34d399; }} .diag-fail {{ color: #f87171; }} .diag-warn {{ color: #fbbf24; }}
.diag-sub  {{ font-size: 12px; color: #9FE1CB; font-family: 'JetBrains Mono', monospace; margin-top: 3px; }}

/* === INTERPRET / AI BOXES === */
.interpret-box {{ background: rgba(29,158,117,0.06); border: 1px solid rgba(29,158,117,0.18); border-radius: 12px; padding: 18px 20px; font-size: 14px; line-height: 1.8; color: #9FE1CB; }}
.interpret-box b {{ color: #e2e8f0; }}

/* === FACTOR INSIGHT CARDS === */
.factor-insight-card {{
    background: rgba(29,158,117,0.05); border: 1px solid rgba(29,158,117,0.15);
    border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;
}}
.factor-insight-card.positive {{ border-left: 3px solid #34d399; }}
.factor-insight-card.negative {{ border-left: 3px solid #f87171; }}
.factor-insight-card.neutral  {{ border-left: 3px solid #6b7280; }}
.fi-header {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 10px; }}
.fi-name {{ font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; color: #e2e8f0; letter-spacing: 1px; }}
.fi-beta {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; padding: 2px 8px; border-radius: 4px; }}
.fi-beta.pos {{ background: rgba(52,211,153,0.12); color: #34d399; }}
.fi-beta.neg {{ background: rgba(248,113,113,0.12); color: #f87171; }}
.fi-sig-label {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 1px; color: #5DCAA5; text-transform: uppercase; }}
.fi-outlook {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase; margin-left: auto;
}}
.fi-outlook.bullish  {{ background: rgba(52,211,153,0.1); color: #34d399; }}
.fi-outlook.bearish  {{ background: rgba(248,113,113,0.1); color: #f87171; }}
.fi-outlook.neutral  {{ background: rgba(107,114,128,0.1); color: #6b7280; }}
.fi-outlook.mixed    {{ background: rgba(251,191,36,0.1);  color: #fbbf24; }}
.fi-body {{ font-size: 13px; color: #9FE1CB; line-height: 1.85; }}
.fi-news {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(29,158,117,0.15); font-size: 13px; color: #5DCAA5; font-family: 'JetBrains Mono', monospace; }}
.fi-news-label {{ font-size: 11px; letter-spacing: 1px; text-transform: uppercase; color: #1D9E75; margin-bottom: 4px; }}

/* === AI SUMMARY BOXES === */
.ai-summary-box {{
    background: rgba(29,158,117,0.07); border: 1px solid rgba(29,158,117,0.2);
    border-radius: 10px; padding: 14px 18px; margin-top: 14px;
    font-size: 13px; color: #9FE1CB; line-height: 1.8;
}}
.ai-summary-box b {{ color: #e2e8f0; }}

/* === AQR BADGE === */
.aqr-badge {{
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase;
    background: rgba(251,191,36,0.12); color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.25); margin-left: 4px; vertical-align: middle;
}}

/* === OLS SUMMARY BOX === */
.ols-box {{
    background: rgba(10,30,25,0.85);
    border: 1px solid rgba(29,158,117,0.25);
    border-radius: 12px;
    padding: 20px 24px;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}}
.ols-box pre {{
    font-family: 'JetBrains Mono', 'Fira Mono', monospace;
    font-size: 12.5px;
    line-height: 1.75;
    color: #9FE1CB;
    margin: 0;
    white-space: pre;
    tab-size: 4;
}}

/* === INPUTS & BUTTONS === */
.stTextInput > div > div > input {{
    background: rgba(29,158,117,0.08) !important;
    border: 1px solid rgba(29,158,117,0.25) !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 8px !important; font-size: 14px !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: rgba(52,211,153,0.5) !important;
    box-shadow: 0 0 0 3px rgba(52,211,153,0.1) !important;
}}
.stButton > button {{
    background: rgba(29,158,117,0.12) !important; color: #9FE1CB !important;
    font-family: 'JetBrains Mono', monospace !important; font-weight: 500 !important;
    letter-spacing: 1px !important; border: 1px solid rgba(29,158,117,0.3) !important;
    border-radius: 8px !important; padding: 10px 20px !important; font-size: 13px !important;
}}
.stButton > button:hover {{
    background: rgba(52,211,153,0.15) !important;
    border-color: rgba(52,211,153,0.5) !important;
    color: #34d399 !important;
}}
.error-box {{
    background: rgba(248,113,113,0.06); border: 1px solid rgba(248,113,113,0.2);
    border-radius: 8px; padding: 12px 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #f87171;
}}
.info-box {{
    background: rgba(29,158,117,0.08); border: 1px solid rgba(29,158,117,0.25);
    border-radius: 8px; padding: 12px 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #5DCAA5;
}}

/* === PORTFOLIO === */
.port-summary-card {{
    background: linear-gradient(135deg, rgba(29,158,117,0.1), rgba(15,110,86,0.1));
    border: 1px solid rgba(29,158,117,0.2); border-radius: 12px;
    padding: 18px 20px; margin-bottom: 14px;
}}
.port-attribution-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 12px; }}
@media (min-width: 500px) {{ .port-attribution-grid {{ grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }} }}
.port-attr-cell {{
    background: rgba(29,158,117,0.06); border: 1px solid rgba(29,158,117,0.15);
    border-radius: 8px; padding: 12px 14px;
}}
.port-attr-factor {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #5DCAA5; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px; }}
.port-attr-beta {{ font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 600; }}
.port-attr-beta.pos {{ color: #34d399; }}
.port-attr-beta.neg {{ color: #f87171; }}

/* === REGIME BADGES === */
.regime-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase;
}}
.regime-badge.low  {{ background: rgba(52,211,153,0.12); color: #34d399; }}
.regime-badge.med  {{ background: rgba(251,191,36,0.12); color: #fbbf24; }}
.regime-badge.high {{ background: rgba(248,113,113,0.12); color: #f87171; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

# FF5 + Momentum factors
FF_FACTORS  = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
# AQR factors
AQR_FACTORS = ["QMJ", "BAB"]
ALL_FACTORS = FF_FACTORS + AQR_FACTORS

# Which factors come from AQR (used for badge rendering)
AQR_FACTOR_SET = set(AQR_FACTORS)

def sig_badge(p):
    if p < 0.001: return '<span class="sig-badge badge-001">★★★</span>'
    elif p < 0.01: return '<span class="sig-badge badge-01">★★★</span>'
    elif p < 0.05: return '<span class="sig-badge badge-05">★★</span>'
    elif p < 0.10: return '<span class="sig-badge badge-10">★</span>'
    else: return '<span class="sig-badge badge-ns">n.s.</span>'

def row_class(name, p):
    if name == "const": return "factor-row alpha"
    if p < 0.05: return "factor-row sig"
    if p < 0.10: return "factor-row marg"
    return "factor-row insig"

def fmt_beta(v):
    color = "#34d399" if v > 0 else "#f87171"
    return f'<span style="color:{color};font-weight:600">{v:+.4f}</span>'

def fmt_pval(p):
    if p < 0.001: color = "#34d399"
    elif p < 0.05: color = "#6ee7b7"
    elif p < 0.10: color = "#fbbf24"
    else: color = "#5DCAA5"
    return f'<span style="color:{color}">{p:.4f}</span>'

def fmt_tstat(t):
    color = "#34d399" if abs(t) > 1.96 else "#5DCAA5"
    return f'<span style="color:{color}">{t:+.3f}</span>'

FACTOR_NAMES = {
    "const":  "Alpha (α)",
    "Mkt-RF": "Market (β)",
    "SMB":    "Size",
    "HML":    "Value",
    "RMW":    "Profitability",
    "CMA":    "Investment",
    "Mom":    "Momentum",
    "QMJ":    "Quality (QMJ)",
    "BAB":    "Betting Agst Beta",
}

FACTOR_DESCRIPTIONS = {
    "Mkt-RF": "Market risk premium (excess return of market over risk-free rate)",
    "SMB":    "Size factor — Small Minus Big (small-cap vs large-cap premium)",
    "HML":    "Value factor — High Minus Low (value vs growth premium)",
    "RMW":    "Profitability factor — Robust Minus Weak (profitable vs unprofitable firms)",
    "CMA":    "Investment factor — Conservative Minus Aggressive (low vs high investment firms)",
    "Mom":    "Momentum factor (past winners vs past losers)",
    "QMJ":    "Quality Minus Junk (AQR) — long high-quality stocks, short low-quality/junk stocks",
    "BAB":    "Betting Against Beta (AQR) — long low-beta stocks, short high-beta stocks (leverage-constrained anomaly)",
}

FACTOR_COLORS = {
    "Mkt-RF": "#60a5fa", "SMB": "#34d399", "HML": "#fbbf24",
    "RMW": "#a78bfa", "CMA": "#f97316", "Mom": "#ec4899",
    "QMJ":  "#facc15",   # gold — AQR quality
    "BAB":  "#38bdf8",   # sky blue — AQR BAB
    "const": "#94a3b8",
}

# Regime display config
REGIME_LABELS_2 = {0: "Low Volatility", 1: "High Volatility"}
REGIME_LABELS_3 = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility"}
REGIME_COLORS   = {0: "#34d399", 1: "#fbbf24", 2: "#f87171"}
REGIME_CSS_CLASS = {0: "low", 1: "med", 2: "high"}

def regime_label(k, n_regimes):
    if n_regimes == 2:
        return REGIME_LABELS_2.get(k, f"Regime {k}")
    return REGIME_LABELS_3.get(k, f"Regime {k}")

def regime_css(k, n_regimes):
    if n_regimes == 2:
        return "low" if k == 0 else "high"
    return REGIME_CSS_CLASS.get(k, "med")

@st.cache_data(show_spinner=False)
def load_factors():
    ff = pd.read_csv(FF_URL, index_col=0)
    ff.columns = ff.columns.str.strip()
    ff.index = ff.index.astype(str).str.strip()
    ff = ff.drop(columns=[c for c in ff.columns if "Unnamed" in str(c)])
    ff = ff[ff.index.str.match(r"^\d{6}$")].copy()
    ff.index = pd.to_datetime(ff.index, format="%Y%m").to_period("M")
    return ff


@st.cache_data(show_spinner=False)
def load_aqr_factors():
    """
    Load QMJ and BAB monthly factor returns from AQR's public Excel files.
    Returns a DataFrame indexed by Period("M") with columns QMJ and BAB (as decimals).
    Falls back to empty DataFrame on any error.
    """
    import io

    results = {}

    # ── QMJ ──────────────────────────────────────────────────────────────────
    try:
        resp_qmj = pd.read_excel(AQR_QMJ_URL, sheet_name="QMJ Factors", header=18, index_col=0)
        resp_qmj.index = pd.to_datetime(resp_qmj.index, errors="coerce")
        resp_qmj = resp_qmj[resp_qmj.index.notna()]
        resp_qmj.index = resp_qmj.index.to_period("M")
        # AQR stores USA column; fall back to first numeric column
        if "USA" in resp_qmj.columns:
            qmj_series = pd.to_numeric(resp_qmj["USA"], errors="coerce").dropna()
        else:
            qmj_series = pd.to_numeric(resp_qmj.iloc[:, 0], errors="coerce").dropna()
        results["QMJ"] = qmj_series
    except Exception as e:
        st.warning(f"Could not load AQR QMJ data: {e}. QMJ will be unavailable.")

    # ── BAB ──────────────────────────────────────────────────────────────────
    try:
        resp_bab = pd.read_excel(AQR_BAB_URL, sheet_name="BAB Factors", header=18, index_col=0)
        resp_bab.index = pd.to_datetime(resp_bab.index, errors="coerce")
        resp_bab = resp_bab[resp_bab.index.notna()]
        resp_bab.index = resp_bab.index.to_period("M")
        if "USA" in resp_bab.columns:
            bab_series = pd.to_numeric(resp_bab["USA"], errors="coerce").dropna()
        else:
            bab_series = pd.to_numeric(resp_bab.iloc[:, 0], errors="coerce").dropna()
        results["BAB"] = bab_series
    except Exception as e:
        st.warning(f"Could not load AQR BAB data: {e}. BAB will be unavailable.")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # AQR returns are already in decimal form in recent files;
    # if values look like percentages (mean abs > 0.5) divide by 100
    for col in df.columns:
        if df[col].abs().mean() > 0.5:
            df[col] = df[col] / 100.0
    return df


def merge_all_factors(ff_df, aqr_df, selected_factors):
    """
    Merge FF and AQR factor DataFrames, return only the selected columns that exist.
    """
    combined = ff_df.copy()
    for col in AQR_FACTORS:
        if col in selected_factors and col in aqr_df.columns:
            combined[col] = aqr_df[col]
    return combined


def get_live_price(ticker_symbol):
    t = yf.Ticker(ticker_symbol)
    try:
        fi = t.fast_info
        lp = getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None)
        pc = getattr(fi, "previous_close", None) or getattr(fi, "previousClose", None)
        cy = getattr(fi, "currency", "USD") or "USD"
        if lp and pc:
            return float(lp), float(pc), cy
    except Exception:
        pass
    try:
        hist = t.history(period="5d")
        if not hist.empty and len(hist) >= 2:
            return float(hist["Close"].iloc[-1]), float(hist["Close"].iloc[-2]), "USD"
        elif not hist.empty:
            p = float(hist["Close"].iloc[-1])
            return p, p, "USD"
    except Exception:
        pass
    try:
        info = t.info
        lp = info.get("currentPrice") or info.get("regularMarketPrice")
        pc = info.get("previousClose") or info.get("regularMarketPreviousClose")
        cy = info.get("currency", "USD")
        if lp and pc:
            return float(lp), float(pc), cy
    except Exception:
        pass
    raise ValueError("All price-fetch layers failed")


def _call_groq(client, system_msg, user_msg, max_tokens=800):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        max_tokens=max_tokens, temperature=0.2,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def _get_stock_context(ticker):
    ctx = {}
    try:
        t_obj = yf.Ticker(ticker)
        info = t_obj.info
        for k, v in [
            ("current_price", info.get("currentPrice") or info.get("regularMarketPrice")),
            ("market_cap", info.get("marketCap")), ("sector", info.get("sector", "Unknown")),
            ("industry", info.get("industry", "Unknown")), ("pe_ratio", info.get("trailingPE")),
            ("pb_ratio", info.get("priceToBook")), ("ps_ratio", info.get("priceToSalesTrailing12Months")),
            ("revenue_growth", info.get("revenueGrowth")), ("earnings_growth", info.get("earningsGrowth")),
            ("profit_margin", info.get("profitMargins")), ("gross_margin", info.get("grossMargins")),
            ("roe", info.get("returnOnEquity")), ("debt_to_equity", info.get("debtToEquity")),
            ("free_cashflow", info.get("freeCashflow")), ("dividend_yield", info.get("dividendYield")),
            ("beta_1y", info.get("beta")), ("52w_high", info.get("fiftyTwoWeekHigh")),
            ("52w_low", info.get("fiftyTwoWeekLow")), ("short_ratio", info.get("shortRatio")),
            ("short_pct_float", info.get("shortPercentOfFloat")), ("analyst_target", info.get("targetMeanPrice")),
            ("rec_mean", info.get("recommendationMean")), ("num_analysts", info.get("numberOfAnalystOpinions")),
            ("forward_pe", info.get("forwardPE")), ("peg_ratio", info.get("pegRatio")),
            ("enterprise_ev_ebitda", info.get("enterpriseToEbitda")),
            ("held_pct_inst", info.get("heldPercentInstitutions")),
            ("company_name", info.get("longName", ticker)),
        ]:
            ctx[k] = v
        for period, key in [("ytd", "ytd_return"), ("1mo", "return_1m"), ("3mo", "return_3m")]:
            h = t_obj.history(period=period)
            if not h.empty and len(h) >= 2:
                ctx[key] = (h["Close"].iloc[-1] / h["Close"].iloc[0]) - 1
    except Exception:
        pass
    return ctx


def _fmt_ctx(ctx):
    lines = []
    if ctx.get("company_name"):    lines.append(f"Company: {ctx['company_name']}")
    if ctx.get("sector"):          lines.append(f"Sector: {ctx['sector']} | Industry: {ctx.get('industry','?')}")
    if ctx.get("current_price"):   lines.append(f"Price: ${ctx['current_price']:.2f}")
    if ctx.get("market_cap"):      lines.append(f"Market Cap: ${ctx['market_cap']/1e9:.1f}B")
    if ctx.get("ytd_return") is not None: lines.append(f"YTD Return: {ctx['ytd_return']:+.1%}")
    if ctx.get("return_1m") is not None:  lines.append(f"1-Month Return: {ctx['return_1m']:+.1%}")
    if ctx.get("return_3m") is not None:  lines.append(f"3-Month Return: {ctx['return_3m']:+.1%}")
    if ctx.get("52w_high") and ctx.get("52w_low"):
        lines.append(f"52-Week Range: ${ctx['52w_low']:.2f} - ${ctx['52w_high']:.2f}")
    if ctx.get("pe_ratio"):        lines.append(f"P/E (TTM): {ctx['pe_ratio']:.1f}")
    if ctx.get("forward_pe"):      lines.append(f"Forward P/E: {ctx['forward_pe']:.1f}")
    if ctx.get("pb_ratio"):        lines.append(f"P/B: {ctx['pb_ratio']:.2f}")
    if ctx.get("peg_ratio"):       lines.append(f"PEG: {ctx['peg_ratio']:.2f}")
    if ctx.get("enterprise_ev_ebitda"): lines.append(f"EV/EBITDA: {ctx['enterprise_ev_ebitda']:.1f}")
    if ctx.get("revenue_growth") is not None:  lines.append(f"Revenue Growth (YoY): {ctx['revenue_growth']:+.1%}")
    if ctx.get("earnings_growth") is not None: lines.append(f"Earnings Growth (YoY): {ctx['earnings_growth']:+.1%}")
    if ctx.get("profit_margin") is not None:   lines.append(f"Net Margin: {ctx['profit_margin']:.1%}")
    if ctx.get("gross_margin") is not None:    lines.append(f"Gross Margin: {ctx['gross_margin']:.1%}")
    if ctx.get("roe") is not None:             lines.append(f"ROE: {ctx['roe']:.1%}")
    if ctx.get("debt_to_equity") is not None:  lines.append(f"Debt/Equity: {ctx['debt_to_equity']:.2f}")
    if ctx.get("beta_1y"):         lines.append(f"1Y Beta (Yahoo): {ctx['beta_1y']:.2f}")
    if ctx.get("dividend_yield"):  lines.append(f"Dividend Yield: {ctx['dividend_yield']:.2%}")
    if ctx.get("short_pct_float") is not None: lines.append(f"Short % of Float: {ctx['short_pct_float']:.1%}")
    if ctx.get("short_ratio"):     lines.append(f"Short Ratio (days): {ctx['short_ratio']:.1f}")
    if ctx.get("analyst_target"):  lines.append(f"Analyst Mean PT: ${ctx['analyst_target']:.2f} ({ctx.get('num_analysts','?')} analysts)")
    if ctx.get("rec_mean"):
        rec_map = {1:"Strong Buy", 2:"Buy", 3:"Hold", 4:"Underperform", 5:"Sell"}
        lines.append(f"Consensus: {rec_map.get(round(ctx['rec_mean']), str(ctx['rec_mean']))}")
    if ctx.get("held_pct_inst") is not None: lines.append(f"Institutional Ownership: {ctx['held_pct_inst']:.1%}")
    return "\n".join(lines)


def _factor_prompt(ticker, factor_code, factor_name, beta, p_value, significant, description, ctx_str):
    today = date.today().strftime("%B %d, %Y")
    is_aqr = factor_code in AQR_FACTOR_SET
    aqr_note = (
        "\nThis is an AQR factor. For QMJ, discuss quality metrics (profitability, earnings quality, safety, payout). "
        "For BAB, discuss the low-beta/low-volatility anomaly and leverage constraints."
    ) if is_aqr else ""
    system = (
        "You are a senior quantitative analyst at a top-tier hedge fund. "
        "You have been given LIVE fundamental and price data for the stock. Use it. "
        "Your analysis must be SPECIFIC to this exact ticker. "
        "You MUST respond with a single valid JSON object and nothing else."
        "You have to check various news sources and check all the finance related developments and any investments decided."
        "Make use of prnewswire to get the latest quarterly results and 8-k announcements to use more numbers."
        "But dont explicitly keep saying 8-k or pnr. Make heavy use of cnbc to get the company developments."
        + aqr_note
    )
    sig_word = "statistically significant (p<0.05)" if significant else f"not statistically significant (p={p_value:.3f})"
    user = f"""Today is {today}. Analyze the {factor_name} ({factor_code}) loading for {ticker}.

=== LIVE STOCK DATA ===
{ctx_str}

=== REGRESSION RESULT ===
Factor: {factor_name} ({factor_code})
Definition: {description}
Beta: {beta:+.4f}
P-value: {p_value:.4f} ({sig_word})

Return EXACTLY this JSON (no line breaks inside strings):
{{
  "code": "{factor_code}",
  "name": "{factor_name}",
  "beta": {beta},
  "significant": {"true" if significant else "false"},
  "outlook": "<bullish|bearish|neutral|mixed>",
  "what_it_means": "<2-3 sentences using ACTUAL numbers from the data above.>",
  "current_macro_context": "<2-3 sentences on current macro environment relevant to THIS factor.>",
  "recent_stock_news": "<2-3 sentences on the most recent material developments for {ticker}.>",
  "forward_forecast": "<2 sentences: near-term directional call for {ticker} on this factor.>",
  "key_risks": "<1-2 sentences: the single most specific risk that could flip this outlook.>"
}}"""
    return system, user


def _summary_prompt(ticker, alpha_ann, alpha_p, r2, factor_summaries, ctx_str):
    today = date.today().strftime("%B %d, %Y")
    system = (
        "You are a senior portfolio strategist. "
        "Use specific numbers from the live data. "
        "Respond with a single valid JSON object only. No markdown."
    )
    sig_word = "statistically significant" if alpha_p < 0.05 else "not statistically significant"
    lines = "\n".join(f"- {f['name']}: beta={f['beta']:+.4f}, outlook={f.get('outlook','?')}" for f in factor_summaries)
    user = f"""Today: {today}. Stock: {ticker}
Annualized alpha: {alpha_ann:+.2%} (p={alpha_p:.4f}, {sig_word})
R2: {r2:.4f}
Significant factor loadings:
{lines if lines else "None"}

=== LIVE STOCK DATA ===
{ctx_str}

Return EXACTLY this JSON:
{{
  "alpha_analysis": "<Is alpha={alpha_ann:+.2%} meaningful for {ticker}? What explains it? What threatens persistence?>",
  "positioning_context": "<Analyst consensus, institutional ownership, short interest, factor profile alignment.>",
  "portfolio_verdict": "<Risk-adjusted verdict as of {today}. Best-case and worst-case. One-line bottom line: buy/hold/avoid.>"
}}"""
    return system, user


def get_ai_insight(ticker, model_result, available, alpha_ann, alpha_p, r2, n, start_date, end_date):
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        return None, "No API key found. Add GROQ_API_KEY to Streamlit secrets."
    ctx = _get_stock_context(ticker)
    ctx_str = _fmt_ctx(ctx)
    try:
        client = Groq(api_key=api_key)
        sig_factors = [f for f in available if float(model_result.pvalues[f]) < 0.05]
        if not sig_factors:
            sys_msg, usr_msg = _summary_prompt(ticker, alpha_ann, alpha_p, r2, [], ctx_str)
            try:
                summary = _call_groq(client, sys_msg, usr_msg, max_tokens=700)
            except Exception as e:
                summary = {"alpha_analysis": f"Summary unavailable: {e}", "positioning_context": "", "portfolio_verdict": ""}
            return {"factors": [], **summary}, None
        factors_out = []
        errors = []
        for f in sig_factors:
            beta = float(model_result.params[f])
            pval = float(model_result.pvalues[f])
            sys_msg, usr_msg = _factor_prompt(ticker, f, FACTOR_NAMES.get(f, f), beta, pval, True, FACTOR_DESCRIPTIONS.get(f, f), ctx_str=ctx_str)
            try:
                result = _call_groq(client, sys_msg, usr_msg, max_tokens=900)
                result["beta"] = beta
                result["significant"] = True
                factors_out.append(result)
            except Exception as e:
                factors_out.append({"code": f, "name": FACTOR_NAMES.get(f, f), "beta": beta, "significant": True,
                                    "outlook": "neutral", "what_it_means": "Analysis unavailable.",
                                    "current_macro_context": f"Error: {e}", "recent_stock_news": "",
                                    "forward_forecast": "", "key_risks": ""})
                errors.append(f"{f}: {e}")
        sys_msg, usr_msg = _summary_prompt(ticker, alpha_ann, alpha_p, r2, factors_out, ctx_str)
        try:
            summary = _call_groq(client, sys_msg, usr_msg, max_tokens=700)
        except Exception as e:
            summary = {"alpha_analysis": f"Summary unavailable: {e}", "positioning_context": "", "portfolio_verdict": ""}
            errors.append(f"summary: {e}")
        return {"factors": factors_out, **summary}, ("; ".join(errors) if errors else None)
    except Exception as e:
        return None, str(e)


def render_ai_insight(ticker, insight_data):
    factors = insight_data.get("factors", [])
    alpha_analysis = insight_data.get("alpha_analysis", "")
    positioning_context = insight_data.get("positioning_context", "")
    portfolio_verdict = insight_data.get("portfolio_verdict", "")

    def bold(text):
        return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', str(text))

    today_str = date.today().strftime("%B %d, %Y")
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;letter-spacing:1px;'
        f'text-transform:uppercase;color:#1D9E75;margin-bottom:16px;">'
        f'Factor Intelligence · {ticker} · {today_str} · Significant factors only</div>',
        unsafe_allow_html=True)

    if not factors:
        st.markdown('<div class="info-box">No statistically significant factors (p&lt;0.05). See summary below.</div>', unsafe_allow_html=True)
    else:
        for fac in factors:
            code = fac.get("code", ""); name = fac.get("name", code); beta = fac.get("beta", 0)
            outlook = fac.get("outlook", "neutral").lower()
            card_cls = "positive" if beta > 0 else "negative"
            beta_cls = "pos" if beta > 0 else "neg"
            outlook_cls  = {"bullish":"bullish","bearish":"bearish","neutral":"neutral","mixed":"mixed"}.get(outlook,"neutral")
            outlook_icon = {"bullish":"↑ BULLISH","bearish":"↓ BEARISH","neutral":"→ NEUTRAL","mixed":"⇅ MIXED"}.get(outlook,"→ NEUTRAL")
            what = bold(fac.get("what_it_means", "")); macro = bold(fac.get("current_macro_context", ""))
            news = bold(fac.get("recent_stock_news", "")); forecast = bold(fac.get("forward_forecast", ""))
            risks = bold(fac.get("key_risks", ""))
            # AQR badge for QMJ/BAB
            aqr_tag = '<span class="aqr-badge">AQR</span>' if code in AQR_FACTOR_SET else ""
            st.markdown(f"""
            <div class="factor-insight-card {card_cls}">
              <div class="fi-header">
                <div class="fi-name">{name}{aqr_tag}</div>
                <div class="fi-beta {beta_cls}">{beta:+.4f}</div>
                <div class="fi-sig-label">SIGNIFICANT</div>
                <div class="fi-outlook {outlook_cls}">{outlook_icon}</div>
              </div>
              <div class="fi-body">
                <div style="margin-bottom:8px;"><span style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#1D9E75;">WHAT IT MEANS FOR {ticker}</span><div style="margin-top:4px;">{what}</div></div>
                <div style="margin-bottom:8px;"><span style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#1D9E75;">CURRENT MACRO CONTEXT</span><div style="margin-top:4px;">{macro}</div></div>
                {"" if not news else f'<div style="margin-bottom:8px;"><span style="font-family:JetBrains Mono,monospace;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#1D9E75;">RECENT NEWS & CATALYSTS</span><div style="margin-top:4px;">{news}</div></div>'}
                <div style="margin-bottom:8px;"><span style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#1D9E75;">FORWARD FORECAST</span><div style="margin-top:4px;">{forecast}</div></div>
                <div class="fi-news"><div class="fi-news-label">⚠ KEY RISK</div>{risks}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    if alpha_analysis:
        st.markdown(f'<div class="ai-summary-box"><div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#1D9E75;margin-bottom:8px;">ALPHA PERSISTENCE ANALYSIS</div>{bold(alpha_analysis)}</div>', unsafe_allow_html=True)
    if positioning_context:
        st.markdown(f'<div class="ai-summary-box" style="border-color:rgba(52,211,153,0.2);color:#6ee7b7;background:rgba(52,211,153,0.05);"><div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#1D9E75;margin-bottom:8px;">POSITIONING & SENTIMENT</div>{bold(positioning_context)}</div>', unsafe_allow_html=True)
    if portfolio_verdict:
        st.markdown(f'<div class="ai-summary-box" style="border-color:rgba(167,139,250,0.2);color:#c4b5fd;background:rgba(167,139,250,0.05);"><div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#7c3aed;margin-bottom:8px;">PORTFOLIO VERDICT</div>{bold(portfolio_verdict)}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Portfolio Attribution
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_monthly_returns(ticker_sym, start_str, end_str):
    try:
        raw = yf.download(ticker_sym, start=start_str + "-01", end=end_str + "-28",
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None, f"No data for {ticker_sym}"
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        monthly = close.resample("ME").last()
        returns = monthly.pct_change().dropna()
        returns.index = returns.index.to_period("M")
        return returns, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def run_single_regression(ticker_sym, start_str, end_str, hac_lags, ff_key, selected_factors_key):
    try:
        ff_raw  = load_factors()
        aqr_raw = load_aqr_factors()
        ff_raw  = merge_all_factors(ff_raw, aqr_raw, list(selected_factors_key))

        ff = ff_raw.loc[start_str:end_str].copy()
        selected  = list(selected_factors_key)
        available = [c for c in selected if c in ff.columns]
        has_rf    = "RF" in ff.columns
        ff[available] = ff[available].astype(float)
        if has_rf:
            ff["RF"] = ff["RF"].astype(float)

        returns, err = fetch_monthly_returns(ticker_sym, start_str, end_str)
        if returns is None:
            return None, err

        data = pd.DataFrame({"Stock": returns}).join(ff[available + (["RF"] if has_rf else [])], how="inner")
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 12:
            return None, f"Too few obs for {ticker_sym} ({len(data)})"
        data["Y"] = data["Stock"] - data["RF"] if has_rf else data["Stock"]

        X = sm.add_constant(data[available])
        model = sm.OLS(data["Y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        alpha_ann = (1 + model.params["const"]) ** 12 - 1

        return {
            "ticker":    ticker_sym,
            "params":    dict(model.params),
            "pvalues":   dict(model.pvalues),
            "r2":        model.rsquared,
            "alpha_ann": alpha_ann,
            "available": available,
            "nobs":      int(model.nobs),
            "returns":   returns,
            "has_rf":    has_rf,
            "ff_rf":     ff["RF"].copy() if has_rf else None,
        }, None
    except Exception as e:
        return None, str(e)


def build_true_portfolio_model(port_results, weights, available_factors, start_str, end_str, hac_lags):
    try:
        ff_raw  = load_factors()
        aqr_raw = load_aqr_factors()
        ff_raw  = merge_all_factors(ff_raw, aqr_raw, available_factors)

        ff = ff_raw.loc[start_str:end_str].copy()
        has_rf = "RF" in ff.columns
        ff[available_factors] = ff[available_factors].astype(float)
        if has_rf:
            ff["RF"] = ff["RF"].astype(float)

        all_returns = {}
        for tkr, res in port_results.items():
            all_returns[tkr] = res["returns"]

        port_ret = pd.Series(dtype=float)
        for tkr, w in weights.items():
            if tkr in all_returns:
                s = all_returns[tkr] * w
                port_ret = s if port_ret.empty else port_ret.add(s, fill_value=0)

        data = pd.DataFrame({"Port": port_ret}).join(
            ff[available_factors + (["RF"] if has_rf else [])], how="inner"
        )
        data["Y"] = data["Port"] - data["RF"] if has_rf else data["Port"]

        X = sm.add_constant(data[available_factors])
        model = sm.OLS(data["Y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

        return {
            "alpha_ann":  (1 + model.params["const"]) ** 12 - 1,
            "alpha_p":    model.pvalues["const"],
            "r2":         model.rsquared,
            "mkt_beta":   model.params.get("Mkt-RF", 0.0),
            "betas":      dict(model.params),
            "nobs":       int(model.nobs),
        }
    except Exception as e:
        return None


def render_portfolio_attribution(port_results, weights, available_factors, true_port, start_str, end_str, hac_lags):
    tickers  = list(port_results.keys())
    n_stocks = len(tickers)

    if true_port:
        port_alpha_ann = true_port["alpha_ann"]
        port_alpha_p   = true_port["alpha_p"]
        port_r2        = true_port["r2"]
        port_mkt_beta  = true_port["mkt_beta"]
        port_betas     = true_port["betas"]
        method_note    = "Single model fitted on blended portfolio returns"
    else:
        port_betas = {f: sum(weights[t] * port_results[t]["params"].get(f, 0.0) for t in tickers) for f in ["const"] + available_factors}
        port_alpha_ann = sum(weights[t] * port_results[t]["alpha_ann"] for t in tickers)
        port_alpha_p   = None
        port_r2        = np.mean([port_results[t]["r2"] for t in tickers])
        port_mkt_beta  = port_betas.get("Mkt-RF", 0.0)
        method_note    = "Weighted average of individual regressions"

    alpha_color = "#34d399" if port_alpha_ann > 0 else "#f87171"
    alpha_p_str = f"p={port_alpha_p:.4f}" if port_alpha_p is not None else "n/a"

    st.markdown(f"""
    <div class="port-summary-card">
      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#1D9E75;margin-bottom:14px;">
        Portfolio Factor Attribution · {n_stocks} Holdings
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:16px 24px;align-items:baseline;">
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#5DCAA5;margin-bottom:2px;">ANNUALIZED ALPHA</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:{alpha_color};">{port_alpha_ann:+.2%}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#9FE1CB;margin-top:2px;">{alpha_p_str}</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#5DCAA5;margin-bottom:2px;">PORTFOLIO R²</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#60a5fa;">{port_r2:.4f}</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#5DCAA5;margin-bottom:2px;">MARKET BETA</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#a78bfa;">{port_mkt_beta:+.3f}</div>
        </div>
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#1D9E75;margin-top:10px;">{method_note}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#5DCAA5;margin-bottom:10px;">Portfolio Factor Exposures</div>', unsafe_allow_html=True)
    grid_html = '<div class="port-attribution-grid">'
    for f in available_factors:
        b    = port_betas.get(f, 0)
        bcls = "pos" if b >= 0 else "neg"
        aqr_tag = ' <span class="aqr-badge">AQR</span>' if f in AQR_FACTOR_SET else ""
        grid_html += f'<div class="port-attr-cell"><div class="port-attr-factor">{FACTOR_NAMES.get(f, f)}{aqr_tag}</div><div class="port-attr-beta {bcls}">{b:+.4f}</div></div>'
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#5DCAA5;margin:20px 0 10px 0;">Factor Loadings by Holding</div>', unsafe_allow_html=True)

    all_betas_flat = [abs(port_results[t]["params"].get(f, 0)) for t in tickers for f in available_factors]
    max_abs = max(all_betas_flat) if all_betas_flat else 1
    grid_cols = f"80px 60px " + " ".join(["1fr"] * len(available_factors)) + " 80px 60px"

    rows_html = f'<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;"><div style="display:grid;grid-template-columns:{grid_cols};gap:6px;min-width:600px;">'
    rows_html += '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:1px;color:#1D9E75;text-transform:uppercase;padding:4px 0;">TICKER</div>'
    rows_html += '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:1px;color:#1D9E75;text-transform:uppercase;padding:4px 0;">WT</div>'
    for f in available_factors:
        fc = FACTOR_COLORS.get(f, "#9FE1CB")
        aqr_tag = ' <span class="aqr-badge">AQR</span>' if f in AQR_FACTOR_SET else ""
        rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:1px;color:{fc};text-transform:uppercase;padding:4px 0;">{FACTOR_NAMES.get(f,f)}{aqr_tag}</div>'
    rows_html += '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:1px;color:#a78bfa;text-transform:uppercase;padding:4px 0;">ANN α</div>'
    rows_html += '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:1px;color:#60a5fa;text-transform:uppercase;padding:4px 0;">R²</div>'

    for tkr in tickers:
        res     = port_results[tkr]
        w       = weights[tkr]
        alpha_c = "#34d399" if res["alpha_ann"] > 0 else "#f87171"
        rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;font-weight:700;color:#e2e8f0;padding:8px 0;border-top:1px solid rgba(29,158,117,0.15);">{tkr}</div>'
        rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#60a5fa;padding:8px 0;border-top:1px solid rgba(29,158,117,0.15);">{w:.0%}</div>'
        for f in available_factors:
            b   = res["params"].get(f, 0)
            bc  = "#34d399" if b >= 0 else "#f87171"
            pct = min(abs(b) / max_abs * 100, 100) if max_abs > 0 else 0
            rows_html += f'<div style="padding:8px 0;border-top:1px solid rgba(29,158,117,0.15);"><div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{bc};">{b:+.3f}</div><div style="height:3px;background:rgba(29,158,117,0.1);border-radius:2px;margin-top:3px;"><div style="width:{pct:.1f}%;height:100%;background:{bc};border-radius:2px;"></div></div></div>'
        rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{alpha_c};padding:8px 0;border-top:1px solid rgba(29,158,117,0.15);">{res["alpha_ann"]:+.2%}</div>'
        rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#60a5fa;padding:8px 0;border-top:1px solid rgba(29,158,117,0.15);">{res["r2"]:.3f}</div>'

    alpha_c = "#34d399" if port_alpha_ann > 0 else "#f87171"
    rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;font-weight:700;color:#a78bfa;padding:8px 0;border-top:2px solid rgba(167,139,250,0.3);">PORTFOLIO</div>'
    rows_html += f'<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:#a78bfa;padding:8px 0;border-top:2px solid rgba(167,139,250,0.3);">100%</div>'
    for f in available_factors:
        b   = port_betas.get(f, 0)
        bc  = "#34d399" if b >= 0 else "#f87171"
        pct = min(abs(b) / max_abs * 100, 100) if max_abs > 0 else 0
        rows_html += f'<div style="padding:8px 0;border-top:2px solid rgba(167,139,250,0.3);"><div style="font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:{bc};">{b:+.3f}</div><div style="height:3px;background:rgba(29,158,117,0.1);border-radius:2px;margin-top:3px;"><div style="width:{pct:.1f}%;height:100%;background:{bc};border-radius:2px;"></div></div></div>'
    rows_html += f'<div style="font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:{alpha_c};padding:8px 0;border-top:2px solid rgba(167,139,250,0.3);">{port_alpha_ann:+.2%}</div>'
    rows_html += f'<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:#60a5fa;padding:8px 0;border-top:2px solid rgba(167,139,250,0.3);">{port_r2:.3f}</div>'
    rows_html += '</div></div>'
    st.markdown(f'<div style="background:rgba(29,158,117,0.04);border:1px solid rgba(29,158,117,0.15);border-radius:10px;padding:16px 18px;">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#5DCAA5;margin:20px 0 10px 0;">Factor Concentration Risk</div>', unsafe_allow_html=True)
    risk_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;">'
    for f in available_factors:
        vals = [port_results[t]["params"].get(f, 0) for t in tickers]
        disp = np.std(vals)
        rc   = "#f87171" if disp > 0.5 else "#fbbf24" if disp > 0.2 else "#34d399"
        rl   = "HIGH" if disp > 0.5 else "MODERATE" if disp > 0.2 else "LOW"
        aqr_tag = ' <span class="aqr-badge">AQR</span>' if f in AQR_FACTOR_SET else ""
        risk_html += f'<div style="background:rgba(29,158,117,0.06);border:1px solid rgba(29,158,117,0.15);border-radius:8px;padding:12px 14px;min-width:100px;flex:1;"><div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#5DCAA5;margin-bottom:4px;text-transform:uppercase;">{FACTOR_NAMES.get(f,f)}{aqr_tag}</div><div style="font-family:\'JetBrains Mono\',monospace;font-size:13px;font-weight:600;color:{rc};">{rl}</div><div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#9FE1CB;margin-top:2px;">σ = {disp:.3f}</div></div>'
    risk_html += '</div>'
    st.markdown(risk_html, unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#5DCAA5;margin:20px 0 10px 0;">Export</div>', unsafe_allow_html=True)
    rows = []
    for tkr in tickers:
        res = port_results[tkr]
        row = {"Ticker": tkr, "Weight": weights[tkr], "Ann_Alpha_Individual": res["alpha_ann"], "R2_Individual": res["r2"]}
        for f in available_factors:
            row[FACTOR_NAMES.get(f, f)] = res["params"].get(f, 0)
        rows.append(row)
    port_row = {"Ticker": "PORTFOLIO (TRUE MODEL)", "Weight": 1.0,
                "Ann_Alpha_Individual": port_alpha_ann, "R2_Individual": port_r2}
    for f in available_factors:
        port_row[FACTOR_NAMES.get(f, f)] = port_betas.get(f, 0)
    rows.append(port_row)
    st.download_button("⬇  Download Attribution CSV", pd.DataFrame(rows).to_csv(index=False),
                       file_name="portfolio_factor_attribution.csv", mime="text/csv")


# ─────────────────────────────────────────────
#  Markov Regime-Switching EGARCH
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_daily_returns(ticker_sym, start_str, end_str):
    try:
        raw = yf.download(ticker_sym, start=start_str, end=end_str, auto_adjust=True, progress=False)
        if raw.empty:
            return None, f"No data for {ticker_sym}"
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        rets = 100 * close.pct_change().dropna()  # in percent for arch package numerical stability
        rets.name = "ret"
        return rets, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def run_markov_egarch(ticker_sym, start_str, end_str, n_regimes, p, o, q, dist):
    """
    Two-step approach:
      1) Fit a Markov-switching mean/variance model (statsmodels MarkovRegression)
         on daily returns to identify volatility regimes & transition dynamics.
      2) Fit an EGARCH(p,o,q) model (arch package) on the full sample for
         conditional-volatility forecasting, plus separate EGARCH fits on the
         data belonging to each identified regime for regime-conditional vol.
    """
    try:
        rets, err = fetch_daily_returns(ticker_sym, start_str, end_str)
        if rets is None:
            return None, err
        if len(rets) < 250:
            return None, f"Too few observations for {ticker_sym} ({len(rets)})"

        rets_vals = rets.values.astype(float)

        # 1) Markov-switching mean/variance model to identify regimes
        ms_model = MarkovRegression(rets_vals, k_regimes=n_regimes, trend="c", switching_variance=True)
        ms_res = ms_model.fit(disp=False, search_reps=20)

        smoothed_probs = ms_res.smoothed_marginal_probabilities  # shape (n, k_regimes)
        regime_assign = np.argmax(smoothed_probs, axis=1)

        # Order regimes by variance (low -> high)
        regime_var_raw = {k: ms_res.params[f"sigma2[{k}]"] for k in range(n_regimes)}
        order = sorted(regime_var_raw, key=regime_var_raw.get)
        regime_map = {old: new for new, old in enumerate(order)}
        regime_assign_ordered = np.vectorize(regime_map.get)(regime_assign)

        # Current regime (last observation)
        current_regime = int(regime_assign_ordered[-1])
        current_probs_raw = smoothed_probs[-1]
        current_probs_ordered = [float(current_probs_raw[order[i]]) for i in range(n_regimes)]

        # Transition matrix (regime_transition shape: k x k x 1, [to, from])
        raw_trans = ms_res.regime_transition[:, :, 0]
        trans_ordered = np.zeros((n_regimes, n_regimes))
        for i_to in range(n_regimes):
            for j_from in range(n_regimes):
                trans_ordered[regime_map[i_to], regime_map[j_from]] = raw_trans[i_to, j_from]

        # Expected duration (in days) of each regime = 1 / (1 - p_ii)
        expected_duration = {}
        for k in range(n_regimes):
            p_stay = trans_ordered[k, k]
            expected_duration[k] = 1.0 / (1.0 - p_stay) if p_stay < 0.999999 else float("inf")

        # 2) Fit overall EGARCH on full series
        am_full = arch_model(rets, mean="Constant", vol="EGARCH", p=p, o=o, q=q, dist=dist)
        res_full = am_full.fit(disp="off")

        cond_vol_full = res_full.conditional_volatility
        ann_vol_full = float(cond_vol_full.iloc[-1] * np.sqrt(252) / 100)  # annualized decimal

        # Forecast next 5 steps vol
        fc = res_full.forecast(horizon=5, reindex=False)
        fc_vol_daily = np.sqrt(fc.variance.values[-1]) / 100  # daily decimal vol forecasts

        # 3) Fit EGARCH separately within each regime's data subset
        regime_egarch = {}
        for k in range(n_regimes):
            mask = regime_assign_ordered == k
            sub_rets = rets[mask]
            if len(sub_rets) < 60:
                regime_egarch[k] = None
                continue
            try:
                am_k = arch_model(sub_rets.reset_index(drop=True), mean="Constant", vol="EGARCH", p=p, o=o, q=q, dist=dist)
                res_k = am_k.fit(disp="off")
                ann_vol_k = float(res_k.conditional_volatility.iloc[-1] * np.sqrt(252) / 100)
                regime_egarch[k] = {
                    "params": dict(res_k.params),
                    "ann_vol_current": ann_vol_k,
                    "loglik": res_k.loglikelihood,
                    "aic": res_k.aic,
                    "n_obs": int(len(sub_rets)),
                    "mean_ret_ann": float(sub_rets.mean() * 252 / 100),
                    "vol_ann_avg": float(sub_rets.std() * np.sqrt(252) / 100),
                }
            except Exception:
                regime_egarch[k] = None

        # leverage / asymmetry param from full model (gamma terms)
        gamma_params = {pn: float(v) for pn, v in res_full.params.items() if "gamma" in pn.lower()}

        regime_means_ann = [float(ms_res.params[f"const[{order[k]}]"] * 252 / 100) for k in range(n_regimes)]
        regime_var_daily = [float(regime_var_raw[order[k]]) for k in range(n_regimes)]

        return {
            "ticker": ticker_sym,
            "n_regimes": n_regimes,
            "current_regime": current_regime,
            "current_probs": current_probs_ordered,
            "transition_matrix": trans_ordered,
            "expected_duration": expected_duration,
            "regime_means_ann": regime_means_ann,
            "regime_var_daily": regime_var_daily,
            "ann_vol_full": ann_vol_full,
            "fc_vol_daily": fc_vol_daily,
            "egarch_params_full": dict(res_full.params),
            "gamma_params": gamma_params,
            "regime_egarch": regime_egarch,
            "ms_loglik": float(ms_res.llf),
            "ms_aic": float(ms_res.aic),
            "full_loglik": float(res_full.loglikelihood),
            "full_aic": float(res_full.aic),
            "rets": rets,
            "regime_assign": regime_assign_ordered,
            "smoothed_probs_ordered": smoothed_probs[:, order],
            "n_obs": int(len(rets)),
            "p": p, "o": o, "q": q, "dist": dist,
        }, None
    except Exception as e:
        return None, str(e)


def render_regime_egarch(result):
    tkr = result["ticker"]
    nk  = result["n_regimes"]
    cur = result["current_regime"]
    probs = result["current_probs"]

    st.markdown(f'<div class="section-title">{tkr} · Markov Regime-Switching EGARCH</div>', unsafe_allow_html=True)

    cur_color = REGIME_COLORS.get(cur, "#9FE1CB")
    cur_label = regime_label(cur, nk)
    cur_css   = regime_css(cur, nk)

    fc_ann = float(np.mean(result["fc_vol_daily"]) * np.sqrt(252))

    st.markdown(f"""
    <div class="port-summary-card">
      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#1D9E75;margin-bottom:14px;">
        Current Regime · {tkr}
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:16px 24px;align-items:baseline;">
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#5DCAA5;margin-bottom:4px;">REGIME</div>
          <div><span class="regime-badge {cur_css}" style="font-size:14px;padding:4px 10px;">{cur_label}</span></div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#9FE1CB;margin-top:6px;">P = {probs[cur]:.1%}</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#5DCAA5;margin-bottom:2px;">CURRENT ANN. VOL (FULL EGARCH)</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#60a5fa;">{result['ann_vol_full']:.1%}</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#5DCAA5;margin-bottom:2px;">5-DAY VOL FORECAST (ANN, AVG)</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#a78bfa;">{fc_ann:.1%}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Regime probability bars
    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#5DCAA5;margin:16px 0 8px 0;">Regime Probabilities (Smoothed, Current)</div>', unsafe_allow_html=True)
    bars_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;">'
    for k in range(nk):
        c = REGIME_COLORS.get(k if nk == 3 else (0 if k == 0 else 2), REGIME_COLORS.get(k, "#9FE1CB"))
        lbl = regime_label(k, nk)
        bars_html += f'''<div style="background:rgba(29,158,117,0.06);border:1px solid rgba(29,158,117,0.15);border-radius:10px;padding:12px 14px;min-width:140px;flex:1;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:{c};margin-bottom:4px;">{lbl}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:600;color:#e2e8f0;">{probs[k]:.1%}</div>
          <div style="height:4px;background:rgba(29,158,117,0.1);border-radius:2px;margin-top:6px;"><div style="width:{probs[k]*100:.1f}%;height:100%;background:{c};border-radius:2px;"></div></div>
        </div>'''
    bars_html += '</div>'
    st.markdown(bars_html, unsafe_allow_html=True)

    # Regime stats table
    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#5DCAA5;margin:16px 0 8px 0;">Regime Characteristics</div>', unsafe_allow_html=True)
    st.markdown('<div class="factor-row header"><div>REGIME</div><div>ANN. RETURN</div><div>ANN. VOL (RAW)</div><div>EGARCH ANN. VOL</div><div>EXP. DURATION</div><div>N OBS</div></div>', unsafe_allow_html=True)
    for k in range(nk):
        c = REGIME_COLORS.get(k if nk == 3 else (0 if k == 0 else 2), "#9FE1CB")
        lbl = regime_label(k, nk)
        ret_ann = result["regime_means_ann"][k]
        eg = result["regime_egarch"].get(k)
        eg_vol = f"{eg['ann_vol_current']:.1%}" if eg else "n/a"
        n_obs = eg['n_obs'] if eg else "—"
        dur = result["expected_duration"][k]
        dur_str = f"{dur:.1f} days" if np.isfinite(dur) else "∞"
        raw_vol = float(np.sqrt(result["regime_var_daily"][k]) * np.sqrt(252) / 100)
        ret_color = "#34d399" if ret_ann > 0 else "#f87171"
        st.markdown(f'''<div class="factor-row" style="border-left:2px solid {c};">
          <div style="color:#e2e8f0;font-weight:500">{lbl}</div>
          <div style="color:{ret_color};font-weight:600">{ret_ann:+.1%}</div>
          <div style="color:#9FE1CB">{raw_vol:.1%}</div>
          <div style="color:#60a5fa;font-weight:600">{eg_vol}</div>
          <div style="color:#9FE1CB">{dur_str}</div>
          <div style="color:#9FE1CB">{n_obs}</div>
        </div>''', unsafe_allow_html=True)

    # Transition matrix
    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#5DCAA5;margin:16px 0 8px 0;">Transition Matrix (Daily, row → column)</div>', unsafe_allow_html=True)
    tm = result["transition_matrix"]
    grid_cols = " ".join(["110px"] * (nk + 1))
    tm_html = f'<div style="display:grid;grid-template-columns:{grid_cols};gap:6px;max-width:{(nk+1)*120}px;overflow-x:auto;">'
    tm_html += '<div></div>'
    for k in range(nk):
        c = REGIME_COLORS.get(k if nk == 3 else (0 if k == 0 else 2), "#9FE1CB")
        tm_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:{c};text-align:center;text-transform:uppercase;">→ {regime_label(k, nk).split()[0][:4]}</div>'
    for j in range(nk):
        c_row = REGIME_COLORS.get(j if nk == 3 else (0 if j == 0 else 2), "#9FE1CB")
        tm_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:{c_row};text-transform:uppercase;">{regime_label(j, nk).split()[0][:4]} →</div>'
        for k in range(nk):
            v = tm[k, j]
            shade = "#34d399" if v > 0.9 else "#9FE1CB"
            tm_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:13px;color:{shade};text-align:center;background:rgba(29,158,117,0.06);border:1px solid rgba(29,158,117,0.12);border-radius:6px;padding:6px;">{v:.3f}</div>'
    tm_html += '</div>'
    st.markdown(tm_html, unsafe_allow_html=True)

    # Full-sample EGARCH params
    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#5DCAA5;margin:16px 0 8px 0;">Full-Sample EGARCH Parameters</div>', unsafe_allow_html=True)
    p_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;">'
    for pname, pval in result["egarch_params_full"].items():
        p_html += f'''<div style="background:rgba(29,158,117,0.06);border:1px solid rgba(29,158,117,0.15);border-radius:8px;padding:10px 12px;min-width:90px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#5DCAA5;">{pname}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;color:#e2e8f0;">{pval:.4f}</div>
        </div>'''
    p_html += '</div>'
    st.markdown(p_html, unsafe_allow_html=True)
    if result["gamma_params"]:
        leverage_note = "Negative γ → negative shocks raise volatility more than positive shocks of the same size (the classic leverage effect)."
        st.markdown(f'<div class="info-box" style="margin-top:8px;">Asymmetry (γ) terms: {", ".join(f"{k}={v:+.4f}" for k, v in result["gamma_params"].items())} — {leverage_note}</div>', unsafe_allow_html=True)

    # Regime timeline chart — probability of being in highest-vol regime
    smoothed = result["smoothed_probs_ordered"]
    high_idx = nk - 1
    series = smoothed[:, high_idx]
    if len(series) > 2000:
        step = len(series) // 2000
        series_plot = series[::step]
    else:
        series_plot = series

    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#5DCAA5;margin:16px 0 8px 0;">Regime History — P(Highest-Vol Regime)</div>', unsafe_allow_html=True)
    W, H = 800, 140
    n_pts = len(series_plot)
    pts = " ".join(f"{int(i/(n_pts-1)*W) if n_pts>1 else W//2},{int(H-(v*H))}" for i, v in enumerate(series_plot))
    fill_pts = f"0,{H} " + pts + f" {W},{H}"
    timeline_html = (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:rgba(29,158,117,0.05);border:1px solid rgba(29,158,117,0.15);border-radius:10px;width:100%;margin-bottom:8px;">'
        f'<polygon points="{fill_pts}" fill="rgba(248,113,113,0.12)" stroke="none"/>'
        f'<polyline points="{pts}" fill="none" stroke="#f87171" stroke-width="1.5"/>'
        f'<line x1="0" y1="{int(H*0.5)}" x2="{W}" y2="{int(H*0.5)}" stroke="rgba(29,158,117,0.25)" stroke-width="1" stroke-dasharray="4,4"/>'
        f'<text x="6" y="14" fill="#f87171" font-family="JetBrains Mono,monospace" font-size="10">P(High Vol) = 1.0</text>'
        f'<text x="6" y="{H-6}" fill="#5DCAA5" font-family="JetBrains Mono,monospace" font-size="10">P(High Vol) = 0.0</text>'
        f'</svg>'
    )
    st.markdown(timeline_html, unsafe_allow_html=True)

    # Model fit comparison
    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#5DCAA5;margin:16px 0 8px 0;">Model Fit</div>', unsafe_allow_html=True)
    fit_html = (
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;">'
        f'<div class="diag-item" style="min-width:140px;"><div class="diag-name">Markov-Switching LogLik</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{result["ms_loglik"]:.2f}</div></div>'
        f'<div class="diag-item" style="min-width:140px;"><div class="diag-name">Markov-Switching AIC</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{result["ms_aic"]:.2f}</div></div>'
        f'<div class="diag-item" style="min-width:140px;"><div class="diag-name">EGARCH LogLik</div><div class="diag-val" style="color:#a78bfa;font-size:16px;">{result["full_loglik"]:.2f}</div></div>'
        f'<div class="diag-item" style="min-width:140px;"><div class="diag-name">EGARCH AIC</div><div class="diag-val" style="color:#a78bfa;font-size:16px;">{result["full_aic"]:.2f}</div></div>'
        f'<div class="diag-item" style="min-width:140px;"><div class="diag-name">N Observations</div><div class="diag-val" style="color:#9FE1CB;font-size:16px;">{result["n_obs"]}</div></div>'
        f'<div class="diag-item" style="min-width:140px;"><div class="diag-name">EGARCH Spec</div><div class="diag-val" style="color:#9FE1CB;font-size:16px;">({result["p"]},{result["o"]},{result["q"]}) {result["dist"]}</div></div>'
        f'</div>'
    )
    st.markdown(fit_html, unsafe_allow_html=True)

    # Export
    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#5DCAA5;margin:16px 0 8px 0;">Export</div>', unsafe_allow_html=True)
    export_rows = []
    for k in range(nk):
        eg = result["regime_egarch"].get(k)
        export_rows.append({
            "Regime": regime_label(k, nk),
            "Current_Prob": probs[k],
            "Ann_Return": result["regime_means_ann"][k],
            "Raw_Ann_Vol": float(np.sqrt(result["regime_var_daily"][k]) * np.sqrt(252) / 100),
            "EGARCH_Ann_Vol": eg["ann_vol_current"] if eg else np.nan,
            "Expected_Duration_Days": result["expected_duration"][k],
            "N_Obs": eg["n_obs"] if eg else 0,
        })
    st.download_button(
        f"⬇  Download {tkr} Regime/EGARCH CSV",
        pd.DataFrame(export_rows).to_csv(index=False),
        file_name=f"{tkr}_markov_egarch.csv", mime="text/csv",
        key=f"dl_regime_{tkr}"
    )


def render_portfolio_regime_summary(regime_results, weights):
    """Aggregate cross-holding regime view for the portfolio."""
    tickers = list(regime_results.keys())
    n_regimes = regime_results[tickers[0]]["n_regimes"]

    st.markdown('<div class="section-title">Portfolio Regime Overview</div>', unsafe_allow_html=True)

    # Weighted average current annualized vol & blended high-vol probability
    w_total = sum(weights[t] for t in tickers if t in weights)
    weighted_vol = sum(weights.get(t, 0) * regime_results[t]["ann_vol_full"] for t in tickers) / (w_total or 1)
    weighted_high_prob = sum(
        weights.get(t, 0) * regime_results[t]["current_probs"][n_regimes - 1] for t in tickers
    ) / (w_total or 1)

    n_in_high = sum(1 for t in tickers if regime_results[t]["current_regime"] == n_regimes - 1)

    st.markdown(f"""
    <div class="port-summary-card">
      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#1D9E75;margin-bottom:14px;">
        Portfolio Volatility Regime Snapshot · {len(tickers)} Holdings
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:16px 24px;align-items:baseline;">
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#5DCAA5;margin-bottom:2px;">WEIGHTED ANN. VOL (EGARCH)</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#60a5fa;">{weighted_vol:.1%}</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#5DCAA5;margin-bottom:2px;">WEIGHTED P(HIGH-VOL REGIME)</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#f87171;">{weighted_high_prob:.1%}</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#5DCAA5;margin-bottom:2px;">HOLDINGS CURRENTLY IN HIGH-VOL</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#fbbf24;">{n_in_high} / {len(tickers)}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:#5DCAA5;margin:16px 0 10px 0;">Holdings · Current Regime</div>', unsafe_allow_html=True)
    st.markdown('<div class="factor-row header"><div>TICKER</div><div>WT</div><div>REGIME</div><div>P(REGIME)</div><div>ANN. VOL</div><div>5D FCST VOL</div></div>', unsafe_allow_html=True)
    for tkr in tickers:
        res = regime_results[tkr]
        cur = res["current_regime"]
        css = regime_css(cur, n_regimes)
        lbl = regime_label(cur, n_regimes)
        c = REGIME_COLORS.get(cur if n_regimes == 3 else (0 if cur == 0 else 2), "#9FE1CB")
        fc_ann = float(np.mean(res["fc_vol_daily"]) * np.sqrt(252))
        st.markdown(f'''<div class="factor-row" style="border-left:2px solid {c};">
          <div style="color:#e2e8f0;font-weight:500">{tkr}</div>
          <div style="color:#60a5fa">{weights.get(tkr,0):.0%}</div>
          <div><span class="regime-badge {css}">{lbl}</span></div>
          <div style="color:#9FE1CB">{res["current_probs"][cur]:.1%}</div>
          <div style="color:#60a5fa;font-weight:600">{res["ann_vol_full"]:.1%}</div>
          <div style="color:#a78bfa;font-weight:600">{fc_ann:.1%}</div>
        </div>''', unsafe_allow_html=True)

    # Portfolio export
    rows = []
    for tkr in tickers:
        res = regime_results[tkr]
        cur = res["current_regime"]
        rows.append({
            "Ticker": tkr,
            "Weight": weights.get(tkr, 0),
            "Current_Regime": regime_label(cur, n_regimes),
            "Regime_Probability": res["current_probs"][cur],
            "Ann_Vol_EGARCH": res["ann_vol_full"],
            "5D_Forecast_Vol_Ann": float(np.mean(res["fc_vol_daily"]) * np.sqrt(252)),
        })
    st.download_button("⬇  Download Portfolio Regime Summary CSV", pd.DataFrame(rows).to_csv(index=False),
                       file_name="portfolio_regime_summary.csv", mime="text/csv", key="dl_port_regime_summary")

# ════════════════════════════════════════════
#  SESSION STATE INIT
# ════════════════════════════════════════════

for key, default in [
    ("run", False), ("ai_insight", None), ("ai_error", None),
    ("port_run", False), ("port_results", None),
    ("single_stock_cache", None), ("portfolio_cache", None),
    ("active_mode", "Single Stock"),
    ("regime_run", False), ("regime_results", None), ("regime_cache", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "selected_factors" not in st.session_state:
    st.session_state["selected_factors"] = list(ALL_FACTORS)


# ════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════

st.markdown("# Factor Regression")
st.markdown(
    '<p style="font-family:\'JetBrains Mono\',monospace;color:#1D9E75;font-size:13px;letter-spacing:1px;">'
    'FF5 + MOMENTUM + AQR (QMJ · BAB) · NEWEY-WEST ROBUST STANDARD ERRORS · MARKOV REGIME EGARCH</p>',
    unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### Configuration")

    mode = st.radio(
        "Mode",
        ["Single Stock", "Portfolio Attribution", "Markov Regime EGARCH"],
        horizontal=True, key="mode_radio"
    )
    if mode != st.session_state["active_mode"]:
        st.session_state["active_mode"] = mode
        st.rerun()

    st.markdown("---")

    if mode == "Single Stock":
        ticker = st.text_input("Stock Ticker", "", placeholder="e.g. AAPL").upper().strip()

    elif mode == "Portfolio Attribution":
        st.markdown("**Portfolio Holdings**")
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#5DCAA5;margin-bottom:10px;">'
            'Add stocks with ticker &amp; amount invested. Weights are auto-calculated.</div>',
            unsafe_allow_html=True
        )
        if "port_holdings" not in st.session_state:
            st.session_state["port_holdings"] = [
                {"ticker": "AAPL", "amount": 30000},
                {"ticker": "MSFT", "amount": 25000},
                {"ticker": "NVDA", "amount": 25000},
                {"ticker": "GOOGL", "amount": 20000},
            ]
        to_remove = None
        for idx, holding in enumerate(st.session_state["port_holdings"]):
            col_t, col_a, col_x = st.columns([2, 2, 0.6])
            with col_t:
                new_ticker = st.text_input(
                    "Ticker", value=holding["ticker"],
                    key=f"port_ticker_{idx}", label_visibility="collapsed", placeholder="TICKER"
                ).upper().strip()
                st.session_state["port_holdings"][idx]["ticker"] = new_ticker
            with col_a:
                new_amount = st.number_input(
                    "Amount", value=float(holding["amount"]), min_value=0.0, step=1000.0,
                    key=f"port_amount_{idx}", label_visibility="collapsed", format="%.0f"
                )
                st.session_state["port_holdings"][idx]["amount"] = new_amount
            with col_x:
                if st.button("×", key=f"port_remove_{idx}", help="Remove"):
                    to_remove = idx
        if to_remove is not None:
            st.session_state["port_holdings"].pop(to_remove)
            st.rerun()
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("＋  Add Stock", use_container_width=True, key="port_add"):
            st.session_state["port_holdings"].append({"ticker": "", "amount": 10000})
            st.rerun()
        total_amt = sum(h["amount"] for h in st.session_state["port_holdings"] if h["amount"] > 0)
        if total_amt > 0:
            preview_lines = [
                f'{h["ticker"]} · {h["amount"]/total_amt:.1%}'
                for h in st.session_state["port_holdings"] if h["ticker"] and h["amount"] > 0
            ]
            if preview_lines:
                st.markdown(
                    '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#5DCAA5;'
                    'margin-top:8px;padding:8px 10px;background:rgba(29,158,117,0.06);border-radius:6px;'
                    'border:1px solid rgba(29,158,117,0.15);line-height:1.8;">'
                    + " &nbsp;|&nbsp; ".join(preview_lines) + '</div>',
                    unsafe_allow_html=True
                )

    else:  # Markov Regime EGARCH — reuse the same portfolio holdings UI
        st.markdown("**Portfolio Holdings**")
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#5DCAA5;margin-bottom:10px;">'
            'Add stocks with ticker &amp; amount invested. Each holding gets a 2/3-state '
            'Markov Regime-Switching EGARCH volatility model.</div>',
            unsafe_allow_html=True
        )
        if "port_holdings" not in st.session_state:
            st.session_state["port_holdings"] = [
                {"ticker": "AAPL", "amount": 30000},
                {"ticker": "MSFT", "amount": 25000},
                {"ticker": "NVDA", "amount": 25000},
                {"ticker": "GOOGL", "amount": 20000},
            ]
        to_remove = None
        for idx, holding in enumerate(st.session_state["port_holdings"]):
            col_t, col_a, col_x = st.columns([2, 2, 0.6])
            with col_t:
                new_ticker = st.text_input(
                    "Ticker", value=holding["ticker"],
                    key=f"regime_ticker_{idx}", label_visibility="collapsed", placeholder="TICKER"
                ).upper().strip()
                st.session_state["port_holdings"][idx]["ticker"] = new_ticker
            with col_a:
                new_amount = st.number_input(
                    "Amount", value=float(holding["amount"]), min_value=0.0, step=1000.0,
                    key=f"regime_amount_{idx}", label_visibility="collapsed", format="%.0f"
                )
                st.session_state["port_holdings"][idx]["amount"] = new_amount
            with col_x:
                if st.button("×", key=f"regime_remove_{idx}", help="Remove"):
                    to_remove = idx
        if to_remove is not None:
            st.session_state["port_holdings"].pop(to_remove)
            st.rerun()
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("＋  Add Stock", use_container_width=True, key="regime_add"):
            st.session_state["port_holdings"].append({"ticker": "", "amount": 10000})
            st.rerun()
        total_amt = sum(h["amount"] for h in st.session_state["port_holdings"] if h["amount"] > 0)
        if total_amt > 0:
            preview_lines = [
                f'{h["ticker"]} · {h["amount"]/total_amt:.1%}'
                for h in st.session_state["port_holdings"] if h["ticker"] and h["amount"] > 0
            ]
            if preview_lines:
                st.markdown(
                    '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#5DCAA5;'
                    'margin-top:8px;padding:8px 10px;background:rgba(29,158,117,0.06);border-radius:6px;'
                    'border:1px solid rgba(29,158,117,0.15);line-height:1.8;">'
                    + " &nbsp;|&nbsp; ".join(preview_lines) + '</div>',
                    unsafe_allow_html=True
                )

    st.markdown("---")

    st.markdown("**Date Range**")
    start_date = st.date_input("Start", value=pd.to_datetime("2010-01-01"))
    end_date   = st.date_input("End",   value=pd.to_datetime("2026-04-30"))

    st.markdown("---")

    if mode in ("Single Stock", "Portfolio Attribution"):
        # ── Factor Selection ──────────────────────────────────────────────────────
        st.markdown("**Factor Selection**")
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#5DCAA5;margin-bottom:8px;">'
            'FF5 + Momentum factors loaded from Ken French\'s library.<br>'
            '<span style="color:#fbbf24;">AQR</span> factors (QMJ, BAB) loaded from AQR\'s public data library.</div>',
            unsafe_allow_html=True
        )
        col_sel, col_desel = st.columns(2)
        with col_sel:
            if st.button("✓ All", use_container_width=True, key="select_all_factors"):
                st.session_state["selected_factors"] = list(ALL_FACTORS)
                st.session_state["single_stock_cache"] = None
                st.session_state["portfolio_cache"]    = None
                st.rerun()
        with col_desel:
            if st.button("✗ None", use_container_width=True, key="deselect_all_factors"):
                st.session_state["selected_factors"] = []
                st.session_state["single_stock_cache"] = None
                st.session_state["portfolio_cache"]    = None
                st.rerun()

        new_selection = []
        # FF factors group
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:1px;'
            'text-transform:uppercase;color:#1D9E75;margin:6px 0 2px 0;">Fama-French + Momentum</div>',
            unsafe_allow_html=True
        )
        for f in FF_FACTORS:
            checked = f in st.session_state["selected_factors"]
            if st.checkbox(FACTOR_NAMES.get(f, f), value=checked, key=f"chk_{f}"):
                new_selection.append(f)

        # AQR factors group
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:1px;'
            'text-transform:uppercase;color:#fbbf24;margin:10px 0 2px 0;">AQR Factors</div>',
            unsafe_allow_html=True
        )
        for f in AQR_FACTORS:
            checked = f in st.session_state["selected_factors"]
            label   = f"{FACTOR_NAMES.get(f, f)} ⚡"
            if st.checkbox(label, value=checked, key=f"chk_{f}"):
                new_selection.append(f)

        if new_selection != st.session_state["selected_factors"]:
            st.session_state["selected_factors"] = new_selection
            st.session_state["single_stock_cache"] = None
            st.session_state["portfolio_cache"]    = None

        if st.session_state["selected_factors"]:
            active_labels = [FACTOR_NAMES.get(f, f) for f in st.session_state["selected_factors"]]
            st.markdown(
                '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#1D9E75;'
                'margin-top:6px;padding:6px 8px;background:rgba(29,158,117,0.06);'
                'border-radius:6px;border:1px solid rgba(29,158,117,0.15);">'
                f'Active: {", ".join(active_labels)}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#f87171;margin-top:6px;">'
                '⚠ No factors selected.</div>', unsafe_allow_html=True
            )

        st.markdown("---")

        st.markdown("**Regression Settings**")
        hac_lags  = st.slider("HAC Max Lags", 1, 12, 3, help="Newey-West lags for HAC robust SE")
        annualize = st.selectbox("Annualization", [12, 52, 252], help="12=monthly, 52=weekly, 252=daily")

        st.markdown("---")

    else:
        # Defaults so variables exist even if not used in this mode
        hac_lags  = 3
        annualize = 12

        st.markdown("**Regime-Switching EGARCH Settings**")
        n_regimes_sel = st.selectbox("Number of Regimes", [2, 3], index=0,
                                      help="2 = Low/High volatility, 3 = Low/Medium/High volatility")
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            egarch_p = st.number_input("EGARCH p", min_value=1, max_value=2, value=1, step=1, help="ARCH order")
        with col_e2:
            egarch_o = st.number_input("EGARCH o", min_value=0, max_value=2, value=1, step=1, help="Asymmetry order")
        with col_e3:
            egarch_q = st.number_input("EGARCH q", min_value=1, max_value=2, value=1, step=1, help="GARCH order")
        dist_choice = st.selectbox("Error Distribution", ["t", "normal", "skewt"], index=0,
                                    help="Student-t is recommended for fat-tailed daily equity returns")

        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#1D9E75;margin-top:8px;">'
            'Uses daily returns. Recommend at least 1-2 years of history for stable regime estimates.</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

    if mode == "Single Stock":
        if st.button("▶  RUN REGRESSION", use_container_width=True):
            if not ticker:
                st.error("Enter a ticker.")
            elif not st.session_state["selected_factors"]:
                st.error("Select at least one factor before running.")
            else:
                st.session_state["run"]        = True
                st.session_state["ticker_ran"] = ticker
                st.session_state["start_ran"]  = start_date
                st.session_state["end_ran"]    = end_date
                st.session_state["hac_ran"]    = hac_lags
                st.session_state["ann_ran"]    = annualize
                st.session_state["ai_insight"] = None
                st.session_state["ai_error"]   = None
                st.session_state["single_stock_cache"] = None

    elif mode == "Portfolio Attribution":
        if st.button("▶  RUN PORTFOLIO ATTRIBUTION", use_container_width=True):
            if not st.session_state["selected_factors"]:
                st.error("Select at least one factor before running.")
            else:
                holdings = st.session_state.get("port_holdings", [])
                parsed   = [(h["ticker"], h["amount"]) for h in holdings if h["ticker"] and h["amount"] > 0]
                if parsed:
                    total_w    = sum(w for _, w in parsed)
                    normalized = {tkr: w / total_w for tkr, w in parsed}
                    st.session_state["port_weights"]    = normalized
                    st.session_state["port_start"]      = start_date
                    st.session_state["port_end"]        = end_date
                    st.session_state["port_hac"]        = hac_lags
                    st.session_state["port_run"]        = True
                    st.session_state["port_results"]    = None
                    st.session_state["portfolio_cache"] = None
                else:
                    st.error("Add at least one valid holding.")

    else:  # Markov Regime EGARCH
        if st.button("▶  RUN REGIME EGARCH", use_container_width=True):
            holdings = st.session_state.get("port_holdings", [])
            parsed   = [(h["ticker"], h["amount"]) for h in holdings if h["ticker"] and h["amount"] > 0]
            if not parsed:
                st.error("Add at least one valid holding.")
            else:
                total_w    = sum(w for _, w in parsed)
                normalized = {tkr: w / total_w for tkr, w in parsed}
                st.session_state["regime_weights"]    = normalized
                st.session_state["regime_start"]      = start_date
                st.session_state["regime_end"]        = end_date
                st.session_state["regime_n"]          = n_regimes_sel
                st.session_state["regime_egarch_p"]   = int(egarch_p)
                st.session_state["regime_egarch_o"]   = int(egarch_o)
                st.session_state["regime_egarch_q"]   = int(egarch_q)
                st.session_state["regime_dist"]       = dist_choice
                st.session_state["regime_run"]        = True
                st.session_state["regime_results"]    = None
                st.session_state["regime_cache"]      = None

    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#1D9E75;">'
        'FF5 + Momentum · AQR QMJ + BAB · Markov Regime EGARCH · Monthly/Daily</div>',
        unsafe_allow_html=True)


# ════════════════════════════════════════════
#  MAIN AREA ROUTING
# ════════════════════════════════════════════

current_mode = st.session_state.get("active_mode", "Single Stock")

if not st.session_state["run"] and not st.session_state["port_run"] and not st.session_state["regime_run"] \
        and st.session_state["single_stock_cache"] is None \
        and st.session_state["portfolio_cache"] is None \
        and st.session_state["regime_cache"] is None:
    if current_mode == "Single Stock":
        st.markdown(
            '<div class="interpret-box" style="margin-top:40px;text-align:center;">'
            '<b>Single Stock Mode</b><br><br>'
            '1. Enter a ticker<br>2. Set date range<br>'
            '3. Select factors — including <span style="color:#fbbf24;">AQR QMJ &amp; BAB</span><br>'
            '4. Click <b>▶ RUN REGRESSION</b><br><br>'
            '<span style="color:#1D9E75;font-size:13px;">Factor analysis runs only on statistically significant loadings (p&lt;0.05).</span>'
            '</div>', unsafe_allow_html=True)
    elif current_mode == "Portfolio Attribution":
        st.markdown(
            '<div class="interpret-box" style="margin-top:40px;text-align:center;">'
            '<b>Portfolio Attribution Mode</b><br><br>'
            '1. Add your holdings (ticker + amount)<br>2. Set date range<br>'
            '3. Select factors — including <span style="color:#fbbf24;">AQR QMJ &amp; BAB</span><br>'
            '4. Click <b>▶ RUN PORTFOLIO ATTRIBUTION</b>'
            '</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="interpret-box" style="margin-top:40px;text-align:center;">'
            '<b>Markov Regime EGARCH Mode</b><br><br>'
            '1. Add your holdings (ticker + amount)<br>2. Set date range (daily data, longer history recommended)<br>'
            '3. Choose number of regimes (2 or 3) and EGARCH(p,o,q) settings<br>'
            '4. Click <b>▶ RUN REGIME EGARCH</b><br><br>'
            '<span style="color:#1D9E75;font-size:13px;">Identifies volatility regimes via Markov-switching, then fits EGARCH per regime for asymmetric vol forecasting.</span>'
            '</div>', unsafe_allow_html=True)
    st.stop()
