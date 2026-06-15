import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from scipy import stats
from groq import Groq
from datetime import date
import re
import json
import base64
import warnings
warnings.filterwarnings("ignore")

FF_URL = "https://raw.githubusercontent.com/lakshyaworkch-cell/Lakshy/main/F-F_Research_Data_5_Factors_2x3.csv"

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
    """background: #0a1f1a;
    background-image:
        radial-gradient(ellipse at 15% 0%, rgba(29,158,117,0.18) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 100%, rgba(15,110,86,0.14) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 50%, rgba(4,52,44,0.92) 0%, transparent 100%);"""
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

/* ── APP BACKGROUND ── */
.stApp {{
    {_BG_CSS}
    color: #e2e8f0;
}}

/* ── KILL THE WHITE TOP BANNER ── */
[data-testid="stHeader"] {{
    background: transparent !important;
    background-color: transparent !important;
}}
[data-testid="stHeader"] * {{
    background: transparent !important;
    background-color: transparent !important;
}}
[data-testid="stDecoration"] {{
    background: transparent !important;
    background-image: none !important;
}}
[data-testid="stToolbar"] {{
    background: transparent !important;
    right: 1rem;
}}
[data-testid="stStatusWidget"] {{
    background: transparent !important;
}}

/* ── SIDEBAR BACKGROUND (stable selector, works across domains) ── */
[data-testid="stSidebar"] {{
    {_BG_CSS}
    border-right: 1px solid rgba(29, 158, 117, 0.2) !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    background: transparent !important;
}}
/* Sidebar inner scrollable container */
[data-testid="stSidebar"] .stSidebarContent,
[data-testid="stSidebarContent"] {{
    background: transparent !important;
}}
/* Catch all generated emotion-cache sidebar classes */
section[data-testid="stSidebar"] * {{
    --background-color: transparent;
}}

/* ── SIDEBAR TEXT ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {{
    color: #e2e8f0 !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: #e2e8f0 !important;
}}

/* ── SIDEBAR INPUTS ── */
[data-testid="stSidebar"] .stTextInput > div > div > input {{
    background: rgba(255,255,255,0.92) !important;
    border: 1px solid rgba(29,158,117,0.35) !important;
    color: #111111 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div > input {{
    background: rgba(255,255,255,0.92) !important;
    border: 1px solid rgba(29,158,117,0.35) !important;
    color: #111111 !important;
    border-radius: 8px !important;
}}

/* ── SIDEBAR BUTTONS ── */
[data-testid="stSidebar"] .stButton > button {{
    background: rgba(29,158,117,0.12) !important;
    color: #9FE1CB !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
    border: 1px solid rgba(29,158,117,0.3) !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(52,211,153,0.15) !important;
    border-color: rgba(52,211,153,0.5) !important;
    color: #34d399 !important;
}}

/* ── SIDEBAR RADIO / CHECKBOX ── */
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label {{
    color: #e2e8f0 !important;
}}

/* ── SIDEBAR SLIDER ── */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {{
    background: rgba(29,158,117,0.2) !important;
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

/* === FACTOR ROWS === */
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

/* === DIAGNOSTICS GRID === */
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

/* === MAIN AREA INPUTS & BUTTONS === */
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

/* === FACTOR REGIME STRIP === */
.regime-row {{
    display: flex; flex-direction: column; gap: 8px; margin-bottom: 18px;
}}
.regime-card {{
    background: rgba(29,158,117,0.06); border: 1px solid rgba(29,158,117,0.15);
    border-radius: 10px; padding: 10px 14px;
    display: flex; align-items: center; gap: 12px;
}}
.regime-name {{
    font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #e2e8f0;
    font-weight: 500; white-space: nowrap;
    flex: 0 0 170px;
}}
.regime-cells {{ display: flex; gap: 10px; flex: 1; }}
.regime-cell {{ flex: 1 1 0; min-width: 0; text-align: center; }}
.regime-label {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.5px;
    color: #5DCAA5; margin-bottom: 2px; white-space: nowrap;
}}
.regime-val {{
    font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600;
    white-space: nowrap;
}}
.regime-val.na {{ color: #6b7280; font-weight: 400; }}
.regime-card-dim {{ opacity: 0.55; }}
.regime-name-dim {{ color: #6b7280 !important; }}
.regime-ns-tag {{
    font-size: 10px; letter-spacing: 1px; color: #4a7c6f;
    text-transform: uppercase; font-weight: 400;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

FF_FACTORS  = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
AQR_FACTORS = ["QMJ", "BAB"]
ALL_FACTORS = FF_FACTORS + AQR_FACTORS
AQR_FACTOR_SET = set(AQR_FACTORS)

# Fixed benchmark for "Active Exposure" comparisons.
BENCHMARK_TICKER = "SPY"

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
    "QMJ":  "#facc15",
    "BAB":  "#38bdf8",
    "const": "#94a3b8",
}


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
    results = {}
    try:
        resp_qmj = pd.read_excel(AQR_QMJ_URL, sheet_name="QMJ Factors", header=18, index_col=0)
        resp_qmj.index = pd.to_datetime(resp_qmj.index, errors="coerce")
        resp_qmj = resp_qmj[resp_qmj.index.notna()]
        resp_qmj.index = resp_qmj.index.to_period("M")
        if "USA" in resp_qmj.columns:
            qmj_series = pd.to_numeric(resp_qmj["USA"], errors="coerce").dropna()
        else:
            qmj_series = pd.to_numeric(resp_qmj.iloc[:, 0], errors="coerce").dropna()
        results["QMJ"] = qmj_series
    except Exception as e:
        st.warning(f"Could not load AQR QMJ data: {e}. QMJ will be unavailable.")

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
    for col in df.columns:
        if df[col].abs().mean() > 0.5:
            df[col] = df[col] / 100.0
    return df


def merge_all_factors(ff_df, aqr_df, selected_factors):
    combined = ff_df.copy()
    for col in AQR_FACTORS:
        if col in selected_factors and col in aqr_df.columns:
            combined[col] = aqr_df[col]
    return combined


@st.cache_data(show_spinner=False)
def get_scaled_factor_df(start_str, end_str, selected_factors_key):
    """
    Returns the factor dataframe sliced to [start_str, end_str], with FF5+Mom
    factors converted from percent to decimal (consistent with how Single Stock
    mode scales them) and AQR factors left as-is (already decimal).
    Also returns RF in decimal form if present.
    """
    ff_raw  = load_factors()
    aqr_raw = load_aqr_factors()
    ff_raw  = merge_all_factors(ff_raw, aqr_raw, list(selected_factors_key))
    ff = ff_raw.loc[start_str:end_str].copy()
    available = [c for c in selected_factors_key if c in ff.columns]
    for col in available:
        if col in FF_FACTORS:
            ff[col] = ff[col].astype(float) / 100
        else:
            ff[col] = ff[col].astype(float)
    if "RF" in ff.columns:
        ff["RF"] = ff["RF"].astype(float) / 100
    return ff, available


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
        "Analyze the War in Iran and take that into consideration also and its affect on the stock"
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
#  Factor Regime Strip & Benchmark Active Exposure
# ─────────────────────────────────────────────

def render_factor_regime_strip(ff_scaled, available_factors, pvalues=None):
    """
    Renders trailing 1M / 3M / 12M realized returns for each selected factor,
    using an already-scaled (decimal) factor dataframe sliced to the analysis window.

    Cards stack vertically — factor name on the left, 1M/3M/12M values on the right.
    If `pvalues` (a dict of factor -> p-value from the regression) is provided,
    factors that are NOT statistically significant (p >= 0.05) for this ticker
    have their name dimmed and tagged "n.s." — the realized-return values
    themselves stay at full visibility since they're market data, not
    stock-specific.
    """
    if ff_scaled is None or ff_scaled.empty or not available_factors:
        return ""

    windows = [("1M", 1), ("3M", 3), ("12M", 12)]
    cards = ""
    for f in available_factors:
        if f not in ff_scaled.columns:
            continue
        series = ff_scaled[f].dropna()
        cells = ""
        for label, months in windows:
            if len(series) >= months:
                window_vals = series.iloc[-months:]
                cumret = (1 + window_vals).prod() - 1
                color = "#34d399" if cumret > 0 else "#f87171"
                val_html = f'<div class="regime-val" style="color:{color};">{cumret:+.2%}</div>'
            else:
                val_html = '<div class="regime-val na">N/A</div>'
            cells += (
                f'<div class="regime-cell">'
                f'<div class="regime-label">{label}</div>'
                f'{val_html}</div>'
            )
        aqr_tag = ' <span class="aqr-badge">AQR</span>' if f in AQR_FACTOR_SET else ""

        is_significant = True
        if pvalues is not None:
            p = pvalues.get(f)
            if p is not None:
                is_significant = p < 0.05

        if is_significant:
            name_html = f'<div class="regime-name">{FACTOR_NAMES.get(f, f)}{aqr_tag}</div>'
            card_cls = "regime-card"
        else:
            name_html = (
                f'<div class="regime-name regime-name-dim">{FACTOR_NAMES.get(f, f)}{aqr_tag} '
                f'<span class="regime-ns-tag">n.s.</span></div>'
            )
            card_cls = "regime-card regime-card-dim"

        cards += (
            f'<div class="{card_cls}">'
            f'{name_html}'
            f'<div class="regime-cells">{cells}</div>'
            f'</div>'
        )

    if not cards:
        return ""

    return (
        '<div class="section-title">Factor Regime · Trailing Returns</div>'
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#5DCAA5;margin-bottom:10px;">'
        'Realized factor returns over the most recent 1 / 3 / 12 months in the analysis window — '
        'shows whether each factor has recently been a tailwind or a headwind.</div>'
        f'<div class="regime-row">{cards}</div>'
    )


def render_active_exposure(entity_params, bench_params, available_factors, bench_ticker, entity_label="POSITION"):
    grid_cols = "160px 1fr 1fr 1fr"
    html = (
        f'<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">'
        f'<div style="display:grid;grid-template-columns:{grid_cols};gap:6px;min-width:520px;">'
    )
    for h in ["FACTOR", entity_label, bench_ticker, "ACTIVE (DIFF)"]:
        html += (
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:1px;'
            f'color:#1D9E75;text-transform:uppercase;padding:4px 0;">{h}</div>'
        )
    for f in ["const"] + available_factors:
        eb = entity_params.get(f, 0.0)
        bb = bench_params.get(f, 0.0)
        active = eb - bb
        ec = "#34d399" if eb >= 0 else "#f87171"
        bc = "#34d399" if bb >= 0 else "#f87171"
        ac = "#34d399" if active >= 0 else "#f87171"
        aqr_tag = ' <span class="aqr-badge">AQR</span>' if f in AQR_FACTOR_SET else ""
        label = FACTOR_NAMES.get(f, f)
        html += (
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#e2e8f0;'
            f'font-weight:500;padding:8px 0;border-top:1px solid rgba(29,158,117,0.15);">{label}{aqr_tag}</div>'
        )
        html += (
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{ec};'
            f'padding:8px 0;border-top:1px solid rgba(29,158,117,0.15);">{eb:+.4f}</div>'
        )
        html += (
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{bc};'
            f'padding:8px 0;border-top:1px solid rgba(29,158,117,0.15);">{bb:+.4f}</div>'
        )
        html += (
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;font-weight:700;color:{ac};'
            f'padding:8px 0;border-top:1px solid rgba(29,158,117,0.15);">{active:+.4f}</div>'
        )
    html += '</div></div>'
    return html


def render_active_exposure_section(entity_params, bench_result, bench_err, available_factors, bench_ticker, entity_label="POSITION"):
    html = '<div class="section-title">Active Exposure vs Benchmark</div>'
    if bench_err or not bench_result:
        html += (
            f'<div class="error-box">Benchmark regression unavailable for {bench_ticker}: '
            f'{bench_err or "no data returned"}</div>'
        )
        return html
    html += (
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#5DCAA5;margin-bottom:10px;">'
        f'Positive ACTIVE = more exposure than {bench_ticker} on that factor · negative = less. '
        f'The Alpha (α) row shows excess alpha after netting out {bench_ticker}\'s own alpha over the same window '
        f'({bench_result.get("nobs","?")} obs).</div>'
    )
    html += render_active_exposure(entity_params, bench_result["params"], available_factors, bench_ticker, entity_label=entity_label)
    return html


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
        ff, available = get_scaled_factor_df(start_str, end_str, selected_factors_key)
        has_rf = "RF" in ff.columns

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
        ff, _ = get_scaled_factor_df(start_str, end_str, tuple(available_factors))
        has_rf = "RF" in ff.columns

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

    return port_betas, port_alpha_ann


# ════════════════════════════════════════════
#  SESSION STATE INIT
# ════════════════════════════════════════════

for key, default in [
    ("run", False), ("ai_insight", None), ("ai_error", None),
    ("port_run", False), ("port_results", None),
    ("single_stock_cache", None), ("portfolio_cache", None),
    ("active_mode", "Single Stock"),
    # FIX: store the ticker input value in session state so it persists
    # and is available when the run button is clicked
    ("ticker_input", ""),
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
    'FF5 + MOMENTUM + AQR (QMJ · BAB) · NEWEY-WEST ROBUST STANDARD ERRORS</p>',
    unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### Configuration")

    mode = st.radio("Mode", ["Single Stock", "Portfolio Attribution"], horizontal=True, key="mode_radio")
    if mode != st.session_state["active_mode"]:
        st.session_state["active_mode"] = mode
        st.rerun()

    st.markdown("---")

    if mode == "Single Stock":
        # FIX 1: No default value ("AVGO" removed → blank placeholder instead)
        # FIX 2: Use key= so the value is stored in session_state["ticker_input"]
        #         and is immediately readable when the Run button is clicked
        #         on the same script execution pass.
        ticker_input = st.text_input(
            "Stock Ticker",
            value=st.session_state["ticker_input"],
            placeholder="e.g. AAPL",
            key="ticker_input_widget",
        ).upper().strip()
        # Mirror to session state immediately so the run handler below always
        # has the freshest value regardless of widget re-render order.
        st.session_state["ticker_input"] = ticker_input

    else:
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

    st.markdown("---")

    st.markdown("**Date Range**")
    start_date = st.date_input("Start", value=pd.to_datetime("2010-01-01"))
    end_date   = st.date_input("End",   value=pd.to_datetime("2026-04-30"))

    st.markdown("---")

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
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:1px;'
        'text-transform:uppercase;color:#1D9E75;margin:6px 0 2px 0;">Fama-French + Momentum</div>',
        unsafe_allow_html=True
    )
    for f in FF_FACTORS:
        checked = f in st.session_state["selected_factors"]
        if st.checkbox(FACTOR_NAMES.get(f, f), value=checked, key=f"chk_{f}"):
            new_selection.append(f)

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

    st.markdown("**Benchmark (Active Exposure)**")
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#5DCAA5;">'
        f'Regression betas vs. <span style="color:#fbbf24;">{BENCHMARK_TICKER}</span> over the same window show your '
        '<span style="color:#fbbf24;">active</span> factor bets.</div>',
        unsafe_allow_html=True
    )
    # Benchmark is fixed to SPY — no text input needed.
    benchmark_ticker = BENCHMARK_TICKER

    st.markdown("---")

    if mode == "Single Stock":
        run_clicked = st.button("▶  RUN REGRESSION", use_container_width=True)
        if run_clicked:
            # FIX 3: Read ticker directly from session state (set above from the widget)
            #         This avoids the "no ticker on first boot" bug where the widget
            #         value isn't yet propagated through the normal flow on the first
            #         script run after clicking the button.
            _ticker_to_run = st.session_state["ticker_input"]
            if not _ticker_to_run:
                st.error("Please enter a stock ticker before running.")
            elif not st.session_state["selected_factors"]:
                st.error("Select at least one factor before running.")
            else:
                st.session_state["run"]        = True
                st.session_state["ticker_ran"] = _ticker_to_run
                st.session_state["start_ran"]  = start_date
                st.session_state["end_ran"]    = end_date
                st.session_state["hac_ran"]    = hac_lags
                st.session_state["ann_ran"]    = annualize
                st.session_state["ai_insight"] = None
                st.session_state["ai_error"]   = None
                st.session_state["single_stock_cache"] = None
    else:
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

    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#1D9E75;">'
        'FF5 + Momentum · AQR QMJ + BAB · Monthly</div>',
        unsafe_allow_html=True)


# ════════════════════════════════════════════
#  MAIN AREA ROUTING
# ════════════════════════════════════════════

current_mode = st.session_state.get("active_mode", "Single Stock")

if not st.session_state["run"] and not st.session_state["port_run"] \
        and st.session_state["single_stock_cache"] is None \
        and st.session_state["portfolio_cache"] is None:
    if current_mode == "Single Stock":
        st.markdown(
            '<div class="interpret-box" style="margin-top:40px;text-align:center;">'
            '<b>Single Stock Mode</b><br><br>'
            '1. Enter a ticker in the left panel<br>2. Set date range<br>'
            '3. Select factors — including <span style="color:#fbbf24;">AQR QMJ &amp; BAB</span><br>'
            '4. Click <b>▶ RUN REGRESSION</b><br><br>'
            '<span style="color:#1D9E75;font-size:13px;">Factor analysis runs only on statistically significant loadings (p&lt;0.05).</span>'
            '</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="interpret-box" style="margin-top:40px;text-align:center;">'
            '<b>Portfolio Attribution Mode</b><br><br>'
            '1. Add your holdings (ticker + amount)<br>2. Set date range<br>'
            '3. Select factors — including <span style="color:#fbbf24;">AQR QMJ &amp; BAB</span><br>'
            '4. Click <b>▶ RUN PORTFOLIO ATTRIBUTION</b>'
            '</div>', unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════
#  PORTFOLIO ATTRIBUTION MODE
# ════════════════════════════════════════════

if current_mode == "Portfolio Attribution":
    if st.session_state["portfolio_cache"] is not None and not st.session_state.get("port_run"):
        cache = st.session_state["portfolio_cache"]
        st.markdown("## Portfolio Factor Attribution")
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:13px;color:#5DCAA5;margin-bottom:20px;">'
            f'{len(cache["weights"])} holdings · {len(cache["ordered_factors"])} factors · HAC SE · {cache["port_start"]} → {cache["port_end"]}'
            f'</div>', unsafe_allow_html=True)
        if cache.get("regime_html"):
            st.markdown(cache["regime_html"], unsafe_allow_html=True)
        render_portfolio_attribution(
            cache["port_results"], cache["weights"], cache["ordered_factors"],
            cache.get("true_port"), str(cache["port_start"])[:7], str(cache["port_end"])[:7],
            cache.get("hac_lags", 3)
        )
        if cache.get("active_exposure_html"):
            st.markdown(cache["active_exposure_html"], unsafe_allow_html=True)
        st.stop()

    if not st.session_state.get("port_run"):
        st.stop()

    weights     = st.session_state.get("port_weights", {})
    port_start  = st.session_state.get("port_start", start_date)
    port_end    = st.session_state.get("port_end",   end_date)
    port_hac    = st.session_state.get("port_hac",   hac_lags)
    sel_factors = tuple(st.session_state["selected_factors"])
    bench_ticker = BENCHMARK_TICKER

    if not weights:
        st.markdown('<div class="error-box">No valid tickers found.</div>', unsafe_allow_html=True)
        st.stop()

    st.markdown("## Portfolio Factor Attribution")
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:13px;color:#5DCAA5;margin-bottom:20px;">'
        f'{len(weights)} holdings · {len(sel_factors)} factors · HAC SE · {port_start} → {port_end}</div>',
        unsafe_allow_html=True)

    port_results = {}
    errors       = []
    progress_bar = st.progress(0, text="Running regressions…")
    start_str    = str(port_start)[:7]
    end_str      = str(port_end)[:7]

    for i, (tkr, w) in enumerate(weights.items()):
        progress_bar.progress(i / len(weights), text=f"Regressing {tkr}…")
        res, err = run_single_regression(tkr, start_str, end_str, port_hac, f"{start_str}_{end_str}", sel_factors)
        if res:
            port_results[tkr] = res
        else:
            errors.append(f"{tkr}: {err}")
    progress_bar.progress(1.0, text="Done.")

    for e in errors:
        st.markdown(f'<div class="error-box">⚠ {e}</div>', unsafe_allow_html=True)
    if not port_results:
        st.markdown('<div class="error-box">No stocks could be regressed.</div>', unsafe_allow_html=True)
        st.stop()

    valid_weights = {t: weights[t] for t in port_results}
    total = sum(valid_weights.values())
    valid_weights = {t: v / total for t, v in valid_weights.items()}

    avail_sets      = [set(port_results[t]["available"]) for t in port_results]
    common_factors  = list(avail_sets[0].intersection(*avail_sets[1:])) if len(avail_sets) > 1 else list(avail_sets[0])
    ordered_factors = [f for f in ALL_FACTORS if f in common_factors]

    with st.spinner("Building portfolio regression model…"):
        true_port = build_true_portfolio_model(port_results, valid_weights, ordered_factors, start_str, end_str, port_hac)

    # ── Factor Regime Strip (trailing factor returns over the analysis window) ──
    ff_regime, regime_available = get_scaled_factor_df(start_str, end_str, tuple(ordered_factors))
    regime_html = render_factor_regime_strip(ff_regime, regime_available)
    if regime_html:
        st.markdown(regime_html, unsafe_allow_html=True)

    # ── Render attribution, then compute entity betas for active exposure ──
    port_betas, port_alpha_ann = render_portfolio_attribution(
        port_results, valid_weights, ordered_factors, true_port, start_str, end_str, port_hac
    )

    # ── Benchmark regression + Active Exposure ──
    if true_port:
        entity_params = true_port["betas"]
    else:
        entity_params = port_betas

    with st.spinner(f"Regressing benchmark {bench_ticker}…"):
        bench_result, bench_err = run_single_regression(
            bench_ticker, start_str, end_str, port_hac, f"{start_str}_{end_str}_{bench_ticker}", tuple(ordered_factors)
        )
    active_exposure_html = render_active_exposure_section(
        entity_params, bench_result, bench_err, ordered_factors, bench_ticker, entity_label="PORTFOLIO"
    )
    st.markdown(active_exposure_html, unsafe_allow_html=True)

    st.session_state["portfolio_cache"] = {
        "port_results": port_results, "weights": valid_weights,
        "ordered_factors": ordered_factors, "port_start": port_start, "port_end": port_end,
        "true_port": true_port, "hac_lags": port_hac,
        "regime_html": regime_html, "active_exposure_html": active_exposure_html,
        "bench_ticker": bench_ticker,
    }
    st.session_state["port_run"] = False
    st.stop()


# ════════════════════════════════════════════
#  SINGLE STOCK MODE
# ════════════════════════════════════════════

def _render_ols_summary(ols_text):
    escaped = ols_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f'<div class="ols-box"><pre>{escaped}</pre></div>',
        unsafe_allow_html=True
    )

if current_mode == "Single Stock":

    if st.session_state["single_stock_cache"] is not None and not st.session_state["run"]:
        c = st.session_state["single_stock_cache"]
        st.markdown(c["price_html"], unsafe_allow_html=True)
        if c.get("regime_html"):
            st.markdown(c["regime_html"], unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(c["metric_alpha"], unsafe_allow_html=True)
        with col2: st.markdown(c["metric_r2"],    unsafe_allow_html=True)
        with col3: st.markdown(c["metric_ir"],    unsafe_allow_html=True)
        with col4: st.markdown(c["metric_f"],     unsafe_allow_html=True)
        st.markdown('<div class="section-title">Factor Loadings</div>', unsafe_allow_html=True)
        for row_html in c["factor_rows"]:
            st.markdown(row_html, unsafe_allow_html=True)
        st.markdown(c["factor_legend"], unsafe_allow_html=True)
        if c.get("active_exposure_html"):
            st.markdown(c["active_exposure_html"], unsafe_allow_html=True)
        st.markdown('<div class="section-title">Factor Intelligence · Significant Loadings</div>', unsafe_allow_html=True)
        st.markdown(c["ai_header"], unsafe_allow_html=True)
        if c.get("ai_error") and not c.get("ai_insight"):
            st.markdown(f'<div class="error-box">Analysis Error: {c["ai_error"]}</div>', unsafe_allow_html=True)
        elif c.get("ai_error") and c.get("ai_insight"):
            st.markdown(f'<div class="error-box" style="border-color:rgba(251,191,36,0.3);color:#fbbf24;">⚠ Some factors had errors: {c["ai_error"]}</div>', unsafe_allow_html=True)
        if c.get("ai_insight"):
            render_ai_insight(c["ticker"], c["ai_insight"])
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["95% Confidence Intervals","Regression Diagnostics","Variance Inflation Factors","Model Fit","Rolling Market Beta"])
        with tab1: st.markdown(c["ci_html"],   unsafe_allow_html=True)
        with tab2: st.markdown(c["diag_html"], unsafe_allow_html=True)
        with tab3: st.markdown(c["vif_html"],  unsafe_allow_html=True)
        with tab4: st.markdown(c["fit_html"],  unsafe_allow_html=True)
        with tab5:
            if c.get("rolling_html"): st.markdown(c["rolling_html"], unsafe_allow_html=True)
            else: st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:13px;color:#5DCAA5;padding:20px 0;">Need at least 36 months of data and Mkt-RF selected.</div>', unsafe_allow_html=True)
        with st.expander("Full OLS Summary"):
            _render_ols_summary(c["ols_summary"])
        st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("⬇  Download Results CSV", c["export_csv"], file_name=f"{c['ticker']}_factor_regression.csv", mime="text/csv", use_container_width=True)
        with col_b:
            st.download_button("⬇  Download Merged Data CSV", c["merged_csv"], file_name=f"{c['ticker']}_merged_data.csv", mime="text/csv", use_container_width=True)
        st.stop()

    if not st.session_state["run"]:
        st.stop()

    ticker     = st.session_state.get("ticker_ran", "")
    start_date = st.session_state.get("start_ran", start_date)
    end_date   = st.session_state.get("end_ran",   end_date)
    hac_lags   = st.session_state.get("hac_ran",   hac_lags)
    annualize  = st.session_state.get("ann_ran",   annualize)
    sel_factors= st.session_state["selected_factors"]
    bench_ticker = BENCHMARK_TICKER

    # FIX 3 continued: guard against empty ticker making it this far
    if not ticker:
        st.markdown('<div class="error-box">No ticker specified. Please enter a ticker in the left panel and click Run.</div>', unsafe_allow_html=True)
        st.session_state["run"] = False
        st.stop()

    try:
        with st.spinner("Loading factor data..."):
            try:
                ff, available = get_scaled_factor_df(str(start_date)[:7], str(end_date)[:7], tuple(sel_factors))
                has_rf = "RF" in ff.columns
            except Exception as e:
                st.markdown(f'<div class="error-box">Failed to load factor data: {e}</div>', unsafe_allow_html=True)
                st.stop()

        if not available:
            st.markdown('<div class="error-box">None of the selected factors found in the dataset.</div>', unsafe_allow_html=True)
            st.stop()

        loaded_aqr = [f for f in AQR_FACTORS if f in available]
        missing_aqr = [f for f in sel_factors if f in AQR_FACTORS and f not in available]
        if loaded_aqr:
            st.markdown(
                f'<div class="info-box">✓ AQR factors loaded: {", ".join(loaded_aqr)}</div>',
                unsafe_allow_html=True
            )
        if missing_aqr:
            st.markdown(
                f'<div class="error-box">⚠ AQR factors unavailable (check network): {", ".join(missing_aqr)}</div>',
                unsafe_allow_html=True
            )

        with st.spinner(f"Downloading {ticker} price history..."):
            raw = yf.download(ticker, start=str(start_date), end=str(end_date), auto_adjust=True, progress=False)

        if raw.empty:
            st.markdown(f'<div class="error-box">No price data found for: <b>{ticker}</b>. Check the ticker symbol is valid.</div>', unsafe_allow_html=True)
            st.session_state["run"] = False
            st.stop()

        close   = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        monthly = close.resample("ME").last()
        returns = monthly.pct_change().dropna()
        returns.index = returns.index.to_period("M")

        data = pd.DataFrame({"Stock": returns}).join(ff[available + (["RF"] if has_rf else [])], how="inner")
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        if len(data) < 24:
            st.markdown(f'<div class="error-box">Too few observations ({len(data)}). Check date range.</div>', unsafe_allow_html=True)
            st.stop()

        data["Y"] = data["Stock"] - data["RF"] if has_rf else data["Stock"]

        X = sm.add_constant(data[available])
        y = data["Y"]
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

        alpha     = model.params["const"]
        alpha_ann = (1 + alpha) ** annualize - 1
        alpha_p   = model.pvalues["const"]
        n         = int(model.nobs)
        r2        = model.rsquared
        r2_adj    = model.rsquared_adj
        f_stat    = model.fvalue
        f_p       = model.f_pvalue
        aic       = model.aic
        bic       = model.bic
        resid     = model.resid
        te        = resid.std() * np.sqrt(annualize)
        ir        = alpha_ann / te if te > 0 else np.nan

        bp_stat, bp_p, _, _ = sm.stats.diagnostic.het_breuschpagan(resid, X)
        dw            = sm.stats.stattools.durbin_watson(resid)
        jb_stat, jb_p = stats.jarque_bera(resid)[:2]
        cond          = np.linalg.cond(X.values)

        vif_data = {}
        for col in available:
            others = [c for c in available if c != col]
            if others:
                r2_vif = sm.OLS(X[col], sm.add_constant(X[others])).fit().rsquared
                vif_data[col] = 1 / (1 - r2_vif) if r2_vif < 1 else np.inf
            else:
                vif_data[col] = 1.0

        dw_status = "pass" if 1.5 < dw < 2.5 else "warn" if 1.2 < dw < 2.8 else "fail"
        bp_status = "pass" if bp_p > 0.05 else "warn" if bp_p > 0.01 else "fail"
        jb_status = "pass" if jb_p > 0.05 else "warn" if jb_p > 0.01 else "fail"
        cn_status = "pass" if cond < 30 else "warn" if cond < 100 else "fail"

        def diag_card(name, val, sub, status):
            cls = {"pass":"diag-pass","fail":"diag-fail","warn":"diag-warn"}.get(status,"diag-pass")
            return f'<div class="diag-item"><div class="diag-name">{name}</div><div class="diag-val {cls}">{val}</div><div class="diag-sub">{sub}</div></div>'

        try:
            _live_price, _prev_close, _currency = get_live_price(ticker)
            _chg = _live_price - _prev_close; _chg_pct = (_chg / _prev_close) * 100
            _clr = "#34d399" if _chg >= 0 else "#f87171"; _arrow = "&#9650;" if _chg >= 0 else "&#9660;"
            _price_html = (
                f'<div style="display:flex;flex-wrap:wrap;align-items:baseline;gap:10px 16px;margin-bottom:20px;'
                f'padding:14px 16px;background:rgba(29,158,117,0.08);border:1px solid rgba(29,158,117,0.2);border-radius:12px;">'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:18px;font-weight:700;color:#e2e8f0;">{ticker}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:22px;font-weight:700;color:{_clr};">{_currency} {_live_price:,.2f}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:13px;color:{_clr};">{_arrow} {abs(_chg):,.2f} ({abs(_chg_pct):.2f}%)</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:13px;color:#5DCAA5;width:100%;">prev close {_currency} {_prev_close:,.2f}</span>'
                f'</div>'
            )
        except Exception:
            _price_html = (
                f'<div style="padding:14px 16px;background:rgba(29,158,117,0.08);border:1px solid rgba(29,158,117,0.2);'
                f'border-radius:12px;margin-bottom:20px;font-family:JetBrains Mono,monospace;'
                f'font-size:18px;font-weight:700;color:#e2e8f0;">{ticker} <span style="font-size:13px;color:#5DCAA5;">· price unavailable</span></div>'
            )
        st.markdown(_price_html, unsafe_allow_html=True)

        active_chips = []
        for f in sel_factors:
            label = FACTOR_NAMES.get(f, f)
            if f in AQR_FACTOR_SET:
                label += ' <span class="aqr-badge">AQR</span>'
            active_chips.append(label)
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:13px;color:#5DCAA5;'
            f'margin-bottom:16px;padding:8px 12px;background:rgba(29,158,117,0.06);'
            f'border:1px solid rgba(29,158,117,0.15);border-radius:8px;">'
            f'Regressing on: <span style="color:#e2e8f0;">{" · ".join(active_chips)}</span></div>',
            unsafe_allow_html=True)

        # ── Factor Regime Strip (trailing factor returns over the analysis window) ──
        regime_html = render_factor_regime_strip(ff, available, pvalues=dict(model.pvalues))
        if regime_html:
            st.markdown(regime_html, unsafe_allow_html=True)

        _alpha_card = f"""
        <div class="metric-card {'green' if alpha_ann > 0 else 'red'}">
          <div class="metric-label">Annual Alpha</div>
          <div class="metric-value" style="color:{'#34d399' if alpha_ann>0 else '#f87171'}">{alpha_ann:+.2%}</div>
          <div class="metric-sub">Monthly: {alpha:+.4f} · p={alpha_p:.4f}</div>
        </div>"""
        _r2_card = f"""
        <div class="metric-card blue">
          <div class="metric-label">R² / Adj R²</div>
          <div class="metric-value">{r2:.4f}</div>
          <div class="metric-sub">Adj: {r2_adj:.4f}</div>
        </div>"""
        ir_color = "#34d399" if not np.isnan(ir) and ir > 0.5 else "#fbbf24" if not np.isnan(ir) and ir > 0 else "#f87171"
        ir_str   = f"{ir:.3f}" if not np.isnan(ir) else "N/A"
        _ir_card = f"""
        <div class="metric-card gold">
          <div class="metric-label">Information Ratio</div>
          <div class="metric-value" style="color:{ir_color}">{ir_str}</div>
          <div class="metric-sub">Ann. TE: {te:.2%}</div>
        </div>"""
        _f_card = f"""
        <div class="metric-card gray">
          <div class="metric-label">F-Stat / Obs</div>
          <div class="metric-value">{f_stat:.2f}</div>
          <div class="metric-sub">p={f_p:.4f} · N={n}</div>
        </div>"""

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(_alpha_card, unsafe_allow_html=True)
        with c2: st.markdown(_r2_card,    unsafe_allow_html=True)
        with c3: st.markdown(_ir_card,    unsafe_allow_html=True)
        with c4: st.markdown(_f_card,     unsafe_allow_html=True)

        st.markdown('<div class="section-title">Factor Loadings</div>', unsafe_allow_html=True)
        st.markdown('<div class="factor-row header"><div>FACTOR</div><div>BETA</div><div>STD ERR</div><div>T-STAT</div><div>P-VALUE</div><div>SIG</div></div>', unsafe_allow_html=True)

        factor_rows_html = []
        for name in ["const"] + available:
            b = model.params[name]; se = model.bse[name]; t = model.tvalues[name]; p = model.pvalues[name]
            aqr_tag = ' <span class="aqr-badge">AQR</span>' if name in AQR_FACTOR_SET else ""
            display_name = f"{FACTOR_NAMES.get(name, name)}{aqr_tag}"
            row_h = f"""
            <div class="{row_class(name, p)}">
              <div style="color:#e2e8f0;font-weight:500">{display_name}</div>
              <div>{fmt_beta(b)}</div><div style="color:#9FE1CB">{se:.4f}</div>
              <div>{fmt_tstat(t)}</div><div>{fmt_pval(p)}</div><div>{sig_badge(p)}</div>
            </div>"""
            st.markdown(row_h, unsafe_allow_html=True)
            factor_rows_html.append(row_h)

        n_sig = sum(1 for f in available if model.pvalues[f] < 0.05)
        legend_html = (
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#1D9E75;margin-top:6px;">'
            f'★★★ p&lt;0.01 · ★★ p&lt;0.05 · ★ p&lt;0.10 · n.s. not significant'
            f' | Newey-West SE, maxlags={hac_lags}'
            f' | <span style="color:#34d399;">{n_sig} of {len(available)} factors significant at p&lt;0.05</span>'
            f' | <span style="color:#fbbf24;">AQR = Quality Minus Junk / Betting Against Beta</span>'
            f'</div>'
        )
        st.markdown(legend_html, unsafe_allow_html=True)

        # ── Benchmark regression + Active Exposure ──
        with st.spinner(f"Regressing benchmark {bench_ticker}…"):
            bench_result, bench_err = run_single_regression(
                bench_ticker, str(start_date)[:7], str(end_date)[:7], hac_lags,
                f"{start_date}_{end_date}_{bench_ticker}", tuple(available)
            )
        active_exposure_html = render_active_exposure_section(
            dict(model.params), bench_result, bench_err, available, bench_ticker, entity_label=ticker
        )
        st.markdown(active_exposure_html, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Factor Intelligence · Significant Loadings</div>', unsafe_allow_html=True)
        today_display = date.today().strftime("%B %d, %Y")
        ai_header_html = (
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:13px;color:#5DCAA5;margin-bottom:12px;">'
            f'Analysis restricted to statistically significant factors (p&lt;0.05) · {n_sig} factor(s) qualify · {today_display}'
            f'<br><span style="color:#1D9E75;">Live price · fundamentals · analyst consensus · short interest</span>'
            f'</div>'
        )
        st.markdown(ai_header_html, unsafe_allow_html=True)

        _ai_key = f"{ticker}_{start_date}_{end_date}_{hac_lags}_{annualize}_{'_'.join(sel_factors)}"
        if st.session_state.get("ai_run_key") != _ai_key:
            st.session_state["ai_run_key"] = _ai_key
            st.session_state["ai_insight"] = None
            st.session_state["ai_error"]   = None
            with st.spinner(f"Analyzing {n_sig} significant factor(s) for {ticker}..."):
                insight, error = get_ai_insight(ticker, model, available, alpha_ann, alpha_p, r2, n, start_date, end_date)
            st.session_state["ai_insight"] = insight
            st.session_state["ai_error"]   = error

        if st.session_state["ai_error"] and not st.session_state["ai_insight"]:
            st.markdown(f'<div class="error-box">Analysis Error: {st.session_state["ai_error"]}</div>', unsafe_allow_html=True)
        elif st.session_state["ai_error"] and st.session_state["ai_insight"]:
            st.markdown(f'<div class="error-box" style="border-color:rgba(251,191,36,0.3);color:#fbbf24;">⚠ Partial errors: {st.session_state["ai_error"]}</div>', unsafe_allow_html=True)
        if st.session_state["ai_insight"]:
            render_ai_insight(ticker, st.session_state["ai_insight"])

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["95% Confidence Intervals","Regression Diagnostics","Variance Inflation Factors","Model Fit","Rolling Market Beta"])

        ci = model.conf_int()
        ci_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:12px;">'
        for name in ["const"] + available:
            lo = ci.loc[name, 0]; hi = ci.loc[name, 1]; b = model.params[name]
            spans_zero = lo < 0 < hi
            bar_color  = "#34d399" if b > 0 and not spans_zero else "#f87171" if b < 0 and not spans_zero else "#6b7280"
            aqr_tag = ' <span class="aqr-badge">AQR</span>' if name in AQR_FACTOR_SET else ""
            ci_html += f'<div style="background:rgba(29,158,117,0.06);border:1px solid rgba(29,158,117,0.15);border-radius:10px;padding:14px 16px;min-width:140px;flex:1;"><div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#5DCAA5;margin-bottom:6px;">{FACTOR_NAMES.get(name, name)}{aqr_tag}</div><div style="font-family:\'JetBrains Mono\',monospace;font-size:14px;font-weight:600;color:{bar_color};">{b:+.4f}</div><div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#9FE1CB;margin-top:4px;">[{lo:+.4f}, {hi:+.4f}]</div><div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#1D9E75;margin-top:6px;">{"spans zero" if spans_zero else "excl. zero"}</div></div>'
        ci_html += '</div>'

        diag_html = f'<div style="height:12px"></div><div class="diag-grid">{diag_card("Durbin-Watson",f"{dw:.4f}","Autocorrelation · ideal ≈ 2.0",dw_status)}{diag_card("Breusch-Pagan",f"p = {bp_p:.4f}",f"Heteroscedasticity · stat={bp_stat:.3f}",bp_status)}{diag_card("Jarque-Bera",f"p = {jb_p:.4f}",f"Residual normality · stat={jb_stat:.3f}",jb_status)}{diag_card("Condition Number",f"{cond:.1f}","Multicollinearity · ideal < 30",cn_status)}</div>'

        vif_html = '<div style="height:12px"></div><div style="display:flex;flex-wrap:wrap;gap:10px;">'
        for col, v in vif_data.items():
            vc = "#34d399" if v < 5 else "#fbbf24" if v < 10 else "#f87171"
            aqr_tag = ' <span class="aqr-badge">AQR</span>' if col in AQR_FACTOR_SET else ""
            vif_html += f'<div style="background:rgba(29,158,117,0.06);border:1px solid rgba(29,158,117,0.15);border-radius:10px;padding:14px 16px;min-width:100px;"><div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#5DCAA5;margin-bottom:4px;">{FACTOR_NAMES.get(col, col)}{aqr_tag}</div><div style="font-family:\'JetBrains Mono\',monospace;font-size:18px;font-weight:600;color:{vc}">{v:.2f}</div><div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#9FE1CB;margin-top:2px;">{"OK" if v < 5 else "MODERATE" if v < 10 else "HIGH"}</div></div>'
        vif_html += '</div>'

        fit_html = f'<div style="height:12px"></div><div style="display:flex;flex-wrap:wrap;gap:10px;"><div class="diag-item" style="min-width:120px;"><div class="diag-name">AIC</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{aic:.2f}</div></div><div class="diag-item" style="min-width:120px;"><div class="diag-name">BIC</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{bic:.2f}</div></div><div class="diag-item" style="min-width:120px;"><div class="diag-name">Log-Likelihood</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{model.llf:.2f}</div></div><div class="diag-item" style="min-width:120px;"><div class="diag-name">Residual Std</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{resid.std():.4f}</div></div><div class="diag-item" style="min-width:120px;"><div class="diag-name">Skewness</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{float(stats.skew(resid)):.4f}</div></div><div class="diag-item" style="min-width:120px;"><div class="diag-name">Kurtosis</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{float(stats.kurtosis(resid)):.4f}</div></div></div>'

        rolling_html = None
        if "Mkt-RF" in available and len(data) >= 36:
            window = 24; roll_betas, roll_dates = [], []
            for i in range(window, len(data) + 1):
                sub = data.iloc[i - window: i]
                try:
                    rb = sm.OLS(sub["Y"], sm.add_constant(sub[["Mkt-RF"]])).fit().params["Mkt-RF"]
                    roll_betas.append(rb); roll_dates.append(str(data.index[i - 1]))
                except Exception:
                    pass
            if roll_betas:
                mn, mx = min(roll_betas), max(roll_betas); rng = mx - mn if mx != mn else 1
                W, H = 800, 140
                pts = " ".join(f"{int(i/(len(roll_betas)-1)*W) if len(roll_betas)>1 else W//2},{int(H-((b-mn)/rng)*H)}" for i, b in enumerate(roll_betas))
                one_y = int(H - ((1.0 - mn) / rng) * H)
                rolling_html = f'<div style="margin-top:12px;"><svg viewBox="0 0 {W} {H+30}" xmlns="http://www.w3.org/2000/svg" style="background:rgba(29,158,117,0.05);border:1px solid rgba(29,158,117,0.15);border-radius:10px;width:100%;margin-bottom:8px;"><line x1="0" y1="{one_y}" x2="{W}" y2="{one_y}" stroke="rgba(29,158,117,0.25)" stroke-width="1" stroke-dasharray="4,4"/><polyline points="{pts}" fill="none" stroke="#34d399" stroke-width="2"/><text x="6" y="{H+20}" fill="#1D9E75" font-family="JetBrains Mono,monospace" font-size="10">{roll_dates[0]}</text><text x="{W-6}" y="{H+20}" fill="#1D9E75" text-anchor="end" font-family="JetBrains Mono,monospace" font-size="10">{roll_dates[-1]}</text><text x="{W//2}" y="18" fill="#5DCAA5" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="10">Current β = {roll_betas[-1]:.3f}</text></svg></div>'

        with tab1: st.markdown(ci_html,   unsafe_allow_html=True)
        with tab2: st.markdown(diag_html, unsafe_allow_html=True)
        with tab3: st.markdown(vif_html,  unsafe_allow_html=True)
        with tab4: st.markdown(fit_html,  unsafe_allow_html=True)
        with tab5:
            if rolling_html: st.markdown(rolling_html, unsafe_allow_html=True)
            else: st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:13px;color:#5DCAA5;padding:20px 0;">Need at least 36 months and Mkt-RF selected.</div>', unsafe_allow_html=True)

        with st.expander("Full OLS Summary"):
            _render_ols_summary(model.summary().as_text())

        st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)
        export_df = pd.DataFrame({
            "Factor":   [FACTOR_NAMES.get(n, n) for n in model.params.index],
            "Beta":     model.params.values, "Std_Err": model.bse.values,
            "T_Stat":   model.tvalues.values, "P_Value": model.pvalues.values,
            "CI_Lower": model.conf_int()[0].values, "CI_Upper": model.conf_int()[1].values,
        })
        data_export = data.copy(); data_export.index = data_export.index.astype(str)
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("⬇  Download Results CSV", export_df.to_csv(index=False), file_name=f"{ticker}_factor_regression.csv", mime="text/csv", use_container_width=True)
        with col_b:
            st.download_button("⬇  Download Merged Data CSV", data_export.to_csv(), file_name=f"{ticker}_merged_data.csv", mime="text/csv", use_container_width=True)

        st.session_state["single_stock_cache"] = {
            "ticker": ticker, "price_html": _price_html, "regime_html": regime_html,
            "metric_alpha": _alpha_card, "metric_r2": _r2_card, "metric_ir": _ir_card, "metric_f": _f_card,
            "factor_rows": factor_rows_html, "factor_legend": legend_html,
            "active_exposure_html": active_exposure_html,
            "ai_header": ai_header_html, "ai_insight": st.session_state["ai_insight"], "ai_error": st.session_state["ai_error"],
            "ci_html": ci_html, "diag_html": diag_html, "vif_html": vif_html, "fit_html": fit_html,
            "rolling_html": rolling_html, "ols_summary": model.summary().as_text(),
            "export_csv": export_df.to_csv(index=False), "merged_csv": data_export.to_csv(),
        }
        st.session_state["run"] = False

    except Exception as e:
        st.markdown(f'<div class="error-box">Error: {str(e)}</div>', unsafe_allow_html=True)
        raise e
