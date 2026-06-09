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

st.set_page_config(page_title="Factor Regression", page_icon="📈", layout="wide")

# ── Background image (embedded as base64 so no external file needed) ──────────
def get_bg_base64():
    try:
        with open("ranger-4df6c1b6.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

_BG = get_bg_base64()
_BG_CSS = (
    f"""background-image:
        linear-gradient(rgba(15,22,35,0.78), rgba(15,22,35,0.78)),
        url("data:image/png;base64,{_BG}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;"""
    if _BG else
    """background: #0f1623;
    background-image: radial-gradient(ellipse at 20% 0%, rgba(56,189,248,0.04) 0%, transparent 60%),
                      radial-gradient(ellipse at 80% 100%, rgba(99,102,241,0.04) 0%, transparent 60%);"""
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
.metric-card {{
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-left: 2px solid #34d399; border-radius: 12px; padding: 16px 18px;
    margin-bottom: 12px; backdrop-filter: blur(8px); transition: border-color 0.2s;
}}
.metric-card:hover {{ border-color: rgba(255,255,255,0.14); }}
.metric-card.red  {{ border-left-color: #f87171; }}
.metric-card.blue {{ border-left-color: #60a5fa; }}
.metric-card.gold {{ border-left-color: #fbbf24; }}
.metric-card.gray {{ border-left-color: #6b7280; }}
.metric-label {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 2px; color: #475569; text-transform: uppercase; margin-bottom: 4px; }}
.metric-value {{ font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 600; color: #e2e8f0; }}
.metric-sub {{ font-size: 11px; color: #334155; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }}
.factor-row {{
    display: grid; grid-template-columns: 140px 90px 90px 90px 80px 1fr;
    gap: 8px; align-items: center; padding: 10px 14px; border-radius: 8px;
    margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; font-size: 12px;
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06); transition: background 0.15s;
}}
.factor-row:hover {{ background: rgba(255,255,255,0.04); }}
.factor-row.header {{ background: transparent; border-color: transparent; font-size: 10px; letter-spacing: 1.5px; color: #334155; text-transform: uppercase; }}
.factor-row.sig   {{ border-left: 2px solid #34d399; }}
.factor-row.marg  {{ border-left: 2px solid #fbbf24; }}
.factor-row.insig {{ border-left: 2px solid rgba(255,255,255,0.08); }}
.factor-row.alpha {{ border-left: 2px solid #a78bfa; }}
.sig-badge {{ display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 10px; font-weight: 500; letter-spacing: 1px; }}
.badge-001 {{ background: rgba(52,211,153,0.12); color: #34d399; }}
.badge-01  {{ background: rgba(52,211,153,0.08); color: #6ee7b7; }}
.badge-05  {{ background: rgba(251,191,36,0.1);  color: #fbbf24; }}
.badge-10  {{ background: rgba(255,255,255,0.05); color: #6b7280; }}
.badge-ns  {{ background: rgba(255,255,255,0.03); color: #334155; }}
.section-title {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 2.5px;
    text-transform: uppercase; color: #334155; border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 8px; margin: 28px 0 16px 0;
}}
.diag-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
.diag-item {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 14px 16px; }}
.diag-name {{ font-size: 11px; color: #475569; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; }}
.diag-val  {{ font-size: 16px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }}
.diag-pass {{ color: #34d399; }} .diag-fail {{ color: #f87171; }} .diag-warn {{ color: #fbbf24; }}
.diag-sub  {{ font-size: 10px; color: #1e293b; font-family: 'JetBrains Mono', monospace; margin-top: 3px; }}
.interpret-box {{ background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 20px 24px; font-size: 13px; line-height: 1.8; color: #94a3b8; }}
.interpret-box b {{ color: #e2e8f0; }}
.ai-box {{
    background: rgba(167,139,250,0.04);
    border: 1px solid rgba(167,139,250,0.15);
    border-radius: 12px;
    padding: 24px 28px;
    font-size: 15px;
    line-height: 1.9;
    color: #c4b5fd;
}}
.ai-box b {{ color: #e2e8f0; }}
.ai-box h4 {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 2px;
    text-transform: uppercase; color: #7c3aed; margin-bottom: 16px;
}}
.factor-insight-card {{
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}}
.factor-insight-card.positive {{ border-left: 3px solid #34d399; }}
.factor-insight-card.negative {{ border-left: 3px solid #f87171; }}
.factor-insight-card.neutral  {{ border-left: 3px solid #6b7280; }}
.fi-header {{
    display: flex; align-items: center; gap: 12px; margin-bottom: 10px;
}}
.fi-name {{
    font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600;
    color: #e2e8f0; letter-spacing: 1px;
}}
.fi-beta {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    padding: 2px 8px; border-radius: 4px;
}}
.fi-beta.pos {{ background: rgba(52,211,153,0.12); color: #34d399; }}
.fi-beta.neg {{ background: rgba(248,113,113,0.12); color: #f87171; }}
.fi-sig-label {{
    font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 1.5px;
    color: #475569; text-transform: uppercase;
}}
.fi-outlook {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase; margin-left: auto;
}}
.fi-outlook.bullish  {{ background: rgba(52,211,153,0.1); color: #34d399; }}
.fi-outlook.bearish  {{ background: rgba(248,113,113,0.1); color: #f87171; }}
.fi-outlook.neutral  {{ background: rgba(107,114,128,0.1); color: #6b7280; }}
.fi-outlook.mixed    {{ background: rgba(251,191,36,0.1);  color: #fbbf24; }}
.fi-body {{ font-size: 14px; color: #94a3b8; line-height: 1.85; }}
.fi-news {{
    margin-top: 10px; padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-size: 13px; color: #475569; font-family: 'JetBrains Mono', monospace;
}}
.fi-news-label {{
    font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: #334155; margin-bottom: 4px;
}}
.ai-summary-box {{
    background: rgba(96,165,250,0.04);
    border: 1px solid rgba(96,165,250,0.12);
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 16px;
    font-size: 14px;
    color: #93c5fd;
    line-height: 1.8;
}}
.ai-summary-box b {{ color: #e2e8f0; }}
.stTextInput > div > div > input {{ background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #e2e8f0 !important; font-family: 'JetBrains Mono', monospace !important; border-radius: 8px !important; }}
.stTextInput > div > div > input:focus {{ border-color: rgba(96,165,250,0.4) !important; box-shadow: 0 0 0 3px rgba(96,165,250,0.08) !important; }}
.stButton > button {{ background: rgba(255,255,255,0.06) !important; color: #e2e8f0 !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 500 !important; letter-spacing: 1px !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 8px !important; padding: 10px 28px !important; transition: all 0.2s !important; }}
.stButton > button:hover {{ background: rgba(52,211,153,0.1) !important; border-color: rgba(52,211,153,0.3) !important; color: #34d399 !important; }}
.error-box {{ background: rgba(248,113,113,0.06); border: 1px solid rgba(248,113,113,0.2); border-radius: 8px; padding: 12px 16px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #f87171; }}

/* ── Portfolio Attribution Styles ── */
.port-header {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 2.5px;
    text-transform: uppercase; color: #334155; border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 8px; margin: 28px 0 16px 0;
}}
.port-stock-card {{
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
}}
.port-stock-ticker {{
    font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700;
    color: #e2e8f0; letter-spacing: 1px;
}}
.port-weight-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    background: rgba(96,165,250,0.1); color: #60a5fa; margin-left: 8px;
}}
.port-factor-bar-container {{
    display: flex; align-items: center; gap: 10px; margin: 3px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
}}
.port-factor-label {{ color: #475569; width: 110px; flex-shrink: 0; }}
.port-bar-wrap {{
    flex: 1; height: 8px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden;
}}
.port-bar-pos {{ height: 100%; background: #34d399; border-radius: 4px; }}
.port-bar-neg {{ height: 100%; background: #f87171; border-radius: 4px; }}
.port-bar-val {{ width: 60px; text-align: right; }}
.port-summary-card {{
    background: linear-gradient(135deg, rgba(96,165,250,0.06), rgba(167,139,250,0.06));
    border: 1px solid rgba(96,165,250,0.15); border-radius: 12px; padding: 20px 24px;
    margin-bottom: 16px;
}}
.port-attribution-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-top: 12px;
}}
.port-attr-cell {{
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px; padding: 12px 14px;
}}
.port-attr-factor {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #475569; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px; }}
.port-attr-beta {{ font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 600; }}
.port-attr-beta.pos {{ color: #34d399; }}
.port-attr-beta.neg {{ color: #f87171; }}
.port-compare-row {{
    display: grid; gap: 8px; align-items: center; padding: 8px 12px;
    border-radius: 8px; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace;
    font-size: 11px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05);
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

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
    else: color = "#475569"
    return f'<span style="color:{color}">{p:.4f}</span>'

def fmt_tstat(t):
    color = "#34d399" if abs(t) > 1.96 else "#475569"
    return f'<span style="color:{color}">{t:+.3f}</span>'


FACTOR_NAMES = {
    "const":  "Alpha (α)",
    "Mkt-RF": "Market (β)",
    "SMB":    "Size",
    "HML":    "Value",
    "RMW":    "Profitability",
    "CMA":    "Investment",
    "Mom":    "Momentum",
}

FACTOR_DESCRIPTIONS = {
    "Mkt-RF": "Market risk premium (excess return of market over risk-free rate)",
    "SMB":    "Size factor — Small Minus Big (small-cap vs large-cap premium)",
    "HML":    "Value factor — High Minus Low (value vs growth premium)",
    "RMW":    "Profitability factor — Robust Minus Weak (profitable vs unprofitable firms)",
    "CMA":    "Investment factor — Conservative Minus Aggressive (low vs high investment firms)",
    "Mom":    "Momentum factor (past winners vs past losers)",
}

FACTOR_COLORS = {
    "Mkt-RF": "#60a5fa",
    "SMB":    "#34d399",
    "HML":    "#fbbf24",
    "RMW":    "#a78bfa",
    "CMA":    "#f97316",
    "Mom":    "#ec4899",
    "const":  "#94a3b8",
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
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$",     "", raw)
    return json.loads(raw)


def _get_stock_context(ticker):
    ctx = {}
    try:
        t_obj = yf.Ticker(ticker)
        info = t_obj.info
        ctx["current_price"]   = info.get("currentPrice") or info.get("regularMarketPrice")
        ctx["market_cap"]      = info.get("marketCap")
        ctx["sector"]          = info.get("sector", "Unknown")
        ctx["industry"]        = info.get("industry", "Unknown")
        ctx["pe_ratio"]        = info.get("trailingPE")
        ctx["pb_ratio"]        = info.get("priceToBook")
        ctx["ps_ratio"]        = info.get("priceToSalesTrailing12Months")
        ctx["revenue_growth"]  = info.get("revenueGrowth")
        ctx["earnings_growth"] = info.get("earningsGrowth")
        ctx["profit_margin"]   = info.get("profitMargins")
        ctx["gross_margin"]    = info.get("grossMargins")
        ctx["roe"]             = info.get("returnOnEquity")
        ctx["debt_to_equity"]  = info.get("debtToEquity")
        ctx["free_cashflow"]   = info.get("freeCashflow")
        ctx["dividend_yield"]  = info.get("dividendYield")
        ctx["beta_1y"]         = info.get("beta")
        ctx["52w_high"]        = info.get("fiftyTwoWeekHigh")
        ctx["52w_low"]         = info.get("fiftyTwoWeekLow")
        ctx["short_ratio"]     = info.get("shortRatio")
        ctx["short_pct_float"] = info.get("shortPercentOfFloat")
        ctx["analyst_target"]  = info.get("targetMeanPrice")
        ctx["rec_mean"]        = info.get("recommendationMean")
        ctx["num_analysts"]    = info.get("numberOfAnalystOpinions")
        ctx["forward_pe"]      = info.get("forwardPE")
        ctx["peg_ratio"]       = info.get("pegRatio")
        ctx["enterprise_ev_ebitda"] = info.get("enterpriseToEbitda")
        ctx["held_pct_inst"]   = info.get("heldPercentInstitutions")
        ctx["company_name"]    = info.get("longName", ticker)
        hist = t_obj.history(period="ytd")
        if not hist.empty and len(hist) >= 2:
            ctx["ytd_return"] = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1
        hist_1m = t_obj.history(period="1mo")
        if not hist_1m.empty and len(hist_1m) >= 2:
            ctx["return_1m"] = (hist_1m["Close"].iloc[-1] / hist_1m["Close"].iloc[0]) - 1
        hist_3m = t_obj.history(period="3mo")
        if not hist_3m.empty and len(hist_3m) >= 2:
            ctx["return_3m"] = (hist_3m["Close"].iloc[-1] / hist_3m["Close"].iloc[0]) - 1
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
        lines.append(f"52-Week Range: ${ctx['52w_low']:.2f} – ${ctx['52w_high']:.2f}")
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
        rec_label = rec_map.get(round(ctx['rec_mean']), f"{ctx['rec_mean']:.1f}")
        lines.append(f"Consensus Rating: {rec_label} ({ctx['rec_mean']:.2f}/5)")
    if ctx.get("held_pct_inst") is not None: lines.append(f"Institutional Ownership: {ctx['held_pct_inst']:.1%}")
    return "\n".join(lines)


def _factor_prompt(ticker, factor_code, factor_name, beta, p_value, significant,
                   description, ctx_str):
    today = date.today().strftime("%B %d, %Y")
    system = (
        "You are a senior quantitative analyst at a top-tier hedge fund. "
        "You have been given LIVE fundamental and price data for the stock. Use it. "
        "Your analysis must be SPECIFIC to this exact ticker — reference its actual numbers. "
        "Never write generic statements like 'interest rates affect valuations'. "
        "For macro context, use your knowledge of the current environment as of your training data "
        "(Fed funds rate, 10Y yield, CPI, VIX, sector performance). "
        "Be concrete: cite actual figures (e.g. 'P/B of 3.2x vs HML threshold ~1x means growth tilt', "
        "'net margin of 22% supports RMW', '52-week momentum of +47% drives high Mom beta'). "
        "You MUST respond with a single valid JSON object and nothing else. "
        "No markdown, no prose outside the JSON."
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

=== YOUR TASK ===
Use the live data above to give a SPECIFIC, NUMBER-DRIVEN analysis.
Cross-reference the actual fundamentals with what this beta implies.
For example:
- For SMB: compare actual market cap to small/large-cap thresholds
- For HML: compare actual P/B ratio to value thresholds
- For RMW: use actual margin/ROE data to explain the loading
- For Momentum: use actual 1M/3M/YTD returns to explain the loading
- For Market: compare regression beta to Yahoo's 1Y beta

Return EXACTLY this JSON (fill every field, no line breaks inside strings):
{{
  "code": "{factor_code}",
  "name": "{factor_name}",
  "beta": {beta},
  "significant": {"true" if significant else "false"},
  "outlook": "<bullish|bearish|neutral|mixed>",
  "what_it_means": "<2-3 sentences using ACTUAL numbers from the data above.>",
  "current_macro_context": "<2-3 sentences on the current macro environment relevant to THIS factor.>",
  "recent_stock_news": "<2-3 sentences on the most recent material developments for {ticker}.>",
  "forward_forecast": "<2 sentences: near-term directional call for {ticker} on this factor.>",
  "key_risks": "<1-2 sentences: the single most specific risk that could flip this outlook.>"
}}"""
    return system, user


def _summary_prompt(ticker, alpha_ann, alpha_p, r2, factor_summaries, ctx_str):
    today = date.today().strftime("%B %d, %Y")
    system = (
        "You are a senior portfolio strategist at a multi-billion dollar fund. "
        "You have been given live fundamental data. Use specific numbers in your analysis. "
        "Respond with a single valid JSON object only. No markdown."
    )
    sig_word = "statistically significant" if alpha_p < 0.05 else "not statistically significant"
    lines = "\n".join(
        f"- {f['name']}: beta={f['beta']:+.4f}, outlook={f.get('outlook','?')}, "
        f"sig={'yes' if f.get('significant') else 'no'}"
        for f in factor_summaries
    )

    user = f"""Today: {today}. Stock: {ticker}
Annualized alpha: {alpha_ann:+.2%} (p={alpha_p:.4f}, {sig_word})
R²: {r2:.4f}
Factor loadings:
{lines}

=== LIVE STOCK DATA ===
{ctx_str}

Return EXACTLY this JSON (2-4 sentences per field, use specific numbers from the data):
{{
  "alpha_analysis": "<Is alpha={alpha_ann:+.2%} meaningful for {ticker}? Compare to its sector. What specific operational advantage explains it? Reference actual numbers. What threatens persistence?>",
  "positioning_context": "<Based on the live data: analyst consensus, institutional ownership, short interest, and how the factor profile aligns with consensus positioning.>",
  "portfolio_verdict": "<Risk-adjusted verdict as of {today}. Identify biggest factor risk. Best-case and worst-case scenarios with specific triggers. One-line bottom line: buy/hold/avoid.>"
}}"""
    return system, user


def get_ai_insight(ticker, model_result, available, alpha_ann, alpha_p, r2, n, start_date, end_date):
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        return None, "No API key found. Add `GROQ_API_KEY` to your `.streamlit/secrets.toml` file."

    ctx = _get_stock_context(ticker)
    ctx_str = _fmt_ctx(ctx)

    try:
        client = Groq(api_key=api_key)
        factors_out = []
        errors = []

        for f in available:
            beta = float(model_result.params[f])
            pval = float(model_result.pvalues[f])
            sig  = pval < 0.05
            sys_msg, usr_msg = _factor_prompt(
                ticker, f, FACTOR_NAMES.get(f, f), beta, pval, sig,
                FACTOR_DESCRIPTIONS.get(f, f),
                ctx_str=ctx_str,
            )
            try:
                result = _call_groq(client, sys_msg, usr_msg, max_tokens=900)
                result["beta"] = beta
                result["significant"] = sig
                factors_out.append(result)
            except Exception as e:
                factors_out.append({
                    "code": f, "name": FACTOR_NAMES.get(f, f),
                    "beta": beta, "significant": sig, "outlook": "neutral",
                    "what_it_means": "Analysis unavailable.",
                    "current_macro_context": f"Error: {e}",
                    "recent_stock_news": "",
                    "forward_forecast": "", "key_risks": "",
                })
                errors.append(f"{f}: {e}")

        sys_msg, usr_msg = _summary_prompt(ticker, alpha_ann, alpha_p, r2, factors_out, ctx_str)
        try:
            summary = _call_groq(client, sys_msg, usr_msg, max_tokens=700)
        except Exception as e:
            summary = {
                "alpha_analysis": f"Summary unavailable: {e}",
                "positioning_context": "",
                "portfolio_verdict": "",
            }
            errors.append(f"summary: {e}")

        result = {"factors": factors_out, **summary}
        err_msg = "; ".join(errors) if errors else None
        return result, err_msg
    except Exception as e:
        return None, str(e)


def render_ai_insight(ticker, insight_data):
    factors = insight_data.get("factors", [])
    alpha_analysis      = insight_data.get("alpha_analysis", "")
    positioning_context = insight_data.get("positioning_context", "")
    portfolio_verdict   = insight_data.get("portfolio_verdict", "")

    def bold(text):
        return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', str(text))

    today_str = date.today().strftime("%B %d, %Y")
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;'
        f'text-transform:uppercase;color:#7c3aed;margin-bottom:16px;">'
        f'✦ AI Factor Intelligence · {ticker} · As of {today_str}</div>',
        unsafe_allow_html=True,
    )
    for fac in factors:
        code    = fac.get("code", "")
        name    = fac.get("name", code)
        beta    = fac.get("beta", 0)
        sig     = fac.get("significant", False)
        outlook = fac.get("outlook", "neutral").lower()
        card_cls   = "positive" if beta > 0 else "negative"
        beta_cls   = "pos" if beta > 0 else "neg"
        outlook_cls  = {"bullish": "bullish", "bearish": "bearish", "neutral": "neutral", "mixed": "mixed"}.get(outlook, "neutral")
        outlook_icon = {"bullish": "↑ BULLISH", "bearish": "↓ BEARISH", "neutral": "→ NEUTRAL", "mixed": "⇅ MIXED"}.get(outlook, "→ NEUTRAL")
        sig_label = "SIGNIFICANT" if sig else "INSIGNIFICANT"
        what     = bold(fac.get("what_it_means", ""))
        macro    = bold(fac.get("current_macro_context", ""))
        news     = bold(fac.get("recent_stock_news", ""))
        forecast = bold(fac.get("forward_forecast", ""))
        risks    = bold(fac.get("key_risks", ""))
        st.markdown(f"""
        <div class="factor-insight-card {card_cls}">
          <div class="fi-header">
            <div class="fi-name">{name}</div>
            <div class="fi-beta {beta_cls}">{beta:+.4f}</div>
            <div class="fi-sig-label">{sig_label}</div>
            <div class="fi-outlook {outlook_cls}">{outlook_icon}</div>
          </div>
          <div class="fi-body">
            <div style="margin-bottom:8px;">
              <span style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#475569;">WHAT IT MEANS FOR {ticker}</span>
              <div style="margin-top:4px;">{what}</div>
            </div>
            <div style="margin-bottom:8px;">
              <span style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#475569;">CURRENT MACRO CONTEXT</span>
              <div style="margin-top:4px;">{macro}</div>
            </div>
            {"" if not news else f'''<div style="margin-bottom:8px;">
              <span style="font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#475569;">RECENT NEWS &amp; CATALYSTS</span>
              <div style="margin-top:4px;">{news}</div>
            </div>'''}
            <div style="margin-bottom:8px;">
              <span style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#475569;">FORWARD FORECAST</span>
              <div style="margin-top:4px;">{forecast}</div>
            </div>
            <div class="fi-news">
              <div class="fi-news-label">⚠ KEY RISK</div>
              {risks}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    if alpha_analysis:
        st.markdown(f"""
        <div class="ai-summary-box">
          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#3b82f6;margin-bottom:8px;">ALPHA PERSISTENCE ANALYSIS</div>
          {bold(alpha_analysis)}
        </div>""", unsafe_allow_html=True)

    if positioning_context:
        st.markdown(f"""
        <div class="ai-summary-box" style="border-color:rgba(52,211,153,0.15);color:#6ee7b7;background:rgba(52,211,153,0.03);">
          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#059669;margin-bottom:8px;">POSITIONING & SENTIMENT</div>
          {bold(positioning_context)}
        </div>""", unsafe_allow_html=True)

    if portfolio_verdict:
        st.markdown(f"""
        <div class="ai-summary-box" style="border-color:rgba(167,139,250,0.15);color:#c4b5fd;background:rgba(167,139,250,0.04);">
          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#7c3aed;margin-bottom:8px;">PORTFOLIO VERDICT</div>
          {bold(portfolio_verdict)}
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Portfolio Attribution Functions
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_single_regression(ticker_sym, start_str, end_str, hac_lags, ff_key):
    try:
        ff_raw = load_factors()
        ff = ff_raw.loc[start_str:end_str].copy()
        available = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"] if c in ff.columns]
        has_rf = "RF" in ff.columns
        ff[available] = ff[available].astype(float) / 100
        if has_rf:
            ff["RF"] = ff["RF"].astype(float) / 100

        raw = yf.download(ticker_sym, start=start_str[:7] + "-01",
                          end=end_str[:7] + "-28", auto_adjust=True, progress=False)
        if raw.empty:
            return None, f"No data for {ticker_sym}"

        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        monthly = close.resample("ME").last()
        returns = monthly.pct_change().dropna()
        returns.index = returns.index.to_period("M")

        data = pd.DataFrame({"Stock": returns}).join(
            ff[available + (["RF"] if has_rf else [])], how="inner"
        )
        if len(data) < 12:
            return None, f"Too few obs for {ticker_sym} ({len(data)})"

        if has_rf:
            data["Y"] = data["Stock"] - data["RF"]
        else:
            data["Y"] = data["Stock"]

        X = sm.add_constant(data[available])
        y = data["Y"]
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

        alpha = model.params["const"]
        alpha_ann = (1 + alpha) ** 12 - 1
        result = {
            "ticker":     ticker_sym,
            "params":     dict(model.params),
            "pvalues":    dict(model.pvalues),
            "r2":         model.rsquared,
            "alpha_ann":  alpha_ann,
            "available":  available,
            "nobs":       int(model.nobs),
        }
        return result, None
    except Exception as e:
        return None, str(e)


def render_portfolio_attribution(port_results, weights, available_factors):
    tickers = list(port_results.keys())
    n_stocks = len(tickers)

    port_betas = {f: 0.0 for f in ["const"] + available_factors}
    for tkr, w in weights.items():
        if tkr in port_results:
            for f in ["const"] + available_factors:
                port_betas[f] += w * port_results[tkr]["params"].get(f, 0.0)

    port_alpha_ann = sum(
        weights[tkr] * port_results[tkr]["alpha_ann"]
        for tkr in tickers if tkr in port_results
    )
    port_r2_avg = np.mean([port_results[t]["r2"] for t in tickers])

    alpha_color = "#34d399" if port_alpha_ann > 0 else "#f87171"
    st.markdown(f"""
    <div class="port-summary-card">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2px;
                  text-transform:uppercase;color:#60a5fa;margin-bottom:14px;">
        ◈ Portfolio-Level Factor Attribution · {n_stocks} Holdings
      </div>
      <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:baseline;">
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#475569;margin-bottom:2px;">WTDAVG ALPHA (ANN)</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:700;color:{alpha_color};">{port_alpha_ann:+.2%}</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#475569;margin-bottom:2px;">AVG R²</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:700;color:#60a5fa;">{port_r2_avg:.4f}</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#475569;margin-bottom:2px;">MKT BETA (PORT)</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:700;color:#a78bfa;">{port_betas.get("Mkt-RF", 0):+.3f}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;'
        'text-transform:uppercase;color:#475569;margin-bottom:10px;">PORTFOLIO WEIGHTED FACTOR EXPOSURES</div>',
        unsafe_allow_html=True
    )
    grid_html = '<div class="port-attribution-grid">'
    for f in available_factors:
        b = port_betas.get(f, 0)
        bcls = "pos" if b >= 0 else "neg"
        grid_html += f"""
        <div class="port-attr-cell">
          <div class="port-attr-factor">{FACTOR_NAMES.get(f, f)}</div>
          <div class="port-attr-beta {bcls}">{b:+.4f}</div>
        </div>"""
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;'
        'text-transform:uppercase;color:#475569;margin:20px 0 10px 0;">FACTOR LOADING COMPARISON</div>',
        unsafe_allow_html=True
    )

    all_betas_flat = []
    for tkr in tickers:
        for f in available_factors:
            all_betas_flat.append(abs(port_results[tkr]["params"].get(f, 0)))
    max_abs = max(all_betas_flat) if all_betas_flat else 1

    grid_cols = f"80px 60px " + " ".join(["1fr"] * len(available_factors)) + " 70px 60px"
    rows_html = f"""
    <div style="overflow-x:auto;">
    <div style="display:grid;grid-template-columns:{grid_cols};gap:6px;min-width:700px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;color:#334155;text-transform:uppercase;padding:4px 0;">TICKER</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;color:#334155;text-transform:uppercase;padding:4px 0;">WT</div>
    """
    for f in available_factors:
        fc = FACTOR_COLORS.get(f, "#94a3b8")
        rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;letter-spacing:1.5px;color:{fc};text-transform:uppercase;padding:4px 0;">{FACTOR_NAMES.get(f,f)}</div>'
    rows_html += '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;letter-spacing:1.5px;color:#a78bfa;text-transform:uppercase;padding:4px 0;">ANN α</div>'
    rows_html += '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;letter-spacing:1.5px;color:#60a5fa;text-transform:uppercase;padding:4px 0;">R²</div>'

    for tkr in tickers:
        res = port_results[tkr]
        w = weights[tkr]
        alpha_c = "#34d399" if res["alpha_ann"] > 0 else "#f87171"
        rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;font-weight:700;color:#e2e8f0;padding:8px 0;border-top:1px solid rgba(255,255,255,0.05);">{tkr}</div>'
        rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#60a5fa;padding:8px 0;border-top:1px solid rgba(255,255,255,0.05);">{w:.0%}</div>'
        for f in available_factors:
            b = res["params"].get(f, 0)
            bc = "#34d399" if b >= 0 else "#f87171"
            pct = min(abs(b) / max_abs * 100, 100) if max_abs > 0 else 0
            bar_color = "#34d399" if b >= 0 else "#f87171"
            rows_html += f"""
            <div style="padding:8px 0;border-top:1px solid rgba(255,255,255,0.05);">
              <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:{bc};">{b:+.3f}</div>
              <div style="height:3px;background:rgba(255,255,255,0.05);border-radius:2px;margin-top:3px;">
                <div style="width:{pct:.1f}%;height:100%;background:{bar_color};border-radius:2px;"></div>
              </div>
            </div>"""
        rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:{alpha_c};padding:8px 0;border-top:1px solid rgba(255,255,255,0.05);">{res["alpha_ann"]:+.2%}</div>'
        rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#60a5fa;padding:8px 0;border-top:1px solid rgba(255,255,255,0.05);">{res["r2"]:.3f}</div>'

    # Portfolio row
    rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;font-weight:700;color:#a78bfa;padding:8px 0;border-top:2px solid rgba(167,139,250,0.2);">PORTFOLIO</div>'
    rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#a78bfa;padding:8px 0;border-top:2px solid rgba(167,139,250,0.2);">100%</div>'
    for f in available_factors:
        b = port_betas.get(f, 0)
        bc = "#34d399" if b >= 0 else "#f87171"
        pct = min(abs(b) / max_abs * 100, 100) if max_abs > 0 else 0
        bar_color = "#34d399" if b >= 0 else "#f87171"
        rows_html += f"""
        <div style="padding:8px 0;border-top:2px solid rgba(167,139,250,0.2);">
          <div style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;color:{bc};">{b:+.3f}</div>
          <div style="height:3px;background:rgba(255,255,255,0.05);border-radius:2px;margin-top:3px;">
            <div style="width:{pct:.1f}%;height:100%;background:{bar_color};border-radius:2px;"></div>
          </div>
        </div>"""
    alpha_c = "#34d399" if port_alpha_ann > 0 else "#f87171"
    rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;font-weight:600;color:{alpha_c};padding:8px 0;border-top:2px solid rgba(167,139,250,0.2);">{port_alpha_ann:+.2%}</div>'
    rows_html += f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#60a5fa;padding:8px 0;border-top:2px solid rgba(167,139,250,0.2);">{port_r2_avg:.3f}</div>'
    rows_html += '</div></div>'

    st.markdown(
        f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);'
        f'border-radius:10px;padding:16px 18px;">{rows_html}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;'
        'text-transform:uppercase;color:#475569;margin:20px 0 10px 0;">FACTOR CONCENTRATION RISK</div>',
        unsafe_allow_html=True
    )
    risk_html = '<div style="display:flex;gap:10px;flex-wrap:wrap;">'
    for f in available_factors:
        vals = [port_results[t]["params"].get(f, 0) for t in tickers]
        dispersion = np.std(vals)
        risk_color = "#f87171" if dispersion > 0.5 else "#fbbf24" if dispersion > 0.2 else "#34d399"
        risk_label = "HIGH" if dispersion > 0.5 else "MODERATE" if dispersion > 0.2 else "LOW"
        risk_html += f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                    border-radius:8px;padding:12px 14px;min-width:110px;flex:1;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#475569;
                      margin-bottom:4px;text-transform:uppercase;">{FACTOR_NAMES.get(f,f)}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;
                      color:{risk_color};">{risk_label}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#334155;margin-top:2px;">
            σ = {dispersion:.3f}
          </div>
        </div>"""
    risk_html += '</div>'
    st.markdown(risk_html, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;'
        'text-transform:uppercase;color:#475569;margin:20px 0 10px 0;">EXPORT</div>',
        unsafe_allow_html=True
    )
    rows = []
    for tkr in tickers:
        res = port_results[tkr]
        row = {"Ticker": tkr, "Weight": weights[tkr], "Ann_Alpha": res["alpha_ann"], "R2": res["r2"]}
        for f in available_factors:
            row[FACTOR_NAMES.get(f, f)] = res["params"].get(f, 0)
        rows.append(row)
    port_row = {"Ticker": "PORTFOLIO", "Weight": 1.0, "Ann_Alpha": port_alpha_ann, "R2": port_r2_avg}
    for f in available_factors:
        port_row[FACTOR_NAMES.get(f, f)] = port_betas.get(f, 0)
    rows.append(port_row)
    export_df = pd.DataFrame(rows)
    st.download_button(
        "⬇  Download Portfolio Attribution CSV",
        export_df.to_csv(index=False),
        file_name="portfolio_factor_attribution.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════
#  SESSION STATE INIT
# ════════════════════════════════════════════

if "run" not in st.session_state:
    st.session_state["run"] = False
if "ai_insight" not in st.session_state:
    st.session_state["ai_insight"] = None
if "ai_error" not in st.session_state:
    st.session_state["ai_error"] = None
if "port_run" not in st.session_state:
    st.session_state["port_run"] = False
if "port_results" not in st.session_state:
    st.session_state["port_results"] = None
# ── FIX: store results for BOTH modes independently so switching tabs
#    does not wipe the analysis you already ran.
if "single_stock_cache" not in st.session_state:
    st.session_state["single_stock_cache"] = None   # holds full single-stock render data
if "portfolio_cache" not in st.session_state:
    st.session_state["portfolio_cache"] = None       # holds full portfolio render data
if "active_mode" not in st.session_state:
    st.session_state["active_mode"] = "Single Stock"


# ════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════

st.markdown("# Factor Regression")
st.markdown(
    '<p style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:12px;letter-spacing:1px;">'
    'FF5 + MOMENTUM · HAC ROBUST STANDARD ERRORS · DATA AUTO-LOADED FROM GITHUB'
    '</p>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### Configuration")

    mode = st.radio(
        "Mode",
        ["Single Stock", "Portfolio Attribution"],
        horizontal=True,
        key="mode_radio",
    )

    # ── FIX: switching mode no longer clears results — each mode keeps its own cache
    if mode != st.session_state["active_mode"]:
        st.session_state["active_mode"] = mode
        st.rerun()

    st.markdown("---")

    if mode == "Single Stock":
        ticker = st.text_input("Stock Ticker", "AVGO").upper().strip()
    else:
        st.markdown("**Portfolio Holdings**")
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#475569;margin-bottom:10px;">'
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
                    key=f"port_ticker_{idx}",
                    label_visibility="collapsed",
                    placeholder="TICKER"
                ).upper().strip()
                st.session_state["port_holdings"][idx]["ticker"] = new_ticker
            with col_a:
                new_amount = st.number_input(
                    "Amount", value=float(holding["amount"]),
                    min_value=0.0, step=1000.0,
                    key=f"port_amount_{idx}",
                    label_visibility="collapsed",
                    format="%.0f"
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
            preview_lines = []
            for h in st.session_state["port_holdings"]:
                if h["ticker"] and h["amount"] > 0:
                    w = h["amount"] / total_amt
                    preview_lines.append(f'{h["ticker"]} · {w:.1%}')
            if preview_lines:
                st.markdown(
                    '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
                    'color:#475569;margin-top:8px;padding:8px 10px;'
                    'background:rgba(255,255,255,0.02);border-radius:6px;'
                    'border:1px solid rgba(255,255,255,0.05);line-height:1.8;">'
                    + " &nbsp;|&nbsp; ".join(preview_lines) +
                    '</div>',
                    unsafe_allow_html=True
                )

    st.markdown("**Date Range**")
    start_date = st.date_input("Start", value=pd.to_datetime("2010-01-01"))
    end_date   = st.date_input("End",   value=pd.to_datetime("2026-04-30"))
    st.markdown("**Regression Settings**")
    hac_lags  = st.slider("HAC Max Lags", 1, 12, 3, help="Newey-West lags for HAC robust SE")
    annualize = st.selectbox("Annualization", [12, 52, 252], help="12=monthly, 52=weekly, 252=daily")

    if mode == "Single Stock":
        if st.button("▶  RUN REGRESSION", use_container_width=True):
            st.session_state["run"]        = True
            st.session_state["ticker_ran"] = ticker
            st.session_state["start_ran"]  = start_date
            st.session_state["end_ran"]    = end_date
            st.session_state["hac_ran"]    = hac_lags
            st.session_state["ann_ran"]    = annualize
            st.session_state["ai_insight"] = None
            st.session_state["ai_error"]   = None
            st.session_state["single_stock_cache"] = None   # force re-run
    else:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("▶  RUN PORTFOLIO ATTRIBUTION", use_container_width=True):
            holdings = st.session_state.get("port_holdings", [])
            parsed = [
                (h["ticker"], h["amount"])
                for h in holdings
                if h["ticker"] and h["amount"] > 0
            ]
            if parsed:
                total_w = sum(w for _, w in parsed)
                normalized = {tkr: w / total_w for tkr, w in parsed}
                st.session_state["port_weights"] = normalized
                st.session_state["port_start"]   = start_date
                st.session_state["port_end"]      = end_date
                st.session_state["port_hac"]      = hac_lags
                st.session_state["port_run"]      = True
                st.session_state["port_results"]  = None
                st.session_state["portfolio_cache"] = None  # force re-run

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#334155;">'
        'Factor data auto-loaded from:<br>'
        '<span style="color:#475569;">github.com/lakshyaworkch-cell/Lakshy</span>'
        '</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════
#  MAIN AREA ROUTING
# ════════════════════════════════════════════

current_mode = st.session_state.get("active_mode", "Single Stock")

# ── Neither mode has been run yet — show welcome
if not st.session_state["run"] and not st.session_state["port_run"] \
        and st.session_state["single_stock_cache"] is None \
        and st.session_state["portfolio_cache"] is None:
    if current_mode == "Single Stock":
        st.markdown(
            '<div class="interpret-box" style="margin-top:40px;text-align:center;">'
            '<b>Single Stock Mode</b><br><br>'
            '1. Enter a ticker in the sidebar<br>'
            '2. Set your date range<br>'
            '3. Click <b>▶ RUN REGRESSION</b><br><br>'
            '<span style="color:#334155;font-size:11px;">Factor data (FF5 + Momentum) loads automatically. '
            'AI macro intelligence runs on every regression.</span>'
            '</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="interpret-box" style="margin-top:40px;text-align:center;">'
            '<b>Portfolio Attribution Mode</b><br><br>'
            '1. Add your holdings (ticker + amount) in the sidebar<br>'
            '2. Set your date range<br>'
            '3. Click <b>▶ RUN PORTFOLIO ATTRIBUTION</b><br><br>'
            '<span style="color:#334155;font-size:11px;">Runs FF5 + Momentum regression per stock, '
            'then shows weighted factor exposures and concentration risk.</span>'
            '</div>', unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════
#  PORTFOLIO ATTRIBUTION MODE
# ════════════════════════════════════════════

if current_mode == "Portfolio Attribution":
    # ── If cache exists and no new run triggered, just show the cached result
    if st.session_state["portfolio_cache"] is not None and not st.session_state.get("port_run"):
        cache = st.session_state["portfolio_cache"]
        st.markdown("## Portfolio Factor Attribution")
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#475569;margin-bottom:20px;">'
            f'{len(cache["weights"])} holdings · FF5 + Momentum · HAC SE · {cache["port_start"]} → {cache["port_end"]}'
            f'</div>', unsafe_allow_html=True
        )
        render_portfolio_attribution(cache["port_results"], cache["weights"], cache["ordered_factors"])
        st.stop()

    # ── Run (or re-run) portfolio attribution
    if not st.session_state.get("port_run"):
        st.stop()

    weights    = st.session_state.get("port_weights", {})
    port_start = st.session_state.get("port_start", start_date)
    port_end   = st.session_state.get("port_end",   end_date)
    port_hac   = st.session_state.get("port_hac",   hac_lags)

    if not weights:
        st.markdown('<div class="error-box">No valid tickers found. Check your input format.</div>', unsafe_allow_html=True)
        st.stop()

    st.markdown("## Portfolio Factor Attribution")
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#475569;margin-bottom:20px;">'
        f'{len(weights)} holdings · FF5 + Momentum · HAC SE · {port_start} → {port_end}'
        f'</div>', unsafe_allow_html=True
    )

    port_results = {}
    errors = []

    progress_bar = st.progress(0, text="Running regressions…")
    for i, (tkr, w) in enumerate(weights.items()):
        progress_bar.progress((i) / len(weights), text=f"Regressing {tkr}…")
        start_str = str(port_start)[:7]
        end_str   = str(port_end)[:7]
        res, err = run_single_regression(tkr, start_str, end_str, port_hac, f"{start_str}_{end_str}")
        if res:
            port_results[tkr] = res
        else:
            errors.append(f"{tkr}: {err}")
    progress_bar.progress(1.0, text="Done.")

    if errors:
        for e in errors:
            st.markdown(f'<div class="error-box">⚠ {e}</div>', unsafe_allow_html=True)

    if not port_results:
        st.markdown('<div class="error-box">No stocks could be regressed. Check tickers and date range.</div>', unsafe_allow_html=True)
        st.stop()

    valid_weights = {t: weights[t] for t in port_results}
    total = sum(valid_weights.values())
    valid_weights = {t: v / total for t, v in valid_weights.items()}

    avail_sets = [set(port_results[t]["available"]) for t in port_results]
    common_factors = list(avail_sets[0].intersection(*avail_sets[1:])) if len(avail_sets) > 1 else list(avail_sets[0])
    ordered_factors = [f for f in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"] if f in common_factors]

    # ── Save to cache so switching back to this tab re-renders without re-running
    st.session_state["portfolio_cache"] = {
        "port_results":    port_results,
        "weights":         valid_weights,
        "ordered_factors": ordered_factors,
        "port_start":      port_start,
        "port_end":        port_end,
    }
    st.session_state["port_run"] = False   # reset trigger; cache keeps the data

    render_portfolio_attribution(port_results, valid_weights, ordered_factors)
    st.stop()


# ════════════════════════════════════════════
#  SINGLE STOCK MODE
# ════════════════════════════════════════════

if current_mode == "Single Stock":
    # ── If cache exists and no new run triggered, re-render from cache ──────
    if st.session_state["single_stock_cache"] is not None and not st.session_state["run"]:
        c = st.session_state["single_stock_cache"]
        # Re-display the stored price HTML + all sections from cache
        st.markdown(c["price_html"], unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(c["metric_alpha"], unsafe_allow_html=True)
        with col2: st.markdown(c["metric_r2"], unsafe_allow_html=True)
        with col3: st.markdown(c["metric_ir"], unsafe_allow_html=True)
        with col4: st.markdown(c["metric_f"], unsafe_allow_html=True)

        st.markdown('<div class="section-title">Factor Loadings</div>', unsafe_allow_html=True)
        for row_html in c["factor_rows"]:
            st.markdown(row_html, unsafe_allow_html=True)
        st.markdown(c["factor_legend"], unsafe_allow_html=True)

        st.markdown('<div class="section-title">AI Macro Intelligence</div>', unsafe_allow_html=True)
        st.markdown(c["ai_header"], unsafe_allow_html=True)
        if c.get("ai_error") and not c.get("ai_insight"):
            st.markdown(f'<div class="error-box">AI Error: {c["ai_error"]}</div>', unsafe_allow_html=True)
        elif c.get("ai_error") and c.get("ai_insight"):
            st.markdown(f'<div class="error-box" style="border-color:rgba(251,191,36,0.3);color:#fbbf24;">⚠ Some factors had errors: {c["ai_error"]}</div>', unsafe_allow_html=True)
        if c.get("ai_insight"):
            render_ai_insight(c["ticker"], c["ai_insight"])

        # Recreate tabs from stored data
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "95% Confidence Intervals", "Regression Diagnostics",
            "Variance Inflation Factors", "Model Fit", "Rolling Market Beta",
        ])
        with tab1: st.markdown(c["ci_html"], unsafe_allow_html=True)
        with tab2: st.markdown(c["diag_html"], unsafe_allow_html=True)
        with tab3: st.markdown(c["vif_html"], unsafe_allow_html=True)
        with tab4: st.markdown(c["fit_html"], unsafe_allow_html=True)
        with tab5:
            if c.get("rolling_html"):
                st.markdown(c["rolling_html"], unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#334155;padding:20px 0;">Need at least 36 months of data and the Mkt-RF factor to display rolling beta.</div>', unsafe_allow_html=True)

        with st.expander("Full OLS Summary (statsmodels)"):
            st.text(c["ols_summary"])

        st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("⬇  Download Results CSV", c["export_csv"],
                               file_name=f"{c['ticker']}_factor_regression.csv", mime="text/csv", use_container_width=True)
        with col_b:
            st.download_button("⬇  Download Merged Data CSV", c["merged_csv"],
                               file_name=f"{c['ticker']}_merged_data.csv", mime="text/csv", use_container_width=True)
        st.stop()

    # ── No cache and no run triggered → stop (welcome already shown above)
    if not st.session_state["run"]:
        st.stop()

    # ── Fresh regression run ─────────────────────────────────────────────────
    ticker     = st.session_state.get("ticker_ran", "AVGO")
    start_date = st.session_state.get("start_ran", start_date)
    end_date   = st.session_state.get("end_ran", end_date)
    hac_lags   = st.session_state.get("hac_ran", hac_lags)
    annualize  = st.session_state.get("ann_ran", annualize)

    try:
        with st.spinner("Loading factor data from GitHub..."):
            try:
                ff_raw = load_factors()
            except Exception as e:
                st.markdown(f'<div class="error-box">Failed to load factor data from GitHub: {e}</div>', unsafe_allow_html=True)
                st.stop()

        ff = ff_raw.loc[str(start_date)[:7]: str(end_date)[:7]].copy()
        available = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"] if c in ff.columns]
        has_rf    = "RF" in ff.columns

        if not available:
            st.markdown('<div class="error-box">No recognised factor columns found in the CSV.</div>', unsafe_allow_html=True)
            st.stop()

        ff[available] = ff[available].astype(float) / 100
        if has_rf:
            ff["RF"] = ff["RF"].astype(float) / 100

        with st.spinner(f"Downloading {ticker} from Yahoo Finance..."):
            raw = yf.download(ticker, start=str(start_date), end=str(end_date), auto_adjust=True, progress=False)

        if raw.empty:
            st.markdown(f'<div class="error-box">No price data found for ticker: {ticker}</div>', unsafe_allow_html=True)
            st.stop()

        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        monthly = close.resample("ME").last()
        returns = monthly.pct_change().dropna()
        returns.index = returns.index.to_period("M")

        data = pd.DataFrame({"Stock": returns}).join(ff[available + (["RF"] if has_rf else [])], how="inner")

        if len(data) < 24:
            st.markdown(f'<div class="error-box">Too few observations after merge ({len(data)}). Check date range.</div>', unsafe_allow_html=True)
            st.stop()

        if has_rf:
            data["Y"] = data["Stock"] - data["RF"]
        else:
            data["Y"] = data["Stock"]

        X = sm.add_constant(data[available])
        y = data["Y"]
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

        alpha     = model.params["const"]
        alpha_ann = (1 + alpha) ** annualize - 1
        alpha_t   = model.tvalues["const"]
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
        dw = sm.stats.stattools.durbin_watson(resid)
        jb_stat, jb_p = stats.jarque_bera(resid)[:2]
        cond = np.linalg.cond(X.values)

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
            cls = {"pass": "diag-pass", "fail": "diag-fail", "warn": "diag-warn"}.get(status, "diag-pass")
            return f'<div class="diag-item"><div class="diag-name">{name}</div><div class="diag-val {cls}">{val}</div><div class="diag-sub">{sub}</div></div>'

        # ── Live price banner
        try:
            _live_price, _prev_close, _currency = get_live_price(ticker)
            _chg     = _live_price - _prev_close
            _chg_pct = (_chg / _prev_close) * 100
            _price_color = "#34d399" if _chg >= 0 else "#f87171"
            _arrow       = "&#9650;" if _chg >= 0 else "&#9660;"
            _price_html  = (
                f'<div style="display:flex;align-items:baseline;gap:16px;margin-bottom:20px;'
                f'padding:14px 20px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
                f'border-radius:12px;flex-wrap:wrap;">'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:22px;font-weight:700;color:#e2e8f0;">{ticker}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:28px;font-weight:700;color:{_price_color};">'
                f'{_currency} {_live_price:,.2f}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:15px;color:{_price_color};">'
                f'{_arrow} {abs(_chg):,.2f} ({abs(_chg_pct):.2f}%)</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:#334155;margin-left:auto;">'
                f'LIVE &nbsp;·&nbsp; prev close {_currency} {_prev_close:,.2f}</span>'
                f'</div>'
            )
        except Exception:
            _price_html = (
                f'<div style="padding:14px 20px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
                f'border-radius:12px;margin-bottom:20px;font-family:JetBrains Mono,monospace;'
                f'font-size:22px;font-weight:700;color:#e2e8f0;">{ticker} '
                f'<span style="font-size:12px;color:#475569;">· price unavailable</span></div>'
            )
        st.markdown(_price_html, unsafe_allow_html=True)

        # ── Key metrics
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
        ir_str = f"{ir:.3f}" if not np.isnan(ir) else "N/A"
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
        with c2: st.markdown(_r2_card, unsafe_allow_html=True)
        with c3: st.markdown(_ir_card, unsafe_allow_html=True)
        with c4: st.markdown(_f_card, unsafe_allow_html=True)

        # ── Factor loadings table
        st.markdown('<div class="section-title">Factor Loadings</div>', unsafe_allow_html=True)
        header_html = """
        <div class="factor-row header">
          <div>FACTOR</div><div>BETA</div><div>STD ERR</div><div>T-STAT</div><div>P-VALUE</div><div>SIG</div>
        </div>"""
        st.markdown(header_html, unsafe_allow_html=True)

        factor_rows_html = []
        for name in ["const"] + available:
            b = model.params[name]; se = model.bse[name]
            t = model.tvalues[name]; p = model.pvalues[name]
            row_h = f"""
            <div class="{row_class(name, p)}">
              <div style="color:#e2e8f0;font-weight:500">{FACTOR_NAMES.get(name, name)}</div>
              <div>{fmt_beta(b)}</div>
              <div style="color:#475569">{se:.4f}</div>
              <div>{fmt_tstat(t)}</div>
              <div>{fmt_pval(p)}</div>
              <div>{sig_badge(p)}</div>
            </div>"""
            st.markdown(row_h, unsafe_allow_html=True)
            factor_rows_html.append(row_h)

        legend_html = (
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#334155;margin-top:6px;">'
            f'★★★ p&lt;0.01 · ★★ p&lt;0.05 · ★ p&lt;0.10 · n.s. not significant'
            f' | HAC Newey-West SE, maxlags={hac_lags}</div>'
        )
        st.markdown(legend_html, unsafe_allow_html=True)

        # ── AI Macro Intelligence
        st.markdown('<div class="section-title">AI Macro Intelligence</div>', unsafe_allow_html=True)
        today_display = date.today().strftime("%B %d, %Y")
        ai_header_html = (
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#475569;margin-bottom:12px;">'
            f'Live fundamentals-grounded analysis · Per-factor deep dive · Current macro context as of {today_display} · Forward forecast · Key risks'
            f'<br><span style="color:#334155;">Powered by Groq · Llama 3.3 70b · Uses live price, P/E, P/B, margins, analyst targets, short interest</span>'
            f'</div>'
        )
        st.markdown(ai_header_html, unsafe_allow_html=True)

        _ai_key = f"{ticker}_{start_date}_{end_date}_{hac_lags}_{annualize}"
        if st.session_state.get("ai_run_key") != _ai_key:
            st.session_state["ai_run_key"] = _ai_key
            st.session_state["ai_insight"] = None
            st.session_state["ai_error"]   = None
            with st.spinner(f"Fetching live fundamentals + analyzing {len(available)} factors for {ticker}..."):
                insight, error = get_ai_insight(
                    ticker, model, available, alpha_ann, alpha_p, r2, n, start_date, end_date
                )
            st.session_state["ai_insight"] = insight
            st.session_state["ai_error"]   = error

        if st.session_state["ai_error"] and not st.session_state["ai_insight"]:
            st.markdown(f'<div class="error-box">AI Error: {st.session_state["ai_error"]}</div>', unsafe_allow_html=True)
        elif st.session_state["ai_error"] and st.session_state["ai_insight"]:
            st.markdown(
                f'<div class="error-box" style="border-color:rgba(251,191,36,0.3);color:#fbbf24;">'
                f'⚠ Some factors had errors (showing placeholders): {st.session_state["ai_error"]}</div>',
                unsafe_allow_html=True)

        if st.session_state["ai_insight"]:
            render_ai_insight(ticker, st.session_state["ai_insight"])

        # ── Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "95% Confidence Intervals",
            "Regression Diagnostics",
            "Variance Inflation Factors",
            "Model Fit",
            "Rolling Market Beta",
        ])

        ci = model.conf_int()
        ci_html = '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:12px;">'
        for name in ["const"] + available:
            lo = ci.loc[name, 0]; hi = ci.loc[name, 1]; b = model.params[name]
            spans_zero = lo < 0 < hi
            bar_color = "#34d399" if b > 0 and not spans_zero else "#f87171" if b < 0 and not spans_zero else "#6b7280"
            ci_html += f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:14px 16px;min-width:160px;flex:1;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#475569;margin-bottom:6px;">{FACTOR_NAMES.get(name, name)}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;color:{bar_color};">{b:+.4f}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#334155;margin-top:4px;">[{lo:+.4f}, {hi:+.4f}]</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#334155;margin-top:6px;">{'spans zero' if spans_zero else 'excl. zero'}</div>
            </div>"""
        ci_html += '</div>'

        diag_html = f"""
        <div style="height:12px"></div>
        <div class="diag-grid">
          {diag_card("Durbin-Watson", f"{dw:.4f}", "Autocorrelation · ideal ≈ 2.0", dw_status)}
          {diag_card("Breusch-Pagan", f"p = {bp_p:.4f}", f"Heteroscedasticity · stat={bp_stat:.3f}", bp_status)}
          {diag_card("Jarque-Bera", f"p = {jb_p:.4f}", f"Residual normality · stat={jb_stat:.3f}", jb_status)}
          {diag_card("Condition Number", f"{cond:.1f}", "Multicollinearity · ideal < 30", cn_status)}
        </div>"""

        vif_html = '<div style="height:12px"></div><div style="display:flex;gap:10px;flex-wrap:wrap;">'
        for col, v in vif_data.items():
            vif_color = "#34d399" if v < 5 else "#fbbf24" if v < 10 else "#f87171"
            vif_html += f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:14px 16px;min-width:110px;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#475569;margin-bottom:4px;">{FACTOR_NAMES.get(col, col)}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:600;color:{vif_color}">{v:.2f}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#334155;margin-top:2px;">{'OK' if v < 5 else 'MODERATE' if v < 10 else 'HIGH'}</div>
            </div>"""
        vif_html += '</div>'

        fit_html = f"""
        <div style="height:12px"></div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
          <div class="diag-item" style="min-width:130px;"><div class="diag-name">AIC</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{aic:.2f}</div></div>
          <div class="diag-item" style="min-width:130px;"><div class="diag-name">BIC</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{bic:.2f}</div></div>
          <div class="diag-item" style="min-width:130px;"><div class="diag-name">Log-Likelihood</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{model.llf:.2f}</div></div>
          <div class="diag-item" style="min-width:130px;"><div class="diag-name">Residual Std</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{resid.std():.4f}</div></div>
          <div class="diag-item" style="min-width:130px;"><div class="diag-name">Skewness</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{float(stats.skew(resid)):.4f}</div></div>
          <div class="diag-item" style="min-width:130px;"><div class="diag-name">Kurtosis</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{float(stats.kurtosis(resid)):.4f}</div></div>
        </div>"""

        rolling_html = None
        if "Mkt-RF" in available and len(data) >= 36:
            window = 24
            roll_betas, roll_dates = [], []
            for i in range(window, len(data) + 1):
                sub = data.iloc[i - window: i]
                try:
                    rb = sm.OLS(sub["Y"], sm.add_constant(sub[["Mkt-RF"]])).fit().params["Mkt-RF"]
                    roll_betas.append(rb)
                    roll_dates.append(str(data.index[i - 1]))
                except Exception:
                    pass
            if roll_betas:
                mn, mx = min(roll_betas), max(roll_betas)
                rng = mx - mn if mx != mn else 1
                W, H = 800, 140
                pts = " ".join(
                    f"{int(i/(len(roll_betas)-1)*W) if len(roll_betas)>1 else W//2},{int(H-((b-mn)/rng)*H)}"
                    for i, b in enumerate(roll_betas)
                )
                one_y = int(H - ((1.0 - mn) / rng) * H)
                rolling_html = f"""
                <div style="margin-top:12px;">
                <svg viewBox="0 0 {W} {H+30}" xmlns="http://www.w3.org/2000/svg"
                     style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);border-radius:10px;width:100%;margin-bottom:8px;">
                  <line x1="0" y1="{one_y}" x2="{W}" y2="{one_y}" stroke="rgba(255,255,255,0.08)" stroke-width="1" stroke-dasharray="4,4"/>
                  <polyline points="{pts}" fill="none" stroke="#60a5fa" stroke-width="2"/>
                  <text x="6" y="{H+20}" fill="#334155" font-family="JetBrains Mono,monospace" font-size="10">{roll_dates[0]}</text>
                  <text x="{W-6}" y="{H+20}" fill="#334155" text-anchor="end" font-family="JetBrains Mono,monospace" font-size="10">{roll_dates[-1]}</text>
                  <text x="{W//2}" y="18" fill="#475569" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="10">Current β = {roll_betas[-1]:.3f}</text>
                </svg>
                </div>"""

        with tab1: st.markdown(ci_html, unsafe_allow_html=True)
        with tab2: st.markdown(diag_html, unsafe_allow_html=True)
        with tab3: st.markdown(vif_html, unsafe_allow_html=True)
        with tab4: st.markdown(fit_html, unsafe_allow_html=True)
        with tab5:
            if rolling_html:
                st.markdown(rolling_html, unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#334155;padding:20px 0;">Need at least 36 months of data and the Mkt-RF factor to display rolling beta.</div>', unsafe_allow_html=True)

        with st.expander("Full OLS Summary (statsmodels)"):
            st.text(model.summary().as_text())

        # ── Export
        st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)
        export_df = pd.DataFrame({
            "Factor":   [FACTOR_NAMES.get(n, n) for n in model.params.index],
            "Beta":     model.params.values,
            "Std_Err":  model.bse.values,
            "T_Stat":   model.tvalues.values,
            "P_Value":  model.pvalues.values,
            "CI_Lower": model.conf_int()[0].values,
            "CI_Upper": model.conf_int()[1].values,
        })
        data_export = data.copy()
        data_export.index = data_export.index.astype(str)

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("⬇  Download Results CSV", export_df.to_csv(index=False),
                               file_name=f"{ticker}_factor_regression.csv", mime="text/csv", use_container_width=True)
        with col_b:
            st.download_button("⬇  Download Merged Data CSV", data_export.to_csv(),
                               file_name=f"{ticker}_merged_data.csv", mime="text/csv", use_container_width=True)

        # ── Save everything to cache so switching modes and coming back is instant
        st.session_state["single_stock_cache"] = {
            "ticker":        ticker,
            "price_html":    _price_html,
            "metric_alpha":  _alpha_card,
            "metric_r2":     _r2_card,
            "metric_ir":     _ir_card,
            "metric_f":      _f_card,
            "factor_rows":   factor_rows_html,
            "factor_legend": legend_html,
            "ai_header":     ai_header_html,
            "ai_insight":    st.session_state["ai_insight"],
            "ai_error":      st.session_state["ai_error"],
            "ci_html":       ci_html,
            "diag_html":     diag_html,
            "vif_html":      vif_html,
            "fit_html":      fit_html,
            "rolling_html":  rolling_html,
            "ols_summary":   model.summary().as_text(),
            "export_csv":    export_df.to_csv(index=False),
            "merged_csv":    data_export.to_csv(),
        }
        st.session_state["run"] = False   # reset trigger; cache keeps the data

    except Exception as e:
        st.markdown(f'<div class="error-box">Error: {str(e)}</div>', unsafe_allow_html=True)
        raise e
