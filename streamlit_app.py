import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from scipy import stats
from groq import Groq
import re
import json
import warnings
warnings.filterwarnings("ignore")

FF_URL = "https://raw.githubusercontent.com/lakshyaworkch-cell/Lakshy/main/F-F_Research_Data_5_Factors_2x3.csv"

st.set_page_config(page_title="Factor Regression", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: #0f1623;
    background-image: radial-gradient(ellipse at 20% 0%, rgba(56,189,248,0.04) 0%, transparent 60%),
                      radial-gradient(ellipse at 80% 100%, rgba(99,102,241,0.04) 0%, transparent 60%);
    color: #e2e8f0;
}
h1, h2, h3 { font-family: 'JetBrains Mono', monospace !important; letter-spacing: -0.5px; }
.metric-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-left: 2px solid #34d399; border-radius: 12px; padding: 16px 18px;
    margin-bottom: 12px; backdrop-filter: blur(8px); transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(255,255,255,0.14); }
.metric-card.red  { border-left-color: #f87171; }
.metric-card.blue { border-left-color: #60a5fa; }
.metric-card.gold { border-left-color: #fbbf24; }
.metric-card.gray { border-left-color: #6b7280; }
.metric-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 2px; color: #475569; text-transform: uppercase; margin-bottom: 4px; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 600; color: #e2e8f0; }
.metric-sub { font-size: 11px; color: #334155; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
.factor-row {
    display: grid; grid-template-columns: 140px 90px 90px 90px 80px 1fr;
    gap: 8px; align-items: center; padding: 10px 14px; border-radius: 8px;
    margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; font-size: 12px;
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06); transition: background 0.15s;
}
.factor-row:hover { background: rgba(255,255,255,0.04); }
.factor-row.header { background: transparent; border-color: transparent; font-size: 10px; letter-spacing: 1.5px; color: #334155; text-transform: uppercase; }
.factor-row.sig   { border-left: 2px solid #34d399; }
.factor-row.marg  { border-left: 2px solid #fbbf24; }
.factor-row.insig { border-left: 2px solid rgba(255,255,255,0.08); }
.factor-row.alpha { border-left: 2px solid #a78bfa; }
.sig-badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 10px; font-weight: 500; letter-spacing: 1px; }
.badge-001 { background: rgba(52,211,153,0.12); color: #34d399; }
.badge-01  { background: rgba(52,211,153,0.08); color: #6ee7b7; }
.badge-05  { background: rgba(251,191,36,0.1);  color: #fbbf24; }
.badge-10  { background: rgba(255,255,255,0.05); color: #6b7280; }
.badge-ns  { background: rgba(255,255,255,0.03); color: #334155; }
.section-title {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 2.5px;
    text-transform: uppercase; color: #334155; border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 8px; margin: 28px 0 16px 0;
}
.diag-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.diag-item { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 14px 16px; }
.diag-name { font-size: 11px; color: #475569; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; }
.diag-val  { font-size: 16px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.diag-pass { color: #34d399; } .diag-fail { color: #f87171; } .diag-warn { color: #fbbf24; }
.diag-sub  { font-size: 10px; color: #1e293b; font-family: 'JetBrains Mono', monospace; margin-top: 3px; }
.interpret-box { background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 20px 24px; font-size: 13px; line-height: 1.8; color: #94a3b8; }
.interpret-box b { color: #e2e8f0; }

/* ── Enhanced AI Box ── */
.ai-box {
    background: rgba(167,139,250,0.04);
    border: 1px solid rgba(167,139,250,0.15);
    border-radius: 12px;
    padding: 24px 28px;
    font-size: 13px;
    line-height: 1.9;
    color: #c4b5fd;
}
.ai-box b { color: #e2e8f0; }
.ai-box h4 {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 2px;
    text-transform: uppercase; color: #7c3aed; margin-bottom: 16px;
}

/* ── Per-Factor Cards ── */
.factor-insight-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.factor-insight-card.positive { border-left: 3px solid #34d399; }
.factor-insight-card.negative { border-left: 3px solid #f87171; }
.factor-insight-card.neutral  { border-left: 3px solid #6b7280; }
.fi-header {
    display: flex; align-items: center; gap: 12px; margin-bottom: 10px;
}
.fi-name {
    font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600;
    color: #e2e8f0; letter-spacing: 1px;
}
.fi-beta {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    padding: 2px 8px; border-radius: 4px;
}
.fi-beta.pos { background: rgba(52,211,153,0.12); color: #34d399; }
.fi-beta.neg { background: rgba(248,113,113,0.12); color: #f87171; }
.fi-sig-label {
    font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 1.5px;
    color: #475569; text-transform: uppercase;
}
.fi-outlook {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase; margin-left: auto;
}
.fi-outlook.bullish  { background: rgba(52,211,153,0.1); color: #34d399; }
.fi-outlook.bearish  { background: rgba(248,113,113,0.1); color: #f87171; }
.fi-outlook.neutral  { background: rgba(107,114,128,0.1); color: #6b7280; }
.fi-outlook.mixed    { background: rgba(251,191,36,0.1);  color: #fbbf24; }
.fi-body { font-size: 12px; color: #94a3b8; line-height: 1.75; }
.fi-news {
    margin-top: 10px; padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-size: 11px; color: #475569; font-family: 'JetBrains Mono', monospace;
}
.fi-news-label {
    font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: #334155; margin-bottom: 4px;
}

/* Summary section */
.ai-summary-box {
    background: rgba(96,165,250,0.04);
    border: 1px solid rgba(96,165,250,0.12);
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 16px;
    font-size: 12px;
    color: #93c5fd;
    line-height: 1.8;
}
.ai-summary-box b { color: #e2e8f0; }

.stTextInput > div > div > input { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #e2e8f0 !important; font-family: 'JetBrains Mono', monospace !important; border-radius: 8px !important; }
.stTextInput > div > div > input:focus { border-color: rgba(96,165,250,0.4) !important; box-shadow: 0 0 0 3px rgba(96,165,250,0.08) !important; }
.stButton > button { background: rgba(255,255,255,0.06) !important; color: #e2e8f0 !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 500 !important; letter-spacing: 1px !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 8px !important; padding: 10px 28px !important; transition: all 0.2s !important; }
.stButton > button:hover { background: rgba(52,211,153,0.1) !important; border-color: rgba(52,211,153,0.3) !important; color: #34d399 !important; }
.error-box { background: rgba(248,113,113,0.06); border: 1px solid rgba(248,113,113,0.2); border-radius: 8px; padding: 12px 16px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #f87171; }
</style>
""", unsafe_allow_html=True)

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

@st.cache_data(show_spinner=False)
def load_factors():
    ff = pd.read_csv(FF_URL, index_col=0)
    ff.columns = ff.columns.str.strip()
    ff.index = ff.index.astype(str).str.strip()
    ff = ff.drop(columns=[c for c in ff.columns if "Unnamed" in str(c)])
    ff = ff[ff.index.str.match(r"^\d{6}$")].copy()
    ff.index = pd.to_datetime(ff.index, format="%Y%m").to_period("M")
    return ff


# ─────────────────────────────────────────────
#  AI: one small focused call per factor + summary
#  Uses response_format json_object to guarantee
#  valid JSON and avoids truncation/escape issues
# ─────────────────────────────────────────────

def _call_groq(client, system_msg, user_msg, max_tokens=600):
    """Single Groq call with json_object mode and robust cleaning."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
        response_format={"type": "json_object"},   # ← guaranteed valid JSON
    )
    raw = response.choices[0].message.content.strip()
    # Belt-and-suspenders: strip fences in case model ignores json_object mode
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$",     "", raw)
    return json.loads(raw)


def _factor_prompt(ticker, factor_code, factor_name, beta, p_value, significant, description):
    system = (
        "You are a senior quantitative analyst. "
        "You MUST respond with a single valid JSON object and nothing else. "
        "No markdown, no prose outside the JSON."
    )
    sig_word = "statistically significant" if significant else "not statistically significant"
    user = f"""Analyze the {factor_name} ({factor_code}) loading for stock {ticker}.

Factor details:
- Beta: {beta:+.4f}
- P-value: {p_value:.4f} ({sig_word})
- Factor definition: {description}

Return EXACTLY this JSON (fill in every field, 1-3 sentences each, no line breaks inside strings):
{{
  "code": "{factor_code}",
  "name": "{factor_name}",
  "beta": {beta},
  "significant": {"true" if significant else "false"},
  "outlook": "<bullish|bearish|neutral|mixed>",
  "what_it_means": "<What this beta magnitude reveals about {ticker} behaviour. Be specific.>",
  "current_macro_context": "<Mid-2025 macro conditions relevant to this factor: Fed policy, inflation, AI capex cycle, tariffs, credit spreads, sector rotation. Be concrete.>",
  "forward_forecast": "<Near-term tailwind or headwind for {ticker} from this factor given current macro. State direction and reason.>",
  "key_risks": "<One main risk that could flip this outlook.>"
}}"""
    return system, user


def _summary_prompt(ticker, alpha_ann, alpha_p, r2, factor_summaries):
    system = (
        "You are a senior portfolio strategist. "
        "Respond with a single valid JSON object only. No markdown."
    )
    sig_word = "significant" if alpha_p < 0.05 else "not significant"
    lines = "\n".join(
        f"- {f['name']}: beta={f['beta']:+.4f}, outlook={f.get('outlook','?')}"
        for f in factor_summaries
    )
    user = f"""Stock: {ticker}
Annualized alpha: {alpha_ann:+.2%} (p={alpha_p:.4f}, {sig_word})
R²: {r2:.4f}
Factor outlooks:
{lines}

Return EXACTLY this JSON (2-4 sentences per field, no line breaks inside strings):
{{
  "alpha_analysis": "<Is this alpha persistent or episodic? What drives it for {ticker} — moat, pricing power, earnings quality? What threatens persistence?>",
  "portfolio_verdict": "<Overall: is {ticker} well-positioned or vulnerable in mid-2025 macro regime? Best-case and worst-case macro scenario. Concise risk-adjusted conclusion.>"
}}"""
    return system, user


def get_ai_insight(ticker, model_result, available, alpha_ann, alpha_p, r2, n, start_date, end_date):
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        return None, "No API key found. Add `GROQ_API_KEY` to your `.streamlit/secrets.toml` file."

    try:
        client = Groq(api_key=api_key)
        factors_out = []
        errors = []

        # ── One call per factor (small, never truncated) ──
        for f in available:
            beta = float(model_result.params[f])
            pval = float(model_result.pvalues[f])
            sig  = pval < 0.05
            sys_msg, usr_msg = _factor_prompt(
                ticker, f, FACTOR_NAMES.get(f, f), beta, pval, sig,
                FACTOR_DESCRIPTIONS.get(f, f)
            )
            try:
                result = _call_groq(client, sys_msg, usr_msg, max_tokens=550)
                # Ensure numeric beta comes from our data, not the model
                result["beta"] = beta
                result["significant"] = sig
                factors_out.append(result)
            except Exception as e:
                # Gracefully degrade: include a placeholder so rendering still works
                factors_out.append({
                    "code": f, "name": FACTOR_NAMES.get(f, f),
                    "beta": beta, "significant": sig, "outlook": "neutral",
                    "what_it_means": "Analysis unavailable.",
                    "current_macro_context": f"Error: {e}",
                    "forward_forecast": "", "key_risks": "",
                })
                errors.append(f"{f}: {e}")

        # ── Single summary call ──
        sys_msg, usr_msg = _summary_prompt(ticker, alpha_ann, alpha_p, r2, factors_out)
        try:
            summary = _call_groq(client, sys_msg, usr_msg, max_tokens=500)
        except Exception as e:
            summary = {
                "alpha_analysis": f"Summary unavailable: {e}",
                "portfolio_verdict": "",
            }
            errors.append(f"summary: {e}")

        result = {"factors": factors_out, **summary}
        err_msg = "; ".join(errors) if errors else None
        return result, err_msg

    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
#  NEW: Render the enhanced AI output as cards
# ─────────────────────────────────────────────

def render_ai_insight(ticker, insight_data):
    """Render the structured JSON insight as rich factor cards."""
    factors = insight_data.get("factors", [])
    alpha_analysis = insight_data.get("alpha_analysis", "")
    portfolio_verdict = insight_data.get("portfolio_verdict", "")

    def bold(text):
        return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;'
        f'text-transform:uppercase;color:#7c3aed;margin-bottom:16px;">✦ AI Factor Intelligence · {ticker}</div>',
        unsafe_allow_html=True,
    )

    for fac in factors:
        code = fac.get("code", "")
        name = fac.get("name", code)
        beta = fac.get("beta", 0)
        sig  = fac.get("significant", False)
        outlook = fac.get("outlook", "neutral").lower()

        card_cls = "positive" if beta > 0 else "negative"
        beta_cls = "pos" if beta > 0 else "neg"
        outlook_cls = {"bullish": "bullish", "bearish": "bearish", "neutral": "neutral", "mixed": "mixed"}.get(outlook, "neutral")
        outlook_icon = {"bullish": "↑ BULLISH", "bearish": "↓ BEARISH", "neutral": "→ NEUTRAL", "mixed": "⇅ MIXED"}.get(outlook, "→ NEUTRAL")

        sig_label = "SIGNIFICANT" if sig else "INSIGNIFICANT"

        what      = bold(fac.get("what_it_means", ""))
        macro     = bold(fac.get("current_macro_context", ""))
        forecast  = bold(fac.get("forward_forecast", ""))
        risks     = bold(fac.get("key_risks", ""))

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
              <span style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;
                           text-transform:uppercase;color:#475569;">WHAT IT MEANS FOR {ticker}</span>
              <div style="margin-top:4px;">{what}</div>
            </div>
            <div style="margin-bottom:8px;">
              <span style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;
                           text-transform:uppercase;color:#475569;">CURRENT MACRO CONTEXT</span>
              <div style="margin-top:4px;">{macro}</div>
            </div>
            <div style="margin-bottom:8px;">
              <span style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;
                           text-transform:uppercase;color:#475569;">FORWARD FORECAST</span>
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
          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;
                      text-transform:uppercase;color:#3b82f6;margin-bottom:8px;">ALPHA PERSISTENCE ANALYSIS</div>
          {bold(alpha_analysis)}
        </div>""", unsafe_allow_html=True)

    if portfolio_verdict:
        st.markdown(f"""
        <div class="ai-summary-box" style="border-color:rgba(167,139,250,0.15);color:#c4b5fd;background:rgba(167,139,250,0.04);">
          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:1.5px;
                      text-transform:uppercase;color:#7c3aed;margin-bottom:8px;">PORTFOLIO VERDICT</div>
          {bold(portfolio_verdict)}
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  MAIN UI — unchanged from original below
# ════════════════════════════════════════════

st.markdown("# Factor Regression")
st.markdown(
    '<p style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:12px;letter-spacing:1px;">'
    'FF5 + MOMENTUM · HAC ROBUST STANDARD ERRORS · DATA AUTO-LOADED FROM GITHUB'
    '</p>', unsafe_allow_html=True)
st.markdown("---")

if "run" not in st.session_state:
    st.session_state["run"] = False
if "ai_insight" not in st.session_state:
    st.session_state["ai_insight"] = None
if "ai_error" not in st.session_state:
    st.session_state["ai_error"] = None

with st.sidebar:
    st.markdown("### Configuration")
    ticker = st.text_input("Stock Ticker", "AVGO").upper().strip()
    st.markdown("**Date Range**")
    start_date = st.date_input("Start", value=pd.to_datetime("2010-01-01"))
    end_date   = st.date_input("End",   value=pd.to_datetime("2026-04-30"))
    st.markdown("**Regression Settings**")
    hac_lags  = st.slider("HAC Max Lags", 1, 12, 3, help="Newey-West lags for HAC robust SE")
    annualize = st.selectbox("Annualization", [12, 52, 252], help="12=monthly, 52=weekly, 252=daily")

    if st.button("▶  RUN REGRESSION", use_container_width=True):
        st.session_state["run"]        = True
        st.session_state["ticker_ran"] = ticker
        st.session_state["start_ran"]  = start_date
        st.session_state["end_ran"]    = end_date
        st.session_state["hac_ran"]    = hac_lags
        st.session_state["ann_ran"]    = annualize
        st.session_state["ai_insight"] = None
        st.session_state["ai_error"]   = None

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#334155;">'
        'Factor data auto-loaded from:<br>'
        '<span style="color:#475569;">github.com/lakshyaworkch-cell/Lakshy</span>'
        '</div>', unsafe_allow_html=True)

if not st.session_state["run"]:
    st.markdown(
        '<div class="interpret-box" style="margin-top:40px;text-align:center;">'
        '<b>How to use</b><br><br>'
        '1. Enter a stock ticker in the sidebar<br>'
        '2. Choose your date range<br>'
        '3. Adjust HAC lags &amp; annualization if needed<br>'
        '4. Click <b>RUN REGRESSION</b><br><br>'
        '<span style="color:#334155;font-size:11px;">Factor data (FF5) is loaded automatically — no file upload needed.</span>'
        '</div>', unsafe_allow_html=True)
    st.stop()

ticker     = st.session_state.get("ticker_ran", ticker)
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
        y_label = "Excess Return (Stock − Rf)"
    else:
        data["Y"] = data["Stock"]
        y_label = "Raw Return (no RF in file)"

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

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card {'green' if alpha_ann > 0 else 'red'}">
          <div class="metric-label">Annual Alpha</div>
          <div class="metric-value" style="color:{'#34d399' if alpha_ann>0 else '#f87171'}">{alpha_ann:+.2%}</div>
          <div class="metric-sub">Monthly: {alpha:+.4f} · p={alpha_p:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card blue">
          <div class="metric-label">R² / Adj R²</div>
          <div class="metric-value">{r2:.4f}</div>
          <div class="metric-sub">Adj: {r2_adj:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        ir_color = "#34d399" if not np.isnan(ir) and ir > 0.5 else "#fbbf24" if not np.isnan(ir) and ir > 0 else "#f87171"
        ir_str = f"{ir:.3f}" if not np.isnan(ir) else "N/A"
        st.markdown(f"""
        <div class="metric-card gold">
          <div class="metric-label">Information Ratio</div>
          <div class="metric-value" style="color:{ir_color}">{ir_str}</div>
          <div class="metric-sub">Ann. TE: {te:.2%}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card gray">
          <div class="metric-label">F-Stat / Obs</div>
          <div class="metric-value">{f_stat:.2f}</div>
          <div class="metric-sub">p={f_p:.4f} · N={n}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Factor Loadings</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="factor-row header">
      <div>FACTOR</div><div>BETA</div><div>STD ERR</div><div>T-STAT</div><div>P-VALUE</div><div>SIG</div>
    </div>""", unsafe_allow_html=True)

    for name in ["const"] + available:
        b = model.params[name]; se = model.bse[name]
        t = model.tvalues[name]; p = model.pvalues[name]
        st.markdown(f"""
        <div class="{row_class(name, p)}">
          <div style="color:#e2e8f0;font-weight:500">{FACTOR_NAMES.get(name, name)}</div>
          <div>{fmt_beta(b)}</div>
          <div style="color:#475569">{se:.4f}</div>
          <div>{fmt_tstat(t)}</div>
          <div>{fmt_pval(p)}</div>
          <div>{sig_badge(p)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#334155;margin-top:6px;">'
        f'★★★ p&lt;0.01 · ★★ p&lt;0.05 · ★ p&lt;0.10 · n.s. not significant'
        f' | HAC Newey-West SE, maxlags={hac_lags}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)
    sig_factors   = [FACTOR_NAMES.get(f, f) for f in available if model.pvalues[f] < 0.05]
    insig_factors = [FACTOR_NAMES.get(f, f) for f in available if model.pvalues[f] >= 0.05]
    alpha_sig    = "statistically significant" if alpha_p < 0.05 else "not statistically significant"
    alpha_interp = "outperforms" if alpha_ann > 0 else "underperforms"

    interp = f"""<div class="interpret-box">
    <b>{ticker}</b> — {y_label}<br><br>
    The model explains <b>{r2:.1%}</b> of return variation (Adj R² = {r2_adj:.1%}) over <b>{n}</b> monthly observations.<br><br>
    The annualized alpha of <b>{alpha_ann:+.2%}</b> is {alpha_sig} (t = {alpha_t:.3f}, p = {alpha_p:.4f}),
    suggesting the stock <b>{alpha_interp}</b> what factor exposures alone would predict.<br><br>"""
    if sig_factors:
        interp += f"<b>Significant factors (p&lt;0.05):</b> {', '.join(sig_factors)}.<br>"
    if insig_factors:
        interp += f"<b>Insignificant factors:</b> {', '.join(insig_factors)}.<br>"
    mkt_b = model.params.get("Mkt-RF", None)
    if mkt_b is not None:
        interp += f"<br>Market beta of <b>{mkt_b:.4f}</b> implies the stock is {'more' if mkt_b > 1 else 'less'} volatile than the market."
    if dw_status != "pass":
        interp += f"<br><br>⚠ Durbin-Watson ({dw:.3f}) suggests possible autocorrelation. HAC correction applied."
    if bp_status != "pass":
        interp += f"<br>⚠ Breusch-Pagan (p={bp_p:.4f}) indicates heteroscedasticity. HAC robust SEs account for this."
    if jb_status != "pass":
        interp += f"<br>⚠ Jarque-Bera (p={jb_p:.4f}) suggests non-normal residuals."
    if cn_status != "pass":
        interp += f"<br>⚠ Condition number ({cond:.1f}) indicates {'moderate' if cn_status=='warn' else 'severe'} multicollinearity."
    interp += "</div>"
    st.markdown(interp, unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    #  AI MACRO INSIGHT — Enhanced Section
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-title">AI Macro Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#475569;margin-bottom:12px;">'
        'Per-factor deep analysis · Current macro context · Forward forecast · Key risks'
        '<br><span style="color:#334155;">Powered by Groq · Llama 3.3 70b · ~3× more detail than standard analysis</span>'
        '</div>', unsafe_allow_html=True)

    if st.button("✦  Generate Deep Factor Intelligence", use_container_width=False):
        with st.spinner("Analyzing each factor against current macro environment..."):
            insight, error = get_ai_insight(
                ticker, model, available, alpha_ann, alpha_p, r2, n, start_date, end_date
            )
        st.session_state["ai_insight"] = insight
        st.session_state["ai_error"]   = error

    if st.session_state["ai_error"]:
        st.markdown(f'<div class="error-box">AI Error: {st.session_state["ai_error"]}</div>', unsafe_allow_html=True)
    elif st.session_state["ai_insight"]:
        render_ai_insight(ticker, st.session_state["ai_insight"])

    # ─────────────────────────────────────────────
    #  TABS — unchanged
    # ─────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "95% Confidence Intervals",
        "Regression Diagnostics",
        "Variance Inflation Factors",
        "Model Fit",
        "Rolling Market Beta",
    ])

    with tab1:
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
        st.markdown(ci_html, unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="diag-grid">
          {diag_card("Durbin-Watson", f"{dw:.4f}", "Autocorrelation · ideal ≈ 2.0", dw_status)}
          {diag_card("Breusch-Pagan", f"p = {bp_p:.4f}", f"Heteroscedasticity · stat={bp_stat:.3f}", bp_status)}
          {diag_card("Jarque-Bera", f"p = {jb_p:.4f}", f"Residual normality · stat={jb_stat:.3f}", jb_status)}
          {diag_card("Condition Number", f"{cond:.1f}", "Multicollinearity · ideal < 30", cn_status)}
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        vif_html = '<div style="display:flex;gap:10px;flex-wrap:wrap;">'
        for col, v in vif_data.items():
            vif_color = "#34d399" if v < 5 else "#fbbf24" if v < 10 else "#f87171"
            vif_html += f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:14px 16px;min-width:110px;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#475569;margin-bottom:4px;">{FACTOR_NAMES.get(col, col)}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:600;color:{vif_color}">{v:.2f}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#334155;margin-top:2px;">{'OK' if v < 5 else 'MODERATE' if v < 10 else 'HIGH'}</div>
            </div>"""
        vif_html += '</div>'
        st.markdown(vif_html, unsafe_allow_html=True)

    with tab4:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
          <div class="diag-item" style="min-width:130px;"><div class="diag-name">AIC</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{aic:.2f}</div></div>
          <div class="diag-item" style="min-width:130px;"><div class="diag-name">BIC</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{bic:.2f}</div></div>
          <div class="diag-item" style="min-width:130px;"><div class="diag-name">Log-Likelihood</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{model.llf:.2f}</div></div>
          <div class="diag-item" style="min-width:130px;"><div class="diag-name">Residual Std</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{resid.std():.4f}</div></div>
          <div class="diag-item" style="min-width:130px;"><div class="diag-name">Skewness</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{float(stats.skew(resid)):.4f}</div></div>
          <div class="diag-item" style="min-width:130px;"><div class="diag-name">Kurtosis</div><div class="diag-val" style="color:#60a5fa;font-size:16px;">{float(stats.kurtosis(resid)):.4f}</div></div>
        </div>""", unsafe_allow_html=True)

    with tab5:
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
                st.markdown(f"""
                <div style="margin-top:12px;">
                <svg viewBox="0 0 {W} {H+30}" xmlns="http://www.w3.org/2000/svg"
                     style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);border-radius:10px;width:100%;margin-bottom:8px;">
                  <line x1="0" y1="{one_y}" x2="{W}" y2="{one_y}" stroke="rgba(255,255,255,0.08)" stroke-width="1" stroke-dasharray="4,4"/>
                  <polyline points="{pts}" fill="none" stroke="#60a5fa" stroke-width="2"/>
                  <text x="6" y="{H+20}" fill="#334155" font-family="JetBrains Mono,monospace" font-size="10">{roll_dates[0]}</text>
                  <text x="{W-6}" y="{H+20}" fill="#334155" text-anchor="end" font-family="JetBrains Mono,monospace" font-size="10">{roll_dates[-1]}</text>
                  <text x="{W//2}" y="18" fill="#475569" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="10">Current β = {roll_betas[-1]:.3f}</text>
                </svg>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#334155;padding:20px 0;">'
                'Need at least 36 months of data and the Mkt-RF factor to display rolling beta.</div>',
                unsafe_allow_html=True)

    with st.expander("Full OLS Summary (statsmodels)"):
        st.text(model.summary().as_text())

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
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("⬇  Download Results CSV", export_df.to_csv(index=False),
                           file_name=f"{ticker}_factor_regression.csv", mime="text/csv", use_container_width=True)
    with col_b:
        data_export = data.copy()
        data_export.index = data_export.index.astype(str)
        st.download_button("⬇  Download Merged Data CSV", data_export.to_csv(),
                           file_name=f"{ticker}_merged_data.csv", mime="text/csv", use_container_width=True)

except Exception as e:
    st.markdown(f'<div class="error-box">Error: {str(e)}</div>', unsafe_allow_html=True)
    raise e
