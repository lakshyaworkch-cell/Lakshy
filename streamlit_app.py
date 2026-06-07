import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

FF_URL = "https://raw.githubusercontent.com/lakshyaworkch-cell/Lakshy/main/F-F_Research_Data_5_Factors_2x3.csv"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Factor Regression", page_icon="📈", layout="wide")

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0a0a0f; color: #e8e8f0; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; letter-spacing: -0.5px; }
.metric-card { background: #12121a; border: 1px solid #1e1e2e; border-left: 3px solid #4ade80; border-radius: 4px; padding: 18px 20px; margin-bottom: 12px; }
.metric-card.red  { border-left-color: #f87171; }
.metric-card.blue { border-left-color: #60a5fa; }
.metric-card.gold { border-left-color: #fbbf24; }
.metric-card.gray { border-left-color: #6b7280; }
.metric-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 2px; color: #6b7280; text-transform: uppercase; margin-bottom: 4px; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 26px; font-weight: 600; color: #e8e8f0; }
.metric-sub { font-size: 11px; color: #4b5563; margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }
.factor-row { display: grid; grid-template-columns: 140px 90px 90px 90px 80px 1fr; gap: 8px; align-items: center; padding: 10px 14px; border-radius: 3px; margin-bottom: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 13px; background: #12121a; border: 1px solid #1a1a28; }
.factor-row.header { background: #0a0a0f; border-color: #1e1e2e; font-size: 10px; letter-spacing: 1.5px; color: #4b5563; text-transform: uppercase; }
.factor-row.sig   { border-left: 3px solid #4ade80; }
.factor-row.marg  { border-left: 3px solid #fbbf24; }
.factor-row.insig { border-left: 3px solid #374151; }
.factor-row.alpha { border-left: 3px solid #a78bfa; }
.sig-badge { display: inline-block; padding: 2px 7px; border-radius: 2px; font-size: 10px; font-weight: 600; letter-spacing: 1px; }
.badge-001 { background: #14532d; color: #4ade80; }
.badge-01  { background: #1a3a1a; color: #86efac; }
.badge-05  { background: #2a2a14; color: #fbbf24; }
.badge-10  { background: #1c1c1c; color: #9ca3af; }
.badge-ns  { background: #111118; color: #4b5563; }
.section-title { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: #4b5563; border-bottom: 1px solid #1e1e2e; padding-bottom: 8px; margin: 28px 0 16px 0; }
.diag-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.diag-item { background: #12121a; border: 1px solid #1e1e2e; border-radius: 4px; padding: 14px 16px; }
.diag-name { font-size: 11px; color: #6b7280; font-family: 'IBM Plex Mono', monospace; margin-bottom: 4px; }
.diag-val  { font-size: 16px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.diag-pass { color: #4ade80; }
.diag-fail { color: #f87171; }
.diag-warn { color: #fbbf24; }
.diag-sub  { font-size: 10px; color: #374151; font-family: 'IBM Plex Mono', monospace; margin-top: 3px; }
.interpret-box { background: #0d0d15; border: 1px solid #1e1e2e; border-radius: 4px; padding: 16px 20px; font-size: 13px; line-height: 1.7; color: #9ca3af; }
.interpret-box b { color: #e8e8f0; }
.stTextInput > div > div > input { background: #12121a !important; border: 1px solid #1e1e2e !important; color: #e8e8f0 !important; font-family: 'IBM Plex Mono', monospace !important; border-radius: 4px !important; }
.stButton > button { background: #e8e8f0 !important; color: #0a0a0f !important; font-family: 'IBM Plex Mono', monospace !important; font-weight: 600 !important; letter-spacing: 1px !important; border: none !important; border-radius: 3px !important; padding: 10px 28px !important; }
.stButton > button:hover { background: #4ade80 !important; }
.error-box { background: #1a0000; border: 1px solid #f87171; border-radius: 4px; padding: 12px 16px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #f87171; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
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
    color = "#4ade80" if v > 0 else "#f87171"
    return f'<span style="color:{color};font-weight:600">{v:+.4f}</span>'

def fmt_pval(p):
    if p < 0.001: color = "#4ade80"
    elif p < 0.05: color = "#86efac"
    elif p < 0.10: color = "#fbbf24"
    else: color = "#6b7280"
    return f'<span style="color:{color}">{p:.4f}</span>'

def fmt_tstat(t):
    color = "#4ade80" if abs(t) > 1.96 else "#6b7280"
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

# ─────────────────────────────────────────────
# LOAD FACTOR DATA FROM GITHUB (cached)
# ─────────────────────────────────────────────
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
# HEADER
# ─────────────────────────────────────────────
st.markdown("# Factor Regression")
st.markdown(
    '<p style="font-family:\'IBM Plex Mono\',monospace;color:#4b5563;font-size:12px;letter-spacing:1px;">'
    'FF5 + MOMENTUM · HAC ROBUST STANDARD ERRORS · DATA AUTO-LOADED FROM GITHUB'
    '</p>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    ticker = st.text_input("Stock Ticker", "AVGO").upper().strip()
    st.markdown("**Date Range**")
    start_date = st.date_input("Start", value=pd.to_datetime("2010-01-01"))
    end_date   = st.date_input("End",   value=pd.to_datetime("2026-04-30"))
    st.markdown("**Regression Settings**")
    hac_lags  = st.slider("HAC Max Lags", 1, 12, 3, help="Newey-West lags for HAC robust SE")
    annualize = st.selectbox("Annualization", [12, 52, 252], help="12=monthly, 52=weekly, 252=daily")
    run = st.button("▶  RUN REGRESSION", use_container_width=True)
    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#374151;">'
        'Factor data auto-loaded from:<br>'
        '<span style="color:#4b5563;">github.com/lakshyaworkch-cell/Lakshy</span>'
        '</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# IDLE STATE
# ─────────────────────────────────────────────
if not run:
    st.markdown(
        '<div class="interpret-box" style="margin-top:40px;text-align:center;">'
        '<b>How to use</b><br><br>'
        '1. Enter a stock ticker in the sidebar<br>'
        '2. Choose your date range<br>'
        '3. Adjust HAC lags &amp; annualization if needed<br>'
        '4. Click <b>RUN REGRESSION</b><br><br>'
        '<span style="color:#374151;font-size:11px;">Factor data (FF5) is loaded automatically — no file upload needed.</span>'
        '</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
try:
    # ── Load factor data ─────────────────────
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

    # ── Stock data ───────────────────────────
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

    # ── Merge ────────────────────────────────
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

    # ── Regression ───────────────────────────
    X = sm.add_constant(data[available])
    y = data["Y"]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    # ── Stats ────────────────────────────────
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

    # ── Diagnostics ──────────────────────────
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

    # ─────────────────────────────────────────
    # DISPLAY
    # ─────────────────────────────────────────

    # ── Top metrics ──────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card {'green' if alpha_ann > 0 else 'red'}">
          <div class="metric-label">Annual Alpha</div>
          <div class="metric-value" style="color:{'#4ade80' if alpha_ann>0 else '#f87171'}">{alpha_ann:+.2%}</div>
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
        ir_color = "#4ade80" if not np.isnan(ir) and ir > 0.5 else "#fbbf24" if not np.isnan(ir) and ir > 0 else "#f87171"
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

# ── Factor Loadings ───────────────────────
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
          <div style="color:#6b7280">{se:.4f}</div>
          <div>{fmt_tstat(t)}</div>
          <div>{fmt_pval(p)}</div>
          <div>{sig_badge(p)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#334155;margin-top:6px;">'
        f'★★★ p&lt;0.01 · ★★ p&lt;0.05 · ★ p&lt;0.10 · n.s. not significant'
        f' | HAC Newey-West SE, maxlags={hac_lags}</div>', unsafe_allow_html=True)

    # ── Interpretation (always visible) ───────
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

    # ── TABS ──────────────────────────────────
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

    # ── Full OLS Summary ──────────────────────
    with st.expander("Full OLS Summary (statsmodels)"):
        st.text(model.summary().as_text())

    # ── Export ────────────────────────────────
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
