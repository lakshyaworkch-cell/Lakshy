import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Factor Regression",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background-color: #0a0a0f;
    color: #e8e8f0;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.5px;
}

.metric-card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #4ade80;
    border-radius: 4px;
    padding: 18px 20px;
    margin-bottom: 12px;
}

.metric-card.red  { border-left-color: #f87171; }
.metric-card.blue { border-left-color: #60a5fa; }
.metric-card.gold { border-left-color: #fbbf24; }
.metric-card.gray { border-left-color: #6b7280; }

.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: #6b7280;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: #e8e8f0;
}

.metric-sub {
    font-size: 11px;
    color: #4b5563;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}

.factor-row {
    display: grid;
    grid-template-columns: 120px 90px 90px 90px 80px 1fr;
    gap: 8px;
    align-items: center;
    padding: 10px 14px;
    border-radius: 3px;
    margin-bottom: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    background: #12121a;
    border: 1px solid #1a1a28;
}

.factor-row.header {
    background: #0a0a0f;
    border-color: #1e1e2e;
    font-size: 10px;
    letter-spacing: 1.5px;
    color: #4b5563;
    text-transform: uppercase;
}

.factor-row.sig    { border-left: 3px solid #4ade80; }
.factor-row.marg   { border-left: 3px solid #fbbf24; }
.factor-row.insig  { border-left: 3px solid #374151; }
.factor-row.alpha  { border-left: 3px solid #a78bfa; }

.sig-badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 2px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
}
.badge-001 { background: #14532d; color: #4ade80; }
.badge-01  { background: #1a3a1a; color: #86efac; }
.badge-05  { background: #2a2a14; color: #fbbf24; }
.badge-10  { background: #1c1c1c; color: #9ca3af; }
.badge-ns  { background: #111118; color: #4b5563; }

.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4b5563;
    border-bottom: 1px solid #1e1e2e;
    padding-bottom: 8px;
    margin: 28px 0 16px 0;
}

.diag-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}

.diag-item {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    padding: 14px 16px;
}

.diag-name  { font-size: 11px; color: #6b7280; font-family: 'IBM Plex Mono', monospace; margin-bottom: 4px; }
.diag-val   { font-size: 16px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.diag-pass  { color: #4ade80; }
.diag-fail  { color: #f87171; }
.diag-warn  { color: #fbbf24; }
.diag-sub   { font-size: 10px; color: #374151; font-family: 'IBM Plex Mono', monospace; margin-top: 3px; }

.interpret-box {
    background: #0d0d15;
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    padding: 16px 20px;
    font-size: 13px;
    line-height: 1.7;
    color: #9ca3af;
}

.interpret-box b { color: #e8e8f0; }

.stTextInput > div > div > input {
    background: #12121a !important;
    border: 1px solid #1e1e2e !important;
    color: #e8e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 4px !important;
}

.stButton > button {
    background: #e8e8f0 !important;
    color: #0a0a0f !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 10px 28px !important;
}

.stButton > button:hover {
    background: #4ade80 !important;
}

.warning-box {
    background: #1a1500;
    border: 1px solid #fbbf24;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #fbbf24;
    margin-bottom: 12px;
}

.error-box {
    background: #1a0000;
    border: 1px solid #f87171;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #f87171;
}

div[data-testid="stFileUploader"] {
    background: #12121a;
    border: 1px dashed #1e1e2e;
    border-radius: 4px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def sig_badge(p):
    if p < 0.01:
        return '<span class="sig-badge badge-001">★★★</span>'
    elif p < 0.05:
        return '<span class="sig-badge badge-05">★★</span>'
    elif p < 0.10:
        return '<span class="sig-badge badge-10">★</span>'
    else:
        return '<span class="sig-badge badge-ns">n.s.</span>'

def row_class(name, p):
    if name == "const":
        return "factor-row alpha"
    if p < 0.05:
        return "factor-row sig"
    if p < 0.10:
        return "factor-row marg"
    return "factor-row insig"

def fmt_beta(v):
    color = "#4ade80" if v > 0 else "#f87171"
    return f'<span style="color:{color};font-weight:600">{v:+.4f}</span>'

def fmt_pval(p):
    if p < 0.001:
        color = "#4ade80"
    elif p < 0.05:
        color = "#86efac"
    elif p < 0.10:
        color = "#fbbf24"
    else:
        color = "#6b7280"
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
# HEADER
# ─────────────────────────────────────────────
st.markdown("# Factor Regression")
st.markdown(
    '<p style="font-family:\'IBM Plex Mono\',monospace;color:#4b5563;font-size:12px;letter-spacing:1px;">'
    'FF5 + MOMENTUM · HAC ROBUST STANDARD ERRORS · PROFESSIONAL DIAGNOSTICS'
    '</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR / INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")

    ticker = st.text_input("Stock Ticker", "AVGO").upper().strip()

    st.markdown("**Date Range**")
    start_date = st.date_input("Start", value=pd.to_datetime("2010-01-01"))
    end_date   = st.date_input("End",   value=pd.to_datetime("2026-04-30"))

    st.markdown("**Factor File**")
    uploaded = st.file_uploader(
        "Upload F-F CSV",
        type=["csv"],
        help="Upload your F-F_Research_Data_5_Factors_2x3.csv file"
    )

    st.markdown("**Regression Settings**")
    hac_lags = st.slider("HAC Max Lags", 1, 12, 3,
                         help="Newey-West lags for HAC robust SE")
    annualize = st.selectbox("Annualization", [12, 52, 252],
                              help="12=monthly, 52=weekly, 252=daily")

    run = st.button("▶  RUN REGRESSION", use_container_width=True)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if not run:
    st.markdown(
        '<div class="interpret-box" style="margin-top:40px;text-align:center;">'
        '<b>How to use</b><br><br>'
        '1. Enter a ticker in the sidebar<br>'
        '2. Upload your Fama-French CSV file<br>'
        '3. Adjust settings as needed<br>'
        '4. Click <b>RUN REGRESSION</b>'
        '</div>',
        unsafe_allow_html=True
    )
    st.stop()

# ── Run ──────────────────────────────────────
try:
    # ── Stock Data ───────────────────────────
    with st.spinner(f"Downloading {ticker}..."):
        raw = yf.download(
            ticker,
            start=str(start_date),
            end=str(end_date),
            auto_adjust=True,
            progress=False
        )

    if raw.empty:
        st.markdown(f'<div class="error-box">No data found for ticker: {ticker}</div>',
                    unsafe_allow_html=True)
        st.stop()

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    monthly = close.resample("ME").last()
    returns = monthly.pct_change().dropna()
    returns.index = returns.index.to_period("M")

    # ── Factor File ──────────────────────────
    if uploaded is None:
        st.markdown(
            '<div class="warning-box">⚠ No factor file uploaded. Please upload your CSV in the sidebar.</div>',
            unsafe_allow_html=True
        )
        st.stop()

    ff = pd.read_csv(uploaded, index_col=0)
    ff.columns = ff.columns.str.strip()
    ff.index   = ff.index.astype(str).str.strip()

    # Drop trailing unnamed columns (trailing commas in Ken French files)
    ff = ff.drop(columns=[c for c in ff.columns if "Unnamed" in str(c)])

    # Keep only YYYYMM rows
    ff = ff[ff.index.str.match(r"^\d{6}$")].copy()
    ff.index = pd.to_datetime(ff.index, format="%Y%m").to_period("M")

    # Filter date range
    ff = ff.loc[str(start_date)[:7]: str(end_date)[:7]]

    # Detect available factors
    available = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"] if c in ff.columns]
    has_rf    = "RF" in ff.columns

    ff[available] = ff[available].astype(float) / 100
    if has_rf:
        ff["RF"] = ff["RF"].astype(float) / 100

    # ── Merge ────────────────────────────────
    data = pd.DataFrame({"Stock": returns}).join(ff[available + (["RF"] if has_rf else [])], how="inner")

    if len(data) < 24:
        st.markdown(
            '<div class="error-box">Too few observations after merge. Check date range and file.</div>',
            unsafe_allow_html=True
        )
        st.stop()

    # ── Excess Return ────────────────────────
    if has_rf:
        data["Y"] = data["Stock"] - data["RF"]
        y_label = "Excess Return (Stock − Rf)"
    else:
        data["Y"] = data["Stock"]
        y_label = "Raw Return (no RF in file)"

    # ── Regression ───────────────────────────
    X = sm.add_constant(data[available])
    y = data["Y"]

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags}
    )

    # ── Derived Stats ────────────────────────
    alpha        = model.params["const"]
    alpha_ann    = (1 + alpha) ** annualize - 1
    alpha_t      = model.tvalues["const"]
    alpha_p      = model.pvalues["const"]

    n            = int(model.nobs)
    k            = len(model.params) - 1
    r2           = model.rsquared
    r2_adj       = model.rsquared_adj
    f_stat       = model.fvalue
    f_p          = model.f_pvalue
    aic          = model.aic
    bic          = model.bic
    resid        = model.resid

    # Information ratio
    te           = resid.std() * np.sqrt(annualize)
    ir           = alpha_ann / te if te > 0 else np.nan

    # ── Diagnostics ──────────────────────────
    # Breusch-Pagan heteroscedasticity
    bp_stat, bp_p, _, _ = sm.stats.diagnostic.het_breuschpagan(resid, X)

    # Durbin-Watson autocorrelation
    dw = sm.stats.stattools.durbin_watson(resid)

    # Jarque-Bera normality
    jb_stat, jb_p = stats.jarque_bera(resid)[:2]

    # Condition number (multicollinearity)
    cond = np.linalg.cond(X.values)

    # VIF
    vif_data = {}
    for col in available:
        others = [c for c in available if c != col]
        r2_vif = sm.OLS(X[col], sm.add_constant(X[others])).fit().rsquared
        vif_data[col] = 1 / (1 - r2_vif) if r2_vif < 1 else np.inf

    # ─────────────────────────────────────────
    # DISPLAY
    # ─────────────────────────────────────────

    # ── Top Metrics ──────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    alpha_color = "green" if alpha_ann > 0 else "red"
    with c1:
        st.markdown(f"""
        <div class="metric-card {'green' if alpha_ann > 0 else 'red'}">
          <div class="metric-label">Annual Alpha</div>
          <div class="metric-value" style="color:{'#4ade80' if alpha_ann>0 else '#f87171'}">
            {alpha_ann:+.2%}
          </div>
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
        st.markdown(f"""
        <div class="metric-card gold">
          <div class="metric-label">Information Ratio</div>
          <div class="metric-value" style="color:{ir_color}">{ir:.3f}</div>
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

    header = """
    <div class="factor-row header">
      <div>FACTOR</div><div>BETA</div><div>STD ERR</div>
      <div>T-STAT</div><div>P-VALUE</div><div>SIG</div>
    </div>"""
    st.markdown(header, unsafe_allow_html=True)

    for name in ["const"] + available:
        b   = model.params[name]
        se  = model.bse[name]
        t   = model.tvalues[name]
        p   = model.pvalues[name]
        rc  = row_class(name, p)
        label = FACTOR_NAMES.get(name, name)

        row = f"""
        <div class="{rc}">
          <div style="color:#e8e8f0;font-weight:600">{label}</div>
          <div>{fmt_beta(b)}</div>
          <div style="color:#6b7280">{se:.4f}</div>
          <div>{fmt_tstat(t)}</div>
          <div>{fmt_pval(p)}</div>
          <div>{sig_badge(p)}</div>
        </div>"""
        st.markdown(row, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#374151;margin-top:6px;">'
        '★★★ p&lt;0.01 &nbsp;·&nbsp; ★★ p&lt;0.05 &nbsp;·&nbsp; ★ p&lt;0.10 &nbsp;·&nbsp; n.s. not significant'
        '&nbsp;&nbsp;|&nbsp;&nbsp;HAC Newey-West SE, maxlags=' + str(hac_lags) +
        '</div>',
        unsafe_allow_html=True
    )

    # ── Diagnostics ───────────────────────────
    st.markdown('<div class="section-title">Regression Diagnostics</div>', unsafe_allow_html=True)

    def diag_card(name, val, sub, status):
        cls = {"pass": "diag-pass", "fail": "diag-fail", "warn": "diag-warn"}.get(status, "diag-pass")
        return f"""
        <div class="diag-item">
          <div class="diag-name">{name}</div>
          <div class="diag-val {cls}">{val}</div>
          <div class="diag-sub">{sub}</div>
        </div>"""

    dw_status = "pass" if 1.5 < dw < 2.5 else "warn" if 1.2 < dw < 2.8 else "fail"
    bp_status = "pass" if bp_p > 0.05 else "warn" if bp_p > 0.01 else "fail"
    jb_status = "pass" if jb_p > 0.05 else "warn" if jb_p > 0.01 else "fail"
    cn_status = "pass" if cond < 30 else "warn" if cond < 100 else "fail"

    st.markdown(f"""
    <div class="diag-grid">
      {diag_card("Durbin-Watson", f"{dw:.4f}", "Autocorrelation · ideal ≈ 2.0", dw_status)}
      {diag_card("Breusch-Pagan", f"p = {bp_p:.4f}", f"Heteroscedasticity · stat={bp_stat:.3f}", bp_status)}
      {diag_card("Jarque-Bera", f"p = {jb_p:.4f}", f"Residual normality · stat={jb_stat:.3f}", jb_status)}
      {diag_card("Condition Number", f"{cond:.1f}", "Multicollinearity · ideal < 30", cn_status)}
    </div>
    """, unsafe_allow_html=True)

    # ── VIF ───────────────────────────────────
    st.markdown('<div class="section-title">Variance Inflation Factors</div>', unsafe_allow_html=True)

    vif_html = '<div style="display:flex;gap:10px;flex-wrap:wrap;">'
    for col, v in vif_data.items():
        vif_color = "#4ade80" if v < 5 else "#fbbf24" if v < 10 else "#f87171"
        vif_html += f"""
        <div style="background:#12121a;border:1px solid #1e1e2e;border-radius:4px;padding:12px 16px;min-width:100px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#4b5563;margin-bottom:4px;">{FACTOR_NAMES.get(col,col)}</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;font-weight:600;color:{vif_color}">{v:.2f}</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#374151;margin-top:2px;">
            {'OK' if v < 5 else 'MODERATE' if v < 10 else 'HIGH'}
          </div>
        </div>"""
    vif_html += '</div>'
    st.markdown(vif_html, unsafe_allow_html=True)

    # ── Model Fit ────────────────────────────
    st.markdown('<div class="section-title">Model Fit</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
      <div class="diag-item" style="min-width:140px;">
        <div class="diag-name">AIC</div>
        <div class="diag-val" style="color:#60a5fa;font-size:16px;">{aic:.2f}</div>
      </div>
      <div class="diag-item" style="min-width:140px;">
        <div class="diag-name">BIC</div>
        <div class="diag-val" style="color:#60a5fa;font-size:16px;">{bic:.2f}</div>
      </div>
      <div class="diag-item" style="min-width:140px;">
        <div class="diag-name">Log-Likelihood</div>
        <div class="diag-val" style="color:#60a5fa;font-size:16px;">{model.llf:.2f}</div>
      </div>
      <div class="diag-item" style="min-width:140px;">
        <div class="diag-name">Residual Std</div>
        <div class="diag-val" style="color:#60a5fa;font-size:16px;">{resid.std():.4f}</div>
      </div>
      <div class="diag-item" style="min-width:140px;">
        <div class="diag-name">Skewness</div>
        <div class="diag-val" style="color:#60a5fa;font-size:16px;">{float(stats.skew(resid)):.4f}</div>
      </div>
      <div class="diag-item" style="min-width:140px;">
        <div class="diag-name">Kurtosis</div>
        <div class="diag-val" style="color:#60a5fa;font-size:16px;">{float(stats.kurtosis(resid)):.4f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Interpretation ───────────────────────
    st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)

    sig_factors = [FACTOR_NAMES.get(f, f) for f in available if model.pvalues[f] < 0.05]
    insig_factors = [FACTOR_NAMES.get(f, f) for f in available if model.pvalues[f] >= 0.05]
    alpha_sig = "statistically significant" if alpha_p < 0.05 else "not statistically significant"
    alpha_interp = "outperforms" if alpha_ann > 0 else "underperforms"

    interp = f"""
    <div class="interpret-box">
    <b>{ticker}</b> — {y_label}<br><br>
    The model explains <b>{r2:.1%}</b> of return variation (Adj R² = {r2_adj:.1%}) over {n} monthly observations.<br><br>
    The annualized alpha of <b>{alpha_ann:+.2%}</b> is {alpha_sig} (t = {alpha_t:.3f}, p = {alpha_p:.4f}),
    suggesting the stock <b>{alpha_interp}</b> what factor exposures alone would predict.<br><br>
    """

    if sig_factors:
        interp += f"<b>Significant factors (p&lt;0.05):</b> {', '.join(sig_factors)}.<br>"
    if insig_factors:
        interp += f"<b>Insignificant factors:</b> {', '.join(insig_factors)}.<br>"

    mkt_b = model.params.get("Mkt-RF", None)
    if mkt_b is not None:
        interp += f"<br>Market beta of <b>{mkt_b:.4f}</b> implies the stock is "
        interp += "more volatile than the market." if mkt_b > 1 else "less volatile than the market."

    if dw_status != "pass":
        interp += f"<br><br>⚠ Durbin-Watson ({dw:.3f}) suggests possible autocorrelation in residuals. HAC correction applied."
    if bp_status != "pass":
        interp += f"<br>⚠ Breusch-Pagan test (p={bp_p:.4f}) indicates heteroscedasticity. HAC robust SEs account for this."

    interp += "</div>"
    st.markdown(interp, unsafe_allow_html=True)

    # ── Raw Summary ───────────────────────────
    with st.expander("Full OLS Summary (statsmodels)"):
        st.text(model.summary().as_text())

    # ── Export ───────────────────────────────
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)

    export_df = pd.DataFrame({
        "Factor":   [FACTOR_NAMES.get(n, n) for n in model.params.index],
        "Beta":     model.params.values,
        "Std_Err":  model.bse.values,
        "T_Stat":   model.tvalues.values,
        "P_Value":  model.pvalues.values,
        "CI_Lower": model.conf_int()[0].values,
        "CI_Upper": model.conf_int()[1].values,
    })

    st.download_button(
        "⬇  Download Results CSV",
        export_df.to_csv(index=False),
        file_name=f"{ticker}_factor_regression.csv",
        mime="text/csv"
    )

except Exception as e:
    st.markdown(
        f'<div class="error-box">Error: {str(e)}</div>',
        unsafe_allow_html=True
    )
    raise e
