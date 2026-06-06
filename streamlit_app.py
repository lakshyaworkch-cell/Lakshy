import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm

st.set_page_config(page_title="Factor Analyzer", layout="wide")

st.title("Fama-French Factor Analyzer")

ticker = st.text_input(
"Enter Ticker",
value="AVGO"
)

if st.button("Analyze"):

```
try:

    ff = pd.read_csv(
        "F-F_Research_Data_5_Factors_2x3.csv"
    )

    ff.rename(
        columns={ff.columns[0]: "Date"},
        inplace=True
    )

    ff = ff[
        ff["Date"]
        .astype(str)
        .str.match(r"^\d{6}$")
    ]

    ff["Date"] = pd.to_datetime(
        ff["Date"].astype(str),
        format="%Y%m"
    )

    ff.set_index(
        "Date",
        inplace=True
    )

    for col in ff.columns:
        ff[col] = pd.to_numeric(
            ff[col],
            errors="coerce"
        ) / 100

    stock = yf.download(
        ticker,
        start=ff.index.min(),
        auto_adjust=True,
        progress=False
    )

    monthly_prices = (
        stock["Close"]
        .resample("ME")
        .last()
    )

    monthly_log_returns = np.log(
        monthly_prices /
        monthly_prices.shift(1)
    )

    returns = pd.DataFrame({
        "Return": monthly_log_returns
    })

    returns.index = (
        returns.index
        .to_period("M")
        .to_timestamp()
    )

    data = returns.merge(
        ff,
        left_index=True,
        right_index=True,
        how="inner"
    )

    data["Excess_Return"] = (
        data["Return"] -
        data["RF"]
    )

    X = data[
        [
            "Mkt-RF",
            "SMB",
            "HML",
            "RMW",
            "CMA"
        ]
    ]

    X = sm.add_constant(X)

    y = data["Excess_Return"]

    model = sm.OLS(
        y,
        X,
        missing="drop"
    ).fit()

    st.subheader(
        f"{ticker.upper()} Factor Exposures"
    )

    results = pd.DataFrame({
        "Factor": model.params.index,
        "Beta": model.params.values
    })

    st.dataframe(
        results,
        use_container_width=True
    )

    st.metric(
        "R²",
        round(model.rsquared, 4)
    )

    st.subheader(
        "Regression Summary"
    )

    st.text(
        model.summary()
    )

except Exception as e:
    st.error(str(e))
```
