import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm

st.title("Fama-French 5 Factor Regression")

ticker = st.text_input("Enter Stock Ticker", "AVGO")

if st.button("Run Regression"):

    try:

        # ----------------------------
        # Download stock data
        # ----------------------------
        stock = yf.download(
            ticker.strip().upper(),
            start="2015-01-01",
            progress=False,
            auto_adjust=True
        )

        st.subheader("Downloaded Stock Data")
        st.write(stock.head())

        if stock.empty:
            st.error(
                f"No stock data returned for ticker '{ticker}'. "
                "Try AAPL, MSFT, NVDA, META, AVGO."
            )
            st.stop()

        # ----------------------------
        # Monthly returns
        # ----------------------------
        monthly_prices = stock["Close"].resample("ME").last()

        monthly_returns = np.log(
            monthly_prices / monthly_prices.shift(1)
        )

        monthly_returns = monthly_returns.dropna()

        returns_df = pd.DataFrame(monthly_returns)
        returns_df.columns = ["Return"]

        returns_df.index = returns_df.index.to_period("M")

        # ----------------------------
        # Load Fama-French CSV
        # ----------------------------
        ff = pd.read_csv(
            "F-F_Research_Data_5_Factors_2x3.csv"
        )

        st.subheader("Raw Factor File")
        st.write(ff.head(10))

        # ----------------------------
        # Clean FF file
        # ----------------------------
        ff.iloc[:, 0] = ff.iloc[:, 0].astype(str)

        ff = ff[
            ff.iloc[:, 0].str.match(r"^\d{6}$")
        ].copy()

        ff["Date"] = pd.to_datetime(
            ff.iloc[:, 0],
            format="%Y%m"
        )

        ff.index = ff["Date"].dt.to_period("M")

        ff.columns = [
            str(c).strip()
            for c in ff.columns
        ]

        factor_cols = [
            "Mkt-RF",
            "SMB",
            "HML",
            "RMW",
            "CMA",
            "RF"
        ]

        ff[factor_cols] = (
            ff[factor_cols]
            .astype(float)
            / 100
        )

        # ----------------------------
        # Merge returns with factors
        # ----------------------------
        merged = returns_df.join(
            ff[factor_cols],
            how="inner"
        )

        st.subheader("Merged Dataset")
        st.write(merged.head())

        if merged.empty:
            st.error(
                "Merged dataset is empty. "
                "Check factor file dates."
            )
            st.stop()

        # ----------------------------
        # Excess Return
        # ----------------------------
        merged["Excess_Return"] = (
            merged["Return"]
            - merged["RF"]
        )

        X = merged[
            ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        ]

        X = sm.add_constant(X)

        y = merged["Excess_Return"]

        # ----------------------------
        # Regression
        # ----------------------------
        model = sm.OLS(y, X).fit()

        st.subheader("Factor Loadings")

        st.write(
            f"Alpha: {model.params['const']:.4f}"
        )

        st.write(
            f"Market Beta: {model.params['Mkt-RF']:.4f}"
        )

        st.write(
            f"SMB: {model.params['SMB']:.4f}"
        )

        st.write(
            f"HML: {model.params['HML']:.4f}"
        )

        st.write(
            f"RMW: {model.params['RMW']:.4f}"
        )

        st.write(
            f"CMA: {model.params['CMA']:.4f}"
        )

        st.write(
            f"R²: {model.rsquared:.4f}"
        )

        st.subheader("Regression Summary")

        st.text(model.summary())

    except Exception as e:
        st.error(f"ERROR: {str(e)}")
