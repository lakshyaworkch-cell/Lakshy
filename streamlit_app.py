import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm

st.title("Fama-French 5 Factor Regression")

ticker = st.text_input("Enter Stock Ticker", "AVGO")

if st.button("Run Regression"):

    try:

        # -----------------------------
        # Download stock data
        # -----------------------------
        stock = yf.download(
            ticker,
            start="2015-01-01",
            auto_adjust=True,
            progress=False
        )

        if stock.empty:
            st.error("No stock data found.")
            st.stop()

        # -----------------------------
        # Monthly returns
        # -----------------------------
        monthly_prices = stock["Close"].resample("M").last()

        monthly_returns = np.log(
            monthly_prices /
            monthly_prices.shift(1)
        )

        monthly_returns = monthly_returns.dropna()

        returns_df = pd.DataFrame(monthly_returns)
        returns_df.columns = ["Return"]

        returns_df.index = returns_df.index.to_period("M")

        # -----------------------------
        # Load FF factor file
        # -----------------------------
        ff = pd.read_csv(
            "F-F_Research_Data_5_Factors_2x3.csv"
        )

        st.write("Factor file preview")
        st.dataframe(ff.head())

        # -----------------------------
        # Convert date column
        # -----------------------------
        ff.iloc[:, 0] = ff.iloc[:, 0].astype(str)

        ff = ff[
            ff.iloc[:, 0].str.len() == 6
        ]

        ff["Date"] = pd.to_datetime(
            ff.iloc[:, 0],
            format="%Y%m"
        )

        ff.index = ff["Date"].dt.to_period("M")

        # -----------------------------
        # Rename columns if needed
        # -----------------------------
        ff.columns = [
            c.strip() for c in ff.columns
        ]

        # -----------------------------
        # Convert percentages to decimals
        # -----------------------------
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

        # -----------------------------
        # Merge
        # -----------------------------
        merged = returns_df.join(
            ff[factor_cols],
            how="inner"
        )

        # Excess Return
        merged["Excess_Return"] = (
            merged["Return"]
            - merged["RF"]
        )

        # -----------------------------
        # Regression
        # -----------------------------
        X = merged[
            ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        ]

        X = sm.add_constant(X)

        y = merged["Excess_Return"]

        model = sm.OLS(y, X).fit()

        # -----------------------------
        # Output
        # -----------------------------
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
        st.error(str(e))
