import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm

st.title("Fama-French 5 Factor Regression")

ticker = st.text_input("Enter Stock Ticker", "AVGO")

if st.button("Run Regression"):

    try:

        # Download stock data
        stock = yf.download(
            ticker.strip().upper(),
            start="2015-01-01",
            auto_adjust=True,
            progress=False
        )

        if stock.empty:
            st.error("No stock data found.")
            st.stop()

        # Monthly returns
        monthly_prices = stock["Close"].resample("ME").last()

        monthly_returns = np.log(
            monthly_prices / monthly_prices.shift(1)
        ).dropna()

        returns_df = pd.DataFrame(monthly_returns)
        returns_df.columns = ["Return"]
        returns_df.index = returns_df.index.to_period("M")

        # Load Fama-French data
        ff = pd.read_csv("F-F_Research_Data_5_Factors_2x3.csv")

        ff.iloc[:, 0] = ff.iloc[:, 0].astype(str)

        ff = ff[
            ff.iloc[:, 0].str.match(r"^\d{6}$")
        ].copy()

        ff["Date"] = pd.to_datetime(
            ff.iloc[:, 0],
            format="%Y%m"
        )

        ff.index = ff["Date"].dt.to_period("M")

        ff.columns = [str(c).strip() for c in ff.columns]

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

        # Merge returns and factors
        merged = returns_df.join(
            ff[factor_cols],
            how="inner"
        )

        merged["Excess_Return"] = (
            merged["Return"]
            - merged["RF"]
        )

        # Regression
        X = merged[
            ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        ]

        X = sm.add_constant(X)

        y = merged["Excess_Return"]

        model = sm.OLS(y, X).fit()

        # Results table
        results = pd.DataFrame({
            "Factor": model.params.index,
            "Beta": model.params.values,
            "P-Value": model.pvalues.values
        })

        alpha = results[
            results["Factor"] == "const"
        ]["Beta"].iloc[0]

        significant = results[
            (results["Factor"] != "const")
            &
            (results["P-Value"] < 0.05)
        ]

        st.subheader("Results")

        st.write(f"Alpha: {alpha:.4f}")
        st.write(f"R²: {model.rsquared:.4f}")

        st.subheader("Significant Factors (p < 0.05)")

        if len(significant) == 0:
            st.write("No significant factors found.")
        else:
            st.dataframe(
                significant[
                    ["Factor", "Beta", "P-Value"]
                ]
                .sort_values(
                    "P-Value"
                )
            )

    except Exception as e:
        st.error(str(e))
