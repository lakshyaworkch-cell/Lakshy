import streamlit as st
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from datetime import date
from dateutil.relativedelta import relativedelta

st.title("Fama-French 6 Factor Regression (Robust HAC)")

ticker = st.text_input("Ticker", "AVGO")

if st.button("Run Regression"):

    try:

        END_DATE = date.today()
        START_DATE = END_DATE - relativedelta(years=10)

        # ------------------------------------
        # Download stock data
        # ------------------------------------
        prices = yf.download(
            ticker.upper(),
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False
        )

        prices = prices["Close"]

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name="Stock")
        else:
            prices.columns = ["Stock"]

        returns = prices.pct_change().dropna()

        # ------------------------------------
        # FF5 Daily Factors
        # ------------------------------------
        ff = pd.read_csv(
            "F-F_Research_Data_5_Factors_2x3_daily.csv",
            skiprows=3
        )

        ff.columns = [
            "Date",
            "Mkt-RF",
            "SMB",
            "HML",
            "RMW",
            "CMA",
            "RF"
        ]

        ff = ff[
            ff["Date"].astype(str).str.isnumeric()
        ]

        ff["Date"] = pd.to_datetime(
            ff["Date"],
            format="%Y%m%d"
        )

        ff.set_index("Date", inplace=True)

        ff = ff / 100

        ff = ff.loc[
            str(START_DATE):
            str(END_DATE)
        ]

        # ------------------------------------
        # Momentum
        # ------------------------------------
        mom = pd.read_csv(
            "F-F_Momentum_Factor_daily.csv",
            skiprows=13
        )

        mom.columns = [
            "Date",
            "MOM"
        ]

        mom = mom[
            mom["Date"].astype(str).str.isnumeric()
        ]

        mom["Date"] = pd.to_datetime(
            mom["Date"],
            format="%Y%m%d"
        )

        mom.set_index("Date", inplace=True)

        mom = mom / 100

        mom = mom.loc[
            str(START_DATE):
            str(END_DATE)
        ]

        # ------------------------------------
        # Merge
        # ------------------------------------
        data = returns.merge(
            ff,
            left_index=True,
            right_index=True,
            how="inner"
        )

        data = data.merge(
            mom,
            left_index=True,
            right_index=True,
            how="inner"
        )

        data["Excess_Stock"] = (
            data["Stock"]
            - data["RF"]
        )

        # ------------------------------------
        # Regression
        # ------------------------------------
        X = data[
            [
                "Mkt-RF",
                "SMB",
                "HML",
                "RMW",
                "CMA",
                "MOM"
            ]
        ]

        X = sm.add_constant(X)

        y = data["Excess_Stock"]

        model = sm.OLS(
            y,
            X
        ).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 5}
        )

        # ------------------------------------
        # Results
        # ------------------------------------
        results = pd.DataFrame({
            "Factor": model.params.index,
            "Beta": model.params.values,
            "P-Value": model.pvalues.values
        })

        alpha = results[
            results["Factor"] == "const"
        ]["Beta"].iloc[0]

        st.subheader("Alpha")
        st.write(f"{alpha:.6f}")

        st.subheader("R²")
        st.write(f"{model.rsquared:.4f}")

        significant = results[
            (results["Factor"] != "const")
            &
            (results["P-Value"] < 0.05)
        ]

        st.subheader(
            "Significant Factors (p < 0.05)"
        )

        if len(significant) == 0:
            st.write(
                "No significant factors found."
            )
        else:
            st.dataframe(
                significant.sort_values(
                    "P-Value"
                )
            )

    except Exception as e:
        st.error(str(e))
