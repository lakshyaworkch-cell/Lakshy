import streamlit as st
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

st.title("Fama-French Factor Regression")

ticker = st.text_input("Enter Stock Ticker", "AVGO")

if st.button("Run Regression"):

    try:

        # ----------------------------
        # Download stock data
        # ----------------------------

        stock = yf.download(
            ticker.upper(),
            start="2015-01-01",
            auto_adjust=True,
            progress=False
        )

        if stock.empty:
            st.error("No stock data found.")
            st.stop()

        close = stock["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        monthly_prices = close.resample("ME").last()

        monthly_returns = monthly_prices.pct_change().dropna()

        returns_df = pd.DataFrame({
            "Return": monthly_returns
        })

        returns_df.index = returns_df.index.to_period("M")

        # ----------------------------
        # Load factor file
        # ----------------------------

        ff = pd.read_csv(
            "F-F_Research_Data_5_Factors_2x3.csv"
        )

        # Rename first column to Date
        ff.rename(
            columns={ff.columns[0]: "Date"},
            inplace=True
        )

        # Keep only rows with YYYYMM dates
        ff = ff[
            ff["Date"].astype(str).str.match(r"^\d{6}$")
        ].copy()

        ff["Date"] = pd.to_datetime(
            ff["Date"].astype(str),
            format="%Y%m"
        )

        ff.set_index("Date", inplace=True)

        ff.index = ff.index.to_period("M")

        # Convert factors to decimals
        factor_cols = [
            "Mkt-RF",
            "SMB",
            "HML",
            "RMW",
            "CMA",
            "RF",
            "Mom"
        ]

        ff[factor_cols] = (
            ff[factor_cols]
            .astype(float)
            / 100
        )

        # ----------------------------
        # Merge
        # ----------------------------

        data = returns_df.join(
            ff,
            how="inner"
        )

        if len(data) == 0:
            st.error("No matching dates found.")
            st.stop()

        # ----------------------------
        # Excess Return
        # ----------------------------

        data["Excess_Return"] = (
            data["Return"]
            - data["RF"]
        )

        # ----------------------------
        # 6 Factor Regression
        # ----------------------------

        X = data[
            [
                "Mkt-RF",
                "SMB",
                "HML",
                "RMW",
                "CMA",
                "Mom"
            ]
        ]

        X = sm.add_constant(X)

        y = data["Excess_Return"]

        model = sm.OLS(
            y,
            X
        ).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 3}
        )

        # ----------------------------
        # Results
        # ----------------------------

        results = pd.DataFrame({
            "Factor": model.params.index,
            "Beta": model.params.values,
            "P-Value": model.pvalues.values
        })

        alpha = results.loc[
            results["Factor"] == "const",
            "Beta"
        ].iloc[0]

        st.subheader("Summary")

        st.write(f"Alpha: {alpha:.4f}")
        st.write(f"R²: {model.rsquared:.4f}")
        st.write(f"Observations: {len(data)}")

        significant = results[
            (results["Factor"] != "const")
            &
            (results["P-Value"] < 0.05)
        ]

        st.subheader("Significant Factors")

        if len(significant) == 0:
            st.write("No significant factors found.")
        else:
            st.dataframe(
                significant[
                    ["Factor", "Beta", "P-Value"]
                ].sort_values("P-Value")
            )

    except Exception as e:
        st.error(str(e))
