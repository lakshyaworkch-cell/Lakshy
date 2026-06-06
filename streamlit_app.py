import streamlit as st

st.title("Factor Dashboard")

ticker = st.text_input("Ticker")

if st.button("Analyze"):
    st.write("You entered:", ticker)
