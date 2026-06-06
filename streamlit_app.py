I am building a Streamlit website that performs a Fama-French factor regression.

Current status:

* I have a GitHub repository connected to Streamlit Community Cloud.
* My app file is named `streamlit_app.py`.
* I have successfully deployed a basic Streamlit app.
* I have uploaded the file `F-F_Research_Data_5_Factors_2x3.csv` to my GitHub repository.
* I want users to enter a stock ticker (e.g. AVGO, NVDA, META).
* The app should download historical stock prices using yfinance.
* Calculate monthly log returns.
* Merge them with my Fama-French factor data.
* Run an OLS regression using statsmodels.
* Display:

  * Alpha
  * Market Beta (Mkt-RF)
  * SMB
  * HML
  * RMW
  * CMA
  * R²
  * Full regression summary

Important:

* I am a beginner with Streamlit and GitHub.
* Explain exactly where to paste code and which files to create/edit.
* If there is an error, help me debug it step-by-step.
* Assume my factor file may only contain the standard Fama-French 5 factors plus RF.
* Ask me for any error messages before changing the code.

Please continue from this point and help me get the regression working on my live Streamlit website.
