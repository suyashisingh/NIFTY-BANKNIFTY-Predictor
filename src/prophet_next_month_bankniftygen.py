from prophet import Prophet
import pandas as pd
import joblib

# For NIFTY
nifty_df = pd.read_csv("path/to/nifty_data.csv")  # Must have columns 'ds' (date), 'y' (close)
prophet_nifty = Prophet()
prophet_nifty.fit(nifty_df)
joblib.dump(prophet_nifty, "prophet_next_month_nifty.pkl")

# For BANKNIFTY
banknifty_df = pd.read_csv("path/to/banknifty_data.csv")
prophet_banknifty = Prophet()
prophet_banknifty.fit(banknifty_df)
joblib.dump(prophet_banknifty, "prophet_next_month_banknifty.pkl")
