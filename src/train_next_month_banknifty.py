import pandas as pd
from prophet import Prophet
import joblib

# Load your BANKNIFTY data (ensure columns: 'Date', 'Close')
df = pd.read_csv(r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/data/processed/banknifty_merged_preprocessed1.csv")
df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

model = Prophet()
model.fit(df)
joblib.dump(model, r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/models/prophet_next_month_banknifty.pkl")
