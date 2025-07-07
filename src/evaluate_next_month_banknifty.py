import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- CONFIGURE PATHS ---
features_path = r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/data/processed/nifty_merged_preprocessed1.csv"
prophet_model_path = r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/models/prophet_next_month_banknifty.pkl"

# --- LOAD DATA ---
df = pd.read_csv(features_path)
df['ds'] = pd.to_datetime(df['Date'])
df['y'] = df['Close']
df = df.dropna(subset=['ds'])  # Remove rows with missing dates

# --- LOAD PROPHET MODEL ---
prophet_model = joblib.load(prophet_model_path)

# --- MAKE FORECAST FOR ALL DATES IN DATA ---
future = df[['ds']].copy()
forecast = prophet_model.predict(future)

# --- COMPARE LAST 30 DAYS ---
actuals = df['y'].values[-30:]
preds = forecast['yhat'].values[-30:]

# --- CALCULATE METRICS ---
mae = mean_absolute_error(actuals, preds)
rmse = np.sqrt(mean_squared_error(actuals, preds))
mape = np.mean(np.abs((actuals - preds) / actuals)) * 100

print(f"Next Month (last 30 days) Prediction Accuracy:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
