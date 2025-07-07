import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- CONFIGURE THESE PATHS ---
features_path = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\data\processed\nifty_engineered_features1.csv"
model_path = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\models\xgboost_next_day1.pkl"
scaler_path = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\models\xgboost_next_day_scaler1.pkl"

# --- LOAD DATA ---
df = pd.read_csv(features_path)
df.columns = df.columns.map(str)

# Create target if missing
if 'Target' not in df.columns:
    df['Target'] = df['Close'].shift(-1)
df = df.iloc[:-1, :]  # Drop last row (no next-day close)

# --- SPLIT INTO FEATURES AND TARGET ---
drop_cols = ['Date', 'Close', 'Target', 'Repaired?']
feature_cols = [col for col in df.columns if col not in drop_cols and pd.api.types.is_numeric_dtype(df[col])]
X = df[feature_cols].fillna(df[feature_cols].mean())
y_true = df['Target']

# --- LOAD MODEL AND SCALER ---
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# --- SCALE FEATURES ---
X_scaled = scaler.transform(X)

# --- MAKE PREDICTIONS ---
y_pred = model.predict(X_scaled)

# --- CALCULATE METRICS ---
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # <-- FIXED HERE
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
