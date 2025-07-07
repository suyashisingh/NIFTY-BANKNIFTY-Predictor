import pandas as pd
import os
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

features_path = r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/data/processed/nifty_engineered_features2.csv"
model_path = r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/models/xgboost_next_week2.pkl"
scaler_path = r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/models/xgboost_next_week_scaler2.pkl"

print(os.path.exists(features_path), features_path)
print(os.path.exists(model_path), model_path)
print(os.path.exists(scaler_path), scaler_path)

# Load and clean DataFrame
df = pd.read_csv(features_path)
df.columns = df.columns.map(str)
df = df.loc[:, ~df.columns.duplicated()]
# Remove tuple-looking columns
df = df[[col for col in df.columns if not (col.startswith("(") and col.endswith(")"))]]
print("Final columns used for training:", df.columns.tolist())

# Create weekly target: next week's close (shift -5)
df['Target_Week'] = df['Close'].shift(-5)
df = df.dropna(subset=['Target_Week'])

# Use ONLY the features your model was trained on!
feature_cols = ['momentum_rsi', 'trend_macd', 'Close_Lag_1', 'Close_Lag_5']  # <-- update if you trained on more features
X = df[feature_cols].fillna(df[feature_cols].mean())
y_true = df['Target_Week']

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Scale features
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"Next Week Prediction Accuracy:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
