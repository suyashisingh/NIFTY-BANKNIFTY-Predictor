import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

features_path = r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/data/processed/nifty_engineered_features_2023.csv"
scaler_path = r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/models/xgboost_next_week_scaler_nifty_2023.pkl"
model_path = r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/models/xgboost_next_week_nifty_2023.pkl"

# Load and clean data
df = pd.read_csv(features_path)
df.columns = df.columns.map(str)
df = df.loc[:, ~df.columns.duplicated()]
df = df[[col for col in df.columns if not (col.startswith("(") and col.endswith(")"))]]

print("Final columns used for training:", df.columns.tolist())

# Create weekly target
df['Target_Week'] = df['Close'].shift(-5)
df = df.dropna(subset=['Target_Week'])

# Select features
drop_cols = ['Date', 'Close', 'Target', 'Target_Week', 'Repaired?']
feature_cols = [col for col in df.columns if col not in drop_cols and pd.api.types.is_numeric_dtype(df[col])]
X = df[feature_cols].fillna(df[feature_cols].mean())
y = df['Target_Week']

# Fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaler features:", scaler.feature_names_in_)
joblib.dump(scaler, scaler_path)

# Fit XGBoost regressor on DataFrame (not NumPy array!)
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
print("Model features:", model.feature_names_in_)
joblib.dump(model, model_path)

print("Weekly model and scaler retrained and saved successfully.")
