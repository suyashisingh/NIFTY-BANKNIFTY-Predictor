import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# --- Paths ---
INPUT_PATH = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\data\processed\nifty_engineered_features2024.csv"
MODEL_DIR = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load data ---
df = pd.read_csv(INPUT_PATH, parse_dates=['Date'])
print(f"Loaded data shape: {df.shape}")

# --- Create next-day target ---
df['Target'] = df['Close'].shift(-1)
df = df.dropna(subset=['Target'])

# --- Feature selection ---
drop_cols = ['Date', 'Close', 'Target']
feature_cols = [col for col in df.columns if col not in drop_cols and pd.api.types.is_numeric_dtype(df[col])]

# --- Ensure all feature names are strings ---
feature_cols = [str(col) for col in feature_cols]

# --- Train/test split ---
split_date = '2025-01-01'  # Adjust as needed
train = df[df['Date'] < split_date]
test = df[df['Date'] >= split_date]
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

X_train = train[feature_cols].copy()
X_test = test[feature_cols].copy()
y_train = train['Target']
y_test = test['Target']

# --- Handle non-numeric columns robustly ---
non_numeric_cols = X_train.select_dtypes(include=['object']).columns.tolist()
for col in non_numeric_cols:
    try:
        X_train[col] = X_train[col].astype(str).str.replace('%', '').astype(float)
        X_test[col] = X_test[col].astype(str).str.replace('%', '').astype(float)
    except Exception:
        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

# --- Drop columns with too many NaNs ---
nan_threshold = 0.2
cols_to_drop = [col for col in X_train.columns if X_train[col].isna().mean() > nan_threshold]
X_train = X_train.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)

# --- Impute remaining NaNs with column mean ---
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

# --- Ensure all columns are strings (again, for safety) ---
X_train.columns = X_train.columns.map(str)
X_test.columns = X_test.columns.map(str)

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train XGBoost model ---
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# --- Evaluate model ---
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE: {mae:.2f}")

# --- Save model and scaler ---
model_path = os.path.join(MODEL_DIR, 'xgboost_next_day_nifty_2024.pkl')
scaler_path = os.path.join(MODEL_DIR, 'xgboost_next_day_scaler_nifty_2024.pkl')
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
