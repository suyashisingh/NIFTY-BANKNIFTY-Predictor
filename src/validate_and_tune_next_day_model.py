import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import joblib
import os

# 1. Load data
DATA_PATH = "data/processed/nifty_engineered_features1.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df['Target'] = df['Close'].shift(-1)
df = df.dropna(subset=['Target'])

# 2. Split train/test
train = df[df['Date'] < '2023-01-01']
test = df[df['Date'] >= '2023-01-01']

feature_cols = [col for col in df.columns if col not in ['Date', 'Close', 'Target']]
X_train = train[feature_cols].copy()
X_test = test[feature_cols].copy()
y_train = train['Target']
y_test = test['Target']

# 3. Handle non-numeric columns and NaNs
non_numeric_cols = X_train.select_dtypes(include=['object']).columns.tolist()
for col in non_numeric_cols:
    try:
        X_train[col] = X_train[col].astype(str).str.replace('%', '').astype(float)
        X_test[col] = X_test[col].astype(str).str.replace('%', '').astype(float)
    except Exception as e:
        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

nan_threshold = 0.2
cols_to_drop = [col for col in X_train.columns if X_train[col].isna().mean() > nan_threshold]
X_train = X_train.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train initial model
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Initial Test RMSE: {rmse:.2f}")
print(f"Initial Test MAE: {mae:.2f}")
print(f"Initial Test MAPE: {mape:.2f}%")

# 6. Visualize predictions vs actuals
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Next-Day Prediction: Actual vs Predicted')
plt.show()

# 7. Hyperparameter tuning with GridSearchCV and TimeSeriesSplit
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=1)
grid.fit(X_train_scaled, y_train)
print("Best parameters:", grid.best_params_)
print("Best CV score (RMSE):", np.sqrt(-grid.best_score_))

# 8. Retrain with best parameters and evaluate
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
mae_best = mean_absolute_error(y_test, y_pred_best)
mape_best = np.mean(np.abs((y_test - y_pred_best) / y_test)) * 100

print(f"Tuned Test RMSE: {rmse_best:.2f}")
print(f"Tuned Test MAE: {mae_best:.2f}")
print(f"Tuned Test MAPE: {mape_best:.2f}%")

# 9. Cross-validation (optional, but recommended)
cv_scores = cross_val_score(
    best_model, X_train_scaled, y_train, cv=tscv, scoring='neg_mean_squared_error'
)
cv_rmses = np.sqrt(-cv_scores)
print("Cross-validated RMSEs:", cv_rmses)
print("Mean CV RMSE:", np.mean(cv_rmses))

# 10. Save best model and scaler
joblib.dump(best_model, os.path.join(MODEL_DIR, "xgboost_next_day.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "xgboost_next_day_scaler.pkl"))
print("Tuned model and scaler saved to models/")
