import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib

def train_and_save(df_path, model_path, scaler_path, index_name):
    print(f"\n--- {index_name} ---")
    print(f"Loading: {df_path}")
    df = pd.read_csv(df_path)
    df.columns = df.columns.map(str)

    # Create target if missing
    if 'Target' not in df.columns:
        df['Target'] = df['Close'].shift(-1)

    # Drop last row (no target possible)
    df = df.iloc[:-1, :]

    # Prepare features
    drop_cols = ['Date', 'Close', 'Target', 'Repaired?']
    feature_cols = [col for col in df.columns if col not in drop_cols and pd.api.types.is_numeric_dtype(df[col])]
    X = df[feature_cols]
    y = df['Target']

    # Only drop rows where target is NaN
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # Impute remaining NaNs in features with column mean
    X = X.fillna(X.mean())

    print(f"Shape after dropping NaNs in target: X={X.shape}, y={y.shape}")
    if X.empty or y.empty:
        print("ERROR: No data left after cleaning. Check your CSV for NaNs or misaligned columns.")
        return

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")

# --- NIFTY ---
train_and_save(
    df_path = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\data\processed\nifty_engineered_features1.csv",
    model_path = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\models\xgboost_next_day1.pkl",
    scaler_path = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\models\xgboost_next_day_scaler1.pkl",
    index_name = "NIFTY"
)

# --- BANKNIFTY ---
train_and_save(
    df_path = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\data\processed\banknifty_engineered_features1.csv",
    model_path = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\models\xgboost_next_day_banknifty1.pkl",
    scaler_path = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\models\xgboost_next_day_scaler_banknifty1.pkl",
    index_name = "BANKNIFTY"
)
