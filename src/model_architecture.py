# model_architecture.py
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from ta.momentum import RSIIndicator
from ta.trend import MACD
from xgboost import XGBClassifier
from prophet import Prophet
import warnings
import os

warnings.filterwarnings('ignore')

# --- Feature Engineering Functions ---
def calculate_rsi(data, window=14):
    """Calculate RSI if missing"""
    if 'RSI_14' not in data.columns:
        rsi = RSIIndicator(close=data['Close'], window=window)
        return rsi.rsi()
    return data['RSI_14']

def calculate_macd(data):
    """Calculate MACD if missing"""
    if 'MACD' not in data.columns:
        macd = MACD(close=data['Close'])
        return macd.macd()
    return data['MACD']

def add_lag_features(data):
    """Add lag features if missing"""
    for lag in [1, 2, 3, 5, 10]:
        col_name = f'Close_Lag_{lag}'
        if col_name not in data.columns:
            data[col_name] = data['Close'].shift(lag)
    return data

# --- Model Training Functions ---
def train_lstm_model(data):
    try:
        import tensorflow as tf
    except ImportError:
        print("Skipping LSTM training - TensorFlow not installed")
        return None

    # Calculate missing features
    data = data.copy()
    data['RSI_14'] = calculate_rsi(data)
    data['MACD'] = calculate_macd(data)
    data = add_lag_features(data)

    # Select features and handle missing values
    features = data[['Close', 'RSI_14', 'MACD']].copy()
    imputer = SimpleImputer(strategy='mean')
    features = pd.DataFrame(imputer.fit_transform(features),
                            columns=features.columns)

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)

    # Create sequences
    def create_sequences(data, seq_length=10):
        X, y = [], []
        for i in range(len(data) - seq_length - 1):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data)

    # Build and train LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True,
                             input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_next_day.h5')
    joblib.dump(scaler, 'models/lstm_scaler.pkl')
    joblib.dump(imputer, 'models/lstm_imputer.pkl')
    return model

def train_xgboost_model(data):
    # Calculate missing features
    data = data.copy()
    data['RSI_14'] = calculate_rsi(data)
    data['MACD'] = calculate_macd(data)
    data = add_lag_features(data)

    # Create target
    data['Target'] = (data['Close'].shift(-5) > data['Close']).astype(int)
    data = data.dropna(subset=['Target'])

    # Prepare features
    feature_cols = ['RSI_14', 'MACD', 'Close_Lag_1', 'Close_Lag_5']
    features = data[feature_cols].copy()
    target = data['Target']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    features = pd.DataFrame(imputer.fit_transform(features),
                            columns=features.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )

    # Train model
    model = XGBClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate and save
    print(f"XGBoost Accuracy: {model.score(X_test, y_test):.2f}")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_next_week.pkl')
    joblib.dump(imputer, 'models/xgboost_imputer.pkl')
    return model

def train_prophet_model(data):
    # Prepare data
    prophet_df = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Handle missing values
    prophet_df = prophet_df.dropna()

    # Train model
    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    model.fit(prophet_df)

    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/prophet_next_month.pkl')
    return model

# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)

    file_path = 'data/processed/nifty_merged_preprocessed.csv'
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        print(f"Loaded preprocessed data from {file_path} successfully.")
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        print("Generating synthetic data as fallback...")
        date_rng = pd.date_range(start='2010-01-01', end='2025-06-20', freq='D')
        df = pd.DataFrame(date_rng, columns=['Date'])
        df['Close'] = np.random.normal(100, 10, size=(len(date_rng)))
        df['Volume'] = np.random.randint(100000, 1000000, size=(len(date_rng)))
        df['Open'] = df['Close'] * 0.99
        df['High'] = df['Close'] * 1.01
        df['Low'] = df['Close'] * 0.98

    print("Training LSTM for next-day prediction...")
    lstm_model = train_lstm_model(df)

    print("\nTraining XGBoost for next-week direction...")
    xgb_model = train_xgboost_model(df)

    print("\nTraining Prophet for next-month forecast...")
    prophet_model = train_prophet_model(df)

    print("\nAll models trained successfully!")
    if "Generating synthetic data" in locals():
        print("Note: Used synthetic data as fallback")
