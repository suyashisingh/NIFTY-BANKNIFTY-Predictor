import pandas as pd
import numpy as np
import os

try:
    import talib
    USE_TALIB = True
except ImportError:
    USE_TALIB = False

# --- Parameters ---
INDICATOR_PERIODS = {
    'SMA': 14,
    'EMA': 14,
    'RSI': 14,
    'BBANDS': 20,
    'ATR': 14
}
LAGS = [1, 2, 3, 5, 10]

# --- Feature Engineering Functions ---
def calculate_technical_indicators(df):
    if USE_TALIB:
        df['SMA'] = talib.SMA(df['Close'], timeperiod=INDICATOR_PERIODS['SMA'])
        df['EMA'] = talib.EMA(df['Close'], timeperiod=INDICATOR_PERIODS['EMA'])
        df['RSI'] = talib.RSI(df['Close'], timeperiod=INDICATOR_PERIODS['RSI'])
        macd, macd_signal, _ = talib.MACD(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        upper, middle, lower = talib.BBANDS(
            df['Close'],
            timeperiod=INDICATOR_PERIODS['BBANDS'],
            nbdevup=2,
            nbdevdn=2
        )
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        df['ATR'] = talib.ATR(
            df['High'],
            df['Low'],
            df['Close'],
            timeperiod=INDICATOR_PERIODS['ATR']
        )
    else:
        df['SMA'] = df['Close'].rolling(window=INDICATOR_PERIODS['SMA']).mean()
        df['EMA'] = df['Close'].ewm(span=INDICATOR_PERIODS['EMA'], adjust=False).mean()
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=INDICATOR_PERIODS['RSI']).mean()
        avg_loss = loss.rolling(window=INDICATOR_PERIODS['RSI']).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        sma = df['Close'].rolling(window=INDICATOR_PERIODS['BBANDS']).mean()
        std = df['Close'].rolling(window=INDICATOR_PERIODS['BBANDS']).std()
        df['BB_Upper'] = sma + (std * 2)
        df['BB_Middle'] = sma
        df['BB_Lower'] = sma - (std * 2)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift()).abs()
        tr3 = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=INDICATOR_PERIODS['ATR']).mean()
    return df

def add_lagged_features(df):
    for lag in LAGS:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    return df

def clean_features(df):
    indicator_cols = [
        'SMA', 'EMA', 'RSI', 'MACD', 'MACD_Signal',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR'
    ]
    for col in indicator_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
    key_cols = ['Close'] + [f'Close_Lag_{lag}' for lag in LAGS] + indicator_cols
    df = df.dropna(subset=key_cols).reset_index(drop=True)
    return df

if __name__ == "__main__":
    INPUT_PATH = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\data\processed\nifty_merged_preprocessed2024.csv"
    OUTPUT_PATH = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\data\processed\nifty_engineered_features2024.csv"

    df = pd.read_csv(INPUT_PATH, parse_dates=['Date'])
    df = calculate_technical_indicators(df)
    df = add_lagged_features(df)
    df = clean_features(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Feature engineering complete. Saved to {OUTPUT_PATH}")
