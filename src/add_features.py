import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Load your data
df = pd.read_csv('data/processed/nifty_merged_preprocessed.csv')

# Add RSI_14 if missing
if 'RSI_14' not in df.columns:
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()

# Add MACD if missing
if 'MACD' not in df.columns:
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()

# You can also add lag features if needed by your model
for lag in [1, 5]:
    col_name = f'Close_Lag_{lag}'
    if col_name not in df.columns:
        df[col_name] = df['Close'].shift(lag)

# Save back to the same file (overwrites it)
df.to_csv('data/processed/nifty_merged_preprocessed.csv', index=False)
print("Features added and file updated!")
