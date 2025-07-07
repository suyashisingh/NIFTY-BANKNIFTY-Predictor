import pandas as pd

df = pd.read_csv("data/processed/nifty_engineered_features1.csv")

# Ensure Date is datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

print("Date range:", df['Date'].min(), "to", df['Date'].max())

df = df.sort_values(by='Date')

cutoff_date = pd.to_datetime('2023-01-01')

train = df[df['Date'] < cutoff_date]
test = df[df['Date'] >= cutoff_date]

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

train.to_csv("data/processed/nifty_train.csv", index=False)
test.to_csv("data/processed/nifty_test.csv", index=False)
