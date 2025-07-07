import yfinance as yf
import pandas as pd

# Load existing data
existing = pd.read_csv("data/processed/nifty_merged_preprocessed1.csv")


# Download new data
new_data = yf.download("^NSEI", start="2024-12-13", end="2025-07-02", interval="1d")
new_data.reset_index(inplace=True)
new_data['Date'] = new_data['Date'].dt.strftime('%d-%m-%Y')

# Append and remove duplicates
combined = pd.concat([existing, new_data], ignore_index=True)
combined = combined.drop_duplicates(subset=['Date'])
combined = combined.sort_values(by='Date')

# Save back to CSV
combined.to_csv(r"data\processed\nifty_merged_preprocessed1.csv", index=False)

