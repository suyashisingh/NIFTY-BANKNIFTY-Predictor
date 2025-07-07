import os
import pandas as pd

base_dir = r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/data/processed"
file1 = os.path.join(base_dir, "banknifty_merged_preprocessed.csv")
file2 = os.path.join(base_dir, "banknifty_merged_preprocessed1.csv")
output = os.path.join(base_dir, "banknifty_merged_preprocessed1.csv")

def load_csv_with_header(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"File not found or empty: {path}")
        return pd.DataFrame()
    with open(path, 'r') as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.lower().startswith('date,'):
            header_idx = i
            break
    if header_idx is not None:
        df = pd.read_csv(path, skiprows=header_idx)
    else:
        print(f"No valid header found in {path}. Skipping this file.")
        return pd.DataFrame()
    if 'Date' not in df.columns:
        print(f"'Date' column missing in {path}. Skipping this file.")
        return pd.DataFrame()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    return df

df1 = load_csv_with_header(file1)
df2 = load_csv_with_header(file2)

if df1.empty and df2.empty:
    print("Both files are missing, empty, or malformed. Nothing to combine.")
else:
    combined = pd.concat([df1, df2], ignore_index=True)
    combined = combined.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
    combined.to_csv(output, index=False)
    print(f"Combined BANKNIFTY data saved to {output}")
    print(f"Final shape: {combined.shape}")
    print("First few rows:")
    print(combined.head())
