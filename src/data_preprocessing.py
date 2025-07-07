# import pandas as pd
# import os
#
#
# def safe_read_csv(path, **kwargs):
#     if os.path.exists(path):
#         try:
#             if "nse_fii_dii_flows" in path:
#                 return pd.read_csv(path, encoding='latin1', **kwargs)
#             return pd.read_csv(path, encoding='utf-8', **kwargs)
#         except UnicodeDecodeError:
#             return pd.read_csv(path, encoding='latin1', **kwargs)
#         except Exception as e:
#             print(f'Error reading {path}: {e}')
#             return pd.DataFrame()
#     else:
#         print(f'Warning: {path} not found. Skipping.')
#         return pd.DataFrame()
#
#
# # Ensure folders exist
# os.makedirs('data/raw/macro', exist_ok=True)
# os.makedirs('data/processed', exist_ok=True)
#
# # Paths to your data files
# paths = {
#     'cpi': 'data/raw/macro/india_cpi_history.csv',
#     'repo': 'data/raw/macro/rbi_repo_rate_history.csv',
#     'fii_dii': 'data/raw/macro/nse_fii_dii_flows.csv',
#     'nifty': 'data/raw/nifty_raw.csv',
#     'banknifty': 'data/raw/banknifty_raw.csv',
#     'twitter': 'data/raw/twitter_nifty_sentiment.csv',
#     'reddit': 'data/raw/reddit_nifty_sentiment.csv'
# }
#
# # Load datasets
# dfs = {name: safe_read_csv(path) for name, path in paths.items()}
#
# # --- Standardize and parse date columns ---
# for key in dfs:
#     if not dfs[key].empty:
#         # Find date column
#         date_col = next((col for col in dfs[key].columns if 'date' in col.lower()), None)
#         if not date_col:
#             print(f"Warning: No date column found in {key} data")
#             continue
#
#         # Parse dates with format detection
#         for fmt in ('%d-%m-%Y', '%Y-%m-%d', '%d-%b-%Y', '%Y/%m/%d'):
#             try:
#                 dfs[key]['Date'] = pd.to_datetime(dfs[key][date_col], format=fmt, errors='coerce')
#                 if dfs[key]['Date'].notnull().any():
#                     break
#             except:
#                 continue
#         else:
#             dfs[key]['Date'] = pd.to_datetime(dfs[key][date_col], errors='coerce')
#
# # --- Handle CPI Data ---
# if not dfs['cpi'].empty:
#     # Find and rename CPI column
#     cpi_col = next((col for col in dfs['cpi'].columns if 'cpi' in col.lower()), None)
#     if cpi_col:
#         dfs['cpi'] = dfs['cpi'].rename(columns={cpi_col: 'CPI'})
#     else:
#         print("Warning: CPI column not found in CPI data")
#
# # --- Handle Repo Rate Data ---
# if not dfs['repo'].empty:
#     # Find and rename Repo Rate column
#     repo_col = next((col for col in dfs['repo'].columns if 'repo' in col.lower()), None)
#     if repo_col:
#         dfs['repo'] = dfs['repo'].rename(columns={repo_col: 'Repo_Rate'})
#
# # --- Merge all datasets on Date ---
# df = pd.DataFrame()
# for key in ['nifty', 'banknifty', 'cpi', 'repo', 'fii_dii']:
#     if not dfs[key].empty and 'Date' in dfs[key].columns:
#         if df.empty:
#             df = dfs[key].copy()
#         else:
#             df = df.merge(dfs[key], on='Date', how='outer')
#
# # --- Handle Sentiment Data ---
# for platform in ['twitter', 'reddit']:
#     if not dfs[platform].empty and 'Date' in dfs[platform].columns:
#         sentiment_col = next((col for col in dfs[platform].columns if 'sentiment' in col.lower()), None)
#         if sentiment_col:
#             temp = dfs[platform].groupby('Date')[sentiment_col].mean().reset_index()
#             df = df.merge(temp, on='Date', how='outer')
#
# # --- Clean and fill data ---
# # Forward-fill macro/sentiment data
# macro_cols = ['CPI', 'Repo_Rate', 'FII_Net', 'DII_Net', 'sentiment']
# for col in macro_cols:
#     if col in df.columns:
#         df[col] = df[col].ffill().bfill()
#
# # Keep only rows with price data
# if 'Close' in df.columns:
#     df = df[df['Close'].notnull()]
#
# # --- Final processing ---
# df = df.sort_values('Date').reset_index(drop=True)
# df.to_csv('data/processed/merged_preprocessed_data.csv', index=False)
# print("Merged data saved successfully!")
# print(f"Total rows: {len(df)}")
# print(df.head())
import os
import numpy as np
import pandas as pd
from ta import add_all_ta_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- Utility Functions ---

def load_csv(path, parse_dates=None):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, parse_dates=parse_dates)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    else:
        print(f"File not found: {path}")
        return None

def add_technical_indicators(df):
    df = df.copy()
    # Ensure 'Volume' column exists for ta
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    return df

def add_lag_features(df, cols, lags=[1, 2, 3, 5, 10]):
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df = df.copy()  # Defragment after adding columns
    return df

def add_rolling_features(df, cols, windows=[3, 5, 10, 20]):
    for col in cols:
        for win in windows:
            df[f"{col}_rollmean{win}"] = df[col].rolling(win).mean()
            df[f"{col}_rollstd{win}"] = df[col].rolling(win).std()
    df = df.copy()  # Defragment after adding columns
    return df

def preprocess_sentiment(df, date_col, sentiment_col, freq='D'):
    if df is None or date_col not in df.columns or sentiment_col not in df.columns:
        return None
    # Try to parse as seconds since epoch, fallback to direct parsing
    try:
        df[date_col] = pd.to_datetime(df[date_col], unit='s', errors='coerce')
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    daily_sentiment = df.groupby(df[date_col].dt.date)[sentiment_col].mean().reset_index()
    daily_sentiment.columns = ['Date', f'{sentiment_col}_daily']
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'], errors='coerce')
    return daily_sentiment

def merge_on_date(main_df, other_df, how='left'):
    # Ensure both Date columns are datetime64[ns]
    main_df['Date'] = pd.to_datetime(main_df['Date'], errors='coerce')
    other_df['Date'] = pd.to_datetime(other_df['Date'], errors='coerce')
    return pd.merge(main_df, other_df, on='Date', how=how)

def scale_features(df, cols, scaler_type='standard'):
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

# --- Load All Raw Data ---

# Market data
nifty = load_csv('data/raw/nifty_raw.csv', parse_dates=['Date'])
banknifty = load_csv('data/raw/banknifty_raw.csv', parse_dates=['Date'])
tcs = load_csv('data/raw/tcs_alpha_vantage.csv', parse_dates=['Date'])

# Macro data
repo = load_csv('data/processed/fii_dii_flows_cleaned.csv', parse_dates=['Date'])
rbi = load_csv('data/raw/macro/rbi_repo_rate_history.csv')
if rbi is not None and 'Effective Date' in rbi.columns:
    # Robust date parsing for RBI data
    rbi['Date'] = pd.to_datetime(rbi['Effective Date'], format='%d-%b-%y', errors='coerce')
    mask_na = rbi['Date'].isna()
    if mask_na.any():
        rbi.loc[mask_na, 'Date'] = pd.to_datetime(rbi.loc[mask_na, 'Effective Date'], format='%b-%y', errors='coerce')
    if rbi['Date'].isna().any():
        print("Dropping rows with unparseable dates in RBI data:")
        print(rbi[rbi['Date'].isna()])
        rbi = rbi.dropna(subset=['Date'])
    rbi = rbi.sort_values('Date').reset_index(drop=True)
    rbi = rbi.drop(columns=['Effective Date'])

# Sentiment data
reddit = load_csv('data/raw/reddit_nifty_sentiment.csv')
twitter = load_csv('data/raw/twitter_nifty_sentiment.csv')
# Optional: If you have these files
newsapi = load_csv('data/raw/newsapi_nifty_articles.csv')
gdelt = load_csv('data/raw/gdelt_nifty_news.csv')

# --- Preprocess Each Data Source ---

# Clean and feature engineer market data
if nifty is not None:
    nifty = nifty.sort_values('Date').reset_index(drop=True)
    nifty = add_technical_indicators(nifty)
    nifty = add_lag_features(nifty, ['Close', 'Open', 'High', 'Low'])
    nifty = add_rolling_features(nifty, ['Close', 'Open', 'High', 'Low'])

if banknifty is not None:
    banknifty = banknifty.sort_values('Date').reset_index(drop=True)
    banknifty = add_technical_indicators(banknifty)
    banknifty = add_lag_features(banknifty, ['Close', 'Open', 'High', 'Low'])
    banknifty = add_rolling_features(banknifty, ['Close', 'Open', 'High', 'Low'])

# Preprocess sentiment data
reddit_daily = preprocess_sentiment(reddit, 'created_utc', 'sentiment') if reddit is not None else None
twitter_daily = preprocess_sentiment(twitter, 'created_at', 'sentiment') if twitter is not None else None

# Macro data: clean and align
if repo is not None:
    repo = repo.sort_values('Date').reset_index(drop=True)
    repo['Date'] = pd.to_datetime(repo['Date'], errors='coerce')
if rbi is not None:
    rbi = rbi.sort_values('Date').reset_index(drop=True)
    rbi['Date'] = pd.to_datetime(rbi['Date'], errors='coerce')

# --- Merge All Data on Date ---

main_df = nifty.copy() if nifty is not None else None

if main_df is not None:
    # Merge with macro data
    if repo is not None:
        main_df = merge_on_date(main_df, repo)
    if rbi is not None:
        main_df = merge_on_date(main_df, rbi)
    # Merge with sentiment
    if reddit_daily is not None:
        main_df = merge_on_date(main_df, reddit_daily)
    if twitter_daily is not None:
        main_df = merge_on_date(main_df, twitter_daily)
    # Merge with other data sources as available...

    # Sort and fill missing values
    main_df = main_df.sort_values('Date').reset_index(drop=True)
    main_df.fillna(method='ffill', inplace=True)
    main_df.fillna(0, inplace=True)  # For any remaining NaNs

    # Optional: scale features
    # numeric_cols = main_df.select_dtypes(include=[np.number]).columns.tolist()
    # main_df = scale_features(main_df, numeric_cols, scaler_type='standard')

    # Save processed dataset
    os.makedirs('data/processed', exist_ok=True)
    main_df.to_csv('data/processed/nifty_merged_preprocessed.csv', index=False)
    print("Preprocessing complete. Output saved to data/processed/nifty_merged_preprocessed.csv")
else:
    print("NIFTY data not found. Cannot proceed with preprocessing.")
