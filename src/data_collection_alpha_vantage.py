import requests
import pandas as pd
import os

API_KEY = '9YCTOOOG28O5NEN6'  # Replace with your real API key
BASE_URL = 'https://www.alphavantage.co/query'

def fetch_daily_ohlcv(symbol):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY,
        'outputsize': 'full'
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    if 'Time Series (Daily)' not in data:
        raise ValueError('Invalid API response. Make sure the symbol is correct and try again.')
    ts_data = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(ts_data, orient='index')
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

if __name__ == "__main__":
    os.makedirs('data/data/raw', exist_ok=True)
    df = fetch_daily_ohlcv('TCS.BSE')  # Change symbol as needed
    print(df.head())
    df.to_csv('data/raw/tcs_alpha_vantage.csv')
