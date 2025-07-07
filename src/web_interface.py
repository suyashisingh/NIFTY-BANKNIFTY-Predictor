import os
import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
import numpy as np
import requests
from textblob import TextBlob
import altair as alt

# --- Custom CSS for dark, bold, centered ticker ---
st.markdown("""
  <style>
  .custom-ticker {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      background: #222831;
      color: #fff;
      font-size: 2.1rem;
      font-weight: bold;
      text-align: center;
      z-index: 9999;
      padding: 18px 0 18px 0;
      margin: 0;
      border-bottom: 2px solid #393e46;
      letter-spacing: 1px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.25);
  }
  .main .block-container {
      padding-top: 80px !important;
  }
  </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=10)
def fetch_price(ticker):
    data = yf.download(ticker, period="2d", interval="1d")
    if data.empty or 'Close' not in data.columns:
        return None, None
    current_price = float(data['Close'].iloc[-1])
    previous_close = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
    return current_price, previous_close

def live_price_ticker():
    nifty_ticker = "^NSEI"
    banknifty_ticker = "^NSEBANK"
    current_nifty, prev_nifty = fetch_price(nifty_ticker)
    current_banknifty, prev_banknifty = fetch_price(banknifty_ticker)
    if current_nifty is None or current_banknifty is None:
        st.error("Failed to fetch live prices.")
        return
    nifty_color = "#00ff00" if (current_nifty - prev_nifty) >= 0 else "#ff4d4d"
    banknifty_color = "#00ff00" if (current_banknifty - prev_banknifty) >= 0 else "#ff4d4d"
    st.markdown(f"""
    <div class="custom-ticker">
        NIFTY: <span style='color:{nifty_color};'>₹{current_nifty:,.2f} ({current_nifty - prev_nifty:+.2f})</span>
        &nbsp;&nbsp;&nbsp;&nbsp;
        BANKNIFTY: <span style='color:{banknifty_color};'>₹{current_banknifty:,.2f} ({current_banknifty - prev_banknifty:+.2f})</span>
    </div>
    """, unsafe_allow_html=True)

live_price_ticker()

# --- Sentiment Analysis for Headlines ---
st.header("📰 Stock News Sentiment Analyzer")
user_stock = st.text_input("Enter Stock Name or Symbol (e.g., TCS, RELIANCE, INFY):", "")

def fetch_headlines(stock_query):
    api_key = "861ab8cd1c9649528a9df719632ae9b6"
    url = (
        f"https://newsapi.org/v2/everything?q={stock_query}&language=en&pageSize=5&sortBy=publishedAt&apiKey={api_key}"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        return []
    data = resp.json()
    return [article['title'] for article in data.get('articles', [])]

def analyze_sentiment(text):
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    if score > 0.15:
        return "Positive ✅", f"+{score:.2f}", "green"
    elif score < -0.15:
        return "Negative ❌", f"{score:.2f}", "red"
    else:
        return "Neutral ⚠️", f"{score:.2f}", "orange"

if user_stock:
    with st.spinner(f"Fetching headlines for '{user_stock}'..."):
        headlines = fetch_headlines(user_stock)
    if not headlines:
        st.warning("No headlines found or API limit reached.")
    else:
        st.subheader(f"Latest Headlines for {user_stock.upper()}:")
        for headline in headlines:
            sentiment, score, color = analyze_sentiment(headline)
            st.markdown(
                f"<span style='color:{color}; font-weight:bold;'>{sentiment}</span> &nbsp; "
                f"<span style='color:gray;'>{headline}</span> "
                f"<span style='color:{color};'>(Score: {score})</span>",
                unsafe_allow_html=True
            )

# --- Prediction Section ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def get_model_path(filename):
    return os.path.join(MODELS_DIR, filename)

ticker_map = {
    "NIFTY": {
        "yf": "^NSEI",
        "csv": "nifty_merged_preprocessed2024.csv",
        "eng": "nifty_engineered_features2024.csv",
        "week_model": "xgboost_next_week_nifty_2023.pkl",
        "week_scaler": "xgboost_next_week_scaler_nifty_2023.pkl",
        "day_model": "xgboost_next_day_nifty_2024.pkl",
        "day_scaler": "xgboost_next_day_scaler_nifty_2024.pkl",
        "features": "nifty_next_day_feature_names_2024.txt"
    },
    "BANKNIFTY": {
        "yf": "^NSEBANK",
        "csv": "banknifty_merged_preprocessed2025.csv",
        "eng": "banknifty_engineered_features2025.csv",
        "week_model": "xgboost_next_week_banknifty2025.pkl",
        "week_scaler": "xgboost_next_week_scaler_banknifty2025.pkl"
    }
}

def get_latest_price(ticker):
    data = yf.download(ticker, period="2d", interval="1d")
    if data.empty or 'Close' not in data.columns:
        st.error(f"No data found or missing 'Close' for {ticker}")
        return None
    return float(data['Close'].iloc[-1])

def align_features(data_eng, feature_names_path):
    with open(os.path.join(MODELS_DIR, feature_names_path)) as f:
        feature_names = [line.strip() for line in f]
    for feat in feature_names:
        if feat not in data_eng.columns:
            data_eng[feat] = 0
    return data_eng[feature_names]

def predict_next_day_nifty(data_eng, scaler, model, feature_names_path, st=None):
    data_eng.columns = data_eng.columns.map(str)
    data_eng = align_features(data_eng, feature_names_path)
    latest_row = data_eng.tail(1)
    latest_row = latest_row.fillna(data_eng.mean())
    if latest_row.empty or latest_row.shape[1] == 0:
        if st:
            st.error("No valid features found for next-day prediction.")
        else:
            print("No valid features found for next-day prediction.")
        return np.nan
    try:
        latest_row_scaled = scaler.transform(latest_row)
        pred = model.predict(latest_row_scaled)
        return pred[0]
    except Exception as e:
        if st:
            st.error(f"Prediction failed: {e}")
        else:
            print(f"Prediction failed: {e}")
        return np.nan

def predict_next_day_banknifty(data_eng, scaler, model, st=None):
    data_eng.columns = data_eng.columns.map(str)
    drop_cols = ['Date', 'Close', 'Target', 'Repaired?']
    feature_cols = [col for col in data_eng.columns if col not in drop_cols and pd.api.types.is_numeric_dtype(data_eng[col])]
    latest_row = data_eng[feature_cols].tail(1)
    latest_row = latest_row.fillna(data_eng[feature_cols].mean())
    if latest_row.empty or latest_row.shape[1] == 0:
        if st:
            st.error("No valid features found for next-day prediction.")
        else:
            print("No valid features found for next-day prediction.")
        return np.nan
    try:
        latest_row_scaled = scaler.transform(latest_row)
        pred = model.predict(latest_row_scaled)
        return pred[0]
    except Exception as e:
        if st:
            st.error(f"Prediction failed: {e}")
        else:
            print(f"Prediction failed: {e}")
        return np.nan

def predict_next_week_fixed(data_eng, scaler, model, st=None):
    data_eng.columns = data_eng.columns.map(str)
    data_eng = data_eng.loc[:, ~data_eng.columns.duplicated()]
    data_eng = data_eng[[col for col in data_eng.columns if not (col.startswith("(") and col.endswith(")"))]]

    if hasattr(model, 'feature_names_in_'):
        expected_features = list(map(str, model.feature_names_in_))
    elif hasattr(scaler, 'feature_names_in_'):
        expected_features = list(map(str, scaler.feature_names_in_))
    else:
        drop_cols = ['Date', 'Close', 'Target', 'Repaired?']
        expected_features = [col for col in data_eng.columns if col not in drop_cols and pd.api.types.is_numeric_dtype(data_eng[col])]

    drop_cols = ['Date', 'Close', 'Target', 'Repaired?']
    feature_cols = [col for col in data_eng.columns if col not in drop_cols and pd.api.types.is_numeric_dtype(data_eng[col])]
    latest_row = data_eng[feature_cols].tail(1).copy()
    latest_row = latest_row.fillna(data_eng[feature_cols].mean())

    for col in expected_features:
        if col not in latest_row.columns:
            latest_row[col] = 0
    latest_row = latest_row[expected_features]

    if latest_row.empty or latest_row.shape[1] == 0:
        if st:
            st.error("No valid features found for next-week prediction.")
        else:
            print("No valid features found for next-week prediction.")
        return np.nan

    try:
        latest_row = latest_row.loc[:, ~latest_row.columns.duplicated()]
        latest_row_scaled = scaler.transform(latest_row)
        pred = model.predict(latest_row_scaled)
        return float(pred[0]) if hasattr(pred, '__len__') and not isinstance(pred, str) else float(pred)
    except Exception as e:
        if st:
            st.error(f"Weekly prediction failed: {e}")
        else:
            print(f"Weekly prediction failed: {e}")
        return np.nan

prophet_models = {
    "NIFTY": joblib.load(get_model_path("prophet_next_month_nifty.pkl")),
    "BANKNIFTY": joblib.load(get_model_path("prophet_next_month_banknifty.pkl"))
}

def predict_next_month(data, prophet_model):
    future = prophet_model.make_future_dataframe(periods=30, include_history=False)
    forecast = prophet_model.predict(future)
    return forecast.iloc[-1]['yhat']

st.title("📈 NIFTY & BANKNIFTY Predictor")

ticker = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY"], key="main_index_select")

price = get_latest_price(ticker_map[ticker]["yf"])
if price is None:
    st.stop()

data_path = os.path.join(DATA_DIR, ticker_map[ticker]["csv"])
data = pd.read_csv(data_path)
engineered_features_path = os.path.join(DATA_DIR, ticker_map[ticker]["eng"])
if os.path.exists(engineered_features_path):
    data_eng = pd.read_csv(engineered_features_path)
else:
    st.error(f"Engineered features file not found: {engineered_features_path}")
    st.stop()

week_model = joblib.load(get_model_path(ticker_map[ticker]["week_model"]))
week_scaler = joblib.load(get_model_path(ticker_map[ticker]["week_scaler"]))

if ticker == "NIFTY":
    day_model = joblib.load(get_model_path(ticker_map[ticker]["day_model"]))
    day_scaler = joblib.load(get_model_path(ticker_map[ticker]["day_scaler"]))
    feature_names_path = ticker_map[ticker]["features"]
    next_day = predict_next_day_nifty(data_eng, day_scaler, day_model, feature_names_path, st=st)
elif ticker == "BANKNIFTY":
    model = joblib.load(get_model_path('xgboost_next_day_banknifty1.pkl'))
    scaler = joblib.load(get_model_path('xgboost_next_day_scaler_banknifty1.pkl'))
    next_day = predict_next_day_banknifty(data_eng, scaler, model, st=st)

if st.button("Generate Predictions"):
    with st.spinner("Generating predictions..."):
        next_week = predict_next_week_fixed(data_eng, week_scaler, week_model, st=st)
        next_month = predict_next_month(data, prophet_models[ticker])
        # next_day already computed above

        next_day_low = next_day * (1 - 0.0025)
        next_day_high = next_day * (1 + 0.0025)
        next_week_low = next_week * (1 - 0.005)
        next_week_high = next_week * (1 + 0.005)
        next_month_low = next_month * (1 - 0.0075)
        next_month_high = next_month * (1 + 0.0075)

    st.subheader("Prediction Ranges")

    st.write("### Next Day Prediction Range")
    st.slider(
        "Next Day Range (₹)",
        min_value=float(next_day_low),
        max_value=float(next_day_high),
        value=(float(next_day_low), float(next_day_high)),
        step=0.01,
        disabled=True
    )
    st.write(f"Low: ₹{next_day_low:,.2f}   |   High: ₹{next_day_high:,.2f}")

    st.write("### Next Week Prediction Range")
    st.slider(
        "Next Week Range (₹)",
        min_value=float(next_week_low),
        max_value=float(next_week_high),
        value=(float(next_week_low), float(next_week_high)),
        step=0.01,
        disabled=True
    )
    st.write(f"Low: ₹{next_week_low:,.2f}   |   High: ₹{next_week_high:,.2f}")

    st.write("### Next Month Prediction Range")
    st.slider(
        "Next Month Range (₹)",
        min_value=float(next_month_low),
        max_value=float(next_month_high),
        value=(float(next_month_low), float(next_month_high)),
        step=0.01,
        disabled=True
    )
    st.write(f"Low: ₹{next_month_low:,.2f}   |   High: ₹{next_month_high:,.2f}")

    st.subheader("Prediction Comparison Charts")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Low Predictions")
        low_df = pd.DataFrame({
            "Prediction Type": ["Current Price", "Next Day Low", "Next Week Low", "Next Month Low"],
            "Value": [price, next_day_low, next_week_low, next_month_low]
        })
        color_scale = alt.Scale(
            domain=["Current Price", "Next Day Low", "Next Week Low", "Next Month Low"],
            range=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        )
        low_chart = alt.Chart(low_df).mark_point(filled=True, size=120).encode(
            x=alt.X('Prediction Type:N', title='Prediction Type'),
            y=alt.Y('Value:Q', title='Low Value (₹)'),
            color=alt.Color('Prediction Type:N', scale=color_scale, legend=alt.Legend(title="Prediction Type")),
            tooltip=['Prediction Type', 'Value']
        ).properties(
            width=350,
            height=400
        )
        st.altair_chart(low_chart, use_container_width=True)

    with col2:
        st.markdown("#### High Predictions")
        high_df = pd.DataFrame({
            "Prediction Type": ["Current Price", "Next Day High", "Next Week High", "Next Month High"],
            "Value": [price, next_day_high, next_week_high, next_month_high]
        })
        color_scale = alt.Scale(
            domain=["Current Price", "Next Day High", "Next Week High", "Next Month High"],
            range=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        )
        high_chart = alt.Chart(high_df).mark_point(filled=True, size=120).encode(
            x=alt.X('Prediction Type:N', title='Prediction Type'),
            y=alt.Y('Value:Q', title='High Value (₹)'),
            color=alt.Color('Prediction Type:N', scale=color_scale, legend=alt.Legend(title="Prediction Type")),
            tooltip=['Prediction Type', 'Value']
        ).properties(
            width=350,
            height=400
        )
        st.altair_chart(high_chart, use_container_width=True)
