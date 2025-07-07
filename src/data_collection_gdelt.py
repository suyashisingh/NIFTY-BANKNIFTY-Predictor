# import requests
# import pandas as pd
# import os
# import time
#
# os.makedirs("data/raw", exist_ok=True)
#
#
# def fetch_gdelt_news(query, filename):
#     try:
#         # Add proper URL encoding and rate limiting
#         encoded_query = requests.utils.quote(f'{query} sourcelang:English')
#         url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={encoded_query}&mode=artlist&format=csv"
#
#         # Add delay to avoid rate limits
#         time.sleep(5)  # Wait 5 seconds between requests
#
#         response = requests.get(url)
#
#         # Check for error messages in response
#         if b"limit requests" in response.content.lower():
#             raise Exception("GDELT rate limit exceeded. Try again later.")
#
#         with open(filename, "wb") as f:
#             f.write(response.content)
#
#         df = pd.read_csv(filename)
#
#         if df.empty or 'V2Tone' not in df.columns:
#             raise ValueError("No valid data received from GDELT")
#
#         # Process sentiment
#         df['sentiment'] = df['V2Tone'].apply(
#             lambda x: float(str(x).split(',')[0]) if pd.notnull(x) else None
#         )
#         return df
#
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return pd.DataFrame()
#
#
# if __name__ == "__main__":
#     df = fetch_gdelt_news("NIFTY", "data/raw/gdelt_nifty_news.csv")
#
#     if not df.empty:
#         print(df[['DocumentIdentifier', 'sentiment']].head())
#     else:
#         print("No data could be retrieved. Try again later.")



import requests
import pandas as pd
import os
import time

def fetch_gdelt_news(query, filename, max_retries=5):
    """
    Fetches news articles from GDELT API for a given query and saves as CSV.
    Includes robust error and rate limit handling.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    # Example: Fetch news from Jan 1, 2024 to June 6, 2025 (customize as needed)
    startdatetime = "20240101"
    enddatetime = "20250606"
    params = {
        'query': f'{query} sourcelang:English',
        'mode': 'artlist',
        'format': 'csv',
        'startdatetime': startdatetime,
        'enddatetime': enddatetime
    }
    backoff = 60  # Start with a 1-minute wait if rate-limited

    for attempt in range(max_retries):
        try:
            print(f"Fetching GDELT data (attempt {attempt+1})...")
            response = requests.get(base_url, params=params, timeout=60)
            if response.status_code == 429 or b"limit requests" in response.content.lower():
                print("GDELT rate limit exceeded. Waiting before retrying...")
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
                continue
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

            # Validate if the response is a CSV with expected header
            content_str = response.content.decode(errors='ignore')
            if not content_str.startswith("DocumentIdentifier"):
                raise ValueError("Invalid CSV response from GDELT (no valid header).")

            with open(filename, "w", encoding="utf-8") as f:
                f.write(content_str)

            df = pd.read_csv(filename)
            if df.empty or 'V2Tone' not in df.columns:
                raise ValueError("No valid data received from GDELT.")

            # Process sentiment
            df['sentiment'] = df['V2Tone'].apply(
                lambda x: float(str(x).split(',')[0]) if pd.notnull(x) else None
            )
            print(f"Success! Retrieved {len(df)} articles.")
            return df

        except Exception as e:
            print(f"Error: {str(e)}")
            if attempt == max_retries - 1:
                print("Max retries reached. No data could be retrieved.")
                return pd.DataFrame()
            print(f"Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2  # Exponential backoff

    return pd.DataFrame()

if __name__ == "__main__":
    # Narrow query for relevance and manageability
    query = "NIFTY AND India AND stockmarket"
    output_file = "data/raw/gdelt_nifty_news.csv"
    df = fetch_gdelt_news(query, output_file)

    if not df.empty:
        print(df[['DocumentIdentifier', 'sentiment']].head())
    else:
        print("No data could be retrieved. Try again later.")
