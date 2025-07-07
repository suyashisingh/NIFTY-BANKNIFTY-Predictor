# # sentiment_analysis.py
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from transformers import BertTokenizer, BertForSequenceClassification
# import torch
#
#
# class SentimentAnalyzer:
#     def __init__(self):
#         # Initialize VADER
#         self.vader_analyzer = SentimentIntensityAnalyzer()
#
#         # Initialize FinBERT
#         self.finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
#         self.finbert_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
#         self.finbert_model.eval()  # Set to evaluation mode
#
#     def vader_sentiment(self, text):
#         """Returns VADER compound sentiment score between -1 (negative) and 1 (positive)"""
#         scores = self.vader_analyzer.polarity_scores(text)
#         return scores['compound']
#
#     def finbert_sentiment(self, text):
#         """Returns FinBERT sentiment: 0=Neutral, 1=Positive, 2=Negative"""
#         inputs = self.finbert_tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=512,
#             padding=True
#         )
#
#         with torch.no_grad():  # Disable gradient calculation
#             outputs = self.finbert_model(**inputs)
#
#         sentiment = torch.argmax(outputs.logits).item()
#         return sentiment
#
#     def analyze(self, text):
#         """Get both VADER and FinBERT sentiment scores"""
#         return {
#             "text": text,
#             "vader": self.vader_sentiment(text),
#             "finbert": self.finbert_sentiment(text)
#         }
#
#
# # Example usage
# if __name__ == "__main__":
#     analyzer = SentimentAnalyzer()
#
#     samples = [
#         "The stock market is showing strong growth today.",
#         "Company profits plummeted amid economic downturn.",
#         "Fed announces interest rate hike of 0.25%",
#         "Tesla shares surge 15% after record earnings report"
#     ]
#
#     for text in samples:
#         result = analyzer.analyze(text)
#         print(f"Text: {text}")
#         print(
#             f"VADER: {result['vader']:.4f} (Score: {'Positive' if result['vader'] > 0 else 'Negative' if result['vader'] < 0 else 'Neutral'})")
#         print(f"FinBERT: {result['finbert']} ({['Neutral', 'Positive', 'Negative'][result['finbert']]})")
#         print("-" * 80)



#
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import time

# --- USER CONFIGURATION ---
NEWSAPI_KEY = "861ab8cd1c9649528a9df719632ae9b6"  # <-- Replace with your NewsAPI key

# --- Sentiment Analyzer Classes ---
class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        # Load FinBERT model and tokenizer
        self.finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.finbert_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        self.finbert_model.eval()

    def vader_sentiment(self, text):
        return self.vader.polarity_scores(text)['compound']

    def finbert_sentiment(self, text):
        inputs = self.finbert_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            outputs = self.finbert_model(**inputs)
        return torch.argmax(outputs.logits).item()  # 0: Neutral, 1: Positive, 2: Negative

    def analyze(self, text):
        return {
            "text": text,
            "vader": self.vader_sentiment(text),
            "finbert": self.finbert_sentiment(text)
        }

# --- Fetch News Headlines ---
def fetch_news_headlines(query, api_key, page_size=20):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data.get("status") != "ok":
        print("Error fetching news:", data.get("message", "Unknown error"))
        return []
    headlines = [article["title"] for article in data["articles"] if article.get("title")]
    return headlines

# --- Main Workflow ---
if __name__ == "__main__":
    # 1. Get user input
    stock_name = input("Enter the stock/company name to analyze sentiment for: ").strip()
    if not stock_name:
        print("No stock name entered. Exiting.")
        exit(1)

    # 2. Fetch news headlines
    print(f"\nFetching latest news headlines for '{stock_name}'...")
    headlines = fetch_news_headlines(stock_name, NEWSAPI_KEY)
    if not headlines:
        print("No news headlines found or error fetching news.")
        exit(1)

    # 3. Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    time.sleep(1)  # Give the model a moment to load

    # 4. Analyze sentiment for each headline
    results = []
    print(f"\nAnalyzing {len(headlines)} headlines...")
    for headline in headlines:
        sentiment = analyzer.analyze(headline)
        results.append(sentiment)
        vader_score = sentiment["vader"]
        finbert_score = sentiment["finbert"]
        vader_label = "Positive" if vader_score > 0.05 else "Negative" if vader_score < -0.05 else "Neutral"
        finbert_label = ["Neutral", "Positive", "Negative"][finbert_score]
        print(f"\nHeadline: {headline}")
        print(f"  VADER: {vader_score:.3f} ({vader_label})")
        print(f"  FinBERT: {finbert_score} ({finbert_label})")

    # 5. Summarize sentiment
    avg_vader = sum(r['vader'] for r in results) / len(results)
    finbert_counts = [0, 0, 0]
    for r in results:
        finbert_counts[r['finbert']] += 1
    print("\n--- Sentiment Summary ---")
    print(f"Average VADER score: {avg_vader:.3f}")
    print(f"FinBERT: {finbert_counts[1]} Positive, {finbert_counts[2]} Negative, {finbert_counts[0]} Neutral out of {len(results)} headlines.")

    print("\nDone.")




# import tweepy
# import pandas as pd
# import matplotlib.pyplot as plt
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from transformers import BertTokenizer, BertForSequenceClassification
# import torch
# import datetime
#
# # --- USER CONFIGURATION ---
# TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAK2s2AEAAAAATNGeEJ33f2o399oQ%2FHcqmUCFBMk%3DhNElCGN6Hc2rNPvHPgw8oQPCPQUW2piLNK7tjaLihJETMXEpO0"  # <-- Replace with your token
#
# # --- Sentiment Analyzer Classes ---
# class SentimentAnalyzer:
#     def __init__(self):
#         self.vader = SentimentIntensityAnalyzer()
#         # Load FinBERT model and tokenizer
#         self.finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
#         self.finbert_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
#         self.finbert_model.eval()
#
#     def vader_sentiment(self, text):
#         return self.vader.polarity_scores(text)['compound']
#
#     def finbert_sentiment(self, text):
#         inputs = self.finbert_tokenizer(
#             text, return_tensors="pt", truncation=True, max_length=512, padding=True
#         )
#         with torch.no_grad():
#             outputs = self.finbert_model(**inputs)
#         return torch.argmax(outputs.logits).item()  # 0: Neutral, 1: Positive, 2: Negative
#
#     def analyze(self, text):
#         return {
#             "vader": self.vader_sentiment(text),
#             "finbert": self.finbert_sentiment(text)
#         }
#
# # --- Fetch Tweets ---
# def fetch_tweets(query, bearer_token, max_results=50):
#     client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
#     tweets = client.search_recent_tweets(
#         query=query + " -is:retweet lang:en",
#         max_results=max_results,
#         tweet_fields=['created_at', 'text']
#     )
#     tweet_texts = [tweet.text for tweet in tweets.data] if tweets.data else []
#     return tweet_texts
#
# # --- Visualization ---
# def visualize_results(df, stock_name):
#     # FinBERT sentiment counts
#     sentiment_labels = ["Neutral", "Positive", "Negative"]
#     counts = df['FinBERT_Label'].value_counts().reindex(sentiment_labels, fill_value=0)
#     plt.figure(figsize=(8,4))
#     counts.plot(kind='bar', color=['gray','green','red'])
#     plt.title(f"FinBERT Sentiment for {stock_name} Tweets")
#     plt.xlabel("Sentiment")
#     plt.ylabel("Number of Tweets")
#     plt.tight_layout()
#     plt.show()
#
#     # VADER score histogram
#     plt.figure(figsize=(8,4))
#     plt.hist(df['VADER_Score'], bins=20, color='skyblue', edgecolor='black')
#     plt.title(f"VADER Sentiment Score Distribution for {stock_name} Tweets")
#     plt.xlabel("VADER Score")
#     plt.ylabel("Number of Tweets")
#     plt.tight_layout()
#     plt.show()
#
# # --- Main Workflow ---
# if __name__ == "__main__":
#     # 1. Get user input
#     stock_name = input("Enter the stock/company name to analyze tweet sentiment for: ").strip()
#     if not stock_name:
#         print("No stock name entered. Exiting.")
#         exit(1)
#
#     # 2. Fetch tweets
#     print(f"\nFetching latest tweets for '{stock_name}'...")
#     tweets = fetch_tweets(stock_name, TWITTER_BEARER_TOKEN)
#     if not tweets:
#         print("No tweets found or error fetching tweets.")
#         exit(1)
#
#     # 3. Sentiment analysis
#     analyzer = SentimentAnalyzer()
#     results = []
#     print(f"\nAnalyzing {len(tweets)} tweets...")
#     for tweet in tweets:
#         sentiment = analyzer.analyze(tweet)
#         vader_score = sentiment["vader"]
#         finbert_score = sentiment["finbert"]
#         vader_label = "Positive" if vader_score > 0.05 else "Negative" if vader_score < -0.05 else "Neutral"
#         finbert_label = ["Neutral", "Positive", "Negative"][finbert_score]
#         results.append({
#             "Tweet": tweet,
#             "VADER_Score": vader_score,
#             "VADER_Label": vader_label,
#             "FinBERT_Score": finbert_score,
#             "FinBERT_Label": finbert_label
#         })
#         print(f"\nTweet: {tweet}")
#         print(f"  VADER: {vader_score:.3f} ({vader_label})")
#         print(f"  FinBERT: {finbert_score} ({finbert_label})")
#
#     # 4. Results DataFrame
#     df = pd.DataFrame(results)
#
#     # 5. Visualization
#     visualize_results(df, stock_name)
#
#     # 6. Save daily report
#     today = datetime.datetime.now().strftime("%Y-%m-%d")
#     report_path = f"sentiment_report_{stock_name.replace(' ','_')}_{today}.csv"
#     df.to_csv(report_path, index=False)
#     print(f"\nDaily sentiment report saved to: {report_path}")
#
#     # 7. Summary
#     avg_vader = df['VADER_Score'].mean()
#     finbert_counts = df['FinBERT_Label'].value_counts()
#     print("\n--- Sentiment Summary ---")
#     print(f"Average VADER score: {avg_vader:.3f}")
#     print("FinBERT counts:")
#     print(finbert_counts)
#     print("\nDone.")
