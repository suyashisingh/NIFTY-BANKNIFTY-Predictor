import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Reddit API credentials
REDDIT_CLIENT_ID = 'XvhEBeBUBe5koDKC0XgGlw'
REDDIT_CLIENT_SECRET = 'jbEID3Lqi9RM15PCacfi4xE2HXFK0w'
REDDIT_USER_AGENT = 'sentiment_analyzer/0.1 by suyashisingh'

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Choose subreddit and search parameters
subreddit_name = 'IndiaInvestments'  # or 'StockMarketIndia', etc.
query = 'NIFTY'
limit = 50  # Number of posts to fetch

analyzer = SentimentIntensityAnalyzer()
data = []

# Fetch posts and comments
for submission in reddit.subreddit(subreddit_name).search(query, limit=limit):
    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        text = comment.body
        sentiment = analyzer.polarity_scores(text)['compound']
        data.append({
            'post_title': submission.title,
            'comment': text,
            'sentiment': sentiment,
            'created_utc': comment.created_utc
        })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('data/raw/reddit_nifty_sentiment.csv', index=False)
print(df.head())
