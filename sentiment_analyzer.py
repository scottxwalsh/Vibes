import tweepy
from newsapi import NewsApiClient
from textblob import TextBlob
import nltk
from datetime import datetime, timedelta
import pandas as pd
import praw
from config import *
import numpy as np
import logging

class SentimentAnalyzer:
    def __init__(self):
        # Initialize Twitter API
        auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
        auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
        self.twitter_api = tweepy.API(auth)
        
        # Initialize News API
        self.news_api = NewsApiClient(api_key=NEWS_API_KEY)
        
        # Initialize Reddit API
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        # Initialize sentiment history
        self.sentiment_history = pd.DataFrame(columns=[
            'timestamp', 'twitter_sentiment', 'news_sentiment', 
            'reddit_sentiment', 'combined_sentiment'
        ])
        
        # Set up logging
        self.logger = logging.getLogger('sentiment_analyzer')
        
    def analyze_twitter_sentiment(self):
        """Analyze sentiment from Twitter posts"""
        tweets = []
        sentiment_scores = []
        
        # Search for tweets containing crypto keywords
        for keyword in CRYPTO_KEYWORDS:
            try:
                search_results = self.twitter_api.search_tweets(
                    q=keyword,
                    lang='en',
                    count=100,
                    tweet_mode='extended'
                )
                
                for tweet in search_results:
                    tweets.append(tweet.full_text)
                    
            except tweepy.TweepError as e:
                self.logger.error(f"Error fetching tweets: {e}")
                continue
        
        if len(tweets) < MIN_TWEETS:
            return None
        
        # Analyze sentiment of tweets
        for tweet in tweets:
            analysis = TextBlob(tweet)
            sentiment_scores.append(analysis.sentiment.polarity)
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        return avg_sentiment
    
    def analyze_news_sentiment(self):
        """Analyze sentiment from news articles"""
        articles = []
        sentiment_scores = []
        
        # Get news articles
        try:
            news = self.news_api.get_everything(
                q=' OR '.join(CRYPTO_KEYWORDS),
                from_param=(datetime.now() - timedelta(hours=LOOKBACK_PERIOD)).isoformat(),
                language='en',
                sort_by='relevancy'
            )
            
            if len(news['articles']) < MIN_NEWS_ARTICLES:
                return None
            
            for article in news['articles']:
                if article['title'] and article['description']:
                    text = f"{article['title']} {article['description']}"
                    articles.append(text)
                    
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return None
        
        # Analyze sentiment of articles
        for article in articles:
            analysis = TextBlob(article)
            sentiment_scores.append(analysis.sentiment.polarity)
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        return avg_sentiment
    
    def analyze_reddit_sentiment(self):
        """Analyze sentiment from Reddit posts and comments"""
        posts = []
        comments = []
        sentiment_scores = []
        
        try:
            # Get posts from specified subreddits
            for subreddit_name in REDDIT_SUBREDDITS:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get hot posts
                for post in subreddit.hot(limit=10):
                    posts.append(post.title)
                    # Get top comments
                    post.comments.replace_more(limit=0)  # Remove MoreComments objects
                    for comment in post.comments.list()[:20]:  # Get top 20 comments
                        if hasattr(comment, 'body'):
                            comments.append(comment.body)
            
            if len(posts) < MIN_REDDIT_POSTS or len(comments) < MIN_REDDIT_COMMENTS:
                return None
            
            # Analyze sentiment of posts and comments
            for text in posts + comments:
                analysis = TextBlob(text)
                sentiment_scores.append(analysis.sentiment.polarity)
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            return avg_sentiment
            
        except Exception as e:
            self.logger.error(f"Error fetching Reddit data: {e}")
            return None
    
    def calculate_combined_sentiment(self):
        """Calculate combined sentiment from all sources"""
        twitter_sentiment = self.analyze_twitter_sentiment()
        news_sentiment = self.analyze_news_sentiment()
        reddit_sentiment = self.analyze_reddit_sentiment()
        
        if any(sentiment is None for sentiment in [twitter_sentiment, news_sentiment, reddit_sentiment]):
            return None
        
        # Weight the different sentiment sources
        combined_sentiment = (
            twitter_sentiment * TWITTER_WEIGHT +
            news_sentiment * NEWS_WEIGHT +
            reddit_sentiment * REDDIT_WEIGHT
        )
        
        # Store in history
        self.sentiment_history = self.sentiment_history.append({
            'timestamp': datetime.now(),
            'twitter_sentiment': twitter_sentiment,
            'news_sentiment': news_sentiment,
            'reddit_sentiment': reddit_sentiment,
            'combined_sentiment': combined_sentiment
        }, ignore_index=True)
        
        # Keep only last 24 hours of sentiment history
        cutoff_time = datetime.now() - timedelta(hours=LOOKBACK_PERIOD)
        self.sentiment_history = self.sentiment_history[
            self.sentiment_history['timestamp'] > cutoff_time
        ]
        
        return combined_sentiment
    
    def get_sentiment_trend(self):
        """Calculate sentiment trend over time"""
        if len(self.sentiment_history) < 2:
            return 0
        
        # Calculate the slope of sentiment over time
        sentiment_values = self.sentiment_history['combined_sentiment'].values
        time_values = range(len(sentiment_values))
        
        slope = np.polyfit(time_values, sentiment_values, 1)[0]
        return slope
    
    def generate_trading_signal(self):
        """Generate trading signal based on sentiment analysis"""
        combined_sentiment = self.calculate_combined_sentiment()
        sentiment_trend = self.get_sentiment_trend()
        
        if combined_sentiment is None:
            return 0
        
        # Generate signal based on sentiment and trend
        if combined_sentiment > SENTIMENT_THRESHOLD and sentiment_trend > 0:
            return 1  # Bullish signal
        elif combined_sentiment < -SENTIMENT_THRESHOLD and sentiment_trend < 0:
            return -1  # Bearish signal
        else:
            return 0  # Neutral signal 