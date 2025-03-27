import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Exchange configuration
EXCHANGE = 'coinbase'  # Changed from binance to coinbase
EXCHANGE_API_KEY = os.getenv('EXCHANGE_API_KEY')
EXCHANGE_SECRET = os.getenv('EXCHANGE_SECRET')

# Trading parameters
TRADING_PAIR = 'BTC/USDC'  # Changed from BTC/USDT to BTC/USDC
TIMEFRAME = '1h'  # Timeframe for analysis (1m, 5m, 15m, 1h, 4h, 1d)
POSITION_SIZE = 0.001  # Size of each trade in BTC

# Technical analysis parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MA_FAST = 12
MA_SLOW = 26
MACD_SIGNAL = 9

# Risk management
STOP_LOSS_PERCENTAGE = 1.5  # Reduced from 2% to 1.5% for USDC's lower volatility
TAKE_PROFIT_PERCENTAGE = 3  # Reduced from 4% to 3% for more frequent trades
MAX_OPEN_TRADES = 1  # Maximum number of open trades
MAX_DAILY_LOSS = 3  # Reduced from 5% to 3% for more conservative USDC trading
MAX_POSITION_SIZE = 0.01  # Maximum position size in BTC
MIN_BALANCE_REQUIRED = 200  # Increased from 100 to 200 USDC for better liquidity

# Sentiment Analysis Configuration
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Reddit API Configuration
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = 'Crypto Trading Bot v1.0'

# Reddit Subreddits to Monitor
REDDIT_SUBREDDITS = [
    'Bitcoin',
    'CryptoCurrency',
    'CryptoMarkets',
    'BitcoinMarkets',
    'CryptoTechnology',
    'CryptoMoonShots'
]

# Sentiment Analysis Parameters
SENTIMENT_WEIGHT = 0.25  # Reduced from 0.3 to 0.25 for more conservative sentiment influence
TECHNICAL_WEIGHT = 0.75  # Increased from 0.7 to 0.75 for stronger technical analysis influence
SENTIMENT_THRESHOLD = 0.25  # Increased from 0.2 to 0.25 for more conservative sentiment signals
LOOKBACK_PERIOD = 24  # Hours to look back for sentiment analysis
MIN_TWEETS = 150  # Increased from 100 to 150 for more reliable sentiment analysis
MIN_NEWS_ARTICLES = 15  # Increased from 10 to 15 for more reliable news sentiment
MIN_REDDIT_POSTS = 50  # Minimum number of Reddit posts to analyze
MIN_REDDIT_COMMENTS = 200  # Minimum number of Reddit comments to analyze

# Sentiment Source Weights
TWITTER_WEIGHT = 0.5  # Increased from 0.4 to 0.5 for highest priority
NEWS_WEIGHT = 0.3  # Kept at 0.3 for second priority
REDDIT_WEIGHT = 0.2  # Reduced from 0.3 to 0.2 for lowest priority

# Keywords for sentiment analysis
CRYPTO_KEYWORDS = [
    'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain',
    'ethereum', 'eth', 'defi', 'nft', 'web3', 'usdc', 'stablecoin'
]

# Sentiment Analysis Timeframes
SENTIMENT_UPDATE_INTERVAL = 1800  # Update sentiment every 30 minutes (in seconds)
VOLATILITY_CHECK_INTERVAL = 300  # Check for high volatility every 5 minutes

# Volatility Analysis Parameters
HIGH_VOLATILITY_THRESHOLD = 0.015  # Reduced from 0.02 to 0.015 for USDC's lower volatility
VOLATILITY_WINDOW = 20  # Number of periods to calculate volatility
VOLATILITY_MULTIPLIER = 1.8  # Reduced from 2.0 to 1.8 for more conservative volatility detection
MAX_VOLATILITY_THRESHOLD = 0.03  # Reduced from 0.05 to 0.03 for USDC's lower volatility
VOLUME_SPIKE_THRESHOLD = 2.5  # Increased from 2.0 to 2.5 for more reliable volume spike detection

# Logging Configuration
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = 'trading_bot.log'
LOG_TO_FILE = True
LOG_TO_CONSOLE = True

# Market Condition Parameters
MARKET_TREND_WINDOW = 24  # Hours to determine market trend
MIN_VOLUME_REQUIRED = 2000000  # Increased from 1M to 2M USDC for better liquidity
MAX_SPREAD_PERCENTAGE = 0.3  # Reduced from 0.5 to 0.3 for tighter USDC spreads 