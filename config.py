import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Exchange configuration
EXCHANGE = 'coinbase'  # Changed from binance to coinbase
EXCHANGE_API_KEY = os.getenv('EXCHANGE_API_KEY')
EXCHANGE_SECRET = os.getenv('EXCHANGE_SECRET')

# Trading parameters
TRADING_PAIR = 'BTC/USDT'  # The trading pair to trade
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
STOP_LOSS_PERCENTAGE = 2  # Stop loss percentage
TAKE_PROFIT_PERCENTAGE = 4  # Take profit percentage
MAX_OPEN_TRADES = 1  # Maximum number of open trades 