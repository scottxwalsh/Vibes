import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import ta
import logging
from config import *
from sentiment_analyzer import SentimentAnalyzer
from performance_tracker import PerformanceTracker

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE) if LOG_TO_FILE else logging.NullHandler(),
        logging.StreamHandler() if LOG_TO_CONSOLE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.exchange = getattr(ccxt, EXCHANGE)({
            'apiKey': EXCHANGE_API_KEY,
            'secret': EXCHANGE_SECRET,
            'enableRateLimit': True
        })
        self.trading_pair = TRADING_PAIR
        self.timeframe = TIMEFRAME
        self.position_size = POSITION_SIZE
        self.sentiment_analyzer = SentimentAnalyzer()
        self.performance_tracker = PerformanceTracker()
        self.last_sentiment_update = 0
        self.last_volatility_check = 0
        self.last_price = None
        self.daily_loss = 0
        self.last_trade_time = None
        self.trade_history = []
        self.active_trades = {}
        
        # Initialize market state
        self.market_state = {
            'volatility': 0,
            'volume_24h': 0,
            'spread': 0,
            'trend': 'neutral'
        }

    def setup_logging(self):
        """Set up logging for the trading bot"""
        logger.info(f"Starting trading bot for {self.trading_pair} on {self.timeframe} timeframe")
        logger.info(f"Initial position size: {self.position_size} BTC")
        logger.info(f"Maximum daily loss: {MAX_DAILY_LOSS}%")

    def check_market_conditions(self):
        """Check overall market conditions"""
        try:
            # Get 24h ticker data
            ticker = self.exchange.fetch_ticker(self.trading_pair)
            
            # Update market state
            self.market_state['volume_24h'] = ticker['quoteVolume']
            self.market_state['spread'] = (ticker['ask'] - ticker['bid']) / ticker['bid']
            
            # Log market conditions
            logger.info(f"Market Conditions - Volume: ${self.market_state['volume_24h']:,.2f}, "
                       f"Spread: {self.market_state['spread']:.2%}")
            
            # Check if market conditions are suitable for trading
            if (self.market_state['volume_24h'] < MIN_VOLUME_REQUIRED or 
                self.market_state['spread'] > MAX_SPREAD_PERCENTAGE):
                logger.warning("Market conditions not suitable for trading")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
            return False

    def calculate_volatility(self, df):
        """Calculate volatility using multiple metrics"""
        try:
            # Calculate price volatility
            returns = df['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate volume volatility
            volume_ma = df['volume'].rolling(window=VOLATILITY_WINDOW).mean()
            volume_std = df['volume'].rolling(window=VOLATILITY_WINDOW).std()
            volume_volatility = (volume_std / volume_ma).iloc[-1]
            
            # Check for volume spikes
            current_volume = df['volume'].iloc[-1]
            volume_spike = current_volume > (volume_ma.iloc[-1] * VOLUME_SPIKE_THRESHOLD)
            
            # Update market state
            self.market_state['volatility'] = volatility
            
            # Log volatility metrics
            logger.info(f"Volatility Metrics - Price: {volatility:.2%}, "
                       f"Volume: {volume_volatility:.2%}, "
                       f"Volume Spike: {'Yes' if volume_spike else 'No'}")
            
            return {
                'price_volatility': volatility,
                'volume_volatility': volume_volatility,
                'volume_spike': volume_spike
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return None

    def check_risk_limits(self):
        """Check if risk limits are exceeded"""
        try:
            # Check daily loss limit
            if self.daily_loss >= MAX_DAILY_LOSS:
                logger.warning(f"Daily loss limit ({MAX_DAILY_LOSS}%) exceeded")
                return False
            
            # Check balance
            balance = self.exchange.fetch_balance()
            usdc_balance = balance['USDC']['free']
            
            if usdc_balance < MIN_BALANCE_REQUIRED:
                logger.warning(f"Insufficient balance: ${usdc_balance:.2f} USDC")
                return False
            
            # Check position size
            if self.position_size > MAX_POSITION_SIZE:
                logger.warning(f"Position size ({self.position_size} BTC) exceeds maximum allowed")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False

    def update_trade_history(self, trade_type, price, size, order_id):
        """Update trade history and calculate daily loss"""
        trade = {
            'timestamp': datetime.now(),
            'type': trade_type,
            'price': price,
            'size': size,
            'value': price * size,
            'order_id': order_id
        }
        self.trade_history.append(trade)
        
        # Calculate daily loss
        today = datetime.now().date()
        daily_trades = [t for t in self.trade_history if t['timestamp'].date() == today]
        if daily_trades:
            total_value = sum(t['value'] for t in daily_trades)
            self.daily_loss = abs(total_value) / MIN_BALANCE_REQUIRED * 100

    def check_volatility(self):
        """Enhanced volatility check with multiple metrics"""
        current_time = time.time()
        
        if current_time - self.last_volatility_check < VOLATILITY_CHECK_INTERVAL:
            return False
            
        self.last_volatility_check = current_time
        
        try:
            # Get current price
            current_price = self.exchange.fetch_ticker(self.trading_pair)['last']
            
            # Calculate price change
            if self.last_price is not None:
                price_change = abs(current_price - self.last_price) / self.last_price
                
                # Get volatility metrics
                df = self.fetch_ohlcv(limit=VOLATILITY_WINDOW)
                volatility_metrics = self.calculate_volatility(df)
                
                if volatility_metrics:
                    # Check multiple volatility conditions
                    high_volatility = (
                        price_change > HIGH_VOLATILITY_THRESHOLD or
                        volatility_metrics['price_volatility'] > MAX_VOLATILITY_THRESHOLD or
                        volatility_metrics['volume_spike']
                    )
                    
                    if high_volatility:
                        logger.warning(f"High volatility detected: {price_change:.2%} price change, "
                                     f"{volatility_metrics['price_volatility']:.2%} volatility")
                        return True
            
            self.last_price = current_price
            return False
            
        except Exception as e:
            logger.error(f"Error checking volatility: {e}")
            return False

    def fetch_ohlcv(self, limit=100):
        """Fetch OHLCV data from the exchange"""
        ohlcv = self.exchange.fetch_ohlcv(self.trading_pair, self.timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=RSI_PERIOD).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'], window_slow=MA_SLOW, window_fast=MA_FAST, window_sign=MACD_SIGNAL)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        return df

    def get_technical_signal(self, df):
        """Get trading signal based on technical analysis"""
        last_row = df.iloc[-1]
        
        # Technical buy conditions
        rsi_oversold = last_row['rsi'] < RSI_OVERSOLD
        macd_crossover = (df['macd_diff'].iloc[-1] > 0 and df['macd_diff'].iloc[-2] <= 0)
        
        # Technical sell conditions
        rsi_overbought = last_row['rsi'] > RSI_OVERBOUGHT
        macd_crossunder = (df['macd_diff'].iloc[-1] < 0 and df['macd_diff'].iloc[-2] >= 0)
        
        if rsi_oversold and macd_crossover:
            return 1  # Buy signal
        elif rsi_overbought or macd_crossunder:
            return -1  # Sell signal
        return 0  # Neutral signal

    def get_sentiment_signal(self, force_update=False):
        """Get trading signal based on sentiment analysis"""
        current_time = time.time()
        
        # Update sentiment analysis if:
        # 1. It's time for regular update
        # 2. High volatility is detected
        if force_update or current_time - self.last_sentiment_update >= SENTIMENT_UPDATE_INTERVAL:
            sentiment_signal = self.sentiment_analyzer.generate_trading_signal()
            self.last_sentiment_update = current_time
            return sentiment_signal
        
        return 0  # Return neutral if not time to update

    def combine_signals(self, technical_signal, sentiment_signal):
        """Combine technical and sentiment signals"""
        # Weight the signals according to configuration
        combined_signal = (technical_signal * TECHNICAL_WEIGHT + 
                         sentiment_signal * SENTIMENT_WEIGHT)
        
        # Convert to final trading decision
        if combined_signal > 0.5:
            return 1  # Buy
        elif combined_signal < -0.5:
            return -1  # Sell
        return 0  # Hold

    def execute_trade(self, side):
        """Execute a trade with enhanced safety checks"""
        try:
            # Check market conditions and risk limits
            if not self.check_market_conditions() or not self.check_risk_limits():
                logger.warning("Trade execution prevented due to risk limits or market conditions")
                return
            
            # Get current price
            ticker = self.exchange.fetch_ticker(self.trading_pair)
            current_price = ticker['last']
            
            # Execute trade
            if side == 'buy':
                order = self.exchange.create_market_buy_order(
                    self.trading_pair,
                    self.position_size
                )
                logger.info(f"Buy order executed: {order}")
                
                # Record trade in performance tracker
                trade_data = {
                    'entry_time': datetime.now(),
                    'exit_time': None,  # Will be updated when position is closed
                    'entry_price': current_price,
                    'exit_price': None,  # Will be updated when position is closed
                    'position_size': self.position_size,
                    'profit_loss': None,  # Will be calculated when position is closed
                    'type': 'buy'
                }
                self.active_trades[order['id']] = trade_data
                
            else:
                # Close existing positions
                for order_id, trade in self.active_trades.items():
                    if trade['type'] == 'buy':
                        close_order = self.exchange.create_market_sell_order(
                            self.trading_pair,
                            trade['position_size']
                        )
                        logger.info(f"Sell order executed: {close_order}")
                        
                        # Calculate profit/loss
                        trade['exit_time'] = datetime.now()
                        trade['exit_price'] = current_price
                        trade['profit_loss'] = (trade['exit_price'] - trade['entry_price']) * trade['position_size']
                        
                        # Add to performance tracker
                        self.performance_tracker.add_trade(trade)
                        
                        # Remove from active trades
                        del self.active_trades[order_id]
            
            # Update trade history
            self.update_trade_history(side, current_price, self.position_size, order['id'])
            
            # Update daily statistics
            self.performance_tracker.update_daily_stats()
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")

    def run(self):
        """Main trading loop with enhanced monitoring"""
        self.setup_logging()
        
        while True:
            try:
                # Check for high volatility
                high_volatility = self.check_volatility()
                
                # Fetch and analyze data
                df = self.fetch_ohlcv()
                df = self.calculate_indicators(df)
                
                # Get signals from different sources
                technical_signal = self.get_technical_signal(df)
                sentiment_signal = self.get_sentiment_signal(force_update=high_volatility)
                
                # Combine signals
                final_signal = self.combine_signals(technical_signal, sentiment_signal)
                
                # Log trading signals
                logger.info(f"Trading Signals - Technical: {technical_signal}, "
                          f"Sentiment: {sentiment_signal}, "
                          f"Final: {final_signal}")
                
                # Execute trades based on combined signal
                if final_signal == 1:
                    logger.info("Buy signal detected")
                    if high_volatility:
                        logger.warning("Signal triggered during high volatility period")
                    self.execute_trade('buy')
                elif final_signal == -1:
                    logger.info("Sell signal detected")
                    if high_volatility:
                        logger.warning("Signal triggered during high volatility period")
                    self.execute_trade('sell')
                
                # Wait before next iteration
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = TradingBot()
    bot.run() 