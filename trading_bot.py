import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import ta
from config import *
from sentiment_analyzer import SentimentAnalyzer

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
        self.last_sentiment_update = 0

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

    def get_sentiment_signal(self):
        """Get trading signal based on sentiment analysis"""
        current_time = time.time()
        
        # Update sentiment analysis at configured intervals
        if current_time - self.last_sentiment_update >= SENTIMENT_UPDATE_INTERVAL:
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
        """Execute a trade on the exchange"""
        try:
            if side == 'buy':
                order = self.exchange.create_market_buy_order(
                    self.trading_pair,
                    self.position_size
                )
                print(f"Buy order executed: {order}")
            else:
                order = self.exchange.create_market_sell_order(
                    self.trading_pair,
                    self.position_size
                )
                print(f"Sell order executed: {order}")
        except Exception as e:
            print(f"Error executing trade: {e}")

    def run(self):
        """Main trading loop"""
        print(f"Starting trading bot for {self.trading_pair} on {self.timeframe} timeframe")
        
        while True:
            try:
                # Fetch and analyze data
                df = self.fetch_ohlcv()
                df = self.calculate_indicators(df)
                
                # Get signals from different sources
                technical_signal = self.get_technical_signal(df)
                sentiment_signal = self.get_sentiment_signal()
                
                # Combine signals
                final_signal = self.combine_signals(technical_signal, sentiment_signal)
                
                # Execute trades based on combined signal
                if final_signal == 1:
                    print(f"Buy signal detected at {datetime.now()}")
                    print(f"Technical signal: {technical_signal}, Sentiment signal: {sentiment_signal}")
                    self.execute_trade('buy')
                elif final_signal == -1:
                    print(f"Sell signal detected at {datetime.now()}")
                    print(f"Technical signal: {technical_signal}, Sentiment signal: {sentiment_signal}")
                    self.execute_trade('sell')
                
                # Wait before next iteration
                time.sleep(60)  # Adjust based on your timeframe
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = TradingBot()
    bot.run() 