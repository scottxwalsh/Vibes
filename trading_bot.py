import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import ta
from config import *

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

    def check_buy_signal(self, df):
        """Check for buy signals based on RSI and MACD"""
        last_row = df.iloc[-1]
        
        # Buy conditions
        rsi_oversold = last_row['rsi'] < RSI_OVERSOLD
        macd_crossover = (df['macd_diff'].iloc[-1] > 0 and df['macd_diff'].iloc[-2] <= 0)
        
        return rsi_oversold and macd_crossover

    def check_sell_signal(self, df):
        """Check for sell signals based on RSI and MACD"""
        last_row = df.iloc[-1]
        
        # Sell conditions
        rsi_overbought = last_row['rsi'] > RSI_OVERBOUGHT
        macd_crossunder = (df['macd_diff'].iloc[-1] < 0 and df['macd_diff'].iloc[-2] >= 0)
        
        return rsi_overbought or macd_crossunder

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
                
                # Check for trading signals
                if self.check_buy_signal(df):
                    print(f"Buy signal detected at {datetime.now()}")
                    self.execute_trade('buy')
                elif self.check_sell_signal(df):
                    print(f"Sell signal detected at {datetime.now()}")
                    self.execute_trade('sell')
                
                # Wait before next iteration
                time.sleep(60)  # Adjust based on your timeframe
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = TradingBot()
    bot.run() 