import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
from config import *

class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.daily_stats = {}
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit_loss': 0,
            'win_rate': 0,
            'average_profit': 0,
            'average_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'average_hold_time': timedelta(0),
            'best_trade': 0,
            'worst_trade': 0,
            'largest_winning_streak': 0,
            'largest_losing_streak': 0
        }
        self.equity_curve = []
        self.current_streak = 0
        self.setup_logging()

    def setup_logging(self):
        """Set up logging for performance tracking"""
        self.logger = logging.getLogger('performance_tracker')
        self.logger.setLevel(logging.INFO)

    def add_trade(self, trade):
        """Add a new trade to the history"""
        self.trades.append(trade)
        self.update_metrics()
        self.generate_report()

    def update_metrics(self):
        """Update performance metrics based on trade history"""
        if not self.trades:
            return

        # Convert trades to DataFrame for easier analysis
        df = pd.DataFrame(self.trades)
        
        # Basic metrics
        self.performance_metrics['total_trades'] = len(df)
        self.performance_metrics['winning_trades'] = len(df[df['profit_loss'] > 0])
        self.performance_metrics['losing_trades'] = len(df[df['profit_loss'] < 0])
        self.performance_metrics['total_profit_loss'] = df['profit_loss'].sum()
        
        # Win rate
        self.performance_metrics['win_rate'] = (
            self.performance_metrics['winning_trades'] / 
            self.performance_metrics['total_trades'] * 100
        )
        
        # Average profit/loss
        winning_trades = df[df['profit_loss'] > 0]['profit_loss']
        losing_trades = df[df['profit_loss'] < 0]['profit_loss']
        
        self.performance_metrics['average_profit'] = winning_trades.mean() if not winning_trades.empty else 0
        self.performance_metrics['average_loss'] = losing_trades.mean() if not losing_trades.empty else 0
        
        # Profit factor
        total_profits = winning_trades.sum() if not winning_trades.empty else 0
        total_losses = abs(losing_trades.sum()) if not losing_trades.empty else 0
        self.performance_metrics['profit_factor'] = (
            total_profits / total_losses if total_losses != 0 else float('inf')
        )
        
        # Best and worst trades
        self.performance_metrics['best_trade'] = df['profit_loss'].max()
        self.performance_metrics['worst_trade'] = df['profit_loss'].min()
        
        # Calculate streaks
        df['streak'] = (df['profit_loss'] > 0).astype(int)
        df['streak_change'] = df['streak'].diff()
        
        winning_streaks = df[df['streak'] == 1].groupby((df['streak_change'] != 0).cumsum()).size()
        losing_streaks = df[df['streak'] == 0].groupby((df['streak_change'] != 0).cumsum()).size()
        
        self.performance_metrics['largest_winning_streak'] = winning_streaks.max() if not winning_streaks.empty else 0
        self.performance_metrics['largest_losing_streak'] = losing_streaks.max() if not losing_streaks.empty else 0
        
        # Calculate drawdown
        cumulative_returns = df['profit_loss'].cumsum()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns - rolling_max
        self.performance_metrics['max_drawdown'] = abs(drawdowns.min())
        
        # Calculate Sharpe and Sortino ratios
        returns = df['profit_loss'].pct_change().dropna()
        if len(returns) > 0:
            risk_free_rate = 0.02  # Assuming 2% risk-free rate
            excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
            self.performance_metrics['sharpe_ratio'] = (
                np.sqrt(252) * excess_returns.mean() / excess_returns.std()
                if excess_returns.std() != 0 else 0
            )
            
            # Sortino ratio (using only downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std()
            self.performance_metrics['sortino_ratio'] = (
                np.sqrt(252) * returns.mean() / downside_std
                if downside_std != 0 else 0
            )
        
        # Average hold time
        hold_times = pd.to_datetime(df['exit_time']) - pd.to_datetime(df['entry_time'])
        self.performance_metrics['average_hold_time'] = hold_times.mean()

    def generate_report(self):
        """Generate a performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': self.performance_metrics,
            'daily_stats': self.daily_stats
        }
        
        # Save report to file
        with open('performance_report.json', 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        # Log summary
        self.logger.info("Performance Report Summary:")
        self.logger.info(f"Total Trades: {self.performance_metrics['total_trades']}")
        self.logger.info(f"Win Rate: {self.performance_metrics['win_rate']:.2f}%")
        self.logger.info(f"Total P/L: ${self.performance_metrics['total_profit_loss']:.2f}")
        self.logger.info(f"Profit Factor: {self.performance_metrics['profit_factor']:.2f}")
        self.logger.info(f"Max Drawdown: ${self.performance_metrics['max_drawdown']:.2f}")
        self.logger.info(f"Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}")
        self.logger.info(f"Sortino Ratio: {self.performance_metrics['sortino_ratio']:.2f}")

    def update_daily_stats(self):
        """Update daily trading statistics"""
        today = datetime.now().date()
        today_trades = [t for t in self.trades if t['entry_time'].date() == today]
        
        if today_trades:
            daily_metrics = {
                'date': today.isoformat(),
                'total_trades': len(today_trades),
                'winning_trades': len([t for t in today_trades if t['profit_loss'] > 0]),
                'losing_trades': len([t for t in today_trades if t['profit_loss'] < 0]),
                'total_profit_loss': sum(t['profit_loss'] for t in today_trades),
                'average_profit': np.mean([t['profit_loss'] for t in today_trades if t['profit_loss'] > 0]) if any(t['profit_loss'] > 0 for t in today_trades) else 0,
                'average_loss': np.mean([t['profit_loss'] for t in today_trades if t['profit_loss'] < 0]) if any(t['profit_loss'] < 0 for t in today_trades) else 0,
                'max_drawdown': self.calculate_daily_drawdown(today_trades)
            }
            
            self.daily_stats[today.isoformat()] = daily_metrics
            self.logger.info(f"Daily Statistics for {today}:")
            self.logger.info(f"Total Trades: {daily_metrics['total_trades']}")
            self.logger.info(f"Total P/L: ${daily_metrics['total_profit_loss']:.2f}")
            self.logger.info(f"Max Drawdown: ${daily_metrics['max_drawdown']:.2f}")

    def calculate_daily_drawdown(self, trades):
        """Calculate maximum drawdown for a given day's trades"""
        if not trades:
            return 0
        
        cumulative_returns = np.cumsum([t['profit_loss'] for t in trades])
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - rolling_max
        return abs(min(drawdowns)) if len(drawdowns) > 0 else 0 