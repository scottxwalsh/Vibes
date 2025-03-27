# Cryptocurrency Trading Bot

A Python-based cryptocurrency trading bot that allows you to implement and test trading strategies.

## Features

- Support for multiple cryptocurrency exchanges through CCXT library
- Basic technical analysis indicators
- Configurable trading parameters
- Environment variable support for API keys

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory with your exchange API credentials:
```
EXCHANGE_API_KEY=your_api_key
EXCHANGE_SECRET=your_secret_key
```

3. Configure your trading parameters in `config.py`

## Usage

Run the trading bot:
```bash
python trading_bot.py
```

## Disclaimer

This is a basic trading bot framework. Always test thoroughly with small amounts before using real money. Cryptocurrency trading involves significant risk of loss. 