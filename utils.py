"""
Data fetching and analysis utilities
No external dependencies from this project
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


class DataFetcher:
    """Fetch data from Yahoo Finance"""
    
    def __init__(self):
        # Expanded list of assets and their Yahoo Finance tickers
        self.cryptos = {
            'Bitcoin': 'BTC-USD', 
            'Ethereum': 'ETH-USD', 
            'Solana': 'SOL-USD',     # Added
            'Ripple': 'XRP-USD',     # Added
            'Dogecoin': 'DOGE-USD'   # Added
        }
        self.stocks = {
            'NIFTY 50': '^NSEI', 
            'S&P 500': '^GSPC',
            'NASDAQ 100': '^NDX',    # Added
            'FTSE 100': '^FTSE',     # Added
            'DAX': '^GDAXI'          # Added
        }
        self.TICKER_MAP = {**self.cryptos, **self.stocks}

    # Modified fetch_all_assets to accept a list of assets to fetch
    def fetch_data(self, ticker, period='1y'):
        try:
            # Note: For indices, setting interval='1d' explicitly can sometimes help
            df = yf.download(ticker, period=period, progress=False, interval='1d')
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                # Handle cases where yf.download returns a multi-index (e.g., if multiple tickers were passed, though we only pass one here)
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            # print(f"Error fetching data for {ticker}: {e}") # Debug line
            return None
    
    # Updated to take selected_assets list from app.py
    def fetch_all_assets(self, period='1y', selected_assets=None):
        result = {}
        assets_to_fetch = selected_assets if selected_assets is not None else self.TICKER_MAP.keys()
        
        for name in assets_to_fetch:
            ticker = self.TICKER_MAP.get(name)
            if ticker:
                df = self.fetch_data(ticker, period)
                if df is not None:
                    result[name] = df
        return result


class DataPreprocessor:
    """Add technical indicators"""
    
    @staticmethod
    def calculate_returns(df):
        df['Returns'] = df['Close'].pct_change()
        return df
    
    @staticmethod
    def calculate_moving_averages(df, windows=[7, 14, 30]):
        for w in windows:
            df[f'MA_{w}'] = df['Close'].rolling(window=w).mean()
        return df
    
    @staticmethod
    def calculate_volatility(df, window=30):
        df['Volatility'] = df['Returns'].rolling(window=window).std()
        return df
    
    @staticmethod
    def calculate_rsi(df, period=14):
        delta = df['Close'].diff()
        # Ensure we handle potential NaN values by filling them with 0 before rolling mean
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        # Handle division by zero for rs
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
        
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def add_all_features(df):
        df = DataPreprocessor.calculate_returns(df)
        df = DataPreprocessor.calculate_moving_averages(df)
        df = DataPreprocessor.calculate_volatility(df)
        df = DataPreprocessor.calculate_rsi(df)
        return df


class CorrelationAnalyzer:
    """Correlation calculations"""
    
    @staticmethod
    def calculate_correlation_matrix(data_dict):
        prices = pd.DataFrame()
        for name, df in data_dict.items():
            if 'Close' in df.columns:
                prices[name] = df['Close']
        # Compute correlation on percentage change (returns) for better financial analysis
        return prices.pct_change().corr() 
    
    @staticmethod
    def calculate_rolling_correlation(df1, df2, window=30):
        r1 = df1['Close'].pct_change()
        r2 = df2['Close'].pct_change()
        return r1.rolling(window=window).corr(r2)
    
    @staticmethod
    def get_correlation_strength(val):
        a = abs(val)
        if a > 0.7:
            return "Strong"
        elif a > 0.4:
            return "Moderate"
        elif a > 0.2:
            return "Weak"
        return "Very Weak"


def format_currency(val):
    try:
        # Check if the input is a pandas Series or DataFrame and extract the last value
        if isinstance(val, (pd.Series, pd.DataFrame)):
             val = val.iloc[-1] if not val.empty else float('nan')
        return f"${float(val):,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def format_percentage(val):
    try:
        # Check if the input is a pandas Series or DataFrame and extract the last value
        if isinstance(val, (pd.Series, pd.DataFrame)):
             val = val.iloc[-1] if not val.empty else float('nan')
        return f"{float(val):.2f}%"
    except (ValueError, TypeError):
        return "N/A"


def get_latest_price(df):
    try:
        if df is not None and 'Close' in df.columns and not df['Close'].empty:
            return float(df['Close'].iloc[-1])
    except:
        pass
    return None


def get_price_change(df, days=1):
    try:
        if df is not None and 'Close' in df.columns and len(df) > days and not df['Close'].iloc[[-days-1, -1]].isnull().any():
            curr = float(df['Close'].iloc[-1])
            prev = float(df['Close'].iloc[-days-1])
            if prev != 0:
                return ((curr - prev) / prev) * 100
    except:
        pass
    return None