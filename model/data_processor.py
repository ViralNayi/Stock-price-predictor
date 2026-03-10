"""
Data Processing Module for Stock Price Prediction
Handles data fetching, preprocessing, normalization, and sequence creation.
Now includes technical indicators: RSI, MACD, SMA, EMA, Volume.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


# ── Technical Indicator Helpers ──────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line and signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


# ── Public Functions ─────────────────────────────────────────────────────

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: Historical stock data with OHLCV columns
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        return df
    except Exception as e:
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical-indicator columns to a dataframe that already has
    'Close' and 'Volume'.  Drops warm-up NaN rows automatically.

    New columns: RSI_14, MACD, MACD_Signal, SMA_20, EMA_12
    """
    df = df.copy()
    close = df['Close']

    df['RSI_14'] = _rsi(close, 14)

    macd_line, signal_line = _macd(close)
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line

    df['SMA_20'] = close.rolling(window=20).mean()
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()

    # Drop rows where indicators are still warming up
    df.dropna(inplace=True)
    return df


def preprocess_data(df):
    """
    Preprocess stock data — keep Close + Volume + technical indicators.

    Args:
        df (pd.DataFrame): Raw stock data

    Returns:
        tuple: (feature_df, close_only_df)
            - feature_df has all columns used for LSTM input
            - close_only_df has only 'Close' (for charting)
    """
    # Keep relevant columns
    df = df[['Close', 'Volume']].copy()

    # Handle missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Add technical indicators (also drops warm-up NaN rows)
    df = compute_technical_indicators(df)

    # Feature order: Close must be column 0 (we inverse-transform it later)
    feature_cols = ['Close', 'Volume', 'RSI_14', 'MACD', 'MACD_Signal', 'SMA_20', 'EMA_12']
    feature_df = df[feature_cols].copy()

    return feature_df


def normalize_data(data, scaler=None):
    """
    Normalize data to 0-1 range using MinMaxScaler.

    Args:
        data (np.array): Data to normalize  (2-D array)
        scaler: Optional pre-fitted scaler. If None a new one is fitted.

    Returns:
        tuple: (normalized_data, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(data)
    else:
        normalized_data = scaler.transform(data)
    return normalized_data, scaler


def create_sequences(data, seq_length=60):
    """
    Create sequences for LSTM training.
    Uses sliding window approach: last seq_length days predict next day's Close.

    Args:
        data (np.array): Normalized multi-feature data  (samples, features)
        seq_length (int): Number of days to look back

    Returns:
        tuple: (X, y)
            X – (samples, seq_length, n_features)
            y – (samples,) target values (Close price, column 0)
    """
    X, y = [], []

    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])          # all features
        y.append(data[i, 0])                       # Close is col 0

    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        raise ValueError(
            f"Not enough data to create sequences. "
            f"You need at least {seq_length + 1} data points, "
            f"but only {len(data)} were available. "
            f"Try selecting a longer date range or reducing 'Sequence Length'."
        )

    return X, y


def split_data(X, y, train_ratio=0.8):
    """
    Split data into training and testing sets (time-series safe).

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


def prepare_data_pipeline(ticker, start_date, end_date, seq_length=60, train_ratio=0.8):
    """
    Complete data preparation pipeline with multi-feature input and
    scaler fitted on training data only (no lookahead bias).

    Returns:
        dict with keys:
            X_train, X_test, y_train, y_test,
            scaler, original_df, seq_length, n_features,
            normalized_data, feature_df
    """
    # Fetch & preprocess
    raw_df = fetch_stock_data(ticker, start_date, end_date)
    feature_df = preprocess_data(raw_df)

    # Keep an un-modified copy for charting (Close + index)
    original_df = feature_df[['Close']].copy()

    data = feature_df.values   # shape: (n_days, n_features)
    n_features = data.shape[1]

    # ── Fix scaler leakage: fit on training portion only ──
    train_end = int(len(data) * train_ratio)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:train_end])                       # fit on train only
    normalized_data = scaler.transform(data)           # transform all

    # Create sequences
    X, y = create_sequences(normalized_data, seq_length)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'original_df': original_df,
        'seq_length': seq_length,
        'n_features': n_features,
        'normalized_data': normalized_data,
        'feature_df': feature_df,
    }
