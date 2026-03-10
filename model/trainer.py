"""
Model Training Module
Handles training, evaluation, and prediction (multi-feature aware).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, callbacks=None):
    """
    Train the LSTM model.

    Returns:
        History: Training history object
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks if callbacks else [],
        verbose=1
    )
    return history


def make_predictions(model, X):
    """
    Make predictions using trained model.

    Returns:
        np.array: Predictions (normalized Close values)
    """
    predictions = model.predict(X, verbose=0)
    return predictions


def evaluate_model(y_true, y_pred):
    """
    Calculate evaluation metrics including directional accuracy.

    Args:
        y_true (np.array): True price values (original scale)
        y_pred (np.array): Predicted price values (original scale)

    Returns:
        dict: RMSE, MAE, MAPE, R², Directional Accuracy
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # Directional accuracy: did the model predict UP/DOWN correctly?
    if len(y_true) > 1:
        actual_direction = np.diff(y_true) >= 0     # True = up/flat
        pred_direction = np.diff(y_pred) >= 0
        dir_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        dir_accuracy = 0.0

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Dir_Accuracy': dir_accuracy,
    }


def inverse_transform_close(predictions, scaler, n_features):
    """
    Inverse-transform only the Close column (column 0) from normalized
    predictions back to the original price scale.

    Args:
        predictions (np.array): Normalized Close predictions, shape (n,) or (n,1)
        scaler: MinMaxScaler fitted on all features
        n_features (int): Total number of features the scaler was fitted on

    Returns:
        np.array: Prices in original scale, shape (n, 1)
    """
    predictions = predictions.reshape(-1, 1)
    # Build a dummy array with the right number of columns
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, 0] = predictions[:, 0]
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0].reshape(-1, 1)


def predict_future(model, normalized_data, scaler, feature_df, seq_length=60, n_days=30):
    """
    Predict the next n_days of stock prices using iterative forecasting
    with on-the-fly recomputation of technical indicators.

    Args:
        model: Trained Keras model
        normalized_data (np.array): Full normalized data (n, n_features)
        scaler: Fitted MinMaxScaler
        feature_df (pd.DataFrame): Original (un-normalized) feature dataframe
            with columns [Close, Volume, RSI_14, MACD, MACD_Signal, SMA_20, EMA_12]
        seq_length (int): Input window size
        n_days (int): Number of future days to predict

    Returns:
        np.array: Predicted Close prices (original scale), shape (n_days, 1)
    """
    from model.data_processor import compute_technical_indicators

    n_features = normalized_data.shape[1]

    # We'll keep a running copy of the raw Close/Volume series so we can
    # recompute indicators after each predicted day.
    close_history = feature_df['Close'].values.tolist()
    volume_history = feature_df['Volume'].values.tolist()

    # Current sliding window of normalized multi-feature rows
    window = normalized_data[-seq_length:].tolist()

    future_prices = []

    for _ in range(n_days):
        x = np.array(window[-seq_length:]).reshape(1, seq_length, n_features)
        pred_norm = model.predict(x, verbose=0)[0, 0]

        # Inverse-transform to get the Close price in original scale
        dummy = np.zeros((1, n_features))
        dummy[0, 0] = pred_norm
        pred_price = scaler.inverse_transform(dummy)[0, 0]
        future_prices.append(pred_price)

        # Append to history and recompute indicators
        close_history.append(pred_price)
        volume_history.append(volume_history[-1])  # carry forward last volume

        # Build a temporary df to recompute indicators
        tmp_df = pd.DataFrame({
            'Close': close_history,
            'Volume': volume_history,
        })
        tmp_df = compute_technical_indicators(tmp_df)

        # Take the last row (the new predicted day) and normalize it
        last_row = tmp_df.iloc[-1][['Close', 'Volume', 'RSI_14', 'MACD', 'MACD_Signal', 'SMA_20', 'EMA_12']].values.reshape(1, -1)
        last_row_norm = scaler.transform(last_row)[0]
        window.append(last_row_norm.tolist())

    return np.array(future_prices).reshape(-1, 1)
