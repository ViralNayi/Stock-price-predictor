"""
LSTM Model Architecture for Stock Price Prediction
Defines, builds, and manages the LSTM neural network
"""

from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os


def build_model(input_shape, lstm_units=50, dropout_rate=0.2):
    """
    Build LSTM model architecture
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features)
        lstm_units (int): Number of LSTM units per layer (default: 50)
        dropout_rate (float): Dropout rate to prevent overfitting (default: 0.2)
    
    Returns:
        keras.Model: Compiled LSTM model
    """
    model = Sequential([
        # First LSTM layer with return sequences
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        
        # Second LSTM layer with return sequences
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        
        # Third LSTM layer
        LSTM(lstm_units),
        Dropout(dropout_rate),
        
        # Output layer
        Dense(1)
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


def get_model_summary(model):
    """
    Get model architecture summary as string
    
    Args:
        model: Keras model
    
    Returns:
        str: Model summary
    """
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    return '\n'.join(summary_list)


def save_model(model, filepath):
    """
    Save trained model to disk
    
    Args:
        model: Keras model to save
        filepath (str): Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from disk
    
    Args:
        filepath (str): Path to the saved model
    
    Returns:
        keras.Model: Loaded model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = keras_load_model(filepath)
    print(f"Model loaded from {filepath}")
    
    return model


def create_early_stopping(patience=10, min_delta=0.0001):
    """
    Create EarlyStopping callback to prevent overfitting
    
    Args:
        patience (int): Number of epochs with no improvement to wait
        min_delta (float): Minimum change to qualify as improvement
    
    Returns:
        EarlyStopping: Keras callback
    """
    return EarlyStopping(
        monitor='val_loss',
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=1
    )
