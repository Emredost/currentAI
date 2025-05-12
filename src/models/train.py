"""
Module for training electricity consumption prediction models.
"""
import os
import pandas as pd
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from src.utils.config import MODEL_DIR, MODEL_PARAMS, TRAINING_PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)

def create_sequence_dataset(data: pd.DataFrame,
                           target_col: str,
                           feature_cols: List[str],
                           lookback: int = 24,
                           forecast_horizon: int = 24,
                           batch_size: int = 32) -> Tuple[tf.data.Dataset, Dict[str, Any]]:
    """
    Create a TensorFlow dataset for sequence prediction.
    
    Args:
        data: Input DataFrame with time-ordered data
        target_col: Column name for the target variable
        feature_cols: List of feature column names
        lookback: Number of time steps to look back
        forecast_horizon: Number of time steps to predict ahead
        batch_size: Batch size for the dataset
    
    Returns:
        TensorFlow dataset and a dictionary with data processing information
    """
    # Extract features and target
    features = data[feature_cols].values
    target = data[target_col].values
    
    # Scale the data
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = [], []
    
    for i in range(len(scaled_features) - lookback - forecast_horizon + 1):
        X.append(scaled_features[i:i+lookback])
        y.append(scaled_target[i+lookback:i+lookback+forecast_horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=MODEL_PARAMS['validation_size'],
        random_state=MODEL_PARAMS['random_state']
    )
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    
    # Create combined dataset for returning
    dataset = {
        'train': train_dataset,
        'val': val_dataset,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val
    }
    
    # Save data processing info for later use
    data_info = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'lookback': lookback,
        'forecast_horizon': forecast_horizon
    }
    
    logging.info(f"Created sequence dataset with {len(X)} sequences")
    logging.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    return dataset, data_info

def create_lstm_model(input_shape: Tuple[int, int], 
                     output_size: int,
                     units: List[int] = [64, 32],
                     dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    Create an LSTM model for time series forecasting.
    
    Args:
        input_shape: Shape of the input sequence (timesteps, features)
        output_size: Number of output time steps to predict
        units: List of units in each LSTM layer
        dropout_rate: Dropout rate to prevent overfitting
    
    Returns:
        Compiled LSTM model
    """
    model = Sequential()
    
    # Add LSTM layers with dropout
    model.add(LSTM(units[0], return_sequences=len(units) > 1, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    
    # Add additional LSTM layers if specified
    for i in range(1, len(units)):
        return_sequences = i < len(units) - 1
        model.add(LSTM(units[i], return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
    
    # Output layer
    model.add(Dense(output_size))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=TRAINING_PARAMS['learning_rate']),
        loss='mse'
    )
    
    return model

def create_gru_model(input_shape: Tuple[int, int],
                    output_size: int,
                    units: List[int] = [64, 32],
                    dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    Create a GRU model for time series forecasting.
    
    Args:
        input_shape: Shape of the input sequence (timesteps, features)
        output_size: Number of output time steps to predict
        units: List of units in each GRU layer
        dropout_rate: Dropout rate to prevent overfitting
    
    Returns:
        Compiled GRU model
    """
    model = Sequential()
    
    # Add GRU layers with dropout
    model.add(GRU(units[0], return_sequences=len(units) > 1, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    
    # Add additional GRU layers if specified
    for i in range(1, len(units)):
        return_sequences = i < len(units) - 1
        model.add(GRU(units[i], return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
    
    # Output layer
    model.add(Dense(output_size))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=TRAINING_PARAMS['learning_rate']),
        loss='mse'
    )
    
    return model

def create_cnn_model(input_shape: Tuple[int, int],
                    output_size: int,
                    filters: List[int] = [64, 32],
                    kernel_size: int = 3,
                    dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    Create a 1D CNN model for time series forecasting.
    
    Args:
        input_shape: Shape of the input sequence (timesteps, features)
        output_size: Number of output time steps to predict
        filters: List of filters in each Conv1D layer
        kernel_size: Size of the convolutional kernel
        dropout_rate: Dropout rate to prevent overfitting
    
    Returns:
        Compiled CNN model
    """
    model = Sequential()
    
    # Add Conv1D layers with dropout
    model.add(Conv1D(filters[0], kernel_size=kernel_size, activation='relu', input_shape=input_shape, padding='same'))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    
    # Add additional Conv1D layers if specified
    for i in range(1, len(filters)):
        model.add(Conv1D(filters[i], kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
    
    # Flatten and output layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_size))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=TRAINING_PARAMS['learning_rate']),
        loss='mse'
    )
    
    return model

def train_deep_learning_model(model: tf.keras.Model,
                             dataset: Dict[str, Any],
                             model_name: str,
                             epochs: int = TRAINING_PARAMS['epochs'],
                             patience: int = TRAINING_PARAMS['early_stopping_patience']) -> tf.keras.Model:
    """
    Train a deep learning model.
    
    Args:
        model: Compiled TensorFlow model
        dataset: Dictionary with train and val datasets
        model_name: Name for the saved model
        epochs: Number of training epochs
        patience: Early stopping patience
    
    Returns:
        Trained model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{timestamp}.keras")
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-6)
    ]
    
    # Train the model
    logging.info(f"Training {model_name} model...")
    history = model.fit(
        dataset['train'],
        validation_data=dataset['val'],
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    history_path = os.path.join(MODEL_DIR, f"{model_name}_{timestamp}_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    logging.info(f"Model trained and saved to {model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Loss (Log Scale)')
    plt.ylabel('Loss (log)')
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{model_name}_{timestamp}_training.png"))
    
    return model

def train_traditional_model(X_train: np.ndarray, 
                          y_train: np.ndarray,
                          model_type: str = 'random_forest',
                          model_params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Train a traditional machine learning model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        model_type: Type of model ('random_forest', 'gradient_boosting', 'linear')
        model_params: Optional parameters for the model
    
    Returns:
        Trained model
    """
    # Default parameters if none provided
    if model_params is None:
        model_params = {}
    
    # Add random state if not specified
    if 'random_state' not in model_params:
        model_params['random_state'] = MODEL_PARAMS['random_state']
    
    # Create and train model
    if model_type == 'random_forest':
        model = RandomForestRegressor(**model_params)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(**model_params)
    elif model_type == 'linear':
        model = LinearRegression(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Handle 3D input for sequence models
    if len(X_train.shape) == 3:
        # Reshape to 2D
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    else:
        X_train_reshaped = X_train
    
    # If y is 2D and model doesn't support multi-output, train separate models
    if len(y_train.shape) == 2 and y_train.shape[1] > 1 and model_type != 'linear':
        logging.info(f"Training {model_type} with multi-output strategy")
        models = []
        for i in range(y_train.shape[1]):
            model_i = type(model)(**model_params)
            model_i.fit(X_train_reshaped, y_train[:, i])
            models.append(model_i)
        model = models
    else:
        logging.info(f"Training {model_type} model")
        model.fit(X_train_reshaped, y_train)
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"{model_type}_{timestamp}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logging.info(f"Model trained and saved to {model_path}")
    
    return model

def evaluate_model(model: Any,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 data_info: Dict[str, Any],
                 model_type: str = 'deep_learning') -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        data_info: Dictionary with data processing information
        model_type: Type of model ('deep_learning', 'traditional')
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    if model_type == 'deep_learning':
        y_pred = model.predict(X_test)
    elif model_type == 'traditional':
        # Handle 3D input for sequence models
        if len(X_test.shape) == 3:
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        else:
            X_test_reshaped = X_test
        
        # Handle multiple models for multi-output
        if isinstance(model, list):
            y_pred = np.column_stack([m.predict(X_test_reshaped) for m in model])
        else:
            y_pred = model.predict(X_test_reshaped)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calculate metrics
    metrics = {}
    
    # Handle different dimensionality
    if len(y_test.shape) == 2 and y_test.shape[1] > 1:
        # Multiple time steps
        mse_per_step = np.mean((y_test - y_pred) ** 2, axis=0)
        mae_per_step = np.mean(np.abs(y_test - y_pred), axis=0)
        
        # Average metrics
        metrics['mse'] = np.mean(mse_per_step)
        metrics['mae'] = np.mean(mae_per_step)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Add per-step metrics
        for i, (mse_i, mae_i) in enumerate(zip(mse_per_step, mae_per_step)):
            metrics[f'mse_step_{i+1}'] = mse_i
            metrics[f'mae_step_{i+1}'] = mae_i
            metrics[f'rmse_step_{i+1}'] = np.sqrt(mse_i)
    else:
        # Single output
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_test, y_pred)
    
    # If we have scalers, add metrics in original scale
    if 'target_scaler' in data_info:
        # Reshape if needed to match scaler expectations
        if len(y_test.shape) == 1:
            y_test_reshaped = y_test.reshape(-1, 1)
            y_pred_reshaped = y_pred.reshape(-1, 1)
        else:
            # Flatten multi-step predictions
            y_test_reshaped = y_test.reshape(-1, 1)
            y_pred_reshaped = y_pred.reshape(-1, 1)
        
        # Inverse transform
        y_test_orig = data_info['target_scaler'].inverse_transform(y_test_reshaped)
        y_pred_orig = data_info['target_scaler'].inverse_transform(y_pred_reshaped)
        
        # Calculate metrics in original scale
        metrics['mse_orig'] = mean_squared_error(y_test_orig, y_pred_orig)
        metrics['mae_orig'] = mean_absolute_error(y_test_orig, y_pred_orig)
        metrics['rmse_orig'] = np.sqrt(metrics['mse_orig'])
    
    # Log and return metrics
    logging.info(f"Model evaluation metrics:")
    for name, value in metrics.items():
        logging.info(f"{name}: {value:.6f}")
    
    return metrics

def train_electricity_forecast_models(data: pd.DataFrame,
                                    target_col: str = 'energy(kWh/hh)',
                                    feature_cols: Optional[List[str]] = None,
                                    lookback: int = 24,
                                    forecast_horizon: int = 24) -> Dict[str, Any]:
    """
    Train multiple models for electricity consumption forecasting.
    
    Args:
        data: Input DataFrame with time-ordered data
        target_col: Column name for the target variable
        feature_cols: List of feature column names (if None, will use all numeric columns except target)
        lookback: Number of time steps to look back
        forecast_horizon: Number of time steps to predict ahead
    
    Returns:
        Dictionary with trained models and evaluation metrics
    """
    # Select features if not provided
    if feature_cols is None:
        feature_cols = data.select_dtypes(include=np.number).columns.tolist()
        # Remove target from features if it's there
        if target_col in feature_cols:
            feature_cols.remove(target_col)
    
    logging.info(f"Training forecast models with {len(feature_cols)} features")
    logging.info(f"Features: {feature_cols}")
    
    # Create sequence dataset
    dataset, data_info = create_sequence_dataset(
        data=data,
        target_col=target_col,
        feature_cols=feature_cols,
        lookback=lookback,
        forecast_horizon=forecast_horizon,
        batch_size=TRAINING_PARAMS['batch_size']
    )
    
    # Get input and output shapes
    input_shape = (lookback, len(feature_cols))
    
    # Train models
    models = {}
    metrics = {}
    
    # LSTM Model
    lstm_model = create_lstm_model(input_shape, forecast_horizon)
    trained_lstm = train_deep_learning_model(lstm_model, dataset, model_name='lstm')
    models['lstm'] = trained_lstm
    metrics['lstm'] = evaluate_model(trained_lstm, dataset['X_val'], dataset['y_val'], data_info)
    
    # GRU Model
    gru_model = create_gru_model(input_shape, forecast_horizon)
    trained_gru = train_deep_learning_model(gru_model, dataset, model_name='gru')
    models['gru'] = trained_gru
    metrics['gru'] = evaluate_model(trained_gru, dataset['X_val'], dataset['y_val'], data_info)
    
    # CNN Model
    cnn_model = create_cnn_model(input_shape, forecast_horizon)
    trained_cnn = train_deep_learning_model(cnn_model, dataset, model_name='cnn')
    models['cnn'] = trained_cnn
    metrics['cnn'] = evaluate_model(trained_cnn, dataset['X_val'], dataset['y_val'], data_info)
    
    # Random Forest (traditional)
    rf_model = train_traditional_model(
        dataset['X_train'], dataset['y_train'], 
        model_type='random_forest', 
        model_params={'n_estimators': 100, 'max_depth': 10}
    )
    models['random_forest'] = rf_model
    metrics['random_forest'] = evaluate_model(
        rf_model, dataset['X_val'], dataset['y_val'], 
        data_info, model_type='traditional'
    )
    
    # Compare models and find the best one
    best_model = min(metrics.items(), key=lambda x: x[1]['rmse'])
    logging.info(f"Best model: {best_model[0]} with RMSE: {best_model[1]['rmse']:.6f}")
    
    # Create a comparison plot
    plt.figure(figsize=(12, 6))
    model_names = list(metrics.keys())
    rmse_values = [metrics[m]['rmse'] for m in model_names]
    mae_values = [metrics[m]['mae'] for m in model_names]
    
    bar_width = 0.35
    x = np.arange(len(model_names))
    
    plt.bar(x - bar_width/2, rmse_values, width=bar_width, label='RMSE')
    plt.bar(x + bar_width/2, mae_values, width=bar_width, label='MAE')
    
    plt.xlabel('Model')
    plt.ylabel('Error')
    plt.title('Model Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(MODEL_DIR, f"model_comparison_{timestamp}.png"))
    
    # Save data info for later use
    with open(os.path.join(MODEL_DIR, f"data_info_{timestamp}.pkl"), 'wb') as f:
        pickle.dump(data_info, f)
    
    return {
        'models': models,
        'metrics': metrics,
        'data_info': data_info,
        'best_model': best_model[0]
    }

if __name__ == "__main__":
    # Example usage
    logging.info("This module is meant to be imported, not run directly")
