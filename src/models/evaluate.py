"""
Module for evaluating trained electricity consumption prediction models.
"""
import os
import pandas as pd
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

from src.utils.config import MODEL_DIR, METRICS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)

def load_trained_model(model_path: str) -> Any:
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        if model_path.endswith('.keras'):
            # Load TensorFlow model
            model = load_model(model_path)
            model_type = 'deep_learning'
        elif model_path.endswith('.pkl'):
            # Load pickle model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            model_type = 'traditional'
        else:
            raise ValueError(f"Unknown model file type: {model_path}")
        
        logging.info(f"Loaded {model_type} model from {model_path}")
        return model, model_type
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise

def load_data_info(data_info_path: str) -> Dict[str, Any]:
    """
    Load data processing info from file.
    
    Args:
        data_info_path: Path to the data info file
        
    Returns:
        Data info dictionary
    """
    if not os.path.exists(data_info_path):
        raise FileNotFoundError(f"Data info file not found: {data_info_path}")
    
    try:
        with open(data_info_path, 'rb') as f:
            data_info = pickle.load(f)
        
        logging.info(f"Loaded data info from {data_info_path}")
        return data_info
    except Exception as e:
        logging.error(f"Error loading data info from {data_info_path}: {e}")
        raise

def prepare_test_data(test_data: pd.DataFrame,
                     data_info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare test data for model evaluation.
    
    Args:
        test_data: Test DataFrame
        data_info: Data processing info dictionary
        
    Returns:
        Tuple of (X_test, y_test) numpy arrays
    """
    feature_cols = data_info['feature_cols']
    target_col = data_info['target_col']
    lookback = data_info['lookback']
    forecast_horizon = data_info['forecast_horizon']
    feature_scaler = data_info['feature_scaler']
    target_scaler = data_info['target_scaler']
    
    # Extract features and target
    features = test_data[feature_cols].values
    target = test_data[target_col].values
    
    # Scale the data
    scaled_features = feature_scaler.transform(features)
    scaled_target = target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = [], []
    
    for i in range(len(scaled_features) - lookback - forecast_horizon + 1):
        X.append(scaled_features[i:i+lookback])
        y.append(scaled_target[i+lookback:i+lookback+forecast_horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    logging.info(f"Prepared test data with {len(X)} sequences")
    
    return X, y

def calculate_advanced_metrics(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             data_info: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Calculate advanced evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        data_info: Optional data info with scalers for original scale metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # Try to calculate percentage metrics (may fail if values are zero)
    try:
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    except:
        metrics['mape'] = np.nan
    
    # R-squared and explained variance
    try:
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
    except:
        metrics['r2'] = np.nan
        metrics['explained_variance'] = np.nan
    
    # If we have data_info with scalers, calculate metrics in original scale
    if data_info and 'target_scaler' in data_info:
        # Reshape data if needed
        if len(y_true.shape) == 1:
            y_true_reshaped = y_true.reshape(-1, 1)
            y_pred_reshaped = y_pred.reshape(-1, 1)
        else:
            # Flatten multi-step predictions
            y_true_reshaped = y_true.reshape(-1, 1)
            y_pred_reshaped = y_pred.reshape(-1, 1)
        
        # Inverse transform
        y_true_orig = data_info['target_scaler'].inverse_transform(y_true_reshaped).flatten()
        y_pred_orig = data_info['target_scaler'].inverse_transform(y_pred_reshaped).flatten()
        
        # Calculate metrics in original scale
        metrics['mse_orig'] = mean_squared_error(y_true_orig, y_pred_orig)
        metrics['rmse_orig'] = np.sqrt(metrics['mse_orig'])
        metrics['mae_orig'] = mean_absolute_error(y_true_orig, y_pred_orig)
        
        try:
            metrics['mape_orig'] = mean_absolute_percentage_error(y_true_orig, y_pred_orig)
        except:
            metrics['mape_orig'] = np.nan
        
        try:
            metrics['r2_orig'] = r2_score(y_true_orig, y_pred_orig)
        except:
            metrics['r2_orig'] = np.nan
    
    return metrics

def predict_and_evaluate(model: Any,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       data_info: Dict[str, Any],
                       model_type: str = 'deep_learning') -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Make predictions and evaluate a model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        data_info: Data processing info
        model_type: Type of model ('deep_learning' or 'traditional')
        
    Returns:
        Tuple of (predictions, metrics)
    """
    # Make predictions
    if model_type == 'deep_learning':
        y_pred = model.predict(X_test)
    else:
        # Handle 3D input for sequence models by reshaping
        if len(X_test.shape) == 3:
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        else:
            X_test_reshaped = X_test
        
        # Handle multiple models for multi-output
        if isinstance(model, list):
            y_pred = np.column_stack([m.predict(X_test_reshaped) for m in model])
        else:
            y_pred = model.predict(X_test_reshaped)
    
    # Calculate metrics
    metrics = calculate_advanced_metrics(y_test, y_pred, data_info)
    
    # Log metrics
    logging.info("Model evaluation metrics:")
    for name, value in metrics.items():
        if not np.isnan(value):
            logging.info(f"{name}: {value:.6f}")
    
    return y_pred, metrics

def plot_predictions(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   data_info: Dict[str, Any],
                   title: str = "Model Predictions",
                   num_samples: int = 5,
                   save_path: Optional[str] = None) -> None:
    """
    Plot true vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        data_info: Data processing info
        title: Plot title
        num_samples: Number of samples to plot
        save_path: Optional path to save the plot
    """
    # Inverse transform if scalers are available
    if 'target_scaler' in data_info:
        # Get a sample of sequences to plot
        indices = np.random.choice(len(y_true), min(num_samples, len(y_true)), replace=False)
        
        # Set up the figure
        fig, axs = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
        if num_samples == 1:
            axs = [axs]
        
        for i, idx in enumerate(indices):
            # Get the current sample
            true_seq = y_true[idx]
            pred_seq = y_pred[idx]
            
            # Reshape for inverse transform
            true_seq_reshaped = true_seq.reshape(-1, 1)
            pred_seq_reshaped = pred_seq.reshape(-1, 1)
            
            # Inverse transform to original scale
            true_orig = data_info['target_scaler'].inverse_transform(true_seq_reshaped).flatten()
            pred_orig = data_info['target_scaler'].inverse_transform(pred_seq_reshaped).flatten()
            
            # Plot
            ax = axs[i]
            ax.plot(true_orig, label='Actual', marker='o')
            ax.plot(pred_orig, label='Predicted', marker='x')
            ax.set_title(f'Sample {idx}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f"{data_info['target_col']} (Original Scale)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02, fontsize=16)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logging.info(f"Saved prediction plot to {save_path}")
        
        plt.close()
    else:
        logging.warning("No target scaler found in data_info, skipping prediction plot")

def plot_forecast_comparison(models_dict: Dict[str, Any],
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           data_info: Dict[str, Any],
                           sample_idx: int = 0,
                           save_path: Optional[str] = None) -> None:
    """
    Plot a comparison of forecasts from different models.
    
    Args:
        models_dict: Dictionary of models {name: (model, model_type)}
        X_test: Test features
        y_test: Test targets
        data_info: Data processing info
        sample_idx: Index of the sample to plot
        save_path: Optional path to save the plot
    """
    # Get predictions from each model
    predictions = {}
    for name, (model, model_type) in models_dict.items():
        if model_type == 'deep_learning':
            pred = model.predict(X_test[sample_idx:sample_idx+1])[0]
        else:
            # Handle 3D input for sequence models
            if len(X_test.shape) == 3:
                X_sample = X_test[sample_idx].reshape(1, -1)
            else:
                X_sample = X_test[sample_idx:sample_idx+1]
            
            # Handle multiple models for multi-output
            if isinstance(model, list):
                pred = np.array([m.predict(X_sample)[0] for m in model])
            else:
                pred = model.predict(X_sample)[0]
        
        predictions[name] = pred
    
    # Get the true values
    true_values = y_test[sample_idx]
    
    # Inverse transform if scalers are available
    if 'target_scaler' in data_info:
        # Reshape for inverse transform
        true_reshaped = true_values.reshape(-1, 1)
        true_orig = data_info['target_scaler'].inverse_transform(true_reshaped).flatten()
        
        # Inverse transform predictions
        pred_orig = {}
        for name, pred in predictions.items():
            pred_reshaped = pred.reshape(-1, 1)
            pred_orig[name] = data_info['target_scaler'].inverse_transform(pred_reshaped).flatten()
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(true_orig, 'k-', marker='o', linewidth=2, label='Actual')
        
        for name, pred in pred_orig.items():
            plt.plot(pred, marker='x', linestyle='--', label=name)
        
        plt.title(f'Forecast Comparison (Sample {sample_idx})')
        plt.xlabel('Forecast Horizon (Time Steps)')
        plt.ylabel(f"{data_info['target_col']} (Original Scale)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logging.info(f"Saved forecast comparison plot to {save_path}")
        
        plt.close()
    else:
        logging.warning("No target scaler found in data_info, skipping forecast comparison plot")

def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                          metric_names: List[str] = ['rmse', 'mae', 'r2'],
                          title: str = "Model Metrics Comparison",
                          save_path: Optional[str] = None) -> None:
    """
    Plot a comparison of metrics across different models.
    
    Args:
        metrics_dict: Dictionary of metrics {model_name: {metric_name: value}}
        metric_names: List of metric names to include in the plot
        title: Plot title
        save_path: Optional path to save the plot
    """
    model_names = list(metrics_dict.keys())
    
    # Create a dataframe for easier plotting
    data = []
    for model_name, metrics in metrics_dict.items():
        for metric_name in metric_names:
            if metric_name in metrics:
                data.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Value': metrics[metric_name]
                })
    
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Metric', y='Value', hue='Model', data=df)
    plt.title(title)
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Saved metrics comparison plot to {save_path}")
    
    plt.close()

def evaluate_and_compare_models(models_dict: Dict[str, Tuple[Any, str]],
                              test_data: pd.DataFrame,
                              data_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate and compare multiple models on test data.
    
    Args:
        models_dict: Dictionary of models {name: (model, model_type)}
        test_data: Test DataFrame
        data_info: Data processing info
        
    Returns:
        Dictionary with evaluation results
    """
    # Prepare test data
    X_test, y_test = prepare_test_data(test_data, data_info)
    
    # Evaluate each model
    predictions = {}
    metrics = {}
    
    for name, (model, model_type) in models_dict.items():
        logging.info(f"Evaluating model: {name}")
        pred, model_metrics = predict_and_evaluate(model, X_test, y_test, data_info, model_type)
        predictions[name] = pred
        metrics[name] = model_metrics
    
    # Find the best model based on RMSE
    best_model = min(metrics.items(), key=lambda x: x[1]['rmse'])
    logging.info(f"Best model: {best_model[0]} with RMSE: {best_model[1]['rmse']:.6f}")
    
    # Create plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Plot predictions for each model
    for name in models_dict.keys():
        plot_predictions(
            y_test, predictions[name], data_info,
            title=f"{name} Predictions",
            save_path=os.path.join(METRICS_DIR, f"{name}_predictions_{timestamp}.png")
        )
    
    # Plot forecast comparison
    plot_forecast_comparison(
        models_dict, X_test, y_test, data_info,
        save_path=os.path.join(METRICS_DIR, f"forecast_comparison_{timestamp}.png")
    )
    
    # Plot metrics comparison
    plot_metrics_comparison(
        metrics,
        save_path=os.path.join(METRICS_DIR, f"metrics_comparison_{timestamp}.png")
    )
    
    # Save evaluation results
    results = {
        'metrics': metrics,
        'best_model': best_model[0],
        'timestamp': timestamp
    }
    
    with open(os.path.join(METRICS_DIR, f"evaluation_results_{timestamp}.pkl"), 'wb') as f:
        pickle.dump(results, f)
    
    return results

def plot_feature_importance(model: Any,
                          feature_names: List[str],
                          model_type: str = 'traditional',
                          top_n: int = 10,
                          save_path: Optional[str] = None) -> None:
    """
    Plot feature importance for a trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_type: Type of model
        top_n: Number of top features to show
        save_path: Optional path to save the plot
    """
    if model_type != 'traditional':
        logging.warning(f"Feature importance plot not supported for {model_type} models")
        return
    
    # Extract feature importances based on model type
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif isinstance(model, list) and hasattr(model[0], 'feature_importances_'):
        # For multi-output models, average the feature importances
        importances = np.mean([m.feature_importances_ for m in model], axis=0)
    else:
        logging.warning("Model does not have feature_importances_ attribute")
        return
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance and take top N
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Saved feature importance plot to {save_path}")
    
    plt.close()

if __name__ == "__main__":
    # Example usage
    logging.info("This module is meant to be imported, not run directly")
