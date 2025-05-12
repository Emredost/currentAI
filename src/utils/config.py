"""
Configuration settings for the project.
Centralizes all paths, parameters and settings for easier maintenance.
"""
import os
from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Data paths
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external")

# Model paths
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Artifact paths
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")
LOG_DIR = os.path.join(ARTIFACT_DIR, "logs")
METRICS_DIR = os.path.join(ARTIFACT_DIR, "metrics")

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# File names
FILES = {
    "household_info": os.path.join(RAW_DATA_DIR, "informations_households.csv"),
    "acorn_details": os.path.join(RAW_DATA_DIR, "acorn_details.csv"),
    "bank_holidays": os.path.join(RAW_DATA_DIR, "uk_bank_holidays.csv"),
    "weather_daily": os.path.join(RAW_DATA_DIR, "weather_daily_darksky.csv"),
    "weather_hourly": os.path.join(RAW_DATA_DIR, "weather_hourly_darksky.csv"),
    "daily_dataset_dir": os.path.join(RAW_DATA_DIR, "daily_dataset"),
    "halfhourly_dataset_dir": os.path.join(RAW_DATA_DIR, "halfhourly_dataset"),
    "processed_daily": os.path.join(PROCESSED_DATA_DIR, "cleaned_daily_dataset.csv"),
    "processed_hourly": os.path.join(PROCESSED_DATA_DIR, "cleaned_halfhourly_data.csv"),
    "processed_weather": os.path.join(PROCESSED_DATA_DIR, "cleaned_weather_hourly.csv"),
    "household_processed": os.path.join(PROCESSED_DATA_DIR, "household_info_processed.csv"),
}

# Model parameters
MODEL_PARAMS = {
    "random_state": 42,
    "test_size": 0.2,
    "validation_size": 0.2,
    "lookback_window": 24,  # Hours to look back for time series prediction
    "forecast_horizon": 24,  # Hours to forecast ahead
}

# Training parameters
TRAINING_PARAMS = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
} 