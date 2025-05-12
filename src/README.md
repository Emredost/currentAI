# Source Code Directory

This directory contains all source code for the Smart Meters in London project, organized into modules.

## Directory Structure

- `data/`: Data loading and preprocessing
- `models/`: Model training and evaluation
- `pipelines/`: Data processing pipelines
- `analysis/`: Analysis and visualization tools
- `utils/`: Utility functions and configuration

## Core Modules

### Data Processing

- `data/load_data.py`: Functions to load raw data from various sources
- `data/preprocess.py`: Data cleaning and preprocessing functions
- `download_data.py`: Utility to download dataset from Kaggle

### Machine Learning

- `models/train.py`: Model training and hyperparameter tuning
- `models/evaluate.py`: Model evaluation and visualization

### Pipelines

- `pipelines/pipeline.py`: Data processing orchestration with `SmartMeterPipeline` class

### Analysis

- `analysis/visualization.py`: Reusable plotting functions
- `analysis/eda.py`: Exploratory data analysis utilities

### Utilities

- `utils/config.py`: Configuration parameters and file paths
- `utils/dataset_utils.py`: Dataset manipulation utilities
- `utils/helpers.py`: General helper functions

## Code Conventions

The codebase follows these conventions:

- PEP 8 style guide for code formatting
- Type hints for function parameters and return values
- Docstrings for all functions and classes
- Logging for all key operations

## Example Usage

```python
# Initialize data pipeline
from src.pipelines.pipeline import SmartMeterPipeline
pipeline = SmartMeterPipeline()

# Load and process data
pipeline.load_raw_data(['household_info', 'weather_hourly', 'halfhourly_dataset'])
pipeline.preprocess_household_data()
pipeline.preprocess_weather_data(hourly=True)
pipeline.preprocess_consumption_data(halfhourly=True)

# Create analysis dataset
analysis_df = pipeline.create_analysis_dataset()
``` 