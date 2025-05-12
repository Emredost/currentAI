# Developer Guide

This document provides information for developers who want to understand or contribute to the Smart Meters in London project.

## Project Structure

```
.
├── app.py                      # Streamlit web application
├── run.py                      # Main startup script
├── run.sh                      # Shell script wrapper
├── dvc.yaml                    # DVC pipeline configuration
├── data/                       # Data directory
│   ├── processed/              # Cleaned & processed data
│   └── raw/                    # Original raw data
├── models/                     # Trained forecasting models
├── notebooks/                  # Jupyter notebooks for exploration
├── src/                        # Source code
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Model training and evaluation
│   ├── pipelines/              # Data processing pipelines
│   ├── analysis/               # Analysis and visualization tools
│   └── utils/                  # Utility functions and config
├── docs/                       # Project documentation
└── tests/                      # Unit tests
```

## Key Modules

### 1. Data Processing

- `src/data/load_data.py`: Functions to load raw data from various sources
- `src/data/preprocess.py`: Data cleaning and preprocessing functions
- `src/pipelines/pipeline.py`: Data processing orchestration with `SmartMeterPipeline` class

### 2. Models

- `src/models/train.py`: Model training and hyperparameter tuning
- `src/models/evaluate.py`: Model evaluation metrics and visualization

### 3. Analysis

- `src/analysis/visualization.py`: Reusable plotting functions
- `src/analysis/eda.py`: Exploratory data analysis utilities

### 4. Utils

- `src/utils/config.py`: Configuration parameters and file paths
- `src/utils/dataset_utils.py`: Dataset manipulation utilities

## Development Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd smart-meters-london
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install development dependencies:
   ```
   pip install -r requirements-dev.txt
   ```

4. Set up DVC for data versioning:
   ```
   dvc init
   dvc add data/raw
   ```

## Data Pipeline

The data pipeline is defined in `dvc.yaml` and can be executed with:

```
dvc repro
```

This will:
1. Process the raw data
2. Generate analysis-ready datasets
3. Train forecasting models

## Contributing Guidelines

### Code Style

- We follow PEP 8 for Python code style
- Use type hints for better code documentation
- Document all functions and classes with docstrings
- Add logging for all key operations

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes and add appropriate tests
3. Run the test suite to ensure all tests pass
4. Submit a pull request with a clear description of the changes

### Testing

Run the test suite with:

```
pytest tests/
```

## Model Development

When developing new models:

1. Create a new module in `src/models/`
2. Implement training and evaluation functions
3. Add appropriate tests in `tests/models/`
4. Update the model comparison in the web app if needed

## Web Application

The Streamlit web application is defined in `app.py`. To extend it:

1. Create a new tab or section
2. Add visualization components
3. Ensure it loads data efficiently using caching
4. Test the UI with different datasets 