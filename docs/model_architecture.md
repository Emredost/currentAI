# Model Architecture Documentation

This document describes the machine learning models used in the Smart Meters London project for electricity consumption forecasting.

## Model Overview

The project implements several time series forecasting models to predict household electricity consumption. The models are designed to capture temporal patterns, weather effects, and household characteristics.

## Input Features

All models use the following feature sets:

1. **Temporal Features**:
   - Hour of day (0-23)
   - Day of week (0-6)
   - Month of year (1-12)
   - Is weekend (binary)
   - Is holiday (binary)
   - Season (encoded)

2. **Weather Features**:
   - Temperature
   - Humidity
   - Wind speed
   - Cloud cover
   - Precipitation intensity

3. **Household Features**:
   - ACORN group (encoded)
   - Tariff type (encoded)

## Model Architectures

### 1. LSTM (Long Short-Term Memory)

The LSTM model is optimized for capturing long-term temporal dependencies in electricity consumption.

```
Input (shape=(sequence_length, features)) → 
LSTM(64, return_sequences=True) → 
Dropout(0.2) → 
LSTM(32) → 
Dropout(0.2) → 
Dense(16, activation='relu') → 
Dense(1, activation='linear')
```

**Hyperparameters**:
- Sequence length: 24-168 (1-7 days)
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Mean Squared Error

**Architecture Details**:
- The first LSTM layer processes the input sequence and returns sequences
- Dropout layers prevent overfitting
- The second LSTM layer consolidates features before final prediction
- Dense layers reduce dimensionality to make the final prediction

### 2. GRU (Gated Recurrent Unit)

The GRU model provides a more computationally efficient alternative to LSTM while maintaining good performance.

```
Input (shape=(sequence_length, features)) → 
GRU(64, return_sequences=True) → 
Dropout(0.2) → 
GRU(32) → 
Dropout(0.2) → 
Dense(16, activation='relu') → 
Dense(1, activation='linear')
```

**Hyperparameters**:
- Sequence length: 24-168 (1-7 days)
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Mean Squared Error

**Architecture Details**:
- Similar to LSTM but uses GRU cells which require fewer parameters
- The dual-layer architecture captures both short and long-term patterns
- Dropout and dense layers match the LSTM architecture for fair comparison

### 3. CNN (Convolutional Neural Network)

The CNN model is designed to capture local patterns within the time series data.

```
Input (shape=(sequence_length, features)) → 
Conv1D(64, kernel_size=3, activation='relu') → 
MaxPooling1D(pool_size=2) → 
Conv1D(32, kernel_size=3, activation='relu') → 
MaxPooling1D(pool_size=2) → 
Flatten() → 
Dense(32, activation='relu') → 
Dropout(0.2) → 
Dense(1, activation='linear')
```

**Hyperparameters**:
- Sequence length: 24-168 (1-7 days)
- Kernel size: 3
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Mean Squared Error

**Architecture Details**:
- 1D convolutional layers detect local patterns in the time series
- Max pooling reduces dimensionality and extracts dominant features
- Flatten layer converts the feature maps to a 1D vector
- Dense layers make the final prediction

### 4. Random Forest

A traditional machine learning model that serves as a baseline and provides good interpretability.

```
RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

**Hyperparameters**:
- Number of trees: 100
- Maximum depth: 20
- Minimum samples to split: 5
- Minimum samples per leaf: 2

**Architecture Details**:
- Ensemble of 100 decision trees
- Each tree trained on a bootstrap sample of the data
- Features randomly selected for each split
- Final prediction is the average of individual tree predictions

## Model Training

All models follow a similar training procedure:

1. Data is split into training (70%), validation (15%), and test (15%) sets
2. Data is normalized using standard scaling (zero mean, unit variance)
3. For sequence models, data is formatted into sliding windows
4. Models are trained with early stopping based on validation loss
5. Hyperparameters are tuned using cross-validation or Bayesian optimization

## Evaluation Metrics

Models are evaluated using several metrics:

1. **Root Mean Squared Error (RMSE)**: Measures the average magnitude of prediction errors
2. **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values
3. **R-squared (R²)**: Measures the proportion of variance in the target variable explained by the model
4. **Mean Absolute Percentage Error (MAPE)**: Measures the percentage difference between predicted and actual values

## Feature Importance

For the Random Forest model, feature importance is calculated to understand which factors most influence electricity consumption. Key features typically include:

1. Hour of day
2. Temperature
3. Day of week
4. ACORN group
5. Month

## Model Selection

The best model is selected based on performance metrics on the test dataset. Currently, the LSTM model shows the best performance with:
- RMSE: 0.114
- MAE: 0.087
- R²: 0.83

## Model Persistence

Models are saved using the following formats:
- Neural network models (LSTM, GRU, CNN): TensorFlow `.keras` format
- Random Forest: Pickle `.pkl` format
- Data preprocessing information: Pickle `.pkl` format

Models are stored in the `models/` directory and can be loaded for inference using the appropriate libraries. 