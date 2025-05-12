# Smart Meter Analytics: Technical Methods

This document outlines the technical approaches, data processing methods, and modeling techniques used in the Smart Meter Analytics project.

## Data Acquisition and Preprocessing

### Data Sources
- **Smart Meter Readings**: Half-hourly electricity consumption from 5,000+ London households (2011-2014)
- **Weather Data**: Hourly weather observations from Dark Sky API
- **Household Information**: ACORN classification, tariff type, and other demographic details

### Preprocessing Pipeline

The data preprocessing pipeline follows these stages:

1. **Initial Data Loading**
   ```python
   # Load raw data with retry mechanism for large files
   consumption_data = load_with_retry('data/raw/halfhourly_dataset.csv')
   weather_data = load_with_retry('data/raw/weather_hourly_darksky.csv')
   household_data = load_with_retry('data/raw/informations_households.csv')
   ```

2. **Missing Value Handling**
   - For consumption data: Linear interpolation for short gaps, mean imputation by time-of-day for longer gaps
   - For weather data: Forward fill for short gaps, interpolation based on neighboring hours for longer gaps
   - For household data: No imputation (complete dataset)

   ```python
   # Example of time-based imputation for consumption data
   def handle_consumption_missing(df):
       # Group by household and time features
       groups = df.groupby(['LCLid', df['tstp'].dt.hour, df['tstp'].dt.dayofweek])
       
       # Fill missing values with mean of same hour and day of week
       df['energy(kWh/hh)'] = df.groupby(['LCLid', 
                                          df['tstp'].dt.hour, 
                                          df['tstp'].dt.dayofweek])['energy(kWh/hh)'].transform(
           lambda x: x.fillna(x.mean()))
       
       return df
   ```

3. **Outlier Detection and Handling**
   - Statistical approach: IQR method with 1.5x multiplier, segmented by household
   - Domain knowledge filters: Maximum consumption thresholds based on typical household capacity
   - Replacement strategy: Capping for mild outliers, removal for extreme outliers

   ```python
   def detect_outliers(df, column, threshold=1.5):
       # Calculate IQR for each household
       Q1 = df.groupby('LCLid')[column].transform('quantile', 0.25)
       Q3 = df.groupby('LCLid')[column].transform('quantile', 0.75)
       IQR = Q3 - Q1
       
       # Create outlier mask
       outlier_mask = (df[column] < (Q1 - threshold * IQR)) | (df[column] > (Q3 + threshold * IQR))
       
       return outlier_mask
   ```

4. **Feature Engineering**
   - **Temporal Features**: Hour of day, day of week, month, season, is_weekend, is_holiday
   - **Weather Derivatives**: Temperature lag features, moving averages, rate of change
   - **Household Features**: ACORN group encoding, tariff type encoding

   ```python
   def add_temporal_features(df, timestamp_col):
       # Extract time-based features
       df['hour'] = df[timestamp_col].dt.hour
       df['day'] = df[timestamp_col].dt.dayofweek
       df['month'] = df[timestamp_col].dt.month
       df['is_weekend'] = df['day'].isin([5, 6])  # 5=Saturday, 6=Sunday
       
       # Define seasons (1=Winter, 2=Spring, 3=Summer, 4=Autumn)
       season_mapping = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1}
       df['season'] = df['month'].map(season_mapping)
       
       return df
   ```

5. **Data Integration**
   - Join consumption data with household information using household ID
   - Merge weather data based on timestamp alignment
   - Time-based aggregation for analysis dataset creation

## Exploratory Data Analysis

### Key Analysis Techniques

1. **Temporal Decomposition**
   - Time series decomposition into trend, seasonality, and residual components
   - Autocorrelation and partial autocorrelation analysis
   - Seasonal patterns visualization using calendar heatmaps

2. **Consumption Pattern Analysis**
   - Hourly consumption profiles by household type
   - Weekend vs. weekday pattern comparison
   - Load duration curve analysis

3. **Weather Relationship Analysis**
   - Temperature response curves using LOESS smoothing
   - Correlation analysis with various weather parameters
   - Multivariate analysis of weather impact

4. **Customer Segmentation**
   - K-means clustering based on consumption patterns
   - ACORN group consumption comparison
   - Tariff impact analysis

## Predictive Modeling

### Feature Selection

Feature importance was evaluated using multiple methods:
- Random Forest feature importance
- Permutation importance
- SHAP values
- Mutual information

The final feature set includes 15 variables with the highest importance scores across methods.

### Model Architectures

1. **LSTM Neural Network**
   ```python
   def create_lstm_model(seq_length, n_features, lstm_units=[64, 32], dropout_rate=0.2):
       model = Sequential()
       
       # First LSTM layer with return sequences
       model.add(LSTM(lstm_units[0], 
                     activation='relu',
                     return_sequences=True,
                     input_shape=(seq_length, n_features)))
       model.add(Dropout(dropout_rate))
       
       # Second LSTM layer
       model.add(LSTM(lstm_units[1], activation='relu'))
       model.add(Dropout(dropout_rate))
       
       # Output layer
       model.add(Dense(1))
       
       model.compile(optimizer='adam', loss='mse', metrics=['mae'])
       return model
   ```

2. **GRU Neural Network**
   ```python
   def create_gru_model(seq_length, n_features, gru_units=[64, 32], dropout_rate=0.2):
       model = Sequential()
       
       # First GRU layer with return sequences
       model.add(GRU(gru_units[0], 
                    activation='relu',
                    return_sequences=True,
                    input_shape=(seq_length, n_features)))
       model.add(Dropout(dropout_rate))
       
       # Second GRU layer
       model.add(GRU(gru_units[1], activation='relu'))
       model.add(Dropout(dropout_rate))
       
       # Output layer
       model.add(Dense(1))
       
       model.compile(optimizer='adam', loss='mse', metrics=['mae'])
       return model
   ```

3. **CNN Model**
   ```python
   def create_cnn_model(seq_length, n_features, filters=[64, 32], kernel_size=3):
       model = Sequential()
       
       # First convolutional layer
       model.add(Conv1D(filters=filters[0],
                        kernel_size=kernel_size,
                        activation='relu',
                        input_shape=(seq_length, n_features)))
       model.add(MaxPooling1D(pool_size=2))
       
       # Second convolutional layer
       model.add(Conv1D(filters=filters[1],
                        kernel_size=kernel_size,
                        activation='relu'))
       model.add(MaxPooling1D(pool_size=2))
       
       # Flatten and dense layers
       model.add(Flatten())
       model.add(Dense(32, activation='relu'))
       model.add(Dropout(0.2))
       model.add(Dense(1))
       
       model.compile(optimizer='adam', loss='mse', metrics=['mae'])
       return model
   ```

4. **Random Forest**
   ```python
   def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=20):
       # Create and train model
       model = RandomForestRegressor(
           n_estimators=n_estimators,
           max_depth=max_depth,
           min_samples_split=5,
           min_samples_leaf=2,
           random_state=42,
           n_jobs=-1
       )
       model.fit(X_train, y_train)
       
       # Evaluate model
       y_pred = model.predict(X_test)
       mse = mean_squared_error(y_test, y_pred)
       mae = mean_absolute_error(y_test, y_pred)
       r2 = r2_score(y_test, y_pred)
       
       return model, mse, mae, r2
   ```

### Training Methodology

1. **Data Preparation**
   - Train/validation/test split: 70%/15%/15%
   - Feature scaling: StandardScaler applied per household
   - Sequence preparation for neural networks: sliding window approach

2. **Hyperparameter Tuning**
   - Random Forest: Grid search with cross-validation
   - Neural Networks: Bayesian optimization using Hyperopt

3. **Training Process**
   - Early stopping based on validation loss
   - Learning rate scheduling
   - Model checkpointing for best performance

4. **Evaluation Metrics**
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R-squared (R²)
   - Mean Absolute Percentage Error (MAPE)

## Deployment Architecture

The application is containerized using Docker with the following components:

1. **Data Processing Service**
   - Handles data ingestion and preprocessing
   - Includes automated quality checks
   - Outputs standardized datasets for analysis and modeling

2. **Model Training Service**
   - On-demand or scheduled model training
   - Hyperparameter optimization
   - Model versioning and storage

3. **Web Dashboard**
   - Streamlit-based interactive visualization
   - Configurable views and parameters
   - Real-time forecasting capabilities

4. **API Service**
   - RESTful API for model inference
   - Data validation endpoints
   - Authentication and rate limiting

## Technical Performance

Performance metrics from production testing:

- **Data Processing**: 5 million records processed in ~3 minutes
- **Model Training**: LSTM model trained in ~25 minutes on CPU
- **Prediction Latency**: <100ms for individual household forecasts
- **Dashboard Responsiveness**: <2 second load time for main views

## Conclusion

The technical approach balances sophisticated modeling techniques with practical performance considerations, making it suitable for both batch processing and real-time applications in energy consumption analysis. 