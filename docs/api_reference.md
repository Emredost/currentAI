# API Reference

This document provides reference information for the key modules and classes in the Smart Meters in London project.

## Data Module

### `src.data.load_data`

#### Functions

- `validate_file(file_path: str) -> bool`: Validate that a file exists
- `load_with_retry(file_path: str, **kwargs) -> pd.DataFrame`: Load a CSV file with retry logic
- `load_household_info() -> pd.DataFrame`: Load household information
- `load_acorn_details() -> pd.DataFrame`: Load ACORN details
- `load_uk_bank_holidays() -> pd.DataFrame`: Load UK bank holidays
- `load_weather_data(daily: bool = True) -> pd.DataFrame`: Load weather data
- `load_from_directory(directory: str, limit: Optional[int] = None) -> pd.DataFrame`: Load all CSV files from a directory
- `load_daily_dataset(limit: Optional[int] = None) -> pd.DataFrame`: Load daily dataset files
- `load_halfhourly_dataset(limit: Optional[int] = None) -> pd.DataFrame`: Load half-hourly dataset files
- `load_processed_data(dataset_type: str) -> pd.DataFrame`: Load already processed data

### `src.data.preprocess`

#### Functions

- `detect_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame`: Detect outliers
- `handle_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr', threshold: float = 1.5, strategy: str = 'clip') -> pd.DataFrame`: Handle outliers
- `clean_missing_values(df: pd.DataFrame, strategy: str = "mean", columns: Optional[List[str]] = None, max_missing_pct: float = 50.0) -> pd.DataFrame`: Fill missing values
- `drop_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame`: Remove duplicate rows
- `format_datetime_column(df: pd.DataFrame, column: str, format: Optional[str] = None, add_components: bool = False) -> pd.DataFrame`: Convert a column to datetime format
- `encode_categorical(df: pd.DataFrame, columns: List[str], method: str = 'onehot', drop_original: bool = True) -> pd.DataFrame`: Encode categorical columns
- `normalize_data(df: pd.DataFrame, columns: List[str], method: str = 'standard', return_scaler: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]`: Normalize numeric columns
- `create_features(df: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame`: Create new features
- `save_processed_data(df: pd.DataFrame, file_name: str) -> None`: Save processed DataFrame to a CSV file
- `load_and_process_data(data_loader_func, preprocessing_steps: List[Dict[str, Any]], output_file: Optional[str] = None) -> pd.DataFrame`: Load and process data

## Models Module

### `src.models.train`

#### Functions

- `create_sequence_dataset(data: pd.DataFrame, target_col: str, feature_cols: List[str], lookback: int = 24, forecast_horizon: int = 24, batch_size: int = 32) -> Tuple[tf.data.Dataset, Dict[str, Any]]`: Create a TensorFlow dataset
- `create_lstm_model(input_shape: Tuple[int, int], output_size: int, units: List[int] = [64, 32], dropout_rate: float = 0.2) -> tf.keras.Model`: Create an LSTM model
- `create_gru_model(input_shape: Tuple[int, int], output_size: int, units: List[int] = [64, 32], dropout_rate: float = 0.2) -> tf.keras.Model`: Create a GRU model
- `create_cnn_model(input_shape: Tuple[int, int], output_size: int, filters: List[int] = [64, 32], kernel_size: int = 3, dropout_rate: float = 0.2) -> tf.keras.Model`: Create a CNN model
- `train_deep_learning_model(model: tf.keras.Model, dataset: Dict[str, Any], model_name: str, epochs: int = TRAINING_PARAMS['epochs'], patience: int = TRAINING_PARAMS['early_stopping_patience']) -> tf.keras.Model`: Train a deep learning model
- `train_traditional_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str = 'random_forest', model_params: Optional[Dict[str, Any]] = None) -> Any`: Train a traditional ML model
- `evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, data_info: Dict[str, Any], model_type: str = 'deep_learning') -> Dict[str, float]`: Evaluate a trained model
- `train_electricity_forecast_models(data: pd.DataFrame, target_col: str = 'energy(kWh/hh)', feature_cols: Optional[List[str]] = None, lookback: int = 24, forecast_horizon: int = 24) -> Dict[str, Any]`: Train multiple forecast models

### `src.models.evaluate`

#### Functions

- `load_trained_model(model_path: str) -> Any`: Load a trained model from file
- `load_data_info(data_info_path: str) -> Dict[str, Any]`: Load data processing info
- `prepare_test_data(test_data: pd.DataFrame, data_info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]`: Prepare test data
- `calculate_advanced_metrics(y_true: np.ndarray, y_pred: np.ndarray, data_info: Optional[Dict[str, Any]] = None) -> Dict[str, float]`: Calculate evaluation metrics
- `predict_and_evaluate(model: Any, X_test: np.ndarray, y_test: np.ndarray, data_info: Dict[str, Any], model_type: str = 'deep_learning') -> Tuple[np.ndarray, Dict[str, float]]`: Make predictions and evaluate
- `plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, data_info: Dict[str, Any], title: str = "Model Predictions", num_samples: int = 5, save_path: Optional[str] = None) -> None`: Plot predictions
- `plot_forecast_comparison(models_dict: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, data_info: Dict[str, Any], sample_idx: int = 0, save_path: Optional[str] = None) -> None`: Plot forecast comparison
- `plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], metric_names: List[str] = ['rmse', 'mae', 'r2'], title: str = "Model Metrics Comparison", save_path: Optional[str] = None) -> None`: Plot metrics comparison
- `evaluate_and_compare_models(models_dict: Dict[str, Tuple[Any, str]], test_data: pd.DataFrame, data_info: Dict[str, Any]) -> Dict[str, Any]`: Evaluate and compare multiple models
- `plot_feature_importance(model: Any, feature_names: List[str], model_type: str = 'traditional', top_n: int = 10, save_path: Optional[str] = None) -> None`: Plot feature importance

## Pipeline Module

### `src.pipelines.pipeline`

#### Classes

- `SmartMeterPipeline`: Main pipeline class for processing and analyzing Smart Meter data

#### Methods

- `__init__(self, config: Optional[Dict[str, Any]] = None)`: Initialize the pipeline
- `load_raw_data(self, dataset_names: List[str]) -> Dict[str, pd.DataFrame]`: Load raw datasets
- `load_processed_data(self, dataset_types: List[str]) -> Dict[str, pd.DataFrame]`: Load processed datasets
- `preprocess_household_data(self) -> pd.DataFrame`: Preprocess household data
- `preprocess_weather_data(self, hourly: bool = True) -> pd.DataFrame`: Preprocess weather data
- `preprocess_consumption_data(self, halfhourly: bool = True, sample_size: Optional[int] = None) -> pd.DataFrame`: Preprocess consumption data
- `merge_datasets(self, base_dataset: str, join_datasets: List[Dict[str, Any]]) -> pd.DataFrame`: Merge multiple datasets
- `create_analysis_dataset(self, output_file: str = "analysis_dataset.csv") -> pd.DataFrame`: Create analysis dataset

#### Functions

- `load_static_datasets() -> Dict[str, pd.DataFrame]`: Load and summarize static datasets
- `analyze_folder_datasets() -> None`: Analyze metrics for large folder-based datasets

## Analysis Module

### `src.analysis.visualization`

#### Functions

- `set_plot_style()`: Set consistent plot style
- `plot_time_series(data: pd.DataFrame, time_col: str, value_col: str, ...) -> plt.Figure`: Create a time series plot
- `plot_comparison(data: pd.DataFrame, time_col: str, value_cols: List[str], ...) -> plt.Figure`: Create a comparison plot
- `plot_distribution(data: pd.DataFrame, column: str, ...) -> plt.Figure`: Create a distribution plot
- `plot_boxplot(data: pd.DataFrame, x_col: str, y_col: str, ...) -> plt.Figure`: Create a box plot
- `plot_heatmap(data: pd.DataFrame, columns: Optional[List[str]] = None, ...) -> plt.Figure`: Create a heatmap
- `plot_scatter(data: pd.DataFrame, x_col: str, y_col: str, ...) -> plt.Figure`: Create a scatter plot
- `plot_bar(data: pd.DataFrame, x_col: str, y_col: str, ...) -> plt.Figure`: Create a bar plot
- `plot_hourly_profile(data: pd.DataFrame, time_col: str, value_col: str, ...) -> plt.Figure`: Create an hourly profile plot
- `plot_forecast_vs_actual(actual: np.ndarray, forecast: np.ndarray, ...) -> plt.Figure`: Compare forecast vs actual values
- `plot_model_comparison(models: Dict[str, Dict[str, float]], ...) -> plt.Figure`: Compare models' performance
- `plot_feature_importance(feature_names: List[str], importance_values: np.ndarray, ...) -> plt.Figure`: Plot feature importance

### `src.analysis.eda`

#### Functions

- `generate_data_profile(df: pd.DataFrame, ...) -> ProfileReport`: Generate a data profile report
- `analyze_missing_values(df: pd.DataFrame, ...) -> Dict[str, Any]`: Analyze missing values
- `analyze_numerical_distributions(df: pd.DataFrame, ...) -> Dict[str, Dict[str, float]]`: Analyze numerical distributions
- `analyze_categorical_distributions(df: pd.DataFrame, ...) -> Dict[str, Dict[str, Any]]`: Analyze categorical distributions
- `analyze_time_series(df: pd.DataFrame, ...) -> Dict[str, Any]`: Analyze time series data
- `analyze_correlations(df: pd.DataFrame, ...) -> Dict[str, Dict[str, float]]`: Analyze correlations
- `identify_skewed_columns(df: pd.DataFrame, ...) -> Dict[str, float]`: Identify skewed columns
- `generate_eda_report(df: pd.DataFrame, ...) -> Dict[str, Any]`: Generate a comprehensive EDA report

## Utils Module

### `src.utils.config`

#### Constants

- `ROOT_DIR`: Root directory of the project
- `DATA_DIR`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `EXTERNAL_DATA_DIR`: Data paths
- `MODEL_DIR`: Model directory
- `ARTIFACT_DIR`, `LOG_DIR`, `METRICS_DIR`: Artifact paths
- `FILES`: Dictionary of file paths
- `MODEL_PARAMS`: Model parameters
- `TRAINING_PARAMS`: Training parameters

### `src.utils.dataset_utils`

#### Functions

- `get_all_csv_files(directory: str) -> List[str]`: Get list of all CSV files in a directory
- `sample_large_csv(input_path: str, output_path: str, sample_size: int, random_state: Optional[int] = None) -> str`: Sample large CSV file
- `split_time_series(df: pd.DataFrame, time_col: str, test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, pd.DataFrame]`: Split time series data
- `create_time_features(df: pd.DataFrame, time_col: str, drop_original: bool = False) -> pd.DataFrame`: Create time features
- `aggregate_time_series(df: pd.DataFrame, time_col: str, value_col: str, freq: str = 'D', agg_func: str = 'sum') -> pd.DataFrame`: Aggregate time series
- `identify_missing_timestamps(df: pd.DataFrame, time_col: str, freq: str = 'H') -> pd.DataFrame`: Find missing timestamps
- `fill_missing_timestamps(df: pd.DataFrame, time_col: str, value_cols: List[str], freq: str = 'H', method: str = 'linear') -> pd.DataFrame`: Fill missing timestamps
- `add_lag_features(df: pd.DataFrame, time_col: str, value_col: str, lag_periods: List[int], groupby_col: Optional[str] = None) -> pd.DataFrame`: Add lag features
- `add_rolling_features(df: pd.DataFrame, time_col: str, value_col: str, windows: List[int], stats: List[str] = ['mean'], groupby_col: Optional[str] = None) -> pd.DataFrame`: Add rolling features
- `detect_and_remove_outliers(df: pd.DataFrame, value_col: str, method: str = 'iqr', threshold: float = 1.5, groupby_col: Optional[str] = None) -> pd.DataFrame`: Remove outliers
- `summarize_dataset(file_path, dataset_name)`: Summarize a dataset

### `src.utils.helpers`

#### Functions

- `save_summary_to_file(summary: Dict[str, Any], file_path: str) -> None`: Save a summary to a text file
- `save_dict_to_json(data: Dict[str, Any], file_path: str) -> None`: Save a dictionary to JSON
- `load_dict_from_json(file_path: str) -> Dict[str, Any]`: Load a dictionary from JSON
- `ensure_dir_exists(directory: str) -> None`: Ensure a directory exists
- `get_unique_values(df: pd.DataFrame, column: str) -> List[Any]`: Get unique values from a column
- `filter_dataframe(df: pd.DataFrame, filters: Dict[str, Any], allow_partial: bool = False) -> pd.DataFrame`: Filter a DataFrame 