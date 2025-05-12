import pandas as pd
import numpy as np
import logging
from typing import List, Union, Optional, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import os
from src.utils.config import FILES, PROCESSED_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)

def detect_outliers(df: pd.DataFrame, 
                  columns: List[str], 
                  method: str = 'iqr', 
                  threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in specified columns using different methods.
    
    Args:
        df: Input DataFrame
        columns: List of column names to check for outliers
        method: Method to use ('iqr', 'zscore', 'percentile')
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with a new boolean column for each input column indicating outliers
    """
    result_df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logging.warning(f"Column {col} not found in DataFrame")
            continue
            
        if not pd.api.types.is_numeric_dtype(df[col]):
            logging.warning(f"Column {col} is not numeric, skipping outlier detection")
            continue
            
        outlier_col = f"{col}_outlier"
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            result_df[outlier_col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            result_df[outlier_col] = abs((df[col] - mean) / std) > threshold
            
        elif method == 'percentile':
            lower = df[col].quantile(threshold / 100)
            upper = df[col].quantile(1 - threshold / 100)
            result_df[outlier_col] = (df[col] < lower) | (df[col] > upper)
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
            
        logging.info(f"Detected {result_df[outlier_col].sum()} outliers in column {col} using {method} method")
        
    return result_df

def handle_outliers(df: pd.DataFrame, 
                   columns: List[str], 
                   method: str = 'iqr', 
                   threshold: float = 1.5,
                   strategy: str = 'clip') -> pd.DataFrame:
    """
    Handle outliers in specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of column names to handle outliers
        method: Method to detect outliers ('iqr', 'zscore', 'percentile')
        threshold: Threshold for outlier detection
        strategy: How to handle outliers ('clip', 'remove', 'replace')
    
    Returns:
        DataFrame with outliers handled according to the specified strategy
    """
    result_df = df.copy()
    
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            outliers = abs((df[col] - mean) / std) > threshold
            
        elif method == 'percentile':
            lower = df[col].quantile(threshold / 100)
            upper = df[col].quantile(1 - threshold / 100)
            outliers = (df[col] < lower) | (df[col] > upper)
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
            
        outlier_count = outliers.sum()
        
        if strategy == 'clip' and outlier_count > 0:
            if method == 'iqr':
                result_df.loc[outliers, col] = result_df.loc[outliers, col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'percentile':
                result_df.loc[outliers, col] = result_df.loc[outliers, col].clip(lower=lower, upper=upper)
            elif method == 'zscore':
                result_df.loc[outliers, col] = result_df.loc[outliers, col].clip(
                    lower=mean - threshold * std, 
                    upper=mean + threshold * std
                )
                
        elif strategy == 'remove' and outlier_count > 0:
            result_df = result_df[~outliers]
            
        elif strategy == 'replace' and outlier_count > 0:
            # Replace with median
            median_value = df[col].median()
            result_df.loc[outliers, col] = median_value
            
        logging.info(f"Handled {outlier_count} outliers in column {col} using {strategy} strategy")
        
    return result_df

def clean_missing_values(df: pd.DataFrame, 
                        strategy: str = "mean", 
                        columns: Optional[List[str]] = None,
                        max_missing_pct: float = 50.0) -> pd.DataFrame:
    """
    Fill missing values in specified columns with the given strategy.
    
    Args:
        df: Input DataFrame
        strategy: Strategy to fill missing values ("mean", "median", "mode", "constant")
        columns: List of column names to clean, if None, all numeric columns are used
        max_missing_pct: Maximum percentage of missing values allowed in a column
                         Columns with higher percentage will be dropped
    
    Returns:
        DataFrame with missing values filled
    """
    result_df = df.copy()
    
    if columns is None:
        # Use only numeric columns if columns not specified
        columns = result_df.select_dtypes(include=np.number).columns.tolist()
    
    # Check percentage of missing values in each column
    for col in columns:
        if col not in result_df.columns:
            logging.warning(f"Column {col} not found in DataFrame")
            continue
            
        missing_pct = result_df[col].isna().mean() * 100
        
        if missing_pct > max_missing_pct:
            logging.warning(f"Column {col} has {missing_pct:.2f}% missing values, exceeding threshold of {max_missing_pct:.2f}%. Dropping column.")
            result_df = result_df.drop(columns=[col])
        elif missing_pct > 0:
            # Different strategies for different column types
            if pd.api.types.is_numeric_dtype(result_df[col]):
                if strategy == "mean":
                    fill_value = result_df[col].mean()
                elif strategy == "median":
                    fill_value = result_df[col].median()
                elif strategy == "mode":
                    fill_value = result_df[col].mode()[0]
                elif strategy == "constant":
                    fill_value = 0  # Default constant value
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                    
                result_df[col] = result_df[col].fillna(fill_value)
                logging.info(f"Filled {missing_pct:.2f}% missing values in column {col} with {strategy} ({fill_value})")
            else:
                # For non-numeric columns, use mode or empty string
                if strategy in ["mode", "most_frequent"]:
                    fill_value = result_df[col].mode()[0] if not result_df[col].mode().empty else ""
                else:
                    fill_value = ""
                    
                result_df[col] = result_df[col].fillna(fill_value)
                logging.info(f"Filled {missing_pct:.2f}% missing values in column {col} with {fill_value}")
    
    return result_df

def drop_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        subset: List of column names to consider when identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    original_shape = df.shape
    result_df = df.drop_duplicates(subset=subset)
    removed = original_shape[0] - result_df.shape[0]
    
    if removed > 0:
        logging.info(f"Removed {removed} duplicate rows ({removed/original_shape[0]*100:.2f}% of data)")
    
    return result_df

def format_datetime_column(df: pd.DataFrame, 
                          column: str, 
                          format: Optional[str] = None,
                          add_components: bool = False) -> pd.DataFrame:
    """
    Convert a column to datetime format and optionally extract date components.
    
    Args:
        df: Input DataFrame
        column: Column name to convert
        format: Optional format string for datetime parsing
        add_components: Whether to add columns with date components (year, month, day, etc.)
    
    Returns:
        DataFrame with datetime column converted and optionally new component columns
    """
    result_df = df.copy()
    
    if column not in result_df.columns:
        logging.warning(f"Column {column} not found in DataFrame")
        return result_df
    
    try:
        if format:
            result_df[column] = pd.to_datetime(result_df[column], format=format)
        else:
            result_df[column] = pd.to_datetime(result_df[column])
            
        logging.info(f"Converted column {column} to datetime format")
        
        if add_components:
            # Add year, month, day, hour, etc. as new columns
            result_df[f"{column}_year"] = result_df[column].dt.year
            result_df[f"{column}_month"] = result_df[column].dt.month
            result_df[f"{column}_day"] = result_df[column].dt.day
            result_df[f"{column}_dayofweek"] = result_df[column].dt.dayofweek
            result_df[f"{column}_quarter"] = result_df[column].dt.quarter
            
            # Add hour if time information is available
            if (result_df[column].dt.hour != 0).any():
                result_df[f"{column}_hour"] = result_df[column].dt.hour
                
            logging.info(f"Added date components for column {column}")
            
    except Exception as e:
        logging.error(f"Error converting column {column} to datetime: {e}")
    
    return result_df

def encode_categorical(df: pd.DataFrame, 
                      columns: List[str], 
                      method: str = 'onehot',
                      drop_original: bool = True) -> pd.DataFrame:
    """
    Encode categorical columns using different methods.
    
    Args:
        df: Input DataFrame
        columns: List of categorical column names
        method: Encoding method ('onehot', 'label', 'ordinal')
        drop_original: Whether to drop original columns
    
    Returns:
        DataFrame with encoded columns
    """
    result_df = df.copy()
    
    for col in columns:
        if col not in result_df.columns:
            logging.warning(f"Column {col} not found in DataFrame")
            continue
            
        if method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(result_df[col], prefix=col, drop_first=False)
            result_df = pd.concat([result_df, dummies], axis=1)
            logging.info(f"One-hot encoded column {col} into {dummies.shape[1]} columns")
            
        elif method == 'label':
            # Label encoding
            unique_values = result_df[col].dropna().unique()
            mapping = {value: idx for idx, value in enumerate(unique_values)}
            result_df[f"{col}_encoded"] = result_df[col].map(mapping)
            result_df[f"{col}_encoded"] = result_df[f"{col}_encoded"].fillna(-1).astype(int)
            logging.info(f"Label encoded column {col} with {len(mapping)} unique values")
            
        elif method == 'ordinal':
            # Ordinal encoding (assumes order is already correct)
            unique_values = sorted(result_df[col].dropna().unique())
            mapping = {value: idx for idx, value in enumerate(unique_values)}
            result_df[f"{col}_ordinal"] = result_df[col].map(mapping)
            result_df[f"{col}_ordinal"] = result_df[f"{col}_ordinal"].fillna(-1).astype(int)
            logging.info(f"Ordinal encoded column {col} with {len(mapping)} unique values")
            
        else:
            raise ValueError(f"Unknown encoding method: {method}")
            
        if drop_original:
            result_df = result_df.drop(columns=[col])
            
    return result_df

def normalize_data(df: pd.DataFrame, 
                  columns: List[str], 
                  method: str = 'standard',
                  return_scaler: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Normalize numeric columns using different methods.
    
    Args:
        df: Input DataFrame
        columns: List of numeric column names
        method: Normalization method ('standard', 'minmax')
        return_scaler: Whether to return the scaler objects
    
    Returns:
        DataFrame with normalized columns, and optionally a dict of scalers
    """
    result_df = df.copy()
    scalers = {}
    
    for col in columns:
        if col not in result_df.columns:
            logging.warning(f"Column {col} not found in DataFrame")
            continue
            
        if not pd.api.types.is_numeric_dtype(result_df[col]):
            logging.warning(f"Column {col} is not numeric, skipping normalization")
            continue
            
        # Handle missing values before normalization
        if result_df[col].isna().any():
            result_df[col] = result_df[col].fillna(result_df[col].mean())
            
        if method == 'standard':
            # Standardization (z-score normalization)
            scaler = StandardScaler()
            result_df[col] = scaler.fit_transform(result_df[[col]])
            logging.info(f"Standardized column {col}")
            
        elif method == 'minmax':
            # Min-max scaling
            scaler = MinMaxScaler()
            result_df[col] = scaler.fit_transform(result_df[[col]])
            logging.info(f"Min-max scaled column {col}")
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        scalers[col] = scaler
    
    if return_scaler:
        return result_df, scalers
    else:
        return result_df

def create_features(df: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create new features based on existing ones.
    
    Args:
        df: Input DataFrame
        feature_configs: List of feature configuration dictionaries
                        Each dict should have: 
                        - 'type': feature type (e.g., 'ratio', 'polynomial', 'rolling', 'lag')
                        - 'columns': columns to use
                        - Other type-specific parameters
    
    Returns:
        DataFrame with new features
    """
    result_df = df.copy()
    
    for config in feature_configs:
        feature_type = config.get('type')
        columns = config.get('columns', [])
        
        if not columns:
            logging.warning(f"No columns specified for feature type {feature_type}")
            continue
            
        if feature_type == 'ratio':
            # Create ratio between two columns
            if len(columns) != 2:
                logging.warning("Ratio feature requires exactly 2 columns")
                continue
                
            col1, col2 = columns
            if col1 in result_df.columns and col2 in result_df.columns:
                new_col = f"{col1}_to_{col2}_ratio"
                # Avoid division by zero
                result_df[new_col] = result_df[col1] / result_df[col2].replace(0, np.nan)
                result_df[new_col] = result_df[new_col].fillna(0)
                logging.info(f"Created ratio feature {new_col}")
                
        elif feature_type == 'polynomial':
            # Create polynomial features
            degree = config.get('degree', 2)
            for col in columns:
                if col in result_df.columns:
                    for i in range(2, degree + 1):
                        new_col = f"{col}_pow{i}"
                        result_df[new_col] = result_df[col] ** i
                        logging.info(f"Created polynomial feature {new_col}")
                        
        elif feature_type == 'rolling':
            # Create rolling window statistics
            window = config.get('window', 3)
            stats = config.get('stats', ['mean'])
            
            for col in columns:
                if col in result_df.columns:
                    for stat in stats:
                        new_col = f"{col}_rolling_{window}_{stat}"
                        
                        if stat == 'mean':
                            result_df[new_col] = result_df[col].rolling(window=window, min_periods=1).mean()
                        elif stat == 'std':
                            result_df[new_col] = result_df[col].rolling(window=window, min_periods=1).std()
                        elif stat == 'min':
                            result_df[new_col] = result_df[col].rolling(window=window, min_periods=1).min()
                        elif stat == 'max':
                            result_df[new_col] = result_df[col].rolling(window=window, min_periods=1).max()
                            
                        logging.info(f"Created rolling {stat} feature {new_col}")
                        
        elif feature_type == 'lag':
            # Create lag features
            lag_periods = config.get('periods', [1])
            
            for col in columns:
                if col in result_df.columns:
                    for period in lag_periods:
                        new_col = f"{col}_lag_{period}"
                        result_df[new_col] = result_df[col].shift(period)
                        result_df[new_col] = result_df[new_col].fillna(method='bfill')
                        logging.info(f"Created lag feature {new_col}")
                        
        else:
            logging.warning(f"Unknown feature type: {feature_type}")
            
    return result_df

def save_processed_data(df: pd.DataFrame, file_name: str) -> None:
    """
    Save processed DataFrame to a CSV file.
    
    Args:
        df: DataFrame to save
        file_name: Name of the file (without path)
    """
    file_path = os.path.join(PROCESSED_DATA_DIR, file_name)
    df.to_csv(file_path, index=False)
    logging.info(f"Saved processed data to {file_path}, shape: {df.shape}")

def load_and_process_data(data_loader_func, preprocessing_steps: List[Dict[str, Any]], output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load data and apply a sequence of preprocessing steps.
    
    Args:
        data_loader_func: Function to load raw data
        preprocessing_steps: List of preprocessing step dictionaries
                            Each dict should have:
                            - 'function': preprocessing function to apply
                            - 'params': parameters for the function
        output_file: Optional name to save the processed file
    
    Returns:
        Processed DataFrame
    """
    logging.info("Loading raw data...")
    df = data_loader_func()
    logging.info(f"Loaded data with shape: {df.shape}")
    
    for step in preprocessing_steps:
        function_name = step.get('function')
        params = step.get('params', {})
        
        logging.info(f"Applying preprocessing step: {function_name}")
        
        if function_name == 'clean_missing_values':
            df = clean_missing_values(df, **params)
        elif function_name == 'drop_duplicate_rows':
            df = drop_duplicate_rows(df, **params)
        elif function_name == 'format_datetime_column':
            df = format_datetime_column(df, **params)
        elif function_name == 'handle_outliers':
            df = handle_outliers(df, **params)
        elif function_name == 'encode_categorical':
            df = encode_categorical(df, **params)
        elif function_name == 'normalize_data':
            df = normalize_data(df, **params)
        elif function_name == 'create_features':
            df = create_features(df, **params)
        else:
            logging.warning(f"Unknown preprocessing function: {function_name}")
    
    if output_file:
        save_processed_data(df, output_file)
    
    return df
