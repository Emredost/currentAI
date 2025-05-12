"""
Utility functions for dataset operations.
"""
import os
import glob
from typing import List, Dict, Tuple, Union, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

def get_all_csv_files(directory: str) -> List[str]:
    """
    Get list of all CSV files in a directory.
    
    Args:
        directory: Path to directory
        
    Returns:
        List of CSV file paths
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    return sorted(glob.glob(os.path.join(directory, "*.csv")))

def sample_large_csv(input_path: str, 
                    output_path: str, 
                    sample_size: int,
                    random_state: Optional[int] = None) -> str:
    """
    Create a random sample from a large CSV file without loading it entirely.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save sampled CSV file
        sample_size: Number of rows to sample
        random_state: Random seed for reproducibility
        
    Returns:
        Path to sampled file
    """
    # Count lines in the file
    with open(input_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    # Calculate sampling probability (accounting for header)
    p = sample_size / (total_lines - 1)
    
    # Set random seed if specified
    if random_state is not None:
        np.random.seed(random_state)
    
    # Sample lines
    header = None
    sampled_indices = []
    
    with open(input_path, 'r') as f:
        # Read header
        header = f.readline().strip()
        
        # Process remaining lines
        for i, line in enumerate(f):
            # Always include line if we haven't reached sample_size
            if len(sampled_indices) < sample_size:
                sampled_indices.append((i + 1, line.strip()))  # +1 to account for header
            else:
                # Reservoir sampling
                j = np.random.randint(0, i + 1)
                if j < sample_size:
                    sampled_indices[j] = (i + 1, line.strip())
    
    # Sort by original index
    sampled_indices.sort(key=lambda x: x[0])
    
    # Write sampled data to output file
    with open(output_path, 'w') as f:
        # Write header
        f.write(f"{header}\n")
        
        # Write sampled lines
        for _, line in sampled_indices:
            f.write(f"{line}\n")
    
    return output_path

def split_time_series(df: pd.DataFrame, 
                     time_col: str,
                     test_size: float = 0.2,
                     val_size: float = 0.1) -> Dict[str, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets.
    
    Args:
        df: DataFrame with time series data
        time_col: Column name for the time/date
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
        
    Returns:
        Dictionary with train, val, and test DataFrames
    """
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort by time
    df = df.sort_values(time_col)
    
    # Calculate split points
    n = len(df)
    test_start_idx = int(n * (1 - test_size))
    val_start_idx = int(test_start_idx * (1 - val_size / (1 - test_size)))
    
    # Split data
    train_df = df.iloc[:val_start_idx].copy()
    val_df = df.iloc[val_start_idx:test_start_idx].copy()
    test_df = df.iloc[test_start_idx:].copy()
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

def create_time_features(df: pd.DataFrame, 
                        time_col: str,
                        drop_original: bool = False) -> pd.DataFrame:
    """
    Create time-based features from a datetime column.
    
    Args:
        df: DataFrame with time series data
        time_col: Column name for the time/date
        drop_original: Whether to drop the original time column
        
    Returns:
        DataFrame with added time features
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df[time_col]):
        result_df[time_col] = pd.to_datetime(result_df[time_col])
    
    # Extract time components
    result_df[f'{time_col}_year'] = result_df[time_col].dt.year
    result_df[f'{time_col}_month'] = result_df[time_col].dt.month
    result_df[f'{time_col}_day'] = result_df[time_col].dt.day
    result_df[f'{time_col}_hour'] = result_df[time_col].dt.hour
    result_df[f'{time_col}_dayofweek'] = result_df[time_col].dt.dayofweek
    result_df[f'{time_col}_quarter'] = result_df[time_col].dt.quarter
    result_df[f'{time_col}_dayofyear'] = result_df[time_col].dt.dayofyear
    result_df[f'{time_col}_weekofyear'] = result_df[time_col].dt.isocalendar().week
    
    # Add cyclical features for month, day of week, hour
    result_df[f'{time_col}_month_sin'] = np.sin(2 * np.pi * result_df[f'{time_col}_month'] / 12)
    result_df[f'{time_col}_month_cos'] = np.cos(2 * np.pi * result_df[f'{time_col}_month'] / 12)
    
    result_df[f'{time_col}_dayofweek_sin'] = np.sin(2 * np.pi * result_df[f'{time_col}_dayofweek'] / 7)
    result_df[f'{time_col}_dayofweek_cos'] = np.cos(2 * np.pi * result_df[f'{time_col}_dayofweek'] / 7)
    
    result_df[f'{time_col}_hour_sin'] = np.sin(2 * np.pi * result_df[f'{time_col}_hour'] / 24)
    result_df[f'{time_col}_hour_cos'] = np.cos(2 * np.pi * result_df[f'{time_col}_hour'] / 24)
    
    # Create is_weekend flag
    result_df[f'{time_col}_is_weekend'] = result_df[f'{time_col}_dayofweek'].isin([5, 6]).astype(int)
    
    # Add business day feature
    bday = pd.tseries.offsets.BusinessDay()
    result_df[f'{time_col}_is_businessday'] = result_df[time_col].map(
        lambda x: 1 if x == bday.rollforward(x) else 0
    )
    
    if drop_original:
        result_df = result_df.drop(columns=[time_col])
    
    return result_df

def aggregate_time_series(df: pd.DataFrame,
                         time_col: str,
                         value_col: str,
                         freq: str = 'D',
                         agg_func: str = 'sum') -> pd.DataFrame:
    """
    Aggregate time series data to a specified frequency.
    
    Args:
        df: DataFrame with time series data
        time_col: Column name for the time/date
        value_col: Column name for the values
        freq: Frequency for aggregation ('D' for daily, 'H' for hourly, 'M' for monthly)
        agg_func: Aggregation function ('sum', 'mean', 'max', 'min', 'std')
        
    Returns:
        Aggregated DataFrame
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df[time_col]):
        result_df[time_col] = pd.to_datetime(result_df[time_col])
    
    # Set time column as index
    result_df = result_df.set_index(time_col)
    
    # Resample and aggregate
    if agg_func == 'sum':
        result_df = result_df.resample(freq)[value_col].sum().reset_index()
    elif agg_func == 'mean':
        result_df = result_df.resample(freq)[value_col].mean().reset_index()
    elif agg_func == 'max':
        result_df = result_df.resample(freq)[value_col].max().reset_index()
    elif agg_func == 'min':
        result_df = result_df.resample(freq)[value_col].min().reset_index()
    elif agg_func == 'std':
        result_df = result_df.resample(freq)[value_col].std().reset_index()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")
    
    return result_df

def identify_missing_timestamps(df: pd.DataFrame,
                              time_col: str,
                              freq: str = 'H') -> pd.DataFrame:
    """
    Identify missing timestamps in a time series.
    
    Args:
        df: DataFrame with time series data
        time_col: Column name for the time/date
        freq: Expected frequency of the time series
        
    Returns:
        DataFrame with the missing timestamps
    """
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort by time
    df = df.sort_values(time_col)
    
    # Create a complete range of timestamps
    start_time = df[time_col].min()
    end_time = df[time_col].max()
    complete_range = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # Find missing timestamps
    existing_times = set(df[time_col])
    missing_times = [t for t in complete_range if t not in existing_times]
    
    if missing_times:
        return pd.DataFrame({time_col: missing_times})
    else:
        return pd.DataFrame({time_col: []})

def fill_missing_timestamps(df: pd.DataFrame,
                          time_col: str,
                          value_cols: List[str],
                          freq: str = 'H',
                          method: str = 'linear') -> pd.DataFrame:
    """
    Fill missing timestamps in a time series.
    
    Args:
        df: DataFrame with time series data
        time_col: Column name for the time/date
        value_cols: List of column names for values to fill
        freq: Expected frequency of the time series
        method: Interpolation method ('linear', 'ffill', 'bfill', 'zero')
        
    Returns:
        DataFrame with filled missing timestamps
    """
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Set time column as index
    result_df = df.copy().set_index(time_col)
    
    # Create a complete range of timestamps
    start_time = df[time_col].min()
    end_time = df[time_col].max()
    
    # Reindex with complete range
    result_df = result_df.reindex(pd.date_range(start=start_time, end=end_time, freq=freq))
    
    # Fill missing values
    for col in value_cols:
        if method == 'linear':
            result_df[col] = result_df[col].interpolate(method='linear')
        elif method == 'ffill':
            result_df[col] = result_df[col].fillna(method='ffill')
        elif method == 'bfill':
            result_df[col] = result_df[col].fillna(method='bfill')
        elif method == 'zero':
            result_df[col] = result_df[col].fillna(0)
        else:
            raise ValueError(f"Unknown fill method: {method}")
    
    # Reset index
    result_df = result_df.reset_index()
    result_df = result_df.rename(columns={'index': time_col})
    
    return result_df

def add_lag_features(df: pd.DataFrame,
                   time_col: str,
                   value_col: str,
                   lag_periods: List[int],
                   groupby_col: Optional[str] = None) -> pd.DataFrame:
    """
    Add lag features to a time series.
    
    Args:
        df: DataFrame with time series data
        time_col: Column name for the time/date
        value_col: Column name for the values
        lag_periods: List of lag periods to create
        groupby_col: Optional column to group by (for multiple time series)
        
    Returns:
        DataFrame with added lag features
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df[time_col]):
        result_df[time_col] = pd.to_datetime(result_df[time_col])
    
    # Sort by time
    if groupby_col:
        result_df = result_df.sort_values([groupby_col, time_col])
    else:
        result_df = result_df.sort_values(time_col)
    
    # Create lag features
    for lag in lag_periods:
        lag_col = f"{value_col}_lag_{lag}"
        
        if groupby_col:
            # Create lag features within each group
            result_df[lag_col] = result_df.groupby(groupby_col)[value_col].shift(lag)
        else:
            # Create lag features for the whole dataset
            result_df[lag_col] = result_df[value_col].shift(lag)
    
    return result_df

def add_rolling_features(df: pd.DataFrame,
                       time_col: str,
                       value_col: str,
                       windows: List[int],
                       stats: List[str] = ['mean'],
                       groupby_col: Optional[str] = None) -> pd.DataFrame:
    """
    Add rolling window features to a time series.
    
    Args:
        df: DataFrame with time series data
        time_col: Column name for the time/date
        value_col: Column name for the values
        windows: List of window sizes
        stats: List of statistics to compute ('mean', 'std', 'min', 'max', 'median')
        groupby_col: Optional column to group by (for multiple time series)
        
    Returns:
        DataFrame with added rolling features
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df[time_col]):
        result_df[time_col] = pd.to_datetime(result_df[time_col])
    
    # Sort by time
    if groupby_col:
        result_df = result_df.sort_values([groupby_col, time_col])
    else:
        result_df = result_df.sort_values(time_col)
    
    # Create rolling features
    for window in windows:
        for stat in stats:
            stat_col = f"{value_col}_rolling_{window}_{stat}"
            
            if groupby_col:
                # Apply rolling window within each group
                grouped = result_df.groupby(groupby_col)
                
                if stat == 'mean':
                    result_df[stat_col] = grouped[value_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                elif stat == 'std':
                    result_df[stat_col] = grouped[value_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                elif stat == 'min':
                    result_df[stat_col] = grouped[value_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min()
                    )
                elif stat == 'max':
                    result_df[stat_col] = grouped[value_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
                elif stat == 'median':
                    result_df[stat_col] = grouped[value_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).median()
                    )
                else:
                    raise ValueError(f"Unknown statistic: {stat}")
            else:
                # Apply rolling window to the whole dataset
                if stat == 'mean':
                    result_df[stat_col] = result_df[value_col].rolling(window=window, min_periods=1).mean()
                elif stat == 'std':
                    result_df[stat_col] = result_df[value_col].rolling(window=window, min_periods=1).std()
                elif stat == 'min':
                    result_df[stat_col] = result_df[value_col].rolling(window=window, min_periods=1).min()
                elif stat == 'max':
                    result_df[stat_col] = result_df[value_col].rolling(window=window, min_periods=1).max()
                elif stat == 'median':
                    result_df[stat_col] = result_df[value_col].rolling(window=window, min_periods=1).median()
                else:
                    raise ValueError(f"Unknown statistic: {stat}")
    
    return result_df

def detect_and_remove_outliers(df: pd.DataFrame,
                             value_col: str,
                             method: str = 'iqr',
                             threshold: float = 1.5,
                             groupby_col: Optional[str] = None) -> pd.DataFrame:
    """
    Detect and remove outliers from a time series.
    
    Args:
        df: DataFrame with time series data
        value_col: Column name for the values
        method: Method for outlier detection ('iqr', 'zscore', 'std')
        threshold: Threshold for outlier detection
        groupby_col: Optional column to group by
        
    Returns:
        DataFrame with outliers removed
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    if groupby_col:
        # Apply outlier detection within each group
        clean_dfs = []
        
        for group, group_df in result_df.groupby(groupby_col):
            if method == 'iqr':
                # IQR method
                Q1 = group_df[value_col].quantile(0.25)
                Q3 = group_df[value_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Filter outliers
                clean_group = group_df[(group_df[value_col] >= lower_bound) & 
                                    (group_df[value_col] <= upper_bound)]
                
            elif method == 'zscore':
                # Z-score method
                mean = group_df[value_col].mean()
                std = group_df[value_col].std()
                
                # Filter outliers
                clean_group = group_df[abs((group_df[value_col] - mean) / std) <= threshold]
                
            elif method == 'std':
                # Standard deviation method
                mean = group_df[value_col].mean()
                std = group_df[value_col].std()
                
                # Filter outliers
                clean_group = group_df[(group_df[value_col] >= mean - threshold * std) & 
                                    (group_df[value_col] <= mean + threshold * std)]
                
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            clean_dfs.append(clean_group)
        
        # Combine the cleaned groups
        result_df = pd.concat(clean_dfs)
        
    else:
        # Apply outlier detection to the whole dataset
        if method == 'iqr':
            # IQR method
            Q1 = result_df[value_col].quantile(0.25)
            Q3 = result_df[value_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Filter outliers
            result_df = result_df[(result_df[value_col] >= lower_bound) & 
                                (result_df[value_col] <= upper_bound)]
            
        elif method == 'zscore':
            # Z-score method
            mean = result_df[value_col].mean()
            std = result_df[value_col].std()
            
            # Filter outliers
            result_df = result_df[abs((result_df[value_col] - mean) / std) <= threshold]
            
        elif method == 'std':
            # Standard deviation method
            mean = result_df[value_col].mean()
            std = result_df[value_col].std()
            
            # Filter outliers
            result_df = result_df[(result_df[value_col] >= mean - threshold * std) & 
                                (result_df[value_col] <= mean + threshold * std)]
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    return result_df

def summarize_dataset(file_path, dataset_name):
    """
    Summarizes a dataset and returns a dictionary of statistics.
    """
    try:
        data = pd.read_csv(file_path, encoding='latin1')
        stats = {
            "shape": data.shape,
            "columns": list(data.columns),
            "missing_values": data.isnull().sum().to_dict(),
            "duplicates": data.duplicated().sum(),
            "summary": data.describe(include="all").to_dict()
        }
        print(f"--- {dataset_name.upper()} ---")
        print(stats)
        return data, stats
    except Exception as e:
        print(f"Error reading {dataset_name}: {e}")
        return None, None

