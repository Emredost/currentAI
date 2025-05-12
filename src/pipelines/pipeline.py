"""
Data processing pipeline module.
This module orchestrates the data processing and model training workflows.
"""
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.data.load_data import (
    load_household_info,
    load_acorn_details,
    load_uk_bank_holidays,
    load_weather_data,
    load_daily_dataset,
    load_halfhourly_dataset,
    load_processed_data
)

from src.data.preprocess import (
    clean_missing_values,
    drop_duplicate_rows,
    format_datetime_column,
    handle_outliers,
    encode_categorical,
    normalize_data,
    create_features,
    save_processed_data
)

from src.utils.config import (
    FILES,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    LOG_DIR,
    MODEL_PARAMS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log")),
        logging.StreamHandler()
    ]
)

class SmartMeterPipeline:
    """
    Pipeline for processing and analyzing Smart Meter data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.datasets = {}
        self.models = {}
        self.transformers = {}
        
        # Create timestamp for this pipeline run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.info(f"Initialized SmartMeterPipeline with timestamp {self.timestamp}")
    
    def load_raw_data(self, dataset_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load raw datasets.
        
        Args:
            dataset_names: List of dataset names to load
            
        Returns:
            Dictionary of dataset names to DataFrames
        """
        data_loaders = {
            'household_info': load_household_info,
            'acorn_details': load_acorn_details,
            'bank_holidays': load_uk_bank_holidays,
            'weather_daily': lambda: load_weather_data(daily=True),
            'weather_hourly': lambda: load_weather_data(daily=False),
            'daily_dataset': load_daily_dataset,
            'halfhourly_dataset': load_halfhourly_dataset
        }
        
        result = {}
        for name in dataset_names:
            if name not in data_loaders:
                logging.warning(f"Unknown dataset name: {name}")
                continue
                
            try:
                logging.info(f"Loading raw dataset: {name}")
                df = data_loaders[name]()
                result[name] = df
                self.datasets[f"raw_{name}"] = df
                logging.info(f"Loaded {name} with shape {df.shape}")
            except Exception as e:
                logging.error(f"Error loading {name}: {e}")
                
        return result
    
    def load_processed_data(self, dataset_types: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load already processed datasets.
        
        Args:
            dataset_types: List of dataset types to load
            
        Returns:
            Dictionary of dataset types to DataFrames
        """
        result = {}
        for dtype in dataset_types:
            try:
                df = load_processed_data(dtype)
                result[dtype] = df
                self.datasets[f"processed_{dtype}"] = df
                logging.info(f"Loaded processed {dtype} with shape {df.shape}")
            except Exception as e:
                logging.error(f"Error loading processed {dtype}: {e}")
                
        return result
    
    def preprocess_household_data(self) -> pd.DataFrame:
        """
        Preprocess household information data.
        
        Returns:
            Processed household DataFrame
        """
        if 'raw_household_info' not in self.datasets:
            logging.info("Loading household info data")
            self.datasets['raw_household_info'] = load_household_info()
        
        df = self.datasets['raw_household_info'].copy()
        logging.info(f"Preprocessing household data with initial shape {df.shape}")
        
        # Basic preprocessing steps
        df = clean_missing_values(df, strategy="mode")
        df = drop_duplicate_rows(df)
        
        # Convert ACORN categories to proper format
        if 'Acorn' in df.columns:
            df = encode_categorical(df, columns=['Acorn'], method='label')
            
        # Convert Acorn_grouped if present
        if 'Acorn_grouped' in df.columns:
            df = encode_categorical(df, columns=['Acorn_grouped'], method='label')
        
        # Save processed data
        save_processed_data(df, "household_info_processed.csv")
        self.datasets['processed_household'] = df
        
        logging.info(f"Completed household data preprocessing, final shape {df.shape}")
        return df
    
    def preprocess_weather_data(self, hourly: bool = True) -> pd.DataFrame:
        """
        Preprocess weather data.
        
        Args:
            hourly: Whether to process hourly (True) or daily (False) weather data
            
        Returns:
            Processed weather DataFrame
        """
        weather_key = 'raw_weather_hourly' if hourly else 'raw_weather_daily'
        if weather_key not in self.datasets:
            logging.info(f"Loading {'hourly' if hourly else 'daily'} weather data")
            self.datasets[weather_key] = load_weather_data(daily=not hourly)
        
        df = self.datasets[weather_key].copy()
        logging.info(f"Preprocessing {'hourly' if hourly else 'daily'} weather data with initial shape {df.shape}")
        
        # Format datetime
        df = format_datetime_column(df, column='time', add_components=True)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        df = clean_missing_values(df, strategy="mean", columns=numeric_cols)
        
        # Handle outliers
        df = handle_outliers(df, columns=numeric_cols, method='iqr', strategy='clip')
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != 'time']  # Exclude time column
        if categorical_cols:
            df = encode_categorical(df, columns=categorical_cols, method='onehot')
        
        # Create features
        feature_configs = [
            {
                'type': 'ratio',
                'columns': ['temperature', 'dewPoint']
            },
            {
                'type': 'polynomial',
                'columns': ['temperature', 'humidity', 'windSpeed'],
                'degree': 2
            }
        ]
        
        df = create_features(df, feature_configs)
        
        # Save processed data
        file_name = "cleaned_weather_hourly.csv" if hourly else "cleaned_weather_daily.csv"
        save_processed_data(df, file_name)
        
        processed_key = 'processed_weather_hourly' if hourly else 'processed_weather_daily'
        self.datasets[processed_key] = df
        
        logging.info(f"Completed {'hourly' if hourly else 'daily'} weather data preprocessing, final shape {df.shape}")
        return df
    
    def preprocess_consumption_data(self, halfhourly: bool = True, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Preprocess electricity consumption data.
        
        Args:
            halfhourly: Whether to process half-hourly (True) or daily (False) consumption data
            sample_size: Optional sample size to limit data for testing
            
        Returns:
            Processed consumption DataFrame
        """
        data_key = 'raw_halfhourly_dataset' if halfhourly else 'raw_daily_dataset'
        
        if data_key not in self.datasets:
            logging.info(f"Loading {'half-hourly' if halfhourly else 'daily'} consumption data")
            if halfhourly:
                self.datasets[data_key] = load_halfhourly_dataset()
            else:
                self.datasets[data_key] = load_daily_dataset()
        
        df = self.datasets[data_key].copy()
        
        if sample_size is not None and sample_size < len(df):
            df = df.sample(sample_size, random_state=MODEL_PARAMS['random_state'])
            logging.info(f"Sampled {sample_size} rows from consumption data")
        
        logging.info(f"Preprocessing {'half-hourly' if halfhourly else 'daily'} consumption data with initial shape {df.shape}")
        
        # Handle timestamp column
        timestamp_col = 'tstp'
        if timestamp_col in df.columns:
            df = format_datetime_column(df, column=timestamp_col, add_components=True)
        
        # Handle energy column
        energy_col = 'energy(kWh/hh)' if halfhourly else 'energy_consumption'
        if energy_col in df.columns:
            # Handle missing and negative values
            df.loc[df[energy_col] < 0, energy_col] = np.nan
            df = clean_missing_values(df, strategy="mean", columns=[energy_col])
            
            # Handle outliers
            df = handle_outliers(df, columns=[energy_col], method='iqr', strategy='clip')
        
        # Drop duplicates
        df = drop_duplicate_rows(df)
        
        # Save processed data
        file_name = "cleaned_halfhourly_data.csv" if halfhourly else "cleaned_daily_dataset.csv"
        save_processed_data(df, file_name)
        
        processed_key = 'processed_halfhourly' if halfhourly else 'processed_daily'
        self.datasets[processed_key] = df
        
        logging.info(f"Completed {'half-hourly' if halfhourly else 'daily'} consumption data preprocessing, final shape {df.shape}")
        return df
    
    def merge_datasets(self, base_dataset: str, join_datasets: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Merge multiple datasets together.
        
        Args:
            base_dataset: Name of the base dataset in self.datasets
            join_datasets: List of dictionaries with parameters for joining:
                           {'dataset': 'dataset_name', 'on': 'join_column', 'how': 'left'}
        
        Returns:
            Merged DataFrame
        """
        if base_dataset not in self.datasets:
            raise ValueError(f"Base dataset {base_dataset} not found in available datasets")
        
        result = self.datasets[base_dataset].copy()
        logging.info(f"Starting dataset merge with base {base_dataset}, shape {result.shape}")
        
        for join_info in join_datasets:
            dataset_name = join_info.get('dataset')
            join_column = join_info.get('on')
            join_type = join_info.get('how', 'left')
            
            if dataset_name not in self.datasets:
                logging.warning(f"Dataset {dataset_name} not found for merging")
                continue
                
            if not join_column:
                logging.warning(f"No join column specified for dataset {dataset_name}")
                continue
            
            right_df = self.datasets[dataset_name]
            
            # Check if join column exists in both dataframes
            if join_column not in result.columns:
                logging.warning(f"Join column {join_column} not found in base dataset")
                continue
                
            if join_column not in right_df.columns:
                logging.warning(f"Join column {join_column} not found in dataset {dataset_name}")
                continue
            
            # Perform the merge
            original_shape = result.shape
            result = result.merge(right_df, on=join_column, how=join_type)
            
            logging.info(f"Merged {dataset_name} on {join_column}, shape changed from {original_shape} to {result.shape}")
        
        # Save the merged dataset
        merged_key = f"merged_{self.timestamp}"
        self.datasets[merged_key] = result
        
        logging.info(f"Completed dataset merging, final shape {result.shape}")
        return result
    
    def create_analysis_dataset(self, output_file: str = "analysis_dataset.csv") -> pd.DataFrame:
        """
        Create a dataset for analysis by preprocessing and merging all relevant data.
        
        Args:
            output_file: Name of the output file
            
        Returns:
            Analysis-ready DataFrame
        """
        # Load or preprocess required datasets
        try:
            # Try to load from processed files first
            logging.info("Trying to load processed datasets...")
            self.load_processed_data(['hourly', 'weather', 'household'])
        except Exception as e:
            logging.warning(f"Could not load all processed datasets: {e}")
            logging.info("Will preprocess raw data instead")
            
            # Load and preprocess raw data
            self.load_raw_data(['household_info', 'weather_hourly', 'halfhourly_dataset'])
            
            if 'processed_household' not in self.datasets:
                self.preprocess_household_data()
                
            if 'processed_weather_hourly' not in self.datasets:
                self.preprocess_weather_data(hourly=True)
                
            if 'processed_halfhourly' not in self.datasets:
                self.preprocess_consumption_data(halfhourly=True)
        
        # Merge datasets
        if all(k in self.datasets for k in ['processed_halfhourly', 'processed_weather_hourly', 'processed_household']):
            # Create a datetime join key if needed
            consumption_df = self.datasets['processed_halfhourly']
            weather_df = self.datasets['processed_weather_hourly']
            
            # Round consumption timestamp to nearest hour to match weather data
            if 'tstp' in consumption_df.columns:
                consumption_df['join_hour'] = pd.to_datetime(consumption_df['tstp']).dt.floor('H')
                
            if 'time' in weather_df.columns:
                weather_df['join_hour'] = pd.to_datetime(weather_df['time']).dt.floor('H')
            
            # Merge weather with consumption
            join_datasets = [
                {
                    'dataset': 'processed_weather_hourly',
                    'on': 'join_hour',
                    'how': 'left'
                },
                {
                    'dataset': 'processed_household',
                    'on': 'LCLid',
                    'how': 'left'
                }
            ]
            
            analysis_df = self.merge_datasets('processed_halfhourly', join_datasets)
            
            # Save the analysis dataset
            save_processed_data(analysis_df, output_file)
            self.datasets['analysis'] = analysis_df
            
            logging.info(f"Created analysis dataset with shape {analysis_df.shape}")
            return analysis_df
        else:
            logging.error("Could not create analysis dataset, missing required processed datasets")
            raise ValueError("Missing required datasets")

def load_static_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load and summarize static datasets (household info, acorn details, etc.)
    
    Returns:
        Dictionary of dataset names to DataFrames
    """
    datasets = {}
    
    try:
        # Load household info
        household_info = load_household_info()
        datasets['household_info'] = household_info
        print(f"Household Info Shape: {household_info.shape}")
        print("Sample data:")
        print(household_info.head(3))
        print("\n")
        
        # Load ACORN details
        acorn_details = load_acorn_details()
        datasets['acorn_details'] = acorn_details
        print(f"ACORN Details Shape: {acorn_details.shape}")
        print("Sample data:")
        print(acorn_details.head(3))
        print("\n")
        
        # Load bank holidays
        bank_holidays = load_uk_bank_holidays()
        datasets['bank_holidays'] = bank_holidays
        print(f"Bank Holidays Shape: {bank_holidays.shape}")
        print("Sample data:")
        print(bank_holidays.head(3))
        print("\n")
        
        # Load weather data
        weather_daily = load_weather_data(daily=True)
        datasets['weather_daily'] = weather_daily
        print(f"Daily Weather Shape: {weather_daily.shape}")
        print("Sample data:")
        print(weather_daily.head(3))
        print("\n")
        
    except Exception as e:
        print(f"Error loading static datasets: {e}")
    
    return datasets

def analyze_folder_datasets() -> None:
    """
    Analyze metrics for large folder-based datasets without loading them entirely.
    """
    try:
        # Analyze daily dataset structure
        daily_dir = FILES["daily_dataset_dir"]
        if os.path.exists(daily_dir):
            files = [f for f in os.listdir(daily_dir) if f.endswith('.csv')]
            print(f"Daily Dataset: {len(files)} files found")
            
            # Load a small sample to understand the structure
            sample_file = os.path.join(daily_dir, files[0])
            sample_data = pd.read_csv(sample_file, nrows=5)
            print("Sample daily dataset structure:")
            print(sample_data.columns.tolist())
            print(sample_data.head(2))
            print("\n")
        
        # Analyze half-hourly dataset structure
        halfhourly_dir = FILES["halfhourly_dataset_dir"]
        if os.path.exists(halfhourly_dir):
            files = [f for f in os.listdir(halfhourly_dir) if f.endswith('.csv')]
            print(f"Half-hourly Dataset: {len(files)} files found")
            
            # Load a small sample to understand the structure
            sample_file = os.path.join(halfhourly_dir, files[0])
            sample_data = pd.read_csv(sample_file, nrows=5)
            print("Sample half-hourly dataset structure:")
            print(sample_data.columns.tolist())
            print(sample_data.head(2))
            print("\n")
    
    except Exception as e:
        print(f"Error analyzing folder datasets: {e}")

if __name__ == "__main__":
    # Create and run the pipeline
    pipeline = SmartMeterPipeline()
    
    # Example: Create a small analysis dataset for testing
    try:
        analysis_df = pipeline.create_analysis_dataset("test_analysis_dataset.csv")
        print(f"Created analysis dataset with shape {analysis_df.shape}")
    except Exception as e:
        logging.error(f"Error creating analysis dataset: {e}")
