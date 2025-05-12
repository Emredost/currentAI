import os
import pandas as pd
import logging
import glob
from src.utils.config import FILES, RAW_DATA_DIR
from typing import Optional, List, Dict, Union, Tuple


# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)

def validate_file(file_path: str) -> bool:
    """
    Ensure the file exists.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if file exists
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    return True

def load_with_retry(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file with retry logic and various encoding options.
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional parameters for pd.read_csv
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        Exception: If all loading attempts fail
    """
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            logging.warning(f"Failed to load with encoding {encoding}, trying another...")
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            raise
            
    # If all encodings fail
    raise UnicodeDecodeError(f"Could not decode file {file_path} with any of the attempted encodings")

def load_household_info() -> pd.DataFrame:
    """
    Load household information.
    
    Returns:
        pd.DataFrame: DataFrame containing household information.
    """
    path = FILES["household_info"]
    validate_file(path)
    logging.info("Loading household information.")
    return load_with_retry(path)

def load_acorn_details() -> pd.DataFrame:
    """
    Load ACORN details using the confirmed working configuration.
    
    Returns:
        pd.DataFrame: DataFrame containing ACORN details.
    """
    path = FILES["acorn_details"]
    validate_file(path)
    logging.info(f"Loading ACORN details from: {path}")
    try:
        data = load_with_retry(path, low_memory=False)
        logging.info(f"ACORN details loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Error loading ACORN details: {e}")
        raise

def load_uk_bank_holidays() -> pd.DataFrame:
    """
    Load UK bank holidays.
    
    Returns:
        pd.DataFrame: DataFrame containing UK bank holidays.
    """
    path = FILES["bank_holidays"]
    validate_file(path)
    logging.info("Loading UK bank holidays.")
    return load_with_retry(path)

def load_weather_data(daily: bool = True) -> pd.DataFrame:
    """
    Load weather data (daily or hourly).
    
    Args:
        daily: If True, loads daily weather data. Otherwise, loads hourly weather data.
        
    Returns:
        pd.DataFrame: DataFrame containing weather data.
    """
    file_key = "weather_daily" if daily else "weather_hourly"
    path = FILES[file_key]
    validate_file(path)
    logging.info(f"Loading {'daily' if daily else 'hourly'} weather data.")
    return load_with_retry(path)

def load_from_directory(directory: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load all CSV files from a directory and combine them.
    
    Args:
        directory: Directory containing CSV files
        limit: Optional limit on number of files to load
        
    Returns:
        pd.DataFrame: Combined DataFrame from all CSV files
    """
    if not os.path.exists(directory):
        logging.error(f"Directory not found: {directory}")
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    
    if not files:
        logging.error(f"No CSV files found in directory: {directory}")
        raise FileNotFoundError(f"No CSV files found in directory: {directory}")
    
    if limit is not None:
        files = files[:limit]
    
    logging.info(f"Loading {len(files)} files from {directory}")
    
    # Load files in batches to avoid memory issues
    chunk_size = 10
    dataframes = []
    
    for i in range(0, len(files), chunk_size):
        batch = files[i:i + chunk_size]
        batch_dfs = []
        
        for file in batch:
            try:
                df = load_with_retry(file)
                batch_dfs.append(df)
            except Exception as e:
                logging.error(f"Error loading file {file}: {e}")
                continue
                
        if batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)
            dataframes.append(batch_df)
            
    if not dataframes:
        logging.error("No files were successfully loaded")
        raise RuntimeError("No files were successfully loaded")
        
    return pd.concat(dataframes, ignore_index=True)

def load_daily_dataset(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load daily dataset files.
    
    Args:
        limit: Optional limit on number of files to load
        
    Returns:
        pd.DataFrame: Combined DataFrame from daily dataset files
    """
    return load_from_directory(FILES["daily_dataset_dir"], limit)

def load_halfhourly_dataset(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load half-hourly dataset files.
    
    Args:
        limit: Optional limit on number of files to load
        
    Returns:
        pd.DataFrame: Combined DataFrame from half-hourly dataset files
    """
    return load_from_directory(FILES["halfhourly_dataset_dir"], limit)

def load_processed_data(dataset_type: str) -> pd.DataFrame:
    """
    Load already processed data.
    
    Args:
        dataset_type: Type of dataset ('daily', 'hourly', 'weather', 'household')
        
    Returns:
        pd.DataFrame: Processed data
    """
    file_mapping = {
        'daily': FILES["processed_daily"],
        'hourly': FILES["processed_hourly"],
        'weather': FILES["processed_weather"],
        'household': FILES["household_processed"]
    }
    
    if dataset_type not in file_mapping:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available types: {list(file_mapping.keys())}")
    
    file_path = file_mapping[dataset_type]
    validate_file(file_path)
    
    logging.info(f"Loading processed {dataset_type} data")
    return load_with_retry(file_path)

if __name__ == "__main__":
    """
    Entry point for the script. Demonstrates loading some datasets
    and confirms the script execution with logging.
    """
    try:
        logging.info("Starting the data loader script...")

        # Example usage
        household_info = load_household_info()
        logging.info(f"Household info loaded successfully. Shape: {household_info.shape}")
        
        # Load processed data example
        try:
            daily_data = load_processed_data('daily')
            logging.info(f"Processed daily data loaded. Shape: {daily_data.shape}")
        except FileNotFoundError:
            logging.warning("Processed daily data not found. Run preprocessing first.")

        logging.info("Data Loader script executed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")