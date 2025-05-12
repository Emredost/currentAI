"""
General helper functions for the Smart Meters in London project.
"""
import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Optional

def save_summary_to_file(summary: Dict[str, Any], file_path: str) -> None:
    """
    Save a summary dictionary to a text file.
    
    Args:
        summary: Dictionary containing summary information
        file_path: Path to save the summary file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, "w") as file:
            for key, value in summary.items():
                file.write(f"{key}: {value}\n")
        logging.info(f"Summary saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving summary to {file_path}: {e}")

def save_dict_to_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Dictionary saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving dictionary to {file_path}: {e}")

def load_dict_from_json(file_path: str) -> Dict[str, Any]:
    """
    Load a dictionary from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary loaded from the file
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Dictionary loaded from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading dictionary from {file_path}: {e}")
        return {}

def ensure_dir_exists(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    """
    try:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Ensured directory exists: {directory}")
    except Exception as e:
        logging.error(f"Error creating directory {directory}: {e}")
        
def get_unique_values(df: pd.DataFrame, column: str) -> List[Any]:
    """
    Get unique values from a DataFrame column.
    
    Args:
        df: Input DataFrame
        column: Column name
        
    Returns:
        List of unique values
    """
    if column not in df.columns:
        logging.warning(f"Column {column} not found in DataFrame")
        return []
    
    return sorted(df[column].unique().tolist())

def filter_dataframe(df: pd.DataFrame, 
                    filters: Dict[str, Any],
                    allow_partial: bool = False) -> pd.DataFrame:
    """
    Filter a DataFrame based on column values.
    
    Args:
        df: Input DataFrame
        filters: Dictionary of {column: value} pairs
        allow_partial: If True, apply only filters for columns that exist
        
    Returns:
        Filtered DataFrame
    """
    if not filters:
        return df
    
    result_df = df.copy()
    
    for col, value in filters.items():
        if col in result_df.columns:
            result_df = result_df[result_df[col] == value]
        elif not allow_partial:
            logging.warning(f"Column {col} not found in DataFrame")
            return pd.DataFrame()  # Return empty DataFrame if strict filtering
    
    return result_df
