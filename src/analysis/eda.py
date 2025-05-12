"""
Exploratory Data Analysis module for automated data profiling and reporting.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional, Any
import os
from datetime import datetime
import logging
from ydata_profiling import ProfileReport
import json

from src.analysis.visualization import (
    plot_time_series,
    plot_distribution,
    plot_heatmap,
    plot_hourly_profile,
    set_plot_style
)

def generate_data_profile(df: pd.DataFrame,
                         title: str = "Data Profile Report",
                         output_path: Optional[str] = None,
                         minimal: bool = False) -> ProfileReport:
    """
    Generate a comprehensive data profile report using ydata-profiling.
    
    Args:
        df: DataFrame to profile
        title: Report title
        output_path: Optional path to save the HTML report
        minimal: Whether to generate a minimal report (faster)
        
    Returns:
        ProfileReport object
    """
    # Set configuration based on minimal flag
    if minimal:
        report = ProfileReport(
            df, 
            title=title,
            minimal=True,
            progress_bar=True,
            correlations=None
        )
    else:
        report = ProfileReport(
            df, 
            title=title,
            html={'style':{'full_width':True}},
            progress_bar=True,
            correlations={
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "kendall": {"calculate": False},
                "phi_k": {"calculate": False},
                "cramers": {"calculate": False},
            }
        )
    
    # Save the report if output path is specified
    if output_path:
        report.to_file(output_path)
        logging.info(f"Data profile report saved to {output_path}")
    
    return report

def analyze_missing_values(df: pd.DataFrame,
                         threshold: float = 0.1,
                         output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze missing values in the DataFrame.
    
    Args:
        df: DataFrame to analyze
        threshold: Threshold for highlighting columns with high missing percentage
        output_path: Optional path to save the plot
        
    Returns:
        Dictionary with missing value statistics
    """
    # Calculate missing value counts and percentages
    missing_count = df.isnull().sum()
    missing_percent = df.isnull().mean().round(4) * 100
    
    # Create a DataFrame with the results
    missing_df = pd.DataFrame({
        'Column': missing_count.index,
        'Missing Count': missing_count.values,
        'Missing Percent': missing_percent.values
    })
    
    # Sort by missing percentage in descending order
    missing_df = missing_df.sort_values('Missing Percent', ascending=False)
    
    # Columns with missing values
    columns_with_missing = missing_df[missing_df['Missing Count'] > 0]
    
    # Columns with high missing percentage
    high_missing = missing_df[missing_df['Missing Percent'] > threshold * 100]
    
    # Create a plot if there are columns with missing values
    if len(columns_with_missing) > 0 and output_path:
        # Select top 20 columns with highest missing percentage
        plot_df = columns_with_missing.head(20)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(plot_df['Column'], plot_df['Missing Percent'], color='skyblue')
        
        # Add a line for the threshold
        if threshold > 0:
            plt.axvline(x=threshold * 100, color='red', linestyle='--', 
                      label=f"Threshold ({threshold * 100}%)")
            plt.legend()
        
        # Add values on the bars
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 1, 
                bar.get_y() + bar.get_height()/2, 
                f"{width:.1f}%", 
                ha='left', 
                va='center'
            )
        
        plt.xlabel('Missing Percentage')
        plt.ylabel('Column')
        plt.title('Missing Value Analysis')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    # Prepare the result dictionary
    result = {
        'total_columns': len(df.columns),
        'columns_with_missing': len(columns_with_missing),
        'columns_with_high_missing': len(high_missing),
        'missing_stats': missing_df.to_dict(orient='records'),
        'high_missing_columns': high_missing['Column'].tolist()
    }
    
    return result

def analyze_numerical_distributions(df: pd.DataFrame,
                                  columns: Optional[List[str]] = None,
                                  output_dir: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Analyze distributions of numerical columns.
    
    Args:
        df: DataFrame to analyze
        columns: List of numerical columns to analyze (if None, all numeric columns)
        output_dir: Optional directory to save the plots
        
    Returns:
        Dictionary with distribution statistics for each column
    """
    # Select numerical columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize result dictionary
    result = {}
    
    # Set plot style
    set_plot_style()
    
    # Analyze each column
    for col in columns:
        if col not in df.columns:
            continue
            
        # Calculate statistics
        stats = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'range': df[col].max() - df[col].min(),
            'skew': df[col].skew(),
            'kurtosis': df[col].kurtosis(),
            'q1': df[col].quantile(0.25),
            'q3': df[col].quantile(0.75),
            'iqr': df[col].quantile(0.75) - df[col].quantile(0.25),
            'zeros_count': (df[col] == 0).sum(),
            'zeros_percent': ((df[col] == 0).sum() / len(df)) * 100,
            'negative_count': (df[col] < 0).sum(),
            'negative_percent': ((df[col] < 0).sum() / len(df)) * 100
        }
        
        # Create distribution plot
        if output_dir:
            plot_path = os.path.join(output_dir, f"{col}_distribution.png")
            
            plot_distribution(
                df, 
                col, 
                title=f"Distribution of {col}",
                xlabel=col,
                save_path=plot_path
            )
        
        # Add to result dictionary
        result[col] = stats
    
    return result

def analyze_categorical_distributions(df: pd.DataFrame,
                                    columns: Optional[List[str]] = None,
                                    output_dir: Optional[str] = None,
                                    max_categories: int = 20) -> Dict[str, Dict[str, Any]]:
    """
    Analyze distributions of categorical columns.
    
    Args:
        df: DataFrame to analyze
        columns: List of categorical columns to analyze (if None, all object columns)
        output_dir: Optional directory to save the plots
        max_categories: Maximum number of categories to include in plots
        
    Returns:
        Dictionary with distribution statistics for each column
    """
    # Select categorical columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize result dictionary
    result = {}
    
    # Analyze each column
    for col in columns:
        if col not in df.columns:
            continue
            
        # Calculate statistics
        value_counts = df[col].value_counts()
        unique_count = len(value_counts)
        
        stats = {
            'unique_count': unique_count,
            'mode': df[col].mode()[0] if not df[col].mode().empty else None,
            'mode_count': value_counts.iloc[0] if not value_counts.empty else 0,
            'mode_percent': (value_counts.iloc[0] / len(df)) * 100 if not value_counts.empty else 0,
            'missing_count': df[col].isnull().sum(),
            'missing_percent': (df[col].isnull().sum() / len(df)) * 100,
            'is_high_cardinality': unique_count > max_categories
        }
        
        # Get top categories
        top_categories = value_counts.head(max_categories).to_dict()
        stats['top_categories'] = top_categories
        
        # Create bar plot
        if output_dir and unique_count <= max_categories:
            plot_path = os.path.join(output_dir, f"{col}_distribution.png")
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(top_categories.keys(), top_categories.values(), color='skyblue')
            
            # Add values on top of the bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2, 
                    height, 
                    str(int(height)), 
                    ha='center', 
                    va='bottom'
                )
            
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        
        # Add to result dictionary
        result[col] = stats
    
    return result

def analyze_time_series(df: pd.DataFrame,
                      time_col: str,
                      value_col: str,
                      groupby_col: Optional[str] = None,
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze a time series column.
    
    Args:
        df: DataFrame with time series data
        time_col: Column name for the time/date
        value_col: Column name for the values
        groupby_col: Optional column to group by
        output_dir: Optional directory to save the plots
        
    Returns:
        Dictionary with time series analysis results
    """
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort by time
    df = df.sort_values(time_col)
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize result dictionary
    result = {
        'time_range': {
            'start': df[time_col].min().strftime('%Y-%m-%d %H:%M:%S'),
            'end': df[time_col].max().strftime('%Y-%m-%d %H:%M:%S'),
            'duration_days': (df[time_col].max() - df[time_col].min()).days
        },
        'value_stats': {
            'mean': df[value_col].mean(),
            'median': df[value_col].median(),
            'std': df[value_col].std(),
            'min': df[value_col].min(),
            'max': df[value_col].max()
        }
    }
    
    # Check for missing timestamps (daily frequency)
    df['date'] = df[time_col].dt.date
    daily_dates = df['date'].unique()
    date_range = pd.date_range(start=min(daily_dates), end=max(daily_dates)).date
    missing_dates = [d for d in date_range if d not in daily_dates]
    
    result['missing_dates'] = {
        'count': len(missing_dates),
        'percent': len(missing_dates) / len(date_range) * 100,
        'dates': [d.strftime('%Y-%m-%d') for d in missing_dates[:10]]
    }
    
    # Create time series plot
    if output_dir:
        # Full time series
        plot_path = os.path.join(output_dir, f"{value_col}_time_series.png")
        
        plot_time_series(
            df, 
            time_col, 
            value_col, 
            title=f"Time Series of {value_col}",
            ylabel=value_col,
            save_path=plot_path
        )
        
        # Hourly profile
        hourly_plot_path = os.path.join(output_dir, f"{value_col}_hourly_profile.png")
        
        plot_hourly_profile(
            df, 
            time_col, 
            value_col, 
            title=f"Hourly Profile of {value_col}",
            save_path=hourly_plot_path
        )
        
        # Compare weekday vs weekend
        weekday_plot_path = os.path.join(output_dir, f"{value_col}_weekday_weekend.png")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create weekday flag
        df['is_weekend'] = df[time_col].dt.dayofweek >= 5
        
        # Group by hour and weekday/weekend
        hourly_data = df.groupby([df[time_col].dt.hour, 'is_weekend'])[value_col].mean().reset_index()
        hourly_data.columns = ['hour', 'is_weekend', value_col]
        
        # Plot weekday vs weekend
        for is_weekend, label in [(False, 'Weekday'), (True, 'Weekend')]:
            subset = hourly_data[hourly_data['is_weekend'] == is_weekend]
            ax.plot(subset['hour'], subset[value_col], marker='o', label=label)
        
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel(f'Average {value_col}')
        ax.set_title(f'Weekday vs Weekend Profile of {value_col}')
        ax.set_xticks(range(0, 24))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(weekday_plot_path)
        plt.close()
        
        # If we have group by column, create grouped analysis
        if groupby_col and groupby_col in df.columns:
            # Calculate stats by group
            group_stats = df.groupby(groupby_col)[value_col].agg(['mean', 'std', 'min', 'max']).reset_index()
            result['group_stats'] = group_stats.to_dict(orient='records')
            
            # Create a plot comparing groups
            groups_plot_path = os.path.join(output_dir, f"{value_col}_by_{groupby_col}.png")
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x=groupby_col, y=value_col, data=df, estimator=np.mean)
            plt.title(f"Average {value_col} by {groupby_col}")
            plt.xlabel(groupby_col)
            plt.ylabel(f"Average {value_col}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(groups_plot_path)
            plt.close()
    
    return result

def analyze_correlations(df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       output_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Analyze correlations between numerical columns.
    
    Args:
        df: DataFrame to analyze
        columns: List of numerical columns to analyze (if None, all numeric columns)
        output_path: Optional path to save the correlation heatmap
        
    Returns:
        Dictionary with correlation values for each column pair
    """
    # Select numerical columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create correlation heatmap
    if output_path:
        plot_heatmap(
            df[columns], 
            columns=columns, 
            title="Correlation Matrix",
            save_path=output_path
        )
    
    # Convert correlation matrix to dictionary
    result = {}
    for col1 in columns:
        result[col1] = {}
        for col2 in columns:
            result[col1][col2] = corr_matrix.loc[col1, col2]
    
    return result

def identify_skewed_columns(df: pd.DataFrame,
                          threshold: float = 1.0) -> Dict[str, float]:
    """
    Identify skewed numerical columns.
    
    Args:
        df: DataFrame to analyze
        threshold: Threshold for considering a column skewed
        
    Returns:
        Dictionary with skewed columns and their skewness values
    """
    # Select numerical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Calculate skewness
    skewness = df[num_cols].skew()
    
    # Identify skewed columns
    skewed_cols = skewness[abs(skewness) > threshold].to_dict()
    
    return skewed_cols

def generate_eda_report(df: pd.DataFrame,
                       title: str = "Exploratory Data Analysis Report",
                       output_dir: str = "eda_report",
                       time_col: Optional[str] = None,
                       value_col: Optional[str] = None,
                       groupby_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive EDA report.
    
    Args:
        df: DataFrame to analyze
        title: Report title
        output_dir: Directory to save the report files
        time_col: Optional time column for time series analysis
        value_col: Optional value column for time series analysis
        groupby_col: Optional column to group by
        
    Returns:
        Dictionary with all analysis results
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize result dictionary
    result = {
        'title': title,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=np.number).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        }
    }
    
    # Generate data profile
    profile_path = os.path.join(output_dir, "data_profile.html")
    generate_data_profile(df, title=title, output_path=profile_path)
    result['profile_path'] = profile_path
    
    # Analyze missing values
    missing_plot_path = os.path.join(output_dir, "missing_values.png")
    result['missing_values'] = analyze_missing_values(df, output_path=missing_plot_path)
    
    # Analyze numerical distributions
    num_dir = os.path.join(output_dir, "numerical")
    result['numerical'] = analyze_numerical_distributions(df, output_dir=num_dir)
    
    # Analyze categorical distributions
    cat_dir = os.path.join(output_dir, "categorical")
    result['categorical'] = analyze_categorical_distributions(df, output_dir=cat_dir)
    
    # Analyze correlations
    corr_path = os.path.join(output_dir, "correlations.png")
    result['correlations'] = analyze_correlations(df, output_path=corr_path)
    
    # Identify skewed columns
    result['skewed_columns'] = identify_skewed_columns(df)
    
    # Time series analysis if time column is provided
    if time_col and value_col and time_col in df.columns and value_col in df.columns:
        ts_dir = os.path.join(output_dir, "time_series")
        result['time_series'] = analyze_time_series(
            df, time_col, value_col, groupby_col, output_dir=ts_dir
        )
    
    # Save the report as JSON
    report_path = os.path.join(output_dir, "eda_report.json")
    
    # Prepare a serializable version of the results
    serializable_result = {}
    for key, value in result.items():
        if key != 'profile_path':  # Skip the profile path
            try:
                # Test if it can be serialized
                json.dumps(value)
                serializable_result[key] = value
            except (TypeError, OverflowError):
                # If not serializable, convert to a string representation
                serializable_result[key] = str(value)
    
    with open(report_path, 'w') as f:
        json.dump(serializable_result, f, indent=2)
    
    return result

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logging.info("EDA module loaded")
