"""
Visualization module for electricity consumption data analysis.
Provides reusable plotting functions for notebooks and the web app.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any
import os
from datetime import datetime, timedelta

def set_plot_style():
    """Set consistent plot style for all visualizations"""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

def plot_time_series(data: pd.DataFrame,
                    time_col: str,
                    value_col: str,
                    title: str = "Time Series Plot",
                    xlabel: str = "Time",
                    ylabel: str = "Value",
                    figsize: Tuple[int, int] = (12, 6),
                    marker: str = 'o',
                    color: str = 'blue',
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a time series plot.
    
    Args:
        data: DataFrame containing time series data
        time_col: Column name for the time/date
        value_col: Column name for the values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        marker: Marker style
        color: Line color
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        data = data.copy()
        data[time_col] = pd.to_datetime(data[time_col])
    
    # Sort by time
    data = data.sort_values(time_col)
    
    # Plot data
    ax.plot(data[time_col], data[value_col], marker=marker, color=color, linestyle='-')
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_comparison(data: pd.DataFrame,
                  time_col: str,
                  value_cols: List[str],
                  labels: Optional[List[str]] = None,
                  colors: Optional[List[str]] = None,
                  markers: Optional[List[str]] = None,
                  title: str = "Comparison Plot",
                  xlabel: str = "Time",
                  ylabel: str = "Value",
                  figsize: Tuple[int, int] = (12, 6),
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comparison plot with multiple time series.
    
    Args:
        data: DataFrame containing time series data
        time_col: Column name for the time/date
        value_cols: List of column names for the values to compare
        labels: Optional list of labels for the series
        colors: Optional list of colors for the series
        markers: Optional list of markers for the series
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        data = data.copy()
        data[time_col] = pd.to_datetime(data[time_col])
    
    # Sort by time
    data = data.sort_values(time_col)
    
    # Default parameters if not provided
    if labels is None:
        labels = value_cols
    
    if colors is None:
        colors = [f"C{i}" for i in range(len(value_cols))]
    
    if markers is None:
        markers = ['o'] * len(value_cols)
    
    # Plot each series
    for i, col in enumerate(value_cols):
        ax.plot(
            data[time_col], 
            data[col], 
            marker=markers[i] if i < len(markers) else 'o',
            color=colors[i] if i < len(colors) else f"C{i}",
            label=labels[i] if i < len(labels) else col,
            linestyle='-'
        )
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_distribution(data: pd.DataFrame,
                    column: str,
                    title: str = "Distribution Plot",
                    xlabel: str = "Value",
                    ylabel: str = "Frequency",
                    figsize: Tuple[int, int] = (12, 6),
                    bins: int = 30,
                    kde: bool = True,
                    color: str = 'blue',
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a distribution plot (histogram with KDE).
    
    Args:
        data: DataFrame containing data
        column: Column name for the values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        bins: Number of histogram bins
        kde: Whether to include KDE curve
        color: Histogram color
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot distribution
    sns.histplot(data[column], bins=bins, kde=kde, color=color, ax=ax)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_boxplot(data: pd.DataFrame,
               x_col: str,
               y_col: str,
               title: str = "Box Plot",
               xlabel: str = "Category",
               ylabel: str = "Value",
               figsize: Tuple[int, int] = (12, 6),
               color: str = 'skyblue',
               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a box plot for categorical comparison.
    
    Args:
        data: DataFrame containing data
        x_col: Column name for the categories
        y_col: Column name for the values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        color: Box color
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot boxplot
    sns.boxplot(x=x_col, y=y_col, data=data, color=color, ax=ax)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Rotate x labels if there are many categories
    if data[x_col].nunique() > 5:
        plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_heatmap(data: pd.DataFrame,
               columns: Optional[List[str]] = None,
               title: str = "Correlation Heatmap",
               figsize: Tuple[int, int] = (10, 8),
               cmap: str = 'coolwarm',
               annot: bool = True,
               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a correlation heatmap.
    
    Args:
        data: DataFrame containing data
        columns: List of column names to include (if None, use all numeric columns)
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        annot: Whether to annotate cells with values
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Select columns if specified, otherwise use all numeric columns
    if columns:
        df = data[columns].copy()
    else:
        df = data.select_dtypes(include=['number'])
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(corr, annot=annot, cmap=cmap, ax=ax, fmt=".2f", linewidths=0.5)
    
    # Set title
    ax.set_title(title)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_scatter(data: pd.DataFrame,
               x_col: str,
               y_col: str,
               hue_col: Optional[str] = None,
               title: str = "Scatter Plot",
               xlabel: Optional[str] = None,
               ylabel: Optional[str] = None,
               figsize: Tuple[int, int] = (10, 6),
               alpha: float = 0.7,
               add_regline: bool = False,
               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a scatter plot.
    
    Args:
        data: DataFrame containing data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        hue_col: Optional column name for color grouping
        title: Plot title
        xlabel: X-axis label (if None, use x_col)
        ylabel: Y-axis label (if None, use y_col)
        figsize: Figure size
        alpha: Point transparency
        add_regline: Whether to add regression line
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set default labels if not provided
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col
    
    # Plot scatter plot
    if add_regline:
        # Use regplot
        if hue_col:
            # For hue, we need to create regplots for each group
            for category in data[hue_col].unique():
                subset = data[data[hue_col] == category]
                sns.regplot(
                    x=x_col, 
                    y=y_col, 
                    data=subset, 
                    scatter_kws={'alpha': alpha, 'label': category},
                    line_kws={'label': f"{category} trend"},
                    ax=ax
                )
        else:
            # Simple regplot
            sns.regplot(x=x_col, y=y_col, data=data, scatter_kws={'alpha': alpha}, ax=ax)
    else:
        # Use scatterplot
        sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=data, alpha=alpha, ax=ax)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend if there's a hue column
    if hue_col:
        ax.legend(title=hue_col)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_bar(data: pd.DataFrame,
           x_col: str,
           y_col: str,
           hue_col: Optional[str] = None,
           title: str = "Bar Plot",
           xlabel: Optional[str] = None,
           ylabel: Optional[str] = None,
           figsize: Tuple[int, int] = (12, 6),
           rotation: int = 0,
           save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a bar plot.
    
    Args:
        data: DataFrame containing data
        x_col: Column name for x-axis (categories)
        y_col: Column name for y-axis (values)
        hue_col: Optional column name for color grouping
        title: Plot title
        xlabel: X-axis label (if None, use x_col)
        ylabel: Y-axis label (if None, use y_col)
        figsize: Figure size
        rotation: X-tick label rotation
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set default labels if not provided
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col
    
    # Plot bar plot
    sns.barplot(x=x_col, y=y_col, hue=hue_col, data=data, ax=ax)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Rotate x labels if requested
    if rotation != 0:
        plt.xticks(rotation=rotation)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_hourly_profile(data: pd.DataFrame,
                      time_col: str,
                      value_col: str,
                      title: str = "Hourly Consumption Profile",
                      figsize: Tuple[int, int] = (12, 6),
                      day_type: Optional[str] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Create an hourly profile plot (average by hour of day).
    
    Args:
        data: DataFrame containing time series data
        time_col: Column name for the datetime
        value_col: Column name for the consumption values
        title: Plot title
        figsize: Figure size
        day_type: Optional filter for 'weekday' or 'weekend'
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        data = data.copy()
        data[time_col] = pd.to_datetime(data[time_col])
    
    # Create a copy to avoid modifying original
    df = data.copy()
    
    # Extract hour of day
    df['hour'] = df[time_col].dt.hour
    
    # Filter for weekday/weekend if specified
    if day_type:
        # Create day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df[time_col].dt.dayofweek
        
        if day_type.lower() == 'weekday':
            df = df[df['day_of_week'] < 5]  # Monday to Friday
            title += " (Weekdays)"
        elif day_type.lower() == 'weekend':
            df = df[df['day_of_week'] >= 5]  # Saturday and Sunday
            title += " (Weekends)"
    
    # Group by hour and calculate average
    hourly_profile = df.groupby('hour')[value_col].mean().reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot hourly profile
    ax.plot(hourly_profile['hour'], hourly_profile[value_col], marker='o', linestyle='-')
    
    # Set labels and title
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Consumption')
    ax.set_title(title)
    
    # Set x-ticks to show all hours
    ax.set_xticks(range(0, 24))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_forecast_vs_actual(actual: np.ndarray,
                          forecast: np.ndarray,
                          start_date: Union[str, datetime],
                          freq: str = 'H',
                          title: str = "Forecast vs Actual",
                          figsize: Tuple[int, int] = (12, 6),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a plot comparing forecast vs actual values.
    
    Args:
        actual: Array of actual values
        forecast: Array of forecasted values
        start_date: Start date for the time axis
        freq: Frequency for the time axis ('H' for hourly, 'D' for daily, etc.)
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Convert start_date to datetime if it's a string
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    # Create date range for x-axis
    dates = pd.date_range(start=start_date, periods=len(actual), freq=freq)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual and forecast
    ax.plot(dates, actual, marker='o', linestyle='-', label='Actual', color='blue')
    ax.plot(dates, forecast, marker='x', linestyle='--', label='Forecast', color='red')
    
    # Calculate error metrics
    mse = np.mean((actual - forecast) ** 2)
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(mse)
    
    # Add error metrics to title
    title = f"{title} (RMSE: {rmse:.4f}, MAE: {mae:.4f})"
    
    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_model_comparison(models: Dict[str, Dict[str, float]],
                        metric: str = 'rmse',
                        title: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a bar plot comparing multiple models.
    
    Args:
        models: Dictionary of model metrics {model_name: {metric_name: value}}
        metric: Metric to compare ('rmse', 'mae', etc.)
        title: Plot title (if None, automatically generated)
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Create DataFrame from the models dictionary
    data = []
    for model_name, metrics in models.items():
        if metric in metrics:
            data.append({
                'Model': model_name,
                'Value': metrics[metric],
                'Metric': metric.upper()
            })
    
    df = pd.DataFrame(data)
    
    # Generate title if not provided
    if title is None:
        title = f"Model Comparison by {metric.upper()}"
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bar chart
    sns.barplot(x='Model', y='Value', hue='Metric', data=df, ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_feature_importance(feature_names: List[str],
                          importance_values: np.ndarray,
                          title: str = "Feature Importance",
                          figsize: Tuple[int, int] = (10, 8),
                          top_n: Optional[int] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a horizontal bar plot of feature importance.
    
    Args:
        feature_names: List of feature names
        importance_values: Array of importance values
        title: Plot title
        figsize: Figure size
        top_n: Number of top features to show (if None, show all)
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False)
    
    # Limit to top_n if specified
    if top_n is not None and top_n < len(df):
        df = df.head(top_n)
        title = f"Top {top_n} {title}"
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    sns.barplot(x='Importance', y='Feature', data=df, ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig
