"""
Streamlit web application for electricity consumption data visualization and forecasting.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.data.load_data import load_processed_data
from src.pipelines.pipeline import SmartMeterPipeline
from src.utils.config import PROCESSED_DATA_DIR, MODEL_DIR, MODEL_PARAMS

# Set page configuration
st.set_page_config(
    page_title="Smart Meter Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("⚡ Smart Meter Analytics Dashboard")
st.markdown("""
This dashboard provides insights into household electricity consumption data from smart meters in London.
You can explore consumption patterns, analyze weather impacts, and test forecasting models.
""")

# Error handling and instructions sidebar
with st.sidebar:
    st.header("Dashboard Information")
    
    # Check if data directory exists
    if not os.path.exists(PROCESSED_DATA_DIR):
        st.warning("⚠️ Processed data directory not found!")
        st.info("To generate the required datasets, run:\n\n```python run.py --process```")
        
    # Check if models directory exists and contains models
    if not os.path.exists(MODEL_DIR) or not any(f.endswith(('.keras', '.pkl')) for f in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, f))):
        st.warning("⚠️ No trained models found!")
        st.info("To train the forecasting models, run:\n\n```python run.py --train```")
    
    st.subheader("Getting Started")
    st.markdown("""
    1. Make sure data and models are loaded
    2. Navigate through the tabs to explore the data
    3. Use the household selector in the Time Series tab
    4. Try the forecasting tool in the Forecast tab
    """)
    
    # Add refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Create tabs for different sections
tabs = st.tabs(["Data Overview", "Time Series Analysis", "Weather Impact", "Consumption Patterns", "Forecast"])

# Cache data loading
@st.cache_data
def load_data():
    """Load the necessary datasets"""
    results = {'loaded': False}
    
    try:
        # Check if processed data directory exists
        if not os.path.exists(PROCESSED_DATA_DIR):
            return {
                'error': "Processed data directory not found. Run 'python run.py --process' to generate datasets.",
                'loaded': False
            }
        
        # Try loading processed data
        consumption = load_processed_data('hourly')
        results['consumption'] = consumption
        
        try:
            weather = load_processed_data('weather')
            results['weather'] = weather
        except Exception as e:
            results['weather_error'] = str(e)
        
        try:
            households = load_processed_data('household')
            results['households'] = households
        except Exception as e:
            results['household_error'] = str(e)
        
        # If at least consumption data loaded successfully
        if 'consumption' in results:
            results['loaded'] = True
        
        return results
    except Exception as e:
        return {
            'error': str(e),
            'loaded': False
        }

# Cache model loading
@st.cache_resource
def load_models():
    """Load the trained forecasting models"""
    models = {}
    
    try:
        # Check if model directory exists
        if not os.path.exists(MODEL_DIR):
            return {'error': "Model directory not found"}
        
        # Find the most recent models
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras') or f.endswith('.pkl')]
        
        if not model_files:
            return {'error': "No model files found in the models directory"}
        
        # Filter by model types
        lstm_files = [f for f in model_files if 'lstm' in f.lower() and f.endswith('.keras')]
        gru_files = [f for f in model_files if 'gru' in f.lower() and f.endswith('.keras')]
        rf_files = [f for f in model_files if 'random_forest' in f.lower() and f.endswith('.pkl')]
        
        # Load most recent models
        if lstm_files:
            newest_lstm = max(lstm_files, key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)))
            models['lstm'] = load_model(os.path.join(MODEL_DIR, newest_lstm))
        
        if gru_files:
            newest_gru = max(gru_files, key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)))
            models['gru'] = load_model(os.path.join(MODEL_DIR, newest_gru))
        
        if rf_files:
            newest_rf = max(rf_files, key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)))
            with open(os.path.join(MODEL_DIR, newest_rf), 'rb') as f:
                models['random_forest'] = pickle.load(f)
        
        # Load data preprocessing info
        info_files = [f for f in model_files if 'data_info' in f and f.endswith('.pkl')]
        if info_files:
            newest_info = max(info_files, key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)))
            with open(os.path.join(MODEL_DIR, newest_info), 'rb') as f:
                models['data_info'] = pickle.load(f)
        
        return models
    except Exception as e:
        return {'error': str(e)}

# Load data
data = load_data()

# Display error message if data couldn't be loaded
if not data['loaded']:
    error_message = data.get('error', "Unknown error loading data")
    st.error(f"⚠️ {error_message}")
    
    st.warning("The dashboard requires processed data to function.")
    st.info("""
    ### Troubleshooting Steps:
    
    1. Ensure the data has been downloaded and processed:
       ```
       python -m src.download_data
       python run.py --process
       ```
    
    2. Check the logs for specific errors
    
    3. Verify that the data paths in config.py are correct
    """)
    st.stop()  # Stop dashboard execution here

# If data loaded successfully, populate the dashboard
consumption_df = data.get('consumption')
weather_df = data.get('weather')
households_df = data.get('households')

# Check for partial data loading issues
missing_data = []
if 'weather_error' in data:
    missing_data.append(f"Weather data: {data['weather_error']}")
if 'household_error' in data:
    missing_data.append(f"Household data: {data['household_error']}")

if missing_data:
    st.warning("⚠️ Some datasets could not be loaded. Limited functionality available.")
    for msg in missing_data:
        st.info(msg)

# Try to load trained models
models = load_models()
if 'error' in models:
    st.warning(f"⚠️ Could not load prediction models: {models['error']}")
    st.info("The forecasting tab will not be fully functional.")

# Data Overview Tab
with tabs[0]:
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Consumption Data Sample")
        st.dataframe(consumption_df.head())
        
        # Display simple consumption stats
        st.subheader("Consumption Statistics")
        
        # Format datetime column if present
        if 'tstp' in consumption_df.columns:
            consumption_df['tstp'] = pd.to_datetime(consumption_df['tstp'])
            min_date = consumption_df['tstp'].min().strftime('%Y-%m-%d')
            max_date = consumption_df['tstp'].max().strftime('%Y-%m-%d')
            st.write(f"Date Range: {min_date} to {max_date}")
        
        # Show energy consumption statistics
        energy_col = 'energy(kWh/hh)' if 'energy(kWh/hh)' in consumption_df.columns else None
        
        if energy_col:
            st.write(f"Total households: {consumption_df['LCLid'].nunique()}")
            st.write(f"Average consumption: {consumption_df[energy_col].mean():.2f} kWh")
            st.write(f"Max consumption: {consumption_df[energy_col].max():.2f} kWh")
            st.write(f"Min consumption: {consumption_df[energy_col].min():.2f} kWh")
    
    with col2:
        if weather_df is not None:
            st.subheader("Weather Data Sample")
            st.dataframe(weather_df.head())
            
            # Display weather stats
            st.subheader("Weather Statistics")
            
            # Format datetime column if present
            if 'time' in weather_df.columns:
                weather_df['time'] = pd.to_datetime(weather_df['time'])
                min_date = weather_df['time'].min().strftime('%Y-%m-%d')
                max_date = weather_df['time'].max().strftime('%Y-%m-%d')
                st.write(f"Date Range: {min_date} to {max_date}")
            
            # Show temperature statistics if available
            if 'temperature' in weather_df.columns:
                st.write(f"Average temperature: {weather_df['temperature'].mean():.2f}°C")
                st.write(f"Max temperature: {weather_df['temperature'].max():.2f}°C")
                st.write(f"Min temperature: {weather_df['temperature'].min():.2f}°C")
        else:
            st.error("Weather data is not available")
            st.info("Some visualizations will be limited without weather data")
    
    # Households summary
    if households_df is not None:
        st.subheader("Household Information")
        st.dataframe(households_df.head())
        
        # ACORN groups distribution if available
        if 'Acorn_grouped' in households_df.columns:
            st.subheader("ACORN Group Distribution")
            acorn_counts = households_df['Acorn_grouped'].value_counts().reset_index()
            acorn_counts.columns = ['ACORN Group', 'Count']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='ACORN Group', y='Count', data=acorn_counts, ax=ax)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error("Household information is not available")
        st.info("Some visualizations will be limited without household metadata")

# Time Series Analysis Tab
with tabs[1]:
    st.header("Time Series Analysis")
    
    # Select household for analysis
    households = sorted(consumption_df['LCLid'].unique())
    selected_household = st.selectbox("Select Household", households)
    
    # Filter data for selected household
    household_data = consumption_df[consumption_df['LCLid'] == selected_household]
    
    # Ensure datetime is properly formatted
    if 'tstp' in household_data.columns:
        household_data['tstp'] = pd.to_datetime(household_data['tstp'])
        household_data = household_data.sort_values('tstp')
        
        # Resample to daily for better visualization
        energy_col = 'energy(kWh/hh)' if 'energy(kWh/hh)' in household_data.columns else None
        
        if energy_col:
            # Create daily and monthly views
            daily_data = household_data.set_index('tstp').resample('D')[energy_col].sum().reset_index()
            monthly_data = household_data.set_index('tstp').resample('M')[energy_col].sum().reset_index()
            
            # Date range selector
            min_date = household_data['tstp'].min().date()
            max_date = household_data['tstp'].max().date()
            
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, min(min_date + timedelta(days=90), max_date)),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                
                # Filter by date range
                filtered_daily = daily_data[(daily_data['tstp'].dt.date >= start_date) & 
                                         (daily_data['tstp'].dt.date <= end_date)]
                
                # Plot daily consumption
                st.subheader(f"Daily Consumption for {selected_household}")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(filtered_daily['tstp'], filtered_daily[energy_col], marker='o', linestyle='-')
                ax.set_xlabel('Date')
                ax.set_ylabel('Daily Consumption (kWh)')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add monthly view
                st.subheader("Monthly Consumption Trend")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(monthly_data['tstp'], monthly_data[energy_col], marker='o', linestyle='-', color='orange')
                ax.set_xlabel('Month')
                ax.set_ylabel('Monthly Consumption (kWh)')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add day of week analysis
                st.subheader("Consumption by Day of Week")
                
                # Create day of week column
                filtered_daily['day_of_week'] = filtered_daily['tstp'].dt.day_name()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Group by day of week
                dow_data = filtered_daily.groupby('day_of_week')[energy_col].mean().reindex(day_order).reset_index()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='day_of_week', y=energy_col, data=dow_data, ax=ax)
                ax.set_xlabel('Day of Week')
                ax.set_ylabel('Average Daily Consumption (kWh)')
                plt.tight_layout()
                st.pyplot(fig)

# Weather Impact Tab
with tabs[2]:
    st.header("Weather Impact Analysis")
    
    # Merge consumption and weather data
    if 'tstp' in consumption_df.columns and 'time' in weather_df.columns:
        # Prepare timestamps for joining
        consumption_sample = consumption_df.copy()
        consumption_sample['tstp'] = pd.to_datetime(consumption_sample['tstp'])
        consumption_sample['hour'] = consumption_sample['tstp'].dt.floor('H')
        
        weather_sample = weather_df.copy()
        weather_sample['time'] = pd.to_datetime(weather_sample['time'])
        weather_sample['hour'] = weather_sample['time'].dt.floor('H')
        
        # Join datasets
        merged_data = pd.merge(
            consumption_sample, 
            weather_sample, 
            left_on='hour', 
            right_on='hour', 
            how='inner'
        )
        
        # Temperature vs. consumption
        if 'temperature' in merged_data.columns and 'energy(kWh/hh)' in merged_data.columns:
            st.subheader("Temperature vs. Energy Consumption")
            
            # Scatter plot with regression line
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x='temperature', y='energy(kWh/hh)', data=merged_data.sample(1000), ax=ax)
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('Energy Consumption (kWh)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Temperature bins for bar chart
            st.subheader("Average Consumption by Temperature Range")
            merged_data['temp_bin'] = pd.cut(
                merged_data['temperature'], 
                bins=range(-5, 30, 5), 
                labels=[f"{i}°C to {i+5}°C" for i in range(-5, 25, 5)]
            )
            
            temp_group = merged_data.groupby('temp_bin')['energy(kWh/hh)'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='temp_bin', y='energy(kWh/hh)', data=temp_group, ax=ax)
            ax.set_xlabel('Temperature Range')
            ax.set_ylabel('Average Energy Consumption (kWh)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Humidity impact if available
            if 'humidity' in merged_data.columns:
                st.subheader("Humidity vs. Energy Consumption")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.regplot(x='humidity', y='energy(kWh/hh)', data=merged_data.sample(1000), ax=ax)
                ax.set_xlabel('Humidity')
                ax.set_ylabel('Energy Consumption (kWh)')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Weather summary by season
            if 'time_month' in merged_data.columns or 'time' in merged_data.columns:
                if 'time_month' not in merged_data.columns:
                    merged_data['time_month'] = merged_data['time'].dt.month
                
                st.subheader("Seasonal Consumption Patterns")
                
                # Define seasons
                season_mapping = {
                    1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn',
                    11: 'Autumn', 12: 'Winter'
                }
                
                merged_data['season'] = merged_data['time_month'].map(season_mapping)
                season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
                
                seasonal_data = merged_data.groupby('season')['energy(kWh/hh)'].mean().reindex(season_order).reset_index()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='season', y='energy(kWh/hh)', data=seasonal_data, ax=ax)
                ax.set_xlabel('Season')
                ax.set_ylabel('Average Energy Consumption (kWh)')
                plt.tight_layout()
                st.pyplot(fig)

# Consumption Patterns Tab
with tabs[3]:
    st.header("Consumption Pattern Analysis")
    
    # Check if we have hourly data
    if 'tstp' in consumption_df.columns:
        consumption_df['tstp'] = pd.to_datetime(consumption_df['tstp'])
        consumption_df['hour'] = consumption_df['tstp'].dt.hour
        consumption_df['day_of_week'] = consumption_df['tstp'].dt.dayofweek  # 0=Monday, 6=Sunday
        
        # Hourly consumption patterns
        st.subheader("Hourly Consumption Patterns")
        
        hourly_avg = consumption_df.groupby('hour')['energy(kWh/hh)'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hourly_avg['hour'], hourly_avg['energy(kWh/hh)'], marker='o', linestyle='-')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Energy Consumption (kWh)')
        ax.set_xticks(range(0, 24))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Weekday vs. weekend patterns
        st.subheader("Weekday vs. Weekend Consumption")
        
        # Create a weekday/weekend flag
        consumption_df['is_weekend'] = consumption_df['day_of_week'].isin([5, 6])  # 5=Saturday, 6=Sunday
        
        # Group by hour and weekday/weekend
        hourly_weekday = consumption_df.groupby(['hour', 'is_weekend'])['energy(kWh/hh)'].mean().reset_index()
        
        # Create separate series for weekday and weekend
        weekday_data = hourly_weekday[hourly_weekday['is_weekend'] == False].copy()
        weekend_data = hourly_weekday[hourly_weekday['is_weekend'] == True].copy()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(weekday_data['hour'], weekday_data['energy(kWh/hh)'], marker='o', linestyle='-', label='Weekday')
        ax.plot(weekend_data['hour'], weekend_data['energy(kWh/hh)'], marker='x', linestyle='--', label='Weekend')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Energy Consumption (kWh)')
        ax.set_xticks(range(0, 24))
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # ACORN group comparison if available
        if 'Acorn_grouped' in households_df.columns:
            st.subheader("Consumption by ACORN Group")
            
            # Merge consumption with household data
            acorn_consumption = pd.merge(
                consumption_df,
                households_df[['LCLid', 'Acorn_grouped']],
                on='LCLid',
                how='inner'
            )
            
            # Group by ACORN and calculate average
            acorn_avg = acorn_consumption.groupby('Acorn_grouped')['energy(kWh/hh)'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Acorn_grouped', y='energy(kWh/hh)', data=acorn_avg, ax=ax)
            ax.set_xlabel('ACORN Group')
            ax.set_ylabel('Average Energy Consumption (kWh)')
            plt.tight_layout()
            st.pyplot(fig)

# Forecast Tab
with tabs[4]:
    st.header("Electricity Consumption Forecasting")
    
    if models:
        st.subheader("Forecast New Data")
        
        # Select a household for forecasting
        households = sorted(consumption_df['LCLid'].unique())
        forecast_household = st.selectbox("Select Household for Forecasting", households, key="forecast_house")
        
        # Filter data for selected household
        household_data = consumption_df[consumption_df['LCLid'] == forecast_household].copy()
        
        # Ensure data is sorted by time
        if 'tstp' in household_data.columns:
            household_data['tstp'] = pd.to_datetime(household_data['tstp'])
            household_data = household_data.sort_values('tstp')
            
            # Select model for forecasting
            available_models = list(models.keys())
            available_models.remove('data_info') if 'data_info' in available_models else None
            
            selected_model = st.selectbox("Select Forecasting Model", available_models)
            
            if selected_model and 'data_info' in models:
                data_info = models['data_info']
                model = models[selected_model]
                
                # Prepare the most recent data for forecasting
                lookback = data_info['lookback']
                forecast_horizon = data_info['forecast_horizon']
                feature_cols = data_info['feature_cols']
                target_col = data_info['target_col']
                
                # Check if we have all required feature columns
                missing_cols = [col for col in feature_cols if col not in household_data.columns]
                
                if missing_cols:
                    st.warning(f"Missing columns for forecasting: {missing_cols}")
                else:
                    # Get the most recent data for forecasting
                    recent_data = household_data.tail(lookback).copy()
                    
                    if len(recent_data) >= lookback:
                        # Prepare features for prediction
                        features = recent_data[feature_cols].values
                        scaled_features = data_info['feature_scaler'].transform(features)
                        
                        # Reshape for model input
                        X_pred = scaled_features.reshape(1, lookback, len(feature_cols))
                        
                        # Generate forecast
                        if selected_model in ['lstm', 'gru', 'cnn']:
                            # Deep learning model
                            forecast = model.predict(X_pred)[0]
                        else:
                            # Traditional model
                            X_pred_reshaped = X_pred.reshape(1, -1)
                            if isinstance(model, list):
                                # Multiple models for multi-output
                                forecast = np.array([m.predict(X_pred_reshaped)[0] for m in model])
                            else:
                                forecast = model.predict(X_pred_reshaped)[0]
                        
                        # Inverse transform to get original scale
                        forecast_reshaped = forecast.reshape(-1, 1)
                        forecast_orig = data_info['target_scaler'].inverse_transform(forecast_reshaped).flatten()
                        
                        # Create a timestamp index for the forecast
                        last_time = household_data['tstp'].iloc[-1]
                        forecast_times = [last_time + timedelta(hours=i+1) for i in range(forecast_horizon)]
                        
                        # Create a forecast dataframe
                        forecast_df = pd.DataFrame({
                            'Time': forecast_times,
                            'Forecast': forecast_orig
                        })
                        
                        # Display the forecast
                        st.subheader(f"{forecast_horizon} Hour Forecast")
                        st.dataframe(forecast_df)
                        
                        # Plot the forecast
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot historical data
                        hist_data = household_data.tail(lookback * 2)
                        ax.plot(hist_data['tstp'], hist_data[target_col], label='Historical', marker='o')
                        
                        # Plot forecast
                        ax.plot(forecast_df['Time'], forecast_df['Forecast'], label='Forecast', marker='x', linestyle='--', color='red')
                        
                        ax.set_xlabel('Time')
                        ax.set_ylabel(f'Energy Consumption ({target_col})')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show forecast statistics
                        st.subheader("Forecast Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Average Consumption", f"{forecast_orig.mean():.2f} kWh")
                        
                        with col2:
                            st.metric("Peak Consumption", f"{forecast_orig.max():.2f} kWh")
                        
                        with col3:
                            st.metric("Minimum Consumption", f"{forecast_orig.min():.2f} kWh")
                    else:
                        st.warning(f"Not enough historical data for forecasting. Need at least {lookback} records.")
    else:
        st.info("No trained models found. Please train models first.")

# Add footer
st.markdown("---")
st.markdown("Smart Meter Analytics Dashboard | Created with Streamlit | Data from London Smart Meters Dataset") 