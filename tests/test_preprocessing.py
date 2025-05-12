import unittest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
from src.data.preprocess import (
    clean_consumption_data,
    clean_weather_data,
    clean_household_data,
    add_temporal_features
)

class TestDataPreprocessing(unittest.TestCase):
    
    def setUp(self):
        # Create sample dataframes for testing
        
        # Sample consumption data
        dates = pd.date_range(
            start='2013-01-01', 
            end='2013-01-02', 
            freq='30Min'
        )
        self.consumption_df = pd.DataFrame({
            'LCLid': ['MAC000001'] * len(dates) + ['MAC000002'] * len(dates),
            'tstp': list(dates) * 2,
            'energy(kWh/hh)': [0.5, 0.7, 0.3, 0.0, 0.2, 1.0, 0.6, 0.4, 0.8, 0.3, 0.9, 0.1,
                               0.5, 0.7, 0.3, 0.0, 0.2, 1.0, 0.6, 0.4, 0.8, 0.3, 0.9, 0.1,
                               0.7, 0.9, 0.5, 0.2, 0.4, 1.2, 0.8, 0.6, 1.0, 0.5, 1.1, 0.3,
                               0.7, 0.9, 0.5, 0.2, 0.4, 1.2, 0.8, 0.6, 1.0, 0.5, 1.1, 0.3]
        })
        
        # Add some outliers and null values for testing
        self.consumption_df.loc[5, 'energy(kWh/hh)'] = 50.0  # outlier
        self.consumption_df.loc[10, 'energy(kWh/hh)'] = np.nan  # null value
        
        # Sample weather data
        weather_dates = pd.date_range(
            start='2013-01-01', 
            end='2013-01-02', 
            freq='1H'
        )
        self.weather_df = pd.DataFrame({
            'time': weather_dates,
            'temperature': [10.5, 11.2, 12.0, 12.5, 13.1, 14.0, 15.2, 16.0, 
                           17.1, 16.8, 16.0, 15.5, 14.8, 14.0, 13.2, 12.5,
                           11.8, 11.5, 11.0, 10.5, 10.2, 10.0, 9.8, 9.5, 9.3],
            'humidity': [0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82,
                        0.85, 0.83, 0.80, 0.78, 0.75, 0.73, 0.70, 0.68,
                        0.65, 0.63, 0.60, 0.58, 0.55, 0.53, 0.50, 0.48, 0.45],
            'windSpeed': [5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
                         8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0,
                         4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
            'precipIntensity': [0.0, 0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.1, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
        # Add some null values for testing
        self.weather_df.loc[3, 'temperature'] = np.nan
        self.weather_df.loc[7, 'humidity'] = np.nan
        
        # Sample household data
        self.household_df = pd.DataFrame({
            'LCLid': ['MAC000001', 'MAC000002', 'MAC000003', 'MAC000004', 'MAC000005'],
            'Acorn': ['ACORN-A', 'ACORN-B', 'ACORN-C', 'ACORN-D', 'ACORN-E'],
            'Acorn_grouped': ['Affluent', 'Comfortable', 'Adversity', 'Comfortable', 'Affluent'],
            'stdorToU': ['Std', 'ToU', 'Std', 'ToU', 'Std']
        })
    
    def test_clean_consumption_data(self):
        # Test consumption data cleaning
        cleaned_df = clean_consumption_data(self.consumption_df)
        
        # Check that outliers were removed
        self.assertLess(cleaned_df['energy(kWh/hh)'].max(), 50.0)
        
        # Check that null values were handled
        self.assertEqual(cleaned_df.isna().sum().sum(), 0)
        
        # Check that the right number of households remain
        self.assertEqual(len(cleaned_df['LCLid'].unique()), 2)
    
    def test_clean_weather_data(self):
        # Test weather data cleaning
        cleaned_df = clean_weather_data(self.weather_df)
        
        # Check that null values were handled
        self.assertEqual(cleaned_df.isna().sum().sum(), 0)
        
        # Check that the time column is preserved
        self.assertTrue('time' in cleaned_df.columns)
        
        # Check that all original rows are present (minus any filtered)
        self.assertLessEqual(len(cleaned_df), len(self.weather_df))
    
    def test_clean_household_data(self):
        # Test household data cleaning
        cleaned_df = clean_household_data(self.household_df)
        
        # Check that all households are present
        self.assertEqual(len(cleaned_df), len(self.household_df))
        
        # Check that the categorical columns are properly encoded
        self.assertTrue('stdorToU' in cleaned_df.columns)
        self.assertTrue('Acorn_grouped' in cleaned_df.columns)
    
    def test_add_temporal_features(self):
        # Test adding temporal features
        df = self.consumption_df.copy()
        df_with_features = add_temporal_features(df, 'tstp')
        
        # Check that new features were added
        expected_features = ['hour', 'day', 'month', 'is_weekend']
        for feature in expected_features:
            self.assertTrue(feature in df_with_features.columns)
        
        # Check that values make sense (test first row)
        first_date = df_with_features.iloc[0]['tstp']
        self.assertEqual(df_with_features.iloc[0]['hour'], first_date.hour)
        self.assertEqual(df_with_features.iloc[0]['day'], first_date.dayofweek)
        self.assertEqual(df_with_features.iloc[0]['month'], first_date.month)
        self.assertEqual(df_with_features.iloc[0]['is_weekend'], first_date.dayofweek >= 5)

if __name__ == "__main__":
    unittest.main() 