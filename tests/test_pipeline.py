import unittest
import os
import pandas as pd
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock
from src.pipelines.pipeline import SmartMeterPipeline

class TestSmartMeterPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create a mock pipeline for testing
        self.pipeline = SmartMeterPipeline()
        
        # Create sample dataframes to mock loading
        self.mock_household_data = pd.DataFrame({
            'LCLid': ['MAC000001', 'MAC000002', 'MAC000003'],
            'Acorn': ['ACORN-A', 'ACORN-B', 'ACORN-C'],
            'Acorn_grouped': ['Affluent', 'Comfortable', 'Adversity'],
            'stdorToU': ['Std', 'ToU', 'Std']
        })
        
        self.mock_weather_data = pd.DataFrame({
            'time': pd.date_range(start='2013-01-01', periods=24, freq='H'),
            'temperature': np.random.uniform(5, 15, 24),
            'humidity': np.random.uniform(0.5, 0.9, 24),
            'windSpeed': np.random.uniform(0, 10, 24),
            'precipIntensity': np.random.uniform(0, 0.3, 24)
        })
        
        self.mock_consumption_data = pd.DataFrame({
            'LCLid': ['MAC000001', 'MAC000001', 'MAC000002', 'MAC000002', 'MAC000003', 'MAC000003'],
            'tstp': pd.date_range(start='2013-01-01', periods=6, freq='30Min'),
            'energy(kWh/hh)': [0.5, 0.7, 0.3, 0.2, 0.8, 0.4]
        })
    
    @patch('src.pipelines.pipeline.pd.read_csv')
    def test_load_raw_data(self, mock_read_csv):
        # Configure the mock to return our sample dataframes
        mock_read_csv.side_effect = [
            self.mock_household_data,
            self.mock_weather_data,
            self.mock_consumption_data
        ]
        
        # Call the method we're testing
        self.pipeline.load_raw_data(['household_info', 'weather_hourly', 'halfhourly_dataset'])
        
        # Check that read_csv was called the expected number of times
        self.assertEqual(mock_read_csv.call_count, 3)
        
        # Check that the data was correctly assigned
        self.assertTrue(hasattr(self.pipeline, 'household_data'))
        self.assertTrue(hasattr(self.pipeline, 'weather_data'))
        self.assertTrue(hasattr(self.pipeline, 'consumption_data'))
    
    @patch('src.pipelines.pipeline.clean_household_data')
    def test_preprocess_household_data(self, mock_clean):
        # Setup the pipeline with mock data
        self.pipeline.household_data = self.mock_household_data
        
        # Configure the mock to return the same dataframe (simplified for test)
        mock_clean.return_value = self.mock_household_data
        
        # Call the method we're testing
        self.pipeline.preprocess_household_data()
        
        # Check that the cleaning function was called
        mock_clean.assert_called_once_with(self.mock_household_data)
        
        # Check that the processed data was saved
        self.assertTrue(hasattr(self.pipeline, 'processed_household_data'))
    
    @patch('src.pipelines.pipeline.clean_weather_data')
    @patch('src.pipelines.pipeline.add_temporal_features')
    def test_preprocess_weather_data(self, mock_add_features, mock_clean):
        # Setup the pipeline with mock data
        self.pipeline.weather_data = self.mock_weather_data
        
        # Configure the mocks
        mock_clean.return_value = self.mock_weather_data
        mock_add_features.return_value = self.mock_weather_data  # simplified
        
        # Call the method we're testing
        self.pipeline.preprocess_weather_data(hourly=True)
        
        # Check that the functions were called
        mock_clean.assert_called_once_with(self.mock_weather_data)
        mock_add_features.assert_called_once()
        
        # Check that the processed data was saved
        self.assertTrue(hasattr(self.pipeline, 'processed_weather_hourly'))
    
    @patch('src.pipelines.pipeline.clean_consumption_data')
    @patch('src.pipelines.pipeline.add_temporal_features')
    def test_preprocess_consumption_data(self, mock_add_features, mock_clean):
        # Setup the pipeline with mock data
        self.pipeline.consumption_data = self.mock_consumption_data
        
        # Configure the mocks
        mock_clean.return_value = self.mock_consumption_data
        mock_add_features.return_value = self.mock_consumption_data  # simplified
        
        # Call the method we're testing
        self.pipeline.preprocess_consumption_data(halfhourly=True)
        
        # Check that the functions were called
        mock_clean.assert_called_once()
        mock_add_features.assert_called_once()
        
        # Check that the processed data was saved
        self.assertTrue(hasattr(self.pipeline, 'processed_halfhourly'))
    
    @patch('src.pipelines.pipeline.pd.merge')
    def test_create_analysis_dataset(self, mock_merge):
        # Setup the pipeline with processed mock data
        self.pipeline.processed_halfhourly = self.mock_consumption_data
        self.pipeline.processed_weather_hourly = self.mock_weather_data
        self.pipeline.processed_household_data = self.mock_household_data
        
        # Configure the mock to return a merged dataframe
        mock_merged_df = pd.DataFrame({
            'LCLid': ['MAC000001', 'MAC000001', 'MAC000002'],
            'tstp': pd.date_range(start='2013-01-01', periods=3, freq='30Min'),
            'energy(kWh/hh)': [0.5, 0.7, 0.3],
            'temperature': [10.0, 10.0, 11.0],
            'humidity': [0.7, 0.7, 0.75],
            'Acorn_grouped': ['Affluent', 'Affluent', 'Comfortable']
        })
        mock_merge.return_value = mock_merged_df
        
        # Patch the save method to avoid file operations
        with patch.object(self.pipeline, 'save_processed_data'):
            # Call the method we're testing
            self.pipeline.create_analysis_dataset()
            
            # Check that merge was called
            self.assertEqual(mock_merge.call_count, 2)
            
            # Check that the analysis dataset was created
            self.assertTrue(hasattr(self.pipeline, 'analysis_dataset'))
            self.assertEqual(self.pipeline.analysis_dataset.equals(mock_merged_df), True)

if __name__ == "__main__":
    unittest.main() 