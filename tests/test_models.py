import unittest
import os
import pandas as pd
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor

from src.models.train import (
    prepare_data_for_training,
    create_lstm_model,
    create_gru_model,
    create_cnn_model,
    train_random_forest,
    train_electricity_forecast_models
)

class TestModelTraining(unittest.TestCase):
    
    def setUp(self):
        # Create sample data for testing
        dates = pd.date_range(start='2013-01-01', periods=1000, freq='H')
        
        # Create features with predictable patterns for testing
        hour_of_day = np.array([d.hour for d in dates])
        day_of_week = np.array([d.dayofweek for d in dates])
        month = np.array([d.month for d in dates])
        
        # Create target variable with a simple pattern based on the features
        # This helps verify that models actually learn something
        energy = (
            0.5 + 
            0.1 * np.sin(hour_of_day * 2 * np.pi / 24) +  # Daily cycle
            0.05 * np.sin(day_of_week * 2 * np.pi / 7) +  # Weekly cycle
            0.02 * np.sin(month * 2 * np.pi / 12)         # Annual cycle
        )
        
        # Add some noise
        energy += np.random.normal(0, 0.02, size=len(dates))
        
        # Create a sample dataframe
        self.df = pd.DataFrame({
            'tstp': dates,
            'energy(kWh/hh)': energy,
            'hour': hour_of_day,
            'day': day_of_week,
            'month': month,
            'is_weekend': day_of_week >= 5,
            'is_holiday': np.random.choice([False, True], size=len(dates), p=[0.95, 0.05]),
            'temperature': 15 + 10 * np.sin(month * 2 * np.pi / 12) + np.random.normal(0, 2, size=len(dates)),
            'humidity': 0.7 + 0.1 * np.sin(hour_of_day * 2 * np.pi / 24) + np.random.normal(0, 0.05, size=len(dates)),
            'LCLid': np.random.choice(['MAC000001', 'MAC000002', 'MAC000003'], size=len(dates)),
            'Acorn_grouped': np.random.choice(['Affluent', 'Comfortable', 'Adversity'], size=len(dates))
        })
    
    def test_prepare_data_for_training(self):
        # Test data preparation function
        X_train, X_test, y_train, y_test, X_val, y_val, scaler = prepare_data_for_training(
            self.df, 
            target_col='energy(kWh/hh)',
            test_size=0.15,
            val_size=0.15,
            random_state=42
        )
        
        # Check the shapes of the outputs
        expected_n_samples = len(self.df)
        expected_n_train = int(expected_n_samples * 0.7)
        expected_n_test = int(expected_n_samples * 0.15)
        expected_n_val = int(expected_n_samples * 0.15)
        
        self.assertEqual(X_train.shape[0], expected_n_train)
        self.assertEqual(X_test.shape[0], expected_n_test)
        self.assertEqual(X_val.shape[0], expected_n_val)
        self.assertEqual(y_train.shape[0], expected_n_train)
        self.assertEqual(y_test.shape[0], expected_n_test)
        self.assertEqual(y_val.shape[0], expected_n_val)
        
        # Check that scaler was created
        self.assertIsNotNone(scaler)
    
    def test_create_lstm_model(self):
        # Create a simple sequence dataset for testing
        X = np.random.random((100, 24, 5))  # 100 samples, 24 time steps, 5 features
        
        # Test model creation
        model = create_lstm_model(
            seq_length=24,
            n_features=5,
            lstm_units=[64, 32],
            dropout_rate=0.2
        )
        
        # Check model architecture
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(len(model.layers), 6)  # Input + 2 LSTM + 2 Dropout + 1 Dense
        
        # Check model can make predictions
        preds = model.predict(X)
        self.assertEqual(preds.shape, (100, 1))
    
    def test_create_gru_model(self):
        # Create a simple sequence dataset for testing
        X = np.random.random((100, 24, 5))  # 100 samples, 24 time steps, 5 features
        
        # Test model creation
        model = create_gru_model(
            seq_length=24,
            n_features=5,
            gru_units=[64, 32],
            dropout_rate=0.2
        )
        
        # Check model architecture
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(len(model.layers), 6)  # Input + 2 GRU + 2 Dropout + 1 Dense
        
        # Check model can make predictions
        preds = model.predict(X)
        self.assertEqual(preds.shape, (100, 1))
    
    def test_create_cnn_model(self):
        # Create a simple sequence dataset for testing
        X = np.random.random((100, 24, 5))  # 100 samples, 24 time steps, 5 features
        
        # Test model creation
        model = create_cnn_model(
            seq_length=24,
            n_features=5,
            filters=[64, 32],
            kernel_size=3
        )
        
        # Check model architecture
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check model can make predictions
        preds = model.predict(X)
        self.assertEqual(preds.shape, (100, 1))
    
    def test_train_random_forest(self):
        # Create a simple dataset for testing
        X = np.random.random((100, 5))  # 100 samples, 5 features
        y = 0.5 + X[:, 0] * 0.3 + X[:, 1] * 0.2  # Simple linear relationship
        
        # Test model training
        model, mse, mae, r2 = train_random_forest(
            X_train=X[:70],
            y_train=y[:70],
            X_test=X[70:],
            y_test=y[70:],
            n_estimators=50,
            max_depth=10
        )
        
        # Check that a model was returned
        self.assertIsInstance(model, RandomForestRegressor)
        
        # Check that metrics were calculated
        self.assertIsInstance(mse, float)
        self.assertIsInstance(mae, float)
        self.assertIsInstance(r2, float)
        
        # Check that the model performs reasonably well
        self.assertLess(mse, 0.1)  # MSE should be low for this simple relationship
        self.assertGreater(r2, 0.5)  # R² should be decent
    
    @patch('src.models.train.create_lstm_model')
    @patch('src.models.train.create_gru_model')
    @patch('src.models.train.train_random_forest')
    @patch('src.models.train.prepare_data_for_training')
    @patch('src.models.train.create_sequence_dataset')
    @patch('tensorflow.keras.callbacks.EarlyStopping')
    @patch('tensorflow.keras.callbacks.ModelCheckpoint')
    @patch('src.models.train.save_model')
    def test_train_electricity_forecast_models(
        self, mock_save, mock_checkpoint, mock_early_stopping,
        mock_create_sequence, mock_prepare_data, mock_train_rf,
        mock_create_gru, mock_create_lstm
    ):
        # Configure mocks
        mock_prepare_data.return_value = (
            np.random.random((70, 5)),  # X_train
            np.random.random((15, 5)),  # X_test
            np.random.random((70, 1)),  # y_train
            np.random.random((15, 1)),  # y_test
            np.random.random((15, 5)),  # X_val
            np.random.random((15, 1)),  # y_val
            MagicMock()  # scaler
        )
        
        # Sequence dataset for RNNs
        mock_create_sequence.return_value = (
            np.random.random((70, 24, 5)),  # X_train_seq
            np.random.random((15, 24, 5)),  # X_test_seq
            np.random.random((15, 24, 5))   # X_val_seq
        )
        
        # Model mocks
        mock_lstm_model = MagicMock()
        mock_lstm_model.fit.return_value = MagicMock()
        mock_lstm_model.evaluate.return_value = [0.01, 0.08]
        mock_create_lstm.return_value = mock_lstm_model
        
        mock_gru_model = MagicMock()
        mock_gru_model.fit.return_value = MagicMock()
        mock_gru_model.evaluate.return_value = [0.015, 0.09]
        mock_create_gru.return_value = mock_gru_model
        
        # Random forest mock
        mock_train_rf.return_value = (MagicMock(), 0.02, 0.1, 0.75)
        
        # Call the function
        result = train_electricity_forecast_models(self.df)
        
        # Check that the function completed and returned a result
        self.assertIsInstance(result, dict)
        self.assertIn('best_model', result)
        
        # Check that all model types were trained
        self.assertTrue(mock_create_lstm.called)
        self.assertTrue(mock_create_gru.called)
        self.assertTrue(mock_train_rf.called)
        
        # Check that models were saved
        self.assertTrue(mock_save.called)

if __name__ == "__main__":
    unittest.main() 