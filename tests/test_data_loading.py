import unittest
import os
import pandas as pd
from src.data.load_data import load_with_retry, validate_file
import tempfile

class TestDataLoading(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary CSV file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = os.path.join(self.temp_dir.name, "test.csv")
        
        # Create a simple test dataframe and save it
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        df.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        # Clean up temporary files
        self.temp_dir.cleanup()
    
    def test_validate_file(self):
        # Test file validation
        self.assertTrue(validate_file(self.test_file))
        
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            validate_file("non_existent_file.csv")
    
    def test_load_with_retry(self):
        # Test loading a valid CSV file
        df = load_with_retry(self.test_file)
        
        # Check that the data was loaded correctly
        self.assertEqual(len(df), 3)
        self.assertEqual(df['id'].tolist(), [1, 2, 3])
        self.assertEqual(df['value'].tolist(), [10, 20, 30])

if __name__ == "__main__":
    unittest.main() 