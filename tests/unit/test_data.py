import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from data.loader import DataLoader
from data.preprocessor import DataPreprocessor


class TestDataLoader:
    def test_load_data(self, tmp_path):
        # Create test CSV
        data = pd.DataFrame({
            'YearsExperience': [1.0, 2.0, 3.0],
            'Salary': [50000, 60000, 70000]
        })
        test_file = tmp_path / "test.csv"
        data.to_csv(test_file, index=False)
        
        loader = DataLoader(str(test_file))
        loaded_data = loader.load_data()
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == (3, 2)
        assert list(loaded_data.columns) == ['YearsExperience', 'Salary']
    
    def test_validate_data_valid(self, tmp_path):
        data = pd.DataFrame({
            'YearsExperience': [1.0, 2.0, 3.0],
            'Salary': [50000, 60000, 70000]
        })
        test_file = tmp_path / "test.csv"
        data.to_csv(test_file, index=False)
        
        loader = DataLoader(str(test_file))
        loader.load_data()
        assert loader.validate_data() == True
    
    def test_validate_data_missing_column(self, tmp_path):
        data = pd.DataFrame({
            'YearsExperience': [1.0, 2.0, 3.0]
            # Missing Salary column
        })
        test_file = tmp_path / "test.csv"
        data.to_csv(test_file, index=False)
        
        loader = DataLoader(str(test_file))
        loader.load_data()
        
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.validate_data()


class TestDataPreprocessor:
    def test_fit_transform(self):
        preprocessor = DataPreprocessor()
        X = pd.DataFrame({'YearsExperience': [1, 2, 3, 4, 5]})
        y = pd.Series([50000, 60000, 70000, 80000, 90000])
        
        preprocessor.fit(X, y)
        X_scaled, y_scaled = preprocessor.transform(X, y)
        
        assert X_scaled.shape == (5, 1)
        assert y_scaled.shape == (5,)
        
        # Check if scaled data has mean ~0 and std ~1
        assert np.allclose(X_scaled.mean(), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(), 1, atol=1e-10)
    
    def test_inverse_transform(self):
        preprocessor = DataPreprocessor()
        X = pd.DataFrame({'YearsExperience': [1, 2, 3]})
        y = pd.Series([50000, 60000, 70000])
        
        preprocessor.fit(X, y)
        X_scaled, y_scaled = preprocessor.transform(X, y)
        y_original = preprocessor.inverse_transform_y(y_scaled)
        
        assert np.allclose(y_original, y.values, rtol=1e-5)
