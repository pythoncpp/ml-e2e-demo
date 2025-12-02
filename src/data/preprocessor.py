import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocess salary data"""

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit scalers on training data"""
        self.scaler_X.fit(X)
        self.scaler_y.fit(y.values.reshape(-1, 1))
        self.is_fitted = True
        logger.info("Scalers fitted successfully")

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple:
        """Transform data using fitted scalers"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        X_scaled = self.scaler_X.transform(X)
        
        if y is not None:
            y_scaled = self.scaler_y.transform(y.values.reshape(-1, 1)).flatten()
            return X_scaled, y_scaled
        return X_scaled    

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled predictions"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def save(self, filepath: str) -> None:
        """Save preprocessor to disk"""
        joblib.dump({
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """Load preprocessor from disk"""
        preprocessor = cls()
        saved_data = joblib.load(filepath)
        preprocessor.scaler_X = saved_data['scaler_X']
        preprocessor.scaler_y = saved_data['scaler_y']
        preprocessor.is_fitted = True
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor