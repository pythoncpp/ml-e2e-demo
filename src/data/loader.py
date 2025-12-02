import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

# initialize logging
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and split salary data"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(self.filepath)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self) -> bool:
        """Validate data schema and quality"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Check required columns
        required_cols = ['YearsExperience', 'Salary']
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")
        
        # Check for null values
        if self.data.isnull().sum().any():
            raise ValueError("Data contains null values")
        
        # Check data types
        if not np.issubdtype(self.data['YearsExperience'].dtype, np.number):
            raise ValueError("YearsExperience must be numeric")
        if not np.issubdtype(self.data['Salary'].dtype, np.number):
            raise ValueError("Salary must be numeric")
        
        # Check for outliers (optional)
        salary_q1 = self.data['Salary'].quantile(0.25)
        salary_q3 = self.data['Salary'].quantile(0.75)
        salary_iqr = salary_q3 - salary_q1
        outliers = self.data[
            (self.data['Salary'] < salary_q1 - 1.5 * salary_iqr) |
            (self.data['Salary'] > salary_q3 + 1.5 * salary_iqr)
        ]
        
        if len(outliers) > 0:
            logger.warning(f"Found {len(outliers)} potential outliers in Salary")
        
        return True
    
    def train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into train and test sets"""
        from sklearn.model_selection import train_test_split
        
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        X = self.data[['YearsExperience']]
        y = self.data['Salary']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test