#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys

def validate_data(filepath):
    """Validate the salary data file."""
    try:
        df = pd.read_csv(filepath)
        print(f'Data shape: {df.shape}')
        print(f'Columns: {list(df.columns)}')
        print(f'Missing values:\n{df.isnull().sum()}')
        
        # Validate
        assert 'YearsExperience' in df.columns, 'Missing YearsExperience column'
        assert 'Salary' in df.columns, 'Missing Salary column'
        assert df['YearsExperience'].min() > 0, 'YearsExperience must be positive'
        assert df['Salary'].min() > 0, 'Salary must be positive'
        assert not df.isnull().any().any(), 'Data contains null values'
        
        # Check for reasonable values
        if df['YearsExperience'].max() > 50:
            print('⚠️  Warning: YearsExperience seems unusually high')
        if df['Salary'].max() > 1000000:
            print('⚠️  Warning: Salary seems unusually high')
        
        print('✅ Data validation passed!')
        return True
        
    except Exception as e:
        print(f'❌ Data validation failed: {e}')
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_data.py <data_file.csv>")
        sys.exit(1)
    
    success = validate_data(sys.argv[1])
    sys.exit(0 if success else 1)