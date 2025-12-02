import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import logging
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate regression model performance"""
    
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        # Transform test data
        X_test_scaled = self.preprocessor.transform(X_test)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.preprocessor.inverse_transform_y(y_pred_scaled)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': self._calculate_mape(y_test, y_pred)
        }
        
        logger.info(f"Test Metrics: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.2f}")
        return metrics
    
    def _calculate_mape(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
   
    def create_residual_plot(self, X_test: pd.DataFrame, y_test: pd.Series, save_path: str) -> None:
        """Create and save residual plot"""
        # Make predictions
        X_test_scaled = self.preprocessor.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.preprocessor.inverse_transform_y(y_pred_scaled)
        
        # Calculate residuals
        residuals = y_test - y_pred
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.7)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        
        # QQ Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Residual plot saved to {save_path}")

    def save_evaluation_report(self, metrics: Dict[str, float], save_path: str) -> None:
        """Save evaluation metrics to JSON file"""
        report = {
            'metrics': metrics,
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'interpretation': {
                'r2': 'Higher is better (max 1.0)',
                'rmse': 'Lower is better',
                'mape': 'Lower is better (percentage error)'
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {save_path}")