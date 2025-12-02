import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import mlflow
import mlflow.sklearn
import joblib
import json
import logging
from typing import Dict, Any, Tuple
import os

# create logger
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train regression models for salary prediction"""
    
    def __init__(self, experiment_name: str = "salary_prediction"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf

    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
        """Train Linear Regression model"""
        with mlflow.start_run(run_name="linear_regression"):
            # create and train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Log metrics
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            mlflow.log_metric("cv_r2_mean", scores.mean())
            mlflow.log_metric("cv_r2_std", scores.std())

            # Log parameters
            mlflow.log_param("model_type", "linear_regression")
            mlflow.log_param("cv_folds", 5)

            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            self.models['linear_regression'] = {
                'model': model,
                'cv_score': scores.mean()
            }

            if scores.mean() > self.best_score:
                self.best_score = scores.mean()
                self.best_model = model
            
            return model
        
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """Train Random Forest model with hyperparameter tuning"""      
        with mlflow.start_run(run_name="random_forest"):
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }

            # Perform grid search
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='r2', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            # Log best parameters
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(param, value)

            # Log metrics
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            mlflow.log_metric("r2_train", grid_search.score(X_train, y_train))
            
            # Log model
            mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
            
            self.models['random_forest'] = {
                'model': grid_search.best_estimator_,
                'cv_score': grid_search.best_score_,
                'best_params': grid_search.best_params_
            }
            
            if grid_search.best_score_ > self.best_score:
                self.best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
            
            return grid_search.best_estimator_
        
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train all models and select the best one"""
        logger.info("Training Linear Regression...")
        self.train_linear_regression(X_train, y_train)
        
        logger.info("Training Random Forest...")
        self.train_random_forest(X_train, y_train)
        
        # Select best model
        best_model_name = max(self.models.items(), key=lambda x: x[1]['cv_score'])[0]
        logger.info(f"Best model: {best_model_name} with CV R²: {self.best_score:.3f}")
        
        return {
            'best_model': self.best_model,
            'best_model_name': best_model_name,
            'best_score': self.best_score,
            'all_models': self.models
        }
    
    def save_model(self, model, filepath: str, preprocessor=None) -> None:
        """Save model and metadata"""
        # Save model
        joblib.dump(model, filepath)
        
        # Save metadata
        metadata = {
            'model_type': type(model).__name__,
            'training_date': pd.Timestamp.now().isoformat(),
            'features': ['YearsExperience'],
            'target': 'Salary'
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save preprocessor if provided
        if preprocessor:
            preprocessor_path = filepath.replace('.pkl', '_preprocessor.pkl')
            preprocessor.save(preprocessor_path)
        
        logger.info(f"Model saved to {filepath}")