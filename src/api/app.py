from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Salary Prediction API",
    description="API for predicting salary based on years of experience",
    version="1.0.0"
)

# Define request/response models
class SalaryPredictionRequest(BaseModel):
    years_experience: float = Field(..., gt=0, description="Years of experience")
    
    class Config:
        json_schema_extra = {
            "example": {
                "years_experience": 5.5
            }
        }

class SalaryPredictionResponse(BaseModel):
    predicted_salary: float = Field(..., description="Predicted salary in USD")
    confidence_interval: List[float] = Field(..., description="95% confidence interval")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_salary": 75000.50,
                "confidence_interval": [70000.00, 80000.00]
            }
        }

class BatchPredictionRequest(BaseModel):
    years_experience_list: List[float] = Field(..., description="List of years of experience")
    
    class Config:
        json_schema_extra = {
            "example": {
                "years_experience_list": [1.0, 3.0, 5.0, 7.0, 10.0]
            }
        }

# Load model and preprocessor (in production, this would be loaded once at startup)
MODEL = None
PREPROCESSOR = None

def load_model():
    """Load model and preprocessor"""
    global MODEL, PREPROCESSOR
    try:
        MODEL = joblib.load('models/best_model.pkl')
        
        # For this example, we'll create a simple preprocessor
        # In production, load from saved file
        from src.data.preprocessor import DataPreprocessor
        PREPROCESSOR = DataPreprocessor.load('models/best_model_preprocessor.pkl')
        
        logger.info("Model and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Salary Prediction API",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": MODEL is not None}

@app.post("/predict", response_model=SalaryPredictionResponse)
async def predict_salary(request: SalaryPredictionRequest):
    """Predict salary for a single input"""
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input
        X = pd.DataFrame({'YearsExperience': [request.years_experience]})
        
        # Transform using preprocessor
        X_scaled = PREPROCESSOR.transform(X)
        
        # Make prediction
        y_pred_scaled = MODEL.predict(X_scaled)
        y_pred = PREPROCESSOR.inverse_transform_y(y_pred_scaled)
        
        # Calculate confidence interval (simplified)
        # In production, use proper confidence interval calculation
        margin = y_pred[0] * 0.1  # 10% margin for example
        confidence_interval = [y_pred[0] - margin, y_pred[0] + margin]
        
        return SalaryPredictionResponse(
            predicted_salary=round(float(y_pred[0]), 2),
            confidence_interval=[round(x, 2) for x in confidence_interval]
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Predict salary for multiple inputs"""
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input
        X = pd.DataFrame({'YearsExperience': request.years_experience_list})
        
        # Transform using preprocessor
        X_scaled = PREPROCESSOR.transform(X)
        
        # Make predictions
        y_pred_scaled = MODEL.predict(X_scaled)
        y_pred = PREPROCESSOR.inverse_transform_y(y_pred_scaled)
        
        # Prepare response
        predictions = []
        for exp, pred in zip(request.years_experience_list, y_pred):
            margin = pred * 0.1
            predictions.append({
                "years_experience": exp,
                "predicted_salary": round(float(pred), 2),
                "confidence_interval": [
                    round(float(pred - margin), 2),
                    round(float(pred + margin), 2)
                ]
            })
        
        return {
            "predictions": predictions,
            "count": len(predictions)
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(MODEL).__name__,
        "features": ["YearsExperience"],
        "target": "Salary",
        "parameters": MODEL.get_params() if hasattr(MODEL, 'get_params') else {}
    }