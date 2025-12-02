#!/usr/bin/env python3
import json
import sys

def evaluate_model(report_path, r2_threshold=0.8, mape_threshold=15):
    """Evaluate model against thresholds."""
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        metrics = report['metrics']
        
        print('📊 Model Evaluation:')
        print(f'R² Score: {metrics["r2"]:.3f}')
        print(f'RMSE: {metrics["rmse"]:.2f}')
        print(f'MAE: {metrics["mae"]:.2f}')
        print(f'MAPE: {metrics["mape"]:.2f}%')
        
        # Check thresholds
        passed = True
        if metrics['r2'] < r2_threshold:
            print(f'R² below threshold ({r2_threshold})')
            passed = False
        if metrics['mape'] > mape_threshold:
            print(f'MAPE above threshold ({mape_threshold}%)')
            passed = False
        
        if passed:
            print('✅ Model meets quality requirements!')
        
        return passed
        
    except Exception as e:
        print(f'❌ Evaluation failed: {e}')
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py <evaluation_report.json>")
        sys.exit(1)
    
    success = evaluate_model(sys.argv[1])
    sys.exit(0 if success else 1)