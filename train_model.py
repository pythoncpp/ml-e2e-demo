import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.train import ModelTrainer
from src.evaluation.metrics import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train salary prediction model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    
    # get args from caller
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    try:
        # Step 1: Load data
        logger.info("Loading data...")
        loader = DataLoader(args.data_path)
        data = loader.load_data()
        loader.validate_data()
        
        # Step 2: Split data
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = loader.train_test_split(
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Step 3: Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()
        preprocessor.fit(X_train, y_train)
        X_train_scaled, y_train_scaled = preprocessor.transform(X_train, y_train)
        
         # Step 4: Train models
        logger.info("Training models...")
        trainer = ModelTrainer(experiment_name="salary_prediction")
        training_result = trainer.train_all_models(X_train_scaled, y_train_scaled)
        
        # Step 5: Evaluate best model
        logger.info("Evaluating best model...")
        best_model = training_result['best_model']
        evaluator = ModelEvaluator(best_model, preprocessor)
        metrics = evaluator.evaluate(X_test, y_test)

        # Step 6: Save artifacts
        logger.info("Saving artifacts...")

        # Save best model
        model_path = os.path.join(args.output_dir, 'best_model.pkl')
        trainer.save_model(best_model, model_path, preprocessor)

        # Save evaluation report
        evaluator.save_evaluation_report(
            metrics,
            os.path.join('reports', 'evaluation_report.json')
        )
        
        # Create residual plot
        evaluator.create_residual_plot(
            X_test, y_test,
            os.path.join('reports', 'residual_plot.png')
        )

        # Step 7: Print results
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETE")
        logger.info("="*50)
        logger.info(f"Best Model: {training_result['best_model_name']}")
        logger.info(f"CV R² Score: {training_result['best_score']:.3f}")
        logger.info(f"Test R² Score: {metrics['r2']:.3f}")
        logger.info(f"Test RMSE: {metrics['rmse']:.2f}")
        logger.info(f"Test MAE: {metrics['mae']:.2f}")
        logger.info(f"Test MAPE: {metrics['mape']:.2f}%")
        logger.info(f"Model saved to: {model_path}")
        logger.info("="*50)
        
    except:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()