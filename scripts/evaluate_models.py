#!/usr/bin/env python3
"""Evaluation script for P2P lending risk assessment models."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.loader import P2PDataLoader
from features.engineering import FeatureEngineer
from risk.evaluation import CreditRiskEvaluator
from risk.explainability import ModelInterpretability
from utils.helpers import set_deterministic_seeds

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_trained_models(models_dir: str) -> Dict[str, Any]:
    """Load trained models from directory."""
    models = {}
    models_path = Path(models_dir)
    
    if not models_path.exists():
        logger.error(f"Models directory {models_dir} does not exist")
        return models
    
    for model_file in models_path.glob("*.joblib"):
        try:
            model_data = joblib.load(model_file)
            model_name = model_file.stem
            models[model_name] = model_data['model']
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_file}: {e}")
    
    return models


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate P2P lending risk assessment models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="outputs/models",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set random seeds
    set_deterministic_seeds(config['data']['random_state'])
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "assets").mkdir(exist_ok=True)
    
    # Load trained models
    logger.info("Loading trained models...")
    models = load_trained_models(args.models_dir)
    
    if not models:
        logger.error("No trained models found. Please train models first.")
        return
    
    # Initialize components
    data_loader = P2PDataLoader(args.data_path)
    feature_engineer = FeatureEngineer(config['features']['scaler_type'])
    evaluator = CreditRiskEvaluator()
    
    # Load test data
    logger.info("Loading test data...")
    df = data_loader.load_data()
    
    # Create time-based splits
    train_df, val_df, test_df = data_loader.create_time_based_splits(
        df,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        random_state=config['data']['random_state']
    )
    
    # Feature engineering
    logger.info("Engineering features...")
    train_features = feature_engineer.engineer_all_features(
        train_df,
        include_interactions=config['features']['include_interactions'],
        include_polynomials=config['features']['include_polynomials'],
        include_binning=config['features']['include_binning'],
        include_time=config['features']['include_time'],
        polynomial_degree=config['features']['polynomial_degree']
    )
    
    test_features = feature_engineer.engineer_all_features(
        test_df,
        include_interactions=config['features']['include_interactions'],
        include_polynomials=config['features']['include_polynomials'],
        include_binning=config['features']['include_binning'],
        include_time=config['features']['include_time'],
        polynomial_degree=config['features']['polynomial_degree']
    )
    
    # Prepare features and targets
    X_train, y_train = data_loader.prepare_features(train_features)
    X_test, y_test = data_loader.prepare_features(test_features)
    
    # Scale features
    logger.info("Scaling features...")
    X_train_scaled = feature_engineer.fit_transform_features(X_train)
    X_test_scaled = feature_engineer.transform_features(X_test)
    
    # Evaluate models
    logger.info("Evaluating models...")
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_default_probability(X_test_scaled)
        
        # Get loan amounts for portfolio metrics
        loan_amounts = test_df['loan_amount'].values if 'loan_amount' in test_df.columns else None
        
        # Evaluate model
        results = evaluator.evaluate_model(
            model_name,
            y_test.values,
            y_pred,
            y_prob,
            loan_amounts,
            config['evaluation']['thresholds']
        )
        
        logger.info(f"Completed evaluation for {model_name}")
    
    # Create model comparison
    logger.info("Creating model comparison...")
    comparison_df = evaluator.create_model_comparison_table()
    comparison_path = output_dir / "assets" / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Saved model comparison to {comparison_path}")
    
    # Generate evaluation report
    logger.info("Generating evaluation report...")
    report_path = output_dir / "assets" / "evaluation_report.txt"
    evaluator.generate_evaluation_report(str(report_path))
    
    # Model explainability
    logger.info("Analyzing model explainability...")
    
    # Create interpretability analysis
    interpretability = ModelInterpretability(models, X_train_scaled.columns.tolist())
    
    # Compare feature importance
    importance_comparison = interpretability.compare_feature_importance(
        X_test_scaled,
        top_n=config['explainability']['top_n_features'],
        save_path=str(output_dir / "assets" / "feature_importance_comparison.png")
    )
    
    # Save feature importance
    if not importance_comparison.empty:
        importance_path = output_dir / "assets" / "feature_importance_comparison.csv"
        importance_comparison.to_csv(importance_path)
        logger.info(f"Saved feature importance comparison to {importance_path}")
    
    # Create individual model explanations
    for model_name, model in models.items():
        explainer = interpretability.explainers.get(model_name)
        if explainer:
            # Calculate SHAP values
            explainer.calculate_shap_values(
                X_test_scaled, 
                max_samples=config['explainability']['max_samples_shap']
            )
            
            # Plot feature importance
            explainer.plot_feature_importance(
                top_n=config['explainability']['top_n_features'],
                save_path=str(output_dir / "assets" / f"{model_name}_feature_importance.png")
            )
            
            # Create explanation report for a sample
            explainer.create_explanation_report(
                X_test_scaled,
                sample_idx=0,
                output_path=str(output_dir / "assets" / f"{model_name}_explanation.txt")
            )
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to {output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("P2P LENDING RISK ASSESSMENT - EVALUATION SUMMARY")
    print("="*60)
    print(f"Models evaluated: {list(models.keys())}")
    print(f"Test samples: {len(X_test_scaled)}")
    print(f"Features: {X_test_scaled.shape[1]}")
    print(f"Default rate: {y_test.mean():.2%}")
    print("\nModel Performance (AUC):")
    for model_name, model in models.items():
        y_prob = model.predict_default_probability(X_test_scaled)
        auc = evaluator.calculate_ml_metrics(y_test.values, model.predict(X_test_scaled), y_prob)['auc_roc']
        print(f"  {model_name}: {auc:.4f}")
    print("\n" + "="*60)
    print("DISCLAIMER: This is a research demonstration project only.")
    print("NOT intended for investment advice or commercial use.")
    print("="*60)


if __name__ == "__main__":
    main()
