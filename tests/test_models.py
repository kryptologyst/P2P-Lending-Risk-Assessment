"""Tests for P2P lending risk assessment."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.loader import P2PDataLoader
from features.engineering import FeatureEngineer
from models.credit_models import create_default_models
from risk.evaluation import CreditRiskEvaluator
from utils.helpers import set_deterministic_seeds, calculate_risk_metrics


class TestDataLoader:
    """Test cases for P2PDataLoader."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        loader = P2PDataLoader()
        df = loader.generate_synthetic_data(n_samples=100, random_state=42)
        
        assert len(df) == 100
        assert 'credit_score' in df.columns
        assert 'default' in df.columns
        assert df['credit_score'].min() >= 300
        assert df['credit_score'].max() <= 850
        assert df['default'].isin([0, 1]).all()
    
    def test_time_based_splits(self):
        """Test time-based data splitting."""
        loader = P2PDataLoader()
        df = loader.generate_synthetic_data(n_samples=100, random_state=42)
        
        train_df, val_df, test_df = loader.create_time_based_splits(df)
        
        assert len(train_df) + len(val_df) + len(test_df) == len(df)
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0


class TestFeatureEngineer:
    """Test cases for FeatureEngineer."""
    
    def test_feature_engineering(self):
        """Test feature engineering pipeline."""
        loader = P2PDataLoader()
        df = loader.generate_synthetic_data(n_samples=100, random_state=42)
        
        engineer = FeatureEngineer()
        engineered_df = engineer.engineer_all_features(df)
        
        assert len(engineered_df) == len(df)
        assert engineered_df.shape[1] > df.shape[1]  # More features after engineering
    
    def test_scaling(self):
        """Test feature scaling."""
        loader = P2PDataLoader()
        df = loader.generate_synthetic_data(n_samples=100, random_state=42)
        
        engineer = FeatureEngineer()
        engineered_df = engineer.engineer_all_features(df)
        
        X, y = loader.prepare_features(engineered_df)
        X_scaled = engineer.fit_transform_features(X)
        
        assert X_scaled.shape == X.shape
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)  # Mean should be ~0
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)   # Std should be ~1


class TestCreditModels:
    """Test cases for credit scoring models."""
    
    def test_model_creation(self):
        """Test model creation."""
        models = create_default_models(random_state=42)
        
        assert len(models) > 0
        assert 'logistic_regression' in models
        assert 'random_forest' in models
    
    def test_model_training(self):
        """Test model training."""
        # Generate test data
        loader = P2PDataLoader()
        df = loader.generate_synthetic_data(n_samples=100, random_state=42)
        
        engineer = FeatureEngineer()
        engineered_df = engineer.engineer_all_features(df)
        
        X, y = loader.prepare_features(engineered_df)
        X_scaled = engineer.fit_transform_features(X)
        
        # Train models
        models = create_default_models(random_state=42)
        
        for model_name, model in models.items():
            model.fit(X_scaled, y)
            
            # Test predictions
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_default_probability(X_scaled)
            
            assert len(y_pred) == len(y)
            assert len(y_prob) == len(y)
            assert y_pred.min() >= 0
            assert y_pred.max() <= 1
            assert y_prob.min() >= 0
            assert y_prob.max() <= 1


class TestEvaluation:
    """Test cases for evaluation metrics."""
    
    def test_risk_metrics(self):
        """Test risk metrics calculation."""
        # Generate test data
        y_true = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.2, 0.9, 0.3])
        
        metrics = calculate_risk_metrics(y_true, y_prob)
        
        assert 'auc' in metrics
        assert 'ks_statistic' in metrics
        assert 'gini_coefficient' in metrics
        assert 'threshold_metrics' in metrics
        
        assert 0 <= metrics['auc'] <= 1
        assert 0 <= metrics['ks_statistic'] <= 1
        assert -1 <= metrics['gini_coefficient'] <= 1
    
    def test_evaluator(self):
        """Test CreditRiskEvaluator."""
        evaluator = CreditRiskEvaluator()
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.2, 0.9, 0.3])
        
        results = evaluator.evaluate_model("test_model", y_true, y_pred, y_prob)
        
        assert 'model_name' in results
        assert 'ml_metrics' in results
        assert 'credit_metrics' in results
        assert 'portfolio_metrics' in results


class TestUtilities:
    """Test cases for utility functions."""
    
    def test_deterministic_seeds(self):
        """Test deterministic seed setting."""
        set_deterministic_seeds(42)
        
        # Generate two arrays with same seed
        arr1 = np.random.random(10)
        set_deterministic_seeds(42)
        arr2 = np.random.random(10)
        
        assert np.allclose(arr1, arr2)
    
    def test_risk_categories(self):
        """Test risk category creation."""
        from utils.helpers import create_risk_categories
        
        probabilities = np.array([0.1, 0.3, 0.5])
        categories = create_risk_categories(probabilities)
        
        assert categories[0] == 'Low Risk'
        assert categories[1] == 'Medium Risk'
        assert categories[2] == 'High Risk'


if __name__ == "__main__":
    pytest.main([__file__])
