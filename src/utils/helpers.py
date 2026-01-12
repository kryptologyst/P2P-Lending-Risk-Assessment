"""Utility functions for P2P lending risk assessment."""

import logging
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def set_deterministic_seeds(seed: int = 42) -> None:
    """Set deterministic seeds for all random number generators.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Set additional seeds for other libraries if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    logger.info(f"Set deterministic seeds to {seed}")


def detect_device() -> str:
    """Detect the best available device for computation.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    
    return 'cpu'


def calculate_risk_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> Dict[str, float]:
    """Calculate comprehensive risk metrics.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        thresholds: List of probability thresholds to evaluate
        
    Returns:
        Dictionary of risk metrics
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    from sklearn.metrics import roc_auc_score, roc_curve
    
    # Basic metrics
    auc = roc_auc_score(y_true, y_prob)
    
    # KS Statistic
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_statistic = np.max(tpr - fpr)
    
    # Gini Coefficient
    gini_coefficient = 2 * auc - 1
    
    # Threshold-based metrics
    threshold_metrics = {}
    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        
        tn, fp, fn, tp = np.bincount(
            y_true * 2 + y_pred_thresh, minlength=4
        ).reshape(2, 2).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr_thresh = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        threshold_metrics[f'threshold_{threshold}'] = {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'fpr': fpr_thresh
        }
    
    return {
        'auc': auc,
        'ks_statistic': ks_statistic,
        'gini_coefficient': gini_coefficient,
        'threshold_metrics': threshold_metrics
    }


def validate_data_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = 'application_date'
) -> bool:
    """Validate that there's no data leakage between train and test sets.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        date_col: Name of the date column
        
    Returns:
        True if no leakage detected, False otherwise
    """
    if date_col not in train_df.columns or date_col not in test_df.columns:
        logger.warning(f"Date column '{date_col}' not found in data")
        return True
    
    train_max_date = pd.to_datetime(train_df[date_col]).max()
    test_min_date = pd.to_datetime(test_df[date_col]).min()
    
    if train_max_date >= test_min_date:
        logger.error("Data leakage detected! Training data contains samples after test data.")
        return False
    
    logger.info("No data leakage detected")
    return True


def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary of features in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with feature summary
    """
    summary_data = []
    
    for col in df.columns:
        col_data = df[col]
        
        summary = {
            'feature': col,
            'dtype': str(col_data.dtype),
            'missing_count': col_data.isnull().sum(),
            'missing_pct': col_data.isnull().sum() / len(df) * 100,
            'unique_count': col_data.nunique(),
            'unique_pct': col_data.nunique() / len(df) * 100
        }
        
        if col_data.dtype in ['int64', 'float64']:
            summary.update({
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'median': col_data.median()
            })
        else:
            summary.update({
                'min': None,
                'max': None,
                'mean': None,
                'std': None,
                'median': None
            })
        
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)


def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format amount as currency string.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == 'USD':
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string.
    
    Args:
        value: Value to format (0-1 range)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_loan_metrics(
    loan_amount: float,
    interest_rate: float,
    loan_term_months: int
) -> Dict[str, float]:
    """Calculate loan-related metrics.
    
    Args:
        loan_amount: Principal loan amount
        interest_rate: Annual interest rate
        loan_term_months: Loan term in months
        
    Returns:
        Dictionary of loan metrics
    """
    monthly_rate = interest_rate / 12
    
    if monthly_rate == 0:
        monthly_payment = loan_amount / loan_term_months
    else:
        monthly_payment = (
            loan_amount * 
            (monthly_rate * (1 + monthly_rate) ** loan_term_months) /
            ((1 + monthly_rate) ** loan_term_months - 1)
        )
    
    total_payment = monthly_payment * loan_term_months
    total_interest = total_payment - loan_amount
    
    return {
        'monthly_payment': monthly_payment,
        'total_payment': total_payment,
        'total_interest': total_interest,
        'interest_rate_monthly': monthly_rate,
        'loan_to_value_ratio': loan_amount / loan_amount  # Placeholder
    }


def create_risk_categories(probabilities: np.ndarray) -> np.ndarray:
    """Create risk categories from probabilities.
    
    Args:
        probabilities: Array of default probabilities
        
    Returns:
        Array of risk categories
    """
    categories = np.zeros_like(probabilities, dtype=object)
    
    categories[probabilities < 0.2] = 'Low Risk'
    categories[(probabilities >= 0.2) & (probabilities < 0.4)] = 'Medium Risk'
    categories[probabilities >= 0.4] = 'High Risk'
    
    return categories


def log_model_info(model: Any, model_name: str) -> None:
    """Log information about a trained model.
    
    Args:
        model: Trained model object
        model_name: Name of the model
    """
    logger.info(f"Model: {model_name}")
    logger.info(f"Model type: {type(model).__name__}")
    
    if hasattr(model, 'feature_importances_'):
        logger.info(f"Number of features: {len(model.feature_importances_)}")
        logger.info(f"Top 5 features: {model.feature_importances_.argsort()[-5:][::-1]}")
    
    if hasattr(model, 'coef_'):
        logger.info(f"Number of coefficients: {len(model.coef_[0])}")
        logger.info(f"Largest coefficient: {np.max(np.abs(model.coef_[0])):.4f}")


def create_performance_summary(results: Dict[str, Any]) -> str:
    """Create a text summary of model performance results.
    
    Args:
        results: Dictionary of evaluation results
        
    Returns:
        Formatted performance summary string
    """
    summary_lines = []
    
    summary_lines.append("Model Performance Summary")
    summary_lines.append("=" * 30)
    
    if 'ml_metrics' in results:
        ml_metrics = results['ml_metrics']
        summary_lines.append(f"AUC: {ml_metrics.get('auc_roc', 0):.4f}")
        summary_lines.append(f"F1 Score: {ml_metrics.get('f1_score', 0):.4f}")
        summary_lines.append(f"Precision: {ml_metrics.get('precision', 0):.4f}")
        summary_lines.append(f"Recall: {ml_metrics.get('recall', 0):.4f}")
    
    if 'credit_metrics' in results:
        credit_metrics = results['credit_metrics']
        summary_lines.append(f"KS Statistic: {credit_metrics.get('ks_statistic', 0):.4f}")
        summary_lines.append(f"Gini Coefficient: {credit_metrics.get('gini_coefficient', 0):.4f}")
    
    return "\n".join(summary_lines)
