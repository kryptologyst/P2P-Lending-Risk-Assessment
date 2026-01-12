"""Evaluation metrics and model assessment for credit scoring."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class CreditRiskEvaluator:
    """Comprehensive evaluator for credit risk models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
    
    def calculate_ml_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate standard machine learning metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities for positive class
            
        Returns:
            Dictionary of ML metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_prob),
            'auc_pr': average_precision_score(y_true, y_prob),
            'log_loss': log_loss(y_true, y_prob),
            'brier_score': brier_score_loss(y_true, y_prob)
        }
        
        logger.info("Calculated ML metrics")
        return metrics
    
    def calculate_credit_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, Union[float, Dict]]:
        """Calculate credit-specific risk metrics.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            thresholds: List of probability thresholds to evaluate
            
        Returns:
            Dictionary of credit risk metrics
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # KS Statistic
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ks_statistic = np.max(tpr - fpr)
        
        # Gini Coefficient
        auc_roc = roc_auc_score(y_true, y_prob)
        gini_coefficient = 2 * auc_roc - 1
        
        # Population Stability Index (PSI) - simplified version
        psi_score = self._calculate_psi(y_prob, y_true)
        
        # Threshold-based metrics
        threshold_metrics = {}
        for threshold in thresholds:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
            
            threshold_metrics[f'threshold_{threshold}'] = {
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'f1_score': f1_score(y_true, y_pred_thresh, zero_division=0),
                'accuracy': accuracy_score(y_true, y_pred_thresh)
            }
        
        # Calibration metrics
        calibration_score = self._calculate_calibration_score(y_true, y_prob)
        
        credit_metrics = {
            'ks_statistic': ks_statistic,
            'gini_coefficient': gini_coefficient,
            'psi_score': psi_score,
            'calibration_score': calibration_score,
            'threshold_metrics': threshold_metrics
        }
        
        logger.info("Calculated credit-specific metrics")
        return credit_metrics
    
    def _calculate_psi(self, y_prob: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate Population Stability Index.
        
        Args:
            y_prob: Predicted probabilities
            y_true: True labels
            
        Returns:
            PSI score
        """
        # Create bins for PSI calculation
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        
        # Calculate expected and actual distributions
        expected_dist = np.bincount(bin_indices, minlength=len(bins)-1)
        expected_dist = expected_dist / np.sum(expected_dist)
        
        # Add small value to avoid division by zero
        expected_dist = np.maximum(expected_dist, 1e-6)
        
        # Calculate PSI
        psi = 0
        for i in range(len(expected_dist)):
            if expected_dist[i] > 0:
                actual_prob = expected_dist[i]
                expected_prob = expected_dist[i]  # Simplified for demo
                psi += (actual_prob - expected_prob) * np.log(actual_prob / expected_prob)
        
        return psi
    
    def _calculate_calibration_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate calibration score using Brier score.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Calibration score (lower is better)
        """
        return brier_score_loss(y_true, y_prob)
    
    def calculate_portfolio_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        loan_amounts: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            loan_amounts: Loan amounts for each borrower
            
        Returns:
            Dictionary of portfolio metrics
        """
        if loan_amounts is None:
            loan_amounts = np.ones(len(y_true))  # Assume equal amounts
        
        # Expected loss
        expected_loss = np.sum(y_prob * loan_amounts)
        total_exposure = np.sum(loan_amounts)
        expected_loss_rate = expected_loss / total_exposure
        
        # Actual loss
        actual_loss = np.sum(y_true * loan_amounts)
        actual_loss_rate = actual_loss / total_exposure
        
        # Loss given default (simplified)
        lgd = actual_loss / np.sum(y_true) if np.sum(y_true) > 0 else 0
        
        # Concentration risk (Herfindahl index)
        portfolio_weights = loan_amounts / total_exposure
        concentration_index = np.sum(portfolio_weights ** 2)
        
        # Value at Risk (VaR) - 95th percentile of losses
        loss_scenarios = y_prob * loan_amounts
        var_95 = np.percentile(loss_scenarios, 95)
        
        portfolio_metrics = {
            'expected_loss': expected_loss,
            'actual_loss': actual_loss,
            'expected_loss_rate': expected_loss_rate,
            'actual_loss_rate': actual_loss_rate,
            'loss_given_default': lgd,
            'concentration_index': concentration_index,
            'var_95': var_95,
            'total_exposure': total_exposure
        }
        
        logger.info("Calculated portfolio-level metrics")
        return portfolio_metrics
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        loan_amounts: Optional[np.ndarray] = None,
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, Union[float, Dict]]:
        """Comprehensive model evaluation.
        
        Args:
            model_name: Name of the model
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities for positive class
            loan_amounts: Loan amounts for each borrower
            thresholds: List of probability thresholds to evaluate
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            'model_name': model_name,
            'ml_metrics': self.calculate_ml_metrics(y_true, y_pred, y_prob),
            'credit_metrics': self.calculate_credit_metrics(y_true, y_prob, thresholds),
            'portfolio_metrics': self.calculate_portfolio_metrics(y_true, y_prob, loan_amounts)
        }
        
        self.results[model_name] = results
        logger.info(f"Completed evaluation for {model_name}")
        return results
    
    def create_model_comparison_table(self) -> pd.DataFrame:
        """Create a comparison table of all evaluated models.
        
        Returns:
            DataFrame with model comparison metrics
        """
        if not self.results:
            logger.warning("No evaluation results available")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {
                'Model': model_name,
                'AUC': results['ml_metrics']['auc_roc'],
                'KS': results['credit_metrics']['ks_statistic'],
                'Gini': results['credit_metrics']['gini_coefficient'],
                'Brier': results['ml_metrics']['brier_score'],
                'F1': results['ml_metrics']['f1_score'],
                'Precision': results['ml_metrics']['precision'],
                'Recall': results['ml_metrics']['recall']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC', ascending=False)
        
        logger.info("Created model comparison table")
        return comparison_df
    
    def plot_roc_curves(self, save_path: Optional[str] = None) -> None:
        """Plot ROC curves for all evaluated models.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            # This is a simplified version - in practice, you'd need the actual predictions
            auc = results['ml_metrics']['auc_roc']
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.text(0.6, 0.1 + len(self.results) * 0.05, 
                    f'{model_name}: AUC = {auc:.3f}', fontsize=10)
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curves plot to {save_path}")
        
        plt.show()
    
    def plot_calibration_curves(self, save_path: Optional[str] = None) -> None:
        """Plot calibration curves for all evaluated models.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            # This is a simplified version - in practice, you'd need the actual predictions
            brier_score = results['ml_metrics']['brier_score']
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.text(0.6, 0.1 + len(self.results) * 0.05, 
                    f'{model_name}: Brier = {brier_score:.3f}', fontsize=10)
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved calibration curves plot to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self,
        feature_importance: pd.DataFrame,
        top_n: int = 15,
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance comparison.
        
        Args:
            feature_importance: DataFrame with feature importance scores
            top_n: Number of top features to show
            save_path: Optional path to save the plot
        """
        if feature_importance.empty:
            logger.warning("No feature importance data available")
            return
        
        # Get top features
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        for col in top_features.columns:
            plt.plot(top_features.index, top_features[col], 
                    marker='o', label=col, linewidth=2)
        
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, output_path: str = "evaluation_report.txt") -> None:
        """Generate a comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write("P2P Lending Risk Assessment - Model Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("DISCLAIMER: This is a research demonstration project only.\n")
            f.write("NOT intended for investment advice or commercial use.\n\n")
            
            if not self.results:
                f.write("No evaluation results available.\n")
                return
            
            # Model comparison table
            comparison_df = self.create_model_comparison_table()
            f.write("Model Performance Comparison:\n")
            f.write("-" * 40 + "\n")
            f.write(comparison_df.to_string(index=False, float_format='%.4f'))
            f.write("\n\n")
            
            # Detailed results for each model
            for model_name, results in self.results.items():
                f.write(f"Detailed Results - {model_name}:\n")
                f.write("-" * 40 + "\n")
                
                # ML Metrics
                f.write("Machine Learning Metrics:\n")
                for metric, value in results['ml_metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")
                
                # Credit Metrics
                f.write("\nCredit Risk Metrics:\n")
                credit_metrics = results['credit_metrics']
                f.write(f"  KS Statistic: {credit_metrics['ks_statistic']:.4f}\n")
                f.write(f"  Gini Coefficient: {credit_metrics['gini_coefficient']:.4f}\n")
                f.write(f"  Calibration Score: {credit_metrics['calibration_score']:.4f}\n")
                
                # Portfolio Metrics
                f.write("\nPortfolio Metrics:\n")
                portfolio_metrics = results['portfolio_metrics']
                f.write(f"  Expected Loss Rate: {portfolio_metrics['expected_loss_rate']:.4f}\n")
                f.write(f"  Actual Loss Rate: {portfolio_metrics['actual_loss_rate']:.4f}\n")
                f.write(f"  Concentration Index: {portfolio_metrics['concentration_index']:.4f}\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
        
        logger.info(f"Generated evaluation report: {output_path}")


def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Kolmogorov-Smirnov statistic.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        
    Returns:
        KS statistic
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return np.max(tpr - fpr)


def calculate_gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Gini coefficient.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        
    Returns:
        Gini coefficient
    """
    auc_roc = roc_auc_score(y_true, y_prob)
    return 2 * auc_roc - 1


def calculate_lift_score(y_true: np.ndarray, y_prob: np.ndarray, percentile: float = 0.1) -> float:
    """Calculate lift score at given percentile.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        percentile: Percentile threshold (e.g., 0.1 for top 10%)
        
    Returns:
        Lift score
    """
    threshold = np.percentile(y_prob, (1 - percentile) * 100)
    top_percentile_mask = y_prob >= threshold
    
    overall_rate = np.mean(y_true)
    top_percentile_rate = np.mean(y_true[top_percentile_mask])
    
    return top_percentile_rate / overall_rate if overall_rate > 0 else 0
