"""Model explainability and interpretability for credit scoring."""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

logger = logging.getLogger(__name__)


class CreditModelExplainer:
    """Explainability module for credit scoring models."""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize the explainer.
        
        Args:
            model: Trained credit scoring model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        if SHAP_AVAILABLE:
            self._create_explainer()
    
    def _create_explainer(self) -> None:
        """Create SHAP explainer based on model type."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping explainer creation")
            return
        
        try:
            # Determine explainer type based on model
            if hasattr(self.model, 'predict_proba'):
                # Tree-based models
                if hasattr(self.model, 'feature_importances_'):
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    # Linear models
                    self.explainer = shap.LinearExplainer(self.model, self.model.coef_)
            else:
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(self.model.predict_proba, self.model)
            
            logger.info("Created SHAP explainer")
        except Exception as e:
            logger.warning(f"Failed to create SHAP explainer: {e}")
            self.explainer = None
    
    def calculate_shap_values(
        self, 
        X: pd.DataFrame, 
        max_samples: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Calculate SHAP values for the given data.
        
        Args:
            X: Features to explain
            max_samples: Maximum number of samples to process (for performance)
            
        Returns:
            SHAP values or None if not available
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            logger.warning("SHAP explainer not available")
            return None
        
        try:
            # Limit samples for performance
            if max_samples and len(X) > max_samples:
                X_sample = X.sample(n=max_samples, random_state=42)
            else:
                X_sample = X
            
            # Calculate SHAP values
            self.shap_values = self.explainer.shap_values(X_sample)
            
            # Handle different model types
            if isinstance(self.shap_values, list):
                # For binary classification, use positive class
                self.shap_values = self.shap_values[1]
            
            logger.info(f"Calculated SHAP values for {len(X_sample)} samples")
            return self.shap_values
            
        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {e}")
            return None
    
    def get_feature_importance_shap(self) -> Optional[pd.Series]:
        """Get feature importance based on SHAP values.
        
        Returns:
            Feature importance scores or None if not available
        """
        if self.shap_values is None or self.feature_names is None:
            return None
        
        # Calculate mean absolute SHAP values
        importance_scores = np.mean(np.abs(self.shap_values), axis=0)
        
        return pd.Series(
            importance_scores,
            index=self.feature_names
        ).sort_values(ascending=False)
    
    def explain_prediction(
        self, 
        X: pd.DataFrame, 
        sample_idx: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Explain a single prediction.
        
        Args:
            X: Features
            sample_idx: Index of the sample to explain
            
        Returns:
            Explanation dictionary or None if not available
        """
        if self.shap_values is None or self.feature_names is None:
            return None
        
        if sample_idx >= len(self.shap_values):
            logger.warning(f"Sample index {sample_idx} out of range")
            return None
        
        # Get SHAP values for the specific sample
        sample_shap = self.shap_values[sample_idx]
        sample_features = X.iloc[sample_idx]
        
        # Create explanation
        explanation = {
            'sample_idx': sample_idx,
            'feature_values': sample_features.to_dict(),
            'shap_values': dict(zip(self.feature_names, sample_shap)),
            'prediction': self.model.predict_proba(X.iloc[[sample_idx]])[0][1],
            'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
        }
        
        return explanation
    
    def plot_feature_importance(
        self, 
        top_n: int = 15,
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance based on SHAP values.
        
        Args:
            top_n: Number of top features to show
            save_path: Optional path to save the plot
        """
        if self.shap_values is None or self.feature_names is None:
            logger.warning("No SHAP values available for plotting")
            return
        
        # Calculate mean absolute SHAP values
        importance_scores = np.mean(np.abs(self.shap_values), axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True).tail(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Top {top_n} Feature Importance (SHAP)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.show()
    
    def plot_shap_summary(
        self, 
        max_display: int = 15,
        save_path: Optional[str] = None
    ) -> None:
        """Plot SHAP summary plot.
        
        Args:
            max_display: Maximum number of features to display
            save_path: Optional path to save the plot
        """
        if not SHAP_AVAILABLE or self.shap_values is None:
            logger.warning("SHAP not available or no SHAP values calculated")
            return
        
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values,
                self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            plt.title('SHAP Summary Plot')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved SHAP summary plot to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create SHAP summary plot: {e}")
    
    def plot_waterfall(
        self, 
        X: pd.DataFrame, 
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ) -> None:
        """Plot SHAP waterfall plot for a single prediction.
        
        Args:
            X: Features
            sample_idx: Index of the sample to explain
            save_path: Optional path to save the plot
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            logger.warning("SHAP not available or no explainer")
            return
        
        try:
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(
                self.explainer.expected_value,
                self.shap_values[sample_idx],
                X.iloc[sample_idx],
                feature_names=self.feature_names,
                show=False
            )
            plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved SHAP waterfall plot to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create SHAP waterfall plot: {e}")
    
    def plot_partial_dependence(
        self, 
        X: pd.DataFrame, 
        feature_name: str,
        save_path: Optional[str] = None
    ) -> None:
        """Plot partial dependence plot for a specific feature.
        
        Args:
            X: Features
            feature_name: Name of the feature to plot
            save_path: Optional path to save the plot
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            logger.warning("SHAP not available or no explainer")
            return
        
        if feature_name not in self.feature_names:
            logger.warning(f"Feature {feature_name} not found in feature names")
            return
        
        try:
            feature_idx = self.feature_names.index(feature_name)
            
            plt.figure(figsize=(10, 6))
            shap.partial_dependence_plot(
                feature_idx,
                self.model.predict_proba,
                X,
                ice=False,
                model_expected_value=True,
                feature_expected_value=True,
                show=False
            )
            plt.title(f'Partial Dependence Plot - {feature_name}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved partial dependence plot to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create partial dependence plot: {e}")
    
    def create_explanation_report(
        self, 
        X: pd.DataFrame, 
        sample_idx: int = 0,
        output_path: str = "explanation_report.txt"
    ) -> None:
        """Create a text report explaining a prediction.
        
        Args:
            X: Features
            sample_idx: Index of the sample to explain
            output_path: Path to save the report
        """
        explanation = self.explain_prediction(X, sample_idx)
        
        if explanation is None:
            logger.warning("No explanation available")
            return
        
        with open(output_path, 'w') as f:
            f.write("P2P Lending Risk Assessment - Prediction Explanation\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("DISCLAIMER: This is a research demonstration project only.\n")
            f.write("NOT intended for investment advice or commercial use.\n\n")
            
            f.write(f"Sample Index: {sample_idx}\n")
            f.write(f"Predicted Default Probability: {explanation['prediction']:.4f}\n")
            f.write(f"Base Value: {explanation['base_value']:.4f}\n\n")
            
            f.write("Feature Contributions (SHAP values):\n")
            f.write("-" * 40 + "\n")
            
            # Sort features by absolute SHAP value
            sorted_features = sorted(
                explanation['shap_values'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            for feature, shap_value in sorted_features:
                feature_value = explanation['feature_values'][feature]
                f.write(f"{feature}: {feature_value:.4f} (SHAP: {shap_value:.4f})\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        logger.info(f"Generated explanation report: {output_path}")


class ModelInterpretability:
    """Comprehensive model interpretability analysis."""
    
    def __init__(self, models: Dict[str, Any], feature_names: List[str]):
        """Initialize interpretability analysis.
        
        Args:
            models: Dictionary of trained models
            feature_names: List of feature names
        """
        self.models = models
        self.feature_names = feature_names
        self.explainers = {}
        
        # Create explainers for each model
        for model_name, model in models.items():
            try:
                explainer = CreditModelExplainer(model, feature_names)
                self.explainers[model_name] = explainer
            except Exception as e:
                logger.warning(f"Failed to create explainer for {model_name}: {e}")
    
    def compare_feature_importance(
        self, 
        X: pd.DataFrame,
        top_n: int = 15,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Compare feature importance across models.
        
        Args:
            X: Features to analyze
            top_n: Number of top features to show
            save_path: Optional path to save the plot
            
        Returns:
            DataFrame with feature importance comparison
        """
        importance_comparison = pd.DataFrame()
        
        for model_name, explainer in self.explainers.items():
            # Calculate SHAP values
            explainer.calculate_shap_values(X, max_samples=1000)
            
            # Get feature importance
            importance = explainer.get_feature_importance_shap()
            if importance is not None:
                importance_comparison[model_name] = importance
        
        if importance_comparison.empty:
            logger.warning("No feature importance data available")
            return pd.DataFrame()
        
        # Normalize importance scores
        importance_comparison = importance_comparison.div(importance_comparison.sum(axis=0), axis=1)
        
        # Get top features
        top_features = importance_comparison.sum(axis=1).sort_values(ascending=False).head(top_n)
        importance_comparison = importance_comparison.loc[top_features.index]
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        importance_comparison.plot(kind='bar', stacked=False)
        plt.title(f'Feature Importance Comparison (Top {top_n})')
        plt.xlabel('Features')
        plt.ylabel('Normalized Importance')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance comparison to {save_path}")
        
        plt.show()
        
        return importance_comparison
    
    def analyze_feature_stability(
        self, 
        X: pd.DataFrame,
        n_bootstrap: int = 100
    ) -> pd.DataFrame:
        """Analyze feature importance stability using bootstrap.
        
        Args:
            X: Features to analyze
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            DataFrame with stability analysis
        """
        stability_results = []
        
        for model_name, explainer in self.explainers.items():
            bootstrap_importance = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                bootstrap_idx = np.random.choice(len(X), size=len(X), replace=True)
                X_bootstrap = X.iloc[bootstrap_idx]
                
                # Calculate SHAP values
                explainer.calculate_shap_values(X_bootstrap, max_samples=500)
                
                # Get feature importance
                importance = explainer.get_feature_importance_shap()
                if importance is not None:
                    bootstrap_importance.append(importance)
            
            if bootstrap_importance:
                # Calculate stability metrics
                importance_df = pd.DataFrame(bootstrap_importance)
                stability_metrics = {
                    'mean_importance': importance_df.mean(),
                    'std_importance': importance_df.std(),
                    'cv_importance': importance_df.std() / importance_df.mean(),
                    'model': model_name
                }
                stability_results.append(stability_metrics)
        
        if not stability_results:
            logger.warning("No stability analysis results available")
            return pd.DataFrame()
        
        # Combine results
        stability_df = pd.concat([sr['mean_importance'] for sr in stability_results], axis=1)
        stability_df.columns = [sr['model'] for sr in stability_results]
        
        logger.info("Completed feature stability analysis")
        return stability_df
