"""Feature engineering module for P2P lending risk assessment."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for P2P lending data."""
    
    def __init__(self, scaler_type: str = 'standard'):
        """Initialize feature engineer.
        
        Args:
            scaler_type: Type of scaler to use ('standard', 'robust', 'power')
        """
        self.scaler_type = scaler_type
        self.scaler = self._get_scaler()
        self.is_fitted = False
        
    def _get_scaler(self):
        """Get the appropriate scaler based on type."""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'power': PowerTransformer(method='yeo-johnson')
        }
        return scalers.get(self.scaler_type, StandardScaler())
    
    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk-related features from raw data.
        
        Args:
            df: Input DataFrame with borrower and loan data
            
        Returns:
            DataFrame with additional risk features
        """
        df_features = df.copy()
        
        # Credit score categories
        df_features['credit_score_category'] = pd.cut(
            df_features['credit_score'],
            bins=[0, 580, 670, 740, 850],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        
        # Income categories
        df_features['income_category'] = pd.cut(
            df_features['annual_income'],
            bins=[0, 30000, 50000, 75000, 100000, float('inf')],
            labels=['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']
        )
        
        # Loan amount categories
        df_features['loan_amount_category'] = pd.cut(
            df_features['loan_amount'],
            bins=[0, 5000, 15000, 25000, 40000, float('inf')],
            labels=['Small', 'Medium', 'Large', 'Very Large', 'Extra Large']
        )
        
        # Risk ratios
        df_features['loan_to_income_ratio'] = df_features['loan_amount'] / df_features['annual_income']
        df_features['payment_to_income_ratio'] = df_features['monthly_payment'] / (df_features['annual_income'] / 12)
        
        # Credit utilization proxy (if we had credit card data)
        df_features['credit_utilization_proxy'] = df_features['debt_to_income'] * 0.8
        
        # Employment stability
        df_features['employment_stability'] = np.where(
            df_features['employment_length'] >= 5, 'Stable', 'Unstable'
        )
        
        # Loan purpose risk
        high_risk_purposes = ['DEBT_CONSOLIDATION', 'CREDIT_CARD']
        df_features['high_risk_purpose'] = df_features['loan_purpose'].isin(high_risk_purposes)
        
        # Interest rate risk
        df_features['high_interest_rate'] = df_features['interest_rate'] > df_features['interest_rate'].quantile(0.75)
        
        # Term risk
        df_features['long_term'] = df_features['loan_term'] > 36
        
        # Combined risk score
        risk_factors = [
            df_features['debt_to_income'] > 0.4,
            df_features['loan_to_income_ratio'] > 0.3,
            df_features['credit_score'] < 650,
            df_features['employment_length'] < 2,
            df_features['high_risk_purpose'],
            df_features['high_interest_rate'],
            df_features['long_term']
        ]
        
        df_features['risk_score'] = sum(risk_factors)
        
        logger.info("Created risk-related features")
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df_interactions = df.copy()
        
        # Credit score and income interaction
        df_interactions['credit_income_interaction'] = (
            df_interactions['credit_score'] * df_interactions['annual_income'] / 1000000
        )
        
        # Loan amount and term interaction
        df_interactions['amount_term_interaction'] = (
            df_interactions['loan_amount'] * df_interactions['loan_term'] / 1000
        )
        
        # Debt-to-income and employment length
        df_interactions['dti_employment_interaction'] = (
            df_interactions['debt_to_income'] * df_interactions['employment_length']
        )
        
        # Credit score and loan amount (higher credit score should allow larger loans)
        df_interactions['credit_amount_interaction'] = (
            df_interactions['credit_score'] * df_interactions['loan_amount'] / 1000000
        )
        
        logger.info("Created interaction features")
        return df_interactions
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for numerical variables.
        
        Args:
            df: Input DataFrame
            degree: Degree of polynomial features
            
        Returns:
            DataFrame with polynomial features
        """
        df_poly = df.copy()
        
        # Select numerical columns for polynomial features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        important_cols = ['credit_score', 'debt_to_income', 'loan_to_income_ratio']
        
        for col in important_cols:
            if col in numerical_cols:
                for d in range(2, degree + 1):
                    df_poly[f'{col}_power_{d}'] = df_poly[col] ** d
        
        logger.info(f"Created polynomial features up to degree {degree}")
        return df_poly
    
    def create_binning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binned features for continuous variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with binned features
        """
        df_binned = df.copy()
        
        # Bin credit score
        df_binned['credit_score_binned'] = pd.cut(
            df_binned['credit_score'],
            bins=10,
            labels=False
        )
        
        # Bin annual income
        df_binned['annual_income_binned'] = pd.cut(
            df_binned['annual_income'],
            bins=10,
            labels=False
        )
        
        # Bin loan amount
        df_binned['loan_amount_binned'] = pd.cut(
            df_binned['loan_amount'],
            bins=10,
            labels=False
        )
        
        logger.info("Created binned features")
        return df_binned
    
    def create_time_features(self, df: pd.DataFrame, date_col: str = 'application_date') -> pd.DataFrame:
        """Create time-based features.
        
        Args:
            df: Input DataFrame
            date_col: Name of the date column
            
        Returns:
            DataFrame with time features
        """
        df_time = df.copy()
        
        if date_col in df.columns:
            df_time[date_col] = pd.to_datetime(df_time[date_col])
            
            # Extract time components
            df_time['application_year'] = df_time[date_col].dt.year
            df_time['application_month'] = df_time[date_col].dt.month
            df_time['application_quarter'] = df_time[date_col].dt.quarter
            df_time['application_dayofweek'] = df_time[date_col].dt.dayofweek
            df_time['application_dayofyear'] = df_time[date_col].dt.dayofyear
            
            # Cyclical encoding for time features
            df_time['month_sin'] = np.sin(2 * np.pi * df_time['application_month'] / 12)
            df_time['month_cos'] = np.cos(2 * np.pi * df_time['application_month'] / 12)
            df_time['dayofweek_sin'] = np.sin(2 * np.pi * df_time['application_dayofweek'] / 7)
            df_time['dayofweek_cos'] = np.cos(2 * np.pi * df_time['application_dayofweek'] / 7)
            
            # Economic cycle proxy (simplified)
            df_time['economic_cycle'] = np.sin(2 * np.pi * df_time['application_year'] / 7)
        
        logger.info("Created time-based features")
        return df_time
    
    def engineer_all_features(
        self, 
        df: pd.DataFrame,
        include_interactions: bool = True,
        include_polynomials: bool = True,
        include_binning: bool = True,
        include_time: bool = True,
        polynomial_degree: int = 2
    ) -> pd.DataFrame:
        """Apply all feature engineering steps.
        
        Args:
            df: Input DataFrame
            include_interactions: Whether to include interaction features
            include_polynomials: Whether to include polynomial features
            include_binning: Whether to include binned features
            include_time: Whether to include time features
            polynomial_degree: Degree for polynomial features
            
        Returns:
            DataFrame with all engineered features
        """
        df_engineered = df.copy()
        
        # Create risk features
        df_engineered = self.create_risk_features(df_engineered)
        
        # Create interaction features
        if include_interactions:
            df_engineered = self.create_interaction_features(df_engineered)
        
        # Create polynomial features
        if include_polynomials:
            df_engineered = self.create_polynomial_features(df_engineered, polynomial_degree)
        
        # Create binned features
        if include_binning:
            df_engineered = self.create_binning_features(df_engineered)
        
        # Create time features
        if include_time:
            df_engineered = self.create_time_features(df_engineered)
        
        logger.info(f"Feature engineering complete. Shape: {df_engineered.shape}")
        return df_engineered
    
    def fit_scaler(self, X: pd.DataFrame) -> None:
        """Fit the scaler on training data.
        
        Args:
            X: Training features
        """
        self.scaler.fit(X)
        self.is_fitted = True
        logger.info(f"Fitted {self.scaler_type} scaler")
    
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming features")
        
        X_transformed = self.scaler.transform(X)
        
        # Convert back to DataFrame with original column names
        if isinstance(X, pd.DataFrame):
            X_transformed = pd.DataFrame(
                X_transformed,
                columns=X.columns,
                index=X.index
            )
        
        logger.info("Transformed features using fitted scaler")
        return X_transformed
    
    def fit_transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform features in one step.
        
        Args:
            X: Features to fit and transform
            
        Returns:
            Transformed features
        """
        self.fit_scaler(X)
        return self.transform_features(X)
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of important feature names for analysis.
        
        Returns:
            List of important feature names
        """
        return [
            'credit_score',
            'debt_to_income',
            'loan_to_income_ratio',
            'payment_to_income_ratio',
            'employment_length',
            'loan_amount',
            'loan_term',
            'interest_rate',
            'risk_score',
            'credit_income_interaction',
            'amount_term_interaction'
        ]
