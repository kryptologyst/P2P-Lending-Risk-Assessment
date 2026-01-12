"""Streamlit demo application for P2P lending risk assessment."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.loader import P2PDataLoader
from features.engineering import FeatureEngineer
from models.credit_models import create_default_models
from risk.evaluation import CreditRiskEvaluator
from risk.explainability import CreditModelExplainer

# Page configuration
st.set_page_config(
    page_title="P2P Lending Risk Assessment",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
    }
    .danger-card {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'feature_engineer' not in st.session_state:
    st.session_state.feature_engineer = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None


def load_disclaimer():
    """Load and display disclaimer."""
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ IMPORTANT DISCLAIMER</h4>
        <p><strong>This is a research and educational demonstration project only.</strong></p>
        <p>This software is <strong>NOT intended for investment advice, financial planning, or any commercial use.</strong></p>
        <p>The models, predictions, and analyses are hypothetical and may be inaccurate. Past performance does not guarantee future results.</p>
        <p><strong>Always consult with qualified financial professionals before making any investment decisions.</strong></p>
    </div>
    """, unsafe_allow_html=True)


def train_models():
    """Train models and store in session state."""
    with st.spinner("Training models... This may take a few minutes."):
        # Initialize components
        data_loader = P2PDataLoader()
        feature_engineer = FeatureEngineer()
        evaluator = CreditRiskEvaluator()
        
        # Generate synthetic data
        df = data_loader.generate_synthetic_data(n_samples=5000, random_state=42)
        
        # Create splits
        train_df, val_df, test_df = data_loader.create_time_based_splits(
            df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42
        )
        
        # Feature engineering
        train_features = feature_engineer.engineer_all_features(train_df)
        test_features = feature_engineer.engineer_all_features(test_df)
        
        # Prepare features
        X_train, y_train = data_loader.prepare_features(train_features)
        X_test, y_test = data_loader.prepare_features(test_features)
        
        # Scale features
        X_train_scaled = feature_engineer.fit_transform_features(X_train)
        X_test_scaled = feature_engineer.transform_features(X_test)
        
        # Create and train models
        models = create_default_models(random_state=42)
        trained_models = {}
        
        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            trained_models[model_name] = model
        
        # Store in session state
        st.session_state.models_trained = True
        st.session_state.trained_models = trained_models
        st.session_state.feature_engineer = feature_engineer
        st.session_state.evaluator = evaluator
        st.session_state.X_test_scaled = X_test_scaled
        st.session_state.y_test = y_test
        st.session_state.feature_names = X_train_scaled.columns.tolist()
        
        st.success("Models trained successfully!")


def create_borrower_input_form():
    """Create input form for borrower characteristics."""
    st.subheader("📝 Borrower Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.slider(
            "Credit Score", 
            min_value=300, 
            max_value=850, 
            value=680, 
            help="FICO credit score (300-850)"
        )
        
        annual_income = st.number_input(
            "Annual Income ($)", 
            min_value=20000, 
            max_value=200000, 
            value=50000,
            step=1000,
            help="Borrower's annual income in USD"
        )
        
        debt_to_income = st.slider(
            "Debt-to-Income Ratio", 
            min_value=0.0, 
            max_value=0.8, 
            value=0.3, 
            step=0.01,
            help="Monthly debt payments divided by monthly income"
        )
        
        employment_length = st.slider(
            "Employment Length (years)", 
            min_value=0, 
            max_value=40, 
            value=5,
            help="Years of employment history"
        )
    
    with col2:
        home_ownership = st.selectbox(
            "Home Ownership",
            options=["RENT", "OWN", "MORTGAGE"],
            help="Current housing situation"
        )
        
        loan_purpose = st.selectbox(
            "Loan Purpose",
            options=["DEBT_CONSOLIDATION", "CREDIT_CARD", "HOME_IMPROVEMENT", 
                    "MAJOR_PURCHASE", "SMALL_BUSINESS", "OTHER"],
            help="Intended use of the loan"
        )
        
        loan_amount = st.number_input(
            "Loan Amount ($)", 
            min_value=1000, 
            max_value=50000, 
            value=15000,
            step=1000,
            help="Requested loan amount in USD"
        )
        
        loan_term = st.selectbox(
            "Loan Term (months)",
            options=[12, 24, 36, 48, 60],
            index=2,
            help="Loan repayment period in months"
        )
    
    return {
        'credit_score': credit_score,
        'annual_income': annual_income,
        'debt_to_income': debt_to_income,
        'employment_length': employment_length,
        'home_ownership': home_ownership,
        'loan_purpose': loan_purpose,
        'loan_amount': loan_amount,
        'loan_term': loan_term
    }


def calculate_derived_features(borrower_data):
    """Calculate derived features from borrower data."""
    # Calculate additional features
    monthly_debt = borrower_data['annual_income'] * borrower_data['debt_to_income'] / 12
    loan_to_income = borrower_data['loan_amount'] / borrower_data['annual_income']
    
    # Estimate interest rate based on credit score
    base_rate = 0.05
    credit_premium = (850 - borrower_data['credit_score']) / 1000 * 0.15
    amount_premium = (borrower_data['loan_amount'] - 1000) / 49000 * 0.05
    term_premium = (borrower_data['loan_term'] - 12) / 48 * 0.03
    
    interest_rate = base_rate + credit_premium + amount_premium + term_premium
    interest_rate = max(0.03, min(0.30, interest_rate))
    
    # Calculate monthly payment
    monthly_payment = (
        borrower_data['loan_amount'] * 
        (interest_rate / 12 * (1 + interest_rate / 12) ** borrower_data['loan_term']) /
        ((1 + interest_rate / 12) ** borrower_data['loan_term'] - 1)
    )
    
    payment_to_income = monthly_payment / (borrower_data['annual_income'] / 12)
    
    # Add derived features
    borrower_data.update({
        'monthly_debt': monthly_debt,
        'loan_to_income_ratio': loan_to_income,
        'interest_rate': interest_rate,
        'monthly_payment': monthly_payment,
        'payment_to_income_ratio': payment_to_income,
        'application_date': pd.Timestamp.now()
    })
    
    return borrower_data


def predict_risk(borrower_data):
    """Predict risk for the given borrower."""
    if not st.session_state.models_trained:
        st.error("Please train models first!")
        return None
    
    # Convert to DataFrame
    borrower_df = pd.DataFrame([borrower_data])
    
    # Apply feature engineering
    feature_engineer = st.session_state.feature_engineer
    engineered_features = feature_engineer.engineer_all_features(borrower_df)
    
    # Prepare features
    X = engineered_features.drop(['borrower_id', 'application_date', 'default_probability'], 
                                axis=1, errors='ignore')
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Scale features
    X_scaled = feature_engineer.transform_features(X)
    
    # Get predictions from all models
    predictions = {}
    for model_name, model in st.session_state.trained_models.items():
        prob = model.predict_default_probability(X_scaled)[0]
        predictions[model_name] = prob
    
    return predictions


def display_risk_assessment(predictions):
    """Display risk assessment results."""
    if not predictions:
        return
    
    st.subheader("🎯 Risk Assessment Results")
    
    # Calculate average prediction
    avg_prob = np.mean(list(predictions.values()))
    
    # Display risk level
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if avg_prob < 0.2:
            st.markdown('<div class="success-card"><h4>✅ LOW RISK</h4><p>Low probability of default</p></div>', 
                       unsafe_allow_html=True)
        elif avg_prob < 0.4:
            st.markdown('<div class="warning-card"><h4>⚠️ MEDIUM RISK</h4><p>Moderate probability of default</p></div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="danger-card"><h4>🚨 HIGH RISK</h4><p>High probability of default</p></div>', 
                       unsafe_allow_html=True)
    
    with col2:
        st.metric("Average Default Probability", f"{avg_prob:.1%}")
    
    with col3:
        st.metric("Risk Score", f"{avg_prob * 100:.0f}/100")
    
    # Model predictions comparison
    st.subheader("📊 Model Predictions")
    
    pred_df = pd.DataFrame(list(predictions.items()), columns=['Model', 'Default Probability'])
    pred_df['Default Probability'] = pred_df['Default Probability'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(pred_df, use_container_width=True)
    
    # Visualization
    fig = px.bar(
        pred_df, 
        x='Model', 
        y='Default Probability',
        title='Default Probability by Model',
        color='Default Probability',
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(yaxis_tickformat='.1%')
    st.plotly_chart(fig, use_container_width=True)


def display_model_performance():
    """Display model performance metrics."""
    if not st.session_state.models_trained:
        st.error("Please train models first!")
        return
    
    st.subheader("📈 Model Performance")
    
    # Get test data
    X_test_scaled = st.session_state.X_test_scaled
    y_test = st.session_state.y_test
    
    # Calculate metrics for each model
    metrics_data = []
    
    for model_name, model in st.session_state.trained_models.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_default_probability(X_test_scaled)
        
        # Calculate metrics
        evaluator = st.session_state.evaluator
        ml_metrics = evaluator.calculate_ml_metrics(y_test.values, y_pred, y_prob)
        credit_metrics = evaluator.calculate_credit_metrics(y_test.values, y_prob)
        
        metrics_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'AUC': ml_metrics['auc_roc'],
            'KS Statistic': credit_metrics['ks_statistic'],
            'Gini': credit_metrics['gini_coefficient'],
            'F1 Score': ml_metrics['f1_score'],
            'Precision': ml_metrics['precision'],
            'Recall': ml_metrics['recall']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display metrics table
    st.dataframe(metrics_df, use_container_width=True)
    
    # Performance visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('AUC Score', 'KS Statistic', 'Gini Coefficient', 'F1 Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    metrics_to_plot = ['AUC', 'KS Statistic', 'Gini', 'F1 Score']
    
    for i, metric in enumerate(metrics_to_plot):
        row = i // 2 + 1
        col = i % 2 + 1
        
        fig.add_trace(
            go.Bar(
                x=metrics_df['Model'],
                y=metrics_df[metric],
                name=metric,
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=600, title_text="Model Performance Metrics")
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">💰 P2P Lending Risk Assessment</h1>', 
                unsafe_allow_html=True)
    
    # Disclaimer
    load_disclaimer()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Risk Assessment", "Model Performance", "About"]
    )
    
    if page == "Risk Assessment":
        st.header("🔍 Individual Risk Assessment")
        
        # Train models button
        if not st.session_state.models_trained:
            st.info("👈 Please train models first using the button below.")
            if st.button("🚀 Train Models", type="primary"):
                train_models()
        else:
            st.success("✅ Models are trained and ready!")
        
        if st.session_state.models_trained:
            # Borrower input form
            borrower_data = create_borrower_input_form()
            
            # Calculate derived features
            borrower_data = calculate_derived_features(borrower_data)
            
            # Predict button
            if st.button("🎯 Assess Risk", type="primary"):
                predictions = predict_risk(borrower_data)
                display_risk_assessment(predictions)
    
    elif page == "Model Performance":
        st.header("📊 Model Performance Analysis")
        display_model_performance()
    
    elif page == "About":
        st.header("ℹ️ About This Project")
        
        st.markdown("""
        ## Project Overview
        
        This is a **research and educational demonstration** of Peer-to-Peer (P2P) lending risk assessment using machine learning techniques.
        
        ### Features
        
        - **Multiple Models**: Logistic Regression, Random Forest, XGBoost, LightGBM
        - **Comprehensive Evaluation**: AUC, KS statistic, Gini coefficient, Brier score
        - **Risk Assessment**: Individual borrower risk scoring
        - **Interactive Demo**: Real-time risk assessment interface
        
        ### Technical Stack
        
        - **Data Processing**: pandas, numpy
        - **Machine Learning**: scikit-learn, XGBoost, LightGBM
        - **Visualization**: plotly, matplotlib
        - **Web Interface**: Streamlit
        
        ### Methodology
        
        1. **Data Generation**: Synthetic P2P lending data with realistic characteristics
        2. **Feature Engineering**: Risk ratios, interaction features, time features
        3. **Model Training**: Multiple algorithms with proper validation
        4. **Evaluation**: Comprehensive metrics for credit risk assessment
        5. **Explainability**: SHAP-based model interpretation
        
        ### Use Cases
        
        - **Research**: Academic research on credit risk modeling
        - **Education**: Learning about machine learning in finance
        - **Prototyping**: Proof-of-concept for risk assessment systems
        
        ### Limitations
        
        - Uses synthetic data for demonstration
        - Models may not generalize to real-world scenarios
        - Simplified feature engineering
        - No real-time data integration
        
        ### Disclaimer
        
        **This software is for research and educational purposes only. It is NOT intended for investment advice, financial planning, or commercial use. Always consult with qualified financial professionals before making investment decisions.**
        """)


if __name__ == "__main__":
    main()
