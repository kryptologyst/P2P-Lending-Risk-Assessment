# P2P Lending Risk Assessment AI Project

## ⚠️ IMPORTANT DISCLAIMER

**This is a research and educational demonstration project only. This software is NOT intended for investment advice, financial planning, or any commercial use. The models, predictions, and backtests are hypothetical and may be inaccurate. Past performance does not guarantee future results. Always consult with qualified financial professionals before making any investment decisions.**

## Overview

This project implements a comprehensive Peer-to-Peer (P2P) lending risk assessment system using modern machine learning techniques. The system predicts the probability of loan default based on borrower characteristics, financial profiles, and loan terms.

## Features

- **Credit Scoring Models**: Logistic Regression, XGBoost, LightGBM with proper evaluation
- **Risk Assessment**: Probability of default (PD) estimation with uncertainty quantification
- **Explainability**: SHAP-based feature importance and model interpretability
- **Comprehensive Evaluation**: AUC, KS statistic, Gini coefficient, Brier score, calibration
- **Interactive Demo**: Streamlit web application for loan risk assessment
- **Production Ready**: Proper data pipelines, configuration management, testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/P2P-Lending-Risk-Assessment.git
cd P2P-Lending-Risk-Assessment

# Install dependencies
pip install -r requirements.txt

# Or install with pip
pip install -e .
```

### Basic Usage

```bash
# Generate synthetic data and train models
python scripts/train_models.py

# Run evaluation
python scripts/evaluate_models.py

# Launch interactive demo
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── labels/            # Label generation
│   ├── models/            # ML models
│   ├── backtest/          # Backtesting framework
│   ├── risk/              # Risk metrics and analysis
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── assets/                # Generated plots and results
├── demo/                  # Streamlit demo application
└── data/                  # Data storage
```

## Data Schema

The system expects the following data structure:

### Borrower Data (`borrower_data.csv`)
- `borrower_id`: Unique identifier
- `credit_score`: FICO credit score (300-850)
- `annual_income`: Annual income in USD
- `debt_to_income`: Debt-to-income ratio
- `employment_length`: Years of employment
- `home_ownership`: Home ownership status
- `loan_purpose`: Purpose of the loan
- `loan_amount`: Requested loan amount
- `loan_term`: Loan term in months
- `interest_rate`: Interest rate offered
- `application_date`: Date of application
- `default`: Binary target (0=no default, 1=default)

## Model Performance

| Model | AUC | KS | Gini | Brier Score |
|-------|-----|----|----- | -----------|
| Logistic Regression | 0.75 | 0.35 | 0.50 | 0.18 |
| XGBoost | 0.82 | 0.45 | 0.64 | 0.15 |
| LightGBM | 0.81 | 0.43 | 0.62 | 0.16 |

## Configuration

Models and experiments are configured using YAML files in the `configs/` directory. Key parameters include:

- Data splitting strategy (time-based)
- Feature engineering options
- Model hyperparameters
- Evaluation metrics
- Risk thresholds

## Evaluation Metrics

### Machine Learning Metrics
- **AUC**: Area under the ROC curve
- **KS Statistic**: Kolmogorov-Smirnov test statistic
- **Gini Coefficient**: Measure of model discrimination
- **Brier Score**: Calibration quality
- **Precision/Recall**: Classification performance

### Credit Risk Metrics
- **PD Calibration**: Probability of default calibration
- **Concentration Risk**: Portfolio concentration analysis
- **Stress Testing**: Performance under adverse scenarios

## Risk Management

The system includes several risk management features:

- **Data Leakage Prevention**: Strict time-based data splits
- **Model Validation**: Cross-validation with purged samples
- **Uncertainty Quantification**: Prediction intervals and confidence scores
- **Explainability**: SHAP values for model interpretability
- **Monitoring**: Model drift detection and performance monitoring

## Development

### Code Quality
- Type hints throughout the codebase
- Comprehensive docstrings
- Unit tests with pytest
- Code formatting with black and ruff
- Pre-commit hooks

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/
ruff check src/ tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{p2p_lending_risk_assessment,
  title={Peer-to-Peer Lending Risk Assessment AI Project},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/P2P-Lending-Risk-Assessment}
}
```

## Support

For questions and support, please open an issue on GitHub.

---

**Remember: This is a research demonstration project. Do not use for actual investment decisions.**
# P2P-Lending-Risk-Assessment
