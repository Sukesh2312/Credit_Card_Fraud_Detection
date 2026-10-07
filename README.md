# Credit Card Fraud Detection Dashboard

## 2013 vs 2023 Dataset Analysis & Machine Learning

A modern, interactive Streamlit web dashboard for detecting credit card fraud using machine learning pipelines trained independently on classic 2013 and newer 2023 datasets.

---

## 📌 Project Overview

This system compares machine learning performance on two separate credit card fraud datasets:
- **2013 Dataset**: Classic highly imbalanced dataset (284,807 transactions, 0.17% fraud rate).
- **2023 Dataset**: Balanced benchmark dataset (568,630 transactions, 50.00% fraud rate).

Models evaluated:
1. **Logistic Regression** (Class-weighted baseline)
2. **Random Forest** (Bagged Decision Trees)
3. **XGBoost** (Gradient Boosted Trees)

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── data/
│   └── raw/
│       ├── creditcard_2013.csv
│       └── creditcard_2023.csv
├── notebooks/
│   └── 01_data_inspection_eda.ipynb
├── models/
│   ├── logistic_regression_2013.pkl
│   ├── random_forest_2013.pkl
│   ├── xgboost_2013.pkl
│   ├── logistic_regression_2023.pkl
│   ├── random_forest_2023.pkl
│   └── xgboost_2023.pkl
├── results/
│   ├── model_comparison.csv
│   ├── cross_validation_results.csv
│   └── class_distribution_comparison.csv
├── app/
│   ├── app.py
│   ├── predictor.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started & Running the Dashboard

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit Application

```bash
streamlit run app/app.py
```

---

## 📊 Dashboard Pages & Features

1. **Overview**: Key metrics, class distributions, and automated dataset insights.
2. **Dataset Analysis**: Exploratory Data Analysis, amount distributions, and correlation heatmaps.
3. **Model Performance**: Multi-metric evaluation tables, metric comparisons, and best model selection based on F1 Score.
4. **Fraud Prediction**: Interactive CSV upload, schema validation, real-time prediction, probability breakdown, and flagged transaction exports.
5. **Feature Importance**: Top 15 model feature importances extracted directly from trained pipelines.
6. **About Project**: Problem statement, methodology, evaluation metrics, and system architecture.
