# Credit Card Fraud Detection

A comprehensive, end-to-end Machine Learning pipeline for detecting fraudulent credit card transactions under severe class imbalance, featuring Random Forest and XGBoost models, rigorous data leakage prevention, and SMOTE oversampling.

---

## Project Overview

This project focuses on identifying fraudulent financial transactions where the positive class (fraud) is heavily underrepresented. Because of the extreme class imbalance, Accuracy was deliberately excluded as a primary evaluation metric to avoid misleading performance estimates.

Instead, the models were rigorously evaluated using:
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- False Negatives (missed fraud cases)

### Key Highlights
- Class Imbalance Handling: Applied SMOTE (Synthetic Minority Oversampling Technique) exclusively to the training set to prevent data leakage.
- Model Comparison: Evaluated and benchmarked Random Forest vs. XGBoost.
- Model Persistence: Saved trained models and scalers using Joblib.

---

## Dataset Information

- Dataset: Credit Card Fraud Detection
- Source: Kaggle - Credit Card Fraud Detection
- Original Dimensions: 284,807 transactions, 30 input features, 1 target variable
- Class Distribution (Cleaned):
  - Legitimate (0): 283,253 (99.83%)
  - Fraudulent (1): 473 (0.17%)

### Features
- Time: Seconds elapsed between each transaction and the first transaction in the dataset.
- V1 to V28: Anonymized principal components obtained via PCA transformation.
- Amount: Transaction amount.
- Class: Target variable (0 = Legitimate, 1 = Fraudulent).

---

## Project Workflow

Load Dataset
     |
Initial Exploration & Data Cleaning (Remove Duplicates)
     |
Stratified Train/Test Split (80/20)
     |
Feature Scaling (StandardScaler on 'Amount' using Train Stats Only)
     |
SMOTE on Training Data Only
     |
Random Forest           XGBoost
     |                     |
     +----------+----------+
                |
     Model Evaluation (Precision, Recall, F1, ROC-AUC)
                |
     Confusion Matrix & False-Negative Analysis
                |
     Model Comparison & Persistence (Joblib)

---

## Exploratory Data Analysis & Cleaning

1. Missing Values: Checked thoroughly; no missing values were found.
2. Duplicate Removal: Identified and removed 1,081 exact duplicate rows prior to splitting to prevent data leakage between training and testing sets.
   - Cleaned dataset shape: 283,726 rows x 31 columns.
3. Transaction Amount Analysis:
   - Legitimate transactions: Mean = 88.29, Median = 22.00
   - Fraudulent transactions: Mean = 122.21, Median = 9.25
   - Insight: Fraudulent transactions are not strictly higher-value transactions; the distribution is heavily skewed.

---

## Data Preprocessing & Leakage Prevention

- Stratified Train/Test Split: Split the data into an 80/20 ratio using stratification to preserve the extremely rare fraud proportion in both sets.
- Feature Scaling: Standardized only the Amount feature using StandardScaler (fitted exclusively on the training split to prevent test-set information leakage). PCA-transformed features (V1–V28) were left untouched.
- SMOTE Application: Applied imbalanced-learn's SMOTE strictly on the training data. The test set remained in its original distribution to reflect real-world performance accurately.

---

## Model Evaluation & Results

### 1. Random Forest
An ensemble bagging algorithm combining multiple decision trees.

- Precision: 90.12%
- Recall: 76.84%
- F1-Score: 82.95%
- ROC-AUC: 96.27%
- Confusion Matrix:
  - True Negatives: 56,643
  - False Positives: 8 (Legit flagged as fraud)
  - False Negatives: 22 (Missed frauds)
  - True Positives: 73

### 2. XGBoost
A gradient boosting algorithm that builds decision trees sequentially to minimize residual errors.

- Precision: 80.00%
- Recall: 80.00%
- F1-Score: 80.00%
- ROC-AUC: 96.23%
- Confusion Matrix:
  - True Negatives: 56,632
  - False Positives: 19 (Legit flagged as fraud)
  - False Negatives: 19 (Missed frauds)
  - True Positives: 76

---

## Model Comparison

| Metric | Random Forest | XGBoost |
| :--- | :--- | :--- |
| Precision | 90.12% | 80.00% |
| Recall | 76.84% | 80.00% |
| F1-Score | 82.95% | 80.00% |
| ROC-AUC | 96.27% | 96.23% |
| False Positives | 8 | 19 |
| False Negatives | 22 | 19 |

### Key Takeaways:
- Random Forest excels in Precision (90.12%) and produced fewer false alarms.
- XGBoost excels in Recall (80.00%) and successfully caught more actual fraud cases, reducing missed frauds.
- Business Recommendation: Because credit card fraud applications prioritize catching unauthorized transactions, XGBoost is preferred when minimizing missed fraud is the primary business goal.

---

## Project Structure

```text
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── amount_scaler.pkl
│
├── outputs/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── time_distribution.png
│   ├── smote_distribution.png
│   ├── rf_confusion_matrix.png
│   ├── xgb_confusion_matrix.png
│   ├── model_comparison.png
│   ├── model_comparison.csv
│   └── rf_feature_importance.png
│
├── README.md
├── requirements.txt
└── .gitignore
```
---

## Installation & Setup

1. Clone the Repository & Navigate to the Project Folder
2. Install Dependencies:
   pip install -r requirements.txt
3. Dataset Setup:
   - Download creditcard.csv from the Kaggle Credit Card Fraud Detection Page.
   - Place the file inside the data/ directory:
     data/creditcard.csv
4. Run the Notebook:
   jupyter notebook
   Open notebooks/credit_card_fraud_detection.ipynb and run all cells sequentially.

---

## Future Improvements

- Hyperparameter tuning using Grid Search / Random Search.
- Classification threshold optimization based on business cost matrices.
- Implementation of cost-sensitive learning models.
- Model deployment via a FastAPI or Flask web application.