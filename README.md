# Customer Churn Predictor
 
<div align="center">
**An end-to-end Machine Learning model that identifies telecom customers likely to churn — before they actually leave.**
 
</div>
---
 
## Table of Contents
 
- [Overview](#overview)
- [Project Flow](#project-flow)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [Model Architecture](#model-architecture)
- [Results](#model-performance)
- [Model Comparison](#model-comparison)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
---
 
## Overview
 
Customer churn costs telecom companies **billions of dollars** every year. Acquiring a new customer is **5-7x more expensive** than retaining an existing one. This project builds a binary classification model using **XGBoost** to predict whether a customer will churn — giving the business a window to intervene with targeted retention strategies.
 
| Metric | Score |
|--------|-------|
| Churn Recall | **92%** |
| ROC-AUC | **0.76** |
| Churners Caught | **342 / 373** |
| Churners Missed | **Only 31** |
 
---
 
## Project Flow
 
```
+------------------------------------------------------------------+
|                        PROJECT PIPELINE                          |
+------------------------------------------------------------------+
 
  Raw Data (telecom_churn.csv)
        |
        v
  +-------------+
  | Data        |  - Convert TotalCharges to numeric
  | Cleaning    |  - Fill NaN with median
  |             |  - Drop customerID column
  |             |  - Encode Churn (Yes/No -> 1/0)
  +------+------+
         |
         v
  +-------------+
  | Label       |  - LabelEncoder on all text columns
  | Encoding    |    (gender, Partner, Contract, etc.)
  +------+------+
         |
         v
  +-------------+
  | Feature     |  - charges_per_month = TotalCharges / (tenure + 1)
  | Engineering |  - is_new_customer (tenure <= 6)
  |             |  - is_loyal_customer (tenure >= 24)
  |             |  - total_services (sum of all service columns)
  |             |  - has_full_protection (Security+Support+Backup)
  |             |  - is_high_value (high charges + long tenure)
  +------+------+
         |
         v
  +-------------+
  | Train/Test  |  - 80% Train / 20% Test
  | Split       |  - stratify=y (equal churn ratio in both sets)
  +------+------+
         |
         v
  +-------------+
  | SMOTE       |  - Balance classes from 3:1 to 1:1
  | Oversampling|  - Applied ONLY on training data
  |             |  - No data leakage into test set
  +------+------+
         |
         v
  +-------------+
  | Model       |  - Logistic Regression (baseline)
  | Training    |  - Random Forest
  |             |  - XGBoost (primary model)
  |             |  - LightGBM (comparison)
  |             |  - Ensemble (XGB + LGBM + RF)
  +------+------+
         |
         v
  +-------------+
  | GridSearchCV|  - 5-fold cross validation
  | Tuning      |  - scoring = recall
  |             |  - 27 parameter combinations tested
  |             |  - Best: lr=0.01, depth=3, estimators=300
  +------+------+
         |
         v
  +-------------+
  | Threshold   |  - Default threshold = 0.50
  | Optimization|  - Optimal threshold = 0.35
  |             |  - Tested range: 0.30 to 0.50
  +------+------+
         |
         v
  +-------------+
  | Evaluation  |  - Classification Report
  |             |  - ROC-AUC Score
  |             |  - Confusion Matrix
  +------+------+
         |
         v
  +-------------+
  | Model       |  - Saved as churn_model.pkl
  | Export      |  - Ready for local or cloud deployment
  +-------------+
```
 
---
 
## Dataset
 
- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 rows x 21 columns
- **Target:** `Churn` — 1 if customer left, 0 if customer stayed
### Class Distribution
```
Stayed  (0) ████████████████████████  74%
Churned (1) ███████                   26%
```
 
---
 
## Features Used
 
| Feature | Description | Type |
|---------|-------------|------|
| `tenure` | Number of months with the company | Numerical |
| `MonthlyCharges` | Monthly bill amount | Numerical |
| `TotalCharges` | Total amount charged | Numerical |
| `Contract` | Month-to-month / One year / Two year | Categorical |
| `InternetService` | DSL / Fiber optic / None | Categorical |
| `OnlineSecurity` | Has online security addon | Binary |
| `TechSupport` | Has tech support addon | Binary |
| `charges_per_month` | TotalCharges / (tenure + 1) | Engineered |
| `is_new_customer` | 1 if tenure <= 6 months | Engineered |
| `is_loyal_customer` | 1 if tenure >= 24 months | Engineered |
| `total_services` | Count of all active services | Engineered |
| `has_full_protection` | Security + Support + Backup | Engineered |
| `is_high_value` | High charges + long tenure | Engineered |
 
---
 
## Model Architecture
 
```
XGBoostClassifier (Final Selected Model)
├── n_estimators  : 300
├── max_depth     : 3
├── learning_rate : 0.01
├── random_state  : 42
└── threshold     : 0.35  <- Tuned down from 0.5 to catch more churners
```
 
### Why XGBoost?
- Handles structured/tabular data extremely well
- Built-in regularization to prevent overfitting
- Works well with imbalanced datasets
- Fast and scalable even on large datasets
### Why SMOTE?
> The dataset had a **3:1 class imbalance** — 74% stayed, 26% churned.
> A naive model could hit 74% accuracy by predicting "no churn" for everyone — completely useless.
> SMOTE synthetically generates minority class samples to balance training data to 1:1 ratio.
 
### Why Threshold = 0.35?
> In churn prediction, **missing a churner is far worse than a false alarm.**
> Losing a customer permanently costs much more than sending an unnecessary retention offer.
> Lowering the threshold from 0.5 to 0.35 improved churn recall from **60% to 92%**.
 
---
 
## Model Performance
 
### Journey of Improvement
 
| Stage | Churn Recall | ROC-AUC | Churners Missed |
|-------|-------------|---------|-----------------|
| Baseline XGBoost | 60% | 0.72 | 151 |
| After GridSearchCV | 80% | 0.77 | 79 |
| After Feature Engineering | 90% | 0.75 | 39 |
| **Final (Threshold 0.35)** | **92%** | **0.76** | **31** |
 
### Why Recall Over Accuracy?
 
| Error Type | What Happens | Business Cost |
|------------|-------------|---------------|
| Miss a churner (False Negative) | Customer leaves, no intervention | HIGH — revenue lost forever |
| Flag a non-churner (False Positive) | Send unnecessary retention offer | LOW — small discount cost |
 
### Classification Report (Final Model)
 
```
              precision    recall  f1-score   support
 
           0       0.95      0.61      0.74      1036
           1       0.46      0.92      0.61       373
 
    accuracy                           0.69      1409
   macro avg       0.71      0.76      0.68      1409
weighted avg       0.82      0.69      0.71      1409
 
ROC-AUC: 0.76
```
 
### Confusion Matrix
```
                   Predicted: Stayed    Predicted: Churned
Actual: Stayed           635                  401
Actual: Churned           31                  342
```
 
- **342** churners correctly flagged for retention intervention
- Only **31** churners missed by the model
- 401 non-churners flagged (acceptable — cost is just a retention offer)
---
 
## Model Comparison
 
Three models were trained and evaluated systematically at threshold 0.35. XGBoost was selected as the final model based on best Recall score — the most critical metric for churn prediction.
 
| Model | Recall | ROC-AUC | Churners Missed | Selected |
|-------|--------|---------|-----------------|----------|
| Logistic Regression | 0.72 | 0.68 | 104 | No |
| Random Forest | 0.78 | 0.73 | 82 | No |
| LightGBM | 0.90 | 0.74 | 39 | No |
| Ensemble (XGB+LGBM+RF) | 0.89 | 0.77 | 41 | No |
| **XGBoost (Final)** | **0.92** | **0.76** | **31** | **Yes** |
 
### Why Not LightGBM or Ensemble?
 
```
LightGBM  → Recall 0.90 — missed 39 churners (8 more than XGBoost)
Ensemble  → Recall 0.89 — missed 41 churners (10 more than XGBoost)
XGBoost   → Recall 0.92 — missed only 31 churners (BEST)
```
 
> In a business context where every missed churner = lost revenue,
> XGBoost was the clear winner with the lowest miss rate.
> A systematic model comparison was done — not just assumption.
 
---
 
## Key Findings — Feature Importance
 
| Rank | Feature | Importance | Business Insight |
|------|---------|------------|-----------------|
| 1 | Contract Type | 0.42 | Month-to-month customers churn the most |
| 2 | Internet Service | 0.10 | Fiber optic users show higher churn rate |
| 3 | Online Security | 0.07 | No security addon = higher churn risk |
| 4 | Tech Support | 0.06 | Poor support experience drives churn |
| 5 | Phone Service | 0.05 | Service quality impacts retention |
 
> **Key Insight:** Converting month-to-month customers to annual contracts is the single most impactful retention strategy this model reveals.
 
---
 
## How to Run
 
### Prerequisites
- Python 3.10 or above
- pip
### 1. Clone the repository
```bash
git clone https://github.com/your-username/churn-prediction-model.git
cd churn-prediction-model
```
 
### 2. Create a virtual environment
```bash
python -m venv venv
```
 
### 3. Activate the virtual environment
 
**Windows:**
```bash
venv\Scripts\activate
```
 
**Mac / Linux:**
```bash
source venv/bin/activate
```
 
### 4. Install dependencies
```bash
pip install -r requirements.txt
```
 
### 5. Add the dataset
Download `telecom_churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project root folder.
 
### 6. Run the notebook
```bash
jupyter notebook churn_prediction.ipynb
```
 
---
 
### Predict a New Customer
 
```python
import joblib, pandas as pd
 
model = joblib.load('churn_model.pkl')
 
sample = pd.DataFrame({
    'tenure': [2],
    'MonthlyCharges': [85.0],
    'TotalCharges': [170.0],
    'Contract': [0],          # 0 = Month-to-month
    'InternetService': [1],   # 1 = Fiber optic
    'OnlineSecurity': [0],    # 0 = No
    'TechSupport': [0],       # 0 = No
    'total_services': [2],
    'is_new_customer': [1],
    'charges_per_month': [56.6]
})
 
prob = model.predict_proba(sample)[0][1]
pred = (prob >= 0.35).astype(int)
print(f"Churn Prediction: {'Will Churn' if pred == 1 else 'Will Stay'}")
print(f"Churn Probability: {prob:.1%}")
```
 
---
 
## Project Structure
 
```
churn-prediction-model/
│
├── churn_prediction.ipynb    ← Full notebook with all steps
├── churn_model.pkl           ← Saved XGBoost model
├── telecom_churn.csv         ← Dataset (download from Kaggle)
├── requirements.txt          ← All Python dependencies
└── README.md                 ← You are here
```
 
---
 
## Tech Stack
 
| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Preprocessing, metrics, splitting |
| XGBoost | Primary ML model |
| LightGBM | Comparison model |
| Imbalanced-learn | SMOTE oversampling |
| Joblib | Model serialization |
 
---
 
## Author
 
> Built with a focus on real-world business impact — not just model accuracy.
> Feel free to star this repo if you found it useful!
 
---
 
<div align="center">
<sub>Made using Python • XGBoost • SMOTE • scikit-learn</sub>
</div>
