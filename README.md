# Customer Churn Prediction — End-to-End Machine Learning Project
 
---
 
## Project Overview
 
Customer churn is one of the most critical business problems in the telecom industry. Losing a customer is far more expensive than retaining one. This project builds a complete, production-ready machine learning pipeline to identify customers who are likely to churn — before they actually do — giving the business a window to intervene with targeted retention strategies.
 
The model was trained on a real-world telecom dataset and optimized specifically for **maximum churn detection (Recall)**, which is the metric that matters most in a business context where missing a churner is costlier than a false alarm.
 
---
 
## Business Problem
 
In telecom, the average cost of acquiring a new customer is **5-7x higher** than retaining an existing one. A model that can flag at-risk customers even a few weeks in advance allows retention teams to:
 
- Offer personalized discounts or plan upgrades
- Proactively reach out with customer support
- Identify pain points causing dissatisfaction
> **The Challenge:** Churn datasets are heavily imbalanced. Only ~26% of customers churn, which means a naive model can achieve 74% accuracy by simply predicting "no churn" for everyone — and still be completely useless for the actual business goal.
 
---
 
## Why Recall Over Accuracy?
 
This is one of the most important design decisions in this project.
 
**Accuracy = 69%** — this number looks low, but it is intentional and correct.
 
Consider the two types of errors this model can make:
 
| Error Type | What Happens | Business Cost |
|:---|:---|:---|
| False Negative (Miss a churner) | Customer leaves, no intervention | HIGH — lost revenue forever |
| False Positive (Flag a non-churner) | Send a retention offer unnecessarily | LOW — small discount cost |
 
> Missing a churner means losing that customer permanently. Flagging a non-churner incorrectly costs only a small retention offer. Therefore, **Recall is the right metric to optimize**, even if it slightly reduces overall accuracy.
 
At threshold 0.35, the model catches **92% of all churners** — only 31 out of 373 churn customers are missed.
 
---
 
## Project Pipeline
 
```
Raw Data
    |
    v
Data Cleaning & Preprocessing
    |-- Convert TotalCharges to numeric
    |-- Fill missing values with median
    |-- Drop irrelevant columns (customerID)
    |
    v
Feature Engineering & Encoding
    |-- Label encode all categorical columns
    |-- Encode target variable (Yes/No -> 1/0)
    |
    v
Train / Test Split (80-20)
    |
    v
SMOTE - Synthetic Minority Oversampling
    |-- Balance classes from 3:1 ratio to 1:1
    |-- Applied ONLY on training data (no data leakage)
    |
    v
Model Training
    |-- Logistic Regression (baseline)
    |-- Random Forest
    |-- XGBoost (primary model)
    |
    v
Hyperparameter Tuning - GridSearchCV
    |-- 5-fold cross validation
    |-- Optimized for Recall
    |-- 27 parameter combinations tested
    |-- Best: learning_rate=0.01, max_depth=3, n_estimators=300
    |
    v
Threshold Optimization
    |-- Default threshold = 0.50
    |-- Optimal threshold = 0.35
    |-- Tested range: 0.30 to 0.50
    |
    v
Final Evaluation + Model Export (.pkl)
```
 
---
 
## Results
 
### Performance at Threshold 0.35 (Final Model)
 
| Metric | Class 0 (Stayed) | Class 1 (Churned) |
|:---|:---:|:---:|
| Precision | 0.95 | 0.46 |
| Recall | 0.61 | **0.92** |
| F1-Score | 0.74 | 0.61 |
 
| Overall Metric | Score |
|:---|:---:|
| Accuracy | 69% |
| ROC-AUC | 0.76 |
| Churn Recall | **92%** |
 
### Confusion Matrix
 
```
                   Predicted: Stayed    Predicted: Churned
Actual: Stayed           630                  406
Actual: Churned           31                  342
```
 
- **342** churners correctly identified and flagged for retention
- Only **31** churners missed by the model
- 406 non-churners flagged (acceptable false alarm rate)
### Journey of Improvement
 
| Stage | Churn Recall | Churners Missed |
|:---|:---:|:---:|
| Baseline XGBoost | 60% | 151 |
| After GridSearchCV | 80% | 79 |
| After Threshold Tuning | **92%** | **31** |
 
---
 
## Key Findings — Feature Importance
 
| Rank | Feature | Importance Score | Business Insight |
|:---:|:---|:---:|:---|
| 1 | Contract Type | 0.42 | Month-to-month customers churn the most |
| 2 | Internet Service | 0.10 | Fiber optic users show higher churn |
| 3 | Online Security | 0.07 | No security addon = higher churn risk |
| 4 | Tech Support | 0.06 | Poor support experience leads to churn |
| 5 | Phone Service | 0.05 | Service quality impacts retention |
 
> **Key Insight:** The single most impactful retention strategy would be converting month-to-month customers to annual contracts through targeted incentives.
 
---
 
## Note on Accuracy vs Recall Tradeoff
 
The final model accuracy is 69%, which is lower than a naive baseline of 74%. This is a **deliberate and correct tradeoff**.
 
By lowering the classification threshold from 0.50 to 0.35, the model becomes more aggressive in flagging potential churners. This increases false positives but dramatically reduces false negatives.
 
In production, the business team would review flagged customers and apply retention strategies selectively. The cost of reviewing extra customers is far lower than the revenue loss from missed churners.
 
---
 
## Tech Stack
 
| Category | Tools |
|:---|:---|
| Language | Python 3.12 |
| ML Framework | Scikit-learn, XGBoost |
| Imbalanced Data | Imbalanced-learn (SMOTE) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Environment | Google Colab |
 
---
 
## How to Run
 
```bash
# Clone the repository
git clone https://github.com/your-username/churn-prediction-model.git
 
# Install dependencies
pip install -r requirements.txt
 
# Open the notebook in Google Colab or Jupyter and run all cells
```
 
---
 
## Project Structure
 
```
churn-prediction-model/
|
|-- churn_prediction.ipynb    <- Full notebook with all steps
|-- churn_model.pkl           <- Saved XGBoost model
|-- dataset.csv               <- Telecom churn dataset
|-- requirements.txt          <- Python dependencies
|-- README.md                 <- Project documentation
```
 
---
 
## Future Improvements
 
- Deploy model as a REST API using FastAPI or Flask
- Build an interactive dashboard using Streamlit
- Experiment with LightGBM and CatBoost
- Add SHAP values for individual prediction explainability
- Implement real-time prediction pipeline
---
 
## Author
 
Built as part of a hands-on machine learning portfolio project focusing on real-world business problem solving, not just model accuracy.
