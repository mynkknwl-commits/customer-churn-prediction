# Customer Churn Prediction

A supervised machine learning mini-project that predicts whether a telecom customer will churn (cancel their subscription) based on account details, subscribed services, and billing information.

## Problem Statement
Customer churn is costly for subscription-based businesses since acquiring a new customer is far more expensive than retaining one. This project builds a classification model to identify customers who are likely to churn, so a business could proactively target them with retention offers.

## Dataset
- **Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM Sample Data Sets)
- **Observations:** 7,043 customers
- **Features:** 20 (demographics, account info, subscribed services, billing)
- **Target:** `Churn` (Yes / No)

## Project Structure
```
churn_project/
├── data/
│   └── telco_churn.csv          # Raw dataset
├── notebook/
│   └── churn_prediction.ipynb   # Full analysis: EDA, preprocessing, modeling, evaluation
└── README.md
```

## Approach
1. **Data Preprocessing** – handled missing values in `TotalCharges`, encoded categorical variables, checked for outliers.
2. **EDA** – explored churn rate, contract type, tenure, and monthly charges in relation to churn.
3. **Modeling** – trained a **Logistic Regression** model (with a Decision Tree for comparison).
4. **Evaluation** – Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC.
5. **Interpretation** – identified month-to-month contracts and short tenure as the strongest churn drivers.

## Key Results
| Metric | Logistic Regression |
|---|---|
| Accuracy | ~80% |
| ROC-AUC | ~0.84 |

**Main finding:** New customers on flexible, no-commitment (month-to-month) plans, without add-on services like Tech Support, are the highest churn risk group.

## Tech Stack
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook notebook/churn_prediction.ipynb
```

## Limitations
- Dataset is a static snapshot; doesn't capture time-varying customer behavior or external factors (e.g., competitor pricing).
- Class imbalance (~26% churn) means recall on churners could be further improved with techniques like SMOTE or class-weighting.
